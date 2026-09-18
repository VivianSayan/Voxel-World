//! A sequence with gaps: values live at integer indices that need not be
//! contiguous, duplicates are allowed, and every value knows which indices
//! hold it (see `SparseIndexed`).
//!
//! Index-based methods follow `BTreeMap`; reordering (`Reorder`) moves
//! values between the occupied indices without changing them.

use crate::misc::random::Random;
use crate::misc::structures::collections::sparse::sparse_core::{One, SparseCore};
use crate::misc::structures::sampling;
use crate::misc::structures::traits::sequence::is_window;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, Reorder, SparseIndexed,
};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::RangeBounds;

#[derive(Clone, Debug)]
/// Values of type `T` stored at arbitrary signed 64-bit indices.
///
/// Indices are ordered and may contain gaps. Equal values may appear at
/// multiple indices and are reverse-indexed for efficient lookup.
pub struct SparseSequence<T> {
    core: SparseCore<T, One<T>>,
}

impl<T> Default for SparseSequence<T> {
    fn default() -> Self {
        Self {
            core: SparseCore::default(),
        }
    }
}

type PairIter<'a, T> = std::iter::Map<
    std::collections::btree_map::Iter<'a, i64, One<T>>,
    fn((&'a i64, &'a One<T>)) -> (i64, &'a T),
>;

impl<T> SparseSequence<T> {
    /// Creates an empty sparse sequence.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of occupied indices.
    pub fn len(&self) -> usize {
        self.core.slots.len()
    }

    /// Returns whether no indices are occupied.
    pub fn is_empty(&self) -> bool {
        self.core.slots.is_empty()
    }

    /// Returns the value stored at `index`.
    pub fn get(&self, index: i64) -> Option<&T> {
        self.core.slots.get(&index).map(|slot| &slot.0)
    }

    /// The value at the lowest index.
    pub fn first(&self) -> Option<&T> {
        self.core.slots.values().next().map(|slot| &slot.0)
    }

    /// The value at the highest index.
    pub fn last(&self) -> Option<&T> {
        self.core.slots.values().next_back().map(|slot| &slot.0)
    }

    /// `(index, value)` pairs in index order.
    pub fn iter(&self) -> PairIter<'_, T> {
        self.core
            .slots
            .iter()
            .map(|(index, slot)| (*index, &slot.0))
    }

    /// Values in index order.
    pub fn values(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        self.core.slots.values().map(|slot| &slot.0)
    }

    /// `(index, value)` pairs whose index lies in `range`.
    pub fn range(
        &self,
        range: impl RangeBounds<i64>,
    ) -> impl DoubleEndedIterator<Item = (i64, &T)> {
        self.core
            .slots
            .range(range)
            .map(|(index, slot)| (*index, &slot.0))
    }

    /// Among the values in index order, the count of leading `is_below` ones.
    pub fn partition_point(&self, mut is_below: impl FnMut(&T) -> bool) -> usize {
        self.values().take_while(|value| is_below(value)).count()
    }
}

impl<T: Element> SparseSequence<T> {
    /// Returns whether `item` occurs at any index.
    pub fn contains(&self, item: &T) -> bool {
        self.core.positions.contains_key(item)
    }

    /// Stores `item` at the first free index from zero, and returns it.
    pub fn push(&mut self, item: T) -> i64 {
        let index = self.core.first_free();
        self.core.put(index, One(item));
        index
    }

    /// Stores `item` at `index`, returning whatever was there.
    pub fn insert_at(&mut self, index: i64, item: T) -> Option<T> {
        self.core.put(index, One(item)).map(|slot| slot.0)
    }

    /// Replaces the value at an occupied `index`, returning the old one.
    pub fn replace(&mut self, index: i64, item: T) -> Option<T> {
        if !self.core.slots.contains_key(&index) {
            return None;
        }
        self.insert_at(index, item)
    }

    /// Stores `item` at `index`, first moving every value at or after
    /// `index` one index up.
    pub fn insert_shifting(&mut self, index: i64, item: T) {
        self.core.shift_from(index);
        self.core.put(index, One(item));
    }

    /// Removes and returns the value stored at `index`.
    pub fn remove_at(&mut self, index: i64) -> Option<T> {
        self.core.take(index).map(|slot| slot.0)
    }

    /// Removes the value at the lowest index.
    pub fn pop_first(&mut self) -> Option<(i64, T)> {
        let index = self.core.min_index()?;
        Some((index, self.remove_at(index)?))
    }

    /// Removes the value at the highest index.
    pub fn pop_last(&mut self) -> Option<(i64, T)> {
        let index = self.core.max_index()?;
        Some((index, self.remove_at(index)?))
    }

    /// Removes the first occurrence of `item`, returning where it was.
    pub fn remove_first_of(&mut self, item: &T) -> Option<i64> {
        let index = self.core.first_index_of(item)?;
        self.core.take(index);
        Some(index)
    }

    /// Removes the last occurrence of `item`, returning where it was.
    pub fn remove_last_of(&mut self, item: &T) -> Option<i64> {
        let index = self.core.last_index_of(item)?;
        self.core.take(index);
        Some(index)
    }

    /// Removes every occurrence, returning how many there were.
    pub fn remove_all_of(&mut self, item: &T) -> usize {
        let indices: Vec<i64> = self.core.indices_of(item).collect();
        for &index in &indices {
            self.core.take(index);
        }
        indices.len()
    }

    /// Removes all entries and reverse-index data.
    pub fn clear(&mut self) {
        self.core.clear();
    }

    /// Keeps only the entries `keep` accepts, at their indices.
    pub fn retain(&mut self, mut keep: impl FnMut(i64, &T) -> bool) {
        let before = self.core.slots.len();
        self.core.slots.retain(|index, slot| keep(*index, &slot.0));
        if self.core.slots.len() != before {
            self.core.rebuild();
        }
    }

    /// Renumbers the values to indices `0..len`, keeping their order.
    pub fn compact(&mut self) {
        self.core.compact();
    }

    /// True when this sequence's values appear consecutively, in order, in
    /// `other`, whatever their indices.
    pub fn is_window_of(&self, other: &Self) -> bool {
        is_window(self.values(), other.values())
    }
}

impl<T: Element> Collection for SparseSequence<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.core.slots.len()
    }

    fn contains(&self, item: &T) -> bool {
        Self::contains(self, item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.values()
    }
}

/// `insert` pushes at the first free index; `remove` takes the first occurrence.
impl<T: Element> CollectionInsert for SparseSequence<T> {
    fn insert(&mut self, item: T) -> bool {
        self.push(item);
        true
    }
}

impl<T: Element> CollectionRemove for SparseSequence<T> {
    fn remove(&mut self, item: &T) -> bool {
        self.remove_first_of(item).is_some()
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, mut keep: F) {
        Self::retain(self, |_, item| keep(item))
    }
}

impl<T: Element> Choose for SparseSequence<T> {}

impl<T: Element> SparseIndexed for SparseSequence<T> {
    type Slot = T;

    fn slot(&self, index: i64) -> Option<&T> {
        self.get(index)
    }

    fn slot_count(&self) -> usize {
        self.core.slots.len()
    }

    fn first_index(&self) -> Option<i64> {
        self.core.min_index()
    }

    fn last_index(&self) -> Option<i64> {
        self.core.max_index()
    }

    fn indices(&self) -> impl DoubleEndedIterator<Item = i64> {
        self.core.slots.keys().copied()
    }

    fn indices_of(&self, item: &T) -> impl DoubleEndedIterator<Item = i64> {
        self.core.indices_of(item)
    }

    fn next_free_index(&self) -> i64 {
        self.core.first_free()
    }
}

/// Moves values between the occupied indices.
impl<T: Element> Reorder for SparseSequence<T> {
    fn sort_by<F: FnMut(&T, &T) -> Ordering>(&mut self, mut compare: F) {
        self.core
            .reorder(|values| values.sort_by(|a, b| compare(&a.0, &b.0)));
    }

    fn reverse(&mut self) {
        self.core.reorder(|values| values.reverse());
    }

    fn shuffle(&mut self, random: &mut Random) {
        self.core
            .reorder(|values| sampling::shuffle(values, random));
    }
}

impl<T: Element> PartialEq for SparseSequence<T> {
    fn eq(&self, other: &Self) -> bool {
        self.core.slots == other.core.slots
    }
}

impl<T: Element> Eq for SparseSequence<T> {}

impl<T: Element> Hash for SparseSequence<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (index, value) in self.iter() {
            index.hash(state);
            value.hash(state);
        }
    }
}

impl<T> std::ops::Index<i64> for SparseSequence<T> {
    type Output = T;

    fn index(&self, index: i64) -> &T {
        self.get(index).expect("no value at this index")
    }
}

/// Packs the values from index zero.
impl<T: Element> FromIterator<T> for SparseSequence<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        let mut sequence = Self::new();
        sequence.extend(items);
        sequence
    }
}

impl<T: Element> FromIterator<(i64, T)> for SparseSequence<T> {
    fn from_iter<I: IntoIterator<Item = (i64, T)>>(pairs: I) -> Self {
        let mut sequence = Self::new();
        sequence.extend(pairs);
        sequence
    }
}

/// Pushes each value at the first free index.
impl<T: Element> Extend<T> for SparseSequence<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        for item in items {
            self.push(item);
        }
    }
}

/// Stores each value at its index, replacing what was there.
impl<T: Element> Extend<(i64, T)> for SparseSequence<T> {
    fn extend<I: IntoIterator<Item = (i64, T)>>(&mut self, pairs: I) {
        for (index, item) in pairs {
            self.insert_at(index, item);
        }
    }
}

impl<'a, T> IntoIterator for &'a SparseSequence<T> {
    type Item = (i64, &'a T);
    type IntoIter = PairIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
