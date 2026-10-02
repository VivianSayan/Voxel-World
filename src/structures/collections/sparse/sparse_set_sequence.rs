//! A sparse sequence whose slots hold sets: several distinct values can
//! share an index, and every value knows which indices hold it (see
//! `SparseIndexed`). Empty slots are removed automatically.

use crate::random::source::StochasticSource;
use crate::structures::collections::sets::set::Set;
use crate::structures::collections::sparse::sparse_core::SparseCore;
use crate::structures::sampling;
use crate::structures::traits::sequence::is_window;
use crate::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, RangeQuery, SparseIndexed,
};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::RangeBounds;

#[derive(Clone, Debug)]
/// Sets of values `T` stored at arbitrary signed 64-bit indices.
///
/// Each slot is a unique set, while the same value may occur in several
/// slots. Empty slots are removed automatically.
pub struct SparseSetSequence<T> {
    core: SparseCore<T, Set<T>>,
    element_count: usize,
}

impl<T> Default for SparseSetSequence<T> {
    fn default() -> Self {
        Self {
            core: SparseCore::default(),
            element_count: 0,
        }
    }
}

impl<T> SparseSetSequence<T> {
    /// Creates an empty sparse set sequence.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of values across all slots.
    pub fn len(&self) -> usize {
        self.element_count
    }

    /// Number of occupied indices.
    pub fn slot_count(&self) -> usize {
        self.core.slots.len()
    }

    /// Returns whether no slots are occupied.
    pub fn is_empty(&self) -> bool {
        self.core.slots.is_empty()
    }

    /// Returns the set stored at `index`.
    pub fn get(&self, index: i64) -> Option<&Set<T>> {
        self.core.slots.get(&index)
    }

    /// The slot at the lowest index.
    pub fn first(&self) -> Option<&Set<T>> {
        self.core.slots.values().next()
    }

    /// The slot at the highest index.
    pub fn last(&self) -> Option<&Set<T>> {
        self.core.slots.values().next_back()
    }

    /// `(index, slot)` pairs in index order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (i64, &Set<T>)> + ExactSizeIterator {
        self.core.slots.iter().map(|(index, slot)| (*index, slot))
    }

    /// `(index, slot)` pairs whose index lies in `range`.
    pub fn range(
        &self,
        range: impl RangeBounds<i64>,
    ) -> impl DoubleEndedIterator<Item = (i64, &Set<T>)> {
        self.core
            .slots
            .range(range)
            .map(|(index, slot)| (*index, slot))
    }
}

impl<T: Element> SparseSetSequence<T> {
    /// Returns whether `item` occurs in any slot.
    pub fn contains(&self, item: &T) -> bool {
        self.core.positions.contains_key(item)
    }

    /// Adds `item` to the slot at `index`, creating the slot if needed.
    /// Returns whether it was new there.
    pub fn insert_at(&mut self, index: i64, item: T) -> bool {
        let slot = self.core.slots.entry(index).or_default();
        let created = slot.is_empty();
        if !slot.insert(item.clone()) {
            return false;
        }
        if created {
            self.core.occupy(index);
        }
        self.core.index_element(&item, index);
        self.element_count += 1;
        true
    }

    /// Adds every value from `items` to the slot at `index`.
    pub fn extend_at(&mut self, index: i64, items: impl IntoIterator<Item = T>) {
        for item in items {
            self.insert_at(index, item);
        }
    }

    /// Adds `item` to a new slot at the first free index from zero.
    pub fn push(&mut self, item: T) -> i64 {
        let index = self.core.first_free();
        self.insert_at(index, item);
        index
    }

    /// Moves every slot at or after `index` one index up, then adds `item`
    /// at `index`.
    pub fn insert_shifting(&mut self, index: i64, item: T) {
        self.core.shift_from(index);
        self.insert_at(index, item);
    }

    /// Moves every slot at or after `index` one index up, then stores
    /// `items` as the slot at `index`.
    pub fn insert_slot_shifting(&mut self, index: i64, items: Set<T>) {
        self.core.shift_from(index);
        if !items.is_empty() {
            self.element_count += items.len();
            self.core.put(index, items);
        }
    }

    /// Removes `item` from the slot at `index`.
    pub fn remove_at(&mut self, index: i64, item: &T) -> bool {
        let Some(slot) = self.core.slots.get_mut(&index) else {
            return false;
        };
        if !slot.remove(item) {
            return false;
        }
        if slot.is_empty() {
            self.core.slots.remove(&index);
            self.core.vacate(index);
        }
        self.core.unindex_element(item, index);
        self.element_count -= 1;
        true
    }

    /// Removes the whole slot at `index`.
    pub fn remove_slot(&mut self, index: i64) -> Option<Set<T>> {
        let slot = self.core.take(index)?;
        self.element_count -= slot.len();
        Some(slot)
    }

    /// Removes and returns the lowest-indexed slot.
    pub fn pop_first(&mut self) -> Option<(i64, Set<T>)> {
        let index = self.core.min_index()?;
        Some((index, self.remove_slot(index)?))
    }

    /// Removes and returns the highest-indexed slot.
    pub fn pop_last(&mut self) -> Option<(i64, Set<T>)> {
        let index = self.core.max_index()?;
        Some((index, self.remove_slot(index)?))
    }

    /// Removes the first occurrence of `item` and returns its index.
    pub fn remove_first_of(&mut self, item: &T) -> Option<i64> {
        let index = self.core.first_index_of(item)?;
        self.remove_at(index, item);
        Some(index)
    }

    /// Removes the last occurrence of `item` and returns its index.
    pub fn remove_last_of(&mut self, item: &T) -> Option<i64> {
        let index = self.core.last_index_of(item)?;
        self.remove_at(index, item);
        Some(index)
    }

    /// Removes `item` from every slot and returns the number of slots changed.
    pub fn remove_all_of(&mut self, item: &T) -> usize {
        let indices: Vec<i64> = self.core.indices_of(item).collect();
        for &index in &indices {
            self.remove_at(index, item);
        }
        indices.len()
    }

    /// Removes every slot and reverse-index entry.
    pub fn clear(&mut self) {
        self.core.clear();
        self.element_count = 0;
    }

    /// Keeps only the values `keep` accepts, in every slot.
    pub fn retain(&mut self, mut keep: impl FnMut(i64, &T) -> bool) {
        self.core.slots.retain(|index, slot| {
            slot.retain(|item| keep(*index, item));
            !slot.is_empty()
        });
        self.after_bulk_change();
    }

    /// Keeps only the slots `keep` accepts.
    pub fn retain_slots(&mut self, mut keep: impl FnMut(i64, &Set<T>) -> bool) {
        self.core.slots.retain(|index, slot| keep(*index, slot));
        self.after_bulk_change();
    }

    /// Renumbers the slots to indices `0..slot_count`, keeping their order.
    pub fn compact(&mut self) {
        self.core.compact();
    }

    /// Reorders the slots while keeping the occupied indices.
    pub fn sort_slots_by(&mut self, compare: impl FnMut(&Set<T>, &Set<T>) -> Ordering) {
        self.core.reorder(|slots| slots.sort_by(compare));
    }

    /// Reverses slot order while leaving occupied index positions unchanged.
    pub fn reverse_slots(&mut self) {
        self.core.reorder(|slots| slots.reverse());
    }

    /// Randomly shuffles slots across occupied indices, from any random source.
    ///
    /// # Why this is a named method and not [`Shuffle`](crate::structures::traits::Shuffle)
    ///
    /// Every other sequence here implements that trait, and this one deliberately does
    /// not. A slot holds a whole set, so "shuffle this collection" has more than one
    /// sensible reading:
    ///
    /// - move the slots around, leaving each set's contents together — this method;
    /// - shuffle the elements *inside* each slot;
    /// - redistribute elements across slots entirely.
    ///
    /// A trait method would have to pick one, and a caller who wanted another would
    /// silently get the wrong thing. Naming it says which one happened, so the
    /// ambiguity is settled at the call site, where the caller is the one who knows.
    ///
    /// The sibling [`SparseSequence`](super::sparse_sequence::SparseSequence) holds one
    /// element per slot, so for it there is only one reading, and it does implement the
    /// trait.
    pub fn shuffle_slots<S: StochasticSource + ?Sized>(&mut self, source: &mut S) {
        self.core.reorder(|slots| sampling::shuffle(slots, source));
    }

    /// True when this sequence's slots appear consecutively, in order, in
    /// `other`, whatever their indices.
    pub fn is_window_of(&self, other: &Self) -> bool {
        is_window(self.core.slots.values(), other.core.slots.values())
    }

    fn after_bulk_change(&mut self) {
        self.element_count = self.core.slots.values().map(Set::len).sum();
        self.core.rebuild();
    }
}

/// Ordered by index, so a range is a span of indices and each match is the set
/// stored at one.
impl<T: Element> RangeQuery for SparseSetSequence<T> {
    type Key = i64;
    type Item<'a>
        = (i64, &'a Set<T>)
    where
        Self: 'a;

    fn range<'a, R: RangeBounds<Self::Key>>(
        &'a self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = Self::Item<'a>> {
        SparseSetSequence::range(self, range)
    }
}

impl<T: Element> Collection for SparseSetSequence<T> {
    type Item = T;

    /// Number of values across all slots.
    fn len(&self) -> usize {
        self.element_count
    }

    fn contains(&self, item: &T) -> bool {
        Self::contains(self, item)
    }

    /// Every value, slot by slot.
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.core.slots.values().flatten()
    }
}

/// `insert` pushes into a new slot; `remove` takes the first occurrence.
impl<T: Element> CollectionInsert for SparseSetSequence<T> {
    fn insert(&mut self, item: T) -> bool {
        self.push(item);
        true
    }
}

impl<T: Element> CollectionRemove for SparseSetSequence<T> {
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

impl<T: Element> Choose for SparseSetSequence<T> {}

impl<T: Element> SparseIndexed for SparseSetSequence<T> {
    type Slot = Set<T>;

    fn slot(&self, index: i64) -> Option<&Set<T>> {
        self.core.slots.get(&index)
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

impl<T: Element> PartialEq for SparseSetSequence<T> {
    fn eq(&self, other: &Self) -> bool {
        self.core.slots == other.core.slots
    }
}

impl<T: Element> Eq for SparseSetSequence<T> {}

impl<T: Element> Hash for SparseSetSequence<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (index, slot) in self.iter() {
            index.hash(state);
            slot.hash(state);
        }
    }
}

impl<T> std::ops::Index<i64> for SparseSetSequence<T> {
    type Output = Set<T>;

    fn index(&self, index: i64) -> &Set<T> {
        self.get(index).expect("no slot at this index")
    }
}

/// Stores each value in the slot at its index.
impl<T: Element> FromIterator<(i64, T)> for SparseSetSequence<T> {
    fn from_iter<I: IntoIterator<Item = (i64, T)>>(pairs: I) -> Self {
        let mut sequence = Self::new();
        sequence.extend(pairs);
        sequence
    }
}

impl<T: Element> Extend<(i64, T)> for SparseSetSequence<T> {
    fn extend<I: IntoIterator<Item = (i64, T)>>(&mut self, pairs: I) {
        for (index, item) in pairs {
            self.insert_at(index, item);
        }
    }
}
