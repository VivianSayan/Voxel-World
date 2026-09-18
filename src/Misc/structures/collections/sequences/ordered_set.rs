//! A unique collection with an explicit order: set-like uniqueness with
//! vector-like positions. Lookups are O(1); inserting or removing at a
//! position is O(n) because later positions shift.
//!
//! Method names follow `Vec` and the `indexmap` crate's `IndexSet`.
//! Sorting, reversing and shuffling come from `Reorder`.

use crate::misc::random::Random;
use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::sampling;
use crate::misc::structures::traits::operators::impl_set_operators;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, InsertAt, Reorder, Sequence,
    SequenceMut, SetAlgebra, UniqueCollection,
};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Range;

#[derive(Clone, Debug)]
/// Unique sequence of elements of type `T` with O(1) membership lookup.
///
/// A vector preserves order while a reverse hash index records each element's
/// position. Insertion and stable removal at arbitrary positions are O(n).
pub struct OrderedSet<T> {
    items: Vec<T>,
    positions: FastHashMap<T, usize>,
}

impl<T> Default for OrderedSet<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            positions: FastHashMap::default(),
        }
    }
}

impl<T> OrderedSet<T> {
    /// Creates an empty ordered set without allocating.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns whether no elements are stored.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns the element at zero-based `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    /// Returns the first element.
    pub fn first(&self) -> Option<&T> {
        self.items.first()
    }

    /// Returns the last element.
    pub fn last(&self) -> Option<&T> {
        self.items.last()
    }

    /// Borrows all elements as an ordered slice.
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    /// Consumes the set and returns its elements in order.
    pub fn into_vec(self) -> Vec<T> {
        self.items
    }

    /// Iterates over elements from first to last.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }

    /// Removes every element while retaining allocated storage.
    pub fn clear(&mut self) {
        self.items.clear();
        self.positions.clear();
    }

    /// The first index whose member is not `is_below`, for a set ordered so
    /// that all `is_below` members come first.
    pub fn partition_point(&self, is_below: impl FnMut(&T) -> bool) -> usize {
        self.items.partition_point(is_below)
    }
}

impl<T: Element> OrderedSet<T> {
    /// Creates an empty ordered set sized for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            positions: FastHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Reserves space for at least `additional` more elements in both indices.
    pub fn reserve(&mut self, additional: usize) {
        self.items.reserve(additional);
        self.positions.reserve(additional);
    }

    /// Releases unused allocation from both internal indices.
    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit();
        self.positions.shrink_to_fit();
    }

    /// Checks that the sequence and its reverse position index agree.
    pub fn check_invariants(&self) -> bool {
        self.items.len() == self.positions.len()
            && self
                .items
                .iter()
                .enumerate()
                .all(|(index, item)| self.positions.get(item) == Some(&index))
    }

    /// Returns whether a value equivalent to borrowed `item` is present.
    ///
    /// `Q` is a borrowed lookup form of `T`, such as `str` for `String`.
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.positions.contains_key(item)
    }

    /// Returns the position of borrowed `item`.
    pub fn index_of<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
    {
        self.positions.get(item).copied()
    }

    /// Appends `item` unless it is already present.
    pub fn push(&mut self, item: T) -> bool {
        if self.positions.contains_key(&item) {
            return false;
        }
        self.positions.insert(item.clone(), self.items.len());
        self.items.push(item);
        true
    }

    /// Inserts `item` at `index` (clamped to the end), shifting later
    /// members back. Does nothing if `item` is already present.
    pub fn insert_at(&mut self, index: usize, item: T) -> bool {
        if self.positions.contains_key(&item) {
            return false;
        }
        let index = index.min(self.items.len());
        self.items.insert(index, item.clone());
        self.positions.insert(item, index);
        self.reindex(index + 1..self.items.len());
        true
    }

    /// Replaces the member at `index`, returning the old one. Fails if
    /// `item` is already elsewhere in the set.
    pub fn replace_at(&mut self, index: usize, item: T) -> Option<T> {
        if index >= self.items.len()
            || self
                .positions
                .get(&item)
                .is_some_and(|existing| *existing != index)
        {
            return None;
        }
        let old = std::mem::replace(&mut self.items[index], item.clone());
        self.positions.remove(&old);
        self.positions.insert(item, index);
        Some(old)
    }

    /// Removes `item`, keeping the order of the rest.
    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        match self.index_of(item) {
            Some(index) => self.remove_at(index).is_some(),
            None => false,
        }
    }

    /// Removes the member at `index`, keeping the order of the rest.
    pub fn remove_at(&mut self, index: usize) -> Option<T> {
        if index >= self.items.len() {
            return None;
        }
        let item = self.items.remove(index);
        self.positions.remove(&item);
        self.reindex(index..self.items.len());
        Some(item)
    }

    /// Removes the member at `index` by moving the last member into its
    /// place. O(1), but changes the order.
    pub fn swap_remove_at(&mut self, index: usize) -> Option<T> {
        if index >= self.items.len() {
            return None;
        }
        let item = self.items.swap_remove(index);
        self.positions.remove(&item);
        if index < self.items.len() {
            *self.positions.get_mut(&self.items[index]).unwrap() = index;
        }
        Some(item)
    }

    /// Removes and returns the first element.
    pub fn pop_first(&mut self) -> Option<T> {
        self.remove_at(0)
    }

    /// Removes and returns the last element.
    pub fn pop_last(&mut self) -> Option<T> {
        let item = self.items.pop()?;
        self.positions.remove(&item);
        Some(item)
    }

    /// Retains elements for which `keep` returns `true`, preserving order.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let before = self.items.len();
        self.items.retain(|item| keep(item));
        if self.items.len() != before {
            self.rebuild_positions();
        }
    }

    /// Moves the member at `from` to `to`, shifting the members between.
    pub fn move_index(&mut self, from: usize, to: usize) -> bool {
        let len = self.items.len();
        if from >= len || to >= len || from == to {
            return false;
        }
        if from < to {
            self.items[from..=to].rotate_left(1);
            self.reindex(from..to + 1);
        } else {
            self.items[to..=from].rotate_right(1);
            self.reindex(to..from + 1);
        }
        true
    }

    /// Moves a member to `index` (clamped to the end).
    pub fn move_to(&mut self, item: &T, index: usize) -> bool {
        match self.index_of(item) {
            Some(from) => self.move_index(from, index.min(self.items.len() - 1)),
            None => false,
        }
    }

    /// Moves a member `offset` places: later if positive, earlier if
    /// negative, stopping at either end.
    pub fn move_by(&mut self, item: &T, offset: isize) -> bool {
        match self.index_of(item) {
            Some(from) => {
                let to = from.saturating_add_signed(offset).min(self.items.len() - 1);
                self.move_index(from, to)
            }
            None => false,
        }
    }

    /// Swaps the elements at indices `a` and `b`.
    ///
    /// # Panics
    /// Panics if either index is out of bounds.
    pub fn swap_indices(&mut self, a: usize, b: usize) {
        self.items.swap(a, b);
        *self.positions.get_mut(&self.items[a]).unwrap() = a;
        *self.positions.get_mut(&self.items[b]).unwrap() = b;
    }

    /// Swaps the positions of two members.
    pub fn swap(&mut self, first: &T, second: &T) -> bool {
        match (self.index_of(first), self.index_of(second)) {
            (Some(a), Some(b)) => {
                self.swap_indices(a, b);
                true
            }
            _ => false,
        }
    }

    /// Clones the members in `range` into a new set.
    pub fn slice(&self, range: Range<usize>) -> Self {
        self.items[range].iter().cloned().collect()
    }

    fn reorder(&mut self, reorder: impl FnOnce(&mut [T])) {
        reorder(&mut self.items);
        self.reindex(0..self.items.len());
    }

    fn reindex(&mut self, range: Range<usize>) {
        for index in range {
            *self.positions.get_mut(&self.items[index]).unwrap() = index;
        }
    }

    fn rebuild_positions(&mut self) {
        self.positions.clear();
        self.positions.extend(
            self.items
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, item)| (item, index)),
        );
    }

    fn filtered(&self, mut keep: impl FnMut(&T) -> bool) -> Self {
        self.items
            .iter()
            .filter(|item| keep(item))
            .cloned()
            .collect()
    }
}

impl<T: Element> Collection for OrderedSet<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn contains(&self, item: &T) -> bool {
        self.positions.contains_key(item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }
}

impl<T: Element> CollectionInsert for OrderedSet<T> {
    fn insert(&mut self, item: T) -> bool {
        self.push(item)
    }
}

impl<T: Element> CollectionRemove for OrderedSet<T> {
    fn remove(&mut self, item: &T) -> bool {
        Self::remove(self, item)
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        Self::retain(self, keep)
    }
}

impl<T: Element> UniqueCollection for OrderedSet<T> {}

impl<T: Element> Choose for OrderedSet<T> {}

impl<T: Element> Sequence for OrderedSet<T> {
    fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    fn index_of(&self, item: &T) -> Option<usize> {
        self.positions.get(item).copied()
    }

    /// O(n) thanks to the position index.
    fn is_subsequence_of(&self, other: &Self) -> bool {
        let mut last: Option<usize> = None;
        self.items.iter().all(|item| match other.index_of(item) {
            Some(index) if last.is_none_or(|last| index > last) => {
                last = Some(index);
                true
            }
            _ => false,
        })
    }
}

impl<T: Element> SequenceMut for OrderedSet<T> {
    fn remove_at(&mut self, index: usize) -> Option<T> {
        Self::remove_at(self, index)
    }

    fn pop_last(&mut self) -> Option<T> {
        Self::pop_last(self)
    }
}

impl<T: Element> InsertAt for OrderedSet<T> {
    fn insert_at(&mut self, index: usize, item: T) -> bool {
        Self::insert_at(self, index, item)
    }
}

impl<T: Element> Reorder for OrderedSet<T> {
    fn sort_by<F: FnMut(&T, &T) -> Ordering>(&mut self, compare: F) {
        self.reorder(|items| items.sort_by(compare));
    }

    fn reverse(&mut self) {
        self.reorder(|items| items.reverse());
    }

    fn shuffle(&mut self, random: &mut Random) {
        self.reorder(|items| sampling::shuffle(items, random));
    }
}

/// Results keep this set's order, with new members from `other` after.
impl<T: Element> SetAlgebra for OrderedSet<T> {
    fn union(&self, other: &Self) -> Self {
        let mut output = self.clone();
        output.extend(other.iter().cloned());
        output
    }

    fn intersection(&self, other: &Self) -> Self {
        self.filtered(|item| other.contains(item))
    }

    fn difference(&self, other: &Self) -> Self {
        self.filtered(|item| !other.contains(item))
    }

    fn is_subset(&self, other: &Self) -> bool {
        self.len() <= other.len() && self.items.iter().all(|item| other.contains(item))
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        !self.items.iter().any(|item| other.contains(item))
    }
}

impl_set_operators!([T: Element] OrderedSet<T>);

/// Equal when the members and their order match.
impl<T: Element> PartialEq for OrderedSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.items == other.items
    }
}

impl<T: Element> Eq for OrderedSet<T> {}

impl<T: Element> Hash for OrderedSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.items.hash(state);
    }
}

impl<T> std::ops::Index<usize> for OrderedSet<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.items[index]
    }
}

impl<T: Element> FromIterator<T> for OrderedSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        let mut set = Self::new();
        set.extend(items);
        set
    }
}

impl<T: Element, const N: usize> From<[T; N]> for OrderedSet<T> {
    fn from(items: [T; N]) -> Self {
        items.into_iter().collect()
    }
}

impl<T: Element> Extend<T> for OrderedSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        for item in items {
            self.push(item);
        }
    }
}

impl<T> IntoIterator for OrderedSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a OrderedSet<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}
