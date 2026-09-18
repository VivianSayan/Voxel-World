//! An `OrderedSet` where every element also carries a unique label, with
//! lookups in both directions. It is a sequence of elements and, at the
//! same time, a one-to-one map from element to label.

use crate::misc::random::Random;
use crate::misc::structures::collections::sequences::ordered_set::OrderedSet;
use crate::misc::structures::mappings::single::bi_map::BiMap;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, Map, MapMut, Reorder,
    Sequence, SequenceMut, UniqueCollection, UniqueValueMap, ValueIndexed,
};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
/// Unique ordered elements of type `T`, each paired one-to-one with a label
/// of type `L`.
///
/// Both elements and labels support O(1) lookup; positional insertion and
/// stable removal are O(n).
pub struct LabeledOrderedSet<T, L> {
    order: OrderedSet<T>,
    labels: BiMap<T, L>,
}

impl<T, L> Default for LabeledOrderedSet<T, L> {
    fn default() -> Self {
        Self {
            order: OrderedSet::default(),
            labels: BiMap::default(),
        }
    }
}

impl<T: Element, L: Element> LabeledOrderedSet<T, L> {
    /// Creates an empty labeled ordered set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of element-label pairs.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Returns whether no pairs are stored.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Returns whether `item` is stored.
    pub fn contains(&self, item: &T) -> bool {
        self.order.contains(item)
    }

    /// Returns whether `label` is assigned.
    pub fn contains_label(&self, label: &L) -> bool {
        self.labels.contains_right(label)
    }

    /// Returns the label assigned to `item`.
    pub fn label_of(&self, item: &T) -> Option<&L> {
        self.labels.get_by_left(item)
    }

    /// Returns the element assigned to `label`.
    pub fn get_by_label(&self, label: &L) -> Option<&T> {
        self.labels.get_by_right(label)
    }

    /// Returns the zero-based position of `item`.
    pub fn index_of(&self, item: &T) -> Option<usize> {
        self.order.index_of(item)
    }

    /// Returns the element-label pair at zero-based `index`.
    pub fn get(&self, index: usize) -> Option<(&T, &L)> {
        let item = self.order.get(index)?;
        Some((item, self.labels.get_by_left(item).unwrap()))
    }

    /// Returns the first element-label pair.
    pub fn first(&self) -> Option<(&T, &L)> {
        self.get(0)
    }

    /// Returns the last element-label pair.
    pub fn last(&self) -> Option<(&T, &L)> {
        self.get(self.len().checked_sub(1)?)
    }

    /// Elements with their labels, in order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&T, &L)> + ExactSizeIterator {
        self.order
            .iter()
            .map(|item| (item, self.labels.get_by_left(item).unwrap()))
    }

    /// The elements alone, as an ordered set.
    pub fn as_ordered_set(&self) -> &OrderedSet<T> {
        &self.order
    }

    /// Appends an element. Fails if the element or the label is taken.
    pub fn push(&mut self, item: T, label: L) -> bool {
        self.insert_at(usize::MAX, item, label)
    }

    /// Inserts at `index` (clamped). Fails if the element or the label is taken.
    pub fn insert_at(&mut self, index: usize, item: T, label: L) -> bool {
        if self.contains(&item) || self.contains_label(&label) {
            return false;
        }
        self.order.insert_at(index, item.clone());
        self.labels.insert(item, label);
        true
    }

    /// Replaces the entry at `index`, returning the old one. Fails if the
    /// new element or label belongs to a different entry.
    pub fn replace_at(&mut self, index: usize, item: T, label: L) -> Option<(T, L)> {
        let old = self.order.get(index)?.clone();
        if item != old && self.contains(&item) {
            return None;
        }
        if self
            .labels
            .get_by_right(&label)
            .is_some_and(|owner| *owner != old)
        {
            return None;
        }

        let (_, old_label) = self.labels.remove_by_left(&old).unwrap();
        if item != old {
            self.order.replace_at(index, item.clone());
        }
        self.labels.insert(item, label);
        Some((old, old_label))
    }

    /// Removes an element, returning its label.
    pub fn remove(&mut self, item: &T) -> Option<L> {
        if !self.order.remove(item) {
            return None;
        }
        self.labels.remove_by_left(item).map(|(_, label)| label)
    }

    /// Removes the element with `label`, returning it.
    pub fn remove_by_label(&mut self, label: &L) -> Option<T> {
        let (item, _) = self.labels.remove_by_right(label)?;
        self.order.remove(&item);
        Some(item)
    }

    /// Removes and returns the pair at `index`, preserving remaining order.
    pub fn remove_at(&mut self, index: usize) -> Option<(T, L)> {
        let item = self.order.remove_at(index)?;
        let (_, label) = self.labels.remove_by_left(&item).unwrap();
        Some((item, label))
    }

    /// Removes and returns the first pair.
    pub fn pop_first(&mut self) -> Option<(T, L)> {
        self.remove_at(0)
    }

    /// Removes and returns the last pair.
    pub fn pop_last(&mut self) -> Option<(T, L)> {
        self.remove_at(self.len().checked_sub(1)?)
    }

    /// Removes all pairs while retaining allocated storage.
    pub fn clear(&mut self) {
        self.order.clear();
        self.labels.clear();
    }

    /// Retains pairs for which `keep(item, label)` returns `true`.
    pub fn retain(&mut self, mut keep: impl FnMut(&T, &L) -> bool) {
        let labels = &mut self.labels;
        self.order.retain(|item| {
            let kept = keep(item, labels.get_by_left(item).unwrap());
            if !kept {
                labels.remove_by_left(item);
            }
            kept
        });
    }

    /// Moves the pair at `from` to `to`, shifting intervening pairs.
    pub fn move_index(&mut self, from: usize, to: usize) -> bool {
        self.order.move_index(from, to)
    }

    /// Swaps the pairs at indices `a` and `b`.
    ///
    /// # Panics
    /// Panics if either index is out of bounds.
    pub fn swap_indices(&mut self, a: usize, b: usize) {
        self.order.swap_indices(a, b);
    }

    /// See `OrderedSet::partition_point`.
    pub fn partition_point(&self, is_below: impl FnMut(&T) -> bool) -> usize {
        self.order.partition_point(is_below)
    }

    /// Sorts pairs by comparing their labels with `compare`.
    pub fn sort_by_label(&mut self, mut compare: impl FnMut(&L, &L) -> Ordering) {
        let labels = &self.labels;
        self.order.sort_by(|a, b| {
            compare(
                labels.get_by_left(a).unwrap(),
                labels.get_by_left(b).unwrap(),
            )
        });
    }
}

impl<T: Element> LabeledOrderedSet<T, T> {
    /// Appends an element that is its own label.
    pub fn push_self_labeled(&mut self, item: T) -> bool {
        self.push(item.clone(), item)
    }
}

impl<T: Element, L: Element> Collection for LabeledOrderedSet<T, L> {
    type Item = T;

    fn len(&self) -> usize {
        self.order.len()
    }

    fn contains(&self, item: &T) -> bool {
        self.order.contains(item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.order.iter()
    }
}

/// Generic insertion is only meaningful when elements are their own labels.
impl<T: Element> CollectionInsert for LabeledOrderedSet<T, T> {
    fn insert(&mut self, item: T) -> bool {
        self.push_self_labeled(item)
    }
}

impl<T: Element, L: Element> CollectionRemove for LabeledOrderedSet<T, L> {
    fn remove(&mut self, item: &T) -> bool {
        Self::remove(self, item).is_some()
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, mut keep: F) {
        Self::retain(self, |item, _| keep(item))
    }
}

impl<T: Element, L: Element> SequenceMut for LabeledOrderedSet<T, L> {
    fn remove_at(&mut self, index: usize) -> Option<T> {
        Self::remove_at(self, index).map(|(item, _)| item)
    }
}

impl<T: Element, L: Element> UniqueCollection for LabeledOrderedSet<T, L> {}

impl<T: Element, L: Element> Choose for LabeledOrderedSet<T, L> {}

impl<T: Element, L: Element> Sequence for LabeledOrderedSet<T, L> {
    fn get(&self, index: usize) -> Option<&T> {
        self.order.get(index)
    }

    fn index_of(&self, item: &T) -> Option<usize> {
        self.order.index_of(item)
    }
}

/// Reorders by element; see `sort_by_label` to order by label.
impl<T: Element, L: Element> Reorder for LabeledOrderedSet<T, L> {
    fn sort_by<F: FnMut(&T, &T) -> Ordering>(&mut self, compare: F) {
        self.order.sort_by(compare);
    }

    fn reverse(&mut self) {
        self.order.reverse();
    }

    fn shuffle(&mut self, random: &mut Random) {
        self.order.shuffle(random);
    }
}

impl<T: Element, L: Element> Map for LabeledOrderedSet<T, L> {
    type Key = T;
    type Value = L;
    type Mapped = L;

    fn len(&self) -> usize {
        self.order.len()
    }

    fn contains_key(&self, item: &T) -> bool {
        self.order.contains(item)
    }

    fn get(&self, item: &T) -> Option<&L> {
        self.label_of(item)
    }

    fn contains_pair(&self, item: &T, label: &L) -> bool {
        self.labels.contains_pair(item, label)
    }

    fn keys(&self) -> impl Iterator<Item = &T> {
        self.order.iter()
    }

    fn pairs(&self) -> impl Iterator<Item = (&T, &L)> {
        self.iter()
    }
}

/// `insert_pair` appends, first removing entries that clash with the pair.
impl<T: Element, L: Element> MapMut for LabeledOrderedSet<T, L> {
    fn insert_pair(&mut self, item: T, label: L) -> bool {
        if self.labels.contains_pair(&item, &label) {
            return false;
        }
        self.remove(&item);
        self.remove_by_label(&label);
        self.push(item, label)
    }

    fn remove(&mut self, item: &T) -> Option<L> {
        Self::remove(self, item)
    }

    fn remove_pair(&mut self, item: &T, label: &L) -> bool {
        self.labels.contains_pair(item, label) && Self::remove(self, item).is_some()
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&T, &L) -> bool>(&mut self, keep: F) {
        Self::retain(self, keep)
    }
}

impl<T: Element, L: Element> ValueIndexed for LabeledOrderedSet<T, L> {
    fn contains_value(&self, label: &L) -> bool {
        self.contains_label(label)
    }

    fn value_count(&self) -> usize {
        self.order.len()
    }

    fn values(&self) -> impl Iterator<Item = &L> {
        self.iter().map(|(_, label)| label)
    }
}

impl<T: Element, L: Element> UniqueValueMap for LabeledOrderedSet<T, L> {
    fn key_of(&self, label: &L) -> Option<&T> {
        self.get_by_label(label)
    }
}

impl<T: Element, L: Element> PartialEq for LabeledOrderedSet<T, L> {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order && self.labels == other.labels
    }
}

impl<T: Element, L: Element> Eq for LabeledOrderedSet<T, L> {}

impl<T, L> std::ops::Index<usize> for LabeledOrderedSet<T, L> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.order[index]
    }
}

impl<T: Element, L: Element> FromIterator<(T, L)> for LabeledOrderedSet<T, L> {
    fn from_iter<I: IntoIterator<Item = (T, L)>>(pairs: I) -> Self {
        let mut set = Self::new();
        set.extend(pairs);
        set
    }
}

/// Pairs clashing with an earlier entry are skipped.
impl<T: Element, L: Element> Extend<(T, L)> for LabeledOrderedSet<T, L> {
    fn extend<I: IntoIterator<Item = (T, L)>>(&mut self, pairs: I) {
        for (item, label) in pairs {
            self.push(item, label);
        }
    }
}
