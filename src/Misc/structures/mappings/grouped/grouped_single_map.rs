//! Each element carries exactly one label, and every label knows its group
//! of elements. For state partitions (active/idle), faction membership and
//! workflow buckets.
//!
//! Structurally this is a `UniqueMultiMap` read from the other side (a
//! label owns many elements, an element has one label), plus a size index
//! so the largest and smallest groups are always at hand.

use crate::misc::structures::buckets::SizeIndex;
use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::mappings::multi::unique_multi_map::UniqueMultiMap;
use crate::misc::structures::traits::{
    Element, GroupSizes, Grouping, Map, MapMut, SharedValueMap, ValueIndexed, ValueIndexedMut,
};

#[derive(Clone, Debug)]
/// Elements `E` partitioned under exactly one label `L` each.
///
/// Groups are reverse-indexed and indexed by size for efficient largest- and
/// smallest-group queries.
pub struct GroupedSingleMap<E, L> {
    groups: UniqueMultiMap<L, E>,
    sizes: SizeIndex<L>,
}

impl<E, L> Default for GroupedSingleMap<E, L> {
    fn default() -> Self {
        Self {
            groups: UniqueMultiMap::default(),
            sizes: SizeIndex::default(),
        }
    }
}

impl<E, L> GroupedSingleMap<E, L> {
    /// Creates an empty grouping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.groups.value_count()
    }

    /// Returns the number of non-empty labels.
    pub fn label_count(&self) -> usize {
        self.groups.len()
    }

    /// Returns whether no elements are labeled.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Each element with its label.
    pub fn iter(&self) -> impl Iterator<Item = (&E, &L)> {
        self.groups.iter_by_value()
    }

    /// Iterates over all labeled elements.
    pub fn elements(&self) -> impl Iterator<Item = &E> {
        self.groups.values()
    }

    /// Removes all elements, labels, and size-index entries.
    pub fn clear(&mut self) {
        self.groups.clear();
        self.sizes.clear();
    }
}

impl<E: Element, L: Element> GroupedSingleMap<E, L> {
    /// Returns whether `element` has a label.
    pub fn contains(&self, element: &E) -> bool {
        self.groups.contains_value(element)
    }

    /// Returns whether `element` has exactly `label`.
    pub fn contains_pair(&self, element: &E, label: &L) -> bool {
        self.groups.contains_pair(label, element)
    }

    /// The label of `element`.
    pub fn get(&self, element: &E) -> Option<&L> {
        self.groups.key_of(element)
    }

    /// The group `element` belongs to.
    pub fn group_of(&self, element: &E) -> Option<&Set<E>> {
        self.groups.get(self.get(element)?)
    }

    /// Labels the element, moving it out of its old group. Returns the old
    /// label.
    pub fn insert(&mut self, element: E, label: L) -> Option<L> {
        let old_label = self.get(&element).cloned();
        if old_label.as_ref() == Some(&label) {
            return old_label;
        }
        if let Some(old_label) = &old_label {
            let old_size = self.group_len(old_label);
            self.sizes.resize(old_label, old_size, old_size - 1);
        }
        let new_size = self.group_len(&label);
        self.sizes.resize(&label, new_size, new_size + 1);
        self.groups.insert_or_move(label, element);
        old_label
    }

    /// Labels the element only if it has no label yet.
    pub fn try_insert(&mut self, element: E, label: L) -> bool {
        if self.contains(&element) {
            return false;
        }
        self.insert(element, label);
        true
    }

    /// Gives every one of `elements` the label `label`.
    pub fn insert_group(&mut self, elements: impl IntoIterator<Item = E>, label: &L) {
        for element in elements {
            self.insert(element, label.clone());
        }
    }

    /// Removes an element, returning its label.
    pub fn remove(&mut self, element: &E) -> Option<L> {
        let label = self.groups.remove_value(element)?;
        let new_size = self.group_len(&label);
        self.sizes.resize(&label, new_size + 1, new_size);
        Some(label)
    }

    /// Removes `(element, label)` only when that exact relationship exists.
    pub fn remove_pair(&mut self, element: &E, label: &L) -> bool {
        self.contains_pair(element, label) && self.remove(element).is_some()
    }

    /// Removes a label and its whole group.
    pub fn remove_group(&mut self, label: &L) -> Option<Set<E>> {
        let group = self.groups.remove(label)?;
        self.sizes.resize(label, group.len(), 0);
        Some(group)
    }

    /// Keeps only the pairs `keep` accepts.
    pub fn retain(&mut self, mut keep: impl FnMut(&E, &L) -> bool) {
        let removed: Vec<E> = self
            .iter()
            .filter(|(element, label)| !keep(element, label))
            .map(|(element, _)| element.clone())
            .collect();
        for element in &removed {
            self.remove(element);
        }
    }
}

impl<E: Element, L: Element> Map for GroupedSingleMap<E, L> {
    type Key = E;
    type Value = L;
    type Mapped = L;

    fn len(&self) -> usize {
        self.groups.value_count()
    }

    fn contains_key(&self, element: &E) -> bool {
        self.contains(element)
    }

    fn get(&self, element: &E) -> Option<&L> {
        Self::get(self, element)
    }

    fn contains_pair(&self, element: &E, label: &L) -> bool {
        Self::contains_pair(self, element, label)
    }

    fn keys(&self) -> impl Iterator<Item = &E> {
        self.groups.values()
    }

    fn pairs(&self) -> impl Iterator<Item = (&E, &L)> {
        self.iter()
    }
}

/// `insert_pair` relabels elements that already have a label.
impl<E: Element, L: Element> MapMut for GroupedSingleMap<E, L> {
    fn insert_pair(&mut self, element: E, label: L) -> bool {
        if self.contains_pair(&element, &label) {
            return false;
        }
        self.insert(element, label);
        true
    }

    fn remove(&mut self, element: &E) -> Option<L> {
        Self::remove(self, element)
    }

    fn remove_pair(&mut self, element: &E, label: &L) -> bool {
        Self::remove_pair(self, element, label)
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&E, &L) -> bool>(&mut self, keep: F) {
        Self::retain(self, keep)
    }
}

impl<E: Element, L: Element> ValueIndexed for GroupedSingleMap<E, L> {
    fn contains_value(&self, label: &L) -> bool {
        self.groups.contains_key(label)
    }

    fn value_count(&self) -> usize {
        self.groups.len()
    }

    fn values(&self) -> impl Iterator<Item = &L> {
        self.groups.keys()
    }
}

impl<E: Element, L: Element> ValueIndexedMut for GroupedSingleMap<E, L> {
    fn remove_value(&mut self, label: &L) -> bool {
        self.remove_group(label).is_some()
    }
}

impl<E: Element, L: Element> SharedValueMap for GroupedSingleMap<E, L> {
    fn keys_of(&self, label: &L) -> Option<&Set<E>> {
        self.groups.get(label)
    }
}

impl<E: Element, L: Element> Grouping for GroupedSingleMap<E, L> {
    type Member = E;
    type Label = L;

    fn group(&self, label: &L) -> Option<&Set<E>> {
        self.groups.get(label)
    }

    fn labels(&self) -> impl Iterator<Item = &L> {
        self.groups.keys()
    }

    fn label_count(&self) -> usize {
        self.groups.len()
    }

    fn groups(&self) -> impl Iterator<Item = (&L, &Set<E>)> {
        self.groups.iter()
    }
}

impl<E: Element, L: Element> GroupSizes for GroupedSingleMap<E, L> {
    fn labels_with_len(&self, len: usize) -> impl Iterator<Item = &L> {
        self.sizes.with_size(len)
    }

    fn largest_group(&self) -> Option<(&L, &Set<E>)> {
        let label = self.sizes.largest()?;
        Some((label, self.groups.get(label)?))
    }

    fn smallest_group(&self) -> Option<(&L, &Set<E>)> {
        let label = self.sizes.smallest()?;
        Some((label, self.groups.get(label)?))
    }
}

impl<E: Element, L: Element> PartialEq for GroupedSingleMap<E, L> {
    fn eq(&self, other: &Self) -> bool {
        self.groups == other.groups
    }
}

impl<E: Element, L: Element> Eq for GroupedSingleMap<E, L> {}

/// Later pairs relabel elements from earlier ones.
impl<E: Element, L: Element> FromIterator<(E, L)> for GroupedSingleMap<E, L> {
    fn from_iter<I: IntoIterator<Item = (E, L)>>(pairs: I) -> Self {
        let mut map = Self::new();
        map.extend(pairs);
        map
    }
}

impl<E: Element, L: Element> Extend<(E, L)> for GroupedSingleMap<E, L> {
    fn extend<I: IntoIterator<Item = (E, L)>>(&mut self, pairs: I) {
        for (element, label) in pairs {
            self.insert(element, label);
        }
    }
}
