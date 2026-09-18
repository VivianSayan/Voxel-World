//! Each element carries any number of labels, and every label knows its
//! group of elements. A `MultiMap` from element to label, plus a size index
//! over the groups.

use crate::misc::structures::buckets::SizeIndex;
use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::mappings::multi::multi_map::MultiMap;
use crate::misc::structures::traits::{
    Element, GroupSizes, Grouping, Map, MapMut, SharedValueMap, ValueIndexed, ValueIndexedMut,
};
use std::collections::hash_map;

#[derive(Clone, Debug)]
/// Elements `E` indexed under any number of labels `L`.
///
/// Both directions are indexed, and groups are additionally indexed by size.
pub struct GroupedMultiMap<E, L> {
    labels: MultiMap<E, L>,
    sizes: SizeIndex<L>,
}

impl<E, L> Default for GroupedMultiMap<E, L> {
    fn default() -> Self {
        Self {
            labels: MultiMap::default(),
            sizes: SizeIndex::default(),
        }
    }
}

impl<E, L> GroupedMultiMap<E, L> {
    /// Creates an empty multi-label grouping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Returns the number of non-empty labels.
    pub fn label_count(&self) -> usize {
        self.labels.value_count()
    }

    /// Returns the number of element-label relationships.
    pub fn pair_count(&self) -> usize {
        self.labels.pair_count()
    }

    /// Returns whether no elements have labels.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Each element with its labels.
    pub fn iter(&self) -> hash_map::Iter<'_, E, Set<L>> {
        self.labels.iter()
    }

    /// Iterates over every `(element, label)` relationship.
    pub fn pairs(&self) -> impl Iterator<Item = (&E, &L)> {
        self.labels.pairs()
    }

    /// Iterates over elements that have at least one label.
    pub fn elements(&self) -> hash_map::Keys<'_, E, Set<L>> {
        self.labels.keys()
    }

    /// Removes every relationship and size-index entry.
    pub fn clear(&mut self) {
        self.labels.clear();
        self.sizes.clear();
    }
}

impl<E: Element, L: Element> GroupedMultiMap<E, L> {
    /// Returns whether `element` has at least one label.
    pub fn contains(&self, element: &E) -> bool {
        self.labels.contains_key(element)
    }

    /// Returns whether `element` carries `label`.
    pub fn contains_pair(&self, element: &E, label: &L) -> bool {
        self.labels.contains_pair(element, label)
    }

    /// The labels of `element`.
    pub fn get(&self, element: &E) -> Option<&Set<L>> {
        self.labels.get(element)
    }

    /// Every group `element` belongs to, with its label.
    pub fn groups_of<'a>(&'a self, element: &E) -> impl Iterator<Item = (&'a L, &'a Set<E>)> + 'a {
        self.get(element)
            .into_iter()
            .flatten()
            .filter_map(|label| Some((label, self.labels.keys_of(label)?)))
    }

    /// Adds a label to an element. Returns whether it was new.
    pub fn insert(&mut self, element: E, label: L) -> bool {
        let old_size = self.group_len(&label);
        if !self.labels.insert(element, label.clone()) {
            return false;
        }
        self.sizes.resize(&label, old_size, old_size + 1);
        true
    }

    /// Adds every label in `labels` to `element`.
    pub fn insert_all(&mut self, element: &E, labels: impl IntoIterator<Item = L>) {
        for label in labels {
            self.insert(element.clone(), label);
        }
    }

    /// Adds `label` to every element in `elements`.
    pub fn insert_group(&mut self, elements: impl IntoIterator<Item = E>, label: &L) {
        for element in elements {
            self.insert(element, label.clone());
        }
    }

    /// Replaces all of an element's labels.
    pub fn replace_labels(&mut self, element: &E, labels: &Set<L>) {
        let stale: Vec<L> = self
            .get(element)
            .into_iter()
            .flatten()
            .filter(|label| !labels.contains(*label))
            .cloned()
            .collect();
        for label in &stale {
            self.remove_pair(element, label);
        }
        for label in labels {
            self.insert(element.clone(), label.clone());
        }
    }

    /// Removes exactly `(element, label)`, returning whether it existed.
    pub fn remove_pair(&mut self, element: &E, label: &L) -> bool {
        let old_size = self.group_len(label);
        if !self.labels.remove_pair(element, label) {
            return false;
        }
        self.sizes.resize(label, old_size, old_size - 1);
        true
    }

    /// Removes an element from every group, returning its labels.
    pub fn remove(&mut self, element: &E) -> Option<Set<L>> {
        let labels = self.labels.remove(element)?;
        for label in &labels {
            let new_size = self.group_len(label);
            self.sizes.resize(label, new_size + 1, new_size);
        }
        Some(labels)
    }

    /// Removes a label from every element, returning its group.
    pub fn remove_group(&mut self, label: &L) -> Option<Set<E>> {
        let group = self.labels.remove_value(label)?;
        self.sizes.resize(label, group.len(), 0);
        Some(group)
    }

    /// Keeps only the pairs `keep` accepts.
    pub fn retain(&mut self, mut keep: impl FnMut(&E, &L) -> bool) {
        let removed: Vec<(E, L)> = self
            .pairs()
            .filter(|(element, label)| !keep(element, label))
            .map(|(element, label)| (element.clone(), label.clone()))
            .collect();
        for (element, label) in &removed {
            self.remove_pair(element, label);
        }
    }
}

impl<E: Element, L: Element> Map for GroupedMultiMap<E, L> {
    type Key = E;
    type Value = L;
    type Mapped = Set<L>;

    fn len(&self) -> usize {
        self.labels.len()
    }

    fn pair_count(&self) -> usize {
        self.labels.pair_count()
    }

    fn contains_key(&self, element: &E) -> bool {
        self.contains(element)
    }

    fn get(&self, element: &E) -> Option<&Set<L>> {
        Self::get(self, element)
    }

    fn contains_pair(&self, element: &E, label: &L) -> bool {
        Self::contains_pair(self, element, label)
    }

    fn keys(&self) -> impl Iterator<Item = &E> {
        self.labels.keys()
    }

    fn pairs(&self) -> impl Iterator<Item = (&E, &L)> {
        self.labels.pairs()
    }

    fn is_submap(&self, other: &Self) -> bool {
        self.labels.is_submap(&other.labels)
    }
}

impl<E: Element, L: Element> MapMut for GroupedMultiMap<E, L> {
    fn insert_pair(&mut self, element: E, label: L) -> bool {
        self.insert(element, label)
    }

    fn remove(&mut self, element: &E) -> Option<Set<L>> {
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

impl<E: Element, L: Element> ValueIndexed for GroupedMultiMap<E, L> {
    fn contains_value(&self, label: &L) -> bool {
        self.labels.contains_value(label)
    }

    fn value_count(&self) -> usize {
        self.labels.value_count()
    }

    fn values(&self) -> impl Iterator<Item = &L> {
        self.labels.values()
    }
}

impl<E: Element, L: Element> ValueIndexedMut for GroupedMultiMap<E, L> {
    fn remove_value(&mut self, label: &L) -> bool {
        self.remove_group(label).is_some()
    }
}

impl<E: Element, L: Element> SharedValueMap for GroupedMultiMap<E, L> {
    fn keys_of(&self, label: &L) -> Option<&Set<E>> {
        self.labels.keys_of(label)
    }
}

impl<E: Element, L: Element> Grouping for GroupedMultiMap<E, L> {
    type Member = E;
    type Label = L;

    fn group(&self, label: &L) -> Option<&Set<E>> {
        self.labels.keys_of(label)
    }

    fn labels(&self) -> impl Iterator<Item = &L> {
        self.labels.values()
    }

    fn label_count(&self) -> usize {
        self.labels.value_count()
    }

    fn groups(&self) -> impl Iterator<Item = (&L, &Set<E>)> {
        self.labels.iter_by_value()
    }
}

impl<E: Element, L: Element> GroupSizes for GroupedMultiMap<E, L> {
    fn labels_with_len(&self, len: usize) -> impl Iterator<Item = &L> {
        self.sizes.with_size(len)
    }

    fn largest_group(&self) -> Option<(&L, &Set<E>)> {
        let label = self.sizes.largest()?;
        Some((label, self.labels.keys_of(label)?))
    }

    fn smallest_group(&self) -> Option<(&L, &Set<E>)> {
        let label = self.sizes.smallest()?;
        Some((label, self.labels.keys_of(label)?))
    }
}

impl<E: Element, L: Element> PartialEq for GroupedMultiMap<E, L> {
    fn eq(&self, other: &Self) -> bool {
        self.labels == other.labels
    }
}

impl<E: Element, L: Element> Eq for GroupedMultiMap<E, L> {}

impl<E: Element, L: Element> FromIterator<(E, L)> for GroupedMultiMap<E, L> {
    fn from_iter<I: IntoIterator<Item = (E, L)>>(pairs: I) -> Self {
        let mut map = Self::new();
        map.extend(pairs);
        map
    }
}

impl<E: Element, L: Element> Extend<(E, L)> for GroupedMultiMap<E, L> {
    fn extend<I: IntoIterator<Item = (E, L)>>(&mut self, pairs: I) {
        for (element, label) in pairs {
            self.insert(element, label);
        }
    }
}
