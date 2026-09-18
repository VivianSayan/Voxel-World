//! A unique collection that also files every member under the labels it
//! carries, so members can be fetched by label (see `Grouping`). The labels
//! come from the members themselves through a labeler function, such as a
//! tag list or a category field.
//!
//! This is the Godot `SortedSet`. It is not ordered; it is sorted into
//! label buckets, hence the new name.
//!
//! The labeler reports labels through a callback instead of returning a
//! collection, so filing a member allocates nothing:
//!
//! ```ignore
//! let set = LabelIndexedSet::new(|item: &Item, emit: &mut dyn FnMut(Tag)| {
//!     for tag in &item.tags {
//!         emit(tag.clone());
//!     }
//! });
//! ```
//!
//! A member's labels must not change while it is in the set.

use crate::misc::structures::buckets::SizeIndex;
use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::traits::operators::impl_set_operators;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, GroupSizes, Grouping,
    SetAlgebra, UniqueCollection,
};
use std::fmt;

#[derive(Clone)]
/// Unique elements `T` indexed under derived labels `L`.
///
/// `F` receives an element and an emitter callback; it must emit every label
/// belonging to that element. Use [`Self::reindex_all`] after labels stored
/// behind interior mutability change.
pub struct LabelIndexedSet<T, L, F> {
    members: Set<T>,
    buckets: FastHashMap<L, Set<T>>,
    sizes: SizeIndex<L>,
    labeler: F,
}

impl<T, L, F> LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)),
{
    /// Creates an empty set using `labeler` to derive labels for each member.
    pub fn new(labeler: F) -> Self {
        Self {
            members: Set::new(),
            buckets: FastHashMap::default(),
            sizes: SizeIndex::default(),
            labeler,
        }
    }

    /// Returns the number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns whether there are no members.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Returns whether `item` is a member.
    pub fn contains(&self, item: &T) -> bool {
        self.members.contains(item)
    }

    /// Iterates over all members.
    pub fn iter(&self) -> std::collections::hash_set::Iter<'_, T> {
        self.members.iter()
    }

    /// Borrows the complete member set.
    pub fn members(&self) -> &Set<T> {
        &self.members
    }

    /// Borrows the label-derivation callback.
    pub fn labeler(&self) -> &F {
        &self.labeler
    }

    /// Returns whether the item was new.
    pub fn insert(&mut self, item: T) -> bool {
        if self.members.contains(&item) {
            return false;
        }
        self.index_item(&item);
        self.members.insert(item)
    }

    /// Replaces the equal stored item and refreshes its derived labels.
    /// Returns the previous stored value, if any.
    pub fn replace(&mut self, item: T) -> Option<T> {
        let previous = self.members.take(&item);
        if let Some(old) = &previous {
            self.unindex_item(old);
        }
        self.index_item(&item);
        self.members.insert(item);
        previous
    }

    fn index_item(&mut self, item: &T) {
        let Self {
            buckets,
            sizes,
            labeler,
            ..
        } = self;
        labeler(item, &mut |label| {
            let bucket = buckets.entry(label.clone()).or_default();
            let old = bucket.len();
            if bucket.insert(item.clone()) {
                sizes.resize(&label, old, old + 1);
            }
        });
    }

    /// Removes `item` and its derived-label index entries.
    pub fn remove(&mut self, item: &T) -> bool {
        let Some(item) = self.members.take(item) else {
            return false;
        };
        self.unindex_item(&item);
        true
    }

    fn unindex_item(&mut self, item: &T) {
        let Self {
            buckets,
            sizes,
            labeler,
            ..
        } = self;
        labeler(item, &mut |label| {
            if let Some(bucket) = buckets.get_mut(&label) {
                let old = bucket.len();
                if bucket.remove(item) {
                    sizes.resize(&label, old, old - 1);
                }
                if bucket.is_empty() {
                    buckets.remove(&label);
                }
            }
        });
    }

    /// Rebuilds every derived-label bucket from the current stored members.
    /// Use this after label data behind interior mutability has changed.
    pub fn reindex_all(&mut self) {
        self.buckets.clear();
        self.sizes.clear();
        let Self {
            members,
            buckets,
            sizes,
            labeler,
        } = self;
        for item in members.iter() {
            labeler(item, &mut |label| {
                let bucket = buckets.entry(label.clone()).or_default();
                let old = bucket.len();
                if bucket.insert(item.clone()) {
                    sizes.resize(&label, old, old + 1);
                }
            });
        }
    }

    /// Removes all members and label buckets while retaining allocation.
    pub fn clear(&mut self) {
        self.members.clear();
        self.buckets.clear();
        self.sizes.clear();
    }

    /// Retains members for which `keep` returns `true` and updates all buckets.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let removed: Vec<T> = self
            .members
            .iter()
            .filter(|item| !keep(item))
            .cloned()
            .collect();
        for item in &removed {
            self.remove(item);
        }
    }

    /// Members carrying any of `labels`.
    pub fn with_any_label(&self, labels: &[L]) -> Set<T> {
        let mut output = Set::new();
        for bucket in labels.iter().filter_map(|label| self.buckets.get(label)) {
            output.extend(bucket.iter().cloned());
        }
        output
    }

    /// Members carrying all of `labels`. With no labels, every member.
    pub fn with_all_labels(&self, labels: &[L]) -> Set<T> {
        let buckets: Option<Vec<&Set<T>>> =
            labels.iter().map(|label| self.buckets.get(label)).collect();
        match buckets.as_deref() {
            Some([first, rest @ ..]) => first.intersection_all(rest),
            Some([]) => self.members.clone(),
            None => Set::new(),
        }
    }
}

impl<T, L, F> LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)) + Clone,
{
    fn with_members<'a>(&self, members: impl IntoIterator<Item = &'a T>) -> Self
    where
        T: 'a,
    {
        let mut output = Self::new(self.labeler.clone());
        for item in members {
            output.insert(item.clone());
        }
        output
    }
}

impl<T, L, F> Collection for LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)),
{
    type Item = T;

    fn len(&self) -> usize {
        self.members.len()
    }

    fn contains(&self, item: &T) -> bool {
        self.members.contains(item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.members.iter()
    }
}

impl<T, L, F> CollectionInsert for LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)),
{
    fn insert(&mut self, item: T) -> bool {
        Self::insert(self, item)
    }
}

impl<T, L, F> CollectionRemove for LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)),
{
    fn remove(&mut self, item: &T) -> bool {
        Self::remove(self, item)
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<P: FnMut(&T) -> bool>(&mut self, keep: P) {
        Self::retain(self, keep)
    }
}

impl<T, L, F> UniqueCollection for LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)),
{
}

impl<T, L, F> Choose for LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)),
{
}

impl<T, L, F> Grouping for LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)),
{
    type Member = T;
    type Label = L;

    fn group(&self, label: &L) -> Option<&Set<T>> {
        self.buckets.get(label)
    }

    fn labels(&self) -> impl Iterator<Item = &L> {
        self.buckets.keys()
    }

    fn label_count(&self) -> usize {
        self.buckets.len()
    }

    fn groups(&self) -> impl Iterator<Item = (&L, &Set<T>)> {
        self.buckets.iter()
    }
}

impl<T, L, F> GroupSizes for LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)),
{
    fn labels_with_len(&self, len: usize) -> impl Iterator<Item = &L> {
        self.sizes.with_size(len)
    }

    fn largest_group(&self) -> Option<(&L, &Set<T>)> {
        let label = self.sizes.largest()?;
        Some((label, self.buckets.get(label)?))
    }

    fn smallest_group(&self) -> Option<(&L, &Set<T>)> {
        let label = self.sizes.smallest()?;
        Some((label, self.buckets.get(label)?))
    }
}

/// Results use this set's labeler.
impl<T, L, F> SetAlgebra for LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)) + Clone,
{
    fn union(&self, other: &Self) -> Self {
        let mut output = self.clone();
        output.extend(other.iter().cloned());
        output
    }

    fn intersection(&self, other: &Self) -> Self {
        self.with_members(&self.members.intersection(&other.members))
    }

    fn difference(&self, other: &Self) -> Self {
        self.with_members(&self.members.difference(&other.members))
    }

    fn is_subset(&self, other: &Self) -> bool {
        self.members.is_subset(&other.members)
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        self.members.is_disjoint(&other.members)
    }
}

impl_set_operators!([T: Element, L: Element, F: Fn(&T, &mut dyn FnMut(L)) + Clone] LabelIndexedSet<T, L, F>);

impl<T: fmt::Debug, L: fmt::Debug, F> fmt::Debug for LabelIndexedSet<T, L, F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LabelIndexedSet")
            .field("members", &self.members)
            .field("buckets", &self.buckets)
            .finish_non_exhaustive()
    }
}

/// Compares members only; labelers cannot be compared.
impl<T: Element, L: Element, F> PartialEq for LabelIndexedSet<T, L, F> {
    fn eq(&self, other: &Self) -> bool {
        self.members == other.members
    }
}

impl<T, L, F> Extend<T> for LabelIndexedSet<T, L, F>
where
    T: Element,
    L: Element,
    F: Fn(&T, &mut dyn FnMut(L)),
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        for item in items {
            self.insert(item);
        }
    }
}

impl<'a, T, L, F> IntoIterator for &'a LabelIndexedSet<T, L, F> {
    type Item = &'a T;
    type IntoIter = std::collections::hash_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.members).into_iter()
    }
}
