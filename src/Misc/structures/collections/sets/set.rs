//! A unique, unordered collection: the base structure most others build on.
//! Other structures use it for their buckets, and any collection can be
//! turned into one with `collect()`.
//!
//! Its API mirrors `HashSet`, with owned results for the set algebra (see
//! `SetAlgebra`) and `choose` for random picks (see `Choose`).

use crate::misc::structures::hashing::{FastHashSet, unordered_hash};
use crate::misc::structures::traits::operators::impl_set_operators;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, Key, SetAlgebra,
    UniqueCollection,
};
use std::borrow::Borrow;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
/// Unordered collection containing at most one equal value of type `T`.
///
/// `T` must implement [`Key`] for lookup and mutation. Operations that create
/// owned result sets additionally require [`Clone`].
pub struct Set<T> {
    members: FastHashSet<T>,
}

impl<T> Default for Set<T> {
    fn default() -> Self {
        Self {
            members: FastHashSet::default(),
        }
    }
}

impl<T> Set<T> {
    /// Creates an empty set without allocating.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of stored members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns whether the set contains no members.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Iterates over members in deterministic hash-table order.
    pub fn iter(&self) -> std::collections::hash_set::Iter<'_, T> {
        self.members.iter()
    }

    /// Removes all members while retaining allocated storage.
    pub fn clear(&mut self) {
        self.members.clear();
    }

    /// Borrows the underlying deterministic hash set.
    pub fn as_hash_set(&self) -> &FastHashSet<T> {
        &self.members
    }

    /// Consumes this wrapper and returns its underlying hash set.
    pub fn into_hash_set(self) -> FastHashSet<T> {
        self.members
    }
}

impl<T: Key> Set<T> {
    /// Creates an empty set sized for at least `capacity` members.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            members: FastHashSet::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Reserves capacity for at least `additional` more members.
    pub fn reserve(&mut self, additional: usize) {
        self.members.reserve(additional);
    }

    /// Releases unused allocation as far as the allocator permits.
    pub fn shrink_to_fit(&mut self) {
        self.members.shrink_to_fit();
    }

    /// Returns whether a value equivalent to borrowed `item` is present.
    ///
    /// `Q` is a borrowed lookup form of `T`, such as `str` for `String`.
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.members.contains(item)
    }

    /// The stored copy of `item`.
    pub fn get<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
    {
        self.members.get(item)
    }

    /// Returns whether the item was new.
    pub fn insert(&mut self, item: T) -> bool {
        self.members.insert(item)
    }

    /// Inserts `item`, returning the equal member it replaced.
    pub fn replace(&mut self, item: T) -> Option<T> {
        self.members.replace(item)
    }

    /// Removes a value equivalent to borrowed `item`.
    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.members.remove(item)
    }

    /// Removes `item` and returns the stored copy.
    pub fn take<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> Option<T>
    where
        T: Borrow<Q>,
    {
        self.members.take(item)
    }

    /// Retains only members for which `keep` returns `true`.
    pub fn retain(&mut self, keep: impl FnMut(&T) -> bool) {
        self.members.retain(keep);
    }

    /// Removes every member and returns an iterator yielding the owned values.
    pub fn drain(&mut self) -> std::collections::hash_set::Drain<'_, T> {
        self.members.drain()
    }

    /// Every combination taking one member from this set, then one from
    /// each of `others` in order.
    pub fn cartesian_product(&self, others: &[&Self]) -> Set<Vec<T>>
    where
        T: Clone,
    {
        let mut tuples: Vec<Vec<T>> = self.iter().map(|item| vec![item.clone()]).collect();

        for set in others {
            let mut next: Vec<Vec<T>> = Vec::with_capacity(tuples.len() * set.len());
            for tuple in &tuples {
                for item in set.iter() {
                    let mut extended = Vec::with_capacity(tuple.len() + 1);
                    extended.extend_from_slice(tuple);
                    extended.push(item.clone());
                    next.push(extended);
                }
            }
            tuples = next;
        }

        tuples.into_iter().collect()
    }

    fn filtered(&self, mut keep: impl FnMut(&T) -> bool) -> Self
    where
        T: Clone,
    {
        self.iter().filter(|item| keep(item)).cloned().collect()
    }
}

impl<T: Key> Collection for Set<T> {
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

impl<T: Key> CollectionInsert for Set<T> {
    fn insert(&mut self, item: T) -> bool {
        self.members.insert(item)
    }
}

impl<T: Key> CollectionRemove for Set<T> {
    fn remove(&mut self, item: &T) -> bool {
        self.members.remove(item)
    }

    fn clear(&mut self) {
        self.members.clear();
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        self.members.retain(keep);
    }
}

impl<T: Key> UniqueCollection for Set<T> {}

impl<T: Key> Choose for Set<T> {}

impl<T: Element> SetAlgebra for Set<T> {
    fn union(&self, other: &Self) -> Self {
        let (larger, smaller) = if self.len() >= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut output = larger.clone();
        output.extend(smaller.iter().cloned());
        output
    }

    fn intersection(&self, other: &Self) -> Self {
        let (smaller, larger) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        smaller.filtered(|item| larger.contains(item))
    }

    fn difference(&self, other: &Self) -> Self {
        self.filtered(|item| !other.contains(item))
    }

    fn symmetric_difference(&self, other: &Self) -> Self {
        self.members
            .symmetric_difference(&other.members)
            .cloned()
            .collect()
    }

    fn is_subset(&self, other: &Self) -> bool {
        self.members.is_subset(&other.members)
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        self.members.is_disjoint(&other.members)
    }

    fn union_all(&self, others: &[&Self]) -> Self {
        let capacity = others
            .iter()
            .fold(self.len(), |total, set| total.saturating_add(set.len()));
        let mut output = Self::with_capacity(capacity);
        output.extend(self.iter().cloned());
        for set in others {
            output.extend(set.iter().cloned());
        }
        output
    }

    /// Filters the smallest set against the others, so the cost follows the
    /// smallest input.
    fn intersection_all(&self, others: &[&Self]) -> Self {
        let smallest =
            others.iter().copied().fold(
                self,
                |best, set| if set.len() < best.len() { set } else { best },
            );
        smallest.filtered(|item| {
            std::iter::once(self)
                .chain(others.iter().copied())
                .all(|set| set.contains(item))
        })
    }
}

impl_set_operators!([T: Element] Set<T>);

impl<T: Key> PartialEq for Set<T> {
    fn eq(&self, other: &Self) -> bool {
        self.members == other.members
    }
}

impl<T: Key> Eq for Set<T> {}

/// Order-independent, so equal sets hash equally however they were built.
impl<T: Key> Hash for Set<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(unordered_hash(&self.members));
    }
}

impl<T: Key> FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self {
            members: items.into_iter().collect(),
        }
    }
}

impl<T: Key, const N: usize> From<[T; N]> for Set<T> {
    fn from(items: [T; N]) -> Self {
        items.into_iter().collect()
    }
}

impl<T> From<FastHashSet<T>> for Set<T> {
    fn from(members: FastHashSet<T>) -> Self {
        Self { members }
    }
}

impl<T: Key> Extend<T> for Set<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        self.members.extend(items);
    }
}

impl<'a, T: Element + 'a> Extend<&'a T> for Set<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, items: I) {
        self.members.extend(items.into_iter().cloned());
    }
}

impl<T> IntoIterator for Set<T> {
    type Item = T;
    type IntoIter = std::collections::hash_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.members.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Set<T> {
    type Item = &'a T;
    type IntoIter = std::collections::hash_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.members.iter()
    }
}
