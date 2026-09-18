//! A unique collection kept in priority order, lowest priority first.
//! Elements with equal priority keep their insertion order.
//!
//! Stored as a B-tree keyed by (priority, insertion number), so push,
//! remove and popping either end are O(log n). Positional access walks the
//! tree and is O(n).
//!
//! Method names follow the `priority-queue` crate and `BTreeMap`. It is
//! also a `Map` from element to priority.

use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, Map, MapMut, Sequence,
    SequenceMut, UniqueCollection,
};
use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::ops::{Bound, RangeBounds};

type Slot<P> = (P, u64);

#[derive(Clone, Debug)]
/// Stable min-priority queue of unique elements `T` ordered by priorities `P`.
///
/// Lower `P` values come first. Equal priorities retain insertion order.
pub struct PriorityQueue<T, P = i64> {
    order: BTreeMap<Slot<P>, T>,
    slots: FastHashMap<T, Slot<P>>,
    next_sequence: u64,
}

impl<T, P> Default for PriorityQueue<T, P> {
    fn default() -> Self {
        Self {
            order: BTreeMap::new(),
            slots: FastHashMap::default(),
            next_sequence: 0,
        }
    }
}

impl<T, P> PriorityQueue<T, P> {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of queued elements.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Returns whether no elements are queued.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Lowest priority first.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&T, &P)> + ExactSizeIterator {
        self.order
            .iter()
            .map(|((priority, _), item)| (item, priority))
    }

    /// O(n).
    pub fn get(&self, index: usize) -> Option<(&T, &P)> {
        self.iter().nth(index)
    }

    /// Removes every entry while retaining hash-index allocation.
    pub fn clear(&mut self) {
        self.order.clear();
        self.slots.clear();
    }

    /// Every entry, lowest priority first.
    pub fn into_sorted_vec(self) -> Vec<(T, P)> {
        self.into_iter().collect()
    }
}

impl<T, P: Ord> PriorityQueue<T, P> {
    /// The element with the lowest priority.
    pub fn first(&self) -> Option<(&T, &P)> {
        self.order
            .first_key_value()
            .map(|((priority, _), item)| (item, priority))
    }

    /// The element with the highest priority.
    pub fn last(&self) -> Option<(&T, &P)> {
        self.order
            .last_key_value()
            .map(|((priority, _), item)| (item, priority))
    }
}

impl<T: Element, P: Ord + Clone> PriorityQueue<T, P> {
    /// Returns whether a value equivalent to borrowed `item` is queued.
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.slots.contains_key(item)
    }

    /// Returns the priority of borrowed `item`.
    pub fn get_priority<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> Option<&P>
    where
        T: Borrow<Q>,
    {
        self.slots.get(item).map(|slot| &slot.0)
    }

    /// O(n).
    pub fn index_of<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
    {
        let slot = self.slots.get(item)?;
        Some(self.order.range(..slot).count())
    }

    /// Queues `item` behind any elements of equal priority. Returns false if
    /// it is already queued; use `change_priority` to move it.
    pub fn push(&mut self, item: T, priority: P) -> bool {
        if self.slots.contains_key(&item) {
            return false;
        }
        if self.order.is_empty() {
            self.next_sequence = 0;
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .expect("priority insertion sequence exhausted");
        let slot = (priority, sequence);
        self.slots.insert(item.clone(), slot.clone());
        self.order.insert(slot, item);
        true
    }

    /// Queues `item`, or moves it if already queued. Returns the old priority.
    pub fn push_or_change(&mut self, item: T, priority: P) -> Option<P> {
        let old = self.remove(&item).map(|(_, old)| old);
        self.push(item, priority);
        old
    }

    /// Moves a queued element behind others of the new priority. Returns the
    /// old priority.
    pub fn change_priority<Q: Hash + Eq + ?Sized>(&mut self, item: &Q, priority: P) -> Option<P>
    where
        T: Borrow<Q>,
    {
        let (item, old) = self.remove(item)?;
        self.push(item, priority);
        Some(old)
    }

    /// Removes an element, returning it with its priority.
    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> Option<(T, P)>
    where
        T: Borrow<Q>,
    {
        let slot = self.slots.remove(item)?;
        let item = self.order.remove(&slot).unwrap();
        Some((item, slot.0))
    }

    /// Removes the element with the lowest priority.
    pub fn pop_first(&mut self) -> Option<(T, P)> {
        let ((priority, _), item) = self.order.pop_first()?;
        self.slots.remove(&item);
        Some((item, priority))
    }

    /// Removes the element with the highest priority.
    pub fn pop_last(&mut self) -> Option<(T, P)> {
        let ((priority, _), item) = self.order.pop_last()?;
        self.slots.remove(&item);
        Some((item, priority))
    }

    /// O(n).
    pub fn remove_at(&mut self, index: usize) -> Option<(T, P)> {
        let item = self.order.values().nth(index)?.clone();
        self.remove(&item)
    }

    /// Retains entries for which `keep(item, priority)` returns `true`.
    pub fn retain(&mut self, mut keep: impl FnMut(&T, &P) -> bool) {
        let slots = &mut self.slots;
        self.order.retain(|(priority, _), item| {
            let kept = keep(item, priority);
            if !kept {
                slots.remove(item);
            }
            kept
        });
    }

    /// Entries whose priority lies in `range`, lowest first. O(log n + matches).
    pub fn range(&self, range: impl RangeBounds<P>) -> impl DoubleEndedIterator<Item = (&T, &P)> {
        // Widen the priority bounds to slot bounds covering every sequence.
        let start = match range.start_bound() {
            Bound::Included(priority) => Bound::Included((priority.clone(), 0)),
            Bound::Excluded(priority) => Bound::Excluded((priority.clone(), u64::MAX)),
            Bound::Unbounded => Bound::Unbounded,
        };
        let end = match range.end_bound() {
            Bound::Included(priority) => Bound::Included((priority.clone(), u64::MAX)),
            Bound::Excluded(priority) => Bound::Excluded((priority.clone(), 0)),
            Bound::Unbounded => Bound::Unbounded,
        };
        self.order
            .range((start, end))
            .map(|((priority, _), item)| (item, priority))
    }
}

impl<T: Element, P: Ord + Clone> Collection for PriorityQueue<T, P> {
    type Item = T;

    fn len(&self) -> usize {
        self.order.len()
    }

    fn contains(&self, item: &T) -> bool {
        self.slots.contains_key(item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.order.values()
    }
}

/// `insert` queues with the default priority.
impl<T: Element, P: Ord + Clone + Default> CollectionInsert for PriorityQueue<T, P> {
    fn insert(&mut self, item: T) -> bool {
        self.push(item, P::default())
    }
}

impl<T: Element, P: Ord + Clone> CollectionRemove for PriorityQueue<T, P> {
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

impl<T: Element, P: Ord + Clone> UniqueCollection for PriorityQueue<T, P> {}

impl<T: Element, P: Ord + Clone> Choose for PriorityQueue<T, P> {}

impl<T: Element, P: Ord + Clone> Sequence for PriorityQueue<T, P> {
    fn get(&self, index: usize) -> Option<&T> {
        self.order.values().nth(index)
    }

    fn first(&self) -> Option<&T> {
        self.order.values().next()
    }

    fn last(&self) -> Option<&T> {
        self.order.values().next_back()
    }

    fn index_of(&self, item: &T) -> Option<usize> {
        Self::index_of(self, item)
    }
}

impl<T: Element, P: Ord + Clone> SequenceMut for PriorityQueue<T, P> {
    fn remove_at(&mut self, index: usize) -> Option<T> {
        Self::remove_at(self, index).map(|(item, _)| item)
    }

    fn pop_first(&mut self) -> Option<T> {
        Self::pop_first(self).map(|(item, _)| item)
    }

    fn pop_last(&mut self) -> Option<T> {
        Self::pop_last(self).map(|(item, _)| item)
    }
}

/// A map from element to priority.
impl<T: Element, P: Ord + Clone> Map for PriorityQueue<T, P> {
    type Key = T;
    type Value = P;
    type Mapped = P;

    fn len(&self) -> usize {
        self.order.len()
    }

    fn contains_key(&self, item: &T) -> bool {
        self.slots.contains_key(item)
    }

    fn get(&self, item: &T) -> Option<&P> {
        self.get_priority(item)
    }

    fn contains_pair(&self, item: &T, priority: &P) -> bool {
        self.get_priority(item) == Some(priority)
    }

    fn keys(&self) -> impl Iterator<Item = &T> {
        self.order.values()
    }

    fn pairs(&self) -> impl Iterator<Item = (&T, &P)> {
        self.iter()
    }
}

/// `insert_pair` queues the element or changes its priority.
impl<T: Element, P: Ord + Clone> MapMut for PriorityQueue<T, P> {
    fn insert_pair(&mut self, item: T, priority: P) -> bool {
        if self.get_priority(&item) == Some(&priority) {
            return false;
        }
        self.push_or_change(item, priority);
        true
    }

    fn remove(&mut self, item: &T) -> Option<P> {
        Self::remove(self, item).map(|(_, priority)| priority)
    }

    fn remove_pair(&mut self, item: &T, priority: &P) -> bool {
        self.get_priority(item) == Some(priority) && Self::remove(self, item).is_some()
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&T, &P) -> bool>(&mut self, keep: F) {
        Self::retain(self, keep)
    }
}

/// Equal when the same elements are queued in the same order with the
/// same priorities.
impl<T: Element, P: Ord + Clone> PartialEq for PriorityQueue<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<T: Element, P: Ord + Clone> Eq for PriorityQueue<T, P> {}

impl<T: Element, P: Ord + Clone + Hash> Hash for PriorityQueue<T, P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (item, priority) in self.iter() {
            item.hash(state);
            priority.hash(state);
        }
    }
}

impl<T: Element, P: Ord + Clone> FromIterator<(T, P)> for PriorityQueue<T, P> {
    fn from_iter<I: IntoIterator<Item = (T, P)>>(pairs: I) -> Self {
        let mut queue = Self::new();
        queue.extend(pairs);
        queue
    }
}

/// Elements already queued are skipped.
impl<T: Element, P: Ord + Clone> Extend<(T, P)> for PriorityQueue<T, P> {
    fn extend<I: IntoIterator<Item = (T, P)>>(&mut self, pairs: I) {
        for (item, priority) in pairs {
            self.push(item, priority);
        }
    }
}

impl<T, P> IntoIterator for PriorityQueue<T, P> {
    type Item = (T, P);
    type IntoIter = std::iter::Map<
        std::collections::btree_map::IntoIter<Slot<P>, T>,
        fn((Slot<P>, T)) -> (T, P),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.order
            .into_iter()
            .map(|((priority, _), item)| (item, priority))
    }
}
