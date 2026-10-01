//! A unique ordered collection that evicts its oldest member at a limit.

use super::OrderedSet;
use crate::structures::traits::{
    Bounded, Collection, CollectionRemove, Element, EvictingInsert, FixedCapacity, Sequence,
    UniqueCollection,
};

/// A FIFO-bounded set: reinserting a present value changes nothing, while a
/// new value at the limit removes the oldest one.
#[derive(Clone, Debug)]
pub struct BoundedOrderedSet<T> {
    values: OrderedSet<T>,
    limit: usize,
}

impl<T: Element> BoundedOrderedSet<T> {
    /// Creates an empty set retaining at most `limit` values.
    pub fn new(limit: usize) -> Self {
        Self {
            values: OrderedSet::with_capacity(limit),
            limit,
        }
    }
    /// Inserts a value and returns the oldest value displaced, if any.
    pub fn insert(&mut self, value: T) -> Option<T> {
        if self.values.contains(&value) {
            return None;
        }
        if self.limit == 0 {
            return Some(value);
        }
        let evicted = (self.values.len() == self.limit)
            .then(|| self.values.pop_first())
            .flatten();
        self.values.push(value);
        evicted
    }
    /// Removes a value.
    pub fn remove(&mut self, value: &T) -> bool {
        self.values.remove(value)
    }
    /// Values from oldest to newest.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.values.iter()
    }
    /// Number of retained values.
    pub fn len(&self) -> usize {
        self.values.len()
    }
    /// Whether no values are retained.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    /// Maximum number retained.
    pub const fn limit(&self) -> usize {
        self.limit
    }
}

impl<T: Element> Bounded for BoundedOrderedSet<T> {
    fn bounded_len(&self) -> usize {
        self.len()
    }
    fn limit(&self) -> usize {
        self.limit
    }
}

impl<T: Element> Collection for BoundedOrderedSet<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.len()
    }

    fn contains(&self, item: &T) -> bool {
        self.values.contains(item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: Element> CollectionRemove for BoundedOrderedSet<T> {
    fn remove(&mut self, item: &T) -> bool {
        self.remove(item)
    }

    fn clear(&mut self) {
        self.values.clear();
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        self.values.retain(keep);
    }
}

impl<T: Element> Sequence for BoundedOrderedSet<T> {
    fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    fn index_of(&self, item: &T) -> Option<usize> {
        self.values.index_of(item)
    }
}

impl<T: Element> UniqueCollection for BoundedOrderedSet<T> {}

impl<T: Element> FixedCapacity for BoundedOrderedSet<T> {
    fn capacity(&self) -> usize {
        self.limit
    }
}

impl<T: Element> EvictingInsert for BoundedOrderedSet<T> {
    type Input = T;
    type Output = Option<T>;
    fn insert_evicting(&mut self, value: T) -> Option<T> {
        self.insert(value)
    }
}
