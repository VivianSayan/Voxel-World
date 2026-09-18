//! A fixed-capacity history that keeps the most recent values. Pushing onto
//! a full buffer drops the value at the other end. Good for rolling logs,
//! recent events and sampling windows.
//!
//! Method names follow `VecDeque`: the front is the oldest value and the
//! back the newest.

use crate::misc::random::Random;
use crate::misc::structures::sampling;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionRemove, FixedCapacity, Reorder, Sequence, SequenceMut,
};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
/// Fixed-capacity double-ended history of values of type `T`.
///
/// Insertion into a full buffer evicts a value from the opposite end and
/// returns it to the caller.
pub struct RingBuffer<T> {
    values: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Creates an empty buffer that retains at most `capacity` values.
    ///
    /// # Panics
    /// Panics when `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "a ring buffer needs room for at least one value"
        );
        Self {
            values: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Returns the current number of retained values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the buffer contains no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns whether the buffer has reached its fixed capacity.
    pub fn is_full(&self) -> bool {
        self.values.len() == self.capacity
    }

    /// Returns the maximum number of values retained at once.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Adds a newest value, returning the oldest one if it was pushed out.
    pub fn push_back(&mut self, value: T) -> Option<T> {
        let dropped = if self.is_full() {
            self.values.pop_front()
        } else {
            None
        };
        self.values.push_back(value);
        dropped
    }

    /// Adds an oldest value, returning the newest one if it was pushed out.
    pub fn push_front(&mut self, value: T) -> Option<T> {
        let dropped = if self.is_full() {
            self.values.pop_back()
        } else {
            None
        };
        self.values.push_front(value);
        dropped
    }

    /// Inserts at `index` (clamped), returning the oldest value if it was
    /// pushed out to make room.
    pub fn insert_at(&mut self, index: usize, value: T) -> Option<T> {
        let mut index = index.min(self.values.len());
        let dropped = if self.is_full() {
            index = index.saturating_sub(1);
            self.values.pop_front()
        } else {
            None
        };
        self.values.insert(index, value);
        dropped
    }

    /// The oldest value.
    pub fn front(&self) -> Option<&T> {
        self.values.front()
    }

    /// The newest value.
    pub fn back(&self) -> Option<&T> {
        self.values.back()
    }

    /// Chronological access: 0 is the oldest value.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    /// The value `steps` pushes before the newest one; 0 is the newest.
    pub fn get_from_back(&self, steps: usize) -> Option<&T> {
        self.values.get(self.values.len().checked_sub(steps + 1)?)
    }

    /// Walks `offset` places from the newest value, wrapping around both
    /// ends: 1 is the oldest value, -1 the one before the newest.
    pub fn get_wrapping(&self, offset: isize) -> Option<&T> {
        let len = self.values.len() as isize;
        if len == 0 {
            return None;
        }
        self.values.get((len - 1 + offset).rem_euclid(len) as usize)
    }

    /// Iterates chronologically from oldest to newest.
    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, T> {
        self.values.iter()
    }

    /// Removes and returns the oldest value.
    pub fn pop_front(&mut self) -> Option<T> {
        self.values.pop_front()
    }

    /// Removes and returns the newest value.
    pub fn pop_back(&mut self) -> Option<T> {
        self.values.pop_back()
    }

    /// Removes the value at `index`; later values close the gap.
    pub fn remove_at(&mut self, index: usize) -> Option<T> {
        self.values.remove(index)
    }

    /// Removes all values while retaining allocated storage.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Retains values for which `keep` returns `true`, preserving order.
    pub fn retain(&mut self, keep: impl FnMut(&T) -> bool) {
        self.values.retain(keep);
    }

    /// Borrows the underlying deque in chronological order.
    pub fn as_deque(&self) -> &VecDeque<T> {
        &self.values
    }
}

impl<T: PartialEq> RingBuffer<T> {
    /// Returns whether an equal `value` is retained.
    pub fn contains(&self, value: &T) -> bool {
        self.values.contains(value)
    }

    /// Removes the newest occurrence of `value`; later values close the gap.
    pub fn remove(&mut self, value: &T) -> Option<T> {
        let index = self.values.iter().rposition(|stored| stored == value)?;
        self.values.remove(index)
    }
}

impl<T: PartialEq> Collection for RingBuffer<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.values.len()
    }

    fn contains(&self, value: &T) -> bool {
        self.values.contains(value)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }
}

impl<T: PartialEq> CollectionRemove for RingBuffer<T> {
    fn remove(&mut self, value: &T) -> bool {
        Self::remove(self, value).is_some()
    }

    fn clear(&mut self) {
        self.values.clear();
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        self.values.retain(keep);
    }
}

impl<T: PartialEq> Choose for RingBuffer<T> {}

impl<T: PartialEq> FixedCapacity for RingBuffer<T> {
    fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: PartialEq> Sequence for RingBuffer<T> {
    fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    fn first(&self) -> Option<&T> {
        self.values.front()
    }

    fn last(&self) -> Option<&T> {
        self.values.back()
    }

    fn index_of(&self, value: &T) -> Option<usize> {
        self.values.iter().position(|stored| stored == value)
    }
}

impl<T: PartialEq> SequenceMut for RingBuffer<T> {
    fn remove_at(&mut self, index: usize) -> Option<T> {
        self.values.remove(index)
    }

    fn pop_first(&mut self) -> Option<T> {
        self.values.pop_front()
    }

    fn pop_last(&mut self) -> Option<T> {
        self.values.pop_back()
    }
}

impl<T: PartialEq> Reorder for RingBuffer<T> {
    fn sort_by<F: FnMut(&T, &T) -> Ordering>(&mut self, compare: F) {
        self.values.make_contiguous().sort_by(compare);
    }

    fn reverse(&mut self) {
        self.values.make_contiguous().reverse();
    }

    fn shuffle(&mut self, random: &mut Random) {
        sampling::shuffle(self.values.make_contiguous(), random);
    }
}

/// Buffers are equal when they hold the same values in the same order,
/// whatever their capacity.
impl<T: PartialEq> PartialEq for RingBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<T: Eq> Eq for RingBuffer<T> {}

impl<T: Hash> Hash for RingBuffer<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.values.hash(state);
    }
}

impl<T> std::ops::Index<usize> for RingBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.values[index]
    }
}

/// Pushes each value as the newest.
impl<T> Extend<T> for RingBuffer<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        for value in values {
            self.push_back(value);
        }
    }
}

impl<T> IntoIterator for RingBuffer<T> {
    type Item = T;
    type IntoIter = std::collections::vec_deque::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a RingBuffer<T> {
    type Item = &'a T;
    type IntoIter = std::collections::vec_deque::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}
