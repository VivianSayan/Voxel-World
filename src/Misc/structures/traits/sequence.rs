//! Traits for collections whose elements have a position.

use crate::misc::random::Random;
use crate::misc::structures::traits::collection::{Collection, CollectionRemove};
use std::cmp::Ordering;

/// Elements at dense positions `0..len`, like a slice.
pub trait Sequence: Collection {
    /// Returns the element at zero-based `index`, or `None` if out of bounds.
    fn get(&self, index: usize) -> Option<&Self::Item>;

    /// Returns the first element, or `None` when empty.
    fn first(&self) -> Option<&Self::Item> {
        self.get(0)
    }

    /// Returns the last element, or `None` when empty.
    fn last(&self) -> Option<&Self::Item> {
        self.get(self.len().checked_sub(1)?)
    }

    /// Position of the first occurrence.
    fn index_of(&self, item: &Self::Item) -> Option<usize>;

    /// True when this sequence's elements appear in `other` in the same
    /// order, not necessarily next to each other.
    fn is_subsequence_of(&self, other: &Self) -> bool
    where
        Self::Item: PartialEq,
    {
        let mut theirs = other.elements();
        self.elements()
            .all(|mine| theirs.any(|their| their == mine))
    }

    /// True when this sequence's elements appear in `other` consecutively
    /// and in order.
    fn is_window_of(&self, other: &Self) -> bool
    where
        Self::Item: PartialEq,
    {
        is_window(self.elements(), other.elements())
    }
}

pub(crate) fn is_window<'a, T: PartialEq + 'a>(
    needle: impl Iterator<Item = &'a T>,
    haystack: impl Iterator<Item = &'a T>,
) -> bool {
    let needle: Vec<&T> = needle.collect();
    let haystack: Vec<&T> = haystack.collect();
    needle.is_empty()
        || haystack
            .windows(needle.len())
            .any(|window| window == needle.as_slice())
}

/// Removing by position.
pub trait SequenceMut: Sequence + CollectionRemove {
    /// Removes and returns the element at `index`, or `None` if out of bounds.
    fn remove_at(&mut self, index: usize) -> Option<Self::Item>;

    /// Removes and returns the first element.
    fn pop_first(&mut self) -> Option<Self::Item> {
        self.remove_at(0)
    }

    /// Removes and returns the last element.
    fn pop_last(&mut self) -> Option<Self::Item> {
        self.remove_at(self.len().checked_sub(1)?)
    }
}

/// Inserting at a chosen position.
pub trait InsertAt: SequenceMut {
    /// Inserts at `index` (clamped to the end), shifting later elements
    /// back. Returns whether the sequence changed.
    fn insert_at(&mut self, index: usize, item: Self::Item) -> bool;
}

/// Changing the order of elements without changing which are stored.
/// Sparse sequences reorder values while keeping their indices.
pub trait Reorder: Collection {
    /// Sorts elements using `compare`.
    fn sort_by<F: FnMut(&Self::Item, &Self::Item) -> Ordering>(&mut self, compare: F);

    /// Sorts elements by their natural ordering.
    fn sort(&mut self)
    where
        Self::Item: Ord,
    {
        self.sort_by(Ord::cmp);
    }

    /// Sorts elements by keys produced by `key`.
    fn sort_by_key<K: Ord, F: FnMut(&Self::Item) -> K>(&mut self, mut key: F) {
        self.sort_by(|a, b| key(a).cmp(&key(b)));
    }

    /// Reverses the current element order.
    fn reverse(&mut self);

    /// Randomly shuffles the current order using `random`.
    fn shuffle(&mut self, random: &mut Random);
}

/// A collection that holds at most a fixed number of elements.
pub trait FixedCapacity: Collection {
    /// Maximum number of elements the collection can retain.
    fn capacity(&self) -> usize;

    /// Returns whether `len()` has reached `capacity()`.
    fn is_full(&self) -> bool {
        self.len() >= self.capacity()
    }
}

/// Elements stored at arbitrary integer indices, with gaps allowed.
pub trait SparseIndexed: Collection {
    /// What one index holds: a single element or a set of them.
    type Slot;

    /// Returns the value or set stored at `index`.
    fn slot(&self, index: i64) -> Option<&Self::Slot>;

    /// Returns the number of occupied indices.
    fn slot_count(&self) -> usize;

    /// Returns whether `index` is occupied.
    fn has_index(&self, index: i64) -> bool {
        self.slot(index).is_some()
    }

    /// Returns the lowest occupied index.
    fn first_index(&self) -> Option<i64>;

    /// Returns the highest occupied index.
    fn last_index(&self) -> Option<i64>;

    /// True when `index` lies between the first and last occupied index.
    fn spans(&self, index: i64) -> bool {
        matches!((self.first_index(), self.last_index()), (Some(first), Some(last)) if (first..=last).contains(&index))
    }

    /// Occupied indices in ascending order.
    fn indices(&self) -> impl DoubleEndedIterator<Item = i64>;

    /// Every index holding `item`, ascending.
    fn indices_of(&self, item: &Self::Item) -> impl DoubleEndedIterator<Item = i64>;

    /// Returns the first occupied index containing `item`.
    fn first_index_of(&self, item: &Self::Item) -> Option<i64> {
        self.indices_of(item).next()
    }

    /// Returns the last occupied index containing `item`.
    fn last_index_of(&self, item: &Self::Item) -> Option<i64> {
        self.indices_of(item).next_back()
    }

    /// The smallest free index that is zero or above.
    fn next_free_index(&self) -> i64;
}
