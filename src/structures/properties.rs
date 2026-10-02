//! Where the property traits are claimed: which collections can reserve room,
//! which promise an iteration order that replays, and which can be reduced to a
//! hash of their contents.
//!
//! Kept together rather than spread across the collections so that the answer
//! to "what does this structure promise?" is one file, and so that a new
//! structure's omissions are visible as gaps in a list rather than as nothing
//! at all.

use crate::structures::collections::measured::{FuzzySet, MultiSet, WeightedSet};
use crate::structures::collections::sequences::{
    LabeledOrderedSet, OrderedSet, PriorityQueue, RingBuffer, Scheduler,
};
use crate::structures::collections::sets::{LabelIndexedSet, NestedSet, Set, SubscriptionSet};
use crate::structures::collections::sparse::{SparseSequence, SparseSetSequence};
use crate::structures::hashing::FastHashMap;
use crate::structures::mappings::{
    BiMap, GroupedMultiMap, GroupedSingleMap, MultiMap, PairMap, UniqueMultiMap,
};
use crate::structures::traits::{
    CanonicalOrder, Capacity, Collection, ContentHashable, DeterministicMapOrder,
    DeterministicOrder, Element, StableHash, stable_hash_ordered, stable_hash_unordered,
};
use crate::units::digest::ContentHash;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Capacity
// ---------------------------------------------------------------------------

/// The standard sequences pass every request straight through.
impl<T: PartialEq> Capacity for Vec<T> {
    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }

    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }

    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
    }

    fn shrink_to_fit(&mut self) {
        Vec::shrink_to_fit(self);
    }
}

impl<T: PartialEq> Capacity for VecDeque<T> {
    fn with_capacity(capacity: usize) -> Self {
        VecDeque::with_capacity(capacity)
    }

    fn capacity(&self) -> usize {
        VecDeque::capacity(self)
    }

    fn reserve(&mut self, additional: usize) {
        VecDeque::reserve(self, additional);
    }

    fn shrink_to_fit(&mut self) {
        VecDeque::shrink_to_fit(self);
    }
}

// ---------------------------------------------------------------------------
// Deterministic iteration order
// ---------------------------------------------------------------------------
//
// Everything below iterates a sequence, an ordered tree, or a fixed-seed hash
// table. The first two may also claim canonical order below; hash-backed types
// promise repeatability only within the documented executable/data version.

impl<T> DeterministicOrder for Vec<T> {}
impl<T> DeterministicOrder for VecDeque<T> {}
impl<T> DeterministicOrder for Set<T> {}
impl<T> DeterministicOrder for NestedSet<T> {}
impl<T, L, F> DeterministicOrder for LabelIndexedSet<T, L, F> {}
impl<S, F> DeterministicOrder for SubscriptionSet<S, F> {}
impl<T> DeterministicOrder for OrderedSet<T> {}
impl<T, L> DeterministicOrder for LabeledOrderedSet<T, L> {}
impl<T, P> DeterministicOrder for PriorityQueue<T, P> {}
impl<T> DeterministicOrder for RingBuffer<T> {}
impl<T> DeterministicOrder for Scheduler<T> {}
impl<T> DeterministicOrder for SparseSequence<T> {}
impl<T> DeterministicOrder for SparseSetSequence<T> {}
impl<T> DeterministicOrder for MultiSet<T> {}
impl<T> DeterministicOrder for WeightedSet<T> {}
impl<T> DeterministicOrder for FuzzySet<T> {}

impl<T> CanonicalOrder for Vec<T> {}
impl<T> CanonicalOrder for VecDeque<T> {}
impl<T> CanonicalOrder for OrderedSet<T> {}
impl<T, L> CanonicalOrder for LabeledOrderedSet<T, L> {}
impl<T, P> CanonicalOrder for PriorityQueue<T, P> {}
impl<T> CanonicalOrder for RingBuffer<T> {}
impl<T> CanonicalOrder for Scheduler<T> {}
impl<T> CanonicalOrder for SparseSequence<T> {}
impl<T> CanonicalOrder for SparseSetSequence<T> {}

impl<K, V> DeterministicMapOrder for FastHashMap<K, V> {}
impl<L, R> DeterministicMapOrder for BiMap<L, R> {}
impl<K, V> DeterministicMapOrder for MultiMap<K, V> {}
impl<K, V> DeterministicMapOrder for UniqueMultiMap<K, V> {}
impl<E, L> DeterministicMapOrder for GroupedSingleMap<E, L> {}
impl<E, L> DeterministicMapOrder for GroupedMultiMap<E, L> {}
impl<T> DeterministicMapOrder for PairMap<T> {}

// ---------------------------------------------------------------------------
// Content hashing
// ---------------------------------------------------------------------------

/// Folds elements in order, so a rearrangement hashes differently.
///
/// Each element is hashed on its own and absorbed into the running state, which
/// is what makes the order matter: the state is mixed between elements, so no
/// two orderings reach the same place.
fn hash_in_order<'a, T: StableHash + 'a>(elements: impl Iterator<Item = &'a T>) -> ContentHash {
    stable_hash_ordered(elements)
}

/// Combines elements commutatively, so two collections built by different
/// routes agree.
///
/// [`unordered_hash`] already does exactly this, and already spreads each
/// element before combining, so a set's hash is one call over its members.
fn hash_unordered<'a, T: StableHash + 'a>(elements: impl Iterator<Item = &'a T>) -> ContentHash {
    stable_hash_unordered(elements)
}

/// In order: a sequence's identity includes its order.
impl<T: StableHash + PartialEq> ContentHashable for Vec<T> {
    fn content_hash(&self) -> ContentHash {
        hash_in_order(self.iter())
    }
}

/// In order, oldest first.
impl<T: StableHash + PartialEq> ContentHashable for VecDeque<T> {
    fn content_hash(&self) -> ContentHash {
        hash_in_order(self.iter())
    }
}

/// In order: an ordered set is a sequence that happens to be unique.
impl<T: Element + StableHash> ContentHashable for OrderedSet<T> {
    fn content_hash(&self) -> ContentHash {
        hash_in_order(self.elements())
    }
}

/// In order, oldest first, so a buffer that has evicted differs from one that
/// has not.
impl<T: Element + StableHash> ContentHashable for RingBuffer<T> {
    fn content_hash(&self) -> ContentHash {
        hash_in_order(self.elements())
    }
}

/// Unordered: a set is its members, however they were added.
impl<T: Element + StableHash> ContentHashable for Set<T> {
    fn content_hash(&self) -> ContentHash {
        hash_unordered(self.elements())
    }
}

/// Unordered, as [`Set`].
impl<T: Element + StableHash> ContentHashable for NestedSet<T> {
    fn content_hash(&self) -> ContentHash {
        hash_unordered(self.elements())
    }
}
