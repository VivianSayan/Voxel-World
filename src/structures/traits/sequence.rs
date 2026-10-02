//! Traits for collections whose elements have a position.

use crate::random::source::StochasticSource;
use crate::structures::sampling;
use crate::structures::traits::collection::{Collection, CollectionRemove};
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
}

/// A collection whose order can be randomised.
///
/// # Question
///
/// "Can this be put into a random order, and what does that mean for it?"
///
/// # Why this is not part of [`Reorder`]
///
/// It used to be, alongside `sort` and `reverse`. They are not the same capability.
/// Sorting and reversing are deterministic rearrangements that any ordered collection
/// can do; shuffling needs a source of randomness, and for some collections it is not
/// a meaningful operation at all — a priority queue in a random order is no longer a
/// priority queue.
///
/// Keeping it separate means implementing it is a *statement* that a random order
/// makes sense for the type, rather than an obligation that came with being sortable.
///
/// # Any source
///
/// Generic over [`StochasticSource`], so the same collection can be shuffled by a
/// [`Random`](crate::random::Random) stream or by a
/// [`Seed`](crate::random::seed::Seed)'s cursor. The second is the one that makes a
/// generated world reproducible: the same seed puts a deck in the same order every
/// time it is built.
///
/// # Example
///
/// ```
/// use voxel_world::random::Random;
/// use voxel_world::random::seed::Seed;
/// use voxel_world::structures::traits::Shuffle;
///
/// let mut deck: Vec<u32> = (0..52).collect();
///
/// // From a stream, which moves on with every draw.
/// let mut random = Random::new(Seed::from_integer(1u64));
/// deck.shuffle(&mut random);
///
/// // Or from a seed, which gives the same order for ever.
/// let seed = Seed::from_integer(7u64).child("deck");
/// let mut one: Vec<u32> = (0..52).collect();
/// let mut two: Vec<u32> = (0..52).collect();
///
/// one.shuffle(&mut seed.cursor());
/// two.shuffle(&mut seed.cursor());
///
/// assert_eq!(one, two, "one seed, one ordering");
///
/// // `Seed::shuffle` is the same thing said more briefly.
/// let mut three: Vec<u32> = (0..52).collect();
/// seed.shuffle(&mut three);
///
/// assert_eq!(one, three);
/// ```
pub trait Shuffle {
    /// Puts the elements into a uniformly random order.
    ///
    /// Every ordering is equally likely, in place, one draw per element.
    fn shuffle<S: StochasticSource + ?Sized>(&mut self, source: &mut S);

    /// Fills just the first `count` places with a random selection, and reports how
    /// many that was.
    ///
    /// # Why this exists next to [`Shuffle::shuffle`]
    ///
    /// A full shuffle costs one draw per element. This costs one per *chosen*
    /// element, so taking three spawn points out of a million candidates is three
    /// draws rather than a million.
    ///
    /// What follows the first `count` places is left in whatever order the partial
    /// swaps happened to leave it. That is **not** a valid shuffle of the remainder
    /// and must not be treated as one.
    ///
    /// Returns `count`, or the length if that is smaller.
    fn partial_shuffle<S: StochasticSource + ?Sized>(&mut self, count: usize, source: &mut S) -> usize;
}

/// Shuffled in place, which is what every slice-backed collection here comes down to.
impl<T> Shuffle for [T] {
    fn shuffle<S: StochasticSource + ?Sized>(&mut self, source: &mut S) {
        sampling::shuffle(self, source);
    }

    fn partial_shuffle<S: StochasticSource + ?Sized>(&mut self, count: usize, source: &mut S) -> usize {
        sampling::partial_shuffle(self, count, source)
    }
}

/// Forwards to the slice implementation.
///
/// # Why this is not redundant with `[T]`
///
/// A `Vec` reaches the slice implementation by deref coercion when the receiver is
/// known to be a slice, which covers `vec.shuffle(&mut source)`. It does **not** cover
/// being passed to a generic parameter: [`Seed::shuffle`](crate::random::seed::Seed::shuffle)
/// takes `&mut T where T: Shuffle`, and Rust infers `T = Vec<_>` there rather than
/// coercing, so without this impl `seed.shuffle(&mut vec)` would not compile.
impl<T> Shuffle for Vec<T> {
    fn shuffle<S: StochasticSource + ?Sized>(&mut self, source: &mut S) {
        self.as_mut_slice().shuffle(source);
    }

    fn partial_shuffle<S: StochasticSource + ?Sized>(&mut self, count: usize, source: &mut S) -> usize {
        self.as_mut_slice().partial_shuffle(count, source)
    }
}

/// Forwards to the slice implementation, for the same reason [`Vec`] does: an array
/// passed to a generic parameter is inferred as `[T; N]` rather than unsized to `[T]`.
impl<T, const N: usize> Shuffle for [T; N] {
    fn shuffle<S: StochasticSource + ?Sized>(&mut self, source: &mut S) {
        self.as_mut_slice().shuffle(source);
    }

    fn partial_shuffle<S: StochasticSource + ?Sized>(&mut self, count: usize, source: &mut S) -> usize {
        self.as_mut_slice().partial_shuffle(count, source)
    }
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
