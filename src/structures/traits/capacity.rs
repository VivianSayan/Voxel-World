//! Control over how much room a collection keeps, separate from how much it
//! holds.
//!
//! A voxel world builds and drops collections constantly: a chunk's entities,
//! a frame's visible set, a generation pass's scratch lists. Reserving once for
//! a size that is usually known beats growing by doubling, and releasing the
//! slack afterwards keeps a long session from holding every peak it ever
//! reached.

use crate::structures::traits::collection::Collection;

/// A structure that can be asked to hold room for more entries than it
/// currently has.
///
/// Not tied to [`Collection`]: a map has storage to reserve as much as a set
/// does, and both implement this.
///
/// Capacity is about memory rather than contents: reserving changes nothing an
/// iterator would see, and [`Capacity::capacity`] is never below
/// [`Collection::len`]. A structure whose storage grows in blocks, or one built
/// on a tree rather than an array, may round a request up or ignore it, which
/// is why the only promise is that room for `additional` more elements exists
/// afterwards.
///
/// Distinct from [`FixedCapacity`](super::FixedCapacity), which is about a
/// limit the collection enforces on itself: this trait asks for room, that one
/// says how much room there will ever be.
pub trait Capacity {
    /// An empty structure with room for `capacity` entries already made.
    fn with_capacity(capacity: usize) -> Self
    where
        Self: Sized;

    /// How many entries fit before anything has to grow. Never below what the
    /// structure currently holds.
    fn capacity(&self) -> usize;

    /// Makes room for `additional` elements beyond those already held, so that
    /// adding that many cannot reallocate.
    ///
    /// May reserve more than asked for, and may do nothing if the room is there
    /// already.
    fn reserve(&mut self, additional: usize);

    /// Gives back the room that is not in use, as far as the storage allows.
    ///
    /// Worth doing after a collection has been drained and will be kept: a
    /// chunk's entity list that peaked at a thousand and now holds three has no
    /// reason to keep the thousand.
    fn shrink_to_fit(&mut self);

    /// How many more elements fit before anything has to grow.
    ///
    /// Available only where the structure is a [`Collection`], since that is
    /// what knows how many entries are in use.
    fn spare_capacity(&self) -> usize
    where
        Self: Collection,
    {
        self.capacity().saturating_sub(Collection::len(self))
    }
}

/// A structure that enforces a maximum number of live entries.
///
/// This is deliberately separate from [`Capacity`]. Reservable capacity is an
/// allocation detail; a bound is observable behaviour because inserting at the
/// limit must reject or evict something.
pub trait Bounded {
    /// Number of entries currently retained.
    fn bounded_len(&self) -> usize;

    /// Maximum number of entries that may be retained.
    fn limit(&self) -> usize;

    /// Whether another distinct entry cannot be retained without eviction.
    fn is_full(&self) -> bool {
        self.bounded_len() >= self.limit()
    }
}

/// Insertion into a bounded structure that may displace existing state.
pub trait EvictingInsert: Bounded {
    /// What is submitted for insertion.
    type Input;
    /// The complete outcome, including anything displaced.
    type Output;

    /// Inserts one input according to the structure's eviction policy.
    fn insert_evicting(&mut self, input: Self::Input) -> Self::Output;
}
