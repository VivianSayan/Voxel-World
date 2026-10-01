//! Hashing a structure by what it holds.

use crate::units::digest::ContentHash;

/// A structure that can be reduced to a [`ContentHash`] of its contents.
///
/// The question it answers is "are these two the same?" for things too large to
/// compare directly, or no longer both in memory: has this chunk changed since
/// it was saved, do these two worlds agree at this node, is the copy that came
/// over the network the one that was sent.
///
/// # What the hash must depend on
///
/// The contents, and nothing else. Not the capacity, not the insertion order
/// where the structure does not consider order part of its identity, and not
/// the address of anything. Two structures that compare equal must hash equal,
/// and the hash must be the same on every machine and every run, which is why
/// it is built from [`ContentHash`] rather than from
/// [`Hash`](std::hash::Hash): the standard hasher is seeded per process.
///
/// Order is part of the contents for a sequence and not for a set. A sequence
/// folds its elements in order, so a rearrangement hashes differently; a set
/// combines them commutatively, so two sets built by different routes agree.
/// Each implementation says which it does.
///
/// # What it is not
///
/// Not a checksum of the bytes a structure occupies, and not stable across a
/// change to the structure itself: adding a field to an element changes its
/// hash. Treat a stored hash as valid for one version of the format.
pub trait ContentHashable {
    /// The hash of everything this structure holds.
    fn content_hash(&self) -> ContentHash;

    /// Whether two structures hold the same contents, as far as their hashes
    /// can tell.
    ///
    /// A false positive needs a 128-bit collision, so this is a practical
    /// answer rather than a certainty; where certainty is wanted, compare the
    /// structures.
    fn content_matches(&self, other: &Self) -> bool
    where
        Self: Sized,
    {
        self.content_hash() == other.content_hash()
    }
}
