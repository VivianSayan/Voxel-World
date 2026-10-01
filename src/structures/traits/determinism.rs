//! A promise about iteration order.

/// A collection whose iteration order replays for the same operations within
/// one data-format and executable version.
///
/// A fixed hash-table seed removes per-process randomisation, but generic
/// [`Hash`](std::hash::Hash) implementations and table internals are not a
/// cross-version storage format. Use [`CanonicalOrder`] when insertion history
/// and hash-table layout must not affect a traversal.
///
/// The standard library's [`HashMap`](std::collections::HashMap) and
/// [`HashSet`](std::collections::HashSet) do **not**. They seed their hasher
/// from the operating system once per process, so their order differs between
/// runs of the same program on the same machine.
///
/// # What it is for
///
/// Anything that has to replay from a seed and walks a collection on the way is
/// only reproducible if that walk is. Picking a spawn point by iterating a set,
/// hashing a chunk's contents, or feeding elements to a sampler in turn all
/// depend on the order, and a structure that does not promise one turns a
/// deterministic world into an intermittently different one.
///
/// As a bound this becomes a compile-time check rather than a habit:
///
/// ```ignore
/// fn place_one<C>(candidates: &C, seed: Seed) -> Option<&C::Item>
/// where
///     C: Choose + DeterministicOrder,
/// {
///     candidates.choose(&mut seed.to_random())
/// }
/// ```
///
/// # Implementing it
///
/// This is a promise, not an algorithm: there is nothing to write. Implement it
/// only where the order truly is settled by the contents, which means the
/// storage underneath is ordered, insertion-ordered, or hashed with a fixed
/// hasher. A structure that iterates a std `HashMap` anywhere inside must not
/// claim it.
pub trait DeterministicOrder {}

/// Iteration determined by logical contents alone.
///
/// Equal values of a type implementing this trait iterate in the same order,
/// regardless of how they were built. This is the appropriate bound for
/// persistent generation decisions and canonical serialization.
pub trait CanonicalOrder: DeterministicOrder {}

/// The map counterpart, for the same reason and with the same promise.
///
/// Separate because a map is not a [`Collection`](super::Collection) in this
/// module's vocabulary, so one bound cannot serve both.
pub trait DeterministicMapOrder {}

/// Map iteration determined by logical key-value contents alone.
pub trait CanonicalMapOrder: DeterministicMapOrder {}
