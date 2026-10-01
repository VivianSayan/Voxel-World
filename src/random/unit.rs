//! Drawing a [`Unit`].
//!
//! The half-open draw and the coin that rests on it. Kept out of
//! [`crate::math::unit_interval`] so that the number itself needs no generator —
//! see there for why the type is split at all.

use crate::math::unit_interval::Unit;
use crate::random::distributions::{Distribution, PortableDistribution};
use crate::random::seed::Seed;
use crate::random::source::RandomSource;

impl Unit {
    /// The fraction a seed stands for, in `[0, 1)`.
    ///
    /// Reads the same word as [`Seed::unit_f64`] — both draw from the seed's
    /// cursor — so the two are the same number at different precisions rather than
    /// two unrelated draws. Earlier this folded the seed directly instead, which
    /// gave a different word from every other one-off fraction in the crate.
    pub fn from_seed(seed: Seed) -> Self {
        Self::from_word(seed.cursor().next_u64())
    }
    /// Whether an event at this chance happens, for the given seed.
    ///
    /// Exact: the probability of a yes is this value, to the last bit, for every
    /// representable chance including zero and one. See the module documentation
    /// for why the half-open draw is what makes that work.
    pub fn decide(self, seed: Seed) -> bool {
        Self::from_seed(seed) < self
    }

    /// Whether an event at this chance happens, drawing from a source.
    pub fn decide_from<S: RandomSource + ?Sized>(self, source: &mut S) -> bool {
        Self::from_word(source.next_u64()) < self
    }
}

/// A uniform draw from `[0, 1)`, on the `2^-63` grid.
///
/// Half-open at the top, which is what makes [`Unit::decide`] exact — see the
/// module documentation. Every draw costs exactly one word, with no rejection, so
/// the number of words a sequence consumes does not depend on the values drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniformUnit;

impl Distribution for UniformUnit {
    type Output = Unit;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> Unit {
        Unit::from_word(source.next_u64())
    }
}

/// Integer-only, so the same seed gives the same draw on every target.
impl PortableDistribution for UniformUnit {}
