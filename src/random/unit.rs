//! Drawing a [`Unit`].
//!
//! The half-open draw and the coin that rests on it. Kept out of
//! [`crate::math::unit_interval`] so that the number itself needs no generator —
//! see there for why the type is split at all.

use crate::math::unit_interval::Unit;
use crate::random::distributions::{Distribution, PortableDistribution};
use crate::random::seed::Seed;
use crate::random::source::StochasticSource;
use crate::units::{NoiseValue, Probability};

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
    pub fn decide_from<S: StochasticSource + ?Sized>(self, source: &mut S) -> bool {
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

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Unit {
        Unit::from_word(source.next_u64())
    }
}

/// Integer-only, so the same seed gives the same draw on every target.
impl PortableDistribution for UniformUnit {}

// ---------------------------------------------------------------------------
// Drawing these types
// ---------------------------------------------------------------------------

/// A uniformly random [`Probability`].
///
/// Half-open in `[0, 1)`, like every uniform draw here — a drawn chance is never
/// exactly certain. Built from a [`Unit`] and converted, so it agrees bit for bit
/// with [`UniformUnit`] on the same word, at the 53 bits
/// an `f64` can hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniformProbability;

impl Distribution for UniformProbability {
    type Output = Probability;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Probability {
        source.unit().to_probability()
    }
}

/// Integer-only, so the same seed gives the same draw on every target.
impl PortableDistribution for UniformProbability {}

/// A uniformly random [`NoiseValue`] in `[-1, 1)`.
///
/// Half-open at the top, inheriting the half-open unit draw it is stretched from.
/// This is a *uniform* sample of the range, not a sample of any noise field: it is
/// what to use for jitter, dithering and test data, not for terrain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniformNoise;

impl Distribution for UniformNoise {
    type Output = NoiseValue;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> NoiseValue {
        // `to_unit_value` cannot fail: the draw is half-open, so never one.
        source
            .unit()
            .to_unit_value()
            .expect("a uniform draw is below one")
            .to_noise_value()
    }
}

/// Integer-only, so the same seed gives the same draw on every target.
impl PortableDistribution for UniformNoise {}
