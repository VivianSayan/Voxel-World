//! Drawing a [`FixedPoint`].
//!
//! The deterministic number type the engine computes with, drawn the same way
//! everything else here is: integers only, so a seed gives the same value on every
//! target. Kept out of [`crate::math::fixed`] so that the number itself needs no
//! generator — the same split as [`crate::random::unit`].
//!
//! # What a "random Fixed" should mean
//!
//! [`UniformFixed`] draws from `[0, 1)`, which is what the rest of the crate means by
//! an unqualified fraction: [`Unit`](crate::math::Unit),
//! [`Probability`](crate::units::Probability) and `next_f64` all do the same. For
//! anything else there is [`UniformFixedRange`], because a draw over a type's *whole*
//! range is almost never the question — a [`Fixed`](crate::math::Fixed) spans `±2^95`, and a value from
//! anywhere in that is not useful for a position, a weight or a speed.

use crate::math::fixed::FixedPoint;
use crate::random::distributions::{Distribution, PortableDistribution};
use crate::random::seed::Seed;
use crate::random::source::StochasticSource;

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// The fraction in `[0, 1)` a random word stands for.
    ///
    /// # Question
    ///
    /// "I have a word of randomness. What fraction is it?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::fraction_from_word(0), Fixed::ZERO);
    ///
    /// // All ones is one step below one, never one itself.
    /// assert_eq!(Fixed::fraction_from_word(u64::MAX), Fixed::ONE - Fixed::from_bits(1));
    /// ```
    ///
    /// # Why the top bits, and why truncated
    ///
    /// The result is the word's leading `FRACTION_BITS` bits read as a count of
    /// `2^-FRACTION_BITS`. That count is below `2^FRACTION_BITS`, so the value is
    /// always under one, which is what keeps the range half-open.
    ///
    /// Going by way of [`Unit`](crate::math::Unit) and
    /// [`Unit::to_fixed`](crate::math::Unit::to_fixed) would be the obvious
    /// implementation and would be wrong: that conversion *rounds*, so a word near the
    /// top carries up to exactly one. Taking bits cannot carry.
    ///
    /// Because both read the same word from the top down, this is
    /// [`Unit::from_word`](crate::math::Unit::from_word) truncated onto the coarser
    /// grid — one draw at two precisions, not two draws.
    pub const fn fraction_from_word(word: u64) -> Self {
        // A layout is between 1 and 64 fractional bits, so this shift is in range and
        // the count that comes out is below `2^FRACTION_BITS`.
        Self::from_bits((word >> (u64::BITS - FRACTION_BITS)) as i128)
    }

    /// The fraction in `[0, 1)` a seed stands for.
    ///
    /// # Question
    ///
    /// "What one fixed fraction belongs to this seed?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    /// use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// let world = Seed::from_integer(1u64).child("moisture");
    /// let here = world.at_position(VoxelPosition3::new(Vector3::new(4, 0, 9)));
    ///
    /// let value = Fixed::fraction_from_seed(here);
    ///
    /// // The same place always has the same value.
    /// assert_eq!(value, Fixed::fraction_from_seed(here));
    /// assert!(value >= Fixed::ZERO && value < Fixed::ONE);
    /// ```
    ///
    /// Reads the seed's cursor, as every other one-off draw in the crate does, so this
    /// is the same word [`Seed::unit`] takes and not an unrelated one.
    pub fn fraction_from_seed(seed: Seed) -> Self {
        Self::fraction_from_word(seed.cursor().next_u64())
    }
}

/// A uniform draw from `[0, 1)`, on the `2^-FRACTION_BITS` grid.
///
/// # Example
///
/// ```
/// use voxel_world::math::{Fixed, FixedPoint};
/// use voxel_world::random::{Random, UniformFixed};
/// use voxel_world::random::seed::Seed;
///
/// let mut random = Random::new(Seed::from_integer(7u64));
///
/// // The default layout, through the convenience method.
/// let a: Fixed = random.fixed();
///
/// // Any other layout, through the distribution.
/// let b: FixedPoint<16> = random.sample(&UniformFixed::<16>);
///
/// assert!(a < Fixed::ONE);
/// assert!(b < FixedPoint::<16>::ONE);
/// ```
///
/// Half-open at the top, and exactly one word per draw with no rejection, so the
/// number of words a sequence consumes does not depend on the values it produced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniformFixed<const FRACTION_BITS: u32>;

impl<const FRACTION_BITS: u32> Distribution for UniformFixed<FRACTION_BITS> {
    type Output = FixedPoint<FRACTION_BITS>;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Self::Output {
        FixedPoint::fraction_from_word(source.next_u64())
    }
}

/// Integer-only, so the same seed gives the same draw on every target.
impl<const FRACTION_BITS: u32> PortableDistribution for UniformFixed<FRACTION_BITS> {}

/// A uniform draw from `[low, high)`, on the `2^-FRACTION_BITS` grid.
///
/// # Question
///
/// "What is a random height between 60 and 80, as a deterministic number?"
///
/// # Example
///
/// ```
/// use voxel_world::math::Fixed;
/// use voxel_world::random::{Random, UniformFixedRange};
/// use voxel_world::random::seed::Seed;
///
/// let mut random = Random::new(Seed::from_integer(3u64));
///
/// let heights = UniformFixedRange::new(Fixed::from_integer(60), Fixed::from_integer(80))
///     .expect("a non-empty range");
///
/// for _ in 0..100 {
///     let height = random.sample(&heights);
///
///     assert!(height >= Fixed::from_integer(60));
///     assert!(height < Fixed::from_integer(80));
/// }
/// ```
///
/// # Every value equally likely, including across the whole range
///
/// The draw is over the *steps* between the bounds, so each representable value in
/// `[low, high)` has the same chance — which a scaled fraction would not give, because
/// multiplying a `[0, 1)` draw by the span rounds several fractions onto the same step
/// and leaves gaps elsewhere.
///
/// The span may be as wide as the type, including from
/// [`MIN`](FixedPoint::MIN) to [`MAX`](FixedPoint::MAX), so the count of steps is
/// taken as a `u128`: as an `i128` that difference would overflow.
///
/// Half-open, like every other range in the crate. `new` returns `None` for an empty
/// or inverted one, since there is no value to return and saying so is better than
/// picking a bound.
///
/// # Cost
///
/// One or two words, with a redraw on the rare rejection that keeps it unbiased — see
/// [`StochasticSource::bounded_u128`]. Unlike [`UniformFixed`], the number of words spent
/// is not fixed, so a sequence of these is not word-for-word alignable with one of
/// something else. The values are still identical on every target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformFixedRange<const FRACTION_BITS: u32> {
    low: i128,
    steps: u128,
}

impl<const FRACTION_BITS: u32> UniformFixedRange<FRACTION_BITS> {
    /// The range `[low, high)`, or `None` if it holds nothing.
    pub fn new(low: FixedPoint<FRACTION_BITS>, high: FixedPoint<FRACTION_BITS>) -> Option<Self> {
        let (low, high): (i128, i128) = (low.to_bits(), high.to_bits());

        if high <= low {
            return None;
        }

        Some(Self {
            low,
            // `high > low`, so the true difference is between 1 and `2^128 - 1`, which
            // is exactly what the wrapping subtraction leaves in a `u128`.
            steps: high.wrapping_sub(low) as u128,
        })
    }

    /// The lowest value the range can produce.
    pub const fn low(self) -> FixedPoint<FRACTION_BITS> {
        FixedPoint::from_bits(self.low)
    }

    /// The bound the range stops below.
    pub const fn high(self) -> FixedPoint<FRACTION_BITS> {
        FixedPoint::from_bits(self.low.wrapping_add(self.steps as i128))
    }
}

impl<const FRACTION_BITS: u32> Distribution for UniformFixedRange<FRACTION_BITS> {
    type Output = FixedPoint<FRACTION_BITS>;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Self::Output {
        let offset: u128 = source.bounded_u128(self.steps);

        // `offset < steps = high - low`, so the sum is below `high` and inside the
        // type; the wrapping add is how two's complement expresses that.
        FixedPoint::from_bits(self.low.wrapping_add(offset as i128))
    }
}

/// Integer-only, so the same seed gives the same draw on every target.
impl<const FRACTION_BITS: u32> PortableDistribution for UniformFixedRange<FRACTION_BITS> {}
