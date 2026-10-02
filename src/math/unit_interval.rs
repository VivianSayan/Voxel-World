//! A number in `[0, 1]`, held as an integer.
//!
//! # Where the rest of it lives
//!
//! This file holds the *number*: its arithmetic, and its exact relationships with
//! [`Ratio`] and [`Fixed`]. Two other things hang off it, and they live where their
//! dependencies do rather than here:
//!
//! - **drawing one** — `Unit::from_seed`, `decide`, and `UniformUnit` — is in
//!   [`crate::random::unit`];
//! - **the bridge to [`Probability`](crate::units::Probability)** is in
//!   [`crate::units::scalar`].
//!
//! Splitting them is what lets [`crate::math`] depend on nothing: a number should
//! not need a random generator to exist. Rustdoc still gathers every method onto
//! the one type, so the split shows in the source and not in the documentation.
//!
//! [`Unit`] is the closed unit interval with `2^63` evenly spaced steps: an
//! ordered, integer-backed alternative to [`Probability`](crate::units::Probability), which holds the same
//! range in an `f64`.
//!
//! What is exact here is the *arithmetic* — complements, and every dyadic rational
//! down to the step. Conversion to and from `f64` is **not** exact in general, in
//! either direction; see [`Unit::from_probability`] for exactly where it loses
//! information and why.
//!
//! # Why an integer rather than an `f64`
//!
//! **The complement is exact.** This is the one that matters. In floating point
//! `1.0 - 1e-20` is `1.0` — the small probability is simply gone, and with it any
//! chance of getting it back. A [`Unit`] complement is `SCALE - k`, a subtraction
//! of integers, which is lossless for every value. Since `and`, `or` and
//! `in_any_of` are all built on complements, they inherit that.
//!
//! **The order is total.** `Unit` is [`Ord`], [`Eq`] and [`Hash`], so it sorts, it
//! keys a [`BTreeMap`](std::collections::BTreeMap), and it has no value that fails
//! to equal itself. [`Probability`](crate::units::Probability) is `PartialOrd` only, because an `f64` drags
//! NaN along with it.
//!
//! **It is finer near one.** An `f64` has a step of `2^-53` in `[½, 1]`; a `Unit`
//! has `2^-63` everywhere. So a probability of `1 - 10^-18` is a distinct value
//! here and rounds to exactly `1` in an `f64`.
//!
//! # What it gives up
//!
//! **Tiny probabilities.** The step is `2^-63 ≈ 1.1×10^-19` *everywhere*, so
//! anything below that is zero. An `f64` reaches `10^-308`. If the interesting
//! numbers are the small ones — a rare event's tail, a likelihood product over
//! thousands of terms — an `f64` or a log-space value is the right tool and this
//! is not. The trade is exactly the one [`Fixed`] makes against
//! `f64`: a constant step instead of a constant relative error.
//!
//! # The representation, and why not "all ones is one"
//!
//! A value is `k / 2^63` with `k` in `[0, 2^63]`, so both ends are included and the
//! step is `2^-63`. That needs `2^63 + 1` distinct values, which is one more than
//! 63 bits holds — hence a `u64` with its top half unused rather than 63 bits
//! exactly full.
//!
//! The tempting alternative is the graphics *unorm* convention, where the all-ones
//! pattern means one and a value is `k / (2^n - 1)`. It gives a full 64 bits and a
//! tidy bit pattern, and it is wrong for this job: `2^n - 1` is **odd**, so
//! `1/2` would need a half-integer numerator and is not representable. Neither is
//! `1/4`, `1/8`, or any other dyadic rational. A probability type that cannot say
//! "a half" is not worth the tidy bit pattern.
//!
//! With `2^63` as the denominator every dyadic rational down to the step is exact,
//! which is what makes bit-driven decisions land on clean numbers.
//!
//! # Closed at both ends, but a draw never is
//!
//! The type includes `1`, because certainty is worth being able to say. A *uniform
//! draw* is `[0, 1)` — it can never be exactly one. That asymmetry is deliberate,
//! and it is what makes [`Unit::decide`] exact:
//!
//! ```text
//! draw is uniform over {0, 1, …, 2^63 - 1}      (2^63 values)
//! p is k / 2^63 with k in [0, 2^63]
//! P(draw < p) = k / 2^63 = p                     exactly, for every k
//! ```
//!
//! including `k = 2^63`, where every draw is below `p` and the answer is always
//! yes. A closed draw would break that at both ends.
//!
//! # Which combination is which
//!
//! The operations are named methods rather than operators, following
//! [`Probability`](crate::units::Probability), because `+` and `*` on probabilities are ambiguous in the way
//! that actually causes bugs:
//!
//! | | means | valid when |
//! |---|---|---|
//! | [`Unit::and`] | `p × q` | the events are **independent** |
//! | [`Unit::or`] | `p + q − pq` | the events are **independent** |
//! | [`Unit::checked_add`] | `p + q` | the events are **disjoint** |
//! | [`Unit::complement`] | `1 − p` | always |
//!
//! Adding two probabilities is right only for mutually exclusive events, and using
//! it for independent ones double-counts the overlap. Naming them stops the two
//! being confused, and is why `checked_add` returns `None` past one rather than
//! quietly saturating.

use crate::math::fixed::Fixed;
use crate::math::decimal::{self, DecimalError};
use crate::math::rational::Ratio;
use std::fmt;

/// How many bits of fraction a [`Unit`] carries.
const SHIFT: u32 = 63;

/// The integer standing for one. A value is `k / SCALE`.
const SCALE: u64 = 1 << SHIFT;

/// How many bits separate this layout from [`Fixed`]'s.
///
/// Derived from both widths rather than written as `31`, so that it stays correct
/// if either changes — and fails the build rather than silently misconverting if
/// `Fixed` ever carries more fractional bits than this type.
const FIXED_SHIFT: u32 = SHIFT - crate::math::fixed::FRACTION_BITS;

/// A number in `[0, 1]`, as a count of `2^-63`.
///
/// # Example
///
/// ```
/// use voxel_world::units::Unit;
///
/// let half = Unit::HALF;
/// let quarter = half.and(half);            // independent AND
///
/// assert_eq!(quarter, Unit::from_ratio_capped(
///     voxel_world::units::Ratio::new(1, 4).unwrap()
/// ));
///
/// // The complement is exact, which is the point of the integer backing.
/// let tiny = Unit::one_in(1_000_000_000_000_000_000);
/// assert_eq!(tiny.complement().complement(), tiny);
///
/// // The same round trip in f64 loses the value entirely.
/// let as_float = 1e-18_f64;
/// assert_eq!(1.0 - (1.0 - as_float), 0.0);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Unit(u64);

impl Unit {
    /// Zero — never.
    pub const ZERO: Self = Self(0);

    /// One — always.
    pub const ONE: Self = Self(SCALE);

    /// A half, exactly.
    pub const HALF: Self = Self(SCALE / 2);

    /// The smallest value above zero, `2^-63`.
    pub const STEP: Self = Self(1);

    /// The largest value below one.
    pub const ALMOST_ONE: Self = Self(SCALE - 1);

    /// How many steps the interval is divided into.
    pub const STEPS: u64 = SCALE;

    // -----------------------------------------------------------------------
    // Building
    // -----------------------------------------------------------------------

    /// A value from its raw count of `2^-63`, or `None` above `2^63`.
    ///
    /// The counterpart of [`Unit::to_bits`]. Checked rather than masking, because a
    /// count above the scale is a mistake somewhere and silently folding it would
    /// hide that.
    pub const fn from_bits(count: u64) -> Option<Self> {
        if count > SCALE {
            return None;
        }

        Some(Self(count))
    }

    /// A value from its raw count, clamped to one.
    /// Reads decimal notation exactly, with no float anywhere in between.
    ///
    /// # Question
    ///
    /// "What `Unit` most closely represents the decimal value in this text?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Unit;
    ///
    /// assert_eq!(Unit::from_decimal_str("0.5").unwrap(), Unit::HALF);
    /// assert_eq!(Unit::from_decimal_str("1e-2").unwrap(), Unit::from_decimal_str("0.01").unwrap());
    ///
    /// // Outside the interval is an error, not a clamp.
    /// assert!(Unit::from_decimal_str("1.1").is_err());
    /// assert!(Unit::from_decimal_str("-0.1").is_err());
    /// ```
    ///
    /// # Why this rather than [`Unit::new`]
    ///
    /// [`Unit::new`] takes an `f64` that already exists. This takes the decimal text
    /// itself, so `0.2` is read as the exact rational `2/10` and rounded once onto
    /// the `k / 2^63` grid, rather than being rounded first to the nearest `f64` and
    /// then again to the grid.
    ///
    /// # Rounding
    ///
    /// Nearest, ties to even — the same rule as
    /// [`Unit::try_from_ratio`], so a literal, a parsed string and a converted ratio
    /// naming the same number all land on the same count.
    pub fn from_decimal_str(text: &str) -> Result<Self, DecimalError> {
        decimal::unit_bits_exact(text).map(Self)
    }

    pub const fn from_bits_clamped(count: u64) -> Self {
        if count > SCALE {
            return Self::ONE;
        }

        Self(count)
    }

    /// The raw count of `2^-63` underneath, for storing or hashing.
    pub const fn to_bits(self) -> u64 {
        self.0
    }

    /// A value from an `f64`, or `None` if it is outside `[0, 1]` or not a number.
    ///
    /// Rounds to the nearest representable value. An `f64` finer than the `2^-63`
    /// grid loses its low bits, and one below `2^-64` becomes [`Unit::ZERO`]; see
    /// [`Unit::from_probability`] for the full account.
    pub fn new(value: f64) -> Option<Self> {
        if !(0.0..=1.0).contains(&value) {
            return None;
        }

        Some(Self::from_f64_unchecked(value))
    }

    /// A value from an `f64`, folded into range.
    ///
    /// For results of arithmetic that may drift a fraction outside. A NaN becomes
    /// zero, since there is nothing else it could reasonably be and this type has
    /// no NaN of its own.
    ///
    /// Rounds as [`Unit::new`] does, with the same loss for values finer than the
    /// grid.
    pub fn clamped(value: f64) -> Self {
        if value.is_nan() || value <= 0.0 {
            return Self::ZERO;
        }

        if value >= 1.0 {
            return Self::ONE;
        }

        Self::from_f64_unchecked(value)
    }

    /// The conversion itself, for a value already known to be in range.
    fn from_f64_unchecked(value: f64) -> Self {
        // `2^63` is exactly representable, so one at the top lands on the scale.
        let scaled: f64 = value * SCALE as f64;

        Self((scaled.round() as u64).min(SCALE))
    }

    /// One chance in `count`, or [`Unit::ZERO`] when `count` is zero.
    pub fn one_in(count: u64) -> Self {
        if count == 0 {
            return Self::ZERO;
        }

        Self(divide_rounded(SCALE as u128, count))
    }

    /// `hits` out of `total`, capped at one, or [`Unit::ZERO`] when `total` is zero.
    pub fn out_of(hits: u64, total: u64) -> Self {
        if total == 0 {
            return Self::ZERO;
        }

        if hits >= total {
            return Self::ONE;
        }

        Self(divide_rounded(hits as u128 * SCALE as u128, total))
    }

    /// A [`Ratio`] as a value here, capped at one.
    ///
    /// A `Ratio` is an unsigned fraction and may exceed one; anything at or above
    /// one becomes [`Unit::ONE`], which is the cap the name promises. Rounds to
    /// nearest, so the nearest representable value is chosen rather than the one
    /// below.
    pub fn from_ratio_capped(ratio: Ratio) -> Self {
        let denominator: u64 = ratio.denominator();

        if denominator == 0 {
            return Self::ZERO;
        }

        if ratio.numerator() >= denominator {
            return Self::ONE;
        }

        Self(divide_rounded(
            ratio.numerator() as u128 * SCALE as u128,
            denominator,
        ))
    }

    /// A uniform word read as a fraction: its top 63 bits over `2^63`.
    ///
    /// **Half-open**, in `[0, 1)` — a value built this way is never exactly one,
    /// which is what makes [`Unit::decide`] land on the asked-for probability
    /// exactly. See the module documentation.
    ///
    /// This is the *single* definition of how a random word becomes a `Unit`.
    /// [`Unit::from_seed`], [`StochasticSource::unit`](crate::random::StochasticSource::unit)
    /// and [`UniformUnit`](crate::random::unit::UniformUnit) all route through it, so every path produces the same
    /// value from the same word — which is what makes a replay reproducible no
    /// matter which one the caller reached for.
    pub const fn from_word(word: u64) -> Self {
        // One bit dropped, leaving 63 — exactly the width of the scale.
        Self(word >> 1)
    }


    // -----------------------------------------------------------------------
    // Reading
    // -----------------------------------------------------------------------

    /// The value as an `f64`, rounded to the nearest one.
    ///
    /// **Lossy near one.** There are 63 fractional bits here and 53 significant
    /// ones in an `f64`, so roughly a thousand `Unit` values share each `f64` at the
    /// top of the range: [`Unit::ALMOST_ONE`] and its neighbour both convert to
    /// exactly `1.0`. Printing the result cannot recover the difference, which is
    /// why [`fmt::Display`] formats from the integer instead.
    ///
    /// Deterministic, and never off by more than half an `f64` step — but not a
    /// round trip.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE as f64
    }

    /// Whether the value is zero.
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Whether the value is one.
    pub const fn is_one(self) -> bool {
        self.0 == SCALE
    }

    // -----------------------------------------------------------------------
    // Exact relationships: Ratio and Fixed
    // -----------------------------------------------------------------------

    /// The exact fraction this value stands for.
    ///
    /// # Why this is exact and the other direction is not
    ///
    /// A `Unit` *is* the fraction `k / 2^63` — that is the whole representation,
    /// not an approximation of anything. So the conversion is a pair of integers
    /// handed straight to [`Ratio`], with no rounding and no `f64` anywhere. The
    /// scale fits a `u64` (`2^63 < 2^64`) and the numerator never exceeds it, so
    /// the fraction is always proper.
    ///
    /// [`Ratio::new`] reduces, so the result comes back in lowest terms:
    ///
    /// ```
    /// use voxel_world::units::{Ratio, Unit};
    ///
    /// assert_eq!(Unit::ZERO.to_ratio(), Ratio::new(0, 1).unwrap());
    /// assert_eq!(Unit::HALF.to_ratio(), Ratio::new(1, 2).unwrap());
    /// assert_eq!(Unit::ONE.to_ratio(), Ratio::new(1, 1).unwrap());
    /// assert_eq!(Unit::STEP.to_ratio().denominator(), 1 << 63);
    /// ```
    ///
    /// Going the other way generally *cannot* be exact: a `Ratio` can be any
    /// fraction of two 64-bit numbers, and only those whose reduced denominator
    /// divides `2^63` land on the grid. See [`Unit::from_ratio_exact`].
    pub fn to_ratio(self) -> Ratio {
        Ratio::new(self.0, SCALE).expect("the scale is not zero")
    }

    /// A [`Ratio`] as a value here, or `None` if it is above one.
    ///
    /// # How this differs from [`Unit::from_ratio_capped`]
    ///
    /// The two do the same arithmetic and differ only in what they mean:
    ///
    /// - [`Unit::from_ratio_capped`] **asks for saturation**. A ratio of `7/2`
    ///   becoming one is the documented answer, and the name says so.
    /// - This one **asserts the input is already a chance**. A ratio of `7/2` is a
    ///   mistake upstream, and it is returned rather than quietly corrected.
    ///
    /// Separate methods because silently folding an out-of-range value hides the
    /// bug that produced it, and because a caller who genuinely wants a cap should
    /// have to say so.
    ///
    /// Within `[0, 1]` the conversion is the same as the capped form's: round to
    /// nearest, ties to even, in widened integer arithmetic. No `f64` is involved,
    /// so the result is the same on every target.
    pub fn try_from_ratio(ratio: Ratio) -> Option<Self> {
        if !ratio.is_proper() {
            return None;
        }

        Some(Self::from_ratio_capped(ratio))
    }

    /// A [`Ratio`] as a value here, or `None` unless it lands on the grid exactly.
    ///
    /// # Which fractions are exact
    ///
    /// The grid is `k / 2^63`, so a reduced fraction `n / d` is representable
    /// exactly when `d` divides `2^63` — that is, when `d` is a power of two. Those
    /// are the dyadic rationals, and they are the only ones: a denominator with any
    /// odd factor above one can never divide a power of two.
    ///
    /// ```text
    /// 1/2, 3/8, 17/64   ->  Some, exactly
    /// 1/3, 1/10         ->  None, no matter how many bits were available
    /// ```
    ///
    /// [`Ratio`] reduces on construction, so `3/12` arrives as `1/4` and is
    /// accepted — the test is on the reduced denominator, not the one written down.
    ///
    /// Every power of two a `u64` can hold is at most `2^63`, so once the
    /// denominator is a power of two it divides the scale and the numerator scales
    /// up without overflowing.
    pub fn from_ratio_exact(ratio: Ratio) -> Option<Self> {
        if !ratio.is_proper() {
            return None;
        }

        let denominator: u64 = ratio.denominator();

        // Dyadic, and nothing else, is representable. `is_power_of_two` also rules
        // out a zero denominator, which `Ratio` does not produce anyway.
        if !denominator.is_power_of_two() {
            return None;
        }

        // The denominator divides the scale, so this is exact. The numerator is at
        // most the denominator, so the product is at most the scale.
        Some(Self(ratio.numerator() * (SCALE / denominator)))
    }

    /// The same number as a [`Fixed`], rounded to its step.
    ///
    /// # Why this loses precision
    ///
    /// A `Fixed` is Q95.32 — 32 fractional bits — where this carries 63. So 31 bits
    /// fall off the bottom, and about two billion `Unit` values share each `Fixed`.
    /// The conversion is a shift of the raw integer with round-to-nearest,
    /// ties-to-even, matching the rounding everywhere else in the maths library.
    ///
    /// No `f64` on the way: both types are integers underneath, so this is a shift
    /// and a comparison, identical on every target.
    pub fn to_fixed(self) -> Fixed {
        Fixed::from_bits(shift_rounded(self.0, FIXED_SHIFT) as i128)
    }

    /// A [`Fixed`] as a value here, or `None` if it is outside `[0, 1]`.
    ///
    /// # Why this direction is exact
    ///
    /// The other way round: a `Fixed` has 32 fractional bits and this has 63, so
    /// every bit of a `Fixed` in range has somewhere to go. The conversion is a
    /// shift left by 31 with nothing discarded, and it round-trips through
    /// [`Unit::to_fixed`] unchanged.
    ///
    /// `None` rather than a clamp for a value outside `[0, 1]`, for the same reason
    /// [`Unit::try_from_ratio`] refuses: a `Fixed` of `-3` is a mistake somewhere,
    /// and folding it to zero would hide that. [`Unit::from_fixed_clamped`] is the
    /// form that asks for folding.
    pub fn try_from_fixed(value: Fixed) -> Option<Self> {
        let bits: i128 = value.to_bits();

        if bits < 0 || bits > Fixed::ONE.to_bits() {
            return None;
        }

        // At most `2^32` shifted up by 31 is `2^63`, which is exactly the scale.
        Some(Self((bits as u64) << FIXED_SHIFT))
    }

    /// A [`Fixed`] as a value here, folded into `[0, 1]`.
    ///
    /// For a value that may have drifted a step outside through arithmetic. Where
    /// out-of-range input would be a bug rather than drift, use
    /// [`Unit::try_from_fixed`].
    pub fn from_fixed_clamped(value: Fixed) -> Self {
        let bits: i128 = value.to_bits().clamp(0, Fixed::ONE.to_bits());

        Self((bits as u64) << FIXED_SHIFT)
    }

    // -----------------------------------------------------------------------
    // Written the way chances usually are
    // -----------------------------------------------------------------------

    /// A whole number of percent, or `None` above `100`.
    ///
    /// Exact where the fraction is dyadic — `percent(50)` is exactly a half — and
    /// correctly rounded otherwise, since a hundredth is not. Built through
    /// [`Ratio`], so the arithmetic is integer throughout.
    pub fn percent(percent: u64) -> Option<Self> {
        Self::try_from_ratio(Ratio::new(percent, 100)?)
    }

    /// A whole number of thousandths, or `None` above `1000`.
    pub fn permille(permille: u64) -> Option<Self> {
        Self::try_from_ratio(Ratio::new(permille, 1_000)?)
    }

    /// A whole number of hundredths of a percent, or `None` above `10_000`.
    ///
    /// The unit rates and small chances are usually quoted in: 250 basis points is
    /// `0.025`.
    pub fn basis_points(points: u64) -> Option<Self> {
        Self::try_from_ratio(Ratio::new(points, 10_000)?)
    }


    // -----------------------------------------------------------------------
    // Combining — closed on [0, 1]
    // -----------------------------------------------------------------------

    /// `1 - p`. Exact for every value.
    ///
    /// An integer subtraction, so nothing is lost however close to either end the
    /// value sits — which is what an `f64` cannot promise and the reason this type
    /// exists.
    pub const fn complement(self) -> Self {
        Self(SCALE - self.0)
    }

    /// `p × q`: both of two **independent** events.
    ///
    /// Rounds to nearest, ties to even. Closed, since a product of two values in
    /// `[0, 1]` is in `[0, 1]`.
    pub const fn and(self, other: Self) -> Self {
        Self(descale(self.0 as u128 * other.0 as u128))
    }

    /// `p + q − pq`: either of two **independent** events.
    ///
    /// Computed as `1 − (1−p)(1−q)`, which is the same number and never leaves the
    /// range on the way — writing it as `p + q − pq` would overflow past one before
    /// the subtraction brought it back.
    pub const fn or(self, other: Self) -> Self {
        self.complement().and(other.complement()).complement()
    }

    /// `1 − (1−p)^tries`: at least one success in `tries` **independent** attempts.
    pub fn in_any_of(self, tries: u32) -> Self {
        self.complement().pow(tries).complement()
    }

    /// `p^exponent`, by repeated squaring.
    pub fn pow(self, exponent: u32) -> Self {
        if exponent == 0 {
            return Self::ONE;
        }

        let mut result: Self = Self::ONE;
        let mut base: Self = self;
        let mut remaining: u32 = exponent;

        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result.and(base);
            }

            remaining >>= 1;

            if remaining == 0 {
                break;
            }

            base = base.and(base);
        }

        result
    }

    /// The value `weight` of the way from this to `other`.
    ///
    /// Closed, since a weighted average of two values in `[0, 1]` stays inside it.
    /// `(1−w)·self + w·other`.
    ///
    /// # Why the two halves are summed before rounding, not after
    ///
    /// Rounding each half and then adding them is wrong, and wrong in a way that
    /// shows on the most obvious input: blending a value with *itself*. At a weight
    /// of a half, both halves are `k/2`, and for odd `k` both round the same way —
    /// so an odd value blended with itself comes back one step larger than it went
    /// in. Summing first and rounding once makes `blend(x, x, w)` exactly `x` for
    /// every weight, which is the least a blend should promise.
    ///
    /// The widened sum cannot overflow: it is at most `SCALE²`, which is `2^126`.
    pub const fn blend(self, other: Self, weight: Unit) -> Self {
        let kept: u128 = weight.complement().0 as u128 * self.0 as u128;
        let taken: u128 = weight.0 as u128 * other.0 as u128;

        Self(descale(kept + taken))
    }

    /// The smaller of two values.
    pub const fn min(self, other: Self) -> Self {
        if self.0 < other.0 { self } else { other }
    }

    /// The larger of two values.
    pub const fn max(self, other: Self) -> Self {
        if self.0 > other.0 { self } else { other }
    }

    // -----------------------------------------------------------------------
    // Combining — not closed, so checked
    // -----------------------------------------------------------------------

    /// `p + q`: either of two **disjoint** events. `None` if the sum exceeds one.
    ///
    /// Refuses rather than saturating, because a sum above one means the events
    /// were not disjoint after all and that is worth hearing about. Use
    /// [`Unit::or`] for independent events and [`Unit::saturating_add`] where a cap
    /// is genuinely wanted.
    pub const fn checked_add(self, other: Self) -> Option<Self> {
        // Widened: two values at the scale would overflow a `u64` between them.
        let sum: u128 = self.0 as u128 + other.0 as u128;

        if sum > SCALE as u128 {
            return None;
        }

        Some(Self(sum as u64))
    }

    /// `p + q`, capped at one.
    pub const fn saturating_add(self, other: Self) -> Self {
        match self.checked_add(other) {
            Some(sum) => sum,
            None => Self::ONE,
        }
    }

    /// `p − q`. `None` if it would fall below zero.
    pub const fn checked_sub(self, other: Self) -> Option<Self> {
        if other.0 > self.0 {
            return None;
        }

        Some(Self(self.0 - other.0))
    }

    /// `p − q`, floored at zero.
    pub const fn saturating_sub(self, other: Self) -> Self {
        match self.checked_sub(other) {
            Some(difference) => difference,
            None => Self::ZERO,
        }
    }

    /// `p / q`, the conditional probability of this given `other`.
    ///
    /// `None` for a zero divisor, and when the quotient exceeds one — which means
    /// this event was not contained in the condition, so the ratio is not a
    /// probability.
    pub fn checked_div(self, other: Self) -> Option<Self> {
        if other.0 == 0 || self.0 > other.0 {
            return None;
        }

        Some(Self(divide_rounded(
            self.0 as u128 * SCALE as u128,
            other.0,
        )))
    }

    // -----------------------------------------------------------------------
    // Scaling to integers
    // -----------------------------------------------------------------------

    /// `p × count`, rounded to nearest with ties to even.
    ///
    /// The result is never above `count`, and reaches it only at [`Unit::ONE`].
    pub const fn scale_rounded(self, count: u64) -> u64 {
        descale(self.0 as u128 * count as u128)
    }

    /// `p × count`, rounded down.
    pub const fn scale_floor(self, count: u64) -> u64 {
        ((self.0 as u128 * count as u128) >> SHIFT) as u64
    }

    /// `p × count`, rounded up.
    pub const fn scale_ceil(self, count: u64) -> u64 {
        let product: u128 = self.0 as u128 * count as u128;
        let floor: u128 = product >> SHIFT;

        (floor + (product & ((1 << SHIFT) - 1) != 0) as u128) as u64
    }

    /// Which of `count` equal buckets the value falls in, from `0` to `count - 1`.
    ///
    /// Zero when `count` is zero. **Always in range**, including at
    /// [`Unit::ONE`]: a plain floor would give `count` there, which is the classic
    /// way a uniform draw indexes one past the end of a table. The clamp is what
    /// makes a closed unit interval safe to index with.
    pub const fn index_of(self, count: u64) -> u64 {
        if count == 0 {
            return 0;
        }

        let floor: u64 = self.scale_floor(count);

        if floor >= count { count - 1 } else { floor }
    }

    /// The value spread across `[low, high]`.
    ///
    /// Closed at both ends, like the value itself: at [`Unit::ONE`] the result is
    /// `high`. Equal bounds give that bound.
    pub fn between(self, low: f64, high: f64) -> f64 {
        assert!(
            low.is_finite() && high.is_finite() && low <= high && (high - low).is_finite(),
            "bounds must be ordered, finite, and have a finite span"
        );

        if low == high {
            return low;
        }

        (low + (high - low) * self.to_f64()).clamp(low, high)
    }

    // -----------------------------------------------------------------------
    // Deciding
    // -----------------------------------------------------------------------

}

/// `product / 2^63`, rounded to nearest with ties to even.
///
/// Ties to even rather than always up because always-up biases every rounded
/// product upwards, and a chain of them drifts. The same choice the float types
/// here make.
const fn descale(product: u128) -> u64 {
    let quotient: u128 = product >> SHIFT;
    let remainder: u128 = product & ((1 << SHIFT) - 1);
    let half: u128 = 1 << (SHIFT - 1);

    // A tie goes to the even neighbour; anything past halfway goes up.
    if remainder > half || (remainder == half && quotient & 1 == 1) {
        (quotient + 1) as u64
    } else {
        quotient as u64
    }
}

/// `value / 2^places`, rounded to nearest with ties to even.
///
/// The same rounding as [`descale`], for the cases where the divisor is a smaller
/// power of two than the scale.
const fn shift_rounded(value: u64, places: u32) -> u64 {
    let quotient: u64 = value >> places;
    let remainder: u64 = value & ((1 << places) - 1);
    let half: u64 = 1 << (places - 1);

    if remainder > half || (remainder == half && quotient & 1 == 1) {
        quotient + 1
    } else {
        quotient
    }
}

/// `numerator / denominator`, rounded to nearest with ties to even.
fn divide_rounded(numerator: u128, denominator: u64) -> u64 {
    let divisor: u128 = denominator as u128;
    let quotient: u128 = numerator / divisor;
    let remainder: u128 = numerator % divisor;
    let doubled: u128 = remainder * 2;

    if doubled > divisor || (doubled == divisor && quotient & 1 == 1) {
        (quotient + 1) as u64
    } else {
        quotient as u64
    }
}

impl Unit {
    /// The exact decimal digits after the point, with no `0.` in front and no
    /// trailing zeros.
    ///
    /// # Why this does not go through `f64`
    ///
    /// Because it cannot. [`Unit::to_f64`] loses the distinction between
    /// neighbouring values near one — an `f64` has 53 significant bits where this
    /// has 63 fractional ones, so about a thousand consecutive `Unit`s share each
    /// `f64` up there, and `(2^63 - 1) / 2^63` converts to exactly `1.0`. Once the
    /// value has been through an `f64`, no number of printed digits can recover what
    /// the conversion threw away.
    ///
    /// # The arithmetic
    ///
    /// The expansion always terminates, because
    ///
    /// ```text
    /// k / 2^63  =  k × 5^63 / 10^63
    /// ```
    ///
    /// so there are at most 63 decimal places. Forming `k × 5^63` directly is not
    /// an option: `5^63` alone needs 147 bits and the product needs 210, both well
    /// past a `u128`. Instead the digits come out one at a time — multiply the
    /// remaining fraction by ten, take whatever carries past the scale as the next
    /// digit, and keep the rest.
    ///
    /// The widest intermediate is `(2^63 - 1) × 10`, about `9.2 × 10^19`. That is 67
    /// bits, so a `u128` has room to spare.
    fn fraction_digits(self) -> String {
        // One has no fractional part; every other value is its own remainder.
        let mut remainder: u128 = u128::from(self.0 % SCALE);
        let mut digits = String::new();

        // Terminates because the denominator is a power of two: each step consumes
        // one factor of five, and there are 63 of them.
        while remainder != 0 {
            remainder *= 10;

            let digit: u32 = (remainder >> SHIFT) as u32;
            digits.push(char::from_digit(digit, 10).expect("a value below ten"));

            remainder &= (1 << SHIFT) - 1;
        }

        digits
    }
}

/// The exact digits rounded to `places`, with any carry passed back through
/// `whole`.
///
/// Round to nearest, ties to even, matching the type's arithmetic. Rounding up can
/// cascade through a run of nines and out of the fractional part entirely, which is
/// how `0.9995` at three places becomes `1.000`.
fn round_digits(whole: &mut u64, digits: &str, places: usize) -> String {
    if digits.len() <= places {
        // Already shorter than asked for, so pad rather than round.
        return format!("{digits}{}", "0".repeat(places - digits.len()));
    }

    let (kept, dropped) = digits.split_at(places);
    let mut kept: Vec<u8> = kept.as_bytes().to_vec();

    let first: u8 = dropped.as_bytes()[0];
    let any_below: bool = dropped.as_bytes()[1..].iter().any(|digit| *digit != b'0');
    // With no digits kept, the parity that decides a tie is the whole part's.
    let last_is_odd: bool = match kept.last() {
        Some(digit) => (digit - b'0') % 2 == 1,
        None => *whole % 2 == 1,
    };

    if first > b'5' || (first == b'5' && (any_below || last_is_odd)) {
        // Carry backwards through the kept digits, and into the whole part if the
        // whole fraction was nines.
        let mut index: usize = kept.len();

        loop {
            if index == 0 {
                *whole += 1;
                break;
            }

            index -= 1;

            if kept[index] == b'9' {
                kept[index] = b'0';
            } else {
                kept[index] += 1;
                break;
            }
        }
    }

    String::from_utf8(kept).expect("the digits were ASCII going in")
}

impl fmt::Display for Unit {
    /// The **exact** decimal value.
    ///
    /// Every value has a terminating expansion, since the denominator is a power of
    /// two, so this is the whole number and not an approximation of it. Distinct
    /// values therefore always print distinctly — which printing through an `f64`
    /// could not promise at any width, because the conversion itself is lossy near
    /// one.
    ///
    /// Trailing zeros are trimmed, so a half is `0.5` rather than a half followed by
    /// sixty-two zeros. Values that are not dyadic run to the full 63 places, which
    /// is what exactness costs; `{:.4}` and friends give a short form, rounded to
    /// nearest with ties to even.
    ///
    /// ```
    /// use voxel_world::units::Unit;
    ///
    /// assert_eq!(Unit::ZERO.to_string(), "0");
    /// assert_eq!(Unit::ONE.to_string(), "1");
    /// assert_eq!(Unit::HALF.to_string(), "0.5");
    /// assert_eq!(format!("{:.4}", Unit::one_in(3)), "0.3333");
    ///
    /// // Neighbours print differently, which is the point.
    /// let almost = Unit::ALMOST_ONE;
    /// let just_below = Unit::from_bits(almost.to_bits() - 1).unwrap();
    /// assert_ne!(almost.to_string(), just_below.to_string());
    ///
    /// // Through an f64 they would both have been exactly 1.0.
    /// assert_eq!(almost.to_f64(), 1.0);
    /// assert_eq!(just_below.to_f64(), 1.0);
    /// ```
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut whole: u64 = self.0 / SCALE;
        let digits: String = self.fraction_digits();

        let text: String = match formatter.precision() {
            Some(places) => {
                let fraction: String = round_digits(&mut whole, &digits, places);

                if places == 0 {
                    format!("{whole}")
                } else {
                    format!("{whole}.{fraction}")
                }
            }
            None if digits.is_empty() => format!("{whole}"),
            None => format!("{whole}.{digits}"),
        };

        formatter.write_str(&text)
    }
}

impl fmt::Debug for Unit {
    /// The same exact decimal as [`fmt::Display`], so that a failing comparison
    /// shows values that actually differ. [`Unit::to_bits`] is there when the raw
    /// count is what is wanted.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self}")
    }
}


/// Exact, so a plain [`From`] is honest here: every `Unit` *is* a fraction.
impl From<Unit> for Ratio {
    fn from(value: Unit) -> Self {
        value.to_ratio()
    }
}

/// Fallible, because a [`Ratio`] can be any fraction and only those in `[0, 1]` are
/// values here.
///
/// # Why there is no `From<Ratio>`
///
/// There was, and it delegated to [`Unit::from_ratio_capped`] — so `7/2` silently
/// became one. That is a reasonable thing for a method whose name says "capped" to
/// do, and a bad thing for the conversion a caller reaches for by default: `From`
/// reads as "the same number in another type", and saturating is not that.
///
/// So the default conversion refuses out-of-range input, and saturation stays
/// available under the name that asks for it.
impl TryFrom<Ratio> for Unit {
    type Error = RatioOutOfRange;

    fn try_from(ratio: Ratio) -> Result<Self, RatioOutOfRange> {
        Self::try_from_ratio(ratio).ok_or(RatioOutOfRange)
    }
}

/// A [`Ratio`] above one, which is not a value in `[0, 1]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RatioOutOfRange;

impl fmt::Display for RatioOutOfRange {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a ratio above one is not a value in [0, 1]")
    }
}

impl std::error::Error for RatioOutOfRange {}

/// Reads decimal notation through [`Unit::from_decimal_str`], which never involves a
/// float.
///
/// The error is [`DecimalError`] rather than a type of its own, because every way the
/// text can fail is already one of its cases — including
/// [`DecimalError::NotAUnit`] for a value
/// outside `[0, 1]`.
impl std::str::FromStr for Unit {
    type Err = DecimalError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::from_decimal_str(text)
    }
}
