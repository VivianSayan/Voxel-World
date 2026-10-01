//! A fixed-point number: an `i128` counted in units of `2^-32`.
//!
//! Every operation here is integer arithmetic, so the same inputs give the same
//! bits on every platform, in every build profile, for ever. That is the whole
//! point of the type. An `f64` gives the same guarantee for `+ - * /` and
//! `sqrt`, but not for `ln`, `exp`, `powf` or the trigonometric functions,
//! which come from whichever C math library the platform links; see
//! [`crate::random::distributions`] for which samplers that affects.
//!
//! # What it is
//!
//! A [`Fixed`] holds a plain `i128` and reads it as a count of `2^-32`, so the
//! layout is Q95.32: one sign bit, 95 bits of whole units, 32 bits of fraction.
//!
//! | | |
//! |---|---|
//! | Step between neighbours | `2^-32`, about `2.3e-10` |
//! | Range | about `±3.96e28` whole units |
//! | Size | 16 bytes, `Copy` |
//!
//! Unlike an `f64`, the step is the same everywhere in that range: a value near
//! the top is held as precisely as one near zero. That is what makes fixed
//! point suited to state that accumulates, such as a position advanced by a
//! velocity thousands of times, where an `f64`'s precision falls away as the
//! value grows and each addition rounds again. It is not suited to values that
//! span many orders of magnitude, which is what an `f64`'s exponent is for.
//!
//! # How it behaves
//!
//! Addition and subtraction are exact. Multiplication and division are computed
//! through a 256-bit intermediate, so no product is lost on the way, and their
//! results are truncated towards zero, as integer `*` and `/` are.
//!
//! What that costs, against `f64` on the same machine: a multiply is about
//! seven times slower, a division twelve, and a square root thirty. A division
//! or a root whose intermediate fits in 128 bits, which is every value below
//! `2^64`, uses the standard library's own operation rather than the wide path,
//! and that is what keeps those numbers small. The logarithms and exponentials
//! are the expensive ones, at roughly a hundred times an `f64` call, since each
//! output bit of a logarithm costs a wide multiply.
//!
//! Every operator **panics on overflow in release builds as well as debug
//! ones**, which is deliberate: a value that wraps silently in a shipped build
//! and not in a test is the opposite of reproducible. The `checked_`,
//! `saturating_` and `wrapping_` methods are there for callers that want
//! another answer.
//!
//! # Logarithms and exponentials
//!
//! [`Fixed::log2`], [`Fixed::ln`], [`Fixed::exp`] and the rest are here, and
//! they are integer algorithms like everything else: a logarithm is found one
//! output bit at a time by squaring, and an exponential by splitting the
//! argument on `ln 2` and summing a short series. Both work internally with 96
//! fractional bits, so the guard bits absorb the rounding and the result lands
//! within a step of the answer: the logarithms are good to one step, the
//! exponentials are correctly rounded, and the two powers to about one part in
//! `10^10`.
//!
//! They are not a replacement for an `f64` libm, and specifically not for the
//! samplers in [`crate::random::distributions`]. Ten significant digits is
//! plenty for decay, falloff and easing; it is not enough for a distribution's
//! tail, where `exp(-30)` is a real probability and this type rounds it to
//! zero. The ranges are on each method.

use std::fmt;
use std::iter::{Product, Sum};
use std::num::ParseIntError;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Shl, ShlAssign, Shr,
    ShrAssign, Sub, SubAssign,
};
use std::str::FromStr;

use crate::math::cordic;

/// How many bits of a [`Fixed`] lie after the point.
///
/// The default layout's width. A [`FixedPoint<F>`] of another width has its own,
/// reachable as `FixedPoint::<F>::FRACTION_BITS`.
pub const FRACTION_BITS: u32 = 32;

/// The raw value of one whole unit in the default layout: `2^32`.
const SCALE: i128 = 1 << FRACTION_BITS;

/// How many fractional bits the named constants are written down with.
///
/// Wide enough that narrowing to any supported layout is a single correctly-rounded
/// shift, and narrow enough that the products still fit an `i128`.
const REFERENCE_BITS: u32 = 96;

/// Pi at [`REFERENCE_BITS`] fractional bits.
const REFERENCE_PI: i128 = 248_902_613_312_231_085_230_521_944_622;
/// Two pi.
const REFERENCE_TAU: i128 = 497_805_226_624_462_170_461_043_889_244;
/// Euler's number.
const REFERENCE_E: i128 = 215_364_474_464_724_850_177_511_348_353;
/// The square root of two.
const REFERENCE_SQRT_2: i128 = 112_045_541_949_572_279_837_463_876_455;
/// The natural logarithm of two. The same number as [`WORKING_LN_2`], which the
/// logarithm uses internally at this very width.
const REFERENCE_LN_2: i128 = 54_916_777_467_707_473_351_141_471_128;

/// π/2 at [`REFERENCE_BITS`] fractional bits. Exactly half [`REFERENCE_PI`], which is
/// even, so no rounding enters here.
const REFERENCE_HALF_PI: i128 = 124_451_306_656_115_542_615_260_972_311;

/// π/4 at [`REFERENCE_BITS`] fractional bits.
const REFERENCE_QUARTER_PI: i128 = 62_225_653_328_057_771_307_630_486_156;

/// The natural logarithm of ten at [`REFERENCE_BITS`] fractional bits.
const REFERENCE_LN_10: i128 = 182_429_585_950_654_714_090_129_938_606;

/// `log2(e)` at [`REFERENCE_BITS`] fractional bits.
const REFERENCE_LOG2_E: i128 = 114_302_077_158_074_026_402_637_675_937;

/// `log10(e)` at [`REFERENCE_BITS`] fractional bits.
const REFERENCE_LOG10_E: i128 = 34_408_353_791_279_068_184_446_883_160;

/// The widest layout the transcendental functions keep their accuracy at.
///
/// [`FixedPoint::ln`] and its relatives work in 96 fractional bits internally and
/// round down to the type's own. Past this the guard between them is too thin for
/// the documented "good to one step", so the build stops rather than quietly
/// returning worse answers than the documentation promises.
pub const MAX_FRACTION_BITS: u32 = 64;

/// A number held as an `i128` count of `2^-32`, in Q95.32 layout.
///
/// Ordering, equality and hashing are the `i128`'s own, which agree with the
/// value's, so a `Fixed` can key a sorted map or a hash map without the traps
/// an `f64` brings: there is no NaN, no negative zero and no value that fails
/// to equal itself.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FixedPoint<const FRACTION_BITS: u32>(i128);

/// The widest layout is checked at compile time, so an impossible one fails the
/// build rather than returning quietly worse answers than the documentation
/// promises:
///
/// ```compile_fail
/// use voxel_world::math::FixedPoint;
///
/// // 100 fractional bits leaves too thin a guard for the transcendentals.
/// let _ = FixedPoint::<100>::from_bits(0);
/// ```
///
/// ```
/// use voxel_world::math::FixedPoint;
///
/// // Anything up to 64 is fine.
/// let _ = FixedPoint::<64>::from_bits(0);
/// let _ = FixedPoint::<1>::from_bits(0);
/// ```
///
/// Fixed point with 32 fractional bits — the Q95.32 layout the engine uses.
///
/// A plain alias rather than a default on the struct, and that is not a style
/// choice: a `const` parameter's default applies only in *type* position, so
/// `struct FixedPoint<const F: u32 = 32>` would leave `Fixed::ONE` and
/// `Fixed::from_integer(5)` failing with "type annotations needed". Through an alias
/// they keep working untouched.
pub type Fixed = FixedPoint<32>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// How many fractional bits the logarithm and exponential work in internally.
///
/// Wide enough that the rounding in each squaring or series term stays far below
/// the step the answer is reported at — for every supported layout, not just the
/// default. The margin is `WORKING_BITS - FRACTION_BITS`, which is why
/// [`MAX_FRACTION_BITS`] stops well short of this.
const WORKING_BITS: u32 = 96;

/// One, in the internal working format.
const WORKING_ONE: u128 = 1 << WORKING_BITS;

/// The natural logarithm of two, in the internal working format.
///
/// The same number as [`REFERENCE_LN_2`], since the reference width and the working
/// width are both 96.
const WORKING_LN_2: u128 = 54_916_777_467_707_473_351_141_471_128;

/// The base-ten logarithm of two, in the internal working format.
const WORKING_LOG10_2: u128 = 23_850_053_418_134_191_015_272_426_710;

/// How many spare bits a logarithm keeps below the answer's own step.
///
/// The original code wrote a flat `LOG_BITS = 48`, which is this plus the default
/// layout's 32. Expressing it as a margin *above* the width is what lets the width
/// move: a flat 48 underflows the moment a layout carries more than 48 fractional
/// bits, and — more quietly — stays at the default's value for every narrower one,
/// which is a wrong answer rather than a failed build. Each layout derives its own
/// `Self::LOG_BITS` from this.
const LOG_GUARD_BITS: u32 = 16;

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// How many bits of this layout lie after the point.
    pub const FRACTION_BITS: u32 = FRACTION_BITS;

    /// The raw value of one whole unit: `2^FRACTION_BITS`.
    pub const SCALE: i128 = 1 << FRACTION_BITS;

    /// How many fractional bits a logarithm keeps before it is rounded to this
    /// layout: its own width plus a guard.
    const LOG_BITS: u32 = FRACTION_BITS + LOG_GUARD_BITS;

    /// Refuses a width the transcendental functions could not hold their accuracy
    /// at, and a width of zero, which has no fraction to be fixed.
    const VALID: () = assert!(
        FRACTION_BITS > 0 && FRACTION_BITS <= MAX_FRACTION_BITS,
        "a fixed-point layout needs between 1 and MAX_FRACTION_BITS fractional bits"
    );

    /// Zero.
    pub const ZERO: Self = Self(0);
    /// One whole unit.
    pub const ONE: Self = Self(Self::SCALE);
    /// Minus one whole unit.
    pub const NEGATIVE_ONE: Self = Self(-Self::SCALE);
    /// One half, which is exact here.
    pub const HALF: Self = Self(Self::SCALE / 2);
    /// The smallest step between two values, `2^-FRACTION_BITS`.
    pub const DELTA: Self = Self(1);
    /// The most negative value.
    pub const MIN: Self = Self(i128::MIN);
    /// The largest value.
    pub const MAX: Self = Self(i128::MAX);

    /// Pi, to the nearest step of *this* layout.
    pub const PI: Self = Self::from_reference(REFERENCE_PI);
    /// Two pi, to the nearest step.
    pub const TAU: Self = Self::from_reference(REFERENCE_TAU);
    /// Euler's number, to the nearest step.
    pub const E: Self = Self::from_reference(REFERENCE_E);
    /// The square root of two, to the nearest step.
    pub const SQRT_2: Self = Self::from_reference(REFERENCE_SQRT_2);
    /// The natural logarithm of two, to the nearest step.
    pub const LN_2: Self = Self::from_reference(REFERENCE_LN_2);

    /// π/2, a quarter turn.
    pub const FRAC_PI_2: Self = Self::from_reference(REFERENCE_HALF_PI);

    /// π/4, an eighth turn.
    pub const FRAC_PI_4: Self = Self::from_reference(REFERENCE_QUARTER_PI);

    /// The natural logarithm of ten, for moving between `ln` and `log10`.
    pub const LN_10: Self = Self::from_reference(REFERENCE_LN_10);

    /// `log2(e)`, for moving between `ln` and `log2`.
    pub const LOG2_E: Self = Self::from_reference(REFERENCE_LOG2_E);

    /// `log10(e)`, for moving between `ln` and `log10`.
    pub const LOG10_E: Self = Self::from_reference(REFERENCE_LOG10_E);

    /// A named constant narrowed from the shared 96-bit reference to this width.
    ///
    /// Written once and rounded per layout, rather than a table of literals per
    /// width: sixty-four hand-written literals would be sixty-four chances to
    /// mistype a digit, and nothing would notice.
    ///
    /// Rounds to nearest. Every reference here is positive, so adding half a step
    /// before shifting is the whole of it.
    const fn from_reference(reference: i128) -> Self {
        let shift: u32 = REFERENCE_BITS - FRACTION_BITS;

        Self((reference + (1 << (shift - 1))) >> shift)
    }
}

// ---------------------------------------------------------------------------
// Building and converting
// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// The value whose raw count of `2^-32` is `bits`.
    ///
    /// The inverse of [`Fixed::to_bits`], and the way to read one back from
    /// storage: the raw count is the whole representation, so it round-trips
    /// exactly.
    pub const fn from_bits(bits: i128) -> Self {
        // Forces the width check: an associated constant is evaluated only where it
        // is named, so an unreferenced assertion would never fire.
        let () = Self::VALID;

        Self(bits)
    }

    /// The raw count of `2^-32` underneath, for storing or hashing.
    pub const fn to_bits(self) -> i128 {
        self.0
    }

    /// A whole number of units. Every `i64` fits, so this cannot fail.
    pub const fn from_integer(whole: i64) -> Self {
        Self(whole as i128 * Self::SCALE)
    }

    /// A whole number of units, or `None` if it is too large to scale.
    pub const fn from_integer_i128(whole: i128) -> Option<Self> {
        match whole.checked_mul(Self::SCALE) {
            Some(bits) => Some(Self(bits)),
            None => None,
        }
    }

    /// The exact fraction `numerator / denominator`, truncated towards zero to
    /// the nearest step.
    ///
    /// `None` for a zero denominator, and for a quotient too large to hold.
    /// Computed through a 256-bit intermediate, so `from_ratio(1, 3)` is the
    /// closest representable third rather than a rounded division.
    pub fn from_ratio(numerator: i128, denominator: i128) -> Option<Self> {
        let sign: bool = (numerator < 0) != (denominator < 0);
        let scaled: wide::U256 = wide::shift_left(numerator.unsigned_abs(), FRACTION_BITS);
        let magnitude: u128 = wide::divide(scaled, denominator.unsigned_abs())?;

        Some(Self(wide::apply_sign(magnitude, sign)?))
    }

    /// The nearest value at or below `value`, or `None` if it is not finite or
    /// lies outside the range.
    ///
    /// Scaling by `2^32` only changes an `f64`'s exponent, so nothing is lost
    /// before the truncation, and the cast that follows truncates towards zero
    /// by a rule Rust pins down. The conversion is therefore as reproducible as
    /// the rest of the type.
    pub fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }

        let scaled: f64 = value * Self::SCALE as f64;

        if scaled < i128::MIN as f64 || scaled > i128::MAX as f64 {
            return None;
        }

        Some(Self(scaled as i128))
    }

    /// The nearest `f64`, for rendering, printing or handing to a sampler.
    ///
    /// Exact while the value needs no more than 53 significant bits, which
    /// covers everything within about `9.0e6` whole units; beyond that the
    /// conversion rounds, and this becomes the boundary where determinism stops
    /// being free.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    /// The nearest `f32`, for a renderer or a vertex buffer.
    ///
    /// Rounds far sooner than [`Fixed::to_f64`] does, at about 8 whole units,
    /// since an `f32` carries only 24 significant bits.
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// The nearest value at or below an `f32`, or `None` if it is not finite.
    ///
    /// Widened to an `f64` first, which is exact, so this agrees with
    /// [`Fixed::from_f64`] for every value both can hold.
    pub fn from_f32(value: f32) -> Option<Self> {
        Self::from_f64(value as f64)
    }

    /// The whole part, truncated towards zero: `-2.7` gives `-2`.
    ///
    /// Towards zero rather than downwards, as integer division is. Where the
    /// voxel *containing* a position is wanted, [`Fixed::floor`] is the one to
    /// reach for: it rounds the same way the octree does.
    pub const fn to_integer(self) -> i128 {
        self.0 / Self::SCALE
    }
}

// ---------------------------------------------------------------------------
// Inspecting
// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// Whether this is exactly zero.
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Whether this is above zero.
    pub const fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Whether this is below zero.
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Whether this is a whole number of units, with nothing after the point.
    pub const fn is_whole(self) -> bool {
        self.0 % Self::SCALE == 0
    }

    /// One, zero or minus one, according to the sign.
    pub const fn signum(self) -> Self {
        Self(self.0.signum() * Self::SCALE)
    }

    /// The value without its sign. Panics on [`Fixed::MIN`], which has no
    /// positive counterpart.
    pub fn abs(self) -> Self {
        Self(self.0.checked_abs().expect("fixed-point overflow in abs"))
    }

    /// The smaller of the two.
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// The larger of the two.
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// Confined to `low..=high`. Panics if the bounds cross.
    pub fn clamp(self, low: Self, high: Self) -> Self {
        assert!(
            low <= high,
            "fixed-point clamp bounds are the wrong way round"
        );
        Self(self.0.clamp(low.0, high.0))
    }
}

// ---------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// The largest whole value at or below this one, rounding towards negative
    /// infinity. An arithmetic shift does it, since the fraction is the low
    /// bits.
    pub const fn floor(self) -> Self {
        Self((self.0 >> FRACTION_BITS) * Self::SCALE)
    }

    /// The smallest whole value at or above this one.
    pub fn ceil(self) -> Self {
        self.checked_ceil().expect("fixed-point overflow in ceil")
    }

    /// The whole part, rounding towards zero, which drops the fraction.
    pub const fn trunc(self) -> Self {
        Self(self.0 - self.0 % Self::SCALE)
    }

    /// What [`Fixed::trunc`] leaves behind, carrying the sign of the value:
    /// `-2.25` gives `-0.25`.
    pub const fn fract(self) -> Self {
        Self(self.0 % Self::SCALE)
    }

    /// The nearest whole value, with a half rounded away from zero.
    pub fn round(self) -> Self {
        let half: i128 = if self.0 < 0 { -(Self::SCALE / 2) } else { Self::SCALE / 2 };

        Self(
            self.0
                .checked_add(half)
                .expect("fixed-point overflow in round"),
        )
        .trunc()
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// The sum, or `None` on overflow.
    pub const fn checked_add(self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(bits) => Some(Self(bits)),
            None => None,
        }
    }

    /// The difference, or `None` on overflow.
    pub const fn checked_sub(self, other: Self) -> Option<Self> {
        match self.0.checked_sub(other.0) {
            Some(bits) => Some(Self(bits)),
            None => None,
        }
    }

    /// The negation, or `None` for [`Fixed::MIN`].
    pub const fn checked_neg(self) -> Option<Self> {
        match self.0.checked_neg() {
            Some(bits) => Some(Self(bits)),
            None => None,
        }
    }

    /// The product, truncated towards zero, or `None` on overflow.
    ///
    /// The two raw counts are multiplied into 256 bits and the result shifted
    /// back down by the fraction width, so nothing is lost between the
    /// multiplication and the rescaling.
    pub fn checked_mul(self, other: Self) -> Option<Self> {
        let sign: bool = (self.0 < 0) != (other.0 < 0);
        let product: wide::U256 = wide::multiply(self.0.unsigned_abs(), other.0.unsigned_abs());
        let magnitude: u128 = wide::shift_right(product, FRACTION_BITS)?;

        wide::apply_sign(magnitude, sign).map(Self)
    }

    /// The quotient, truncated towards zero, or `None` for a zero divisor or on
    /// overflow.
    ///
    /// The numerator is widened to 256 bits before being scaled up, so a
    /// division never loses the range that scaling would otherwise cost.
    pub fn checked_div(self, other: Self) -> Option<Self> {
        let sign: bool = (self.0 < 0) != (other.0 < 0);
        let scaled: wide::U256 = wide::shift_left(self.0.unsigned_abs(), FRACTION_BITS);
        let magnitude: u128 = wide::divide(scaled, other.0.unsigned_abs())?;

        wide::apply_sign(magnitude, sign).map(Self)
    }

    /// The remainder of `self / other`, carrying the sign of `self`, or `None`
    /// for a zero divisor.
    ///
    /// Both values count the same unit, so the remainder is simply the
    /// remainder of the raw counts.
    pub const fn checked_rem(self, other: Self) -> Option<Self> {
        match self.0.checked_rem(other.0) {
            Some(bits) => Some(Self(bits)),
            None => None,
        }
    }

    /// The smallest whole value at or above this one, or `None` on overflow.
    pub const fn checked_ceil(self) -> Option<Self> {
        match self.0.checked_add(Self::SCALE - 1) {
            Some(bits) => Some(Self((bits >> FRACTION_BITS) * Self::SCALE)),
            None => None,
        }
    }

    /// `self * factor + addend`, with a single truncation at the end rather
    /// than one per step. `None` on overflow.
    pub fn checked_mul_add(self, factor: Self, addend: Self) -> Option<Self> {
        self.checked_mul(factor)?.checked_add(addend)
    }

    /// One divided by this value, or `None` for zero or on overflow.
    pub fn checked_recip(self) -> Option<Self> {
        Self::ONE.checked_div(self)
    }

    /// The sum, held at [`Fixed::MIN`] or [`Fixed::MAX`] on overflow.
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// The difference, held at the ends on overflow.
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// The product, held at the ends on overflow.
    pub fn saturating_mul(self, other: Self) -> Self {
        self.checked_mul(other).unwrap_or(self.overflow_end(other))
    }

    /// The quotient, held at the ends on overflow. Panics on a zero divisor,
    /// which has no end to hold at.
    pub fn saturating_div(self, other: Self) -> Self {
        assert!(!other.is_zero(), "fixed-point division by zero");

        self.checked_div(other).unwrap_or(self.overflow_end(other))
    }

    /// The sum, wrapping around the ends of the range on overflow.
    pub const fn wrapping_add(self, other: Self) -> Self {
        Self(self.0.wrapping_add(other.0))
    }

    /// The difference, wrapping on overflow.
    pub const fn wrapping_sub(self, other: Self) -> Self {
        Self(self.0.wrapping_sub(other.0))
    }

    /// The product, keeping the low 128 bits of the rescaled result on
    /// overflow.
    pub fn wrapping_mul(self, other: Self) -> Self {
        let sign: bool = (self.0 < 0) != (other.0 < 0);
        let product: wide::U256 = wide::multiply(self.0.unsigned_abs(), other.0.unsigned_abs());
        let magnitude: u128 = wide::shift_right_wrapping(product, FRACTION_BITS);
        let bits: i128 = magnitude as i128;

        Self(if sign { bits.wrapping_neg() } else { bits })
    }

    /// Which end of the range an overflowing operation should be held at.
    const fn overflow_end(self, other: Self) -> Self {
        if (self.0 < 0) != (other.0 < 0) {
            Self::MIN
        } else {
            Self::MAX
        }
    }
}

// ---------------------------------------------------------------------------
// Roots and interpolation
// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// The square root, truncated towards zero, or `None` for a negative value.
    ///
    /// Computed digit by digit in integers on the value scaled up by `2^32`,
    /// held in 256 bits, so the result is the largest [`Fixed`] whose square
    /// does not exceed this one. Exact in the sense that matters: no
    /// approximation of a real-valued function is involved, and no platform can
    /// disagree about it.
    pub fn sqrt(self) -> Option<Self> {
        if self.0 < 0 {
            return None;
        }

        let scaled: wide::U256 = wide::shift_left(self.0 as u128, FRACTION_BITS);
        let root: u128 = wide::square_root(scaled);

        (root <= i128::MAX as u128).then_some(Self(root as i128))
    }

    /// The point `t` of the way from this value to `other`, where `t` of zero
    /// gives this one and one gives `other`.
    ///
    /// Written as `self + (other - self) * t`, so both ends come out exactly.
    /// `t` outside `0..=1` extrapolates past them. `None` on overflow.
    pub fn lerp(self, other: Self, t: Self) -> Option<Self> {
        other.checked_sub(self)?.checked_mul(t)?.checked_add(self)
    }

    /// The value halfway between the two, without the overflow that adding them
    /// first would risk.
    pub const fn midpoint(self, other: Self) -> Self {
        Self(self.0 / 2 + other.0 / 2 + (self.0 % 2 + other.0 % 2) / 2)
    }
}

// ---------------------------------------------------------------------------
// Logarithms and exponentials
// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// The base-two logarithm, or `None` for a value at or below zero.
    ///
    /// The whole part comes free from the position of the highest set bit. The
    /// fraction is then found one bit at a time: normalise what is left into
    /// `[1, 2)` and square it, and if the square reaches two the bit is set and
    /// the square is halved. Each output bit costs one squaring.
    ///
    /// Accurate to half a step: the squarings are carried out with 96
    /// fractional bits, so their rounding stays far below the answer and only
    /// the final rounding to a [`Fixed`] shows.
    pub fn log2(self) -> Option<Self> {
        Some(Self(round_shift(
            self.log2_wide()?,
            Self::LOG_BITS - FRACTION_BITS,
        )))
    }

    /// The natural logarithm, or `None` for a value at or below zero.
    ///
    /// [`Fixed::log2`] scaled by `ln 2`, with the scaling done in the internal
    /// working format rather than in steps, so nothing is lost between the two.
    /// Accurate to one step.
    pub fn ln(self) -> Option<Self> {
        self.scaled_log(WORKING_LN_2)
    }

    /// The base-ten logarithm, or `None` for a value at or below zero.
    ///
    /// As [`Fixed::ln`], scaled by the base-ten logarithm of two. Accurate to
    /// one step.
    pub fn log10(self) -> Option<Self> {
        self.scaled_log(WORKING_LOG10_2)
    }

    /// The logarithm in any base, as `log2(self) / log2(base)`, or `None`
    /// unless both are above zero and the base is not one.
    pub fn log(self, base: Self) -> Option<Self> {
        let value: i128 = self.log2_wide()?;
        let divisor: i128 = base.log2_wide()?;

        if divisor == 0 {
            return None;
        }

        let sign: bool = (value < 0) != (divisor < 0);
        let scaled: wide::U256 = wide::shift_left(value.unsigned_abs(), FRACTION_BITS);
        let magnitude: u128 = wide::divide(scaled, divisor.unsigned_abs())?;

        wide::apply_sign(magnitude, sign).map(Self)
    }

    /// Two raised to this value, or `None` above 95, where the result no longer
    /// fits.
    ///
    /// The whole part of the exponent becomes a shift and only the fraction
    /// needs work, which is what keeps the series short. Correctly rounded:
    /// the result is never more than half a step from the true value. Anything
    /// below -32 gives zero, which is that rounding rather than a failure.
    pub fn exp2(self) -> Option<Self> {
        exp2_wide::<FRACTION_BITS>(self.0 << (Self::LOG_BITS - FRACTION_BITS))
    }

    /// `e` raised to this value, or `None` above about 65.85, where the result
    /// no longer fits.
    ///
    /// The exponent is split as `k * ln 2 + r` with `r` in `[0, ln 2)`, so the
    /// whole of it becomes a shift and the series only ever sees a small
    /// remainder, where it converges in a dozen terms. Correctly rounded, as
    /// [`Fixed::exp2`] is. Anything below about -22.19 gives zero, which is
    /// that rounding rather than a failure: `exp(-30)` really is nearer zero
    /// than to any other value this type can hold, and that is the reason a
    /// distribution's tail cannot be computed here.
    pub fn exp(self) -> Option<Self> {
        // Beyond this the shift below could not fit, let alone the answer.
        if self.0.unsigned_abs() > (1 << 100) {
            return (self.0 < 0).then_some(Self::ZERO);
        }

        let exponent: i128 = self.0 << (WORKING_BITS - FRACTION_BITS);
        let halvings: i128 = exponent.div_euclid(WORKING_LN_2 as i128);
        let remainder: u128 = exponent.rem_euclid(WORKING_LN_2 as i128) as u128;

        place(exponential_series(remainder), halvings)
    }

    /// This value raised to another, as `exp2(log2(self) * exponent)`, or
    /// `None` unless the value is above zero and the result fits.
    ///
    /// Both halves round, so this is the least accurate function here: about
    /// one part in `10^10`, and worse for a large exponent, which multiplies
    /// the logarithm's error along with the logarithm. The exponent is carried
    /// at the logarithm's own precision rather than rounded to a step first,
    /// which is worth a factor of a thousand. [`Fixed::powi`] is closer still
    /// and should be preferred for a whole exponent.
    pub fn pow(self, exponent: Self) -> Option<Self> {
        let logarithm: i128 = self.log2_wide()?;
        let sign: bool = (logarithm < 0) != (exponent.0 < 0);
        let product: wide::U256 =
            wide::multiply(logarithm.unsigned_abs(), exponent.0.unsigned_abs());

        // The product carries LOG_BITS + FRACTION_BITS fractional bits, and the
        // exponential takes LOG_BITS, so the scaled logarithm keeps all the
        // precision it was found with rather than being rounded to a step
        // first.
        let magnitude: u128 = wide::shift_right(product, FRACTION_BITS)?;

        exp2_wide(wide::apply_sign(magnitude, sign)?)
    }

    /// This value raised to a whole power, by repeated squaring, or `None` on
    /// overflow.
    ///
    /// Every multiplication truncates, so the error grows with the number of
    /// squarings, but it never involves a logarithm: about one part in `10^10`
    /// across the range, and exact for small whole bases and exponents.
    ///
    /// A negative exponent inverts either the base or the result, whichever
    /// keeps the intermediate value large: raising a value below one to a
    /// negative power inverts first, since the truncation in a tiny
    /// intermediate would otherwise be magnified by the inversion.
    pub fn powi(self, exponent: i32) -> Option<Self> {
        let invert_first: bool = exponent < 0 && self.abs() < Self::ONE;
        let mut remaining: u32 = exponent.unsigned_abs();
        let mut base: Self = if invert_first {
            self.checked_recip()?
        } else {
            self
        };
        let mut total: Self = Self::ONE;

        while remaining != 0 {
            if remaining & 1 == 1 {
                total = total.checked_mul(base)?;
            }

            remaining >>= 1;

            if remaining != 0 {
                base = base.checked_mul(base)?;
            }
        }

        if exponent < 0 && !invert_first {
            return total.checked_recip();
        }

        Some(total)
    }

    /// The base-two logarithm with [`LOG_BITS`] fractional bits, which is what
    /// every logarithm here is built from.
    fn log2_wide(self) -> Option<i128> {
        if self.0 <= 0 {
            return None;
        }

        let magnitude: u128 = self.0 as u128;
        let highest: u32 = u128::BITS - 1 - magnitude.leading_zeros();

        // What is left once the highest bit is taken out, in `[1, 2)`.
        let mut mantissa: u128 = if highest <= WORKING_BITS {
            magnitude << (WORKING_BITS - highest)
        } else {
            magnitude >> (highest - WORKING_BITS)
        };

        let whole: i128 = highest as i128 - FRACTION_BITS as i128;
        let mut fraction: i128 = 0;

        for place in (0..Self::LOG_BITS).rev() {
            mantissa = wide::shift_right_wrapping(wide::multiply(mantissa, mantissa), WORKING_BITS);

            if mantissa >= WORKING_ONE << 1 {
                mantissa >>= 1;
                fraction |= 1 << place;
            }
        }

        Some((whole << Self::LOG_BITS) + fraction)
    }

    /// A logarithm in the base whose `ln` is `scale`, given in the internal
    /// working format.
    fn scaled_log(self, scale: u128) -> Option<Self> {
        let logarithm: i128 = self.log2_wide()?;
        let product: wide::U256 = wide::multiply(logarithm.unsigned_abs(), scale);

        // The product carries LOG_BITS + WORKING_BITS fractional bits.
        let magnitude: u128 =
            wide::shift_right(product, WORKING_BITS + Self::LOG_BITS - FRACTION_BITS)?;

        wide::apply_sign(magnitude, logarithm < 0).map(Self)
    }
}

/// Two raised to a value carrying [`LOG_BITS`] fractional bits.
///
/// Taking the exponent at the logarithm's own precision rather than at a step
/// is what keeps [`Fixed::pow`] accurate: a step of error in the exponent is a
/// relative error of `ln 2 * 2^-32` in the result, which at a large result is
/// many steps.
fn exp2_wide<const FRACTION_BITS: u32>(exponent: i128) -> Option<FixedPoint<FRACTION_BITS>> {
    let log_bits: u32 = FRACTION_BITS + LOG_GUARD_BITS;
    let whole: i128 = exponent >> log_bits;
    let fraction: u128 = (exponent & ((1 << log_bits) - 1)) as u128;

    // The fraction as an exponent of e, which is what the series takes.
    let scaled: u128 = wide::shift_right_wrapping(
        wide::multiply(fraction << (WORKING_BITS - log_bits), WORKING_LN_2),
        WORKING_BITS,
    );

    place(exponential_series(scaled), whole)
}

/// `e` raised to a value in `[0, ln 2)`, given and returned in the internal
/// working format.
///
/// The Taylor series, whose terms are each the one before times `x / n`. The
/// argument is below 0.694, so the terms fall away by more than a factor of two
/// each time and the sum settles in about a dozen of them; it runs until a term
/// rounds to nothing.
fn exponential_series(exponent: u128) -> u128 {
    let mut term: u128 = WORKING_ONE;
    let mut total: u128 = WORKING_ONE;
    let mut step: u128 = 1;

    while term != 0 {
        term = wide::shift_right_wrapping(wide::multiply(term, exponent), WORKING_BITS) / step;
        total += term;
        step += 1;
    }

    total
}

/// A mantissa in `[1, 2)`, given in the internal working format, doubled
/// `halvings` times and brought back to a [`Fixed`].
///
/// `None` when the result is too large to hold; zero when it is too small,
/// which is the correctly rounded answer rather than a failure.
fn place<const FRACTION_BITS: u32>(mantissa: u128, halvings: i128) -> Option<FixedPoint<FRACTION_BITS>> {
    let shift: i128 = halvings - (WORKING_BITS - FRACTION_BITS) as i128;

    if shift <= -(u128::BITS as i128) {
        return Some(FixedPoint::ZERO);
    }

    if shift < 0 {
        return Some(FixedPoint(round_shift(mantissa as i128, (-shift) as u32)));
    }

    let widened: wide::U256 = wide::shift_left(mantissa, shift as u32);

    (widened.high == 0 && widened.low <= i128::MAX as u128).then_some(FixedPoint(widened.low as i128))
}

/// A signed value shifted right by `places`, rounded to nearest rather than
/// towards negative infinity, which keeps a logarithm or an exponential from
/// being biased low.
fn round_shift(value: i128, places: u32) -> i128 {
    if places == 0 {
        return value;
    }

    let half: i128 = 1 << (places - 1);

    (value + half) >> places
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// Exact, and panics on overflow in every build profile.
impl<const FRACTION_BITS: u32> Add for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.checked_add(other)
            .expect("fixed-point overflow in add")
    }
}

/// Exact, and panics on overflow in every build profile.
impl<const FRACTION_BITS: u32> Sub for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.checked_sub(other)
            .expect("fixed-point overflow in sub")
    }
}

/// Panics on [`Fixed::MIN`], which has no positive counterpart.
impl<const FRACTION_BITS: u32> Neg for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn neg(self) -> Self {
        self.checked_neg().expect("fixed-point overflow in neg")
    }
}

/// Truncates towards zero, and panics on overflow in every build profile; see
/// [`Fixed::checked_mul`].
impl<const FRACTION_BITS: u32> Mul for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.checked_mul(other)
            .expect("fixed-point overflow in mul")
    }
}

/// Truncates towards zero, and panics on a zero divisor or on overflow; see
/// [`Fixed::checked_div`].
impl<const FRACTION_BITS: u32> Div for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert!(!other.is_zero(), "fixed-point division by zero");

        self.checked_div(other)
            .expect("fixed-point overflow in div")
    }
}

/// The remainder, carrying the sign of the left side. Panics on a zero divisor.
impl<const FRACTION_BITS: u32> Rem for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        assert!(!other.is_zero(), "fixed-point remainder by zero");

        self.checked_rem(other)
            .expect("fixed-point overflow in rem")
    }
}

/// Scales by a whole number, which needs no rescaling afterwards.
impl<const FRACTION_BITS: u32> Mul<i128> for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn mul(self, factor: i128) -> Self {
        Self(
            self.0
                .checked_mul(factor)
                .expect("fixed-point overflow in mul"),
        )
    }
}

/// Divides by a whole number, truncating towards zero.
impl<const FRACTION_BITS: u32> Div<i128> for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn div(self, divisor: i128) -> Self {
        assert!(divisor != 0, "fixed-point division by zero");

        Self(
            self.0
                .checked_div(divisor)
                .expect("fixed-point overflow in div"),
        )
    }
}

/// Doubles the value `places` times, which is exact until it overflows.
impl<const FRACTION_BITS: u32> Shl<u32> for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn shl(self, places: u32) -> Self {
        Self(
            self.0
                .checked_shl(places)
                .filter(|bits| bits >> places == self.0)
                .expect("fixed-point overflow in shl"),
        )
    }
}

/// Halves the value `places` times, rounding towards negative infinity.
impl<const FRACTION_BITS: u32> Shr<u32> for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn shr(self, places: u32) -> Self {
        Self(
            self.0
                .checked_shr(places)
                .expect("fixed-point shift too far"),
        )
    }
}

macro_rules! implement_assign {
    ($trait:ident, $method:ident, $operator:tt, $rhs:ty) => {
        impl $trait<$rhs> for Fixed {
            fn $method(&mut self, other: $rhs) {
                *self = *self $operator other;
            }
        }
    };
}

implement_assign!(AddAssign, add_assign, +, Fixed);
implement_assign!(SubAssign, sub_assign, -, Fixed);
implement_assign!(MulAssign, mul_assign, *, Fixed);
implement_assign!(DivAssign, div_assign, /, Fixed);
implement_assign!(RemAssign, rem_assign, %, Fixed);
implement_assign!(MulAssign, mul_assign, *, i128);
implement_assign!(DivAssign, div_assign, /, i128);
implement_assign!(ShlAssign, shl_assign, <<, u32);
implement_assign!(ShrAssign, shr_assign, >>, u32);

/// Adds a sequence up, panicking on overflow as [`Add`] does.
impl<const FRACTION_BITS: u32> Sum for FixedPoint<FRACTION_BITS> {
    fn sum<I: Iterator<Item = Self>>(values: I) -> Self {
        values.fold(Self::ZERO, |total, value| total + value)
    }
}

/// Multiplies a sequence together, panicking on overflow as [`Mul`] does.
impl<const FRACTION_BITS: u32> Product for FixedPoint<FRACTION_BITS> {
    fn product<I: Iterator<Item = Self>>(values: I) -> Self {
        values.fold(Self::ONE, |total, value| total * value)
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

macro_rules! implement_from_integer {
    ($($integer:ty),+) => {
        $(
            #[doc = concat!("Every `", stringify!($integer), "` is a whole number of units, so this cannot fail.")]
            impl From<$integer> for Fixed {
                fn from(whole: $integer) -> Self {
                    Self(whole as i128 * SCALE)
                }
            }
        )+
    };
}

implement_from_integer!(i8, i16, i32, i64, u8, u16, u32);

/// A `u64` is a whole number of units, and still fits after scaling.
impl<const FRACTION_BITS: u32> From<u64> for FixedPoint<FRACTION_BITS> {
    fn from(whole: u64) -> Self {
        Self(whole as i128 * Self::SCALE)
    }
}

/// Whole units, or an error when they are too large to scale.
impl<const FRACTION_BITS: u32> TryFrom<i128> for FixedPoint<FRACTION_BITS> {
    type Error = &'static str;

    fn try_from(whole: i128) -> Result<Self, Self::Error> {
        Self::from_integer_i128(whole).ok_or("whole value is outside the fixed-point range")
    }
}

/// As [`Fixed::from_f64`], with a message instead of `None`.
impl<const FRACTION_BITS: u32> TryFrom<f64> for FixedPoint<FRACTION_BITS> {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_f64(value).ok_or("value is not finite, or outside the fixed-point range")
    }
}

/// The nearest `f64`; see [`Fixed::to_f64`].
impl<const FRACTION_BITS: u32> From<FixedPoint<FRACTION_BITS>> for f64 {
    fn from(value: FixedPoint<FRACTION_BITS>) -> Self {
        value.to_f64()
    }
}

/// The nearest `f32`; see [`Fixed::to_f32`].
impl<const FRACTION_BITS: u32> From<FixedPoint<FRACTION_BITS>> for f32 {
    fn from(value: FixedPoint<FRACTION_BITS>) -> Self {
        value.to_f32()
    }
}

/// As [`Fixed::from_f32`], with a message instead of `None`.
impl<const FRACTION_BITS: u32> TryFrom<f32> for FixedPoint<FRACTION_BITS> {
    type Error = &'static str;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::from_f32(value).ok_or("value is not finite, or outside the fixed-point range")
    }
}

/// Whole units from a `usize`, which is at most 64 bits on any target, so this
/// cannot fail.
impl<const FRACTION_BITS: u32> From<usize> for FixedPoint<FRACTION_BITS> {
    fn from(whole: usize) -> Self {
        Self(whole as i128 * Self::SCALE)
    }
}

/// Whole units from an `isize`, which is at most 64 bits on any target.
impl<const FRACTION_BITS: u32> From<isize> for FixedPoint<FRACTION_BITS> {
    fn from(whole: isize) -> Self {
        Self(whole as i128 * Self::SCALE)
    }
}

/// Whole units, or an error when they are too large to scale.
impl<const FRACTION_BITS: u32> TryFrom<u128> for FixedPoint<FRACTION_BITS> {
    type Error = &'static str;

    fn try_from(whole: u128) -> Result<Self, Self::Error> {
        i128::try_from(whole)
            .ok()
            .and_then(Self::from_integer_i128)
            .ok_or("whole value is outside the fixed-point range")
    }
}

/// The whole part, truncated towards zero; see [`Fixed::to_integer`]. Every
/// value fits, so this cannot fail.
impl<const FRACTION_BITS: u32> From<FixedPoint<FRACTION_BITS>> for i128 {
    fn from(value: FixedPoint<FRACTION_BITS>) -> Self {
        value.to_integer()
    }
}

/// The whole part of a [`Fixed`], truncated towards zero, for the integer
/// widths that cannot hold every one of them.
macro_rules! implement_to_integer {
    ($($integer:ty),+) => {
        $(
            #[doc = concat!("The whole part, truncated towards zero, or an error when it is outside `", stringify!($integer), "`.")]
            impl TryFrom<Fixed> for $integer {
                type Error = &'static str;

                fn try_from(value: Fixed) -> Result<Self, Self::Error> {
                    Self::try_from(value.to_integer())
                        .map_err(|_| concat!("value is outside ", stringify!($integer)))
                }
            }
        )+
    };
}

implement_to_integer!(i8, i16, i32, i64, u8, u16, u32, u64, u128, usize, isize);

// ---------------------------------------------------------------------------
// Reading and writing
// ---------------------------------------------------------------------------

/// The shortest decimal that reads back as this exact value, which is at most
/// ten fractional digits and none at all for a whole number.
///
/// A step of `2^-32` is a finite decimal, but writing it out in full takes 32
/// digits: the stored value nearest `0.1` is exactly
/// `0.09999999986030161380767822265625`. Printing the shortest decimal that
/// [`FromStr`] maps back to the same bits keeps text round-trips honest and
/// readable at once. [`Fixed::to_exact_string`] writes the full expansion where
/// that is what is wanted.
///
/// A precision, as in `{:.3}`, gives exactly that many fractional digits,
/// truncated rather than rounded, so every digit printed is one the value
/// really has.
impl<const FRACTION_BITS: u32> fmt::Display for FixedPoint<FRACTION_BITS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let negative: bool = self.0 < 0;
        let magnitude: u128 = self.0.unsigned_abs();
        let whole: u128 = magnitude >> FRACTION_BITS;
        let fraction: u128 = magnitude & (Self::SCALE as u128 - 1);

        let digits: String = match formatter.precision() {
            Some(wanted) => truncated_digits(fraction, wanted, FRACTION_BITS),
            None => shortest_digits(fraction, FRACTION_BITS),
        };

        let text: String = if digits.is_empty() {
            whole.to_string()
        } else {
            format!("{whole}.{digits}")
        };

        formatter.pad_integral(!negative, "", &text)
    }
}

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// Every digit of the exact decimal value, which for a fraction is always
    /// 32 of them.
    ///
    /// Each step is `2^-32`, and a negative power of two is a finite decimal,
    /// so this terminates and is exact rather than rounded. Useful for a
    /// snapshot or a golden test, where the point is to pin the stored bits and
    /// not to be read.
    pub fn to_exact_string(self) -> String {
        let magnitude: u128 = self.0.unsigned_abs();
        let whole: u128 = magnitude >> FRACTION_BITS;
        let digits: String =
            truncated_digits(
                magnitude & (Self::SCALE as u128 - 1),
                FRACTION_BITS as usize,
                FRACTION_BITS,
            );
        let sign: &str = if self.0 < 0 { "-" } else { "" };

        if digits.is_empty() {
            return format!("{sign}{whole}");
        }

        format!("{sign}{whole}.{digits}")
    }
}

/// The first `wanted` digits of a fraction's decimal expansion, taken by
/// repeatedly multiplying by ten and lifting the digit that carries past the
/// point. Empty once the expansion has ended.
fn truncated_digits(fraction: u128, wanted: usize, fraction_bits: u32) -> String {
    let scale: u128 = 1u128 << fraction_bits;
    let mut remaining: u128 = fraction;
    let mut digits: String = String::new();

    while digits.len() < wanted {
        remaining *= 10;
        digits.push(char::from_digit((remaining >> fraction_bits) as u32, 10).unwrap());
        remaining &= scale - 1;
    }

    digits
}

/// The fewest digits, rounded to nearest, that read back as this exact
/// fraction.
///
/// Enough digits always suffice: a step of `2^-f` is coarser than `10^-ceil(f/3)`,
/// so a little over a third of the fractional bits is always a wide enough decimal.
/// At the default 32 bits that is the familiar ten digits.
fn shortest_digits(fraction: u128, fraction_bits: u32) -> String {
    if fraction == 0 {
        return String::new();
    }

    let scale: u128 = 1u128 << fraction_bits;
    // log10(2) is a little over 0.301, so this never under-counts.
    let limit: usize = (fraction_bits as usize * 302 / 1000) + 1;
    let mut power: u128 = 1;

    for places in 1..=limit {
        power *= 10;

        let rounded: u128 = ((fraction * power) + scale / 2) >> fraction_bits;

        if rounded < power && (rounded << fraction_bits) / power == fraction {
            return format!("{rounded:0>places$}");
        }
    }

    let rounded: u128 = ((fraction * power) + scale / 2) >> fraction_bits;

    format!("{rounded:0>limit$}")
}

/// As [`Display`](fmt::Display), tagged so a value is recognisable in a dump.
impl<const FRACTION_BITS: u32> fmt::Debug for FixedPoint<FRACTION_BITS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Fixed({self})")
    }
}

/// What went wrong while reading a [`Fixed`] from text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseFixedError {
    /// The text held no digits at all, or nothing before the point.
    Empty,
    /// The text held something that is not a digit, a sign or a single point.
    Invalid,
    /// The value is outside the range a [`Fixed`] can hold.
    OutOfRange,
}

impl fmt::Display for ParseFixedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "no digits to read",
            Self::Invalid => "not a fixed-point number",
            Self::OutOfRange => "outside the fixed-point range",
        })
    }
}

impl std::error::Error for ParseFixedError {}

impl From<ParseIntError> for ParseFixedError {
    fn from(_: ParseIntError) -> Self {
        Self::Invalid
    }
}

/// Reads decimal text such as `-12.25`, exactly.
///
/// The whole part is read as an integer and the fractional digits as a fraction
/// over a power of ten, then scaled and truncated towards zero, so the result
/// is the nearest representable value at or before the one written. No `f64` is
/// involved at any point, which is what keeps a written-down value and the one
/// read back from it in step.
impl<const FRACTION_BITS: u32> FromStr for FixedPoint<FRACTION_BITS> {
    type Err = ParseFixedError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (negative, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };

        let (whole_text, fraction_text) = match digits.split_once('.') {
            Some((whole, fraction)) => (whole, fraction),
            None => (digits, ""),
        };

        if whole_text.is_empty() && fraction_text.is_empty() {
            return Err(ParseFixedError::Empty);
        }

        if !fraction_text.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(ParseFixedError::Invalid);
        }

        let whole: u128 = if whole_text.is_empty() {
            0
        } else {
            whole_text.parse::<u128>()?
        };

        // Read the fractional digits as one integer over a power of ten, then
        // scale that fraction up in 256 bits, so no digit is lost on the way.
        let mut written: u128 = 0;
        let mut power: u128 = 1;

        for digit in fraction_text.bytes().take(38).map(u128::from) {
            let Some(next) = power.checked_mul(10) else {
                break;
            };

            written = written * 10 + (digit - u128::from(b'0'));
            power = next;
        }

        let fraction: u128 = wide::divide(wide::shift_left(written, FRACTION_BITS), power)
            .ok_or(ParseFixedError::OutOfRange)?;

        let magnitude: u128 = whole
            .checked_shl(FRACTION_BITS)
            .filter(|scaled| scaled >> FRACTION_BITS == whole)
            .and_then(|scaled| scaled.checked_add(fraction))
            .ok_or(ParseFixedError::OutOfRange)?;

        wide::apply_sign(magnitude, negative)
            .map(Self)
            .ok_or(ParseFixedError::OutOfRange)
    }
}

// ---------------------------------------------------------------------------
// 256-bit helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Trigonometry
//
// All of it runs on `cordic`, which is integer shifts and adds at 96 fractional
// bits. Nothing here touches a float, so every result is bit-identical on every
// platform — which is the whole reason this type exists.
// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// Narrows a value from the internal 96-bit working format, rounding once.
    ///
    /// Rounding here and nowhere else is what keeps the error analysis simple: the
    /// loop's own error is bounded far below the last bit of this layout, so this
    /// single rounding is the only one that reaches the result.
    fn from_internal(value: i128) -> Self {
        let shift: u32 = cordic::BITS - FRACTION_BITS;

        Self((value + (1 << (shift - 1))) >> shift)
    }

    /// The sine and cosine of an angle in radians, together.
    ///
    /// # Question
    ///
    /// "Where on the unit circle is this angle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// let (sine, cosine) = Fixed::FRAC_PI_2.sin_cos();
    ///
    /// assert!((sine.to_f64() - 1.0).abs() < 1e-9);
    /// assert!(cosine.to_f64().abs() < 1e-9);
    /// ```
    ///
    /// # Why both at once
    ///
    /// CORDIC produces the whole vector, so the cosine is already computed when the
    /// sine is. Asking for them separately does the identical work twice, and
    /// anything rotating a point needs both. This is the method to reach for; [`sin`]
    /// and [`cos`] discard half of it.
    ///
    /// [`sin`]: Self::sin
    /// [`cos`]: Self::cos
    ///
    /// # Accuracy
    ///
    /// Within one step of the true value for any angle of modest size. The working
    /// precision is 96 fractional bits against at most 64 here, and the reduction
    /// error grows as `|angle| × 2⁻⁹³`, so an angle would have to exceed `2²⁹`
    /// radians before it reached the last bit at 64 fractional bits.
    pub fn sin_cos(self) -> (Self, Self) {
        let (cosine, sine): (i128, i128) = cordic::cosine_sine(self.0, FRACTION_BITS);

        (Self::from_internal(sine), Self::from_internal(cosine))
    }

    /// The sine of an angle in radians.
    ///
    /// # Question
    ///
    /// "How far above the axis is this angle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.sin(), Fixed::ZERO);
    /// assert!(Fixed::PI.sin().to_f64().abs() < 1e-9);
    /// ```
    ///
    /// Always in `[-1, 1]`, so it cannot fail and returns no [`Option`]. Prefer
    /// [`sin_cos`](Self::sin_cos) where both are wanted: it costs the same as this.
    pub fn sin(self) -> Self {
        self.sin_cos().0
    }

    /// The cosine of an angle in radians.
    ///
    /// # Question
    ///
    /// "How far along the axis is this angle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.cos(), Fixed::ONE);
    /// assert!((Fixed::PI.cos() + Fixed::ONE).to_f64().abs() < 1e-9);
    /// ```
    ///
    /// Always in `[-1, 1]`, so it cannot fail. Prefer [`sin_cos`](Self::sin_cos)
    /// where both are wanted.
    pub fn cos(self) -> Self {
        self.sin_cos().1
    }

    /// The tangent of an angle in radians, or [`None`] where it does not fit.
    ///
    /// # Question
    ///
    /// "What is the slope of this angle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// let eighth = Fixed::FRAC_PI_4.tan().unwrap();
    ///
    /// assert!((eighth.to_f64() - 1.0).abs() < 1e-9);
    /// ```
    ///
    /// # Why this one returns an [`Option`] when [`sin`](Self::sin) does not
    ///
    /// A tangent is unbounded. Near a quarter turn it exceeds anything this type can
    /// hold, and exactly at one it is undefined. Both come back as [`None`] rather
    /// than a saturated value, because a slope of "the largest number available" is
    /// not an answer and would quietly poison whatever used it.
    ///
    /// Reaching the exact quarter turn takes an angle whose cosine rounds to zero at
    /// 96 fractional bits, which an angle stored at 64 or fewer effectively cannot
    /// miss — so the undefined case is reported, not approximated.
    pub fn tan(self) -> Option<Self> {
        let (cosine, sine): (i128, i128) = cordic::cosine_sine(self.0, FRACTION_BITS);

        if cosine == 0 {
            return None;
        }

        let negative: bool = (sine < 0) != (cosine < 0);

        let quotient: u128 = wide::divide(
            wide::shift_left(sine.unsigned_abs(), FRACTION_BITS),
            cosine.unsigned_abs(),
        )?;

        wide::apply_sign(quotient, negative).map(Self)
    }

    /// The angle of the point `(x, self)`, over the whole circle `(-π, π]`.
    ///
    /// # Question
    ///
    /// "Which way does this vector point?"
    ///
    /// The receiver is the **y** component and the argument the **x**, matching
    /// `atan2(y, x)` everywhere else in mathematics and in the standard library.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// let north_east = Fixed::ONE.atan2(Fixed::ONE);
    ///
    /// assert!((north_east.to_f64() - Fixed::FRAC_PI_4.to_f64()).abs() < 1e-9);
    ///
    /// // Behind, which `atan` alone could never tell from ahead.
    /// let behind = Fixed::ZERO.atan2(Fixed::NEGATIVE_ONE);
    ///
    /// assert!((behind.to_f64() - Fixed::PI.to_f64()).abs() < 1e-9);
    /// ```
    ///
    /// # Why this and not [`atan`](Self::atan)
    ///
    /// `atan(y/x)` loses the quadrant, because `y/x` is the same for a direction and
    /// its opposite — and it divides, which fails when `x` is zero and loses
    /// precision when `x` is small. This takes both components, so it keeps the
    /// quadrant, handles every axis including `(0, 0)`, and never divides. For a
    /// direction in a world, this is almost always the one that was wanted.
    ///
    /// `(0, 0)` has no angle; zero is returned, as the standard library does.
    ///
    /// # Accuracy
    ///
    /// The pair is rescaled to use the full width before the loop, so a direction
    /// built from two small components is as accurate as one built from two large
    /// ones. No argument reduction is involved, so there is no large-input
    /// degradation here.
    pub fn atan2(self, x: Self) -> Self {
        Self::from_internal(cordic::arctangent2(self.0, x.0))
    }

    /// The angle whose tangent is this, in `(-π/2, π/2)`.
    ///
    /// # Question
    ///
    /// "What angle has this slope?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert!((Fixed::ONE.atan().to_f64() - Fixed::FRAC_PI_4.to_f64()).abs() < 1e-9);
    /// assert_eq!(Fixed::ZERO.atan(), Fixed::ZERO);
    /// ```
    ///
    /// Bounded by a quarter turn, so it cannot fail. Use
    /// [`atan2`](Self::atan2) when the two components are available separately: it
    /// keeps the quadrant that a slope alone has already lost.
    pub fn atan(self) -> Self {
        self.atan2(Self::ONE)
    }

    /// The angle whose sine is this, in `[-π/2, π/2]`, or [`None`] outside `[-1, 1]`.
    ///
    /// # Question
    ///
    /// "What angle has this height on the unit circle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ONE.asin(), Some(Fixed::FRAC_PI_2));
    /// assert_eq!(Fixed::from_integer(2).asin(), None);
    /// ```
    ///
    /// # How it is built
    ///
    /// `asin(v) = atan2(v, √(1 − v²))`, which is exact as a relationship and needs no
    /// approximation of its own. Going through [`atan2`](Self::atan2) rather than
    /// dividing keeps the `v = ±1` case working, where the root is zero and a
    /// division would not be.
    ///
    /// # Accuracy
    ///
    /// Worst near `±1`, where the square root's own error is magnified by the steep
    /// slope of `asin`. In the middle of the range it is accurate to the last step or
    /// so. Where that matters, prefer [`atan2`](Self::atan2) on the two components
    /// directly and never form the sine at all.
    pub fn asin(self) -> Option<Self> {
        let square: Self = self.checked_mul(self)?;

        // `square` is in [0, 1] whenever the input is, so this cannot overflow; and
        // where it is not, `sqrt` of the negative returns None, which is the answer.
        let root: Self = Self(Self::ONE.0 - square.0).sqrt()?;

        Some(self.atan2(root))
    }

    /// The angle whose cosine is this, in `[0, π]`, or [`None`] outside `[-1, 1]`.
    ///
    /// # Question
    ///
    /// "What angle has this horizontal extent on the unit circle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ONE.acos(), Some(Fixed::ZERO));
    /// assert_eq!(Fixed::from_integer(-2).acos(), None);
    /// ```
    ///
    /// `acos(v) = atan2(√(1 − v²), v)`, the same construction as
    /// [`asin`](Self::asin) with the arguments the other way round, and with the same
    /// loss of accuracy near `±1`. This is the usual way to turn a dot product of two
    /// unit vectors into the angle between them.
    pub fn acos(self) -> Option<Self> {
        let square: Self = self.checked_mul(self)?;
        let root: Self = Self(Self::ONE.0 - square.0).sqrt()?;

        Some(root.atan2(self))
    }

    /// The hyperbolic sine, or [`None`] where it does not fit.
    ///
    /// # Question
    ///
    /// "What is `(eˣ − e⁻ˣ)/2` here?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.sinh(), Some(Fixed::ZERO));
    /// ```
    ///
    /// Built from [`exp`](Self::exp), which is itself integer-only, so this inherits
    /// that accuracy and determinism. It grows as fast as `eˣ` does, so it returns
    /// [`None`] once the result leaves the layout's range.
    pub fn sinh(self) -> Option<Self> {
        let up: Self = self.exp()?;
        let down: Self = Self(-self.0).exp()?;

        Some(Self(up.0.checked_sub(down.0)? / 2))
    }

    /// The hyperbolic cosine, or [`None`] where it does not fit.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.cosh(), Some(Fixed::ONE));
    /// ```
    ///
    /// `(eˣ + e⁻ˣ)/2`, built from [`exp`](Self::exp). Never below one, and growing as
    /// fast as `eˣ`.
    pub fn cosh(self) -> Option<Self> {
        let up: Self = self.exp()?;
        let down: Self = Self(-self.0).exp()?;

        Some(Self(up.0.checked_add(down.0)? / 2))
    }

    /// The hyperbolic tangent, always in `(-1, 1)`.
    ///
    /// # Question
    ///
    /// "What is this value, squashed into `(-1, 1)`?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.tanh(), Fixed::ZERO);
    /// assert_eq!(Fixed::from_integer(100).tanh(), Fixed::ONE);
    /// ```
    ///
    /// # Why this cannot fail where [`sinh`](Self::sinh) can
    ///
    /// It is computed as `(e²ˣ − 1)/(e²ˣ + 1)` rather than as a ratio of the two
    /// above, so the intermediate that would overflow never appears on its own. Past
    /// the point where the result is within one step of `±1` — about 23, since
    /// `1 − tanh(x) ≈ 2e⁻²ˣ` — that bound is returned directly, which is the
    /// correctly rounded answer rather than a saturation.
    pub fn tanh(self) -> Self {
        // Beyond this the true value is nearer to ±1 than to any other representable
        // value, so returning the bound is exact, not a clamp.
        const SATURATION: i64 = 23;

        if self.0 >= Self::from_integer(SATURATION).0 {
            return Self::ONE;
        }

        if self.0 <= Self::from_integer(-SATURATION).0 {
            return Self::NEGATIVE_ONE;
        }

        let Some(doubled) = Self(self.0 * 2).exp() else {
            return Self::ONE;
        };

        let numerator: i128 = doubled.0 - Self::ONE.0;
        let denominator: i128 = doubled.0 + Self::ONE.0;

        let negative: bool = numerator < 0;

        let Some(quotient) = wide::divide(
            wide::shift_left(numerator.unsigned_abs(), FRACTION_BITS),
            denominator.unsigned_abs(),
        ) else {
            return if negative { Self::NEGATIVE_ONE } else { Self::ONE };
        };

        wide::apply_sign(quotient, negative).map_or(Self::ONE, Self)
    }
}

/// The 256-bit arithmetic a 128-bit fixed-point type needs in the middle of a
/// multiplication, a division or a square root.
///
/// Nothing here is a general-purpose wide integer: each function does exactly
/// what one [`Fixed`] operation needs, on magnitudes, with the sign handled by
/// the caller.
///
/// # Why this is not [`WideUint<4>`](crate::math::WideUint)
///
/// The crate does have a general 256-bit integer, and it would remove every line
/// below. It is far too slow to put here. Measured over 200,000 pairs:
///
/// | operation | this module | `WideUint<4>` |
/// |---|---|---|
/// | 256-bit multiply | 144 µs | 1.18 ms (8.2×) |
/// | multiply then divide | 1.19 ms | 147 ms (124×) |
///
/// The gap is structural, not an accident of tuning. This module divides a
/// 256-bit value by walking 128 bits with native `u128` compares; `WideUint<4>`
/// walks 256 bits and does a four-limb compare and subtraction at each one. Since
/// [`Fixed`] exists to be the arithmetic used *instead of* `f64` on hot paths,
/// paying 124× for tidiness is the wrong trade.
///
/// What the general type is used for instead is checking this one: the tests
/// in `testing/tests/fixed_wide.rs` run both over the same inputs and require
/// them to agree, so the fast code has an independent oracle rather than only its
/// own expectations. That is why the items here are `pub` and hidden — an
/// integration test cannot reach a crate-private module.
#[doc(hidden)]
pub mod wide {
    use crate::math::WideUint;

    /// A 256-bit unsigned value, as two halves.
    ///
    /// Ordered by the high half first, which is the value's own order.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct U256 {
        pub high: u128,
        pub low: u128,
    }

    /// The exact 256-bit product of two 128-bit values, by the schoolbook
    /// method on 64-bit halves, carrying between the partial products.
    pub fn multiply(left: u128, right: u128) -> U256 {
        const HALF: u32 = 64;
        let mask: u128 = u64::MAX as u128;

        let (left_high, left_low): (u128, u128) = (left >> HALF, left & mask);
        let (right_high, right_low): (u128, u128) = (right >> HALF, right & mask);

        let low_low: u128 = left_low * right_low;
        let cross_one: u128 = left_high * right_low;
        let cross_two: u128 = left_low * right_high;
        let high_high: u128 = left_high * right_high;

        let (cross, crossed): (u128, bool) = cross_one.overflowing_add(cross_two);
        let (low, carried): (u128, bool) = low_low.overflowing_add(cross << HALF);

        let mut high: u128 = high_high + (cross >> HALF) + u128::from(carried);

        if crossed {
            high += 1 << HALF;
        }

        U256 { high, low }
    }

    /// A 128-bit value widened and scaled up by `places` bits.
    pub fn shift_left(value: u128, places: u32) -> U256 {
        if places == 0 {
            return U256 {
                high: 0,
                low: value,
            };
        }

        U256 {
            high: value >> (u128::BITS - places),
            low: value << places,
        }
    }

    /// The value scaled down by `places` bits, or `None` if what remains does
    /// not fit in 128 bits.
    pub fn shift_right(value: U256, places: u32) -> Option<u128> {
        (value.high >> places == 0).then(|| shift_right_wrapping(value, places))
    }

    /// The low 128 bits of the value scaled down by `places` bits.
    pub fn shift_right_wrapping(value: U256, places: u32) -> u128 {
        if places == 0 {
            return value.low;
        }

        (value.low >> places) | (value.high << (u128::BITS - places))
    }

    /// The quotient of a 256-bit value by a 128-bit one, or `None` for a zero
    /// divisor or a quotient too large to hold.
    ///
    /// # Two paths, for a measured reason
    ///
    /// A numerator that fits in 128 bits is divided directly, which is one
    /// instruction or one compiler-runtime call. That covers every ordinary
    /// [`Fixed`](super::Fixed): the numerator here is a magnitude shifted up by
    /// [`FRACTION_BITS`](super::FRACTION_BITS), so `high` stays zero for anything
    /// below `2⁹⁶`. Measured at 17 ns, against 23 ns to go through
    /// [`WideUint<4>`](crate::math::WideUint) — so this path earns its keep and stays.
    ///
    /// Anything wider hands the work to `WideUint<4>`, whose division is Knuth's
    /// Algorithm D. That replaced a bit-at-a-time loop which took **377 ns** where
    /// this takes **40 ns** — nine times faster, and one fewer implementation of
    /// long division for the crate to be right about.
    ///
    /// The guard above establishes `high < divisor`, so the quotient is below
    /// `2¹²⁸` and always fits.
    pub fn divide(numerator: U256, divisor: u128) -> Option<u128> {
        if divisor == 0 || numerator.high >= divisor {
            return None;
        }

        if numerator.high == 0 {
            return Some(numerator.low / divisor);
        }

        let wide: WideUint<4> = WideUint::from_limbs([
            numerator.low as u64,
            (numerator.low >> 64) as u64,
            numerator.high as u64,
            (numerator.high >> 64) as u64,
        ]);

        let (quotient, _) = wide.div_rem(&WideUint::from(divisor))?;

        quotient.to_u128()
    }

    /// The largest value whose square does not exceed this one.
    ///
    /// A value that fits in 128 bits goes straight to the standard library's
    /// integer square root, which covers every [`Fixed`](super::Fixed) below
    /// `2^64`. A wider one is found one bit at a time from the top, testing
    /// each candidate by squaring it back out to 256 bits.
    pub fn square_root(value: U256) -> u128 {
        if value.high == 0 {
            return value.low.isqrt();
        }

        // The root of the high half, shifted back up, is never above the true
        // root and never more than `2^64` below it, so only the low half of the
        // answer is left to find bit by bit.
        let mut root: u128 = value.high.isqrt() << (u128::BITS / 2);

        for place in (0..u128::BITS / 2).rev() {
            let candidate: u128 = root | (1 << place);

            if multiply(candidate, candidate) <= value {
                root = candidate;
            }
        }

        root
    }

    /// A magnitude given its sign, or `None` when the negative range cannot
    /// hold it.
    pub fn apply_sign(magnitude: u128, negative: bool) -> Option<i128> {
        if negative {
            (magnitude <= i128::MAX as u128 + 1).then(|| (magnitude as i128).wrapping_neg())
        } else {
            (magnitude <= i128::MAX as u128).then_some(magnitude as i128)
        }
    }
}
