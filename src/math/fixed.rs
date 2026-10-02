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
//! close to the answer. Measured against an independent high-precision
//! evaluation, over every supported width:
//!
//! | | worst error, result at most one | larger results |
//! |---|---|---|
//! | `exp`, `exp2`, `log2` | half a step | grows with the result |
//! | `ln`, `log10`, `sqrt` | one step | one step |
//! | `pow` | half a step | about one part in `10^10` |
//!
//! The qualification matters for the exponentials. A fixed-point step is a fixed
//! *absolute* size, so a large result is a large number of steps, and an
//! algorithm with bounded relative error necessarily spans more of them: `exp`
//! at 64 fractional bits was 100 steps out at a result near `7e10`, which is a
//! relative error below `2^-93`. That is not a defect, but it is not "correctly
//! rounded" either, and this file used to say it was.
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
use crate::math::decimal;

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

    /// `ln 2` as `LN_2_NUMERATOR / 2^16`, rounded **up** so it over-estimates.
    ///
    /// The exponential limits below are derived from it, and they are only sound if
    /// the approximation errs on the generous side: an over-estimate of `ln 2` makes
    /// the admitted domain slightly too wide, which the exact arithmetic downstream
    /// then rejects, where an under-estimate would reject values that do fit.
    const LN_2_NUMERATOR: i128 = 45_427;

    /// The whole part of an exponent at or above which `2^x` cannot fit.
    ///
    /// The largest value is just under `2^(127 - FRACTION_BITS)`, so that exponent
    /// is already out of range. Derived from the layout, not chosen.
    const EXP2_TOO_LARGE: i128 = 127 - FRACTION_BITS as i128;

    /// The whole part of an exponent below which `2^x` rounds to zero.
    ///
    /// Half a step is `2^-(FRACTION_BITS + 1)`, and anything under that is nearer
    /// zero than to the smallest positive value.
    const EXP2_UNDERFLOWS: i128 = -(FRACTION_BITS as i128) - 1;

    /// The whole part of an exponent above which `e^x` cannot fit.
    const EXP_TOO_LARGE: i128 = (Self::EXP2_TOO_LARGE * Self::LN_2_NUMERATOR) >> 16;

    /// The whole part of an exponent below which `e^x` rounds to zero.
    const EXP_UNDERFLOWS: i128 = -(((FRACTION_BITS as i128 + 1) * Self::LN_2_NUMERATOR) >> 16) - 1;

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
        Self(round_shift(reference, REFERENCE_BITS - FRACTION_BITS))
    }
}

// ---------------------------------------------------------------------------
// Building and converting
// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// The value whose raw count of `2^-FRACTION_BITS` is `bits`.
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

    /// The raw count of `2^-FRACTION_BITS` underneath, for storing or hashing.
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

    /// The value obtained by **truncating towards zero**, or `None` if the input is
    /// not finite or lies outside the range.
    ///
    /// Towards zero, not downwards: `1.9` gives `1.9` cut down to the grid, and
    /// `-1.9` gives `-1.9` cut *up* towards zero. The two directions are not the
    /// same, and this is not a floor.
    ///
    /// Scaling by `2^FRACTION_BITS` only changes an `f64`'s exponent, so nothing is
    /// lost before the truncation, and the cast that follows truncates towards zero
    /// by a rule Rust pins down. The conversion is therefore as reproducible as the
    /// rest of the type.
    ///
    /// # When to use this, and when not to
    ///
    /// This answers "I already hold a binary floating-point number — convert it",
    /// which is the right question for a sensor reading, a parsed configuration value
    /// or anything arriving from a float API.
    ///
    /// It is the wrong question for a constant written in source. By the time this
    /// method sees `0.1`, the compiler has already replaced it with the nearest
    /// `f64`, and the original decimal cannot be recovered. Use
    /// [`fixed!`](crate::fixed) for that: it reads the decimal text itself and never
    /// involves a float at all.
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
    /// # Which values survive exactly
    ///
    /// An `f64` carries 53 bits of significand, so the conversion is exact exactly
    /// while the **raw integer** fits in 53 bits — that is, for values under
    /// `2^(53 - FRACTION_BITS)` whole units. At the default 32 fractional bits that
    /// is `2^21`, about 2.1 million whole units, not the 9 million this once claimed.
    ///
    /// Larger values are not all lost: one whose low bits happen to be zero still
    /// converts exactly, because it needs fewer significant bits than its magnitude
    /// suggests. But no *guarantee* covers them, and beyond that bound is where
    /// determinism stops being free.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    /// The nearest `f32`, for a renderer or a vertex buffer.
    ///
    /// # Which values survive exactly
    ///
    /// An `f32` carries 24 bits of significand, so by the same argument as
    /// [`FixedPoint::to_f64`] the exact range is values under `2^(24 - FRACTION_BITS)`
    /// whole units. At the default 32 fractional bits that exponent is negative: only
    /// values below `2^-8` are guaranteed exact, and an ordinary coordinate is not
    /// among them. This is a lossy conversion for rendering, and is meant to be.
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
        let half: i128 = if self.0 < 0 {
            -(Self::SCALE / 2)
        } else {
            Self::SCALE / 2
        };

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
    /// Computed digit by digit in integers on the value scaled up by
    /// `2^FRACTION_BITS`, held in 256 bits, so the result is the largest value of
    /// this layout whose square does not exceed this one. Exact in the sense that
    /// matters: no approximation of a real-valued function is involved, and no
    /// platform can disagree about it.
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
include!("fixed/transcendental.rs");
include!("fixed/operators.rs");
include!("fixed/conversions.rs");
include!("fixed/formatting.rs");
include!("fixed/trigonometry.rs");
