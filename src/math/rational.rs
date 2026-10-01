//! Exact rational numbers: a fraction of two integers, always reduced.
//!
//! The companion to [`Fixed`]. Both avoid the
//! platform-dependent parts of floating point, and they fail in opposite
//! directions: a [`Fixed`] holds any value in its range to a step of `2^-32`
//! and rounds everything else, while a [`Ratio`] holds a
//! third or a seventh exactly and fails when a numerator or denominator no
//! longer fits in 64 bits.
//!
//! Reach for a ratio where the value *is* a count over a count and the answer
//! has to be right: one voxel in a thousand, three chances in seven, a mean of
//! seven halves. A third is not an `f64` and not a [`Fixed`] either, and a
//! chain of arithmetic on the
//! nearest one drifts. The exact samplers in [`crate::random::distributions`]
//! take ratios for exactly that reason, and work in integers all the way down
//! from them.
//!
//! Non-negative, and always in lowest terms: `2/4` and `1/2` are the same value
//! and compare, hash and print alike. The world's ratios are counts over
//! counts, and a signed rational would put a sign check in every operation to
//! buy nothing that is currently needed.
//!
//! Arithmetic is checked rather than wrapping. The operators panic on anything
//! they cannot represent, in release builds as well as debug ones, since a
//! wrong fraction is worse than a stopped program; the `checked_` methods
//! return `None` instead. Where a result cannot be held exactly,
//! [`Ratio::approximate`] gives the closest fraction with a denominator you
//! choose, by continued fractions.

use crate::math::fixed::{Fixed, wide};
use std::cmp::Ordering;
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use std::str::FromStr;

/// A fraction `numerator / denominator`, reduced, with a denominator above
/// zero.
///
/// Reduced at construction by Euclid's algorithm, which is what lets `Eq` and
/// `Hash` be derived: `2/4` and `1/2` become the same pair of integers, so they
/// compare equal and hash alike. Ordering cross-multiplies in 128 bits instead,
/// since two reduced fractions still need a common denominator to be compared.
///
/// The arithmetic is checked rather than wrapping: a product or sum that will
/// not fit in 64 bits gives `None` rather than a silently wrong fraction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Ratio {
    numerator: u64,
    denominator: u64,
}

impl Ratio {
    /// Never: zero over one.
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };

    /// Certainty: one over one. The largest proper ratio.
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    /// A coin: one over two.
    pub const HALF: Self = Self {
        numerator: 1,
        denominator: 2,
    };

    /// `numerator / denominator`, reduced to lowest terms by dividing both by
    /// their greatest common divisor. `None` for a denominator of zero.
    ///
    /// Reducing here rather than on use is what keeps later arithmetic inside
    /// 64 bits for as long as possible, and what makes equality mean equal
    /// value rather than equal spelling.
    pub fn new(numerator: u64, denominator: u64) -> Option<Self> {
        if denominator == 0 {
            return None;
        }

        let divisor: u64 = greatest_common_divisor(numerator, denominator);

        Some(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    /// One in `count`, exactly. `None` when `count` is zero, which is not a
    /// chance.
    ///
    /// The form most of the world's chances are written in: one vein in 400
    /// voxels, one tree in 30 columns. Unlike
    /// [`Probability::one_in`](crate::units::Probability::one_in), nothing is
    /// rounded, so the samplers built on it hit exactly that frequency.
    pub fn one_in(count: u64) -> Option<Self> {
        Self::new(1, count)
    }

    /// `hits` out of `total`, exactly, reduced. `None` when `total` is zero.
    ///
    /// Improper ratios are allowed: more hits than the total gives a value
    /// above one, which is meaningless as a chance but valid as a rate or a
    /// mean, such as the one [`PoissonRatio`](crate::random::PoissonRatio)
    /// takes. Use [`Ratio::is_proper`] where it has to be a chance.
    pub fn out_of(hits: u64, total: u64) -> Option<Self> {
        Self::new(hits, total)
    }

    /// A percentage, exactly: `percent(3)` is `3/100`, not the nearest `f64`
    /// to 0.03. Reduced, so `percent(50)` is `1/2`.
    ///
    /// Never fails, since the denominator is fixed; percentages above 100 give
    /// improper ratios.
    pub fn percent(percent: u64) -> Self {
        Self::new(percent, 100).unwrap_or(Self::ZERO)
    }

    /// The top of the reduced fraction.
    pub const fn numerator(self) -> u64 {
        self.numerator
    }

    /// The bottom of the reduced fraction, always above zero.
    pub const fn denominator(self) -> u64 {
        self.denominator
    }

    /// Whether this is at most 1, which is what it must be to be a chance.
    ///
    /// The check every sampler that takes a chance makes first; the ones that
    /// take a mean or a rate do not.
    pub const fn is_proper(self) -> bool {
        self.numerator <= self.denominator
    }

    /// Whether the fraction is zero, which for a chance means never.
    pub const fn is_zero(self) -> bool {
        self.numerator == 0
    }

    /// The nearest `f64`, for handing to something that wants one.
    ///
    /// This is where the exactness stops: both integers convert exactly only up
    /// to `2^53`, and the division rounds. Do it last, after the arithmetic,
    /// and prefer the samplers that take a [`Ratio`] directly where the
    /// frequency has to be exact.
    pub fn to_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// The chance of this *not* happening, `(denominator - numerator) /
    /// denominator`, or `None` when the ratio is above one and there is nothing
    /// left to take it from.
    ///
    /// Exact, unlike [`Probability::complement`](crate::units::Probability),
    /// which loses the difference for chances very close to either end.
    pub fn complement(self) -> Option<Self> {
        if !self.is_proper() {
            return None;
        }

        Self::new(self.denominator - self.numerator, self.denominator)
    }

    /// Exact product, or `None` if it will not fit in 64 bits.
    ///
    /// Each numerator is reduced against the other's denominator before the
    /// two multiplications, rather than multiplying first and reducing the
    /// result. Both give the same fraction, but cross-reducing keeps far more
    /// pairs inside 64 bits: `(2/3) * (3/2)` never forms `6/6` at all.
    pub fn checked_mul(self, other: Self) -> Option<Self> {
        let first: u64 = greatest_common_divisor(self.numerator, other.denominator);
        let second: u64 = greatest_common_divisor(other.numerator, self.denominator);

        let numerator: u64 = (self.numerator / first).checked_mul(other.numerator / second)?;
        let denominator: u64 =
            (self.denominator / second).checked_mul(other.denominator / first)?;

        Self::new(numerator, denominator)
    }

    /// Exact sum, or `None` if it will not fit in 64 bits.
    ///
    /// Over the least common denominator, formed as
    /// `self.denominator * (other.denominator / gcd)` rather than the product
    /// of the two, which keeps the numbers as small as the fractions allow.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        let divisor: u64 = greatest_common_divisor(self.denominator, other.denominator);
        let denominator: u64 = self.denominator.checked_mul(other.denominator / divisor)?;
        let numerator: u64 = self
            .numerator
            .checked_mul(other.denominator / divisor)?
            .checked_add(other.numerator.checked_mul(self.denominator / divisor)?)?;

        Self::new(numerator, denominator)
    }

    /// Exact difference, or `None` when it would fall below zero or will not
    /// fit in 64 bits.
    ///
    /// Over the least common denominator, as [`Ratio::checked_add`]. A
    /// [`Ratio`] is unsigned, so a larger `other` is an error rather than a
    /// negative result.
    pub fn checked_sub(self, other: Self) -> Option<Self> {
        let divisor: u64 = greatest_common_divisor(self.denominator, other.denominator);
        let denominator: u64 = self.denominator.checked_mul(other.denominator / divisor)?;
        let left: u64 = self.numerator.checked_mul(other.denominator / divisor)?;
        let right: u64 = other.numerator.checked_mul(self.denominator / divisor)?;

        Self::new(left.checked_sub(right)?, denominator)
    }

    /// The fraction turned over, `denominator / numerator`, or `None` for
    /// zero, which has no reciprocal.
    ///
    /// Turns a chance into the average wait for it: one in 400 becomes 400.
    pub fn reciprocal(self) -> Option<Self> {
        Self::new(self.denominator, self.numerator)
    }
}

impl Ratio {
    /// The whole part, rounding towards zero, which for a non-negative value is
    /// also rounding down.
    pub const fn to_integer(self) -> u64 {
        self.numerator / self.denominator
    }

    /// The value as a whole number, or `None` when it has a fractional part.
    pub const fn as_integer(self) -> Option<u64> {
        if self.denominator == 1 {
            Some(self.numerator)
        } else {
            None
        }
    }

    /// Whether the value is a whole number, which for a reduced fraction means
    /// a denominator of one.
    pub const fn is_integer(self) -> bool {
        self.denominator == 1
    }

    /// The largest whole value at or below this one.
    pub const fn floor(self) -> Self {
        Self {
            numerator: self.numerator / self.denominator,
            denominator: 1,
        }
    }

    /// The smallest whole value at or above this one.
    pub const fn ceil(self) -> Self {
        Self {
            numerator: self.numerator.div_ceil(self.denominator),
            denominator: 1,
        }
    }

    /// The nearest whole value, with a half rounded up. `None` when adding the
    /// half overflows.
    pub fn checked_round(self) -> Option<Self> {
        Some(self.checked_add(Self::HALF)?.floor())
    }

    /// What [`Ratio::floor`] leaves behind: the part below one.
    pub const fn fract(self) -> Self {
        Self {
            numerator: self.numerator % self.denominator,
            denominator: self.denominator,
        }
    }

    /// The smaller of the two.
    pub fn min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }

    /// The larger of the two.
    pub fn max(self, other: Self) -> Self {
        if other > self { other } else { self }
    }

    /// Confined to `low..=high`. Panics if the bounds cross.
    pub fn clamp(self, low: Self, high: Self) -> Self {
        assert!(low <= high, "ratio clamp bounds are the wrong way round");

        self.max(low).min(high)
    }

    /// Exact quotient, or `None` for a zero divisor or a result that will not
    /// fit.
    pub fn checked_div(self, other: Self) -> Option<Self> {
        self.checked_mul(other.reciprocal()?)
    }

    /// The remainder of `self / other`: what is left after taking out as many
    /// whole copies of `other` as fit. `None` for a zero divisor or on
    /// overflow.
    pub fn checked_rem(self, other: Self) -> Option<Self> {
        let copies: u64 = self.checked_div(other)?.to_integer();

        self.checked_sub(other.checked_mul(Self::from(copies))?)
    }

    /// This value raised to a whole power, by repeated squaring. `None` on
    /// overflow; a power of zero gives [`Ratio::ONE`] for any value.
    pub fn checked_pow(self, exponent: u32) -> Option<Self> {
        let mut remaining: u32 = exponent;
        let mut base: Self = self;
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

        Some(total)
    }

    /// The fraction halfway between two neighbours in the Stern-Brocot sense:
    /// numerators and denominators added straight across.
    ///
    /// Not the arithmetic mean. What it is good for is that the mediant of two
    /// fractions always lies strictly between them and has the smallest
    /// denominator of anything that does, which is how simple fractions are
    /// found between two bounds. `None` on overflow.
    pub fn mediant(self, other: Self) -> Option<Self> {
        Self::new(
            self.numerator.checked_add(other.numerator)?,
            self.denominator.checked_add(other.denominator)?,
        )
    }

    /// The closest fraction to this one whose denominator is at most
    /// `max_denominator`.
    ///
    /// Walks the continued fraction expansion, keeping the last convergent
    /// whose denominator fits, and then tries the best semiconvergent, which is
    /// what makes the answer the closest rather than merely a close one. A
    /// ratio near pi gives `22/7` at a limit of 50, `355/113` at 200, and the
    /// nearest whole number at 1.
    ///
    /// The value itself is returned when its denominator already fits, and
    /// [`Ratio::ZERO`] when `max_denominator` is zero, which asks for no
    /// fraction at all.
    pub fn approximate(self, max_denominator: u64) -> Self {
        if max_denominator == 0 {
            return Self::ZERO;
        }

        if self.denominator <= max_denominator {
            return self;
        }

        // The last two convergents, as numerator-denominator pairs. They start
        // at the two values the recurrence is seeded with, `1/0` and `0/1`,
        // neither of which is a fraction in its own right.
        let (mut before, mut previous): ((u64, u64), (u64, u64)) = ((0, 1), (1, 0));
        let (mut left, mut right): (u64, u64) = (self.numerator, self.denominator);

        loop {
            let term: u64 = left / right;
            let denominator: u128 = term as u128 * previous.1 as u128 + before.1 as u128;

            if denominator > max_denominator as u128 {
                break;
            }

            let next: (u64, u64) = (term * previous.0 + before.0, denominator as u64);

            before = previous;
            previous = next;

            let remainder: u64 = left % right;

            if remainder == 0 {
                return Self::new(previous.0, previous.1).unwrap_or(Self::ZERO);
            }

            left = right;
            right = remainder;
        }

        let convergent: Self = Self::new(previous.0, previous.1).unwrap_or(Self::ZERO);

        // The semiconvergent: as many further steps towards the convergent that
        // did not fit as the limit allows. Sometimes it is the closer of the
        // two, sometimes not, so both are tried.
        let steps: u64 = (max_denominator - before.1) / previous.1;
        let Some(candidate) =
            Self::new(before.0 + steps * previous.0, before.1 + steps * previous.1)
        else {
            return convergent;
        };

        if self.is_closer_to(candidate, convergent) {
            candidate
        } else {
            convergent
        }
    }

    /// Whether `first` is nearer to this value than `second` is.
    ///
    /// The two distances are `|a d - c b| / (b d)` for each candidate, compared
    /// by cross-multiplying in 256 bits, so nothing is approximated and nothing
    /// overflows on the way.
    fn is_closer_to(self, first: Self, second: Self) -> bool {
        let gap = |other: Self| -> (u128, u128) {
            let left: u128 = self.numerator as u128 * other.denominator as u128;
            let right: u128 = other.numerator as u128 * self.denominator as u128;

            (
                left.abs_diff(right),
                self.denominator as u128 * other.denominator as u128,
            )
        };

        let (first_gap, first_scale): (u128, u128) = gap(first);
        let (second_gap, second_scale): (u128, u128) = gap(second);

        wide::multiply(first_gap, second_scale) < wide::multiply(second_gap, first_scale)
    }

    /// The exact value of an `f64`, which is always a fraction over a power of
    /// two, or `None` when it is negative, not finite, or needs more than 64
    /// bits on either side.
    ///
    /// Exact rather than close: `0.1` gives the enormous fraction the `f64`
    /// really holds, not `1/10`. [`Ratio::from_f64_approximate`] is what gives
    /// `1/10`.
    pub fn from_f64_exact(value: f64) -> Option<Self> {
        if !value.is_finite() || value < 0.0 {
            return None;
        }

        if value == 0.0 {
            return Some(Self::ZERO);
        }

        let bits: u64 = value.to_bits();
        let raw_exponent: i32 = ((bits >> 52) & 0x7FF) as i32;
        let raw_mantissa: u64 = bits & ((1 << 52) - 1);

        // Subnormals carry no implicit leading one and sit one exponent up.
        let (mantissa, exponent): (u64, i32) = if raw_exponent == 0 {
            (raw_mantissa, -1074)
        } else {
            (raw_mantissa | (1 << 52), raw_exponent - 1075)
        };

        if exponent >= 0 {
            return Self::new(mantissa.checked_shl(exponent as u32)?, 1);
        }

        let shift: u32 = (-exponent) as u32;
        let trailing: u32 = mantissa.trailing_zeros().min(shift);

        Self::new(mantissa >> trailing, 1u64.checked_shl(shift - trailing)?)
    }

    /// The closest fraction to an `f64` whose denominator is at most
    /// `max_denominator`, or `None` when the value is negative or not finite.
    ///
    /// Reads the value's continued fraction with the floor and reciprocal of
    /// ordinary arithmetic, which IEEE-754 pins down, so the same `f64` gives
    /// the same fraction everywhere. `0.1` at a limit of 100 gives `1/10`.
    pub fn from_f64_approximate(value: f64, max_denominator: u64) -> Option<Self> {
        if !value.is_finite() || value < 0.0 {
            return None;
        }

        let (mut before, mut previous): ((u64, u64), (u64, u64)) = ((0, 1), (1, 0));
        let mut remaining: f64 = value;

        for _ in 0..64 {
            let term: f64 = remaining.floor();

            if term > u64::MAX as f64 {
                break;
            }

            let whole: u64 = term as u64;
            let Some(denominator) = whole
                .checked_mul(previous.1)
                .and_then(|scaled| scaled.checked_add(before.1))
            else {
                break;
            };

            if denominator > max_denominator.max(1) {
                break;
            }

            let Some(numerator) = whole
                .checked_mul(previous.0)
                .and_then(|scaled| scaled.checked_add(before.0))
            else {
                break;
            };

            before = previous;
            previous = (numerator, denominator);

            let fraction: f64 = remaining - term;

            if fraction == 0.0 {
                break;
            }

            remaining = 1.0 / fraction;
        }

        Self::new(previous.0, previous.1.max(1))
    }

    /// The value as a [`Fixed`], truncated towards zero to its `2^-32` step, or
    /// `None` when it is too large to hold.
    pub fn to_fixed(self) -> Option<Fixed> {
        Fixed::from_ratio(self.numerator as i128, self.denominator as i128)
    }

    /// The exact value of a [`Fixed`], which is a fraction over `2^32`, or
    /// `None` when it is negative or its numerator needs more than 64 bits.
    pub fn from_fixed(value: Fixed) -> Option<Self> {
        let bits: i128 = value.to_bits();

        if bits < 0 {
            return None;
        }

        let trailing: u32 = (bits as u128)
            .trailing_zeros()
            .min(crate::math::fixed::FRACTION_BITS);

        Self::new(
            u64::try_from(bits >> trailing).ok()?,
            1u64 << (crate::math::fixed::FRACTION_BITS - trailing),
        )
    }
}

/// Compared exactly, by cross-multiplying in 128 bits so that no pair of
/// 64-bit fractions can overflow the comparison.
impl Ord for Ratio {
    fn cmp(&self, other: &Self) -> Ordering {
        let left: u128 = self.numerator as u128 * other.denominator as u128;
        let right: u128 = other.numerator as u128 * self.denominator as u128;

        left.cmp(&right)
    }
}

impl PartialOrd for Ratio {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Default for Ratio {
    fn default() -> Self {
        Self::ZERO
    }
}

/// As the reduced fraction, such as `3/7`, and as a plain number when the
/// denominator is one. What [`FromStr`] reads back.
impl fmt::Display for Ratio {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            return write!(formatter, "{}", self.numerator);
        }

        write!(formatter, "{}/{}", self.numerator, self.denominator)
    }
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// Exact, and panics rather than wrapping when the sum will not fit; see
/// [`Ratio::checked_add`].
impl Add for Ratio {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.checked_add(other).expect("ratio overflow in add")
    }
}

/// Exact, and panics when the difference would fall below zero or will not fit;
/// see [`Ratio::checked_sub`].
impl Sub for Ratio {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.checked_sub(other).expect("ratio overflow in sub")
    }
}

/// Exact, and panics when the product will not fit; see
/// [`Ratio::checked_mul`].
impl Mul for Ratio {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.checked_mul(other).expect("ratio overflow in mul")
    }
}

/// Exact, and panics on a zero divisor or when the quotient will not fit; see
/// [`Ratio::checked_div`].
impl Div for Ratio {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert!(!other.is_zero(), "ratio division by zero");

        self.checked_div(other).expect("ratio overflow in div")
    }
}

/// What is left after taking out as many whole copies of the divisor as fit.
/// Panics on a zero divisor; see [`Ratio::checked_rem`].
impl Rem for Ratio {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        assert!(!other.is_zero(), "ratio remainder by zero");

        self.checked_rem(other).expect("ratio overflow in rem")
    }
}

macro_rules! implement_assign {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl $trait for Ratio {
            fn $method(&mut self, other: Self) {
                *self = *self $operator other;
            }
        }
    };
}

implement_assign!(AddAssign, add_assign, +);
implement_assign!(SubAssign, sub_assign, -);
implement_assign!(MulAssign, mul_assign, *);
implement_assign!(DivAssign, div_assign, /);
implement_assign!(RemAssign, rem_assign, %);

/// Adds a sequence up, panicking on overflow as [`Add`] does.
impl Sum for Ratio {
    fn sum<I: Iterator<Item = Self>>(values: I) -> Self {
        values.fold(Self::ZERO, |total, value| total + value)
    }
}

/// Multiplies a sequence together, panicking on overflow as [`Mul`] does.
impl Product for Ratio {
    fn product<I: Iterator<Item = Self>>(values: I) -> Self {
        values.fold(Self::ONE, |total, value| total * value)
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// What went wrong while reading a [`Ratio`] from text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseRatioError {
    /// The text held no digits at all.
    Empty,
    /// The text held something that is not a fraction, a whole number or a
    /// decimal.
    Invalid,
    /// The denominator was zero, which names no fraction.
    ZeroDenominator,
    /// The value needs more than 64 bits on one side.
    OutOfRange,
}

impl fmt::Display for ParseRatioError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "no digits to read",
            Self::Invalid => "not a ratio",
            Self::ZeroDenominator => "a ratio cannot be over zero",
            Self::OutOfRange => "outside the range a ratio can hold",
        })
    }
}

impl std::error::Error for ParseRatioError {}

/// Reads `3/7`, a whole number such as `5`, or a decimal such as `0.25`, which
/// becomes the exact fraction `1/4` rather than the nearest `f64` to it.
///
/// Surrounding whitespace is ignored, as is whitespace around the slash. A
/// decimal with more digits than 64 bits can hold over its power of ten is out
/// of range rather than rounded, since rounding is what this type exists to
/// avoid; [`Ratio::from_f64_approximate`] is the way to ask for a close
/// fraction instead.
impl FromStr for Ratio {
    type Err = ParseRatioError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text: &str = text.trim();

        if text.is_empty() {
            return Err(ParseRatioError::Empty);
        }

        if let Some((numerator, denominator)) = text.split_once('/') {
            let numerator: u64 = numerator
                .trim()
                .parse()
                .map_err(|_| ParseRatioError::Invalid)?;
            let denominator: u64 = denominator
                .trim()
                .parse()
                .map_err(|_| ParseRatioError::Invalid)?;

            return Self::new(numerator, denominator).ok_or(ParseRatioError::ZeroDenominator);
        }

        let Some((whole, fraction)) = text.split_once('.') else {
            return text
                .parse::<u64>()
                .map(Self::from)
                .map_err(|_| ParseRatioError::Invalid);
        };

        let whole: u64 = if whole.is_empty() {
            0
        } else {
            whole.parse().map_err(|_| ParseRatioError::Invalid)?
        };

        if fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(ParseRatioError::Invalid);
        }

        let digits: u64 = fraction.parse().map_err(|_| ParseRatioError::OutOfRange)?;
        let denominator: u64 = 10u64
            .checked_pow(fraction.len() as u32)
            .ok_or(ParseRatioError::OutOfRange)?;
        let numerator: u64 = whole
            .checked_mul(denominator)
            .and_then(|scaled| scaled.checked_add(digits))
            .ok_or(ParseRatioError::OutOfRange)?;

        Self::new(numerator, denominator).ok_or(ParseRatioError::ZeroDenominator)
    }
}

impl From<u64> for Ratio {
    /// A whole number, over one.
    fn from(whole: u64) -> Self {
        Self {
            numerator: whole,
            denominator: 1,
        }
    }
}

impl From<Ratio> for f64 {
    /// The nearest `f64`; see [`Ratio::to_f64`].
    fn from(ratio: Ratio) -> Self {
        ratio.to_f64()
    }
}

/// The greatest common divisor, by Euclid's algorithm: replace the pair with
/// the smaller number and the remainder until the remainder is zero.
///
/// Returns `1` rather than `0` for a pair that is entirely zero, so that
/// reducing by the result is always safe.
const fn greatest_common_divisor(first: u64, second: u64) -> u64 {
    let mut left: u64 = first;
    let mut right: u64 = second;

    while right != 0 {
        let remainder: u64 = left % right;

        left = right;
        right = remainder;
    }

    if left == 0 { 1 } else { left }
}
