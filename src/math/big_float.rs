//! A floating-point number that grows to stay exact.
//!
//! [`BigFloat`] holds `±significand × 2^exponent` with a [`BigUint`] significand
//! and a [`BigInt`] exponent, both of which lengthen as the value demands. There is
//! no precision to choose.
//!
//! # The property that makes this type different
//!
//! **Addition, subtraction and multiplication are exact.** Not "exact to 256 bits"
//! — exact. The significand grows to hold whatever the answer needs, so nothing is
//! ever rounded away.
//!
//! That is a stronger guarantee than any fixed-precision float can make, and it has
//! a consequence worth stating plainly: the ring laws hold **exactly** here.
//!
//! ```text
//! (a + b) + c == a + (b + c)      // true for every input
//! a × (b + c) == a×b + a×c        // true for every input
//! ```
//!
//! For `f64`, [`Fixed`](crate::math::Fixed) and
//! [`WideFloat`](crate::math::WideFloat) those are true only up to a rounding.
//! Here they are simply true, and the tests assert it.
//!
//! # What that costs, and the pressure valve
//!
//! Exactness is not free, and the cost is **growth**. Two facts to keep in mind:
//!
//! - **Multiplying doubles the significand.** Squaring repeatedly is the trap: ten
//!   squarings multiply the length by about a thousand, twenty by about a million.
//! - **Adding across a large exponent gap allocates the gap.** `1 + 2^-1000000`
//!   is exact, and holding it takes a million bits, because that is genuinely how
//!   many bits the answer has.
//!
//! Neither is a defect; both are what "exact" means. When growth is not wanted,
//! [`BigFloat::rounded_to`] trims a value to a chosen number of significant bits,
//! and is the thing to call inside a loop that would otherwise grow without bound.
//! [`BigFloat::significand_bits`] says how large a value has become.
//!
//! Operations refuse rather than exhaust memory: anything that would need more than
//! [`MAX_SIGNIFICAND_BITS`] returns `None` from a `checked_` method.
//!
//! # Division, and why there is no `Field`
//!
//! Division cannot be exact — `1/3` has no finite binary expansion at any size — so
//! there is no honest total division and **[`Field`](crate::math::traits::Field) is
//! deliberately not implemented**. Instead there are two operations that each say
//! what they do:
//!
//! - [`BigFloat::checked_div_exact`] divides when the result is exactly
//!   representable and returns `None` when it is not. `1/4` succeeds; `1/3` does
//!   not. The test is simple: with both significands odd, the quotient terminates
//!   exactly when one divides the other.
//! - [`BigFloat::div_rounded`] takes the precision to round to, as an argument.
//!
//! A `Field` impl would have to invent a precision and hide it inside an operator.
//! That is the kind of silent decision this module avoids — the same reasoning that
//! keeps [`Interval`](crate::math::Interval) out of the ring hierarchy.
//!
//! # Against the fixed-width float
//!
//! | | [`WideFloat<S, E>`](crate::math::WideFloat) | [`BigFloat`] |
//! |---|---|---|
//! | Precision | chosen at compile time | grows; exact for `+ - ×` |
//! | `+ - ×` | rounded | exact |
//! | `/` | rounds, always succeeds | exact-or-`None`, or rounds on request |
//! | Ring laws | up to a rounding | exactly |
//! | `Field` | yes | no, on purpose |
//! | Storage | array, `Copy` | two `Vec`s, `Clone` |
//! | Cost | fixed and predictable | grows with the value |
//!
//! # Representation
//!
//! Canonical, so [`PartialEq`] and [`Ord`] can be trusted:
//!
//! - The significand is **odd**. Every factor of two is pushed out into the
//!   exponent, which makes the representation unique — `2 × 2^0` and `1 × 2^1` are
//!   the same number and are stored the same way.
//! - Zero is a zero significand with a zero exponent, and is never negative.
//!
//! Note this is the opposite convention from
//! [`WideFloat`](crate::math::WideFloat), which normalises the significand's *top*
//! bit to a fixed position. That one wants a constant width; this one wants a
//! minimal one.

use crate::math::big_integer::{BigInt, BigUint};
use crate::math::traits::{CommutativeRing, One, Ring, Semiring, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The most bits a significand is allowed to reach before an operation refuses.
///
/// A gigabit, which is 128 MB of limbs. Past this, an exact answer is almost
/// certainly not what was wanted, and allocating it would be worse than saying so.
/// [`BigFloat::rounded_to`] is how to stay below it deliberately.
pub const MAX_SIGNIFICAND_BITS: u64 = 1 << 30;

/// How many bits the exact decimal conversion will build before printing the
/// hexadecimal form instead.
///
/// `significand × 2^exponent` in decimal needs about `bits + 2.33 × |exponent|`
/// bits of intermediate. 64 Ki bits is some twenty thousand digits — past anything
/// anyone reads.
const MAX_DECIMAL_BITS: u64 = 64 * 1024;

/// A floating-point number whose significand and exponent both grow as needed.
///
/// # Example
///
/// ```
/// use voxel_world::math::BigFloat;
///
/// // A tenth, a million times over. Exact — no drift at all, unlike an f64.
/// let tenth = BigFloat::from_integer(1)
///     .checked_div_exact(&BigFloat::from_integer(10));
/// assert_eq!(tenth, None, "a tenth is not exact in binary, and it says so");
///
/// // A quarter is, though, because four is a power of two.
/// let quarter = BigFloat::from_integer(1)
///     .checked_div_exact(&BigFloat::from_integer(4))
///     .expect("exact");
/// assert_eq!(quarter.to_string(), "0.25");
///
/// // Multiplication never rounds, however far it grows.
/// let mut value = BigFloat::from_integer(3);
/// for _ in 0..10 {
///     value = value.clone() * value;
/// }
/// // 3^1024, exactly.
/// assert_eq!(value.significand_bits(), 1624);
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct BigFloat {
    /// Always `false` when the significand is zero, so there is no negative zero.
    negative: bool,
    /// Always **odd** unless the value is zero.
    significand: BigUint,
    /// The power of two the significand is multiplied by.
    exponent: BigInt,
}

impl BigFloat {
    /// Zero.
    pub fn zero() -> Self {
        Self {
            negative: false,
            significand: BigUint::zero(),
            exponent: BigInt::zero(),
        }
    }

    /// One.
    pub fn one() -> Self {
        Self {
            negative: false,
            significand: BigUint::one(),
            exponent: BigInt::zero(),
        }
    }

    /// Whether the value is zero.
    pub fn is_zero(&self) -> bool {
        self.significand.is_zero()
    }

    /// Whether the value is one.
    pub fn is_one(&self) -> bool {
        !self.negative && self.significand.is_one() && self.exponent.is_zero()
    }

    /// Whether the value is below zero. Never true for zero.
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// The significand, which is odd unless the value is zero.
    pub fn significand(&self) -> &BigUint {
        &self.significand
    }

    /// The power of two the significand is scaled by.
    pub fn exponent(&self) -> &BigInt {
        &self.exponent
    }

    /// How many bits the significand currently occupies.
    ///
    /// The number to watch in a loop: exact arithmetic grows this, and
    /// [`BigFloat::rounded_to`] is how to bring it back down.
    pub fn significand_bits(&self) -> u64 {
        self.significand.bit_length()
    }

    /// The value with its sign removed.
    pub fn abs(&self) -> Self {
        Self {
            negative: false,
            significand: self.significand.clone(),
            exponent: self.exponent.clone(),
        }
    }

    /// The negation. Total — zero stays zero, and there is no `MIN` to fail on.
    pub fn negated(&self) -> Self {
        Self {
            negative: !self.negative && !self.is_zero(),
            significand: self.significand.clone(),
            exponent: self.exponent.clone(),
        }
    }

    /// A value from its parts, normalised so the significand is odd.
    ///
    /// The one place the invariant is established. Pushing the factors of two out
    /// into the exponent is what makes the representation unique, so that
    /// `2 × 2^0` and `1 × 2^1` are not two different spellings of two.
    fn normalised(negative: bool, significand: BigUint, exponent: BigInt) -> Self {
        if significand.is_zero() {
            return Self::zero();
        }

        let shift: u64 = significand.trailing_zeros();

        Self {
            negative,
            significand: significand.shifted_down(shift),
            exponent: exponent.sum(&BigInt::from(shift as i64)),
        }
    }

    /// A whole number.
    pub fn from_integer(value: i64) -> Self {
        Self::normalised(
            value < 0,
            BigUint::from(value.unsigned_abs()),
            BigInt::zero(),
        )
    }

    /// A value from a growable integer.
    pub fn from_big_int(value: &BigInt) -> Self {
        Self::normalised(
            value.is_negative(),
            value.magnitude().clone(),
            BigInt::zero(),
        )
    }

    /// The exact value of an `f64`.
    ///
    /// `None` for a NaN or an infinity. Every finite `f64` is held exactly, since
    /// this type has room for any significand.
    pub fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }

        if value == 0.0 {
            return Some(Self::zero());
        }

        let bits: u64 = value.to_bits();
        let negative: bool = bits >> 63 == 1;
        let raw_exponent: i64 = ((bits >> 52) & 0x7FF) as i64;
        let fraction: u64 = bits & ((1 << 52) - 1);

        // A subnormal has no implicit leading one and a fixed exponent.
        let (significand, exponent) = if raw_exponent == 0 {
            (fraction, -1074_i64)
        } else {
            (fraction | 1 << 52, raw_exponent - 1075)
        };

        Some(Self::normalised(
            negative,
            BigUint::from(significand),
            BigInt::from(exponent),
        ))
    }

    /// The nearest `f64`, saturating to an infinity or to zero beyond its range.
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }

        let bits: u64 = self.significand_bits();

        // The top 64 bits, with everything below folded into the low bit so a
        // halfway value is not mistaken for an exact tie.
        let (leading, dropped): (u64, u64) = if bits <= 64 {
            (
                self.significand.limbs().first().copied().unwrap_or(0),
                0,
            )
        } else {
            let shift: u64 = bits - 64;
            let top: BigUint = self.significand.shifted_down(shift);
            let sticky: bool = self.significand.any_low_bit_set(shift);

            (
                top.limbs().first().copied().unwrap_or(0) | u64::from(sticky),
                shift,
            )
        };

        // `u64 as f64` rounds to nearest, which is the rounding wanted.
        let mut value: f64 = leading as f64;

        if self.negative {
            value = -value;
        }

        let Some(exponent) = self
            .exponent
            .sum(&BigInt::from(dropped as i64))
            .to_i128()
        else {
            // Past `i128`, so certainly past an f64 in whichever direction.
            return if self.exponent.is_negative() {
                0.0
            } else if self.negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        };

        // Scale in bounded steps, so a large exponent overflows to infinity or
        // underflows to zero rather than mis-scaling along the way.
        let mut remaining: i128 = exponent;

        while remaining > 0 {
            let step: i128 = remaining.min(512);
            value *= f64::from_bits((1023 + step as u64) << 52);
            remaining -= step;
        }

        while remaining < 0 {
            let step: i128 = (-remaining).min(512);
            value /= f64::from_bits((1023 + step as u64) << 52);
            remaining += step;
        }

        value
    }

    /// The exact sum, or `None` when it would need more than
    /// [`MAX_SIGNIFICAND_BITS`].
    ///
    /// Exact, always. Both significands are shifted up to a common exponent — the
    /// lower of the two — which loses nothing because the significand can grow. The
    /// cost is that a large exponent gap means a large significand: the answer
    /// genuinely has that many bits.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        if other.is_zero() {
            return Some(self.clone());
        }

        if self.is_zero() {
            return Some(other.clone());
        }

        // Align to the lower exponent, so both shifts are upwards and exact.
        let (base, gap_left, gap_right) = match self.exponent.cmp(&other.exponent) {
            Ordering::Equal => (self.exponent.clone(), 0, 0),
            Ordering::Less => {
                let gap: u64 = other.exponent.difference(&self.exponent).to_i128()?.try_into().ok()?;
                (self.exponent.clone(), 0, gap)
            }
            Ordering::Greater => {
                let gap: u64 = self.exponent.difference(&other.exponent).to_i128()?.try_into().ok()?;
                (other.exponent.clone(), gap, 0)
            }
        };

        // Refuse rather than try to allocate an answer this large.
        let reach: u64 = self
            .significand_bits()
            .max(other.significand_bits())
            .saturating_add(gap_left.max(gap_right));

        if reach > MAX_SIGNIFICAND_BITS {
            return None;
        }

        let left: BigUint = self.significand.shifted_up(gap_left);
        let right: BigUint = other.significand.shifted_up(gap_right);

        if self.negative == other.negative {
            return Some(Self::normalised(self.negative, left.sum(&right), base));
        }

        // Opposite signs: the smaller magnitude comes off the larger, exactly.
        Some(match left.cmp(&right) {
            Ordering::Equal => Self::zero(),
            Ordering::Greater => Self::normalised(
                self.negative,
                left.checked_sub(&right).expect("the larger was established"),
                base,
            ),
            Ordering::Less => Self::normalised(
                other.negative,
                right.checked_sub(&left).expect("the larger was established"),
                base,
            ),
        })
    }

    /// The exact difference, or `None` when it would grow past the limit.
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        self.checked_add(&other.negated())
    }

    /// The exact product, or `None` when it would grow past the limit.
    ///
    /// The significands multiply and the exponents add. No normalisation is needed
    /// afterwards: the product of two odd numbers is odd, so the canonical form is
    /// already in hand.
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        if self.is_zero() || other.is_zero() {
            return Some(Self::zero());
        }

        if self
            .significand_bits()
            .saturating_add(other.significand_bits())
            > MAX_SIGNIFICAND_BITS
        {
            return None;
        }

        Some(Self {
            negative: self.negative != other.negative,
            significand: self.significand.product(&other.significand),
            exponent: self.exponent.sum(&other.exponent),
        })
    }

    /// The quotient when it is exactly representable, `None` when it is not.
    ///
    /// # How the test works
    ///
    /// Both significands are odd, so no power of two can help the division along.
    /// `sa × 2^ea / (sb × 2^eb)` therefore terminates in binary exactly when `sb`
    /// divides `sa` — one remainder check, no searching.
    ///
    /// So `1/4` succeeds, `3/6` succeeds (both reduce to odd 1 and 1), and `1/3`
    /// returns `None` rather than silently rounding. Dividing by zero is also
    /// `None`.
    pub fn checked_div_exact(&self, other: &Self) -> Option<Self> {
        if other.is_zero() {
            return None;
        }

        if self.is_zero() {
            return Some(Self::zero());
        }

        let (quotient, remainder) = self.significand.div_rem(&other.significand)?;

        // A remainder means the expansion does not terminate, at any precision.
        if !remainder.is_zero() {
            return None;
        }

        Some(Self::normalised(
            self.negative != other.negative,
            quotient,
            self.exponent.difference(&other.exponent),
        ))
    }

    /// The quotient rounded to `precision` significant bits, or `None` for a zero
    /// divisor or a zero precision.
    ///
    /// Rounds to nearest, ties to even, as [`WideFloat`](crate::math::WideFloat)
    /// does. The precision is an argument because there is no right default: this is
    /// the one operation where a number has to be chosen, and choosing it silently
    /// is what this type refuses to do.
    pub fn div_rounded(&self, other: &Self, precision: u64) -> Option<Self> {
        if other.is_zero() || precision == 0 {
            return None;
        }

        if self.is_zero() {
            return Some(Self::zero());
        }

        let numerator_bits: i128 = self.significand_bits() as i128;
        let divisor_bits: i128 = other.significand_bits() as i128;

        // Scale so the quotient lands with `precision + 1` or `+ 2` bits: enough to
        // round from, and no more work than that. Whichever side needs shifting is
        // shifted, so nothing is ever shifted *down* and lost.
        let wanted: i128 = precision as i128 + 1 - (numerator_bits - divisor_bits);
        let (numerator, divisor, shift): (BigUint, BigUint, i128) = if wanted >= 0 {
            (
                self.significand.shifted_up(wanted as u64),
                other.significand.clone(),
                -wanted,
            )
        } else {
            (
                self.significand.clone(),
                other.significand.shifted_up((-wanted) as u64),
                -wanted,
            )
        };

        let (quotient, remainder) = numerator.div_rem(&divisor)?;

        // Trim the surplus bits, using the division's remainder as the sticky bit.
        let surplus: u64 = quotient.bit_length().saturating_sub(precision);
        let negative: bool = self.negative != other.negative;
        let exponent: BigInt = self
            .exponent
            .difference(&other.exponent)
            .sum(&BigInt::from(shift as i64));

        // The scaling above puts the quotient at `precision + 1` or `+ 2` bits, so
        // there is always at least one surplus bit to round from. If that ever
        // stopped being true the rounding below would read a bit that does not
        // exist, so it is worth asserting rather than assuming.
        debug_assert!(
            surplus > 0,
            "the scaling should leave at least one bit to round with"
        );

        if surplus == 0 {
            return Some(Self::normalised(negative, quotient, exponent));
        }

        let half: bool = quotient.bit(surplus - 1);
        let sticky: bool = quotient.any_low_bit_set(surplus - 1) || !remainder.is_zero();
        let mut trimmed: BigUint = quotient.shifted_down(surplus);

        // Round to nearest, ties to even.
        if half && (sticky || trimmed.bit(0)) {
            trimmed = trimmed.sum(&BigUint::one());
        }

        Some(Self::normalised(
            negative,
            trimmed,
            exponent.sum(&BigInt::from(surplus as i64)),
        ))
    }

    /// This value rounded to at most `precision` significant bits.
    ///
    /// The pressure valve against unbounded growth: exact arithmetic lengthens the
    /// significand, and this is how to put it back. Call it inside any loop that
    /// multiplies repeatedly, or the significand will double each time.
    ///
    /// Rounds to nearest, ties to even. A value already that short is returned
    /// unchanged, so this never *adds* precision.
    pub fn rounded_to(&self, precision: u64) -> Self {
        if precision == 0 || self.is_zero() {
            return self.clone();
        }

        let surplus: u64 = self.significand_bits().saturating_sub(precision);

        if surplus == 0 {
            return self.clone();
        }

        let half: bool = self.significand.bit(surplus - 1);
        let sticky: bool = self.significand.any_low_bit_set(surplus - 1);
        let mut trimmed: BigUint = self.significand.shifted_down(surplus);

        if half && (sticky || trimmed.bit(0)) {
            trimmed = trimmed.sum(&BigUint::one());
        }

        Self::normalised(
            self.negative,
            trimmed,
            self.exponent.sum(&BigInt::from(surplus as i64)),
        )
    }

    /// The two magnitudes compared, ignoring both signs.
    ///
    /// Unlike [`WideFloat`](crate::math::WideFloat), the exponents cannot simply be
    /// compared first: the significands are minimal rather than a fixed width, so
    /// `3 × 2^0` is larger than `1 × 2^1` despite the smaller exponent. What decides
    /// is the position of the top bit, which is the exponent plus the significand's
    /// length.
    fn compare_magnitude(&self, other: &Self) -> Ordering {
        match (self.is_zero(), other.is_zero()) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            (false, false) => {}
        }

        let left_top: BigInt = self
            .exponent
            .sum(&BigInt::from(self.significand_bits() as i64));
        let right_top: BigInt = other
            .exponent
            .sum(&BigInt::from(other.significand_bits() as i64));

        match left_top.cmp(&right_top) {
            Ordering::Equal => {}
            decided => return decided,
        }

        // The same leading position, so align the significands and compare them.
        // Shifting the shorter one up is exact and cannot change the order.
        match self.significand_bits().cmp(&other.significand_bits()) {
            Ordering::Equal => self.significand.cmp(&other.significand),
            Ordering::Less => {
                let gap: u64 = other.significand_bits() - self.significand_bits();
                self.significand.shifted_up(gap).cmp(&other.significand)
            }
            Ordering::Greater => {
                let gap: u64 = self.significand_bits() - other.significand_bits();
                self.significand.cmp(&other.significand.shifted_up(gap))
            }
        }
    }
}

impl Ord for BigFloat {
    /// A total order: there is no NaN to make it partial, and no negative zero to
    /// make two distinct values compare equal.
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            // Among negatives the order reverses.
            (true, true) => other.compare_magnitude(self),
            (false, false) => self.compare_magnitude(other),
        }
    }
}

impl PartialOrd for BigFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<i64> for BigFloat {
    fn from(value: i64) -> Self {
        Self::from_integer(value)
    }
}

impl From<BigInt> for BigFloat {
    fn from(value: BigInt) -> Self {
        Self::from_big_int(&value)
    }
}

/// Operators panic where the checked methods return `None` — that is, only when the
/// answer would need more than [`MAX_SIGNIFICAND_BITS`].
impl Add for BigFloat {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.checked_add(&other)
            .expect("the exact sum would exceed MAX_SIGNIFICAND_BITS")
    }
}

impl Sub for BigFloat {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.checked_sub(&other)
            .expect("the exact difference would exceed MAX_SIGNIFICAND_BITS")
    }
}

impl Mul for BigFloat {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.checked_mul(&other)
            .expect("the exact product would exceed MAX_SIGNIFICAND_BITS")
    }
}

impl Neg for BigFloat {
    type Output = Self;

    fn neg(self) -> Self {
        self.negated()
    }
}

impl AddAssign for BigFloat {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl SubAssign for BigFloat {
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl MulAssign for BigFloat {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl Zero for BigFloat {
    fn zero() -> Self {
        BigFloat::zero()
    }

    fn is_zero(&self) -> bool {
        BigFloat::is_zero(self)
    }
}

impl One for BigFloat {
    fn one() -> Self {
        BigFloat::one()
    }

    fn is_one(&self) -> bool {
        BigFloat::is_one(self)
    }
}

impl Semiring for BigFloat {}
impl Ring for BigFloat {}

/// A commutative ring whose laws hold **exactly**, not up to a rounding.
///
/// This is the one floating-point type here that can say that. `f64`,
/// [`Fixed`](crate::math::Fixed) and [`WideFloat`](crate::math::WideFloat) all
/// satisfy associativity and distributivity only to within their last bit, because
/// each operation rounds. Nothing rounds here, so the laws are simply true.
///
/// [`Field`](crate::math::traits::Field) is **not** implemented, and that is the
/// price: a field needs a total division, and division cannot be exact. See
/// [`BigFloat::checked_div_exact`] and [`BigFloat::div_rounded`].
impl CommutativeRing for BigFloat {}

impl fmt::Display for BigFloat {
    /// The exact decimal value, which is always finite because two divides ten.
    ///
    /// A value whose exponent makes the decimal form unreasonably long prints as
    /// hexadecimal instead, which is exact at any size.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return formatter.write_str("0");
        }

        let Some(exponent) = self.exponent.to_i128() else {
            return write!(formatter, "{self:x}");
        };

        let width: u128 = u128::from(self.significand_bits())
            + if exponent >= 0 {
                exponent.unsigned_abs()
            } else {
                // log2(5) is a little under 2.33.
                exponent.unsigned_abs() * 233 / 100
            };

        if width > u128::from(MAX_DECIMAL_BITS) {
            return write!(formatter, "{self:x}");
        }

        if self.negative {
            formatter.write_str("-")?;
        }

        formatter.write_str(&decimal_digits(&self.significand, exponent as i64))
    }
}

impl fmt::LowerHex for BigFloat {
    /// `-0x<significand>p<exponent>`, meaning `significand × 2^exponent`.
    ///
    /// Exact at every size, which is why it is the fallback when a decimal expansion
    /// would be unreasonable.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return formatter.write_str("0x0p0");
        }

        if self.negative {
            formatter.write_str("-")?;
        }

        write!(formatter, "0x{:x}p{}", self.significand, self.exponent)
    }
}

impl fmt::Debug for BigFloat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self}")
    }
}

/// The exact decimal text of `significand × 2^exponent`, without a sign.
///
/// For a positive exponent the value is an integer, `significand << exponent`. For a
/// negative one, write `s = -exponent` and use
///
/// ```text
/// significand × 2^-s = significand × 5^s × 10^-s
/// ```
///
/// which is the integer `significand × 5^s` with the decimal point moved `s` places
/// left. Every binary fraction terminates in decimal, so there is never a repeating
/// tail to cut off.
fn decimal_digits(significand: &BigUint, exponent: i64) -> String {
    let mut value: BigUint = significand.clone();
    let mut point: usize = 0;

    if exponent >= 0 {
        value = value.shifted_up(exponent as u64);
    } else {
        let places: u64 = exponent.unsigned_abs();
        point = places as usize;

        // 5^27 is the largest power of five below 2^64, so it is the widest step a
        // single-limb multiply can take.
        const CHUNK: u32 = 27;
        let step: BigUint = BigUint::from(5u64.pow(CHUNK));

        let mut remaining: u64 = places;

        while remaining >= u64::from(CHUNK) {
            value = value.product(&step);
            remaining -= u64::from(CHUNK);
        }

        if remaining > 0 {
            value = value.product(&BigUint::from(5u64.pow(remaining as u32)));
        }
    }

    let digits: String = value.to_string();

    if point == 0 {
        return digits;
    }

    // Place the point `point` digits from the right, padding when the integer part
    // is empty.
    let padded: String = if digits.len() <= point {
        format!("{}{}", "0".repeat(point - digits.len() + 1), digits)
    } else {
        digits
    };

    let split: usize = padded.len() - point;
    let whole: &str = &padded[..split];
    let fraction: &str = padded[split..].trim_end_matches('0');

    if fraction.is_empty() {
        whole.to_string()
    } else {
        format!("{whole}.{fraction}")
    }
}
