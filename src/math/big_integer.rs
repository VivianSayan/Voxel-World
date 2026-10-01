//! Integers that grow to fit, however large they get.
//!
//! [`BigUint`] is a non-negative integer and [`BigInt`] a signed one, each holding
//! its digits in a [`Vec`] that lengthens as the value does. There is no width to
//! choose and no overflow to handle: `BigInt` arithmetic is exact for every input
//! it is given, and the only limit is memory.
//!
//! # Against the fixed-width types
//!
//! [`WideUint`](crate::math::WideUint) and [`WideInt`](crate::math::WideInt) hold
//! the same kind of value in an array whose length is fixed at compile time. Which
//! to use is a real decision, not a matter of taste:
//!
//! | | [`WideUint<N>`](crate::math::WideUint) | [`BigUint`] |
//! |---|---|---|
//! | Width | fixed at compile time | grows at run time |
//! | Storage | array, on the stack | `Vec`, on the heap |
//! | Copying | `Copy`, a memcpy | `Clone`, an allocation |
//! | Overflow | wraps, or refuses | cannot happen |
//! | Size of a value | always the same | depends on the value |
//!
//! Reach for the fixed ones when the size is known and the cost has to be
//! predictable — a hash, a 256-bit coordinate, anything in a hot loop or a struct
//! that gets copied. Reach for these when the size is *not* known: a factorial, an
//! accumulating product, a parsed number of unknown length.
//!
//! The rule of thumb: if you find yourself picking a width large enough that it
//! surely will not overflow, you wanted [`BigUint`].
//!
//! # Representation
//!
//! Digits are `u64` limbs, least significant first — the same order as the
//! fixed-width types, so the two can be converted without reversing anything.
//!
//! The representation is **canonical**: there are never trailing zero limbs, so
//! zero is the empty vector and every value has exactly one spelling. That is what
//! lets [`PartialEq`] be derived and [`Ord`] compare lengths before digits.
//!
//! # Sign, and why it is not two's complement
//!
//! [`BigInt`] is a sign and a magnitude, where
//! [`WideInt`](crate::math::WideInt) is two's complement. That is forced rather
//! than chosen: complementing needs a width to complement *against*, and a growable
//! integer has none. Sign-magnitude costs a branch on each addition and gives
//! something back: there is no asymmetric `MIN` whose negation does not fit, so
//! [`BigInt::negated`] is total where the fixed-width types need a `checked_neg`
//! that can fail.

use crate::math::traits::{CommutativeRing, EuclideanRing, One, Ring, Semiring, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Shl, Shr, Sub, SubAssign};
use std::str::FromStr;

/// How many bits one limb holds.
const LIMB_BITS: u32 = u64::BITS;

// ===========================================================================
// Unsigned
// ===========================================================================

/// A non-negative integer of any size.
///
/// # Example
///
/// ```
/// use voxel_world::math::BigUint;
///
/// // 100 factorial, which is 158 digits and fits in no primitive anywhere.
/// let mut factorial = BigUint::one();
/// for n in 1..=100u64 {
///     factorial = factorial * BigUint::from(n);
/// }
///
/// assert_eq!(factorial.to_string().len(), 158);
/// assert!(factorial.to_string().starts_with("93326215443944152681699238856266700"));
///
/// // And it divides back down exactly.
/// for n in 1..=100u64 {
///     let (quotient, remainder) = factorial.div_rem(&BigUint::from(n)).unwrap();
///     assert!(remainder.is_zero());
///     factorial = quotient;
/// }
/// assert!(factorial.is_one());
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct BigUint {
    /// Least significant limb first, with no trailing zeros. Zero is empty.
    limbs: Vec<u64>,
}

impl BigUint {
    /// Zero.
    pub fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    /// One.
    pub fn one() -> Self {
        Self { limbs: vec![1] }
    }

    /// A value from limbs, least significant first. Trailing zeros are dropped.
    pub fn from_limbs(limbs: Vec<u64>) -> Self {
        let mut value = Self { limbs };
        value.trim();
        value
    }

    /// The limbs, least significant first, with no trailing zeros.
    pub fn limbs(&self) -> &[u64] {
        &self.limbs
    }

    /// Whether the value is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Whether the value is one.
    pub fn is_one(&self) -> bool {
        self.limbs.len() == 1 && self.limbs[0] == 1
    }

    /// How many bits the value needs, which is zero for zero.
    pub fn bit_length(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(top) => {
                (self.limbs.len() as u64 - 1) * u64::from(LIMB_BITS)
                    + u64::from(LIMB_BITS - top.leading_zeros())
            }
        }
    }

    /// How many zero bits sit below the lowest one bit, or zero for zero.
    ///
    /// What [`BigFloat`](crate::math::BigFloat) uses to push factors of two out of
    /// its significand and into its exponent.
    pub fn trailing_zeros(&self) -> u64 {
        for (index, limb) in self.limbs.iter().enumerate() {
            if *limb != 0 {
                return index as u64 * u64::from(LIMB_BITS) + u64::from(limb.trailing_zeros());
            }
        }

        0
    }

    /// Whether any of the lowest `count` bits is set.
    ///
    /// The sticky bit of a rounding, without building the masked value.
    pub fn any_low_bit_set(&self, count: u64) -> bool {
        let whole: usize = (count / u64::from(LIMB_BITS)) as usize;
        let part: u32 = (count % u64::from(LIMB_BITS)) as u32;

        // Every limb wholly below the cut.
        if self.limbs.iter().take(whole).any(|limb| *limb != 0) {
            return true;
        }

        // And the partial one straddling it.
        match self.limbs.get(whole) {
            Some(limb) if part > 0 => limb & ((1 << part) - 1) != 0,
            _ => false,
        }
    }

    /// One bit, counting from zero at the least significant end.
    pub fn bit(&self, index: u64) -> bool {
        let limb: usize = (index / u64::from(LIMB_BITS)) as usize;

        match self.limbs.get(limb) {
            None => false,
            Some(value) => value >> (index % u64::from(LIMB_BITS)) & 1 == 1,
        }
    }

    /// Drops trailing zero limbs, restoring the canonical form.
    ///
    /// Every operation ends with this. Without it, `1 - 1` would be a one-limb
    /// value holding zero, which would not equal [`BigUint::zero`] and would break
    /// the derived [`PartialEq`].
    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    /// The sum.
    pub fn sum(&self, other: &Self) -> Self {
        let mut limbs: Vec<u64> = Vec::with_capacity(self.limbs.len().max(other.limbs.len()) + 1);
        let mut carry: u64 = 0;

        for index in 0..self.limbs.len().max(other.limbs.len()) {
            // A limb past the end of the shorter value reads as zero.
            let left: u64 = self.limbs.get(index).copied().unwrap_or(0);
            let right: u64 = other.limbs.get(index).copied().unwrap_or(0);
            let sum: u128 = left as u128 + right as u128 + carry as u128;

            limbs.push(sum as u64);
            carry = (sum >> LIMB_BITS) as u64;
        }

        // Unlike the fixed-width types, a carry out of the top is not an overflow:
        // the value simply gets one limb longer.
        if carry != 0 {
            limbs.push(carry);
        }

        Self::from_limbs(limbs)
    }

    /// The difference, or `None` when it would go below zero.
    ///
    /// Unsigned, so there is nothing to return for a negative result — that is
    /// [`BigInt`]'s job.
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        if self < other {
            return None;
        }

        let mut limbs: Vec<u64> = Vec::with_capacity(self.limbs.len());
        let mut borrow: u64 = 0;

        for index in 0..self.limbs.len() {
            let left: u64 = self.limbs[index];
            let right: u64 = other.limbs.get(index).copied().unwrap_or(0);

            let (partial, first) = left.overflowing_sub(right);
            let (value, second) = partial.overflowing_sub(borrow);

            limbs.push(value);
            borrow = u64::from(first || second);
        }

        // The comparison above guarantees this, but the arithmetic should say so.
        debug_assert_eq!(borrow, 0, "a checked subtraction should not borrow out");

        Some(Self::from_limbs(limbs))
    }

    /// The product.
    ///
    /// Schoolbook, `O(n × m)` in the two limb counts. See
    /// [`WideUint::widening_mul`](crate::math::WideUint::widening_mul) for why
    /// Karatsuba is not used: the measured crossover is around 2048 bits, and
    /// values that large are the exception even here.
    pub fn product(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        // The product needs at most the sum of the two lengths.
        let mut limbs: Vec<u64> = vec![0; self.limbs.len() + other.limbs.len()];

        for (left, value) in self.limbs.iter().enumerate() {
            if *value == 0 {
                continue;
            }

            let mut carry: u64 = 0;

            for (right, other_value) in other.limbs.iter().enumerate() {
                let index: usize = left + right;
                let product: u128 = *value as u128 * *other_value as u128
                    + limbs[index] as u128
                    + carry as u128;

                limbs[index] = product as u64;
                carry = (product >> LIMB_BITS) as u64;
            }

            limbs[left + other.limbs.len()] = carry;
        }

        Self::from_limbs(limbs)
    }

    /// The quotient and remainder, or `None` when dividing by zero.
    ///
    /// Knuth's Algorithm D, as [`WideUint::div_rem`](crate::math::WideUint::div_rem)
    /// is — one 64-bit quotient digit per step rather than one bit. The structure
    /// is the same; what differs is that the working buffers are sized from the
    /// inputs rather than from a type parameter, which if anything makes it simpler:
    /// the extra digit normalisation needs is just a `push`.
    ///
    /// The rare add-back correction (step D5) is tested here the same way, with
    /// inputs constructed to force it.
    pub fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        if divisor.is_zero() {
            return None;
        }

        if self < divisor {
            return Some((Self::zero(), self.clone()));
        }

        // A single-limb divisor cannot use the two-limb estimate, and does not need
        // to: short division is one hardware divide per limb.
        if divisor.limbs.len() == 1 {
            let (quotient, remainder) = self.div_rem_limb(divisor.limbs[0]);

            return Some((quotient, Self::from_limbs(vec![remainder])));
        }

        let n: usize = divisor.limbs.len();
        let steps: usize = self.limbs.len() - n;

        // D1: normalise so the divisor's top limb has its high bit set, which is
        // what bounds the estimate's error at two.
        let shift: u32 = divisor.limbs[n - 1].leading_zeros();
        let divisor: Self = divisor.shifted_up(u64::from(shift));
        let mut remainder: Self = self.shifted_up(u64::from(shift));

        // Algorithm D indexes one digit past the numerator; growing the vector is
        // all that takes here.
        remainder.limbs.resize(self.limbs.len() + 1, 0);

        let base: u128 = 1 << LIMB_BITS;
        let top: u64 = divisor.limbs[n - 1];
        let second: u64 = divisor.limbs[n - 2];
        let mut quotient: Vec<u64> = vec![0; steps + 1];

        // D2: one quotient digit per step, most significant first.
        for step in (0..=steps).rev() {
            let high: usize = step + n;

            // D3: estimate from the top two digits, then correct against the
            // divisor's second limb.
            let pair: u128 =
                (remainder.limbs[high] as u128) << LIMB_BITS | remainder.limbs[high - 1] as u128;
            let mut estimate: u128 = pair / top as u128;
            let mut rest: u128 = pair % top as u128;

            while estimate >= base
                || estimate * second as u128
                    > (rest << LIMB_BITS) + remainder.limbs[high - 2] as u128
            {
                estimate -= 1;
                rest += top as u128;

                if rest >= base {
                    break;
                }
            }

            // D4: subtract the divisor times the estimate.
            let mut carry: u128 = 0;
            let mut borrow: i128 = 0;

            for index in 0..n {
                let product: u128 = estimate * divisor.limbs[index] as u128 + carry;
                carry = product >> LIMB_BITS;

                let difference: i128 = remainder.limbs[step + index] as i128
                    - (product & (base - 1)) as i128
                    - borrow;

                remainder.limbs[step + index] = difference as u64;
                borrow = i128::from(difference < 0);
            }

            let difference: i128 = remainder.limbs[high] as i128 - carry as i128 - borrow;
            remainder.limbs[high] = difference as u64;

            // D5: the estimate was one too large after all — decrement it and add
            // the divisor back. Roughly one division in 2^63 reaches this.
            if difference < 0 {
                estimate -= 1;

                let mut carry: u128 = 0;

                for index in 0..n {
                    let sum: u128 = remainder.limbs[step + index] as u128
                        + divisor.limbs[index] as u128
                        + carry;

                    remainder.limbs[step + index] = sum as u64;
                    carry = sum >> LIMB_BITS;
                }

                remainder.limbs[high] = (remainder.limbs[high] as u128 + carry) as u64;
            }

            quotient[step] = estimate as u64;
        }

        // D8: what is left below the divisor's width is the remainder, un-normalised.
        remainder.limbs.truncate(n);
        remainder.trim();

        Some((Self::from_limbs(quotient), remainder.shifted_down(u64::from(shift))))
    }

    /// The quotient and remainder of a division by a single limb.
    fn div_rem_limb(&self, divisor: u64) -> (Self, u64) {
        let mut limbs: Vec<u64> = vec![0; self.limbs.len()];
        let mut remainder: u64 = 0;

        for index in (0..self.limbs.len()).rev() {
            let current: u128 = (remainder as u128) << LIMB_BITS | self.limbs[index] as u128;

            limbs[index] = (current / divisor as u128) as u64;
            remainder = (current % divisor as u128) as u64;
        }

        (Self::from_limbs(limbs), remainder)
    }

    /// Shifted up by `places` bits, growing as needed.
    pub fn shifted_up(&self, places: u64) -> Self {
        if self.is_zero() {
            return Self::zero();
        }

        let whole: usize = (places / u64::from(LIMB_BITS)) as usize;
        let part: u32 = (places % u64::from(LIMB_BITS)) as u32;

        // Whole limbs are zeros spliced in at the bottom.
        let mut limbs: Vec<u64> = vec![0; whole];

        if part == 0 {
            limbs.extend_from_slice(&self.limbs);
        } else {
            let mut carry: u64 = 0;

            for limb in &self.limbs {
                limbs.push((limb << part) | carry);
                carry = limb >> (LIMB_BITS - part);
            }

            if carry != 0 {
                limbs.push(carry);
            }
        }

        Self::from_limbs(limbs)
    }

    /// Shifted down by `places` bits, dropping what falls off the bottom.
    pub fn shifted_down(&self, places: u64) -> Self {
        let whole: usize = (places / u64::from(LIMB_BITS)) as usize;
        let part: u32 = (places % u64::from(LIMB_BITS)) as u32;

        if whole >= self.limbs.len() {
            return Self::zero();
        }

        let kept: &[u64] = &self.limbs[whole..];
        let mut limbs: Vec<u64> = Vec::with_capacity(kept.len());

        for index in 0..kept.len() {
            let mut value: u64 = kept[index] >> part;

            // Bits arriving from the limb above. Guarded, because shifting by the
            // full width is undefined.
            if part > 0 && index + 1 < kept.len() {
                value |= kept[index + 1] << (LIMB_BITS - part);
            }

            limbs.push(value);
        }

        Self::from_limbs(limbs)
    }

    /// This value raised to a power, by repeated squaring.
    pub fn pow(&self, exponent: u64) -> Self {
        if exponent == 0 {
            return Self::one();
        }

        let mut result: Self = Self::one();
        let mut base: Self = self.clone();
        let mut remaining: u64 = exponent;

        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result.product(&base);
            }

            remaining >>= 1;

            if remaining == 0 {
                break;
            }

            base = base.product(&base);
        }

        result
    }

    /// The greatest common divisor, by the Euclidean algorithm.
    pub fn gcd(&self, other: &Self) -> Self {
        let mut left: Self = self.clone();
        let mut right: Self = other.clone();

        while !right.is_zero() {
            let remainder: Self = left
                .div_rem(&right)
                .expect("the divisor was just checked to be non-zero")
                .1;

            left = right;
            right = remainder;
        }

        left
    }

    /// A value parsed from decimal digits.
    ///
    /// Nineteen digits at a time, since `10^19` is the largest power of ten that
    /// fits in a limb: multiply the accumulator by that and add the chunk. One
    /// limb-multiply and one limb-add per nineteen digits, rather than per digit.
    ///
    /// Reading a number whose length is not known ahead of time is one of the
    /// reasons this type exists, so this is part of the point rather than a
    /// convenience.
    pub fn parse_decimal(text: &str) -> Option<Self> {
        if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }

        /// How many decimal digits fit in one limb.
        const CHUNK: usize = 19;

        let digits: &[u8] = text.as_bytes();
        let mut value: Self = Self::zero();
        let mut position: usize = 0;

        // The first chunk is short, so that the rest land on a boundary.
        let first: usize = match digits.len() % CHUNK {
            0 => CHUNK,
            remainder => remainder,
        };
        let mut width: usize = first;

        while position < digits.len() {
            let slice: &str = std::str::from_utf8(&digits[position..position + width]).ok()?;
            let chunk: u64 = slice.parse().ok()?;

            value = value
                .product(&Self::from(10u64.pow(width as u32)))
                .sum(&Self::from(chunk));

            position += width;
            width = CHUNK;
        }

        Some(value)
    }

    /// The value as a `u128`, or `None` when it needs more room.
    pub fn to_u128(&self) -> Option<u128> {
        if self.limbs.len() > 2 {
            return None;
        }

        let low: u128 = self.limbs.first().copied().unwrap_or(0) as u128;
        let high: u128 = self.limbs.get(1).copied().unwrap_or(0) as u128;

        Some(low | high << LIMB_BITS)
    }

    /// The value as a fixed-width integer, or `None` when it needs more limbs than
    /// that width has.
    ///
    /// The bridge to [`WideUint`](crate::math::WideUint), for handing a grown value
    /// back to code that wants a predictable size.
    pub fn to_wide<const LIMBS: usize>(&self) -> Option<crate::math::WideUint<LIMBS>> {
        if self.limbs.len() > LIMBS {
            return None;
        }

        let mut limbs = [0u64; LIMBS];

        for (index, limb) in self.limbs.iter().enumerate() {
            limbs[index] = *limb;
        }

        Some(crate::math::WideUint::from_limbs(limbs))
    }

    /// A growable value from a fixed-width one.
    pub fn from_wide<const LIMBS: usize>(value: &crate::math::WideUint<LIMBS>) -> Self {
        Self::from_limbs(value.limbs().to_vec())
    }
}

impl Ord for BigUint {
    /// A longer value is larger, since neither has trailing zeros; equal lengths
    /// compare from the top limb down.
    fn cmp(&self, other: &Self) -> Ordering {
        self.limbs
            .len()
            .cmp(&other.limbs.len())
            .then_with(|| self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Every unsigned primitive converts, since all of them fit.
macro_rules! from_unsigned {
    ($($type:ty),* $(,)?) => {
        $(
            impl From<$type> for BigUint {
                fn from(value: $type) -> Self {
                    Self::from_limbs(vec![u64::from(value)])
                }
            }
        )*
    };
}

from_unsigned!(u8, u16, u32, u64);

impl From<usize> for BigUint {
    fn from(value: usize) -> Self {
        Self::from_limbs(vec![value as u64])
    }
}

impl From<u128> for BigUint {
    fn from(value: u128) -> Self {
        Self::from_limbs(vec![value as u64, (value >> LIMB_BITS) as u64])
    }
}

impl Add for BigUint {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BigUint::sum(&self, &other)
    }
}

/// Panics on a negative result, as the unsigned primitives do. Use
/// [`BigUint::checked_sub`] where that is a possible answer, or [`BigInt`] where it
/// is a meaningful one.
impl Sub for BigUint {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.checked_sub(&other)
            .expect("subtraction went below zero")
    }
}

impl Mul for BigUint {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        BigUint::product(&self, &other)
    }
}

impl Shl<u64> for BigUint {
    type Output = Self;

    fn shl(self, places: u64) -> Self {
        BigUint::shifted_up(&self, places)
    }
}

impl Shr<u64> for BigUint {
    type Output = Self;

    fn shr(self, places: u64) -> Self {
        BigUint::shifted_down(&self, places)
    }
}

impl AddAssign for BigUint {
    fn add_assign(&mut self, other: Self) {
        *self = BigUint::sum(self, &other);
    }
}

impl SubAssign for BigUint {
    fn sub_assign(&mut self, other: Self) {
        *self = self.checked_sub(&other).expect("subtraction went below zero");
    }
}

impl MulAssign for BigUint {
    fn mul_assign(&mut self, other: Self) {
        *self = BigUint::product(self, &other);
    }
}

impl Zero for BigUint {
    fn zero() -> Self {
        BigUint::zero()
    }

    fn is_zero(&self) -> bool {
        BigUint::is_zero(self)
    }
}

impl One for BigUint {
    fn one() -> Self {
        BigUint::one()
    }

    fn is_one(&self) -> bool {
        BigUint::is_one(self)
    }
}

/// A semiring and no more: there is no additive inverse, exactly as for `u64` and
/// [`WideUint`](crate::math::WideUint).
impl Semiring for BigUint {}

/// The inverse of [`fmt::Display`], so a value round-trips through its own text.
impl FromStr for BigUint {
    type Err = ParseBigError;

    fn from_str(text: &str) -> Result<Self, ParseBigError> {
        Self::parse_decimal(text).ok_or(ParseBigError)
    }
}

impl fmt::Display for BigUint {
    /// In decimal, by repeatedly dividing out the largest power of ten that fits in
    /// a limb.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return formatter.write_str("0");
        }

        /// The largest power of ten below `2^64`: nineteen digits per division.
        const CHUNK: u64 = 10_000_000_000_000_000_000;

        let mut remaining: BigUint = self.clone();
        let mut chunks: Vec<u64> = Vec::new();

        while !remaining.is_zero() {
            let (quotient, digits) = remaining.div_rem_limb(CHUNK);

            chunks.push(digits);
            remaining = quotient;
        }

        let mut chunks = chunks.into_iter().rev();

        if let Some(first) = chunks.next() {
            write!(formatter, "{first}")?;
        }

        // Every chunk but the most significant keeps its leading zeros.
        for chunk in chunks {
            write!(formatter, "{chunk:019}")?;
        }

        Ok(())
    }
}

impl fmt::Debug for BigUint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self}")
    }
}

impl fmt::LowerHex for BigUint {
    /// Most significant limb first, the top one without leading zeros.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some((top, rest)) = self.limbs.split_last() else {
            return formatter.write_str("0");
        };

        write!(formatter, "{top:x}")?;

        for limb in rest.iter().rev() {
            write!(formatter, "{limb:016x}")?;
        }

        Ok(())
    }
}

// ===========================================================================
// Signed
// ===========================================================================

/// An integer of any size, positive or negative.
///
/// # Example
///
/// ```
/// use voxel_world::math::BigInt;
/// use voxel_world::math::traits::EuclideanRing;
///
/// // 2^200 - 1 and 2^100 - 1 share a factor no primitive could hold.
/// let big = BigInt::from(2).pow(200) - BigInt::one();
/// let small = BigInt::from(2).pow(100) - BigInt::one();
///
/// // gcd(2^200 - 1, 2^100 - 1) = 2^gcd(200,100) - 1 = 2^100 - 1.
/// assert_eq!(big.gcd_normalised(&small), small);
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct BigInt {
    /// Always `false` when the magnitude is zero, so there is no negative zero.
    negative: bool,
    magnitude: BigUint,
}

impl BigInt {
    /// Zero.
    pub fn zero() -> Self {
        Self {
            negative: false,
            magnitude: BigUint::zero(),
        }
    }

    /// One.
    pub fn one() -> Self {
        Self {
            negative: false,
            magnitude: BigUint::one(),
        }
    }

    /// A value from a sign and a magnitude, collapsing a negative zero.
    pub fn from_parts(negative: bool, magnitude: BigUint) -> Self {
        Self {
            // The one canonicalisation this type needs.
            negative: negative && !magnitude.is_zero(),
            magnitude,
        }
    }

    /// Whether the value is below zero.
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// Whether the value is zero.
    pub fn is_zero(&self) -> bool {
        self.magnitude.is_zero()
    }

    /// Whether the value is one.
    pub fn is_one(&self) -> bool {
        !self.negative && self.magnitude.is_one()
    }

    /// The distance from zero.
    ///
    /// Unlike [`WideInt::magnitude`](crate::math::WideInt::magnitude), this needs no
    /// caveat: there is no most-negative value whose magnitude will not fit.
    pub fn magnitude(&self) -> &BigUint {
        &self.magnitude
    }

    /// The value with its sign removed.
    pub fn abs(&self) -> Self {
        Self::from_parts(false, self.magnitude.clone())
    }

    /// The negation.
    ///
    /// Total, where the fixed-width types need a `checked_neg` that can fail: a
    /// growable integer has no asymmetric `MIN`.
    pub fn negated(&self) -> Self {
        Self::from_parts(!self.negative, self.magnitude.clone())
    }

    /// The sum.
    pub fn sum(&self, other: &Self) -> Self {
        if self.negative == other.negative {
            // Same sign: the magnitudes add and the sign is kept.
            return Self::from_parts(self.negative, self.magnitude.sum(&other.magnitude));
        }

        // Opposite signs: the smaller magnitude comes off the larger, and the
        // larger one's sign wins.
        match self.magnitude.cmp(&other.magnitude) {
            Ordering::Equal => Self::zero(),
            Ordering::Greater => Self::from_parts(
                self.negative,
                self.magnitude
                    .checked_sub(&other.magnitude)
                    .expect("the larger magnitude was just established"),
            ),
            Ordering::Less => Self::from_parts(
                other.negative,
                other
                    .magnitude
                    .checked_sub(&self.magnitude)
                    .expect("the larger magnitude was just established"),
            ),
        }
    }

    /// The difference.
    pub fn difference(&self, other: &Self) -> Self {
        self.sum(&other.negated())
    }

    /// The product.
    pub fn product(&self, other: &Self) -> Self {
        Self::from_parts(
            self.negative != other.negative,
            self.magnitude.product(&other.magnitude),
        )
    }

    /// The quotient and remainder, or `None` when dividing by zero.
    ///
    /// Truncates towards zero and gives the remainder the numerator's sign, as
    /// Rust's own integer division does. Unlike
    /// [`WideInt::div_rem`](crate::math::WideInt::div_rem) this has no second
    /// failure case: there is no `MIN / -1` to overflow.
    pub fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        let (quotient, remainder) = self.magnitude.div_rem(&divisor.magnitude)?;

        Some((
            Self::from_parts(self.negative != divisor.negative, quotient),
            // The remainder takes the numerator's sign, so `-7 % 2` is `-1`.
            Self::from_parts(self.negative, remainder),
        ))
    }

    /// This value raised to a power, by repeated squaring.
    pub fn pow(&self, exponent: u64) -> Self {
        Self::from_parts(
            // A negative base stays negative only through an odd power.
            self.negative && exponent % 2 == 1,
            self.magnitude.pow(exponent),
        )
    }

    /// The value as an `i128`, or `None` when it needs more room.
    pub fn to_i128(&self) -> Option<i128> {
        let magnitude: u128 = self.magnitude.to_u128()?;

        if self.negative {
            // The negative range holds one more value than the positive one.
            (magnitude <= i128::MAX as u128 + 1).then(|| (magnitude as i128).wrapping_neg())
        } else {
            (magnitude <= i128::MAX as u128).then_some(magnitude as i128)
        }
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            // Among negatives the order reverses: the larger magnitude is smaller.
            (true, true) => other.magnitude.cmp(&self.magnitude),
            (false, false) => self.magnitude.cmp(&other.magnitude),
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Every signed primitive converts. Written through `i64` so the sign handling
/// exists once — and note `unsigned_abs`, which is what makes `MIN` work: negating
/// it would overflow, taking its magnitude does not.
macro_rules! from_signed {
    ($($type:ty),* $(,)?) => {
        $(
            impl From<$type> for BigInt {
                fn from(value: $type) -> Self {
                    Self::from(i64::from(value))
                }
            }
        )*
    };
}

from_signed!(i8, i16, i32);

impl From<i64> for BigInt {
    fn from(value: i64) -> Self {
        Self::from_parts(value < 0, BigUint::from(value.unsigned_abs()))
    }
}

impl From<isize> for BigInt {
    fn from(value: isize) -> Self {
        Self::from(value as i64)
    }
}

impl From<i128> for BigInt {
    fn from(value: i128) -> Self {
        Self::from_parts(value < 0, BigUint::from(value.unsigned_abs()))
    }
}

impl From<BigUint> for BigInt {
    fn from(value: BigUint) -> Self {
        Self::from_parts(false, value)
    }
}

impl Add for BigInt {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BigInt::sum(&self, &other)
    }
}

impl Sub for BigInt {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        BigInt::difference(&self, &other)
    }
}

impl Mul for BigInt {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        BigInt::product(&self, &other)
    }
}

impl Neg for BigInt {
    type Output = Self;

    fn neg(self) -> Self {
        self.negated()
    }
}

impl AddAssign for BigInt {
    fn add_assign(&mut self, other: Self) {
        *self = BigInt::sum(self, &other);
    }
}

impl SubAssign for BigInt {
    fn sub_assign(&mut self, other: Self) {
        *self = BigInt::difference(self, &other);
    }
}

impl MulAssign for BigInt {
    fn mul_assign(&mut self, other: Self) {
        *self = BigInt::product(self, &other);
    }
}

impl Zero for BigInt {
    fn zero() -> Self {
        BigInt::zero()
    }

    fn is_zero(&self) -> bool {
        BigInt::is_zero(self)
    }
}

impl One for BigInt {
    fn one() -> Self {
        BigInt::one()
    }

    fn is_one(&self) -> bool {
        BigInt::is_one(self)
    }
}

impl Semiring for BigInt {}
impl Ring for BigInt {}
impl CommutativeRing for BigInt {}

/// Division with a remainder, so greatest common divisors come from the trait.
///
/// Worth noting what this one gets that
/// [`WideInt`](crate::math::WideInt)'s does not: the laws hold for *every* input,
/// because nothing overflows. A fixed-width ring is only a ring until a product
/// leaves its range.
impl EuclideanRing for BigInt {
    /// The magnitude, which is unsigned and so compares the right way.
    type Size = BigUint;

    fn euclidean_size(&self) -> BigUint {
        self.magnitude.clone()
    }

    fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        BigInt::div_rem(self, divisor)
    }

    /// Non-negative, which is the usual convention.
    fn gcd_normalised(&self, other: &Self) -> Self {
        self.gcd(other).abs()
    }
}

/// Accepts an optional leading sign, then decimal digits.
impl FromStr for BigInt {
    type Err = ParseBigError;

    fn from_str(text: &str) -> Result<Self, ParseBigError> {
        let (negative, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };

        let magnitude: BigUint = BigUint::parse_decimal(digits).ok_or(ParseBigError)?;

        Ok(Self::from_parts(negative, magnitude))
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negative {
            formatter.write_str("-")?;
        }

        write!(formatter, "{}", self.magnitude)
    }
}

impl fmt::Debug for BigInt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self}")
    }
}

/// What went wrong parsing a [`BigUint`] or [`BigInt`].
///
/// One variant, because there is only one way to fail: the text was not a sign
/// followed by decimal digits. A growable integer has no range to overflow.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParseBigError;

impl fmt::Display for ParseBigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("expected an optional sign followed by decimal digits")
    }
}

impl std::error::Error for ParseBigError {}
