//! A fixed-width floating-point number, with a significand and exponent both as
//! wide as asked for at compile time.
//!
//! [`WideFloat<SIGNIFICAND, EXPONENT>`](WideFloat) holds a value as
//!
//! ```text
//! ±significand × 2^exponent
//! ```
//!
//! with the significand an [`WideUint<SIGNIFICAND>`] and the exponent an
//! [`WideInt<EXPONENT>`]. Both widths are chosen at compile time, so the type
//! neither allocates nor has a precision that changes underneath a calculation.
//!
//! ```text
//! type Float256 = WideFloat<4>;        // 256 bits of precision, 64-bit exponent
//! type Float1024 = WideFloat<16>;      // 1024 bits of precision
//! type Astronomical = WideFloat<4, 2>; // 128-bit exponent, for the absurd
//! ```
//!
//! # Which half is worth widening
//!
//! The two widths do very different amounts of work, and it is worth being clear
//! about which one is actually useful to widen.
//!
//! **The significand is the one that matters.** It sets how many bits of the
//! answer are right. An `f64` gives 53; `WideFloat<4>` gives 256, and
//! `WideFloat<16>` gives 1024. This is the knob to reach for.
//!
//! **The exponent is almost never worth widening.** The default single limb
//! already reaches `2^(±9.2×10^18)`. Written out, the largest value is a 1
//! followed by about two billion billion binary digits — there is no physical
//! quantity anywhere near it, and no calculation that survives to reach it. A
//! second limb is offered because it costs nothing to leave at the default; it is
//! not offered because anyone needs it.
//!
//! So the exponent width is a type parameter with a default, and the significand
//! width is not.
//!
//! # Chosen precision, enormous range
//!
//! This is the same model as MPFR, and it is worth stating plainly: the precision
//! is *chosen at compile time*, not unlimited. `1/3` has no finite binary
//! expansion, so it is rounded to the width in hand, exactly as an `f64` rounds it
//! — just 256 bits in rather than 53.
//!
//! For a float that grows instead, [`BigFloat`](crate::math::BigFloat) keeps
//! addition, subtraction and multiplication **exact** by widening its significand
//! as needed, and asks for a precision only where one is unavoidable.
//!
//! What this type is not: exact. If exactness is what matters,
//! [`Ratio`](crate::math::Ratio) holds a third as a third for ever, and
//! [`Interval`](crate::math::Interval) carries rigorous bounds around a value
//! rather than a rounded guess at it. The choice between the three:
//!
//! | | exact? | range | cost |
//! |---|---|---|---|
//! | [`Ratio`](crate::math::Ratio) | yes | grows until it overflows | denominators explode |
//! | `WideFloat` | no, rounded | effectively unbounded | fixed, chosen |
//! | [`Interval`](crate::math::Interval) | bounds are rigorous | of its component | two of everything |
//! | [`Fixed`](crate::math::Fixed) | no, truncated | `±4×10^28` | cheapest |
//!
//! # Rounding
//!
//! Every operation rounds **to nearest, ties to even** — what IEEE 754 does by
//! default, and for its reason: truncating instead would bias every result
//! towards zero, and a long accumulation would drift steadily. Ties to even
//! rather than always-up because always-up biases too, just more subtly.
//!
//! # No NaN, no infinity
//!
//! Deliberately, and following [`Fixed`](crate::math::Fixed). An operation with no
//! representable answer returns `None` from a `checked_` method, or panics from an
//! operator, rather than producing a value that poisons everything downstream.
//!
//! What that buys is real: [`Ord`] and [`Eq`] are honest, so an `WideFloat` can key
//! a [`BTreeMap`](std::collections::BTreeMap) or a
//! [`HashMap`](std::collections::HashMap). There is no value that fails to equal
//! itself, and no negative zero — zero has exactly one representation.
//!
//! # What is deliberately simple
//!
//! Significand division is binary long division, one bit at a time, for the same
//! reason [`WideUint::div_rem_binary`] exists: it is obviously correct. Unlike
//! there, it has not yet been replaced by Knuth's Algorithm D, because doing so
//! needs a `2×LIMBS`-by-`LIMBS` division primitive that `WideUint` does not have.
//! That is the known place to make this faster, and it is a contained change; see
//! [`WideFloat::checked_div`].

use crate::math::wide_integer::{WideInt, WideUint};
use crate::math::traits::{CommutativeRing, Field, One, Ring, Semiring, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// How many bits one limb holds.
const LIMB_BITS: u32 = u64::BITS;

/// A floating-point number with a `SIGNIFICAND × 64`-bit significand and an
/// `EXPONENT × 64`-bit exponent.
///
/// # Invariants
///
/// Every value is canonical, which is what lets [`PartialEq`] and [`Ord`] be
/// derived rather than hand-written:
///
/// - A non-zero value has its significand's **top bit set**. So the significand is
///   always in `[2^(BITS-1), 2^BITS)`, and comparing two values can compare
///   exponents before significands.
/// - Zero is exactly `{ negative: false, significand: 0, exponent: 0 }`. There is
///   no negative zero and no other spelling of zero.
///
/// # Example
///
/// ```
/// use voxel_world::math::WideFloat;
///
/// type Float = WideFloat<4>; // 256 bits of precision
///
/// let third: Float = Float::from_integer(1) / Float::from_integer(3);
///
/// // An f64 gets 1/3 right to about 16 digits; this gets it right to about 77.
/// assert!(third.to_string().starts_with("0.333333333333333333333333333"));
///
/// // And three thirds is one again, because the rounding is to nearest.
/// assert_eq!(third * Float::from_integer(3), Float::from_integer(1));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WideFloat<const SIGNIFICAND: usize, const EXPONENT: usize = 1> {
    /// The sign. Always `false` for zero, so there is no negative zero.
    negative: bool,
    /// The significand, with its top bit set unless the value is zero.
    significand: WideUint<SIGNIFICAND>,
    /// The power of two the significand is multiplied by.
    exponent: WideInt<EXPONENT>,
}

impl<const S: usize, const E: usize> WideFloat<S, E> {
    /// How many bits of precision the significand carries.
    pub const PRECISION: u32 = WideUint::<S>::BITS;

    /// Zero.
    pub const ZERO: Self = Self {
        negative: false,
        significand: WideUint::ZERO,
        exponent: WideInt::ZERO,
    };

    /// One.
    ///
    /// Not a constant, because the normalised significand of one is
    /// `2^(PRECISION-1)` and that array has to be built at run time for a generic
    /// width — the same reason [`WideUint::one`] is a function.
    pub fn one() -> Self {
        Self::from_integer(1)
    }

    /// Whether the value is zero.
    pub fn is_zero(&self) -> bool {
        self.significand.is_zero()
    }

    /// Whether the value is below zero. Never true for zero itself.
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// The significand, normalised so its top bit is set — or zero.
    pub fn significand(&self) -> WideUint<S> {
        self.significand
    }

    /// The power of two the significand is scaled by.
    pub fn exponent(&self) -> WideInt<E> {
        self.exponent
    }

    /// The value with its sign removed.
    pub fn abs(&self) -> Self {
        Self {
            negative: false,
            ..*self
        }
    }

    /// A value from its parts, normalised.
    ///
    /// The one place the invariants are established: the significand is shifted up
    /// until its top bit is set, the exponent adjusted to match, and a zero
    /// significand collapsed to the canonical zero. Every other constructor comes
    /// through here.
    ///
    /// `None` if normalising would take the exponent out of range.
    fn normalised(negative: bool, significand: WideUint<S>, exponent: WideInt<E>) -> Option<Self> {
        if significand.is_zero() {
            return Some(Self::ZERO);
        }

        let shift: u32 = significand.leading_zeros();
        let exponent: WideInt<E> = exponent.checked_sub(&WideInt::from(i64::from(shift)))?;

        Some(Self {
            negative,
            significand: significand.wrapping_shl(shift),
            exponent,
        })
    }

    /// A whole number.
    pub fn from_integer(value: i64) -> Self {
        let negative: bool = value < 0;
        let significand: WideUint<S> = WideUint::from(value.unsigned_abs());

        Self::normalised(negative, significand, WideInt::ZERO)
            .expect("an i64 cannot take the exponent out of range")
    }

    /// A value from a wide integer.
    ///
    /// `None` if the integer needs more precision than the significand has, since
    /// the low bits would be lost silently otherwise. Use
    /// [`WideFloat::from_arb_int_rounded`] to accept the rounding.
    pub fn from_arb_int(value: &WideInt<S>) -> Option<Self> {
        let magnitude: WideUint<S> = value.magnitude();

        // The magnitude has at most `PRECISION` bits by construction, so it always
        // fits — this cannot actually fail, but the signature matches the rounded
        // form for symmetry.
        Self::normalised(value.is_negative(), magnitude, WideInt::ZERO)
    }

    /// A value from a wide unsigned integer, rounding if it does not fit.
    pub fn from_arb_uint(value: &WideUint<S>) -> Self {
        Self::normalised(false, *value, WideInt::ZERO)
            .expect("no shift of a same-width value can leave the exponent range")
    }

    /// A value from a wide integer, rounding rather than refusing.
    ///
    /// Identical to [`WideFloat::from_arb_int`] at equal widths; kept as the name
    /// to reach for when the caller means "round it".
    pub fn from_arb_int_rounded(value: &WideInt<S>) -> Self {
        Self::from_arb_int(value).expect("an equal-width integer always fits")
    }

    /// The value of an `f64`, exactly.
    ///
    /// `None` for a NaN or an infinity, which this type has no room for. Every
    /// finite `f64` converts without losing anything, since 53 bits of significand
    /// fit in any width this type offers.
    pub fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }

        if value == 0.0 {
            return Some(Self::ZERO);
        }

        let bits: u64 = value.to_bits();
        let negative: bool = bits >> 63 == 1;
        let raw_exponent: i64 = ((bits >> 52) & 0x7FF) as i64;
        let fraction: u64 = bits & ((1 << 52) - 1);

        // A subnormal has no implicit leading one and a fixed exponent; a normal
        // number has both.
        let (significand, exponent) = if raw_exponent == 0 {
            (fraction, -1074_i64)
        } else {
            (fraction | 1 << 52, raw_exponent - 1075)
        };

        Self::normalised(
            negative,
            WideUint::from(significand),
            WideInt::from(exponent),
        )
    }

    /// The nearest `f64`, saturating to an infinity when the value is too large
    /// and to zero when it is too small.
    ///
    /// Unlike the rest of this type, this returns an infinity rather than `None`,
    /// because an `f64` *has* infinities and that is the honest answer for a value
    /// beyond its range.
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }

        // The top limb, with everything below it folded into the low bit. That
        // "sticky" bit is what stops a halfway value rounding to even when it is
        // not actually halfway.
        let top: u64 = self.significand.limbs()[S - 1];
        let sticky: bool = self.significand.limbs()[..S - 1].iter().any(|l| *l != 0);
        let leading: u64 = top | u64::from(sticky);

        // `u64 as f64` rounds to nearest, which is the rounding wanted here.
        let mut value: f64 = leading as f64;

        if self.negative {
            value = -value;
        }

        // The top limb stood for `significand >> (BITS - 64)`, so put that back.
        let Some(exponent) = self
            .exponent
            .checked_add(&WideInt::from(i64::from(Self::PRECISION - LIMB_BITS)))
            .and_then(|total| total.to_i128())
        else {
            // Past `i128`, so certainly past an f64 either way.
            return if self.exponent.is_negative() {
                0.0
            } else if self.negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        };

        // Scale in steps, so a large exponent overflows to infinity (or underflows
        // to zero) the way it should rather than mis-scaling on the way.
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

    /// The sum, or `None` when the exponent would leave its range.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        if other.is_zero() {
            return Some(*self);
        }

        if self.is_zero() {
            return Some(*other);
        }

        // Work with the larger magnitude first, so the shift is always downwards.
        let (large, small) = if self.compare_magnitude(other) == Ordering::Less {
            (other, self)
        } else {
            (self, other)
        };

        let difference: WideInt<E> = large.exponent.checked_sub(&small.exponent)?;
        let gap: i128 = difference.to_i128().unwrap_or(i128::MAX);

        // Past the precision the smaller value is under half a unit in the last
        // place, so it cannot change the rounded answer — in either direction. For
        // a sum that is plain. For a difference it needs a moment: the answer
        // becomes `large - 1` plus a fraction above a half, which rounds back up to
        // `large`.
        if gap > i128::from(Self::PRECISION) {
            return Some(*large);
        }

        let gap: u32 = gap as u32;

        // The smaller significand aligned against the larger, as an exact
        // `2 × PRECISION`-bit pair: `head` lines up with the significand and `tail`
        // holds everything that falls below it. Nothing is discarded, which is what
        // lets the cancellation case below recover real bits rather than zeros.
        //
        // Both shifts are total: at `gap` of zero the tail shifts out entirely, and
        // at `gap` of `PRECISION` the head does.
        let head: WideUint<S> = small.significand.wrapping_shr(gap);
        let tail: WideUint<S> = small.significand.wrapping_shl(Self::PRECISION - gap);

        if large.negative == small.negative {
            // Same sign: the magnitudes add, and may carry one bit off the top.
            let (sum, carried) = large.significand.overflowing_add(&head);

            if carried {
                // A bit wider than the type, so shift back down. The bit that falls
                // off is now the halfway mark, and the tail is what lies below it.
                let significand: WideUint<S> =
                    sum.wrapping_shr(1) | WideUint::one().wrapping_shl(Self::PRECISION - 1);

                return Self::rounded(
                    large.negative,
                    significand,
                    large.exponent.checked_add(&WideInt::from(1_i64))?,
                    sum.bit(0),
                    !tail.is_zero(),
                );
            }

            return Self::rounded(
                large.negative,
                sum,
                large.exponent,
                tail.bit(Self::PRECISION - 1),
                !tail.wrapping_shl(1).is_zero(),
            );
        }

        // Opposite signs: the magnitudes subtract. A non-zero tail is borrowed from
        // the significand, leaving `2^PRECISION - tail` behind — which is exact, not
        // an approximation.
        let mut value: WideUint<S> = large.significand.wrapping_sub(&head);
        let mut fraction: WideUint<S> = WideUint::ZERO;

        if !tail.is_zero() {
            value = value.wrapping_sub(&WideUint::one());
            fraction = WideUint::ZERO.wrapping_sub(&tail);
        }

        if value.is_zero() && fraction.is_zero() {
            return Some(Self::ZERO);
        }

        // Cancellation: with a `gap` of one the subtraction can annihilate almost
        // every bit, and the answer's low end then has to come from the fraction.
        // Normalising the *pair* together is what supplies those bits. Keeping only
        // two rounding bits instead, as the same-sign path can, would fill the gap
        // with zeros and quietly throw away most of the precision.
        let shift: u32 = if value.is_zero() {
            Self::PRECISION + fraction.leading_zeros()
        } else {
            value.leading_zeros()
        };

        let (significand, fraction) = Self::shift_pair(value, fraction, shift);
        let exponent: WideInt<E> = large
            .exponent
            .checked_sub(&WideInt::from(i64::from(shift)))?;

        Self::rounded(
            large.negative,
            significand,
            exponent,
            fraction.bit(Self::PRECISION - 1),
            !fraction.wrapping_shl(1).is_zero(),
        )
    }

    /// A `2 × PRECISION`-bit pair shifted up together, the low half feeding the
    /// high one.
    fn shift_pair(
        value: WideUint<S>,
        fraction: WideUint<S>,
        shift: u32,
    ) -> (WideUint<S>, WideUint<S>) {
        if shift == 0 {
            return (value, fraction);
        }

        // Past the width, the whole answer comes from the low half.
        if shift >= Self::PRECISION {
            return (
                fraction.wrapping_shl(shift - Self::PRECISION),
                WideUint::ZERO,
            );
        }

        (
            value.wrapping_shl(shift) | fraction.wrapping_shr(Self::PRECISION - shift),
            fraction.wrapping_shl(shift),
        )
    }

    /// The difference, or `None` when the exponent would leave its range.
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        self.checked_add(&other.negated())
    }

    /// The product, or `None` when the exponent would leave its range.
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        if self.is_zero() || other.is_zero() {
            return Some(Self::ZERO);
        }

        let negative: bool = self.negative != other.negative;
        // Exact: nothing of the product is lost, it is just in two pieces.
        let (low, high) = self.significand.widening_mul(&other.significand);

        // Both significands have their top bit set, so the product is in
        // [2^(2·BITS − 2), 2^(2·BITS)) — the high half's top bit is set, or its
        // second from top is. One shift normalises either.
        let exponent: WideInt<E> = self
            .exponent
            .checked_add(&other.exponent)?
            .checked_add(&WideInt::from(i64::from(Self::PRECISION)))?;

        if high.bit(Self::PRECISION - 1) {
            // Already normalised. The low half decides the rounding: its top bit
            // is the half-way mark, the rest is sticky.
            let half: bool = low.bit(Self::PRECISION - 1);
            let sticky: bool = !low.wrapping_shl(1).is_zero();

            return Self::rounded(negative, high, exponent, half, sticky);
        }

        // One bit low: shift the pair up by one, which moves the low half's top bit
        // into the significand.
        let significand: WideUint<S> =
            high.wrapping_shl(1) | WideUint::from(u64::from(low.bit(Self::PRECISION - 1)));
        let half: bool = low.bit(Self::PRECISION - 2);
        let sticky: bool = !low.wrapping_shl(2).is_zero();

        Self::rounded(
            negative,
            significand,
            exponent.checked_sub(&WideInt::from(1_i64))?,
            half,
            sticky,
        )
    }

    /// The quotient, or `None` for a zero divisor or an exponent out of range.
    ///
    /// # How the significand is divided
    ///
    /// Both significands are normalised, so their ratio is in `(1/2, 2)` and the
    /// quotient needs `PRECISION` or `PRECISION + 1` bits. What is computed is
    ///
    /// ```text
    /// floor(a × 2^(PRECISION − 1) / b)
    /// ```
    ///
    /// by binary long division: the remainder is doubled each step and the divisor
    /// subtracted when it fits, one quotient bit at a time. The doubling can carry
    /// past the top of the type, so that bit is tracked separately — the same care
    /// [`WideUint::div_rem_binary`] documents, and for the same reason.
    ///
    /// This is `PRECISION` iterations: 256 for `WideFloat<4>`. Knuth's Algorithm D
    /// would make it a handful of steps instead, but it needs a
    /// `2×LIMBS`-by-`LIMBS` division that [`WideUint`] does not expose yet. That is
    /// the single change that would make this fast, and it is isolated here.
    pub fn checked_div(&self, other: &Self) -> Option<Self> {
        if other.is_zero() {
            return None;
        }

        if self.is_zero() {
            return Some(Self::ZERO);
        }

        let negative: bool = self.negative != other.negative;
        let divisor: WideUint<S> = other.significand;

        let mut remainder: WideUint<S> = self.significand;
        let mut quotient: WideUint<S> = WideUint::ZERO;

        // How many fractional bits to produce. The ratio is in `(1/2, 2)`, so the
        // quotient needs `PRECISION` bits either way — but the binary point sits in
        // a different place on each side of one, and producing a fixed count would
        // be a bit short in the `a < b` case. That bit is not recoverable
        // afterwards: normalising an under-long quotient shifts a *zero* into the
        // last place and rounds at the wrong position.
        let fractional: u32 = if remainder >= divisor {
            // The integer part is one, so that bit is the first of the `PRECISION`.
            remainder = remainder.wrapping_sub(&divisor);
            quotient = WideUint::one();
            Self::PRECISION - 1
        } else {
            // The integer part is zero and is not stored; the first fractional bit
            // is necessarily one, since the ratio is above a half.
            Self::PRECISION
        };

        for _ in 0..fractional {
            let carried: bool = remainder.bit(Self::PRECISION - 1);
            remainder = remainder.wrapping_shl(1);
            quotient = quotient.wrapping_shl(1);

            // A carry out of the top means the true remainder is larger than
            // anything representable, so it certainly exceeds the divisor.
            if carried || remainder >= divisor {
                remainder = remainder.wrapping_sub(&divisor);
                quotient = quotient | WideUint::one();
            }
        }

        // What is left of the remainder says which way to round: compare twice the
        // remainder with the divisor.
        let (doubled, carried) = (remainder.wrapping_shl(1), remainder.bit(Self::PRECISION - 1));
        let half: bool = carried || doubled >= divisor;
        let sticky: bool = if half {
            // Above the halfway point unless the doubled remainder lands exactly on
            // the divisor.
            carried || doubled != divisor
        } else {
            !remainder.is_zero()
        };

        let exponent: WideInt<E> = self
            .exponent
            .checked_sub(&other.exponent)?
            .checked_sub(&WideInt::from(i64::from(fractional)))?;

        // The quotient is normalised in both branches above, so rounding happens at
        // the last place of the answer rather than one short of it.
        Self::rounded(negative, quotient, exponent, half, sticky)
    }

    /// This value with its sign flipped. Zero stays zero.
    fn negated(&self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }

        Self {
            negative: !self.negative,
            ..*self
        }
    }

    /// A value from an un-normalised significand plus the two bits that say how to
    /// round it: whether the discarded part was at least half a unit in the last
    /// place, and whether anything below that was non-zero.
    ///
    /// Round to nearest, ties to even: go up when past halfway, and on an exact
    /// tie go to whichever of the two neighbours has an even last bit.
    fn rounded(
        negative: bool,
        significand: WideUint<S>,
        exponent: WideInt<E>,
        half: bool,
        sticky: bool,
    ) -> Option<Self> {
        let round_up: bool = half && (sticky || significand.bit(0));

        if !round_up {
            return Self::normalised(negative, significand, exponent);
        }

        let (incremented, carried) = significand.overflowing_add(&WideUint::one());

        if carried {
            // The significand was all ones, so rounding up made it a power of two
            // one bit wider. That power of two is the new significand's top bit.
            return Self::normalised(
                negative,
                WideUint::one().wrapping_shl(Self::PRECISION - 1),
                exponent.checked_add(&WideInt::from(1_i64))?,
            );
        }

        Self::normalised(negative, incremented, exponent)
    }

    /// The two values' magnitudes compared, ignoring both signs.
    ///
    /// Both significands are normalised to the same bit length, which is what lets
    /// the exponent decide before the significand is looked at.
    fn compare_magnitude(&self, other: &Self) -> Ordering {
        match (self.is_zero(), other.is_zero()) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            (false, false) => {}
        }

        self.exponent
            .cmp(&other.exponent)
            .then_with(|| self.significand.cmp(&other.significand))
    }
}

impl<const S: usize, const E: usize> Ord for WideFloat<S, E> {
    /// A total order, with no NaN to make it partial and no negative zero to make
    /// two distinct values compare equal.
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            // Among negatives the order reverses: the larger magnitude is smaller.
            (true, true) => other.compare_magnitude(self),
            (false, false) => self.compare_magnitude(other),
        }
    }
}

impl<const S: usize, const E: usize> PartialOrd for WideFloat<S, E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const S: usize, const E: usize> Neg for WideFloat<S, E> {
    type Output = Self;

    fn neg(self) -> Self {
        self.negated()
    }
}

/// Operators panic where the checked methods return `None`, in release builds as
/// well as debug ones — the same choice [`Fixed`](crate::math::Fixed) makes, and
/// for the same reason: a value that silently goes wrong in a shipped build and
/// not in a test is the opposite of reproducible.
impl<const S: usize, const E: usize> Add for WideFloat<S, E> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.checked_add(&other).expect("exponent out of range")
    }
}

impl<const S: usize, const E: usize> Sub for WideFloat<S, E> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.checked_sub(&other).expect("exponent out of range")
    }
}

impl<const S: usize, const E: usize> Mul for WideFloat<S, E> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.checked_mul(&other).expect("exponent out of range")
    }
}

impl<const S: usize, const E: usize> Div for WideFloat<S, E> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self.checked_div(&other)
            .expect("division by zero, or exponent out of range")
    }
}

impl<const S: usize, const E: usize> AddAssign for WideFloat<S, E> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const S: usize, const E: usize> SubAssign for WideFloat<S, E> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<const S: usize, const E: usize> MulAssign for WideFloat<S, E> {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<const S: usize, const E: usize> DivAssign for WideFloat<S, E> {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl<const S: usize, const E: usize> Zero for WideFloat<S, E> {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        WideFloat::is_zero(self)
    }
}

impl<const S: usize, const E: usize> One for WideFloat<S, E> {
    fn one() -> Self {
        Self::one()
    }

    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

impl<const S: usize, const E: usize> Semiring for WideFloat<S, E> {}
impl<const S: usize, const E: usize> Ring for WideFloat<S, E> {}
impl<const S: usize, const E: usize> CommutativeRing for WideFloat<S, E> {}

/// A field up to the rounding its chosen precision forces — the same caveat
/// [`Fixed`](crate::math::Fixed) and `f64` carry, and it is a real one:
/// `(a + b) + c` and `a + (b + c)` can differ in the last bit, so the associativity
/// a ring nominally promises holds only to within a rounding.
///
/// Wider precision pushes the discrepancy further down without removing it. For
/// arithmetic where the laws must hold exactly, [`Ratio`](crate::math::Ratio) is
/// the type that does not round.
impl<const S: usize, const E: usize> Field for WideFloat<S, E> {
    fn inverse(&self) -> Option<Self> {
        Self::one().checked_div(self)
    }

    fn divide(&self, divisor: &Self) -> Option<Self> {
        // Directly, rather than as `self * divisor.inverse()`, which would round
        // twice and be wrong in the last bit for no reason.
        self.checked_div(divisor)
    }
}

impl<const S: usize, const E: usize> From<i64> for WideFloat<S, E> {
    fn from(value: i64) -> Self {
        Self::from_integer(value)
    }
}

// ===========================================================================
// Printing
// ===========================================================================

/// How many bits the exact decimal conversion will build an integer out of before
/// giving up and printing the hexadecimal form instead.
///
/// The decimal form of `significand × 2^exponent` needs an integer of roughly
/// `PRECISION + 2.33 × |exponent|` bits, and then a division per nineteen digits
/// to print it. This bounds that work: 64 Ki bits is a value of some twenty
/// thousand decimal digits, past anything anyone reads and already slow to produce.
const MAX_DECIMAL_BITS: u64 = 64 * 1024;

impl<const S: usize, const E: usize> fmt::Display for WideFloat<S, E> {
    /// The exact decimal value, which is always finite: a binary fraction is a
    /// decimal fraction too, because 2 divides 10.
    ///
    /// Values with an exponent too extreme for that fall back to the hexadecimal
    /// form of [`fmt::LowerHex`], which is exact at any size. A value of `2^(10^18)`
    /// has a quintillion decimal digits and there is nothing useful to print.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return formatter.write_str("0");
        }

        let Some(exponent) = self.exponent.to_i128() else {
            return write!(formatter, "{self:x}");
        };

        // The integer to build is `significand << exponent` when the exponent is
        // positive, and `significand × 5^-exponent` when it is negative — see
        // `decimal_digits`.
        let width: u128 = u128::from(Self::PRECISION)
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

        formatter.write_str(&decimal_digits(
            self.significand.limbs(),
            exponent as i64,
        ))
    }
}

impl<const S: usize, const E: usize> fmt::LowerHex for WideFloat<S, E> {
    /// The significand in hexadecimal and the exponent in decimal, as
    /// `-0x<significand>p<exponent>`, meaning `significand × 2^exponent`.
    ///
    /// Exact for every value at every size, which is what makes it the fallback
    /// when a decimal expansion would be unreasonably long. The same idea as C's
    /// `%a`, though not the same spelling: the significand here is an integer
    /// rather than a value between one and two.
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

impl<const S: usize, const E: usize> fmt::Debug for WideFloat<S, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self}")
    }
}

/// The exact decimal text of `significand × 2^exponent`, without a sign.
///
/// # Why this is exact and finite
///
/// Multiplying by `2^exponent` for a positive exponent is a shift, giving an
/// integer. For a negative one, write `s = -exponent` and use
///
/// ```text
/// significand × 2^-s = significand × 5^s × 10^-s
/// ```
///
/// which is an integer — `significand × 5^s` — with the decimal point moved `s`
/// places left. Every binary fraction terminates in decimal, so there is never a
/// repeating tail to truncate.
///
/// A [`Vec`] rather than an [`WideUint`](crate::math::WideUint) because the width
/// needed depends on the exponent, which is a runtime value. Building a string
/// allocates regardless.
fn decimal_digits(significand: &[u64], exponent: i64) -> String {
    // Little-endian limbs, as everywhere else here.
    let mut value: Vec<u64> = significand.to_vec();
    let mut point: usize = 0;

    if exponent >= 0 {
        shift_up(&mut value, exponent as u64);
    } else {
        let places: u64 = exponent.unsigned_abs();
        point = places as usize;

        // 5^27 is the largest power of five below 2^64, so this is the widest step
        // a single-limb multiply can take.
        const CHUNK: u32 = 27;
        const FIVE_TO_CHUNK: u64 = 5u64.pow(CHUNK);

        let mut remaining: u64 = places;

        while remaining >= u64::from(CHUNK) {
            multiply_small(&mut value, FIVE_TO_CHUNK);
            remaining -= u64::from(CHUNK);
        }

        if remaining > 0 {
            multiply_small(&mut value, 5u64.pow(remaining as u32));
        }
    }

    let digits: String = to_decimal(&mut value);

    if point == 0 {
        return digits;
    }

    // Place the point `point` digits from the right, padding with leading zeros
    // when the integer part is empty.
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

/// Multiplies the limbs in place by a single limb, growing as needed.
fn multiply_small(value: &mut Vec<u64>, factor: u64) {
    let mut carry: u64 = 0;

    for limb in value.iter_mut() {
        let product: u128 = *limb as u128 * factor as u128 + carry as u128;

        *limb = product as u64;
        carry = (product >> LIMB_BITS) as u64;
    }

    if carry != 0 {
        value.push(carry);
    }
}

/// Shifts the limbs in place up by `places` bits, growing as needed.
fn shift_up(value: &mut Vec<u64>, places: u64) {
    let whole: usize = (places / u64::from(LIMB_BITS)) as usize;
    let part: u32 = (places % u64::from(LIMB_BITS)) as u32;

    if part > 0 {
        let mut carry: u64 = 0;

        for limb in value.iter_mut() {
            let shifted: u64 = (*limb << part) | carry;

            carry = *limb >> (LIMB_BITS - part);
            *limb = shifted;
        }

        if carry != 0 {
            value.push(carry);
        }
    }

    if whole > 0 {
        // Whole limbs are just zeros spliced in at the bottom.
        value.splice(0..0, std::iter::repeat_n(0u64, whole));
    }
}

/// The decimal text of the limbs, consuming them.
fn to_decimal(value: &mut [u64]) -> String {
    /// The largest power of ten below `2^64`: nineteen digits per division.
    const CHUNK: u64 = 10_000_000_000_000_000_000;

    let mut chunks: Vec<u64> = Vec::new();

    while value.iter().any(|limb| *limb != 0) {
        chunks.push(divide_small(value, CHUNK));
    }

    if chunks.is_empty() {
        return "0".to_string();
    }

    let mut text: String = String::new();
    let mut chunks = chunks.into_iter().rev();

    if let Some(first) = chunks.next() {
        text.push_str(&first.to_string());
    }

    // Every chunk but the most significant keeps its leading zeros.
    for chunk in chunks {
        text.push_str(&format!("{chunk:019}"));
    }

    text
}

/// Divides the limbs in place by a single limb, returning the remainder.
fn divide_small(value: &mut [u64], divisor: u64) -> u64 {
    let mut remainder: u64 = 0;

    for limb in value.iter_mut().rev() {
        let current: u128 = (remainder as u128) << LIMB_BITS | *limb as u128;

        *limb = (current / divisor as u128) as u64;
        remainder = (current % divisor as u128) as u64;
    }

    remainder
}
