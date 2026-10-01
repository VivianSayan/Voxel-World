//! Fixed-width integers, as wide as asked for at compile time.
//!
//! [`WideUint<LIMBS>`](WideUint) is an unsigned integer of `LIMBS × 64` bits, and
//! [`WideInt<LIMBS>`](WideInt) is its two's-complement signed counterpart. Neither
//! allocates: the limbs are an array, so a value of either lives on the stack and
//! copies like a primitive.
//!
//! The width is **fixed at compile time**, which is the whole trade. These are what
//! to reach for when the size is known and the cost has to be predictable: no
//! allocation, `Copy`, and a `WideUint<4>` that is exactly 32 bytes wherever it
//! goes. They overflow like the primitives do, and say so.
//!
//! When the size is *not* known ahead of time, [`BigUint`](crate::math::BigUint)
//! and [`BigInt`](crate::math::BigInt) grow to fit whatever they are given and
//! never overflow at all.
//!
//! ```text
//! type U256 = WideUint<4>;    // 256 bits
//! type I512 = WideInt<8>;     // 512 bits, signed
//! ```
//!
//! # Why uniform limbs rather than a mixture of widths
//!
//! The tempting design is to build a 24-bit integer from a `u16` and a `u8`,
//! using the widest primitive that fits and making up the remainder with smaller
//! ones. It is worse on every axis that matters:
//!
//! - **It barely saves space.** A `{ u16, u8 }` struct is four bytes, not three,
//!   because alignment rounds it up. Against one masked `u64` limb that saves
//!   four bytes; against a plain `[u8; 3]` it saves nothing at all.
//! - **Arithmetic stops being a loop.** Uniform limbs add with one body for any
//!   width: shift a carry along the array. A mixture puts a width boundary in the
//!   middle of the carry chain, so every distinct shape — 24 bits, 40 bits, 72
//!   bits — needs its own hand-written arithmetic.
//! - **Narrow limbs are slower per byte.** A `u8` limb moves one byte per carry
//!   step where a `u64` limb moves eight, so the small limb added "just to reach
//!   the exact width" makes that part of every operation several times slower.
//!
//! Every serious arbitrary-precision library — GMP, and the crates that follow it
//! — uses one limb type as wide as the machine and masks what it does not need.
//! This does the same.
//!
//! # Why the width is counted in limbs, not bits
//!
//! `WideUint<const BITS: usize>` with the array length worked out from `BITS` does
//! not compile on stable Rust:
//!
//! ```text
//! error: generic parameters may not be used in const operations
//!     limbs: [u64; (BITS + 63) / 64],
//!                   ^^^^ cannot perform const operation using `BITS`
//! ```
//!
//! That needs `generic_const_exprs`, which is still unstable. Counting limbs
//! avoids the problem entirely, at the cost of widths being multiples of 64 —
//! and type aliases hide the arithmetic at the point of use.
//!
//! # Limb order
//!
//! `limbs[0]` is the least significant. That makes a carry a forward loop and a
//! borrow the same, which is the only reason to prefer it.
//!
//! # Division, and how it is trusted
//!
//! [`WideUint::div_rem`] is Knuth's Algorithm D: base-2⁶⁴ long division, one 64-bit
//! quotient digit per step rather than one bit. It is where bignum bugs live,
//! because each digit is *estimated* and then corrected, and the last correction
//! step fires for about one division in 2⁶³.
//!
//! So it is not trusted on its own. [`WideUint::div_rem_binary`] — the obvious
//! bit-at-a-time version — is kept beside it as a reference, and the two are
//! checked against each other. Both are checked against `u128` across the whole
//! overlapping range. And because no amount of random input reaches the rare
//! correction, `testing/tests/wide_integer.rs` carries inputs found by searching for
//! it specifically: a divisor whose *low* limb is large, which the estimate never
//! looks at. Instrumenting the branch confirms the ordinary differential tests
//! never reach it and the constructed ones always do.
//!
//! # Multiplication, and why it is still schoolbook
//!
//! Multiplication is schoolbook, O(LIMBS²). Karatsuba is the asymptotically
//! better algorithm and is the obvious thing to reach for, so the measurement is
//! recorded here rather than left to be re-derived. With a preallocated scratch
//! buffer and a 16-limb threshold:
//!
//! | limbs | bits | schoolbook | Karatsuba | |
//! |---|---|---|---|---|
//! | 4 | 256 | 32 ns | 51 ns | 1.59× loss |
//! | 16 | 1024 | 222 ns | 317 ns | 1.43× loss |
//! | 32 | 2048 | 920 ns | 829 ns | crossover |
//! | 128 | 8192 | 16.2 µs | 8.9 µs | 1.8× win |
//! | 256 | 16384 | 65.8 µs | 28.4 µs | 2.3× win |
//!
//! Karatsuba starts paying at about **2048 bits**. Nothing here is used anywhere
//! near that: `WideUint<4>` is eight times below the crossover, where it loses.
//!
//! The decisive point is where the time actually goes. Before Algorithm D, an
//! `WideUint<4>` multiply-then-divide spent 5.9 ns in the multiply and 735 ns in
//! total — division was **99.2%** of it. A free multiply would have saved six
//! nanoseconds in seven hundred. Algorithm D took that 735 ns to 73 ns; Karatsuba
//! could not have taken it below 729. That is why the division was done and the
//! multiplication is still schoolbook.
//!
//! Two things to know if Karatsuba is ever wanted anyway. It has to split the
//! limbs in half, and `LIMBS / 2` in a type position is the same
//! `generic_const_exprs` wall described above, so it would be slice-based helpers
//! with a threaded scratch buffer rather than a method. And the textbook
//! `(a0+a1)(b0+b1)` form does not terminate: the sum of two k-limb halves needs
//! k+1 limbs, so on an odd split the recursive call gets a slice as long as the one
//! it was given. The subtractive form, `a0·b1 + a1·b0 = z0 + z2 − (a0−a1)(b0−b1)`,
//! keeps every recursive call at exactly n/2 limbs at the cost of tracking two
//! signs.

use crate::math::traits::{
    CommutativeRing, EuclideanRing, One, Ring, Semiring, Zero,
};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
};

/// How many bits one limb holds.
const LIMB_BITS: u32 = u64::BITS;

// ===========================================================================
// Unsigned
// ===========================================================================

/// An unsigned integer of `LIMBS × 64` bits.
///
/// # Example
///
/// ```
/// use voxel_world::math::WideUint;
///
/// type U256 = WideUint<4>;
///
/// let big: U256 = WideUint::from(u128::MAX);
///
/// // Squaring `u128::MAX` needs 256 bits, which this has and `u128` does not.
/// let squared = big.checked_mul(&big).expect("it fits in 256 bits");
/// assert!(squared > big);
///
/// // And it divides back exactly.
/// let (quotient, remainder) = squared.div_rem(&big).expect("not by zero");
/// assert_eq!(quotient, big);
/// assert!(remainder.is_zero());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WideUint<const LIMBS: usize> {
    /// Least significant limb first.
    limbs: [u64; LIMBS],
}

impl<const LIMBS: usize> WideUint<LIMBS> {
    /// Fails the build for a width of no limbs, which has no values.
    const VALID: () = assert!(LIMBS > 0, "an integer needs at least one limb");

    /// How many bits the type holds.
    pub const BITS: u32 = LIMBS as u32 * LIMB_BITS;

    /// Zero.
    pub const ZERO: Self = Self {
        limbs: [0; LIMBS],
    };

    /// The largest value.
    pub const MAX: Self = Self {
        limbs: [u64::MAX; LIMBS],
    };

    /// One.
    ///
    /// A `const fn` rather than a constant because the array has to be built and
    /// then written into, which a constant initialiser cannot express for a
    /// generic length.
    pub const fn one() -> Self {
        let () = Self::VALID;

        let mut limbs = [0u64; LIMBS];
        limbs[0] = 1;

        Self { limbs }
    }

    /// From the least significant limb, with the rest zero.
    pub const fn from_limb(value: u64) -> Self {
        let () = Self::VALID;

        let mut limbs = [0u64; LIMBS];
        limbs[0] = value;

        Self { limbs }
    }

    /// The limbs, least significant first.
    pub const fn limbs(&self) -> &[u64; LIMBS] {
        &self.limbs
    }

    /// From limbs, least significant first.
    pub const fn from_limbs(limbs: [u64; LIMBS]) -> Self {
        Self { limbs }
    }

    /// Whether every bit is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|limb| *limb == 0)
    }

    /// Whether this is one.
    pub fn is_one(&self) -> bool {
        self.limbs[0] == 1 && self.limbs[1..].iter().all(|limb| *limb == 0)
    }

    /// One bit, counting from zero at the least significant end.
    ///
    /// `false` for a position beyond the width, rather than a panic: division
    /// walks every bit and a bound check per bit would be noise.
    pub fn bit(&self, index: u32) -> bool {
        if index >= Self::BITS {
            return false;
        }

        let limb: usize = (index / LIMB_BITS) as usize;
        let offset: u32 = index % LIMB_BITS;

        self.limbs[limb] >> offset & 1 == 1
    }

    /// Sets one bit, ignoring a position beyond the width.
    pub fn set_bit(&mut self, index: u32, value: bool) {
        if index >= Self::BITS {
            return;
        }

        let limb: usize = (index / LIMB_BITS) as usize;
        let offset: u32 = index % LIMB_BITS;

        if value {
            self.limbs[limb] |= 1 << offset;
        } else {
            self.limbs[limb] &= !(1 << offset);
        }
    }

    /// How many bits the value needs, which is zero for zero.
    pub fn bit_length(&self) -> u32 {
        Self::BITS - self.leading_zeros()
    }

    /// How many zero bits sit above the most significant one bit.
    pub fn leading_zeros(&self) -> u32 {
        let mut count: u32 = 0;

        // From the top limb down, stopping at the first that holds anything.
        for limb in self.limbs.iter().rev() {
            if *limb == 0 {
                count += LIMB_BITS;
                continue;
            }

            return count + limb.leading_zeros();
        }

        count
    }

    /// The sum, and whether it carried out of the top.
    pub fn overflowing_add(&self, other: &Self) -> (Self, bool) {
        let mut limbs = [0u64; LIMBS];
        let mut carry: u64 = 0;

        for ((limb, left), right) in limbs.iter_mut().zip(&self.limbs).zip(&other.limbs) {
            // Widened, so that the sum of two limbs and a carry cannot wrap
            // before the carry is taken out of it.
            let sum: u128 = *left as u128 + *right as u128 + carry as u128;

            *limb = sum as u64;
            carry = (sum >> LIMB_BITS) as u64;
        }

        (Self { limbs }, carry != 0)
    }

    /// The difference, and whether it borrowed past the bottom.
    pub fn overflowing_sub(&self, other: &Self) -> (Self, bool) {
        let mut limbs = [0u64; LIMBS];
        let mut borrow: u64 = 0;

        for ((limb, left), right) in limbs.iter_mut().zip(&self.limbs).zip(&other.limbs) {
            // Two steps, because a borrow can come out of either of them: the
            // limb subtraction itself, or taking the previous borrow off.
            let (partial, first) = left.overflowing_sub(*right);
            let (value, second) = partial.overflowing_sub(borrow);

            *limb = value;
            borrow = u64::from(first || second);
        }

        (Self { limbs }, borrow != 0)
    }

    /// The sum, wrapping past the top.
    pub fn wrapping_add(&self, other: &Self) -> Self {
        self.overflowing_add(other).0
    }

    /// The difference, wrapping past the bottom.
    pub fn wrapping_sub(&self, other: &Self) -> Self {
        self.overflowing_sub(other).0
    }

    /// The sum, or `None` when it does not fit.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        let (sum, carried) = self.overflowing_add(other);

        (!carried).then_some(sum)
    }

    /// The difference, or `None` when it would go below zero.
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        let (difference, borrowed) = self.overflowing_sub(other);

        (!borrowed).then_some(difference)
    }

    /// The product, and whether anything was lost off the top.
    ///
    /// Schoolbook multiplication: every pair of limbs, each partial product
    /// widened to 128 bits so the carry can be taken out of it. A pair whose
    /// result would land beyond the last limb is not computed — it is only noted
    /// as an overflow.
    pub fn overflowing_mul(&self, other: &Self) -> (Self, bool) {
        let mut limbs = [0u64; LIMBS];
        let mut overflow: bool = false;

        for left in 0..LIMBS {
            if self.limbs[left] == 0 {
                continue;
            }

            let mut carry: u64 = 0;

            for right in 0..LIMBS {
                let index: usize = left + right;

                if index >= LIMBS {
                    // Past the top: anything non-zero here is lost.
                    overflow |= other.limbs[right] != 0;
                    continue;
                }

                let product: u128 = self.limbs[left] as u128 * other.limbs[right] as u128
                    + limbs[index] as u128
                    + carry as u128;

                limbs[index] = product as u64;
                carry = (product >> LIMB_BITS) as u64;
            }

            // A carry out of the last limb is also lost.
            overflow |= carry != 0;
        }

        (Self { limbs }, overflow)
    }

    /// The product, wrapping past the top.
    pub fn wrapping_mul(&self, other: &Self) -> Self {
        self.overflowing_mul(other).0
    }

    /// The product, or `None` when it does not fit.
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        let (product, overflowed) = self.overflowing_mul(other);

        (!overflowed).then_some(product)
    }

    /// The exact product, as `(low, high)` halves — nothing is ever lost.
    ///
    /// Two `LIMBS`-limb values multiply to `2 × LIMBS` limbs, and `WideUint<2 *
    /// LIMBS>` cannot be named on stable Rust. Returning the halves separately
    /// says the same thing without the const arithmetic, and is how
    /// [`WideFloat`](crate::math::WideFloat) multiplies significands: the new
    /// significand comes from `high`, and `low` decides which way to round.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::WideUint;
    ///
    /// let big: WideUint<2> = WideUint::from(u128::MAX);
    /// let (low, high) = big.widening_mul(&big);
    ///
    /// // (2¹²⁸ − 1)² = 2²⁵⁶ − 2¹²⁹ + 1, so the low half is 1 and the rest is high.
    /// assert_eq!(low, WideUint::one());
    /// assert_eq!(high, big.wrapping_sub(&WideUint::one()));
    /// ```
    pub fn widening_mul(&self, other: &Self) -> (Self, Self) {
        let mut low = [0u64; LIMBS];
        let mut high = [0u64; LIMBS];

        /// One digit of the `2 × LIMBS`-limb accumulator.
        macro_rules! digit {
            ($index:expr) => {
                if $index < LIMBS {
                    low[$index]
                } else {
                    high[$index - LIMBS]
                }
            };
        }

        macro_rules! set_digit {
            ($index:expr, $value:expr) => {
                if $index < LIMBS {
                    low[$index] = $value;
                } else {
                    high[$index - LIMBS] = $value;
                }
            };
        }

        for left in 0..LIMBS {
            if self.limbs[left] == 0 {
                continue;
            }

            let mut carry: u64 = 0;

            for right in 0..LIMBS {
                let index: usize = left + right;
                let product: u128 = self.limbs[left] as u128 * other.limbs[right] as u128
                    + digit!(index) as u128
                    + carry as u128;

                set_digit!(index, product as u64);
                carry = (product >> LIMB_BITS) as u64;
            }

            // The carry out of this row lands above it, always in the high half
            // because the row ended at `left + LIMBS - 1`.
            let mut index: usize = left + LIMBS;

            while carry != 0 && index < 2 * LIMBS {
                let sum: u128 = digit!(index) as u128 + carry as u128;

                set_digit!(index, sum as u64);
                carry = (sum >> LIMB_BITS) as u64;
                index += 1;
            }
        }

        (Self { limbs: low }, Self { limbs: high })
    }

    /// Shifted up, wrapping bits off the top.
    pub fn wrapping_shl(&self, places: u32) -> Self {
        if places >= Self::BITS {
            return Self::ZERO;
        }

        let whole: usize = (places / LIMB_BITS) as usize;
        let part: u32 = places % LIMB_BITS;
        let mut limbs = [0u64; LIMBS];

        for index in (0..LIMBS).rev() {
            if index < whole {
                break;
            }

            let from: usize = index - whole;
            let mut value: u64 = self.limbs[from] << part;

            // The bits pushed out of the limb below arrive at the bottom of this
            // one. Guarded, because shifting by the full width is undefined.
            if part > 0 && from > 0 {
                value |= self.limbs[from - 1] >> (LIMB_BITS - part);
            }

            limbs[index] = value;
        }

        Self { limbs }
    }

    /// Shifted down, dropping bits off the bottom.
    pub fn wrapping_shr(&self, places: u32) -> Self {
        if places >= Self::BITS {
            return Self::ZERO;
        }

        let whole: usize = (places / LIMB_BITS) as usize;
        let part: u32 = places % LIMB_BITS;
        let mut limbs = [0u64; LIMBS];

        for (index, limb) in limbs.iter_mut().enumerate() {
            // The source limb sits `whole` places above the destination, which is
            // why this cannot be a plain zip.
            let from: usize = index + whole;

            if from >= LIMBS {
                break;
            }

            let mut value: u64 = self.limbs[from] >> part;

            if part > 0 && from + 1 < LIMBS {
                value |= self.limbs[from + 1] << (LIMB_BITS - part);
            }

            *limb = value;
        }

        Self { limbs }
    }

    /// How many limbs the value actually uses, which is zero for zero.
    fn significant_limbs(&self) -> usize {
        for index in (0..LIMBS).rev() {
            if self.limbs[index] != 0 {
                return index + 1;
            }
        }

        0
    }

    /// The quotient and remainder of a division by one limb.
    ///
    /// Short division: one 128-by-64-bit step per limb, from the top down,
    /// carrying the remainder into the next. Every step is a single hardware
    /// division, so this is as fast as division gets and worth having as its own
    /// path — a one-limb divisor is the common case for the decimal printing in
    /// [`fmt::Display`], which divides by 10¹⁹ repeatedly.
    fn div_rem_limb(&self, divisor: u64) -> (Self, u64) {
        let mut quotient = [0u64; LIMBS];
        let mut remainder: u64 = 0;

        for index in (0..LIMBS).rev() {
            // The remainder is always below the divisor, so this pair divided by
            // the divisor is always below `2⁶⁴` and the digit fits.
            let current: u128 =
                (remainder as u128) << LIMB_BITS | self.limbs[index] as u128;

            quotient[index] = (current / divisor as u128) as u64;
            remainder = (current % divisor as u128) as u64;
        }

        (Self { limbs: quotient }, remainder)
    }

    /// The quotient and remainder, or `None` when dividing by zero.
    ///
    /// Knuth's Algorithm D (*The Art of Computer Programming* vol. 2, §4.3.1):
    /// schoolbook long division in base 2⁶⁴, producing one 64-bit quotient digit
    /// per step instead of one bit. For a four-limb value that is four steps
    /// rather than 256.
    ///
    /// # How a digit is found
    ///
    /// The quotient digit is *estimated* from the top two limbs of the running
    /// remainder over the top limb of the divisor, then corrected. Normalising the
    /// divisor so its top bit is set (step D1) is what bounds the error: with that,
    /// the estimate is never more than **two** too large, and the two-limb test in
    /// step D3 brings it down to at most one. The remaining case is caught after
    /// the fact — if the multiply-and-subtract goes negative, the digit was one too
    /// large, so it is decremented and the divisor added back (step D5).
    ///
    /// That add-back happens for roughly one division in 2⁶³, which is exactly why
    /// this algorithm is hard to test: no realistic amount of random input will
    /// reach it. [`WideUint::div_rem_binary`] is kept as an independent reference
    /// and both are checked against `u128` over the whole overlapping range, with
    /// inputs constructed specifically to force the correction.
    ///
    /// # Cost
    ///
    /// `O(n·m)` limb operations for an n-limb divisor and m-limb quotient, against
    /// the binary method's `O(bits × n)`. Measured on `WideUint<4>`, this is about
    /// 20× faster; see the module documentation.
    pub fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        let divisor_limbs: usize = divisor.significant_limbs();

        if divisor_limbs == 0 {
            return None;
        }

        if self < divisor {
            return Some((Self::ZERO, *self));
        }

        // A single-limb divisor cannot use the estimation step, which reads two
        // limbs of the divisor — and does not need to.
        if divisor_limbs == 1 {
            let (quotient, remainder) = self.div_rem_limb(divisor.limbs[0]);

            return Some((quotient, Self::from_limb(remainder)));
        }

        let numerator_limbs: usize = self.significant_limbs();
        // The number of quotient digits to produce, less one. `self >= divisor`
        // was established above, so this cannot go negative.
        let steps: usize = numerator_limbs - divisor_limbs;

        // D1: normalise, so the divisor's top limb has its high bit set. This is
        // what makes the estimate below accurate to within two.
        let shift: u32 = divisor.limbs[divisor_limbs - 1].leading_zeros();
        let divisor: Self = divisor.wrapping_shl(shift);

        let mut remainder: Self = self.wrapping_shl(shift);
        // Shifting the numerator up can push bits past the top limb. They belong
        // to the working remainder, so they are kept beside the array: `[u64;
        // LIMBS + 1]` is the same const-arithmetic wall as everywhere else here.
        let mut overflow: u64 = if shift > 0 && numerator_limbs == LIMBS {
            self.limbs[LIMBS - 1] >> (LIMB_BITS - shift)
        } else {
            0
        };

        /// One digit of the working remainder, treating `overflow` as the limb
        /// just past the end.
        macro_rules! digit {
            ($index:expr) => {
                if $index == LIMBS {
                    overflow
                } else {
                    remainder.limbs[$index]
                }
            };
        }

        macro_rules! set_digit {
            ($index:expr, $value:expr) => {
                if $index == LIMBS {
                    overflow = $value;
                } else {
                    remainder.limbs[$index] = $value;
                }
            };
        }

        let base: u128 = 1 << LIMB_BITS;
        let top: u64 = divisor.limbs[divisor_limbs - 1];
        let second: u64 = divisor.limbs[divisor_limbs - 2];
        let mut quotient = [0u64; LIMBS];

        // D2: one quotient digit per step, most significant first.
        for step in (0..=steps).rev() {
            let high: usize = step + divisor_limbs;

            // D3: estimate the digit from the top two limbs over the divisor's top
            // limb. This can come out as large as `2⁶⁴`, hence the u128.
            let pair: u128 = (digit!(high) as u128) << LIMB_BITS | digit!(high - 1) as u128;
            let mut estimate: u128 = pair / top as u128;
            let mut rest: u128 = pair % top as u128;

            // Bring the estimate down until the divisor's second limb agrees with
            // it too. This is the test that reduces the error from two to one.
            while estimate >= base
                || estimate * second as u128 > (rest << LIMB_BITS) + digit!(high - 2) as u128
            {
                estimate -= 1;
                rest += top as u128;

                if rest >= base {
                    break;
                }
            }

            // D4: subtract the divisor times the estimate from the remainder.
            let mut carry: u128 = 0;
            let mut borrow: i128 = 0;

            for index in 0..divisor_limbs {
                let product: u128 = estimate * divisor.limbs[index] as u128 + carry;
                carry = product >> LIMB_BITS;

                // Signed, because this digit can go below zero and borrow from the
                // next. Truncating the negative value to `u64` gives the correct
                // two's-complement digit.
                let difference: i128 =
                    digit!(step + index) as i128 - (product & (base - 1)) as i128 - borrow;

                set_digit!(step + index, difference as u64);
                borrow = i128::from(difference < 0);
            }

            let difference: i128 = digit!(high) as i128 - carry as i128 - borrow;
            set_digit!(high, difference as u64);

            // D5: a negative result means the estimate was one too large after
            // all. Undo it by adding the divisor back — the rare branch that no
            // amount of random testing will reach.
            if difference < 0 {
                estimate -= 1;

                let mut carry: u128 = 0;

                for index in 0..divisor_limbs {
                    let sum: u128 =
                        digit!(step + index) as u128 + divisor.limbs[index] as u128 + carry;

                    set_digit!(step + index, sum as u64);
                    carry = sum >> LIMB_BITS;
                }

                // This carry cancels the borrow that took the digit negative.
                set_digit!(high, (digit!(high) as u128 + carry) as u64);
            }

            // D6.
            quotient[step] = estimate as u64;
        }

        // D8: the remainder is what is left in the low limbs, un-normalised. It is
        // below the divisor and so never reaches the overflow digit.
        for index in divisor_limbs..LIMBS {
            remainder.limbs[index] = 0;
        }

        Some((Self { limbs: quotient }, remainder.wrapping_shr(shift)))
    }

    /// The quotient and remainder by binary long division — the reference
    /// implementation, kept to check [`WideUint::div_rem`] against.
    ///
    /// The remainder is shifted up one bit at a time, taking the next bit of the
    /// numerator, and the divisor subtracted whenever it fits: one compare and at
    /// most one subtraction per bit. Far slower than Algorithm D, and far easier to
    /// be sure of, which is the point of keeping it. It is not `#[cfg(test)]`
    /// because the tests are integration tests and could not then reach it.
    ///
    /// # Why the overflow of the shift is watched
    ///
    /// The remainder is always below the divisor at the top of a step, so after
    /// shifting and taking a bit it is below twice the divisor — which can be one
    /// bit wider than the type. When that bit is pushed out, the true remainder
    /// exceeds anything representable, so it certainly exceeds the divisor, and
    /// the subtraction is correct modulo the width. Missing this is the classic
    /// way to get a wrong answer for a divisor above half the range.
    pub fn div_rem_binary(&self, divisor: &Self) -> Option<(Self, Self)> {
        if divisor.is_zero() {
            return None;
        }

        if self < divisor {
            return Some((Self::ZERO, *self));
        }

        let mut quotient: Self = Self::ZERO;
        let mut remainder: Self = Self::ZERO;

        for index in (0..self.bit_length()).rev() {
            let (shifted, carried) = remainder.shifted_up_one();
            remainder = shifted;
            remainder.set_bit(0, self.bit(index));

            if carried || remainder >= *divisor {
                remainder = remainder.wrapping_sub(divisor);
                quotient.set_bit(index, true);
            }
        }

        Some((quotient, remainder))
    }

    /// The quotient, or `None` for a zero divisor.
    pub fn checked_div(&self, divisor: &Self) -> Option<Self> {
        self.div_rem(divisor).map(|(quotient, _)| quotient)
    }

    /// The remainder, or `None` for a zero divisor.
    pub fn checked_rem(&self, divisor: &Self) -> Option<Self> {
        self.div_rem(divisor).map(|(_, remainder)| remainder)
    }

    /// The largest value whose square does not exceed this one.
    ///
    /// Bit by bit from the top: the answer is built one bit at a time, keeping
    /// each bit whose square still fits. Exact, and no floating point anywhere
    /// near it.
    pub fn integer_sqrt(&self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }

        let mut root: Self = Self::ZERO;

        // The root needs at most half as many bits as the value.
        for index in (0..=self.bit_length() / 2).rev() {
            let mut candidate: Self = root;
            candidate.set_bit(index, true);

            // Overflowing means the square is far past the value, so the bit
            // does not belong in the answer.
            let (square, overflowed) = candidate.overflowing_mul(&candidate);

            if !overflowed && square <= *self {
                root = candidate;
            }
        }

        root
    }

    /// Shifted up one bit, with the bit pushed off the top.
    fn shifted_up_one(&self) -> (Self, bool) {
        let carried: bool = self.bit(Self::BITS - 1);

        (self.wrapping_shl(1), carried)
    }

    /// The value as a `u128`, or `None` when it needs more room.
    pub fn to_u128(&self) -> Option<u128> {
        if self.limbs.iter().skip(2).any(|limb| *limb != 0) {
            return None;
        }

        let low: u128 = self.limbs[0] as u128;
        let high: u128 = if LIMBS > 1 {
            self.limbs[1] as u128
        } else {
            0
        };

        Some(low | high << LIMB_BITS)
    }

    /// The low 128 bits, whatever else the value holds.
    pub fn as_u128_wrapping(&self) -> u128 {
        let low: u128 = self.limbs[0] as u128;
        let high: u128 = if LIMBS > 1 {
            self.limbs[1] as u128
        } else {
            0
        };

        low | high << LIMB_BITS
    }

    /// The same value in a different width, or `None` when it will not fit.
    ///
    /// How to move between widths: there is no way to make `LIMBS` arithmetic
    /// work in a type parameter, so widening and narrowing are one method taking
    /// the new width as its own parameter.
    pub fn resize<const OTHER: usize>(&self) -> Option<WideUint<OTHER>> {
        // Anything above the new width must be zero.
        if self.limbs.iter().skip(OTHER).any(|limb| *limb != 0) {
            return None;
        }

        let mut limbs = [0u64; OTHER];

        for (index, limb) in limbs.iter_mut().enumerate().take(LIMBS) {
            *limb = self.limbs[index];
        }

        Some(WideUint::from_limbs(limbs))
    }
}

impl<const LIMBS: usize> Ord for WideUint<LIMBS> {
    /// From the most significant limb down: the first that differs decides.
    fn cmp(&self, other: &Self) -> Ordering {
        for index in (0..LIMBS).rev() {
            match self.limbs[index].cmp(&other.limbs[index]) {
                Ordering::Equal => continue,
                decided => return decided,
            }
        }

        Ordering::Equal
    }
}

impl<const LIMBS: usize> PartialOrd for WideUint<LIMBS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const LIMBS: usize> From<u64> for WideUint<LIMBS> {
    fn from(value: u64) -> Self {
        Self::from_limb(value)
    }
}

impl<const LIMBS: usize> From<u128> for WideUint<LIMBS> {
    /// Takes both halves, or only the low one in a single-limb width.
    fn from(value: u128) -> Self {
        let mut limbs = [0u64; LIMBS];
        limbs[0] = value as u64;

        if LIMBS > 1 {
            limbs[1] = (value >> LIMB_BITS) as u64;
        }

        Self { limbs }
    }
}

/// Arithmetic that panics on overflow, matching what the primitives do in a
/// debug build — and unlike them, in every build.
///
/// The checked and wrapping forms above are the ones to reach for where an
/// overflow is expected; these exist so that ordinary expressions read normally.
impl<const LIMBS: usize> Add for WideUint<LIMBS> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.checked_add(&other).expect("addition overflowed")
    }
}

impl<const LIMBS: usize> Sub for WideUint<LIMBS> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.checked_sub(&other).expect("subtraction went below zero")
    }
}

impl<const LIMBS: usize> Mul for WideUint<LIMBS> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.checked_mul(&other).expect("multiplication overflowed")
    }
}

impl<const LIMBS: usize> AddAssign for WideUint<LIMBS> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const LIMBS: usize> SubAssign for WideUint<LIMBS> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<const LIMBS: usize> MulAssign for WideUint<LIMBS> {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<const LIMBS: usize> Shl<u32> for WideUint<LIMBS> {
    type Output = Self;

    fn shl(self, places: u32) -> Self {
        self.wrapping_shl(places)
    }
}

impl<const LIMBS: usize> Shr<u32> for WideUint<LIMBS> {
    type Output = Self;

    fn shr(self, places: u32) -> Self {
        self.wrapping_shr(places)
    }
}

impl<const LIMBS: usize> Not for WideUint<LIMBS> {
    type Output = Self;

    fn not(self) -> Self {
        Self {
            limbs: self.limbs.map(|limb| !limb),
        }
    }
}

/// Generates the bitwise operators, which are all the same shape.
macro_rules! implement_bitwise {
    ($($trait:ident, $method:ident, $operator:tt);* $(;)?) => {
        $(
            impl<const LIMBS: usize> $trait for WideUint<LIMBS> {
                type Output = Self;

                fn $method(self, other: Self) -> Self {
                    let mut limbs = [0u64; LIMBS];

                    for ((limb, left), right) in
                        limbs.iter_mut().zip(&self.limbs).zip(&other.limbs)
                    {
                        *limb = left $operator right;
                    }

                    Self { limbs }
                }
            }
        )*
    };
}

implement_bitwise!(
    BitAnd, bitand, &;
    BitOr, bitor, |;
    BitXor, bitxor, ^;
);

impl<const LIMBS: usize> Zero for WideUint<LIMBS> {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        WideUint::is_zero(self)
    }
}

impl<const LIMBS: usize> One for WideUint<LIMBS> {
    fn one() -> Self {
        Self::one()
    }

    fn is_one(&self) -> bool {
        WideUint::is_one(self)
    }
}

/// A semiring and no more: an unsigned integer has no additive inverse, exactly
/// as `u64` has none.
impl<const LIMBS: usize> Semiring for WideUint<LIMBS> {}

impl<const LIMBS: usize> fmt::Display for WideUint<LIMBS> {
    /// In decimal, by repeatedly dividing out the largest power of ten that fits
    /// in a limb.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return formatter.write_str("0");
        }

        /// The largest power of ten below `u64::MAX`, so nineteen digits come out
        /// of each division.
        const CHUNK: u64 = 10_000_000_000_000_000_000;

        let divisor: Self = Self::from_limb(CHUNK);
        let mut remaining: Self = *self;
        let mut chunks: Vec<u64> = Vec::new();

        while !remaining.is_zero() {
            let (quotient, remainder) = remaining
                .div_rem(&divisor)
                .expect("the divisor is not zero");

            chunks.push(remainder.limbs[0]);
            remaining = quotient;
        }

        // The most significant chunk has no leading zeros; the rest are padded to
        // nineteen digits so that their zeros are not lost.
        let mut chunks = chunks.into_iter().rev();

        if let Some(first) = chunks.next() {
            write!(formatter, "{first}")?;
        }

        for chunk in chunks {
            write!(formatter, "{chunk:019}")?;
        }

        Ok(())
    }
}

impl<const LIMBS: usize> fmt::Debug for WideUint<LIMBS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self}")
    }
}

impl<const LIMBS: usize> fmt::LowerHex for WideUint<LIMBS> {
    /// Every limb in full, most significant first, which makes the limb boundaries
    /// visible.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for limb in self.limbs.iter().rev() {
            write!(formatter, "{limb:016x}")?;
        }

        Ok(())
    }
}

// ===========================================================================
// Signed
// ===========================================================================

/// A signed integer of `LIMBS × 64` bits, in two's complement.
///
/// # Why two's complement rather than sign and magnitude
///
/// Because addition, subtraction and multiplication are then *bit for bit* the
/// same as the unsigned ones, and need no code of their own. Only comparison,
/// division and printing have to know about the sign. Sign-and-magnitude would
/// invert that: cheap printing, and four cases in every addition.
///
/// The cost is the usual asymmetry: there is one more negative value than
/// positive, so [`WideInt::MIN`] has no positive counterpart and
/// [`WideInt::checked_abs`] returns `None` for it.
///
/// # Example
///
/// ```
/// use voxel_world::math::WideInt;
///
/// type I256 = WideInt<4>;
///
/// let big: I256 = WideInt::from(i128::MIN);
/// let negated = big.checked_neg().expect("256 bits has room for it");
///
/// assert!(big.is_negative());
/// assert!(!negated.is_negative());
///
/// // Division truncates towards zero, as Rust's own does.
/// let (quotient, remainder) = WideInt::<4>::from(-7i64)
///     .div_rem(&WideInt::from(2i64))
///     .expect("not by zero");
///
/// assert_eq!(quotient, WideInt::from(-3i64));
/// assert_eq!(remainder, WideInt::from(-1i64));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WideInt<const LIMBS: usize> {
    /// The two's-complement bit pattern, held in the unsigned type so that the
    /// arithmetic can be shared.
    bits: WideUint<LIMBS>,
}

impl<const LIMBS: usize> WideInt<LIMBS> {
    /// How many bits the type holds, the topmost being the sign.
    pub const BITS: u32 = WideUint::<LIMBS>::BITS;

    /// Zero.
    pub const ZERO: Self = Self {
        bits: WideUint::ZERO,
    };

    /// The most negative value, which has no positive counterpart.
    pub const MIN: Self = Self {
        bits: WideUint::from_limbs({
            let mut limbs = [0u64; LIMBS];
            limbs[LIMBS - 1] = 1 << (LIMB_BITS - 1);
            limbs
        }),
    };

    /// The largest value.
    pub const MAX: Self = Self {
        bits: WideUint::from_limbs({
            let mut limbs = [u64::MAX; LIMBS];
            limbs[LIMBS - 1] = u64::MAX >> 1;
            limbs
        }),
    };

    /// One.
    pub const fn one() -> Self {
        Self {
            bits: WideUint::one(),
        }
    }

    /// Whether the sign bit is set.
    pub fn is_negative(&self) -> bool {
        self.bits.bit(Self::BITS - 1)
    }

    /// Whether the value is zero.
    pub fn is_zero(&self) -> bool {
        self.bits.is_zero()
    }

    /// Whether the value is one.
    pub fn is_one(&self) -> bool {
        self.bits.is_one()
    }

    /// The bit pattern, read as unsigned.
    pub const fn to_bits(&self) -> WideUint<LIMBS> {
        self.bits
    }

    /// From a bit pattern read as two's complement.
    pub const fn from_bits(bits: WideUint<LIMBS>) -> Self {
        Self { bits }
    }

    /// The value as an `i128`, or `None` when it needs more room.
    ///
    /// The counterpart of [`WideUint::to_u128`], and the way to get a wide value
    /// back into something a shift or an index can use.
    pub fn to_i128(&self) -> Option<i128> {
        let negative: bool = self.is_negative();
        // Above the low two limbs, a representable value is nothing but sign
        // extension: all ones when negative, all zeros when not.
        let fill: u64 = if negative { u64::MAX } else { 0 };
        let limbs: &[u64; LIMBS] = self.bits.limbs();

        if limbs.iter().skip(2).any(|limb| *limb != fill) {
            return None;
        }

        // A single-limb type has no second limb, so the sign extension supplies it.
        let high: u64 = if LIMBS > 1 { limbs[1] } else { fill };
        let value: i128 = (limbs[0] as u128 | (high as u128) << LIMB_BITS) as i128;

        // The low 128 bits must themselves carry the right sign, or the value
        // needed the limb that was just checked away.
        (value.is_negative() == negative).then_some(value)
    }

    /// The distance from zero, as an unsigned value.
    ///
    /// Unsigned because the magnitude of [`WideInt::MIN`] does not fit in the
    /// signed type — which is the whole reason [`WideInt::checked_abs`] can fail
    /// while this cannot.
    pub fn magnitude(&self) -> WideUint<LIMBS> {
        if self.is_negative() {
            // Negating a two's-complement pattern is inverting and adding one,
            // and for MIN that wraps back to MIN — whose *unsigned* reading is
            // the correct magnitude.
            self.bits.wrapping_sub(&WideUint::ONE_PATTERN).not()
        } else {
            self.bits
        }
    }

    /// The negation, or `None` for [`WideInt::MIN`].
    pub fn checked_neg(&self) -> Option<Self> {
        if *self == Self::MIN {
            return None;
        }

        Some(Self {
            bits: self.bits.not().wrapping_add(&WideUint::one()),
        })
    }

    /// The negation, wrapping for [`WideInt::MIN`].
    pub fn wrapping_neg(&self) -> Self {
        Self {
            bits: self.bits.not().wrapping_add(&WideUint::one()),
        }
    }

    /// The distance from zero, or `None` for [`WideInt::MIN`].
    pub fn checked_abs(&self) -> Option<Self> {
        if self.is_negative() {
            self.checked_neg()
        } else {
            Some(*self)
        }
    }

    /// The sum, wrapping past either end.
    ///
    /// Bit for bit the unsigned addition: that is what two's complement buys.
    pub fn wrapping_add(&self, other: &Self) -> Self {
        Self {
            bits: self.bits.wrapping_add(&other.bits),
        }
    }

    /// The difference, wrapping past either end.
    pub fn wrapping_sub(&self, other: &Self) -> Self {
        Self {
            bits: self.bits.wrapping_sub(&other.bits),
        }
    }

    /// The product, wrapping past either end.
    pub fn wrapping_mul(&self, other: &Self) -> Self {
        Self {
            bits: self.bits.wrapping_mul(&other.bits),
        }
    }

    /// The sum, or `None` when the sign shows it did not fit.
    ///
    /// Overflow in two's complement is exactly this: adding two values of one
    /// sign and arriving at the other.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        let sum: Self = self.wrapping_add(other);

        if self.is_negative() == other.is_negative()
            && sum.is_negative() != self.is_negative()
        {
            return None;
        }

        Some(sum)
    }

    /// The difference, or `None` when it did not fit.
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        match other.checked_neg() {
            Some(negated) => self.checked_add(&negated),
            // Subtracting MIN is adding a value the type cannot hold, which fits
            // only when this is negative.
            None => self.is_negative().then(|| self.wrapping_sub(other)),
        }
    }

    /// The product, or `None` when it did not fit.
    ///
    /// Worked out on magnitudes, where the unsigned overflow check applies, and
    /// the sign put back afterwards.
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        let product: WideUint<LIMBS> = self.magnitude().checked_mul(&other.magnitude())?;
        let negative: bool = self.is_negative() != other.is_negative();

        Self::from_magnitude(product, negative)
    }

    /// The quotient and remainder, or `None` for a zero divisor or for
    /// [`WideInt::MIN`] divided by minus one.
    ///
    /// Truncates towards zero and gives the remainder the numerator's sign,
    /// which is what Rust's own integer division does.
    pub fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        let (quotient, remainder) = self.magnitude().div_rem(&divisor.magnitude())?;

        let negative: bool = self.is_negative() != divisor.is_negative();
        let quotient: Self = Self::from_magnitude(quotient, negative)?;
        // The remainder takes the numerator's sign, so `-7 % 2` is `-1`.
        let remainder: Self = Self::from_magnitude(remainder, self.is_negative())?;

        Some((quotient, remainder))
    }

    /// A value from its magnitude and sign, or `None` when it does not fit.
    fn from_magnitude(magnitude: WideUint<LIMBS>, negative: bool) -> Option<Self> {
        let value = Self { bits: magnitude };

        if negative {
            // `MIN` is the one negative value whose magnitude has the sign bit
            // set, and it is representable.
            if magnitude == Self::MIN.bits {
                return Some(Self::MIN);
            }

            if value.is_negative() {
                return None;
            }

            return value.checked_neg();
        }

        (!value.is_negative()).then_some(value)
    }
}

impl<const LIMBS: usize> WideUint<LIMBS> {
    /// One, as a constant, for the places that need it in a `const` context.
    ///
    /// [`WideUint::one`] is a `const fn` and cannot be named where a constant is
    /// required; this can.
    const ONE_PATTERN: Self = Self {
        limbs: {
            let mut limbs = [0u64; LIMBS];
            limbs[0] = 1;
            limbs
        },
    };
}

impl<const LIMBS: usize> Ord for WideInt<LIMBS> {
    /// A negative value is below every positive one; within a sign the unsigned
    /// order of the bit pattern is already correct.
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_negative(), other.is_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => self.bits.cmp(&other.bits),
        }
    }
}

impl<const LIMBS: usize> PartialOrd for WideInt<LIMBS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const LIMBS: usize> From<i64> for WideInt<LIMBS> {
    /// Sign-extended, so that the value is the same in every width.
    fn from(value: i64) -> Self {
        Self::from(value as i128)
    }
}

impl<const LIMBS: usize> From<i128> for WideInt<LIMBS> {
    fn from(value: i128) -> Self {
        let magnitude: WideUint<LIMBS> = WideUint::from(value.unsigned_abs());

        if value < 0 {
            Self { bits: magnitude }.wrapping_neg()
        } else {
            Self { bits: magnitude }
        }
    }
}

impl<const LIMBS: usize> Add for WideInt<LIMBS> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.checked_add(&other).expect("addition overflowed")
    }
}

impl<const LIMBS: usize> Sub for WideInt<LIMBS> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.checked_sub(&other).expect("subtraction overflowed")
    }
}

impl<const LIMBS: usize> Mul for WideInt<LIMBS> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.checked_mul(&other).expect("multiplication overflowed")
    }
}

impl<const LIMBS: usize> Neg for WideInt<LIMBS> {
    type Output = Self;

    fn neg(self) -> Self {
        self.checked_neg().expect("the most negative value has no negation")
    }
}

impl<const LIMBS: usize> AddAssign for WideInt<LIMBS> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const LIMBS: usize> SubAssign for WideInt<LIMBS> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<const LIMBS: usize> MulAssign for WideInt<LIMBS> {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<const LIMBS: usize> Zero for WideInt<LIMBS> {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        WideInt::is_zero(self)
    }
}

impl<const LIMBS: usize> One for WideInt<LIMBS> {
    fn one() -> Self {
        Self::one()
    }

    fn is_one(&self) -> bool {
        WideInt::is_one(self)
    }
}

impl<const LIMBS: usize> Semiring for WideInt<LIMBS> {}
impl<const LIMBS: usize> Ring for WideInt<LIMBS> {}
impl<const LIMBS: usize> CommutativeRing for WideInt<LIMBS> {}

/// Division with a remainder, so greatest common divisors come from the trait.
impl<const LIMBS: usize> EuclideanRing for WideInt<LIMBS> {
    /// The magnitude, which is unsigned and so compares the right way.
    type Size = WideUint<LIMBS>;

    fn euclidean_size(&self) -> WideUint<LIMBS> {
        self.magnitude()
    }

    fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        WideInt::div_rem(self, divisor)
    }

    /// Non-negative, which is the usual convention.
    fn gcd_normalised(&self, other: &Self) -> Self {
        let divisor: Self = self.gcd(other);

        divisor.checked_abs().unwrap_or(divisor)
    }
}

impl<const LIMBS: usize> fmt::Display for WideInt<LIMBS> {
    /// A sign where there is one, then the magnitude.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_negative() {
            write!(formatter, "-{}", self.magnitude())
        } else {
            write!(formatter, "{}", self.bits)
        }
    }
}

impl<const LIMBS: usize> fmt::Debug for WideInt<LIMBS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self}")
    }
}
