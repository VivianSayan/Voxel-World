//! Arithmetic that wraps: `Z/nZ`, and the prime fields among them.
//!
//! [`Modulo`] is the integers modulo some `N`, a commutative ring for every `N`
//! above zero. [`PrimeField`] is the same arithmetic where `N` is prime, which is
//! the case where every value but zero can be divided by — and it checks that
//! primality **while compiling**, so a composite modulus is a build error rather
//! than a division that quietly fails.
//!
//! # The modulus is in the type
//!
//! `Modulo<7>` and `Modulo<11>` are different types, so nothing can add a value
//! of one to a value of the other. Carrying the modulus in each value instead
//! would make that a run-time check at best and a silent wrong answer at worst,
//! and the modulus is almost always known where the code is written.
//!
//! The cost is that a modulus read from a file cannot be used. If that is ever
//! needed it wants a separate type that carries its modulus, and the two can sit
//! beside each other.
//!
//! # Storage
//!
//! Values are held as `u64` and multiplied through `u128`, so any modulus up to
//! `u64::MAX` works without overflow. Being generic over the storage would buy
//! very little: the modulus bounds the values, so a smaller type would only save
//! space in an array of them.

use crate::math::traits::{CommutativeRing, EuclideanRing, Field, One, Ring, Semiring, Zero};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The witnesses that make [`is_prime`] decide rather than guess.
///
/// Miller–Rabin is a probable-prime test for an arbitrary witness, but testing
/// against every prime up to 37 settles every `n` below 3.3 × 10²⁴ — far above
/// `u64::MAX` — so over this range it is a proof, not a probability.
const WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Whether `candidate` is prime, by deterministic Miller–Rabin.
///
/// A `const fn`, so [`PrimeField`] can refuse a composite modulus at compile
/// time.
///
/// # Why not trial division, or a sieve
///
/// Trial division needs up to `√n` steps, which for a modulus near `u64::MAX` is
/// two billion of them — slow enough to notice in a build, and most of those
/// steps divide by a composite that cannot be a factor anyway.
///
/// A sieve does not help either, despite being the right tool for *listing*
/// primes. To test a single `n` it would have to sieve up to `√n` to get the
/// divisors worth trying: for a 64-bit modulus that is 2³² entries, half a
/// gigabyte, to answer one question. Sieving is worth it when many numbers are
/// asked about and the bound is small; this is the opposite case.
///
/// Miller–Rabin instead needs one modular exponentiation per witness — about
/// sixty squarings each — so around eight hundred multiplications settle any
/// `u64`, whatever its size. That is why the cost barely grows with the modulus
/// where trial division's doubles with every two bits.
///
/// # How it decides
///
/// Write `n - 1 = d · 2ʳ` with `d` odd. If `n` is prime then for any witness `a`,
/// Fermat gives `aⁿ⁻¹ ≡ 1`, and since the only square roots of one modulo a prime
/// are `±1`, the sequence `aᵈ, a²ᵈ, a⁴ᵈ, …` must reach 1 through `-1` — or start
/// there. A witness for which it does not proves `n` composite.
pub const fn is_prime(candidate: u64) -> bool {
    if candidate < 2 {
        return false;
    }

    // The witnesses double as trial divisors, which settles every small
    // candidate before the main test and guarantees each witness is below it.
    let mut index: usize = 0;

    while index < WITNESSES.len() {
        if candidate == WITNESSES[index] {
            return true;
        }

        if candidate.is_multiple_of(WITNESSES[index]) {
            return false;
        }

        index += 1;
    }

    // n - 1 = odd * 2^power.
    let mut odd: u64 = candidate - 1;
    let mut power: u32 = 0;

    while odd.is_multiple_of(2) {
        odd /= 2;
        power += 1;
    }

    let mut index: usize = 0;

    while index < WITNESSES.len() {
        if !survives(candidate, WITNESSES[index], odd, power) {
            return false;
        }

        index += 1;
    }

    true
}

/// Whether `candidate` survives one Miller–Rabin witness.
const fn survives(candidate: u64, witness: u64, odd: u64, power: u32) -> bool {
    let mut value: u64 = power_mod(witness, odd, candidate);

    if value == 1 || value == candidate - 1 {
        return true;
    }

    let mut step: u32 = 1;

    while step < power {
        value = multiply_mod(value, value, candidate);

        if value == candidate - 1 {
            return true;
        }

        step += 1;
    }

    false
}

/// `base ^ exponent mod modulus`, by repeated squaring.
const fn power_mod(base: u64, exponent: u64, modulus: u64) -> u64 {
    let mut result: u64 = 1;
    let mut base: u64 = base % modulus;
    let mut exponent: u64 = exponent;

    while exponent > 0 {
        if exponent & 1 == 1 {
            result = multiply_mod(result, base, modulus);
        }

        base = multiply_mod(base, base, modulus);
        exponent >>= 1;
    }

    result
}

/// `left * right mod modulus`, widened so the product cannot wrap.
const fn multiply_mod(left: u64, right: u64, modulus: u64) -> u64 {
    ((left as u128 * right as u128) % modulus as u128) as u64
}

// ---------------------------------------------------------------------------
// Z/nZ
// ---------------------------------------------------------------------------

/// An integer modulo `N`, kept reduced.
///
/// # Example
///
/// ```
/// use voxel_world::math::Modulo;
/// use voxel_world::math::traits::Semiring;
///
/// type Clock = Modulo<12>;
///
/// let ten: Clock = Modulo::new(10);
/// let five: Clock = Modulo::new(5);
///
/// assert_eq!((ten + five).value(), 3, "ten o'clock plus five hours");
///
/// // Powers wrap too, by repeated squaring from the trait.
/// assert_eq!(Modulo::<7>::new(3).power(6).value(), 1);
/// ```
///
/// # Which values can be divided by
///
/// A value has a multiplicative inverse exactly when it shares no factor with
/// `N`, so `Z/12Z` can divide by 5 but not by 4. [`Modulo::inverse`] answers
/// `None` for the rest, which is why this is a ring rather than a
/// [`Field`]. Where every non-zero value should be
/// invertible, use [`PrimeField`] and let the compiler check the modulus.
///
/// # Invariant
///
/// The held value is always below `N`, so two equal values are the same value —
/// which is what lets `PartialEq` be derived and mean what it should.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Modulo<const N: u64>(u64);

impl<const N: u64> Modulo<N> {
    /// Fails the build for a modulus of zero, which has no arithmetic.
    const VALID: () = assert!(N > 0, "a modulus of zero has no residues");

    /// The value of `value` in this ring.
    pub const fn new(value: u64) -> Self {
        let () = Self::VALID;

        Self(value % N)
    }

    /// The value of a possibly negative integer in this ring.
    ///
    /// `-1` in `Z/7Z` is 6: the residue is taken the mathematician's way, never
    /// negative, rather than the way `%` truncates.
    pub const fn from_signed(value: i64) -> Self {
        let () = Self::VALID;

        Self(value.rem_euclid(N as i64) as u64)
    }

    /// The representative held, always in `0..N`.
    pub const fn value(self) -> u64 {
        self.0
    }

    /// The modulus, which is the type's own parameter.
    pub const fn modulus() -> u64 {
        N
    }

    /// The value this multiplies with to give one, or `None` when it shares a
    /// factor with the modulus.
    ///
    /// By the extended Euclidean algorithm, which finds `x` with
    /// `value * x + N * y = gcd(value, N)` — an inverse exactly when that
    /// greatest common divisor is one.
    pub fn inverse(self) -> Option<Self> {
        // Widened to i128 so the coefficients, which may go negative and grow,
        // cannot wrap.
        let (mut previous_remainder, mut remainder): (i128, i128) = (self.0 as i128, N as i128);
        let (mut previous_coefficient, mut coefficient): (i128, i128) = (1, 0);

        while remainder != 0 {
            let quotient: i128 = previous_remainder / remainder;

            (previous_remainder, remainder) =
                (remainder, previous_remainder - quotient * remainder);
            (previous_coefficient, coefficient) =
                (coefficient, previous_coefficient - quotient * coefficient);
        }

        if previous_remainder != 1 {
            return None;
        }

        Some(Self(previous_coefficient.rem_euclid(N as i128) as u64))
    }

    /// This divided by `divisor`, or `None` when the divisor has no inverse.
    pub fn divide(self, divisor: Self) -> Option<Self> {
        Some(self * divisor.inverse()?)
    }

    /// Zero.
    pub const ZERO: Self = Self(0);

    /// One, which is zero in `Z/1Z`, where the two coincide.
    pub const ONE: Self = Self(1 % N);

    /// Whether this ring is a field, which is to say whether `N` is prime.
    ///
    /// Worked out while compiling, so it costs nothing to ask.
    pub const IS_FIELD: bool = is_prime(N);

    /// Whether this value can be divided by.
    pub fn is_invertible(self) -> bool {
        self.inverse().is_some()
    }
}

impl<const N: u64> Add for Modulo<N> {
    type Output = Self;

    /// Widened, since two values below `N` can sum past `u64::MAX` when `N` is
    /// large.
    fn add(self, other: Self) -> Self {
        Self(((self.0 as u128 + other.0 as u128) % N as u128) as u64)
    }
}

impl<const N: u64> Sub for Modulo<N> {
    type Output = Self;

    /// `a - b` as `a + (N - b)`, which stays in the unsigned range throughout.
    fn sub(self, other: Self) -> Self {
        Self(((self.0 as u128 + (N - other.0) as u128) % N as u128) as u64)
    }
}

impl<const N: u64> Mul for Modulo<N> {
    type Output = Self;

    /// Widened, since a product of two values near `N` needs twice the bits.
    fn mul(self, other: Self) -> Self {
        Self(((self.0 as u128 * other.0 as u128) % N as u128) as u64)
    }
}

impl<const N: u64> Neg for Modulo<N> {
    type Output = Self;

    /// `N - a`, and zero for zero: the additive inverse that unsigned storage
    /// cannot express on its own, which is why `Z/nZ` is a ring while `u64` is
    /// only a semiring.
    fn neg(self) -> Self {
        Self((N - self.0) % N)
    }
}

impl<const N: u64> AddAssign for Modulo<N> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const N: u64> SubAssign for Modulo<N> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<const N: u64> MulAssign for Modulo<N> {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<const N: u64> Zero for Modulo<N> {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl<const N: u64> One for Modulo<N> {
    fn one() -> Self {
        Self::ONE
    }

    fn is_one(&self) -> bool {
        self.0 == 1 % N
    }
}

impl<const N: u64> Semiring for Modulo<N> {}
impl<const N: u64> Ring for Modulo<N> {}
impl<const N: u64> CommutativeRing for Modulo<N> {}

impl<const N: u64> fmt::Display for Modulo<N> {
    /// The representative and its modulus, since the value alone is ambiguous.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} (mod {N})", self.0)
    }
}

impl<const N: u64> From<u64> for Modulo<N> {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl<const N: u64> From<i64> for Modulo<N> {
    fn from(value: i64) -> Self {
        Self::from_signed(value)
    }
}

// ---------------------------------------------------------------------------
// GF(p)
// ---------------------------------------------------------------------------

/// The field of `P` elements, for prime `P`.
///
/// `GF(p)`, written this way so that the one thing that makes it a field — `P`
/// being prime — is checked rather than assumed.
///
/// # The primality is a build error, not a run-time one
///
/// [`is_prime`] is a `const fn`, so `PrimeField<9>` fails to compile:
///
/// ```compile_fail
/// use voxel_world::math::PrimeField;
///
/// // 9 = 3 * 3, so 3 has no inverse and this is no field.
/// let _: PrimeField<9> = PrimeField::new(4);
/// ```
///
/// That is the difference from [`Modulo`], which accepts any modulus and answers
/// `None` from `inverse` where it must. Here every value but zero is invertible,
/// so the [`Field`] implementation can promise what the trait requires.
///
/// # Example
///
/// ```
/// use voxel_world::math::PrimeField;
/// use voxel_world::math::traits::{Field, Semiring};
///
/// type F7 = PrimeField<7>;
///
/// let three: F7 = PrimeField::new(3);
///
/// // Every non-zero value divides.
/// assert_eq!((three * three.inverse().unwrap()).value(), 1);
/// assert_eq!(F7::new(0).inverse(), None, "except zero");
///
/// // Fermat's little theorem, which holds because the modulus is prime.
/// assert_eq!(three.power(6).value(), 1);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrimeField<const P: u64>(Modulo<P>);

impl<const P: u64> PrimeField<P> {
    /// Fails the build for a composite modulus, which is the whole point of the
    /// type.
    const PRIME: () = assert!(is_prime(P), "the modulus of a prime field must be prime");

    /// The value of `value` in this field.
    pub const fn new(value: u64) -> Self {
        let () = Self::PRIME;

        Self(Modulo::new(value))
    }

    /// The value of a possibly negative integer in this field.
    pub const fn from_signed(value: i64) -> Self {
        let () = Self::PRIME;

        Self(Modulo::from_signed(value))
    }

    /// Zero.
    pub const ZERO: Self = Self(Modulo::ZERO);

    /// One.
    pub const ONE: Self = Self(Modulo::ONE);

    /// The representative held, always in `0..P`.
    pub const fn value(self) -> u64 {
        self.0.value()
    }

    /// How many elements the field has, which is its modulus.
    pub const fn order() -> u64 {
        P
    }

    /// This as an element of the ring, dropping the promise of primality.
    pub const fn residue(self) -> Modulo<P> {
        self.0
    }
}

impl<const P: u64> Add for PrimeField<P> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl<const P: u64> Sub for PrimeField<P> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl<const P: u64> Mul for PrimeField<P> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl<const P: u64> Neg for PrimeField<P> {
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<const P: u64> AddAssign for PrimeField<P> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const P: u64> SubAssign for PrimeField<P> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<const P: u64> MulAssign for PrimeField<P> {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<const P: u64> Zero for PrimeField<P> {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const P: u64> One for PrimeField<P> {
    fn one() -> Self {
        Self::ONE
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl<const P: u64> Semiring for PrimeField<P> {}
impl<const P: u64> Ring for PrimeField<P> {}
impl<const P: u64> CommutativeRing for PrimeField<P> {}

/// Every value but zero has an inverse, which is exactly what `P` being prime
/// buys and what the compile-time check guarantees.
impl<const P: u64> Field for PrimeField<P> {
    fn inverse(&self) -> Option<Self> {
        self.0.inverse().map(Self)
    }
}

/// A field divides exactly, so the remainder is always zero — which makes this
/// Euclidean in the degenerate way every field is.
///
/// Worth having so that code written against [`EuclideanRing`] — a greatest
/// common divisor, a polynomial division — accepts a prime field without a
/// separate path.
impl<const P: u64> EuclideanRing for PrimeField<P> {
    /// Zero or one: a field has nothing finer to say about size, since every
    /// non-zero value divides every other exactly.
    type Size = u8;

    fn euclidean_size(&self) -> u8 {
        u8::from(!self.is_zero())
    }

    fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        Some((self.divide(divisor)?, Self::zero()))
    }
}

impl<const P: u64> fmt::Display for PrimeField<P> {
    /// Just the representative.
    ///
    /// The field is in the type, so repeating it here would only make a
    /// polynomial over `GF(2)` unreadable — `1x + 1` rather than
    /// `1 (GF(2))x + 1 (GF(2))`. [`Modulo`] prints its modulus because it is
    /// usually read on its own rather than nested.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.value())
    }
}

impl<const P: u64> From<u64> for PrimeField<P> {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl<const P: u64> From<i64> for PrimeField<P> {
    fn from(value: i64) -> Self {
        Self::from_signed(value)
    }
}
