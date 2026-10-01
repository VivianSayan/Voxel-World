//! Polynomials over any ring, and the finite fields they quotient into.
//!
//! [`Polynomial`] is `a₀ + a₁x + a₂x² + …` over whatever its coefficients are: a
//! ring for any [`Ring`], and a Euclidean ring — so with division and greatest
//! common divisors — whenever the coefficients form a [`Field`].
//!
//! [`Extension`] is what that buys: `GF(pⁿ)`, the field of `pⁿ` elements, built
//! as polynomials over [`PrimeField`] taken modulo an irreducible one. Together
//! with [`PrimeField`] itself that covers every finite field, since a finite
//! field's size is always a prime power.
//!
//! # Coefficients ascend
//!
//! Index is degree: `coefficients[0]` is the constant term. It reads backwards
//! from the way polynomials are written, and is the right way round for
//! everything else — the degree is the last index, multiplication adds indices,
//! and dividing works from the top by popping.

use crate::math::PrimeField;
use crate::math::traits::{CommutativeRing, EuclideanRing, Field, One, Ring, Semiring, Zero};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
// The polynomial ring
// ---------------------------------------------------------------------------

/// A polynomial in one variable, with coefficients in `T`.
///
/// # Invariant
///
/// No trailing zero coefficients. That is what makes the degree well defined,
/// equality structural, and the zero polynomial uniquely the empty one — so
/// `PartialEq` can be derived and mean what it should. Every constructor and
/// operation normalises.
///
/// # Example
///
/// ```
/// use voxel_world::math::Polynomial;
/// use voxel_world::math::traits::Semiring;
///
/// // 1 + 2x + 3x², written lowest degree first.
/// let p: Polynomial<i64> = Polynomial::new(vec![1, 2, 3]);
///
/// assert_eq!(p.degree(), Some(2));
/// assert_eq!(p.evaluate(&2), 17, "1 + 4 + 12");
///
/// // Multiplying adds degrees.
/// let doubled = p.clone() * p;
/// assert_eq!(doubled.degree(), Some(4));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Polynomial<T> {
    /// Ascending by degree, with no trailing zeros.
    coefficients: Vec<T>,
}

impl<T: Semiring> Polynomial<T> {
    /// A polynomial from its coefficients, lowest degree first.
    pub fn new(coefficients: Vec<T>) -> Self {
        let mut polynomial = Self { coefficients };
        polynomial.normalise();

        polynomial
    }

    /// The constant polynomial `value`.
    pub fn constant(value: T) -> Self {
        Self::new(vec![value])
    }

    /// `x`, the variable itself.
    pub fn variable() -> Self {
        Self::new(vec![T::zero(), T::one()])
    }

    /// `coefficient * xᵈᵉᵍʳᵉᵉ`, the building block of everything else.
    pub fn term(coefficient: T, degree: usize) -> Self {
        let mut coefficients: Vec<T> = vec![T::zero(); degree];
        coefficients.push(coefficient);

        Self::new(coefficients)
    }

    /// The highest power present, or `None` for the zero polynomial.
    ///
    /// `None` rather than a number because zero has no degree: every convention
    /// for it — zero, minus one, minus infinity — breaks one rule or another, and
    /// making the caller decide is more honest than picking one silently.
    pub fn degree(&self) -> Option<usize> {
        self.coefficients.len().checked_sub(1)
    }

    /// The coefficient of `xᵈᵉᵍʳᵉᵉ`, which is zero beyond the degree.
    pub fn coefficient(&self, degree: usize) -> T {
        self.coefficients.get(degree).cloned().unwrap_or(T::zero())
    }

    /// Every coefficient, ascending by degree.
    pub fn coefficients(&self) -> &[T] {
        &self.coefficients
    }

    /// The coefficient of the highest power, or `None` for zero.
    pub fn leading_coefficient(&self) -> Option<T> {
        self.coefficients.last().cloned()
    }

    /// Whether the leading coefficient is one, which is the canonical form for a
    /// polynomial over a field.
    pub fn is_monic(&self) -> bool {
        matches!(self.leading_coefficient(), Some(leading) if leading.is_one())
    }

    /// Whether this is a constant, including zero.
    pub fn is_constant(&self) -> bool {
        self.coefficients.len() <= 1
    }

    /// This polynomial's value at a point, by Horner's method.
    ///
    /// One multiplication and one addition per coefficient, working down from
    /// the top, rather than raising the point to each power separately.
    pub fn evaluate(&self, at: &T) -> T {
        self.coefficients
            .iter()
            .rev()
            .fold(T::zero(), |total, coefficient| {
                total * at.clone() + coefficient.clone()
            })
    }

    /// The derivative, term by term.
    ///
    /// `d/dx aᵏxᵏ = k·aᵏxᵏ⁻¹`, with `k` as a repeated addition since a ring has
    /// no notion of the integer `k` otherwise. Over a field of characteristic
    /// `p`, a term whose degree is a multiple of `p` therefore vanishes — which
    /// is correct, and the reason derivatives detect repeated roots there.
    pub fn derivative(&self) -> Self {
        Self::new(
            self.coefficients
                .iter()
                .enumerate()
                .skip(1)
                .map(|(degree, coefficient)| coefficient.multiple(degree as u32))
                .collect(),
        )
    }

    /// This multiplied by `xᵈᵉᵍʳᵉᵉ`, which is a shift rather than a
    /// multiplication.
    pub fn shifted(&self, degree: usize) -> Self {
        if self.is_zero() {
            return Self::zero();
        }

        let mut coefficients: Vec<T> = vec![T::zero(); degree];
        coefficients.extend(self.coefficients.iter().cloned());

        Self::new(coefficients)
    }

    /// Drops trailing zeros, restoring the invariant.
    fn normalise(&mut self) {
        while self.coefficients.last().is_some_and(Zero::is_zero) {
            self.coefficients.pop();
        }
    }
}

impl<T: Field> Polynomial<T> {
    /// This scaled so that its leading coefficient is one, or `None` for zero.
    ///
    /// The canonical associate over a field: two polynomials differing only by a
    /// constant factor have the same monic form, which is what makes a greatest
    /// common divisor unique rather than unique-up-to-scaling.
    pub fn monic(&self) -> Option<Self> {
        let inverse: T = self.leading_coefficient()?.inverse()?;

        Some(Self::new(
            self.coefficients
                .iter()
                .map(|coefficient| coefficient.clone() * inverse.clone())
                .collect(),
        ))
    }

    /// The greatest common divisor together with the coefficients that build it:
    /// `(gcd, s, t)` with `s·self + t·other == gcd`.
    ///
    /// What an inverse in a quotient ring is made of — [`Extension::inverse`] is
    /// this applied to the field's defining polynomial — and the reason to have
    /// it rather than only the plain greatest common divisor.
    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        let mut remainders: (Self, Self) = (self.clone(), other.clone());
        let mut first: (Self, Self) = (Self::one(), Self::zero());
        let mut second: (Self, Self) = (Self::zero(), Self::one());

        while !remainders.1.is_zero() {
            let Some((quotient, remainder)) = remainders.0.div_rem(&remainders.1) else {
                break;
            };

            remainders = (remainders.1, remainder);
            first = (
                first.1.clone(),
                first.0.clone() - quotient.clone() * first.1,
            );
            second = (second.1.clone(), second.0.clone() - quotient * second.1);
        }

        (remainders.0, first.0, second.0)
    }
}

impl<T: Semiring> Zero for Polynomial<T> {
    /// The empty polynomial, which the invariant makes the only zero.
    fn zero() -> Self {
        Self {
            coefficients: Vec::new(),
        }
    }

    fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }
}

impl<T: Semiring> One for Polynomial<T> {
    /// The constant one, which needs an allocation and so cannot be a constant.
    fn one() -> Self {
        Self::constant(T::one())
    }

    fn is_one(&self) -> bool {
        self.coefficients.len() == 1 && self.coefficients[0].is_one()
    }
}

impl<T: Semiring> Add for Polynomial<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let length: usize = self.coefficients.len().max(other.coefficients.len());

        Self::new(
            (0..length)
                .map(|degree| self.coefficient(degree) + other.coefficient(degree))
                .collect(),
        )
    }
}

impl<T: Ring> Sub for Polynomial<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let length: usize = self.coefficients.len().max(other.coefficients.len());

        Self::new(
            (0..length)
                .map(|degree| self.coefficient(degree) - other.coefficient(degree))
                .collect(),
        )
    }
}

/// Schoolbook multiplication: every pair of terms, with degrees added.
///
/// Quadratic in the degrees, which is the right choice here. The subquadratic
/// methods — Karatsuba, and transforms above that — only overtake it at degrees
/// far beyond anything this crate multiplies, and cost accuracy or a root of
/// unity to do it.
impl<T: Semiring> Mul for Polynomial<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        let length: usize = self.coefficients.len() + other.coefficients.len() - 1;
        let mut coefficients: Vec<T> = vec![T::zero(); length];

        for (left, first) in self.coefficients.iter().enumerate() {
            for (right, second) in other.coefficients.iter().enumerate() {
                coefficients[left + right] =
                    coefficients[left + right].clone() + first.clone() * second.clone();
            }
        }

        Self::new(coefficients)
    }
}

impl<T: Ring> Neg for Polynomial<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(
            self.coefficients
                .into_iter()
                .map(|coefficient| -coefficient)
                .collect(),
        )
    }
}

impl<T: Semiring> AddAssign for Polynomial<T> {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<T: Ring> SubAssign for Polynomial<T> {
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl<T: Semiring> MulAssign for Polynomial<T> {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<T: Semiring> Semiring for Polynomial<T> {}
impl<T: Ring> Ring for Polynomial<T> {}
impl<T: CommutativeRing> CommutativeRing for Polynomial<T> {}

/// Long division, which needs the leading coefficient to be invertible — and so
/// needs the coefficients to form a field.
///
/// Over a ring that is not a field there is no division in general: `2x` does not
/// divide `x` in `Z[x]`, however small the remainder is allowed to be. That is
/// why this implementation asks for [`Field`] while the ring operations above ask
/// only for [`Ring`].
impl<T: Field> EuclideanRing for Polynomial<T> {
    /// The degree, shifted by one so that zero — which has no degree — is
    /// smaller than every polynomial that has one.
    type Size = usize;

    fn euclidean_size(&self) -> usize {
        self.coefficients.len()
    }

    fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        let divisor_degree: usize = divisor.degree()?;
        let inverse: T = divisor.leading_coefficient()?.inverse()?;

        let Some(dividend_degree) = self.degree() else {
            return Some((Self::zero(), Self::zero()));
        };

        if dividend_degree < divisor_degree {
            return Some((Self::zero(), self.clone()));
        }

        // Written into by degree rather than pushed onto: a step can cancel
        // several leading terms at once, leaving a gap in the quotient that
        // appending would silently close up.
        let mut quotient: Vec<T> = vec![T::zero(); dividend_degree - divisor_degree + 1];
        let mut remainder: Self = self.clone();

        // Work down from the top: each step cancels the remainder's leading
        // term, so the degree falls and the loop ends.
        while let Some(degree) = remainder.degree() {
            if degree < divisor_degree {
                break;
            }

            let scale: T = remainder
                .leading_coefficient()
                .expect("a polynomial with a degree has a leading coefficient")
                * inverse.clone();
            let step: usize = degree - divisor_degree;

            quotient[step] = scale.clone();
            remainder -= Self::term(scale, step) * divisor.clone();
        }

        Some((Self::new(quotient), remainder))
    }

    /// Monic, which is the canonical associate over a field.
    fn gcd_normalised(&self, other: &Self) -> Self {
        self.gcd(other).monic().unwrap_or(Self::zero())
    }
}

impl<T: Semiring + fmt::Display> fmt::Display for Polynomial<T> {
    /// Highest degree first, the way polynomials are written, skipping zero
    /// terms.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return formatter.write_str("0");
        }

        let mut first: bool = true;

        for (degree, coefficient) in self.coefficients.iter().enumerate().rev() {
            if coefficient.is_zero() {
                continue;
            }

            if !first {
                formatter.write_str(" + ")?;
            }
            first = false;

            // `x` rather than `1x`, the way it is written by hand; the constant
            // term keeps its digit, since `+ 1` cannot be left implicit.
            let implicit: bool = coefficient.is_one() && degree > 0;

            match (degree, implicit) {
                (0, _) => write!(formatter, "{coefficient}")?,
                (1, true) => formatter.write_str("x")?,
                (1, false) => write!(formatter, "{coefficient}x")?,
                (_, true) => write!(formatter, "x^{degree}")?,
                (_, false) => write!(formatter, "{coefficient}x^{degree}")?,
            }
        }

        Ok(())
    }
}

impl<T: Semiring> FromIterator<T> for Polynomial<T> {
    /// From coefficients, lowest degree first.
    fn from_iter<I: IntoIterator<Item = T>>(coefficients: I) -> Self {
        Self::new(coefficients.into_iter().collect())
    }
}

// ---------------------------------------------------------------------------
// GF(p^n)
// ---------------------------------------------------------------------------

/// The polynomial that defines an [`Extension`] field.
///
/// # What the implementor promises
///
/// That [`Modulus::polynomial`] is **irreducible** over `GF(P)` — that it has no
/// factors but itself and constants. Nothing here can check that while
/// compiling: irreducibility is a property of the coefficients, not of a const
/// parameter, and testing it means trial division. So it is a promise, and
/// [`is_irreducible`] is provided to verify it in a test.
///
/// If the promise is broken the arithmetic still runs, but the result is a ring
/// with zero divisors rather than a field, and some non-zero value will have no
/// inverse.
///
/// # Example
///
/// ```
/// use voxel_world::math::polynomial::{Extension, Modulus, is_irreducible};
/// use voxel_world::math::{Polynomial, PrimeField};
/// use voxel_world::math::traits::{Field, One};
///
/// /// GF(8), as GF(2) adjoined a root of x³ + x + 1.
/// struct Cubic;
///
/// impl Modulus<2> for Cubic {
///     const DEGREE: usize = 3;
///
///     fn polynomial() -> Polynomial<PrimeField<2>> {
///         // 1 + x + x³, lowest degree first.
///         Polynomial::new(vec![
///             PrimeField::new(1),
///             PrimeField::new(1),
///             PrimeField::new(0),
///             PrimeField::new(1),
///         ])
///     }
/// }
///
/// // The promise, checked.
/// assert!(is_irreducible(&Cubic::polynomial()));
///
/// let x: Extension<2, Cubic> = Extension::variable();
/// assert!(x.inverse().is_some(), "a field inverts everything but zero");
/// ```
pub trait Modulus<const P: u64> {
    /// The degree of the defining polynomial, which is the `n` in `GF(pⁿ)`.
    const DEGREE: usize;

    /// The defining polynomial, monic and irreducible over `GF(P)`.
    fn polynomial() -> Polynomial<PrimeField<P>>;
}

/// Whether a polynomial over a prime field has no factors but constants and
/// itself.
///
/// By trial division against every monic polynomial up to half the degree, which
/// is exhaustive and slow — fine for checking a field's definition once in a
/// test, wrong for anything in a loop.
pub fn is_irreducible<const P: u64>(polynomial: &Polynomial<PrimeField<P>>) -> bool {
    let Some(degree) = polynomial.degree() else {
        return false;
    };

    if degree == 0 {
        return false;
    }

    if degree == 1 {
        return true;
    }

    // A reducible polynomial has a factor of degree at most half its own, so
    // only those need trying.
    for candidate_degree in 1..=degree / 2 {
        for candidate in monic_polynomials::<P>(candidate_degree) {
            if candidate.divides(polynomial) {
                return false;
            }
        }
    }

    true
}

/// Every monic polynomial of one degree over `GF(P)`, which is what trial
/// division has to walk.
fn monic_polynomials<const P: u64>(degree: usize) -> Vec<Polynomial<PrimeField<P>>> {
    let count: u64 = P.pow(degree as u32);

    (0..count)
        .map(|index| {
            let mut coefficients: Vec<PrimeField<P>> = Vec::with_capacity(degree + 1);
            let mut remaining: u64 = index;

            for _ in 0..degree {
                coefficients.push(PrimeField::new(remaining % P));
                remaining /= P;
            }

            coefficients.push(PrimeField::new(1));

            Polynomial::new(coefficients)
        })
        .collect()
}

/// The field of `Pⁿ` elements: polynomials over `GF(P)` modulo an irreducible
/// one of degree `n`.
///
/// Every finite field is one of these or a [`PrimeField`], since a finite field's
/// size is always a prime power — so between the two, all of them are covered.
///
/// # Why a polynomial quotient
///
/// `GF(4)` cannot be `Z/4Z`: 2 × 2 = 0 there, so 2 has no inverse and it is no
/// field. What works instead is to take polynomials over `GF(2)` and divide by
/// one that has no roots — `x² + x + 1` — exactly as the complex numbers are the
/// reals modulo `x² + 1`. The four elements are then `0`, `1`, `x` and `x + 1`.
///
/// # Example
///
/// ```
/// use voxel_world::math::polynomial::{Extension, Modulus};
/// use voxel_world::math::{Polynomial, PrimeField};
/// use voxel_world::math::traits::{Field, One, Semiring};
///
/// struct Quadratic;
///
/// impl Modulus<2> for Quadratic {
///     const DEGREE: usize = 2;
///
///     fn polynomial() -> Polynomial<PrimeField<2>> {
///         Polynomial::new(vec![PrimeField::new(1), PrimeField::new(1), PrimeField::new(1)])
///     }
/// }
///
/// type GF4 = Extension<2, Quadratic>;
///
/// // x is a cube root of one here, since GF(4) has three non-zero elements.
/// let x: GF4 = Extension::variable();
/// assert!(x.power(3).is_one());
/// ```
pub struct Extension<const P: u64, M: Modulus<P>> {
    /// Always reduced below the defining polynomial's degree.
    value: Polynomial<PrimeField<P>>,
    modulus: PhantomData<fn() -> M>,
}

impl<const P: u64, M: Modulus<P>> Extension<P, M> {
    /// The class of a polynomial, reduced.
    pub fn new(value: Polynomial<PrimeField<P>>) -> Self {
        Self {
            value: Self::reduce(value),
            modulus: PhantomData,
        }
    }

    /// The class of a constant from the base field.
    pub fn constant(value: PrimeField<P>) -> Self {
        Self::new(Polynomial::constant(value))
    }

    /// `x`, the root the field was built by adjoining.
    pub fn variable() -> Self {
        Self::new(Polynomial::variable())
    }

    /// The reduced polynomial standing for this element.
    pub fn polynomial(&self) -> &Polynomial<PrimeField<P>> {
        &self.value
    }

    /// How many elements the field has: `Pⁿ`.
    pub fn order() -> u64 {
        P.pow(M::DEGREE as u32)
    }

    /// A polynomial taken modulo the defining one.
    fn reduce(value: Polynomial<PrimeField<P>>) -> Polynomial<PrimeField<P>> {
        match value.div_rem(&M::polynomial()) {
            Some((_, remainder)) => remainder,
            // The defining polynomial is zero, which no correct `Modulus`
            // returns; leaving the value be is the least surprising answer.
            None => value,
        }
    }
}

impl<const P: u64, M: Modulus<P>> Clone for Extension<P, M> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            modulus: PhantomData,
        }
    }
}

impl<const P: u64, M: Modulus<P>> PartialEq for Extension<P, M> {
    /// Reduced on the way in, so equal classes are equal polynomials.
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<const P: u64, M: Modulus<P>> Eq for Extension<P, M> {}

impl<const P: u64, M: Modulus<P>> fmt::Debug for Extension<P, M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Extension({})", self.value)
    }
}

impl<const P: u64, M: Modulus<P>> fmt::Display for Extension<P, M> {
    /// The representative polynomial, and the field it lives in.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} (GF({}))", self.value, Self::order())
    }
}

impl<const P: u64, M: Modulus<P>> Add for Extension<P, M> {
    type Output = Self;

    /// Addition cannot raise the degree, so nothing needs reducing.
    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            modulus: PhantomData,
        }
    }
}

impl<const P: u64, M: Modulus<P>> Sub for Extension<P, M> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            modulus: PhantomData,
        }
    }
}

impl<const P: u64, M: Modulus<P>> Mul for Extension<P, M> {
    type Output = Self;

    /// The one operation that can raise the degree, so the one that reduces.
    fn mul(self, other: Self) -> Self {
        Self::new(self.value * other.value)
    }
}

impl<const P: u64, M: Modulus<P>> Neg for Extension<P, M> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            modulus: PhantomData,
        }
    }
}

impl<const P: u64, M: Modulus<P>> AddAssign for Extension<P, M> {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<const P: u64, M: Modulus<P>> SubAssign for Extension<P, M> {
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl<const P: u64, M: Modulus<P>> MulAssign for Extension<P, M> {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<const P: u64, M: Modulus<P>> Zero for Extension<P, M> {
    fn zero() -> Self {
        Self {
            value: Polynomial::zero(),
            modulus: PhantomData,
        }
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl<const P: u64, M: Modulus<P>> One for Extension<P, M> {
    fn one() -> Self {
        Self {
            value: Polynomial::one(),
            modulus: PhantomData,
        }
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }
}

impl<const P: u64, M: Modulus<P>> Semiring for Extension<P, M> {}
impl<const P: u64, M: Modulus<P>> Ring for Extension<P, M> {}
impl<const P: u64, M: Modulus<P>> CommutativeRing for Extension<P, M> {}

/// Every element but zero has an inverse, provided the defining polynomial is
/// irreducible as [`Modulus`] promises.
///
/// By the extended Euclidean algorithm against the defining polynomial: since it
/// is irreducible, it shares no factor with anything of lower degree, so the
/// greatest common divisor is a constant and the coefficient beside it is the
/// inverse.
impl<const P: u64, M: Modulus<P>> Field for Extension<P, M> {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let (divisor, coefficient, _) = self.value.extended_gcd(&M::polynomial());

        // A constant greatest common divisor is what irreducibility guarantees;
        // anything else means the promise was broken.
        let scale: PrimeField<P> = divisor.leading_coefficient()?.inverse()?;

        if !divisor.is_constant() {
            return None;
        }

        Some(Self::new(coefficient * Polynomial::constant(scale)))
    }
}
