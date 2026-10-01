//! The two quadratic integer rings that tile a plane: `Z[i]` and `Z[ω]`.
//!
//! Both extend the integers by one root of a quadratic, and both keep enough of
//! the integers' behaviour to divide with a remainder — which is what makes
//! greatest common divisors, and therefore unique factorisation, work in them.
//!
//! | | Root | Lattice | Units |
//! |---|---|---|---|
//! | [`Gaussian`] | `i² = -1` | square | 4: `±1`, `±i` |
//! | [`Eisenstein`] | `ω² = -ω - 1` | triangular | 6: `±1`, `±ω`, `±(1 + ω)` |
//!
//! The lattice is the reason to care beyond the algebra. Gaussian integers *are*
//! the square grid, so a walk on a square lattice is arithmetic in `Z[i]`.
//! Eisenstein integers are the triangular grid — the centres of a hexagonal
//! tiling — so hex coordinates are arithmetic in `Z[ω]`, and rotating by sixty
//! degrees is multiplying by a unit.
//!
//! # Genericity
//!
//! Both are generic over their component type, so they work over any signed
//! integer, and over [`Fixed`](crate::math::Fixed) or
//! [`Ratio`](crate::math::Ratio) where a non-integer coefficient is wanted.
//! Division needs [`RoundedDiv`], which is what restricts the interesting
//! operations to types that can round.

use crate::math::traits::{
    AlgebraicNorm, CommutativeRing, Conjugate, EuclideanRing, One, Ring, RoundedDiv, Semiring, Zero,
};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
// Gaussian integers
// ---------------------------------------------------------------------------

/// `a + bi`, where `i² = -1`.
///
/// The integers of the complex plane: a square lattice closed under addition and
/// multiplication. Multiplying by `i` turns the plane a quarter turn, so the
/// four units are the four rotations that map the grid to itself.
///
/// # Against `Complex<i64>`
///
/// [`Complex`](crate::math::Complex) is the same two numbers, and for addition
/// and multiplication the same arithmetic. What this adds is the *ring*: division
/// with a remainder, greatest common divisors, primality, units. Those need the
/// lattice to be integral, which `Complex<f64>` is not.
///
/// # Example
///
/// ```
/// use voxel_world::math::Gaussian;
/// use voxel_world::math::traits::{EuclideanRing, AlgebraicNorm};
///
/// let a: Gaussian<i64> = Gaussian::new(5, 3);
/// let b: Gaussian<i64> = Gaussian::new(2, -1);
///
/// // The norm multiplies, which is what makes it useful.
/// assert_eq!((a * b).algebraic_norm(), a.algebraic_norm() * b.algebraic_norm());
///
/// // And it divides with a remainder smaller than the divisor.
/// let (quotient, remainder) = a.div_rem(&b).expect("b is not zero");
/// assert_eq!(a, quotient * b + remainder);
/// assert!(remainder.algebraic_norm() < b.algebraic_norm());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gaussian<T> {
    /// The integer part.
    pub re: T,
    /// The coefficient of `i`.
    pub im: T,
}

impl<T> Gaussian<T> {
    /// `re + im * i`.
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T: Ring> Gaussian<T> {
    /// `1`.
    pub fn unit() -> Self {
        Self::new(T::one(), T::zero())
    }

    /// `i`, the quarter turn.
    pub fn imaginary() -> Self {
        Self::new(T::zero(), T::one())
    }

    /// Whether this is a real integer, with no imaginary part.
    pub fn is_real(&self) -> bool {
        self.im.is_zero()
    }

    /// The four units of the ring: the values that divide everything.
    ///
    /// They are exactly the values of norm one, and exactly the rotations that
    /// map the square lattice onto itself.
    pub fn units() -> [Self; 4] {
        [
            Self::new(T::one(), T::zero()),
            Self::new(T::zero(), T::one()),
            Self::new(-T::one(), T::zero()),
            Self::new(T::zero(), -T::one()),
        ]
    }

    /// Whether this divides every element, which is to say its norm is one.
    pub fn is_unit(&self) -> bool {
        self.algebraic_norm().is_one()
    }

    /// This turned a quarter turn anticlockwise, which is multiplication by
    /// `i`.
    pub fn turned(&self) -> Self {
        Self::new(-self.im.clone(), self.re.clone())
    }
}

/// `N(a + bi) = a² + b²`, the squared distance from the origin.
impl<T: Ring> AlgebraicNorm for Gaussian<T> {
    type Output = T;

    fn algebraic_norm(&self) -> T {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

/// `conj(a + bi) = a - bi`, the reflection in the real axis.
impl<T: Ring> Conjugate for Gaussian<T> {
    fn conjugate(&self) -> Self {
        Self::new(self.re.clone(), -self.im.clone())
    }
}

impl<T: Ring> Add for Gaussian<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl<T: Ring> Sub for Gaussian<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

/// `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, from `i² = -1`.
impl<T: Ring> Mul for Gaussian<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re.clone() * other.re.clone() - self.im.clone() * other.im.clone(),
            self.re * other.im + self.im * other.re,
        )
    }
}

impl<T: Ring> Neg for Gaussian<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl<T: Ring> AddAssign for Gaussian<T> {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<T: Ring> SubAssign for Gaussian<T> {
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl<T: Ring> MulAssign for Gaussian<T> {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<T: Ring> Zero for Gaussian<T> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
}

impl<T: Ring> One for Gaussian<T> {
    fn one() -> Self {
        Self::new(T::one(), T::zero())
    }

    fn is_one(&self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }
}

impl<T: Ring> Semiring for Gaussian<T> {}
impl<T: Ring> Ring for Gaussian<T> {}

/// `Z[i]` is commutative, unlike the quaternions above it.
impl<T: Ring> CommutativeRing for Gaussian<T> {}

/// Division by nearest lattice point, which is what keeps the remainder small.
///
/// The exact quotient is `a * conj(b) / N(b)`, a point of the plane with
/// rational coordinates. Rounding each coordinate to the nearest integer lands
/// on the closest lattice point, so the remainder is at most half a cell away
/// and its norm at most half the divisor's — comfortably less, which is what
/// Euclid's algorithm needs.
impl<T: Ring + Ord + RoundedDiv> EuclideanRing for Gaussian<T> {
    type Size = T;

    fn euclidean_size(&self) -> T {
        self.algebraic_norm()
    }

    fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        if divisor.is_zero() {
            return None;
        }

        let scaled: Self = self.clone() * divisor.conjugate();
        let norm: T = divisor.algebraic_norm();

        let quotient: Self =
            Self::new(scaled.re.div_rounded(&norm)?, scaled.im.div_rounded(&norm)?);

        let remainder: Self = self.clone() - quotient.clone() * divisor.clone();

        Some((quotient, remainder))
    }
}

impl<T: fmt::Display + Zero + PartialOrd> fmt::Display for Gaussian<T> {
    /// `a + bi`, with the sign folded into the operator.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_pair(formatter, &self.re, &self.im, "i")
    }
}

// ---------------------------------------------------------------------------
// Eisenstein integers
// ---------------------------------------------------------------------------

/// `a + bω`, where `ω² = -ω - 1`: the primitive cube root of one.
///
/// The integers of the triangular lattice. `ω` is a third of a turn, so the six
/// units are the six rotations of a hexagon, and the lattice points are the
/// centres of a hexagonal tiling — which is what makes this the natural
/// arithmetic for hex grids.
///
/// # The relation, and why the norm is not `a² + b²`
///
/// `ω = (-1 + i√3) / 2`, so the axes are sixty degrees apart rather than ninety,
/// and the lattice is sheared. Every power of `ω` reduces through
/// `ω² = -ω - 1`, and the norm picks up the cross term that shear introduces:
///
/// ```text
/// N(a + bω) = a² - ab + b²
/// ```
///
/// It is still positive for everything but zero, and still multiplies, so
/// everything the norm is used for carries over.
///
/// # Example
///
/// ```
/// use voxel_world::math::Eisenstein;
/// use voxel_world::math::traits::{AlgebraicNorm, One, Semiring};
///
/// // ω is a cube root of one, so three sixty-degree turns come back.
/// let omega: Eisenstein<i64> = Eisenstein::omega();
/// assert!(omega.power(3).is_one());
///
/// // Every unit has norm one, and there are six of them.
/// assert!(Eisenstein::<i64>::units().iter().all(|unit| unit.algebraic_norm() == 1));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Eisenstein<T> {
    /// The integer part.
    pub a: T,
    /// The coefficient of `ω`.
    pub b: T,
}

impl<T> Eisenstein<T> {
    /// `a + b * ω`.
    pub const fn new(a: T, b: T) -> Self {
        Self { a, b }
    }
}

impl<T: Ring> Eisenstein<T> {
    /// `ω`, the third of a turn.
    pub fn omega() -> Self {
        Self::new(T::zero(), T::one())
    }

    /// Whether this is a plain integer, with no `ω` part.
    pub fn is_real(&self) -> bool {
        self.b.is_zero()
    }

    /// The six units of the ring.
    ///
    /// `±1`, `±ω` and `±(1 + ω)`, which are the six powers of `-ω` and the six
    /// rotations that map the triangular lattice onto itself.
    pub fn units() -> [Self; 6] {
        [
            Self::new(T::one(), T::zero()),
            Self::new(T::zero(), T::one()),
            Self::new(-T::one(), -T::one()),
            Self::new(-T::one(), T::zero()),
            Self::new(T::zero(), -T::one()),
            Self::new(T::one(), T::one()),
        ]
    }

    /// Whether this divides every element, which is to say its norm is one.
    pub fn is_unit(&self) -> bool {
        self.algebraic_norm().is_one()
    }

    /// This turned a sixth of a turn, which is multiplication by `-ω²`, the
    /// primitive sixth root of one.
    pub fn turned(&self) -> Self {
        self.clone() * Self::new(T::one(), T::one())
    }
}

/// `N(a + bω) = a² - ab + b²`, the squared distance with the shear term the
/// sixty-degree axes introduce.
impl<T: Ring> AlgebraicNorm for Eisenstein<T> {
    type Output = T;

    fn algebraic_norm(&self) -> T {
        self.a.clone() * self.a.clone() - self.a.clone() * self.b.clone()
            + self.b.clone() * self.b.clone()
    }
}

/// `conj(a + bω) = (a - b) - bω`, since `conj(ω) = ω² = -1 - ω`.
impl<T: Ring> Conjugate for Eisenstein<T> {
    fn conjugate(&self) -> Self {
        Self::new(self.a.clone() - self.b.clone(), -self.b.clone())
    }
}

impl<T: Ring> Add for Eisenstein<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.a + other.a, self.b + other.b)
    }
}

impl<T: Ring> Sub for Eisenstein<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.a - other.a, self.b - other.b)
    }
}

/// `(a + bω)(c + dω) = (ac - bd) + (ad + bc - bd)ω`.
///
/// From expanding and reducing the `ω²` term through `ω² = -ω - 1`: the `bd ω²`
/// becomes `-bd - bd ω`, which is where both corrections come from.
impl<T: Ring> Mul for Eisenstein<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let cross: T = self.b.clone() * other.b.clone();

        Self::new(
            self.a.clone() * other.a.clone() - cross.clone(),
            self.a * other.b + self.b * other.a - cross,
        )
    }
}

impl<T: Ring> Neg for Eisenstein<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.a, -self.b)
    }
}

impl<T: Ring> AddAssign for Eisenstein<T> {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<T: Ring> SubAssign for Eisenstein<T> {
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl<T: Ring> MulAssign for Eisenstein<T> {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<T: Ring> Zero for Eisenstein<T> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }
}

impl<T: Ring> One for Eisenstein<T> {
    fn one() -> Self {
        Self::new(T::one(), T::zero())
    }

    fn is_one(&self) -> bool {
        self.a.is_one() && self.b.is_zero()
    }
}

impl<T: Ring> Semiring for Eisenstein<T> {}
impl<T: Ring> Ring for Eisenstein<T> {}
impl<T: Ring> CommutativeRing for Eisenstein<T> {}

/// Division by nearest lattice point, as in [`Gaussian`], allowing for the
/// sheared axes.
///
/// `a * conj(b) / N(b)` is again the exact quotient, but rounding its
/// coordinates is not quite enough here: the triangular cell is not a square, so
/// the nearest point in coordinates is not always the nearest in the plane. The
/// two candidates that rounding can disagree about are both tried, and the one
/// with the smaller remainder wins — which keeps the remainder inside the cell
/// and Euclid's algorithm shrinking.
impl<T: Ring + Ord + RoundedDiv> EuclideanRing for Eisenstein<T> {
    type Size = T;

    fn euclidean_size(&self) -> T {
        self.algebraic_norm()
    }

    fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        if divisor.is_zero() {
            return None;
        }

        let scaled: Self = self.clone() * divisor.conjugate();
        let norm: T = divisor.algebraic_norm();

        let a: T = scaled.a.div_rounded(&norm)?;
        let b: T = scaled.b.div_rounded(&norm)?;

        // The sheared cell means the rounded pair is not always closest, so the
        // neighbours along the diagonal are tried too and the best kept.
        let mut best: Option<(Self, Self)> = None;

        for (da, db) in [
            (T::zero(), T::zero()),
            (T::one(), T::zero()),
            (T::zero(), T::one()),
            (T::one(), T::one()),
            (-T::one(), T::zero()),
            (T::zero(), -T::one()),
            (-T::one(), -T::one()),
        ] {
            let quotient: Self = Self::new(a.clone() + da, b.clone() + db);
            let remainder: Self = self.clone() - quotient.clone() * divisor.clone();
            let size: T = remainder.algebraic_norm();

            let better: bool = match &best {
                Some((_, held)) => size < held.algebraic_norm(),
                None => true,
            };

            if better {
                best = Some((quotient, remainder));
            }
        }

        best
    }
}

impl<T: fmt::Display + Zero + PartialOrd> fmt::Display for Eisenstein<T> {
    /// `a + bω`, with the sign folded into the operator.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_pair(formatter, &self.a, &self.b, "\u{3c9}")
    }
}

/// `a + b<unit>`, or just `a` when the second part is zero, with a minus sign
/// where one belongs.
fn write_pair<T: fmt::Display + Zero + PartialOrd>(
    formatter: &mut fmt::Formatter<'_>,
    first: &T,
    second: &T,
    unit: &str,
) -> fmt::Result {
    if second.is_zero() {
        return write!(formatter, "{first}");
    }

    if *second < T::zero() {
        return write!(formatter, "{first} - {}{unit}", Negated(second));
    }

    write!(formatter, "{first} + {second}{unit}")
}

/// Prints a negative value without its sign, for the case above.
struct Negated<'a, T>(&'a T);

impl<T: fmt::Display> fmt::Display for Negated<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The sign is already in the operator, so it is stripped from the digits
        // rather than negated, which would overflow at the limit.
        let text: String = self.0.to_string();

        formatter.write_str(text.strip_prefix('-').unwrap_or(&text))
    }
}
