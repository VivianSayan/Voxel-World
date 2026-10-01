//! Hypercomplex number systems.
//!
//! Three small algebras, each extending the reals with units that square to
//! something different:
//!
//! - [`Complex`]: `i * i == -1`. Rotation and scaling within a plane.
//! - [`Dual`]: `e * e == 0`, where `e` is usually written epsilon. A dual
//!   number carries a value and a derivative together, so ordinary arithmetic
//!   differentiates itself: evaluate a function at `Dual::variable(x)` and the
//!   dual part of the result is that function's derivative at `x`, exactly,
//!   with no step size and no cancellation error.
//! - [`Quaternion`]: `i*i == j*j == k*k == i*j*k == -1`. Rotation in three
//!   dimensions, free of the gimbal lock that Euler angles suffer from.
//!
//! Each type is generic over its component type, matching
//! [`crate::math::linear`]. Operations that need square roots or trigonometry
//! are implemented for `f64` only.

use crate::math::linear::Vector3;
use crate::math::traits::{
    AlgebraicNorm, CommutativeRing, Conjugate, Field, One, Ring, Semiring, Zero,
};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Generates everything that works identically in all three algebras:
/// construction, addition, subtraction, negation and component scaling.
/// Multiplying two values together differs per algebra and is written out.
macro_rules! implement_componentwise {
    ($name:ident, [$($component:ident),+]) => {
        impl<T> $name<T>
        {
            /// From its components, in the order the struct declares them.
            pub const fn new($($component: T),+) -> Self {
                Self { $($component),+ }
            }
        }

        impl<T> $name<T>
        where
            T: Mul<Output = T> + Copy,
        {
            /// Every component multiplied by `factor`.
            ///
            /// Scaling by a real number, which in all three algebras is the
            /// same component-wise operation, unlike multiplying two values of
            /// the algebra together.
            pub fn scale(self, factor: T) -> Self {
                $name {
                    $($component: self.$component * factor),+
                }
            }
        }

        impl<T> Add for $name<T>
        where
            T: Add<Output = T>,
        {
            type Output = $name<T>;

            fn add(self, rhs: Self) -> Self::Output {
                $name {
                    $($component: self.$component + rhs.$component),+
                }
            }
        }

        impl<T> Sub for $name<T>
        where
            T: Sub<Output = T>,
        {
            type Output = $name<T>;

            fn sub(self, rhs: Self) -> Self::Output {
                $name {
                    $($component: self.$component - rhs.$component),+
                }
            }
        }

        impl<T> Neg for $name<T>
        where
            T: Neg<Output = T>,
        {
            type Output = $name<T>;

            fn neg(self) -> Self::Output {
                $name {
                    $($component: -self.$component),+
                }
            }
        }

        impl<T> AddAssign for $name<T>
        where
            T: Add<Output = T> + Copy,
        {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<T> SubAssign for $name<T>
        where
            T: Sub<Output = T> + Copy,
        {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
    };
}

/// Scalar multiplication and division for the `f64` instantiations. These are
/// written against `f64` rather than a generic `T` so that they cannot overlap
/// with each algebra's own `Mul<Self>`.
macro_rules! implement_real_scalar_ops {
    ($name:ident, [$($component:ident),+]) => {
        impl Mul<f64> for $name<f64> {
            type Output = $name<f64>;

            fn mul(self, scalar: f64) -> Self::Output {
                $name {
                    $($component: self.$component * scalar),+
                }
            }
        }

        impl Div<f64> for $name<f64> {
            type Output = $name<f64>;

            fn div(self, scalar: f64) -> Self::Output {
                $name {
                    $($component: self.$component / scalar),+
                }
            }
        }

        impl MulAssign<f64> for $name<f64> {
            fn mul_assign(&mut self, scalar: f64) {
                *self = *self * scalar;
            }
        }

        impl DivAssign<f64> for $name<f64> {
            fn div_assign(&mut self, scalar: f64) {
                *self = *self / scalar;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Complex: i * i == -1
// ---------------------------------------------------------------------------

/// A number with a real and an imaginary part, where `i * i == -1`.
///
/// Multiplying by one multiplies lengths and adds angles, which makes a unit
/// complex number a rotation of the plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Complex<T> {
    /// The real part.
    pub re: T,
    /// The imaginary part, the coefficient of `i`.
    pub im: T,
}

implement_componentwise!(Complex, [re, im]);
implement_real_scalar_ops!(Complex, [re, im]);

impl<T> Complex<T>
where
    T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Copy,
{
    /// The number mirrored across the real axis: the imaginary part negated.
    ///
    /// Multiplying a number by its conjugate gives its squared norm as a real
    /// number, which is how division is carried out.
    pub fn conjugate(self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    /// The squared distance from the origin.
    ///
    /// Needs no square root, so it stays exact for integer components, and it
    /// is what to compare when only relative sizes matter.
    pub fn norm_squared(self) -> T {
        self.re * self.re + self.im * self.im
    }
}

/// The product, `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, which follows
/// from `i * i == -1`. Lengths multiply and angles add, which is what makes
/// multiplication by a unit complex number a rotation of the plane.
impl<T> Mul for Complex<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
{
    type Output = Complex<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Division, by multiplying above and below by the divisor's conjugate, which
/// leaves a real denominator. Not finite when the divisor is zero.
impl<T> Div for Complex<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Copy,
{
    type Output = Complex<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs.re * rhs.re + rhs.im * rhs.im;

        Complex {
            re: (self.re * rhs.re + self.im * rhs.im) / denominator,
            im: (self.im * rhs.re - self.re * rhs.im) / denominator,
        }
    }
}

impl Complex<f64> {
    /// The origin, and the additive identity.
    pub const ZERO: Self = Self::new(0.0, 0.0);
    /// The multiplicative identity: multiplying by it changes nothing.
    pub const ONE: Self = Self::new(1.0, 0.0);
    /// The imaginary unit, which squares to `-ONE` and is a quarter turn.
    pub const I: Self = Self::new(0.0, 1.0);

    /// A real number with no imaginary part.
    pub fn from_real(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    /// From a radius and an angle in radians, the polar form.
    ///
    /// Uses `sin` and `cos`, so the result may differ in the last bit between
    /// platforms; see [`crate::random::distributions`] on why the generators
    /// avoid them.
    pub fn from_polar(radius: f64, angle: f64) -> Self {
        Self::new(radius * angle.cos(), radius * angle.sin())
    }

    /// The distance from the origin.
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// The angle from the positive real axis, in radians, in `(-pi, pi]`.
    pub fn argument(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// The same direction at distance one from the origin. Not finite for
    /// zero.
    pub fn normalize(self) -> Self {
        self / self.norm()
    }

    /// The number that multiplies with this one to give [`Complex::ONE`]: the
    /// conjugate over the squared norm. Not finite for zero.
    pub fn inverse(self) -> Self {
        self.conjugate() / self.norm_squared()
    }

    /// `e` raised to this number, the bridge between the plane and rotation:
    /// the real part becomes a radius and the imaginary part an angle.
    pub fn exp(self) -> Self {
        Self::from_polar(self.re.exp(), self.im)
    }

    /// The principal logarithm, the inverse of [`Complex::exp`]: the log of
    /// the norm, with the argument as the imaginary part.
    pub fn ln(self) -> Self {
        Self::new(self.norm().ln(), self.argument())
    }

    /// Whether both parts are finite: no infinity and no NaN.
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

// ---------------------------------------------------------------------------
// Dual: e * e == 0
// ---------------------------------------------------------------------------

/// A value paired with its derivative.
///
/// Because `e * e == 0`, every product drops its second-order term, which is
/// exactly the product rule: `(a + a'e)(b + b'e) == ab + (ab' + a'b)e`. Run a
/// calculation on `Dual::variable(x)` and `dual` holds the derivative at `x`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dual<T> {
    pub real: T,
    pub dual: T,
}

implement_componentwise!(Dual, [real, dual]);
implement_real_scalar_ops!(Dual, [real, dual]);

impl<T> Dual<T>
where
    T: Neg<Output = T> + Copy,
{
    /// The dual part negated, which for a derivative is its sign reversed.
    pub fn conjugate(self) -> Self {
        Dual {
            real: self.real,
            dual: -self.dual,
        }
    }
}

/// The product. The `e^2` term vanishes, so what is left is exactly the
/// product rule: `(a + a'e)(b + b'e) = ab + (ab' + a'b)e`.
impl<T> Mul for Dual<T>
where
    T: Add<Output = T> + Mul<Output = T> + Copy,
{
    type Output = Dual<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Dual {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

/// Division, which comes out as the quotient rule. Undefined when the
/// divisor's real part is zero.
impl<T> Div for Dual<T>
where
    T: Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Copy,
{
    type Output = Dual<T>;

    fn div(self, rhs: Self) -> Self::Output {
        Dual {
            real: self.real / rhs.real,
            dual: (self.dual * rhs.real - self.real * rhs.dual) / (rhs.real * rhs.real),
        }
    }
}

impl Dual<f64> {
    /// Zero value, zero derivative.
    pub const ZERO: Self = Self::new(0.0, 0.0);
    /// The multiplicative identity: value one, derivative zero.
    pub const ONE: Self = Self::new(1.0, 0.0);
    /// The unit that squares to zero, usually written epsilon.
    pub const E: Self = Self::new(0.0, 1.0);

    /// A value that does not vary: its derivative is zero.
    pub fn constant(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    /// The variable being differentiated with respect to, so its derivative is
    /// one.
    ///
    /// Feed this into a calculation and the `dual` part of the result is the
    /// derivative at that point, exactly: no step size, and none of the
    /// cancellation a finite difference suffers.
    pub fn variable(value: f64) -> Self {
        Self::new(value, 1.0)
    }

    /// One over the value, carrying the derivative of `1/x`, `-x'/x^2`.
    pub fn recip(self) -> Self {
        Self::new(1.0 / self.real, -self.dual / (self.real * self.real))
    }

    /// The square root, carrying the derivative `x' / (2 sqrt(x))`.
    pub fn sqrt(self) -> Self {
        let root = self.real.sqrt();
        Self::new(root, self.dual / (2.0 * root))
    }

    /// `e` raised to the value, carrying the derivative `x' e^x`.
    pub fn exp(self) -> Self {
        let value = self.real.exp();
        Self::new(value, self.dual * value)
    }

    /// The natural logarithm, carrying the derivative `x' / x`.
    pub fn ln(self) -> Self {
        Self::new(self.real.ln(), self.dual / self.real)
    }

    /// The sine, carrying the derivative `x' cos(x)`.
    pub fn sin(self) -> Self {
        Self::new(self.real.sin(), self.dual * self.real.cos())
    }

    /// The cosine, carrying the derivative `-x' sin(x)`.
    pub fn cos(self) -> Self {
        Self::new(self.real.cos(), -self.dual * self.real.sin())
    }

    /// The tangent, carrying the derivative `x' (1 + tan^2(x))`.
    pub fn tan(self) -> Self {
        let tangent = self.real.tan();
        Self::new(tangent, self.dual * (1.0 + tangent * tangent))
    }

    /// The value raised to a constant power, carrying the derivative
    /// `x' n x^(n-1)`.
    pub fn powf(self, exponent: f64) -> Self {
        Self::new(
            self.real.powf(exponent),
            self.dual * exponent * self.real.powf(exponent - 1.0),
        )
    }

    /// The absolute value, with the derivative flipped along with it.
    /// Undefined at zero, where it returns the value unchanged.
    pub fn abs(self) -> Self {
        if self.real < 0.0 { -self } else { self }
    }

    /// Whether both the value and the derivative are finite.
    pub fn is_finite(self) -> bool {
        self.real.is_finite() && self.dual.is_finite()
    }
}

// ---------------------------------------------------------------------------
// Quaternion: i*i == j*j == k*k == i*j*k == -1
// ---------------------------------------------------------------------------

/// Scalar part first, then the three imaginary parts.
///
/// A unit quaternion represents a rotation in three dimensions, and the product
/// of two composes their rotations. This type is the raw algebra, with no
/// invariant: use [`UnitQuaternion`](crate::math::UnitQuaternion) where a
/// rotation is meant, since that one is kept normalized.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Quaternion<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

implement_componentwise!(Quaternion, [w, x, y, z]);
implement_real_scalar_ops!(Quaternion, [w, x, y, z]);

impl<T> Quaternion<T>
where
    T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Copy,
{
    /// The three imaginary parts negated.
    ///
    /// For a unit quaternion this is the inverse rotation, which is why
    /// rotating by one is so cheap: no division is needed.
    pub fn conjugate(self) -> Self {
        Quaternion {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// The sum of the four squared components. One for a rotation.
    pub fn norm_squared(self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl<T> Mul for Quaternion<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
{
    type Output = Quaternion<T>;

    /// The Hamilton product, from `i*i == j*j == k*k == i*j*k == -1`.
    ///
    /// Not commutative: `a * b` and `b * a` differ, which is exactly what
    /// lets it represent composed rotations, since turning about two axes in
    /// the other order lands somewhere else.
    fn mul(self, rhs: Self) -> Self::Output {
        Quaternion {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

impl<T> Div for Quaternion<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Copy,
{
    type Output = Quaternion<T>;

    /// Right division, `self * rhs.inverse()`, by multiplying by the
    /// conjugate and dividing by the squared norm.
    ///
    /// Because the product does not commute, dividing on the left,
    /// `rhs.inverse() * self`, gives a different answer; this one is not
    /// available as an operator.
    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs.norm_squared();
        let product = self * rhs.conjugate();

        Quaternion {
            w: product.w / denominator,
            x: product.x / denominator,
            y: product.y / denominator,
            z: product.z / denominator,
        }
    }
}

impl Quaternion<f64> {
    /// All four components zero. Not a rotation.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    /// The rotation that does nothing, and the multiplicative identity.
    pub const IDENTITY: Self = Self::new(1.0, 0.0, 0.0, 0.0);

    /// A quaternion with no scalar part, which is how a direction enters the
    /// algebra: the vector becomes the imaginary parts.
    pub fn from_vector(vector: Vector3<f64>) -> Self {
        Self::new(0.0, vector.x, vector.y, vector.z)
    }

    /// The three imaginary parts as a vector, the reverse of
    /// [`Quaternion::from_vector`].
    pub fn vector_part(self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// The length in four dimensions, the square root of
    /// [`Quaternion::norm_squared`].
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// The same quaternion at length one, which is the form a rotation takes.
    /// Not finite for zero.
    pub fn normalize(self) -> Self {
        self / self.norm()
    }

    /// The quaternion that multiplies with this one to give
    /// [`Quaternion::IDENTITY`]: the conjugate over the squared norm. For a
    /// unit quaternion the conjugate alone would do.
    pub fn inverse(self) -> Self {
        self.conjugate() / self.norm_squared()
    }

    /// The four components multiplied pairwise and added, as for a vector.
    /// For two rotations, the cosine of half the angle between them, which is
    /// what [`slerp`](crate::math::UnitQuaternion::slerp) interpolates along.
    pub fn dot(self, rhs: Self) -> f64 {
        self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Whether all four components are finite.
    pub fn is_finite(self) -> bool {
        self.w.is_finite() && self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

// ---------------------------------------------------------------------------
// The split algebras
// ---------------------------------------------------------------------------

/// `a + bj`, where `j² = +1` and `j` is not a real number.
///
/// The third of the three two-dimensional algebras over the reals, and the one
/// that completes the set: a square of `-1` gives [`Complex`], of `0` gives
/// [`Dual`], and of `+1` gives this. There are no others.
///
/// | | Square | AlgebraicNorm | Divides |
/// |---|---|---|---|
/// | [`Complex`] | `i² = -1` | `a² + b²`, positive | everything but zero |
/// | [`Dual`] | `ε² = 0` | `a²`, degenerate | where `a != 0` |
/// | [`SplitComplex`] | `j² = +1` | `a² - b²`, indefinite | off the null lines |
///
/// # Why it has zero divisors
///
/// The norm is `a² - b²`, which is zero whenever `a = ±b` — so `1 + j` and
/// `1 - j` are both non-zero, yet their product is `1 - j² = 0`. Those two null
/// lines are the whole difference from the complex numbers: everywhere off them
/// division works exactly as it does there, and on them nothing can be divided
/// by. That makes this a commutative ring and never a field, whatever the
/// component type.
///
/// # What it is for
///
/// The same thing rotation is for in the complex numbers, hyperbolically.
/// Multiplying by a unit-norm value is a squeeze that preserves `a² - b²` rather
/// than a turn preserving `a² + b²` — which is a Lorentz boost in two
/// dimensions, and why this algebra belongs to spacetime the way the complex
/// numbers belong to the plane.
///
/// # Example
///
/// ```
/// use voxel_world::math::SplitComplex;
/// use voxel_world::math::traits::AlgebraicNorm;
///
/// let j: SplitComplex<i64> = SplitComplex::new(0, 1);
///
/// // j squares to one rather than to minus one.
/// assert_eq!(j * j, SplitComplex::new(1, 0));
///
/// // And the null lines multiply to nothing without being nothing.
/// let up: SplitComplex<i64> = SplitComplex::new(1, 1);
/// let down: SplitComplex<i64> = SplitComplex::new(1, -1);
///
/// assert_eq!(up.algebraic_norm(), 0);
/// assert_eq!(up * down, SplitComplex::new(0, 0));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SplitComplex<T> {
    /// The real part.
    pub re: T,
    /// The coefficient of `j`, whose square is one.
    pub sp: T,
}

implement_componentwise!(SplitComplex, [re, sp]);
implement_real_scalar_ops!(SplitComplex, [re, sp]);

impl<T: Ring> SplitComplex<T> {
    /// `j`, the unit whose square is one.
    pub fn unit() -> Self {
        Self::new(T::zero(), T::one())
    }

    /// Whether this lies on a null line, where the norm vanishes and nothing can
    /// divide by it.
    ///
    /// Exactly the values with `a = ±b`, which is what having zero divisors
    /// means concretely.
    pub fn is_null(&self) -> bool {
        self.algebraic_norm().is_zero()
    }

    /// The two idempotents `(1 ± j)`, which square to themselves up to a factor
    /// of two.
    ///
    /// They span the algebra as a pair of independent lines, which is the
    /// structural reason it is two copies of the reals glued together rather than
    /// a field.
    pub fn null_basis() -> [Self; 2] {
        [
            Self::new(T::one(), T::one()),
            Self::new(T::one(), -T::one()),
        ]
    }
}

/// `N(a + bj) = a² - b²`, indefinite, and the quantity a hyperbolic rotation
/// preserves.
impl<T: Ring> AlgebraicNorm for SplitComplex<T> {
    type Output = T;

    fn algebraic_norm(&self) -> T {
        self.re.clone() * self.re.clone() - self.sp.clone() * self.sp.clone()
    }
}

/// `conj(a + bj) = a - bj`, as in every one of these algebras.
impl<T: Ring> Conjugate for SplitComplex<T> {
    fn conjugate(&self) -> Self {
        Self::new(self.re.clone(), -self.sp.clone())
    }
}

/// `(a + bj)(c + dj) = (ac + bd) + (ad + bc)j`.
///
/// The one sign that differs from [`Complex`]: `j² = +1` adds the cross term
/// where `i² = -1` subtracts it.
impl<T: Ring> Mul for SplitComplex<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re.clone() * other.re.clone() + self.sp.clone() * other.sp.clone(),
            self.re * other.sp + self.sp * other.re,
        )
    }
}

impl<T: Ring> MulAssign for SplitComplex<T> {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<T: Ring> Zero for SplitComplex<T> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.sp.is_zero()
    }
}

impl<T: Ring> One for SplitComplex<T> {
    fn one() -> Self {
        Self::new(T::one(), T::zero())
    }

    fn is_one(&self) -> bool {
        self.re.is_one() && self.sp.is_zero()
    }
}

impl<T: Ring> Semiring for SplitComplex<T> {}
impl<T: Ring> Ring for SplitComplex<T> {}

/// Commutative, as every two-dimensional algebra over the reals is.
impl<T: Ring> CommutativeRing for SplitComplex<T> {}

impl<T: Field> SplitComplex<T> {
    /// The value this multiplies with to give one, or `None` on a null line.
    ///
    /// `conj(x) / N(x)`, exactly as for a complex number — the difference is only
    /// that the norm can vanish without the value doing so.
    pub fn inverse(&self) -> Option<Self> {
        let norm: T = self.algebraic_norm();

        if norm.is_zero() {
            return None;
        }

        let scale: T = norm.inverse()?;
        let conjugate: Self = self.conjugate();

        Some(Self::new(
            conjugate.re * scale.clone(),
            conjugate.sp * scale,
        ))
    }
}

impl<T: fmt::Display + Zero + PartialOrd> fmt::Display for SplitComplex<T> {
    /// `a + bj`, with the sign folded into the operator.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sp.is_zero() {
            return write!(formatter, "{}", self.re);
        }

        if self.sp < T::zero() {
            return write!(formatter, "{} - {}j", self.re, Unsigned(&self.sp));
        }

        write!(formatter, "{} + {}j", self.re, self.sp)
    }
}

/// `w + xi + yj + zk`, where `i² = -1` while `j² = k² = +1`.
///
/// The split quaternions, also called coquaternions: the four-dimensional
/// algebra with one imaginary unit and two split ones. Where
/// [`Quaternion`] has three units that all square to `-1` and is a division
/// algebra, this mixes the signs — and pays for it with zero divisors, exactly as
/// [`SplitComplex`] does against [`Complex`].
///
/// # The relations
///
/// ```text
/// i² = -1        j² = +1        k² = +1
///
/// ij =  k        jk = -i        ki =  j
/// ji = -k        kj =  i        ik = -j
/// ```
///
/// `k` is `ij`, and squares to `+1` because swapping the two units costs a sign
/// that cancels the one from `i²`. Multiplication does not commute: `ij` and `ji`
/// differ in sign, which is why this implements [`Ring`] but not
/// [`CommutativeRing`].
///
/// # What it is
///
/// The two-by-two real matrices, in disguise. The norm below is their
/// determinant, the values of norm one are the matrices of determinant one, and
/// the zero divisors are the singular matrices. That is also why it is not a
/// division algebra: a matrix can be non-zero and have no inverse.
///
/// # Example
///
/// ```
/// use voxel_world::math::SplitQuaternion;
/// use voxel_world::math::traits::AlgebraicNorm;
///
/// let i: SplitQuaternion<i64> = SplitQuaternion::new(0, 1, 0, 0);
/// let j: SplitQuaternion<i64> = SplitQuaternion::new(0, 0, 1, 0);
///
/// // One unit squares to minus one, the other to plus one.
/// assert_eq!(i * i, -SplitQuaternion::new(1, 0, 0, 0));
/// assert_eq!(j * j, SplitQuaternion::new(1, 0, 0, 0));
///
/// // And they anticommute, so the order matters.
/// assert_eq!(i * j, -(j * i));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SplitQuaternion<T> {
    /// The real part.
    pub w: T,
    /// The coefficient of `i`, whose square is minus one.
    pub x: T,
    /// The coefficient of `j`, whose square is one.
    pub y: T,
    /// The coefficient of `k = ij`, whose square is one.
    pub z: T,
}

implement_componentwise!(SplitQuaternion, [w, x, y, z]);
implement_real_scalar_ops!(SplitQuaternion, [w, x, y, z]);

impl<T: Ring> SplitQuaternion<T> {
    /// `i`, the unit that squares to minus one.
    pub fn imaginary() -> Self {
        Self::new(T::zero(), T::one(), T::zero(), T::zero())
    }

    /// `j`, one of the two units that square to one.
    pub fn split() -> Self {
        Self::new(T::zero(), T::zero(), T::one(), T::zero())
    }

    /// `k = ij`, the other.
    pub fn product() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::one())
    }

    /// Whether the norm vanishes, so that nothing can divide by this.
    ///
    /// These are the singular matrices under the correspondence with two-by-two
    /// real matrices, and they are why this is not a division algebra.
    pub fn is_null(&self) -> bool {
        self.algebraic_norm().is_zero()
    }
}

/// `N(q) = w² + x² - y² - z²`, which is the determinant of the matrix this
/// stands for.
///
/// Indefinite, unlike a quaternion's `w² + x² + y² + z²`, so it vanishes on a
/// whole cone rather than only at the origin.
impl<T: Ring> AlgebraicNorm for SplitQuaternion<T> {
    type Output = T;

    fn algebraic_norm(&self) -> T {
        self.w.clone() * self.w.clone() + self.x.clone() * self.x.clone()
            - self.y.clone() * self.y.clone()
            - self.z.clone() * self.z.clone()
    }
}

/// `conj(q) = w - xi - yj - zk`, which gives `q * conj(q) = N(q)`.
impl<T: Ring> Conjugate for SplitQuaternion<T> {
    fn conjugate(&self) -> Self {
        Self::new(
            self.w.clone(),
            -self.x.clone(),
            -self.y.clone(),
            -self.z.clone(),
        )
    }
}

/// The product, from the relations in the type's documentation.
///
/// Every term is one basis product, and the signs are wherever a `j` or a `k`
/// squares to `+1` or two units had to be swapped.
impl<T: Ring> Mul for SplitQuaternion<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let (rw, rx, ry, rz) = (other.w, other.x, other.y, other.z);

        Self::new(
            // i² = -1 subtracts, while j² = k² = +1 add.
            w.clone() * rw.clone() - x.clone() * rx.clone()
                + y.clone() * ry.clone()
                + z.clone() * rz.clone(),
            // jk = -i and kj = +i.
            w.clone() * rx.clone() + x.clone() * rw.clone() - y.clone() * rz.clone()
                + z.clone() * ry.clone(),
            // ik = -j and ki = +j.
            w.clone() * ry.clone() + y.clone() * rw.clone() - x.clone() * rz.clone()
                + z.clone() * rx.clone(),
            // ij = +k and ji = -k.
            w * rz + z * rw + x * ry - y * rx,
        )
    }
}

impl<T: Ring> MulAssign for SplitQuaternion<T> {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<T: Ring> Zero for SplitQuaternion<T> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.w.is_zero() && self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }
}

impl<T: Ring> One for SplitQuaternion<T> {
    fn one() -> Self {
        Self::new(T::one(), T::zero(), T::zero(), T::zero())
    }

    fn is_one(&self) -> bool {
        self.w.is_one() && self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }
}

impl<T: Ring> Semiring for SplitQuaternion<T> {}

/// A ring, but deliberately **not** a
/// [`CommutativeRing`]: `ij = k` while
/// `ji = -k`.
impl<T: Ring> Ring for SplitQuaternion<T> {}

impl<T: Field> SplitQuaternion<T> {
    /// The value this multiplies with to give one, or `None` where the norm
    /// vanishes.
    ///
    /// `conj(q) / N(q)`, as for a quaternion. Because the norm is indefinite this
    /// fails on a cone of non-zero values rather than only at the origin.
    pub fn inverse(&self) -> Option<Self> {
        let norm: T = self.algebraic_norm();

        if norm.is_zero() {
            return None;
        }

        let scale: T = norm.inverse()?;
        let conjugate: Self = self.conjugate();

        Some(Self::new(
            conjugate.w * scale.clone(),
            conjugate.x * scale.clone(),
            conjugate.y * scale.clone(),
            conjugate.z * scale,
        ))
    }
}

impl<T: fmt::Display + Zero + PartialOrd> fmt::Display for SplitQuaternion<T> {
    /// `w + xi + yj + zk`, with signs folded into the operators.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.w)?;

        for (component, unit) in [(&self.x, "i"), (&self.y, "j"), (&self.z, "k")] {
            if *component < T::zero() {
                write!(formatter, " - {}{unit}", Unsigned(component))?;
            } else {
                write!(formatter, " + {component}{unit}")?;
            }
        }

        Ok(())
    }
}

/// Prints a value without its sign, for the displays above.
struct Unsigned<'a, T>(&'a T);

impl<T: fmt::Display> fmt::Display for Unsigned<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The sign is already in the operator, so it is stripped from the text
        // rather than negated, which would overflow at the limit.
        let text: String = self.0.to_string();

        formatter.write_str(text.strip_prefix('-').unwrap_or(&text))
    }
}

// ---------------------------------------------------------------------------
// The older algebras, brought into the trait hierarchy
// ---------------------------------------------------------------------------

/// Expands to `T::zero()`, ignoring the component name it is handed.
///
/// Needed because a repetition has to mention its variable, and these components
/// are all simply zero.
macro_rules! zeroed {
    ($component:ident) => {
        T::zero()
    };
}

/// Implements the algebraic traits for an algebra whose `Mul` is already written.
///
/// Every one of these has the same identities — one in the real component and
/// zero elsewhere — and the same conjugate, so only the norm and what the algebra
/// promises differ.
macro_rules! implement_algebra {
    ($name:ident, $real:ident, [$($imaginary:ident),+]) => {
        impl<T: Ring> Zero for $name<T> {
            fn zero() -> Self {
                Self::new(T::zero(), $(zeroed!($imaginary)),+)
            }

            fn is_zero(&self) -> bool {
                self.$real.is_zero() $(&& self.$imaginary.is_zero())+
            }
        }

        impl<T: Ring> One for $name<T> {
            fn one() -> Self {
                Self::new(T::one(), $(zeroed!($imaginary)),+)
            }

            fn is_one(&self) -> bool {
                self.$real.is_one() $(&& self.$imaginary.is_zero())+
            }
        }

        impl<T: Ring> Conjugate for $name<T> {
            /// The real part kept, every other part negated.
            fn conjugate(&self) -> Self {
                Self::new(self.$real.clone(), $(-self.$imaginary.clone()),+)
            }
        }

        impl<T: Ring + Copy> Semiring for $name<T> {}
        impl<T: Ring + Copy> Ring for $name<T> {}
    };
}

implement_algebra!(Complex, re, [im]);
implement_algebra!(Dual, real, [dual]);
implement_algebra!(Quaternion, w, [x, y, z]);

/// `a² + b²`, the squared magnitude — the quantity that stays in the ring where
/// [`Complex::norm`] does not.
impl<T: Ring> AlgebraicNorm for Complex<T> {
    type Output = T;

    fn algebraic_norm(&self) -> T {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

/// `a²`, which ignores the dual part entirely.
///
/// Degenerate, and the reason `Dual` has zero divisors: every value with a zero
/// real part has norm zero without being zero, so `ε * ε == 0`.
impl<T: Ring> AlgebraicNorm for Dual<T> {
    type Output = T;

    fn algebraic_norm(&self) -> T {
        self.real.clone() * self.real.clone()
    }
}

/// `w² + x² + y² + z²`, positive for everything but zero — which is what makes
/// the quaternions a division algebra where [`SplitQuaternion`] is not.
impl<T: Ring> AlgebraicNorm for Quaternion<T> {
    type Output = T;

    fn algebraic_norm(&self) -> T {
        self.w.clone() * self.w.clone()
            + self.x.clone() * self.x.clone()
            + self.y.clone() * self.y.clone()
            + self.z.clone() * self.z.clone()
    }
}

/// Commutative, as every two-dimensional algebra over a commutative ring is.
impl<T: Ring + Copy> CommutativeRing for Complex<T> {}

/// Commutative: `ε` commutes with everything, since there is only one of it.
impl<T: Ring + Copy> CommutativeRing for Dual<T> {}

// `Quaternion` is deliberately absent: `ij = k` while `ji = -k`, exactly as for
// `SplitQuaternion`.
//
// So is `Field`, for every one of the three. `Dual` and `Quaternion` cannot have
// it — `ε` is a zero divisor, and a quaternion algebra is a field only over a
// field where the norm form stays positive. `Complex<T>` cannot claim it either,
// and the reason is worth stating: it is a field only when `-1` is not a square
// in `T`. Over the reals that holds, but over `GF(5)` it does not — `2² = 4 = -1`
// — so `Complex<PrimeField<5>>` has zero divisors and inverting through the norm
// would fail on them. Nothing in the type system can tell those cases apart, so
// the trait is left unclaimed and the existing inherent `inverse` methods, which
// return what they can, stand instead.
