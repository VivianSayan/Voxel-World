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
//! [`crate::misc::linear`]. Operations that need square roots or trigonometry
//! are implemented for `f64` only.

use crate::misc::linear::Vector3;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Generates everything that works identically in all three algebras:
/// construction, addition, subtraction, negation and component scaling.
/// Multiplying two values together differs per algebra and is written out.
macro_rules! implement_componentwise {
    ($name:ident, [$($component:ident),+]) => {
        impl<T> $name<T>
        {
            pub const fn new($($component: T),+) -> Self {
                Self { $($component),+ }
            }
        }

        impl<T> $name<T>
        where
            T: Mul<Output = T> + Copy,
        {
            /// Multiplies every component by `factor`.
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Complex<T>
{
    pub re: T,
    pub im: T,
}

implement_componentwise!(Complex, [re, im]);
implement_real_scalar_ops!(Complex, [re, im]);

impl<T> Complex<T>
where
    T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Copy,
{
    /// Mirrors the number across the real axis.
    pub fn conjugate(self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared distance from the origin. Needs no square root, so it stays
    /// exact for integer components.
    pub fn norm_squared(self) -> T {
        self.re * self.re + self.im * self.im
    }
}

impl<T> Mul for Complex<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
{
    type Output = Complex<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T> Div for Complex<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Copy,
{
    type Output = Complex<T>;

    fn div(self, rhs: Self) -> Self::Output {
        // Multiply above and below by the conjugate of the divisor.
        let denominator = rhs.re * rhs.re + rhs.im * rhs.im;

        Complex {
            re: (self.re * rhs.re + self.im * rhs.im) / denominator,
            im: (self.im * rhs.re - self.re * rhs.im) / denominator,
        }
    }
}

impl Complex<f64> {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 0.0);
    /// The imaginary unit.
    pub const I: Self = Self::new(0.0, 1.0);

    pub fn from_real(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    /// Builds a number from a radius and an angle in radians.
    pub fn from_polar(radius: f64, angle: f64) -> Self {
        Self::new(radius * angle.cos(), radius * angle.sin())
    }

    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// The angle from the positive real axis, in radians.
    pub fn argument(self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn normalize(self) -> Self {
        self / self.norm()
    }

    pub fn inverse(self) -> Self {
        self.conjugate() / self.norm_squared()
    }

    /// e raised to this number: the bridge between the plane and rotation.
    pub fn exp(self) -> Self {
        Self::from_polar(self.re.exp(), self.im)
    }

    pub fn ln(self) -> Self {
        Self::new(self.norm().ln(), self.argument())
    }

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
pub struct Dual<T>
{
    pub real: T,
    pub dual: T,
}

implement_componentwise!(Dual, [real, dual]);
implement_real_scalar_ops!(Dual, [real, dual]);

impl<T> Dual<T>
where
    T: Neg<Output = T> + Copy,
{
    pub fn conjugate(self) -> Self {
        Dual {
            real: self.real,
            dual: -self.dual,
        }
    }
}

impl<T> Mul for Dual<T>
where
    T: Add<Output = T> + Mul<Output = T> + Copy,
{
    type Output = Dual<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        // The e^2 term vanishes, leaving the product rule.
        Dual {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

impl<T> Div for Dual<T>
where
    T: Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Copy,
{
    type Output = Dual<T>;

    fn div(self, rhs: Self) -> Self::Output {
        // The quotient rule; undefined when the divisor's real part is zero.
        Dual {
            real: self.real / rhs.real,
            dual: (self.dual * rhs.real - self.real * rhs.dual)
                / (rhs.real * rhs.real),
        }
    }
}

impl Dual<f64> {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 0.0);
    /// The unit that squares to zero.
    pub const E: Self = Self::new(0.0, 1.0);

    /// A value that does not vary: its derivative is zero.
    pub fn constant(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    /// The variable being differentiated with respect to, so its derivative
    /// is one. Feed this into a calculation to differentiate it.
    pub fn variable(value: f64) -> Self {
        Self::new(value, 1.0)
    }

    pub fn recip(self) -> Self {
        Self::new(
            1.0 / self.real,
            -self.dual / (self.real * self.real),
        )
    }

    pub fn sqrt(self) -> Self {
        let root = self.real.sqrt();
        Self::new(root, self.dual / (2.0 * root))
    }

    pub fn exp(self) -> Self {
        let value = self.real.exp();
        Self::new(value, self.dual * value)
    }

    pub fn ln(self) -> Self {
        Self::new(self.real.ln(), self.dual / self.real)
    }

    pub fn sin(self) -> Self {
        Self::new(self.real.sin(), self.dual * self.real.cos())
    }

    pub fn cos(self) -> Self {
        Self::new(self.real.cos(), -self.dual * self.real.sin())
    }

    pub fn tan(self) -> Self {
        let tangent = self.real.tan();
        Self::new(tangent, self.dual * (1.0 + tangent * tangent))
    }

    pub fn powf(self, exponent: f64) -> Self {
        Self::new(
            self.real.powf(exponent),
            self.dual * exponent * self.real.powf(exponent - 1.0),
        )
    }

    pub fn abs(self) -> Self {
        if self.real < 0.0 { -self } else { self }
    }

    pub fn is_finite(self) -> bool {
        self.real.is_finite() && self.dual.is_finite()
    }
}

// ---------------------------------------------------------------------------
// Quaternion: i*i == j*j == k*k == i*j*k == -1
// ---------------------------------------------------------------------------

/// Scalar part first, then the three imaginary parts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Quaternion<T>
{
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
    /// Negates the imaginary parts. For a unit quaternion this is the
    /// inverse rotation.
    pub fn conjugate(self) -> Self {
        Quaternion {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn norm_squared(self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl<T> Mul for Quaternion<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
{
    type Output = Quaternion<T>;

    /// The Hamilton product. Not commutative: `a * b` and `b * a` differ,
    /// which is what lets it represent composed rotations.
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

    /// Right division, `self * rhs.inverse()`. Because the product does not
    /// commute, dividing on the left gives a different answer.
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
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    /// The rotation that does nothing.
    pub const IDENTITY: Self = Self::new(1.0, 0.0, 0.0, 0.0);

    /// A quaternion with no scalar part, which is how a direction enters the
    /// algebra.
    pub fn from_vector(vector: Vector3<f64>) -> Self {
        Self::new(0.0, vector.x, vector.y, vector.z)
    }

    pub fn vector_part(self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// The rotation of `angle` radians about `axis`, right-handed. The axis
    /// is normalized here, so it need not arrive as a unit vector.
    pub fn from_axis_angle(axis: Vector3<f64>, angle: f64) -> Self {
        let half = angle * 0.5;
        let direction = axis.normalize() * half.sin();

        Self::new(half.cos(), direction.x, direction.y, direction.z)
    }

    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    pub fn normalize(self) -> Self {
        self / self.norm()
    }

    pub fn inverse(self) -> Self {
        self.conjugate() / self.norm_squared()
    }

    pub fn dot(self, rhs: Self) -> f64 {
        self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Rotates a vector. `self` must be a unit quaternion, which
    /// `from_axis_angle` and `normalize` both guarantee.
    ///
    /// This is the expanded form of `q * v * q.conjugate()`, which avoids
    /// building the two intermediate quaternions.
    pub fn rotate(self, vector: Vector3<f64>) -> Vector3<f64> {
        let imaginary = self.vector_part();
        let twice_cross = imaginary.cross(vector) * 2.0;

        vector + twice_cross * self.w + imaginary.cross(twice_cross)
    }

    /// Interpolates along the shortest arc between two rotations, at constant
    /// angular speed. Both inputs must be unit quaternions.
    pub fn slerp(self, other: Self, t: f64) -> Self {
        let mut cosine = self.dot(other);
        let mut target = other;

        // q and -q are the same rotation; pick the nearer one so the path
        // never takes the long way round.
        if cosine < 0.0 {
            target = -target;
            cosine = -cosine;
        }

        // Nearly parallel: the arc is too short to divide by safely.
        if cosine > 0.999_95 {
            return (self + (target - self) * t).normalize();
        }

        let angle = cosine.clamp(-1.0, 1.0).acos();
        let sine = angle.sin();

        let from = ((1.0 - t) * angle).sin() / sine;
        let to = (t * angle).sin() / sine;

        self * from + target * to
    }

    pub fn is_finite(self) -> bool {
        self.w.is_finite()
            && self.x.is_finite()
            && self.y.is_finite()
            && self.z.is_finite()
    }
}
