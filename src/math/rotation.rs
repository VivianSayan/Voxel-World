//! Normalized rotations, separate from general quaternion arithmetic.
//!
//! A [`Quaternion`] is the raw algebra and may be any length; a
//! [`UnitQuaternion`] is one that is finite and of length one, which is what a
//! rotation has to be. Keeping them apart means a rotation never has to be
//! checked or renormalized on use, and the places where normalization can fail
//! (zero, infinity, NaN) are the constructors, which say so by returning
//! `None`.

use super::{hypercomplex::Quaternion, linear::Vector3};
use crate::math::unit_interval::Unit;

/// A finite quaternion of length one: a rotation in three dimensions.
///
/// The inner quaternion is private, so no component can be changed on its own
/// and the length always holds. Composing two with `*` gives another rotation,
/// applying one to a vector is [`UnitQuaternion::rotate`], and undoing one is
/// [`UnitQuaternion::inverse`], which is only a conjugation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitQuaternion(Quaternion<f64>);

impl UnitQuaternion {
    /// The rotation that does nothing.
    pub const IDENTITY: Self = Self(Quaternion::IDENTITY);

    /// The quaternion scaled to length one, or `None` if it is zero or not
    /// finite, neither of which names a rotation.
    ///
    /// Divides by the largest component before taking the norm. Squaring a
    /// component of a very large or very small quaternion would otherwise
    /// overflow to infinity or underflow to zero, and the length would come out
    /// wrong for a quaternion that was perfectly usable.
    pub fn new(quaternion: Quaternion<f64>) -> Option<Self> {
        if !quaternion.is_finite() {
            return None;
        }
        let scale = quaternion
            .w
            .abs()
            .max(quaternion.x.abs())
            .max(quaternion.y.abs())
            .max(quaternion.z.abs());
        if scale == 0.0 {
            return None;
        }
        let scaled = quaternion / scale;
        Some(Self(scaled / scaled.norm()))
    }

    /// A right-handed rotation of `angle` radians about `axis`, or `None` if
    /// the axis is zero or either input is not finite.
    ///
    /// The axis need not be normalized; it is scaled here, by the largest
    /// component first for the same reason as in [`UnitQuaternion::new`]. The
    /// quaternion holds half the angle, since applying a rotation multiplies by
    /// it twice.
    pub fn from_axis_angle(axis: Vector3<f64>, angle: f64) -> Option<Self> {
        if !axis.is_finite() || !angle.is_finite() {
            return None;
        }
        let scale = axis.abs().max_component();
        if scale == 0.0 {
            return None;
        }
        let axis = axis / scale;
        let direction = axis / axis.norm();
        let half = angle * 0.5;
        let imaginary = direction * half.sin();
        Self::new(Quaternion::new(
            half.cos(),
            imaginary.x,
            imaginary.y,
            imaginary.z,
        ))
    }

    /// The rotation as a plain quaternion, for arithmetic this type does not
    /// offer.
    pub const fn quaternion(self) -> Quaternion<f64> {
        self.0
    }

    /// The rotation undone, which for a unit quaternion is its conjugate: no
    /// division, and no loss of length.
    pub fn inverse(self) -> Self {
        Self(self.0.conjugate())
    }

    /// The vector rotated.
    ///
    /// Uses the cross-product form, `v + 2w(u x v) + 2u x (u x v)` for scalar
    /// part `w` and imaginary part `u`, which gives the same answer as
    /// `q * v * q.conjugate()` with fewer multiplications and without building
    /// the intermediate quaternions.
    pub fn rotate(self, vector: Vector3<f64>) -> Vector3<f64> {
        let imaginary = self.0.vector_part();
        let twice_cross = imaginary.cross(vector) * 2.0;
        vector + twice_cross * self.0.w + imaginary.cross(twice_cross)
    }

    /// The rotation `fraction` of the way from this one to `other`, along the
    /// shortest arc.
    ///
    /// Spherical linear interpolation: the result turns at a constant rate,
    /// unlike interpolating the components and renormalizing. A quaternion and
    /// its negation are the same rotation, so `other` is negated first when
    /// that is the shorter way round. Very close rotations, where the sine of
    /// the angle between them would divide badly, are interpolated straight and
    /// renormalized instead, which is within rounding of the same answer.
    /// The `fraction` is a [`Unit`] rather than a
    /// [`Probability`](crate::units::Probability): it says how far along the arc to
    /// stop, not the chance of anything. Typing it as a probability invited the two
    /// to be confused at a call site.
    pub fn slerp(self, other: Self, fraction: Unit) -> Self {
        let t = fraction.to_f64();
        if t == 0.0 {
            return self;
        }
        if t == 1.0 {
            return other;
        }
        let mut cosine = self.0.dot(other.0);
        let mut target = other.0;
        if cosine < 0.0 {
            target = -target;
            cosine = -cosine;
        }
        let result = if cosine > 0.999_95 {
            self.0 + (target - self.0) * t
        } else {
            let angle = cosine.clamp(-1.0, 1.0).acos();
            let sine = angle.sin();
            self.0 * (((1.0 - t) * angle).sin() / sine) + target * ((t * angle).sin() / sine)
        };
        // Two unit quaternions at most a half turn apart cannot cancel, so
        // the result is never zero and `new` cannot fail.
        Self::new(result).unwrap()
    }
}

/// [`UnitQuaternion::IDENTITY`]: a default rotation is no rotation.
impl Default for UnitQuaternion {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Composes two rotations: `a * b` turns by `b` and then by `a`. The product
/// of two unit quaternions is another, so normalization here cannot fail.
impl std::ops::Mul for UnitQuaternion {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(self.0 * other.0).unwrap()
    }
}

/// Normalizes, as [`UnitQuaternion::new`], with a message instead of `None`.
impl TryFrom<Quaternion<f64>> for UnitQuaternion {
    type Error = &'static str;
    fn try_from(value: Quaternion<f64>) -> Result<Self, Self::Error> {
        Self::new(value).ok_or("a rotation needs a finite nonzero quaternion")
    }
}
