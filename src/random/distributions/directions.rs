//! Directions and points in the unit disc and ball.
//!
//! All exact except [`UnitBall`]. Drawing an angle and taking its sine and
//! cosine would be shorter, but the trigonometric functions are not correctly
//! rounded and may differ between platforms and libm versions, and two machines
//! would then disagree about the same seed. Everything here is arithmetic and
//! `sqrt` on a rejected point, which IEEE-754 pins to one result.

use super::{Distribution, PortableDistribution};
use crate::math::linear::{Vector2, Vector3, Vector4};
use crate::math::{Quaternion, UnitQuaternion};
use crate::random::source::RandomSource;

/// A point drawn uniformly from the whole of the unit disc, not its rim.
///
/// The companion to [`UnitCircle`]: that one scatters on the edge, this one
/// fills the area. Scaling it gives a point inside a circle of any radius,
/// which is what scattering within a distance wants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnitDisc;

/// A point drawn uniformly from the whole of the unit ball.
///
/// Uniform by volume, so it does not crowd the middle: the radius is stretched
/// by a cube root, which is what makes equal volumes equally likely rather
/// than equal radii.
///
/// Platform-dependent, slightly: `cbrt` is not one of the operations IEEE-754
/// requires to be correctly rounded, so the radius may differ in its last bit.
/// The direction is exact, and no draw depends on the radius, so the stream
/// never diverges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnitBall;

/// A uniformly distributed direction in the plane: a point on the unit circle.
///
/// A disc is rotationally symmetric, so pushing a point from it out to the rim
/// leaves the angle uniform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnitCircle;

/// A uniformly distributed direction: a point on the unit sphere (Marsaglia's
/// method, 1972). A disc point lifts to the sphere by way of `z = 1 - 2s`,
/// which spreads it over the surface without bunching anything at the poles.
///
/// Every direction is equally likely. Normalizing a vector drawn from a cube
/// would instead bias directions towards the eight corners, which shows up as
/// visible structure in gradient noise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnitSphere;

/// A uniformly distributed direction on the unit 3-sphere, by Marsaglia's
/// four-dimensional method.
///
/// Two disc points are drawn. The first is used as it stands and the second is
/// scaled so that the four components come to length 1: the squared norm is
/// `s + t * (1 - s) / t`, which is 1 whatever the two radii were.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnitHypersphere;

impl Distribution for UnitDisc {
    type Output = Vector2<f64>;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> Vector2<f64> {
        let (x, y, _): (f64, f64, f64) = disc_point(source);

        Vector2::new(x, y)
    }
}

impl Distribution for UnitBall {
    type Output = Vector3<f64>;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> Vector3<f64> {
        let direction: Vector3<f64> = UnitSphere.sample(source);
        let radius: f64 = source.unit_f64().cbrt();

        direction * radius
    }
}

impl Distribution for UnitCircle {
    type Output = Vector2<f64>;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> Vector2<f64> {
        let (u, v, square_radius): (f64, f64, f64) = disc_point(source);
        let scale: f64 = 1.0 / square_radius.sqrt();

        Vector2::new(u * scale, v * scale)
    }
}

impl Distribution for UnitSphere {
    type Output = Vector3<f64>;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> Vector3<f64> {
        let (u, v, square_radius): (f64, f64, f64) = disc_point(source);
        let factor: f64 = 2.0 * (1.0 - square_radius).sqrt();

        Vector3::new(u * factor, v * factor, 1.0 - 2.0 * square_radius)
    }
}

impl Distribution for UnitHypersphere {
    type Output = Vector4<f64>;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> Vector4<f64> {
        let (x, y, first_square_radius): (f64, f64, f64) = disc_point(source);
        let (z, w, second_square_radius): (f64, f64, f64) = disc_point(source);

        let factor: f64 = ((1.0 - first_square_radius) / second_square_radius).sqrt();

        Vector4::new(x, y, z * factor, w * factor)
    }
}

impl PortableDistribution for UnitDisc {}
impl PortableDistribution for UnitCircle {}
impl PortableDistribution for UnitSphere {}
impl PortableDistribution for UnitHypersphere {}

/// A point drawn uniformly from inside the unit disc, with its squared radius.
/// Every direction above is built out of one or two of these.
///
/// Points are taken from the enclosing square and the corners are thrown away,
/// which keeps about pi/4 of them, so this costs a little over 2.5 draws on
/// average.
///
/// The origin is rejected along with the corners, so callers are free to
/// divide by the radius. The ones that do not lose a region of measure zero by
/// it.
fn disc_point<S: RandomSource + ?Sized>(source: &mut S) -> (f64, f64, f64) {
    loop {
        let u: f64 = 2.0 * source.unit_f64() - 1.0;
        let v: f64 = 2.0 * source.unit_f64() - 1.0;

        let square_radius: f64 = u * u + v * v;

        if square_radius < 1.0 && square_radius != 0.0 {
            return (u, v, square_radius);
        }
    }
}

/// A uniformly random orientation.
///
/// # Why a uniform point on the hypersphere is a uniform rotation
///
/// The unit quaternions double-cover the rotation group: every rotation
/// corresponds to exactly two of them, `q` and `-q`. That covering carries the
/// uniform measure on the sphere onto the uniform (Haar) measure on rotations, so
/// drawing uniformly from [`UnitHypersphere`] and reading the result as a
/// quaternion gives a uniformly random orientation. This is Shoemake's method, and
/// it costs no more than the hypersphere draw it is built on.
///
/// The obvious alternative — three uniform Euler angles — is **not** uniform. It
/// bunches orientations towards the poles, in the same way that picking a latitude
/// uniformly bunches points towards the top of a globe. That is the usual way this
/// gets written wrong, which is reason enough to have it here once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniformRotation;

impl Distribution for UniformRotation {
    type Output = UnitQuaternion;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> UnitQuaternion {
        let point: Vector4<f64> = UnitHypersphere.sample(source);

        UnitQuaternion::new(Quaternion::new(point.x, point.y, point.z, point.w))
            .expect("a point on the unit hypersphere has length one")
    }
}

/// Integer draws and arithmetic only, so the same seed gives the same orientation
/// on every target.
impl PortableDistribution for UniformRotation {}
