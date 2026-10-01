//! A sketch: distances and positions held in fixed point rather than `f64`.
//!
//! Nothing in the crate uses this. It is here to be read beside
//! [`measure`](super::measure) and [`position`](super::position), which hold
//! the same two ideas in floating point, so the trade can be judged from the
//! code rather than from an argument.
//!
//! [`PreciseUnits`] is what [`WorldUnits`](super::WorldUnits) becomes on top of
//! [`Fixed`], and [`PrecisePosition3`] is a position that can sit anywhere
//! inside a voxel, which is what an entity, a camera or a projectile needs and
//! [`VoxelPosition3`] cannot express.
//!
//! Neither type carries arithmetic of its own beyond what it means for the
//! quantity. A position holds a `Vector3<Fixed>`, and vectors of fixed-point
//! components already add, subtract, scale, interpolate and measure distances
//! in [`linear`](crate::math::linear); what is written here is only the part
//! that is about the world: which voxel a point is in, where inside it, and how
//! it reaches a renderer.
//!
//! # Why a position is the case for fixed point
//!
//! **The step never changes.** An `f64` spends its precision on dynamic range:
//! a step of `2^-52` voxels at the origin, but `1.2e-4` a trillion voxels out,
//! and an eighth of a voxel at `10^15`. A [`Fixed`] holds `2^-32` of a voxel
//! everywhere in its range. Past about a million voxels the fixed-point value
//! is the finer of the two, and it stays that way however far the world goes.
//!
//! **Adding is exact.** A velocity added to a position rounds every time in
//! floating point, and a tenth of a voxel per tick drifts by `1.3e-6` voxels
//! over a million ticks. In fixed point the same sum is exact, so a replay, a
//! peer in lockstep and a re-simulation from a save agree bit for bit rather
//! than nearly.
//!
//! **It is already the octree's layout.** The whole bits of a [`Fixed`] are a
//! voxel coordinate and the low 32 are the offset inside that voxel, so
//! [`PrecisePosition3::voxel`] is an arithmetic shift and
//! [`PrecisePosition3::local`] is a mask. No `floor`, and in particular no `as`
//! cast rounding towards zero, which is the standard way a position just below
//! the origin ends up attributed to the wrong node.
//!
//! # What it costs
//!
//! A position update is addition, which is free. Scaling costs a multiply at
//! about seven times an `f64`'s, and a distance costs a square root at about
//! thirty. A `Vector3<Fixed>` is 48 bytes against 24 for `Vector3<f64>`; an
//! `i64` with the same 32 fractional bits would halve that at the cost of
//! reaching only `±2` billion voxels.
//!
//! Rotations and directions are deliberately absent. They are bounded, do not
//! accumulate, and want `sin` and `cos`, which is `f64` work; the boundary this
//! module implies is that what accumulates or spans the world is fixed point,
//! and what is local and bounded stays floating point.
//!
//! [`Fixed`]: crate::math::Fixed
//! [`VoxelPosition3`]: super::VoxelPosition3

use crate::math::Fixed;
use crate::math::fixed::FRACTION_BITS;
use crate::math::linear::Vector3;
use crate::spatial::depth::{Depth, TreeDepth};
use crate::spatial::position::{LocalPosition3, NodePosition3, VoxelPosition3};
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// A distance in voxels, held exactly to a step of `2^-32`.
///
/// The fixed-point twin of [`WorldUnits`](super::WorldUnits), with the same
/// unit: one is one voxel at the tree's deepest level. What it adds is that
/// addition and subtraction cannot round, and what it gives up is range beyond
/// `±4e28` voxels and the ability to hold a value far smaller than a step.
///
/// There is no `is_finite`, because there is no infinity and no NaN to find.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct PreciseUnits(Fixed);

impl PreciseUnits {
    /// No distance at all.
    pub const ZERO: Self = Self(Fixed::ZERO);
    /// One voxel at the deepest level, which is the unit itself.
    pub const VOXEL: Self = Self(Fixed::ONE);
    /// The smallest distance that is not zero, `2^-32` of a voxel.
    pub const STEP: Self = Self(Fixed::DELTA);
    /// The largest distance, about `4e28` voxels.
    pub const MAX: Self = Self(Fixed::MAX);
    /// The most negative distance, since a distance here is signed and doubles
    /// as an offset along an axis.
    pub const MIN: Self = Self(Fixed::MIN);

    /// A distance from a whole number of voxels.
    pub const fn from_voxels(voxels: i64) -> Self {
        Self(Fixed::from_integer(voxels))
    }

    /// A distance from a fixed-point count of voxels.
    pub const fn from_fixed(voxels: Fixed) -> Self {
        Self(voxels)
    }

    /// A distance from a floating-point count of voxels, truncated to the
    /// nearest step, or `None` if it is not finite or out of range.
    ///
    /// The one place a rounding enters, which is why it is a named conversion
    /// rather than a `From`.
    pub fn from_f64(voxels: f64) -> Option<Self> {
        Fixed::from_f64(voxels).map(Self)
    }

    /// The distance as a fixed-point count of voxels.
    pub const fn voxels(self) -> Fixed {
        self.0
    }

    /// The distance as a floating-point count of voxels, for rendering or
    /// printing. Exact below about `9e6` voxels, and rounded beyond.
    pub fn to_f64(self) -> f64 {
        self.0.to_f64()
    }

    /// Rounded down to a whole voxel, which is the one a position at this
    /// distance falls in.
    ///
    /// Rounds towards negative infinity rather than towards zero, by an
    /// arithmetic shift, so a distance just below the origin lands in the voxel
    /// below it.
    pub const fn floor_voxels(self) -> i128 {
        self.0.to_bits() >> FRACTION_BITS
    }

    /// What is left after [`PreciseUnits::floor_voxels`]: the position inside
    /// that voxel, always in `[0, 1)`.
    pub const fn within_voxel(self) -> Fixed {
        Fixed::from_bits(self.0.to_bits() & (Fixed::ONE.to_bits() - 1))
    }

    /// The distance without its sign. Panics on [`PreciseUnits::MIN`], which
    /// has no positive counterpart.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// The smaller of the two distances.
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// The larger of the two distances.
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// Confined to `low..=high`. Panics if the bounds cross.
    pub fn clamp(self, low: Self, high: Self) -> Self {
        Self(self.0.clamp(low.0, high.0))
    }

    /// How many of `other` fit in this, as a plain number with no unit left.
    ///
    /// Dividing a distance by a distance cancels the unit, which is why this is
    /// a method rather than a `Div`. `None` for a zero divisor.
    pub fn ratio_to(self, other: Self) -> Option<Fixed> {
        self.0.checked_div(other.0)
    }

    /// The sum, or `None` on overflow, for callers that would rather not have
    /// the operator panic.
    pub const fn checked_add(self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(total) => Some(Self(total)),
            None => None,
        }
    }

    /// The distance scaled by a fraction, truncated towards zero. `None` on
    /// overflow.
    pub fn checked_scale(self, factor: Fixed) -> Option<Self> {
        self.0.checked_mul(factor).map(Self)
    }
}

/// Distances add, exactly. Panics on overflow, as [`Fixed`] does.
impl Add for PreciseUnits {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

/// Distances subtract, exactly, and may come out negative.
impl Sub for PreciseUnits {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

/// The same distance in the other direction.
impl Neg for PreciseUnits {
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Scaled by a fraction, which keeps the unit.
impl Mul<Fixed> for PreciseUnits {
    type Output = Self;

    fn mul(self, factor: Fixed) -> Self {
        Self(self.0 * factor)
    }
}

/// Scaled by a whole number, which needs no rescaling and cannot lose a step.
impl Mul<i128> for PreciseUnits {
    type Output = Self;

    fn mul(self, factor: i128) -> Self {
        Self(self.0 * factor)
    }
}

impl AddAssign for PreciseUnits {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl SubAssign for PreciseUnits {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

/// Adds a sequence of distances, exactly however many there are.
impl Sum for PreciseUnits {
    fn sum<I: Iterator<Item = Self>>(distances: I) -> Self {
        distances.fold(Self::ZERO, |total, distance| total + distance)
    }
}

impl fmt::Display for PreciseUnits {
    /// As a count of voxels, such as `12.5 voxels`, with as many decimals as
    /// the value actually has.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} voxels", self.0)
    }
}

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

/// A position anywhere in the world, including inside a voxel.
///
/// [`VoxelPosition3`] names a voxel; this names a point. An entity standing
/// three quarters of the way across a voxel has a position here and a voxel
/// there, and the two are the same number read at different widths: the whole
/// bits are the voxel, the low 32 are the offset within it.
///
/// That is what makes the conversions free and, more to the point, correct
/// below the origin. [`PrecisePosition3::voxel`] is an arithmetic shift, which
/// rounds towards negative infinity and so agrees with the octree; a cast
/// through `as i128` would round towards zero and put every position in
/// `(-1, 0)` in the voxel above the one it is actually in.
///
/// # Why a type of its own
///
/// It holds nothing but a `Vector3<Fixed>`, and the arithmetic all lives there:
/// see [`checked_add`](Vector3::checked_add), [`distance`](Vector3::distance),
/// [`lerp`](Vector3::lerp) and the rest, which work for any fixed-point vector.
/// What this adds is the distinction between a *place* and a *displacement*.
/// Subtracting two positions gives a `Vector3<Fixed>`, and adding one to a
/// position gives a position, but adding two positions together means nothing
/// and will not compile. That is the same reason [`VoxelPosition3`] is not a
/// bare `Vector3<i128>`.
///
/// Not ordered, for the same reason [`LocalPosition3`] is not: one point being
/// "less than" another has no meaning in more than one dimension. It is `Eq`
/// and `Hash`, which a floating-point position could not be, so a position can
/// key a map.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct PrecisePosition3(Vector3<Fixed>);

impl PrecisePosition3 {
    /// The origin of the world.
    pub const ORIGIN: Self = Self(Vector3::new(Fixed::ZERO, Fixed::ZERO, Fixed::ZERO));

    /// A position from three fixed-point coordinates.
    pub const fn new(x: Fixed, y: Fixed, z: Fixed) -> Self {
        Self(Vector3::new(x, y, z))
    }

    /// A position at the point a vector of coordinates names.
    pub const fn from_vector(coordinates: Vector3<Fixed>) -> Self {
        Self(coordinates)
    }

    /// The coordinates as a vector, for the arithmetic [`Vector3`] offers.
    pub const fn coordinates(self) -> Vector3<Fixed> {
        self.0
    }

    /// The low corner of a voxel, or `None` for a coordinate too large to
    /// scale.
    pub fn from_voxel(voxel: VoxelPosition3) -> Option<Self> {
        let [x, y, z]: [i128; 3] = voxel.to_array();

        Some(Self::new(
            Fixed::from_integer_i128(x)?,
            Fixed::from_integer_i128(y)?,
            Fixed::from_integer_i128(z)?,
        ))
    }

    /// The middle of a voxel, which is where a thing standing in it usually
    /// belongs. `None` for a coordinate too large to scale.
    pub fn at_voxel_centre(voxel: VoxelPosition3) -> Option<Self> {
        let half: Vector3<Fixed> = Vector3::splat(Fixed::HALF);

        Self::from_voxel(voxel)?.0.checked_add(half).map(Self)
    }

    /// The voxel this position is inside.
    ///
    /// Three arithmetic shifts: the whole bits of each coordinate are the voxel
    /// coordinate, and shifting rounds towards negative infinity, which is the
    /// octree's own convention.
    pub fn voxel(self) -> VoxelPosition3 {
        VoxelPosition3::new(self.0.map(|value| value.to_bits() >> FRACTION_BITS))
    }

    /// Where this position sits inside its own voxel, each component in
    /// `[0, 1)`.
    ///
    /// The low bits of each coordinate, which is the fraction by construction,
    /// so this cannot fall out of range however the coordinates were reached.
    pub fn local(self) -> LocalPosition3 {
        LocalPosition3::clamped(self.0.map(|value| {
            (value.to_bits() & (Fixed::ONE.to_bits() - 1)) as f64 / Fixed::ONE.to_bits() as f64
        }))
    }

    /// The node at `depth` that contains this position, or `None` for a level
    /// the tree cannot address.
    pub fn node_at(self, tree: TreeDepth, depth: Depth) -> Option<NodePosition3> {
        self.voxel().node_at(tree, depth)
    }

    /// This position moved by a displacement, exactly. `None` on overflow.
    ///
    /// The operation a moving thing performs every tick, and the reason for the
    /// whole type: no rounding here means a position is a function of its
    /// history rather than of how that history was accumulated.
    pub fn checked_offset(self, offset: Vector3<Fixed>) -> Option<Self> {
        self.0.checked_add(offset).map(Self)
    }

    /// The displacement from `other` to this position. `None` on overflow.
    pub fn checked_difference(self, other: Self) -> Option<Vector3<Fixed>> {
        self.0.checked_sub(other.0)
    }

    /// This position after `ticks` steps at a constant velocity, in one
    /// operation rather than a loop. `None` on overflow.
    ///
    /// Identical to offsetting that many times, since neither the scaling nor
    /// the addition rounds. In floating point the two would differ, and the
    /// difference would grow with the number of ticks.
    pub fn advanced_by(self, velocity: Vector3<Fixed>, ticks: u64) -> Option<Self> {
        let steps: Fixed = Fixed::from_integer_i128(ticks as i128)?;

        self.checked_offset(velocity.checked_scale(steps)?)
    }

    /// The squared distance to another position, which is what to compare
    /// against a squared radius. `None` on overflow, which a separation beyond
    /// about `2e14` voxels will cause.
    pub fn distance_squared(self, other: Self) -> Option<PreciseUnits> {
        self.0.distance_squared(other.0).map(PreciseUnits)
    }

    /// The distance to another position, truncated to a step. `None` on
    /// overflow.
    pub fn distance_to(self, other: Self) -> Option<PreciseUnits> {
        self.0.distance(other.0).map(PreciseUnits)
    }

    /// The point `t` of the way from this position to `other`, where zero gives
    /// this one and one gives `other`. `None` on overflow.
    pub fn lerp(self, other: Self, t: Fixed) -> Option<Self> {
        self.0.lerp(other.0, t).map(Self)
    }

    /// The displacement from `origin` to this position, in floating point, for
    /// handing to a renderer.
    ///
    /// Taking the difference first is what keeps the precision: a position a
    /// billion voxels out converts to an `f64` with a step of `1.2e-7`, while
    /// its offset from a nearby camera converts exactly. The subtraction itself
    /// is exact, so nothing is lost before the conversion.
    pub fn to_f64_relative(self, origin: Self) -> Option<Vector3<f64>> {
        Some(self.checked_difference(origin)?.to_f64())
    }
}

impl fmt::Display for PrecisePosition3 {
    /// As `(x, y, z)` in voxels, such as `(12.5, -3.25, 0)`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "({}, {}, {})", self.0.x, self.0.y, self.0.z)
    }
}
