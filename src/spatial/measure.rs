//! Distances measured in voxels.
//!
//! One type, [`WorldUnits`], and one unit: a voxel at the tree's deepest level.
//! Time quantities live in [`crate::time`], which is the same idea applied to
//! ticks.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A distance, in voxels at the tree's deepest level.
///
/// One unit is one voxel. Held as an `f64` rather than an integer because it
/// has to survive division, interpolation and the inside of a node, which
/// integers do not.
///
/// The choice of a voxel as the unit is the part most likely to change. If the
/// world later wants metres, this is where the conversion goes, and everything
/// measuring in voxels keeps working because it never spelled out the number.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Debug)]
pub struct WorldUnits(f64);

impl WorldUnits {
    /// No distance at all.
    pub const ZERO: Self = Self(0.0);
    /// One voxel at the deepest level, which is the unit itself.
    pub const VOXEL: Self = Self(1.0);

    /// A distance from a count of voxels, fractions included. Nothing is
    /// checked: an infinite or NaN distance is possible and
    /// [`WorldUnits::is_finite`] is there to catch it.
    pub const fn new(voxels: f64) -> Self {
        Self(voxels)
    }

    /// A distance from a whole number of voxels. Exact up to `2^53` voxels,
    /// past which the conversion rounds.
    pub fn from_voxels(voxels: i64) -> Self {
        Self(voxels as f64)
    }

    /// The distance as a count of voxels.
    pub const fn voxels(self) -> f64 {
        self.0
    }

    /// Rounded down to a whole voxel, which is the one a position falls in.
    ///
    /// Rounds towards negative infinity rather than towards zero, so a position
    /// just below the origin lands in the voxel below it rather than in the one
    /// above.
    pub fn floor_voxels(self) -> i64 {
        self.0.floor() as i64
    }

    /// The distance without its sign.
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

    /// How many of `other` fit in this, as a plain number with no unit left.
    ///
    /// Dividing a distance by a distance cancels the unit, which is why this is
    /// a method rather than a `Div` implementation.
    pub fn ratio_to(self, other: Self) -> f64 {
        self.0 / other.0
    }

    /// Whether the distance is finite: no infinity and no NaN.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

/// Distances add.
impl Add for WorldUnits {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

/// Distances subtract, and may come out negative: a distance here is signed,
/// since it also serves as an offset along an axis.
impl Sub for WorldUnits {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

/// The same distance in the other direction.
impl Neg for WorldUnits {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Scaling by a plain number keeps the unit; dividing by another distance does
/// not, which is why that is [`WorldUnits::ratio_to`] and not `Div`.
impl Mul<f64> for WorldUnits {
    type Output = Self;
    fn mul(self, factor: f64) -> Self {
        Self(self.0 * factor)
    }
}

impl Div<f64> for WorldUnits {
    type Output = Self;
    fn div(self, divisor: f64) -> Self {
        Self(self.0 / divisor)
    }
}

impl AddAssign for WorldUnits {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl SubAssign for WorldUnits {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl fmt::Display for WorldUnits {
    /// With three decimals and the unit named, such as `12.500 voxels`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.3} voxels", self.0)
    }
}
