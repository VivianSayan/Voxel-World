//! Z-order keys: a position folded into one number that keeps nearby positions
//! nearby.
//!
//! A chunk map keyed by three coordinates has no order at all, so walking it
//! jumps about the world and the cache never helps. A Morton key interleaves
//! the bits of the coordinates, which puts the eight children of a node next to
//! one another, and the eight children of *their* parent next to those. Sorting
//! by it walks the world depth-first through the octree, which is the order the
//! octree stores in and the order a generator wants to fill.
//!
//! It is also an octree address in its own right: dropping three bits from the
//! bottom is the parent, which is the same shift a [`NodePosition3`] makes on
//! all three axes at once.
//!
//! Two things bound what it can hold. Three coordinates of 21 bits fit in one
//! 64-bit key, covering a grid about two million across, which is where chunk
//! coordinates comfortably sit; voxel coordinates are `i128` and do not fit, so
//! this is a key for nodes and chunks rather than for every voxel.
//!
//! And the coordinates are **unsigned**. Biasing a signed coordinate into the
//! unsigned range would keep the ordering, but it breaks the hierarchy: the
//! children of a biased node are no longer its key shifted, because the bias is
//! added once per coordinate rather than once per level. A world that addresses
//! negative chunks should add a fixed offset to its coordinates before making
//! keys, and take it off again afterwards; then both properties hold at once.

use crate::math::linear::Vector3;
use crate::spatial::position::NodePosition3;
use std::fmt;

/// How many bits of each coordinate a key carries.
pub const COORDINATE_BITS: u32 = 21;

/// One past the largest coordinate a key covers.
const COORDINATE_LIMIT: u32 = 1 << COORDINATE_BITS;

/// A position as one number, with the bits of its three coordinates
/// interleaved.
///
/// Coordinates are unsigned and below `2^21`. Equality is the position's, and
/// ordering is the curve's: two keys compare as their positions do along it,
/// which is what makes a sorted map of these walk the world in octree order,
/// and what puts the eight children of a node next to one another.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct MortonKey(u64);

impl MortonKey {
    /// The key at the origin, which is zero.
    pub const ORIGIN: Self = Self(0);

    /// The key for a position, or `None` for a coordinate at or above `2^21`.
    pub fn new(x: u32, y: u32, z: u32) -> Option<Self> {
        if x >= COORDINATE_LIMIT || y >= COORDINATE_LIMIT || z >= COORDINATE_LIMIT {
            return None;
        }

        Some(Self(
            spread(x as u64) | (spread(y as u64) << 1) | (spread(z as u64) << 2),
        ))
    }

    /// The key for a node's address, or `None` for a negative or out-of-range
    /// coordinate.
    ///
    /// The depth is not part of the key: two nodes at different levels with the
    /// same address give the same key, so keep the level alongside where it
    /// matters.
    pub fn of_node(node: NodePosition3) -> Option<Self> {
        let [x, y, z]: [i128; 3] = node.to_array();

        Self::new(
            u32::try_from(x).ok()?,
            u32::try_from(y).ok()?,
            u32::try_from(z).ok()?,
        )
    }

    /// The key as a plain number, for storing or as a map key.
    pub const fn value(self) -> u64 {
        self.0
    }

    /// A key from a number that came from [`MortonKey::value`].
    pub const fn from_value(value: u64) -> Self {
        Self(value)
    }

    /// The position the key stands for.
    pub fn coordinates(self) -> Vector3<u32> {
        Vector3::new(
            gather(self.0) as u32,
            gather(self.0 >> 1) as u32,
            gather(self.0 >> 2) as u32,
        )
    }

    /// The key of the node one level up, which is this one with the last three
    /// bits dropped.
    ///
    /// The same step as shifting all three coordinates right by one, and the
    /// reason a Morton key doubles as an octree address.
    pub const fn parent(self) -> Self {
        Self(self.0 >> 3)
    }

    /// The key of the node `levels` up.
    pub const fn ancestor(self, levels: u32) -> Self {
        let places: u32 = 3 * levels;

        if places >= u64::BITS {
            return Self(0);
        }

        Self(self.0 >> places)
    }

    /// The keys of the eight children of this node, in Z-order, or `None` when
    /// the key already uses the top bits and so has no finer level below it.
    ///
    /// Descending is the opposite of [`MortonKey::parent`]: three more bits on
    /// the bottom, one per axis, which is exactly the child's offset within its
    /// parent.
    pub fn children(self) -> Option<[Self; 8]> {
        if self.0 >> (u64::BITS - 3) != 0 {
            return None;
        }

        Some(std::array::from_fn(|child| {
            Self((self.0 << 3) | child as u64)
        }))
    }

    /// Which of its parent's eight children this node is, from 0 to 7.
    pub const fn child_index(self) -> u8 {
        (self.0 & 0b111) as u8
    }

    /// Whether this key names a node inside the one `other` names.
    ///
    /// True when `other`'s key is a prefix of this one, which is the whole test
    /// for containment in an octree: a shift and a comparison.
    pub fn is_inside(self, other: Self, levels_above: u32) -> bool {
        self.ancestor(levels_above) == other
    }
}

impl fmt::Display for MortonKey {
    /// As the position it stands for, such as `morton(3, -1, 7)`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let position: Vector3<u32> = self.coordinates();

        write!(
            formatter,
            "morton({}, {}, {})",
            position.x, position.y, position.z
        )
    }
}

/// Spreads 21 bits out so that each lands two places apart, leaving room for
/// the other two coordinates between them.
///
/// The usual bit-twiddling ladder: each step splits the value into groups and
/// pushes the upper group further left, doubling the gaps until every bit
/// stands alone.
const fn spread(value: u64) -> u64 {
    let mut bits: u64 = value & 0x1F_FFFF;

    bits = (bits | (bits << 32)) & 0x001F_0000_0000_FFFF;
    bits = (bits | (bits << 16)) & 0x001F_0000_FF00_00FF;
    bits = (bits | (bits << 8)) & 0x100F_00F0_0F00_F00F;
    bits = (bits | (bits << 4)) & 0x10C3_0C30_C30C_30C3;
    bits = (bits | (bits << 2)) & 0x1249_2492_4924_9249;

    bits
}

/// The inverse of [`spread`]: collects every third bit back into a run.
const fn gather(value: u64) -> u64 {
    let mut bits: u64 = value & 0x1249_2492_4924_9249;

    bits = (bits | (bits >> 2)) & 0x10C3_0C30_C30C_30C3;
    bits = (bits | (bits >> 4)) & 0x100F_00F0_0F00_F00F;
    bits = (bits | (bits >> 8)) & 0x001F_0000_FF00_00FF;
    bits = (bits | (bits >> 16)) & 0x001F_0000_0000_FFFF;
    bits = (bits | (bits >> 32)) & 0x1F_FFFF;

    bits
}
