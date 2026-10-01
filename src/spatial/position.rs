//! The three spaces a position can be in.
//!
//! In conversation they are all "the position", and as numbers they are all
//! vectors, which is exactly why they get mixed up:
//!
//! - [`VoxelPosition3`] is a coordinate at the tree's deepest level, one unit
//!   per voxel. This is the coordinate the world is addressed in.
//! - [`NodePosition3`] is the address of an octree node, which is a voxel
//!   coordinate shifted right. It carries the depth it is an address in,
//!   because `(3, 4, 5)` at depth 6 and `(3, 4, 5)` at depth 7 are different
//!   places and comparing them as vectors would say otherwise.
//! - [`LocalPosition3`] is where a sample sits inside its node, each component
//!   in `[0, 1)`. This is what interpolation weights are built from.
//!
//! Converting between them needs the [`TreeDepth`], since that is what decides
//! how wide a node is, so the conversions take one rather than assuming.

use crate::math::linear::{Vector2, Vector3, Vector4};
use crate::spatial::depth::{Depth, TreeDepth};
use std::fmt;

/// Writes out the three position types for one dimension, and the conversions
/// between them.
macro_rules! implement_positions {
    ($voxel:ident, $node:ident, $local:ident, $vector:ident, $dimension:literal) => {
        /// A coordinate at the tree's deepest level, one unit per voxel.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
        pub struct $voxel($vector<i128>);

        /// Ordered by axis, x first, so that these can key a sorted map. The
        /// order is an arbitrary but stable one; it says nothing about where
        /// the positions are in space, and nearby positions are not nearby in
        /// it.
        impl Ord for $voxel {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0.to_array().cmp(&other.0.to_array())
            }
        }

        impl PartialOrd for $voxel {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl $voxel {
            /// A voxel coordinate from a vector, at the tree's deepest level.
            pub const fn new(coordinates: $vector<i128>) -> Self {
                Self(coordinates)
            }

            /// The coordinate as a vector, for arithmetic this type does not
            /// offer.
            pub const fn coordinates(self) -> $vector<i128> {
                self.0
            }

            /// The components in axis order, which is the form the seed
            /// derivations take.
            pub fn to_array(self) -> [i128; $dimension] {
                self.0.to_array()
            }

            /// The node at `depth` that contains this voxel.
            ///
            /// A right shift, which rounds towards negative infinity rather
            /// than towards zero, so the node below the origin is the one that
            /// actually contains the voxel.
            pub fn node_at(self, tree: TreeDepth, depth: Depth) -> Option<$node> {
                Some($node {
                    address: self.0 >> tree.shift_for(depth)?,
                    depth,
                })
            }

            /// Where this voxel sits inside its node at `depth`, each
            /// component in `[0, 1)`.
            ///
            /// The node's origin is the coordinate with its low bits cleared,
            /// and what is left is divided by the node's width. `None` for a
            /// level this tree cannot address.
            pub fn local_at(self, tree: TreeDepth, depth: Depth) -> Option<$local> {
                let shift: u32 = tree.shift_for(depth)?;
                let origin: $vector<i128> = (self.0 >> shift) << shift;
                let width: f64 = (1u128 << shift) as f64;

                Some($local::clamped(
                    (self.0 - origin).map(|component| component as f64 / width),
                ))
            }

            /// This voxel moved by a whole number of voxels along each axis.
            /// Wraps on overflow, which no real coordinate comes near.
            pub fn offset(self, by: $vector<i128>) -> Self {
                Self(self.0 + by)
            }
        }

        /// A vector read as a voxel coordinate.
        impl From<$vector<i128>> for $voxel {
            fn from(coordinates: $vector<i128>) -> Self {
                Self(coordinates)
            }
        }

        /// As `voxel [x, y, ...]`.
        impl fmt::Display for $voxel {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "voxel {:?}", self.0.to_array())
            }
        }

        /// The address of an octree node, together with the level it addresses.
        ///
        /// Two of these are equal only when both the address and the depth
        /// match, so a node and its parent never compare equal even where their
        /// addresses coincide.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub struct $node {
            address: $vector<i128>,
            depth: Depth,
        }

        impl $node {
            /// A node address at a given level. Nothing is checked here; the
            /// tree is what decides which levels exist, and the conversions
            /// take one.
            pub const fn new(address: $vector<i128>, depth: Depth) -> Self {
                Self { address, depth }
            }

            /// The address as a vector, without its depth.
            pub const fn address(self) -> $vector<i128> {
                self.address
            }

            /// The level this is an address in.
            pub const fn depth(self) -> Depth {
                self.depth
            }

            /// The address components in axis order, without the depth.
            pub fn to_array(self) -> [i128; $dimension] {
                self.address.to_array()
            }

            /// The node one level up that contains this one, or `None` when
            /// that level cannot be named.
            pub fn parent(self) -> Option<Self> {
                self.ancestor(1)
            }

            /// This node's ancestor `steps` levels up: the address shifted
            /// right by that many bits.
            ///
            /// `None` when the level would fall outside a [`Depth`].
            pub fn ancestor(self, steps: u8) -> Option<Self> {
                let level = i8::try_from(self.depth.level() as i16 - steps as i16).ok()?;
                Some(Self {
                    address: self.address >> (steps as u32).min(i128::BITS - 1),
                    depth: Depth::new(level),
                })
            }

            /// The lowest-addressed voxel this node covers: the address
            /// shifted back left into voxel coordinates.
            ///
            /// `None` for a level this tree cannot address, and also when the
            /// address is so far out that the shift would push bits off the top
            /// of an `i128`, which is why the result is shifted back and
            /// compared rather than the shift count alone being checked.
            pub fn origin(self, tree: TreeDepth) -> Option<$voxel> {
                let shift = tree.shift_for(self.depth)?;
                let origin = self.address << shift;
                ((origin >> shift) == self.address).then_some($voxel(origin))
            }

            /// How many voxels across this node is, or `None` for a level
            /// this tree cannot address.
            pub fn width(self, tree: TreeDepth) -> Option<u128> {
                tree.node_width(self.depth)
            }

            /// Whether this node covers that voxel, by asking which node the
            /// voxel falls in at this level.
            pub fn contains(self, tree: TreeDepth, voxel: $voxel) -> bool {
                voxel.node_at(tree, self.depth) == Some(self)
            }
        }

        /// As `node [x, y, ...] at depth <level>`, since an address without its
        /// level names no place.
        impl fmt::Display for $node {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "node {:?} at depth {}",
                    self.address.to_array(),
                    self.depth.level()
                )
            }
        }

        /// Where a sample sits inside its node, each component in `[0, 1)`.
        ///
        /// Not ordered: one point being "less than" another has no meaning in
        /// more than one dimension, and picking an order would only invite it
        /// to be read as one.
        #[derive(Clone, Copy, PartialEq, Default, Debug)]
        pub struct $local($vector<f64>);

        impl $local {
            /// The fraction as given, or `None` if any component is outside
            /// `[0, 1)` or is NaN.
            pub fn new(fraction: $vector<f64>) -> Option<Self> {
                fraction
                    .all(|value| (0.0..1.0).contains(&value))
                    .then_some(Self(fraction))
            }

            /// The fraction brought into range: a component at or above one
            /// becomes the largest `f64` below one, and one below zero or NaN
            /// becomes zero.
            ///
            /// What the conversions use, since dividing a voxel offset by a
            /// node width can land exactly on a boundary through rounding.
            pub fn clamped(fraction: $vector<f64>) -> Self {
                Self(fraction.map(|component| {
                    if component.is_nan() || component < 0.0 {
                        0.0
                    } else if component >= 1.0 {
                        f64::from_bits(1.0f64.to_bits() - 1)
                    } else {
                        component
                    }
                }))
            }

            /// The low corner of the node: every component zero.
            pub const ORIGIN: Self = Self($vector::ZERO);

            /// The fraction as a vector, each component in `[0, 1)`.
            pub const fn fraction(self) -> $vector<f64> {
                self.0
            }

            /// The components in axis order.
            pub fn to_array(self) -> [f64; $dimension] {
                self.0.to_array()
            }

            /// The offset from one corner of the node to this point.
            ///
            /// What a gradient is dotted against in gradient noise: each corner
            /// contributes its own gradient times the offset from it, and the
            /// contributions are then interpolated.
            pub fn from_corner(self, corner: $vector<i128>) -> $vector<f64> {
                self.0 - corner.map(|component| component as f64)
            }
        }

        /// As `local [x, y, ...]`.
        impl fmt::Display for $local {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "local {:?}", self.0.to_array())
            }
        }
    };
}

implement_positions!(VoxelPosition2, NodePosition2, LocalPosition2, Vector2, 2);
implement_positions!(VoxelPosition3, NodePosition3, LocalPosition3, Vector3, 3);
implement_positions!(VoxelPosition4, NodePosition4, LocalPosition4, Vector4, 4);
