//! White noise: one independent value per octree node.
//!
//! Nothing is interpolated, so neighbouring nodes are unrelated and the field
//! is not continuous. That is the point of it. Use it where a decision belongs
//! to one node and owes nothing to its surroundings: which grass variant a
//! voxel grows, whether an ore vein starts here, a per-node jitter to break up
//! something too regular.
//!
//! It is the cheapest field in the module, one hash per sample, and it is the
//! only one that needs no neighbourhood at all: the answer depends on the
//! node's own coordinates and nothing else.
//!
//! At the tree's deepest level a node is one voxel, so that is the resolution
//! at which every voxel gets its own draw. Shallower depths hand a whole
//! subtree one value, which is what you want for deciding something once per
//! region rather than once per voxel.

use crate::random::mixing::mix128;
use crate::random::seed::Seed;
use crate::spatial::noise::{WHITE_DOMAIN, position_hash};
use crate::spatial::{Depth, TreeDepth, VoxelPosition2, VoxelPosition3, VoxelPosition4};
use crate::units::{NoiseValue, UnitValue};

/// Turns a hash into a value in `[-1, 1]`.
///
/// The top 53 bits are taken, which is every bit an `f64` can hold, and the
/// halving keeps the mapping exact: no rounding happens anywhere, so the values
/// stay uniform rather than clustering on the representable ones.
#[inline]
fn signed_f64(hash: u128) -> f64 {
    let unit: f64 = ((hash >> 75) as u64) as f64 * (1.0 / (1u64 << 53) as f64);

    unit * 2.0 - 1.0
}

/// Writes out `get_2d_white`, `get_3d_white` and `get_4d_white`, and the
/// `[0, 1)` variants beside them.
macro_rules! implement_white_noise {
    ($signed:ident, $unit:ident, $voxel:ident) => {
        /// One independent value in `[-1, 1]` for the node this position falls
        /// in, at the given octree depth.
        ///
        /// Positions arrive at the tree's deepest level, where a node is one
        /// voxel, so a node at `depth` is `tree_depth - depth` bits up from
        /// there. A depth deeper than the tree is clamped to single voxels, and
        /// depths below 0 keep doubling the node past the root, which hands the
        /// same value to a region larger than one tree.
        ///
        /// Two positions in the same node get the same value. Nothing is
        /// interpolated, so the field jumps at every node boundary.
        pub fn $signed(
            seed: Seed,
            tree_depth: TreeDepth,
            depth: Depth,
            position: $voxel,
        ) -> NoiseValue {
            let depth = tree_depth.clamp(depth);
            let node = position.node_at(tree_depth, depth).unwrap();

            NoiseValue::clamped(signed_f64(mix128(position_hash(
                seed.domain(WHITE_DOMAIN),
                depth,
                node.to_array(),
            ))))
        }

        /// As the signed version, in `[0, 1)`.
        ///
        /// This is the one to compare against a probability: `< 0.02` fires on
        /// one node in fifty, whatever the depth.
        pub fn $unit(
            seed: Seed,
            tree_depth: TreeDepth,
            depth: Depth,
            position: $voxel,
        ) -> UnitValue {
            UnitValue::clamped($signed(seed, tree_depth, depth, position).to_unit())
        }
    };
}

implement_white_noise!(get_2d_white, get_2d_white_unit, VoxelPosition2);
implement_white_noise!(get_3d_white, get_3d_white_unit, VoxelPosition3);
implement_white_noise!(get_4d_white, get_4d_white_unit, VoxelPosition4);
