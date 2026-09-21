//! Cellular noise laid out on the octree.
//!
//! One feature point per octree node: the node a sample falls in is exactly
//! `position >> (tree_depth - depth)`, so the cells of this noise and the nodes
//! of the tree are the same partition of space. A sample reports the node that
//! owns it, how far it sits from that node's feature point, and how far from the
//! next one along.
//!
//! It keeps the properties `noise` has, for the same reasons:
//!
//! - **Stateless.** A sample reads the hashes of the nodes around it and
//!   nothing else, so any point can be evaluated without having generated its
//!   neighbours, in any order, from any thread.
//! - **Consistent between levels of detail.** The feature point of a node is
//!   keyed on the node's absolute depth, so a node has the same feature point
//!   however a query arrives at it. A coarse pass and a fine pass agree.
//! - **Unbounded.** Coordinates are hashed rather than looked up in a
//!   permutation table, so there is no period to repeat at.
//! - **Reproducible across machines.** Only `+ - * /`, comparison and `sqrt`,
//!   all of which IEEE-754 pins to one result. No trigonometry, so no
//!   dependence on the platform's libm.
//!
//! Unlike gradient noise this stays useful at every depth. Perlin is zero
//! wherever a sample lands on a lattice corner, which makes its deepest level
//! useless; a feature point sits somewhere inside its node rather than on a
//! corner, so the deepest level here is as meaningful as any other.

use crate::misc::linear::Vector2;
use crate::misc::mixing::mix128;
use crate::misc::noise::{CELLULAR_DOMAIN, CELL_VALUE_DOMAIN, position_hash};
use crate::misc::seed::Seed;

/// How far out to search, in cells. Everything outside a ring this wide is too
/// far away to win.
///
/// A sample sits somewhere in its own cell, so its own cell's feature point is
/// at most `sqrt(2)` cell widths away, while the nearest possible point in a
/// cell `k` rings out is more than `k - 1` away. Searching two rings therefore
/// cannot miss the closest point: nothing beyond is closer than 2, and the
/// answer is never worse than 1.415.
const SEARCH_RADIUS: i128 = 2;

/// What one sample of the field found.
///
/// Distances are in cell widths, not voxels, so they do not change meaning when
/// the depth changes. Multiply by `1 << (tree_depth - depth)` for voxels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell2d {
    /// The octree node whose feature point is nearest, in that depth's node
    /// coordinates. This is the cell the sample belongs to, and it is a node
    /// address the rest of the world can use directly.
    pub node: Vector2<i128>,

    /// Distance to the nearest feature point. Runs from 0 up to about 1.
    pub nearest: f64,

    /// Distance to the second nearest, which always belongs to another node.
    pub second: f64,

    /// A value in `[0, 1)` fixed for the owning node, for choosing whatever the
    /// cell stands for. Unrelated to the distances and to the node's position.
    pub value: f64,
}

impl Cell2d {
    /// How far inside its cell the sample is, as the gap between the two
    /// nearest feature points. Zero exactly on the border between two cells and
    /// largest deep inside one, so thresholding it draws the cell walls.
    ///
    /// This is the one to use for cracks, rivers and region borders: unlike
    /// `nearest` it has no peak in the middle of a cell to be mistaken for an
    /// edge, and it is continuous across the border.
    pub fn edge_distance(self) -> f64 {
        self.second - self.nearest
    }
}

/// Where a node's feature point sits inside it, each component in `[0, 1)`.
///
/// Public because the world may want a node's point without sampling a
/// position: it is the node's own landmark, and the same point every query
/// sees.
pub fn feature_point_2d(seed: Seed, depth: i8, node: Vector2<i128>) -> Vector2<f64> {
    let hash: u128 = mix128(position_hash(
        seed.domain(CELLULAR_DOMAIN),
        depth,
        node.to_array(),
    ));

    Vector2::new(unit_f64(hash as u64), unit_f64((hash >> 64) as u64))
}

/// The value a node carries, in `[0, 1)`.
pub fn node_value_2d(seed: Seed, depth: i8, node: Vector2<i128>) -> f64 {
    let hash: u128 = mix128(position_hash(
        seed.domain(CELL_VALUE_DOMAIN),
        depth,
        node.to_array(),
    ));

    unit_f64(hash as u64)
}

/// Uniform in `[0, 1)` with 53 bits, the same conversion `Random` uses.
#[inline]
fn unit_f64(value: u64) -> f64 {
    (value >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Samples the field at one point and one octree depth.
///
/// `depth` is the octree level whose nodes are the cells. Positions arrive at
/// the tree's deepest level, where a node is one voxel, so a node at `depth` is
/// `tree_depth - depth` bits up from there. Depths below 0 keep doubling the
/// cell past the root, for regions larger than one tree, and a `depth` deeper
/// than the tree is clamped to single voxel cells.
pub fn get_2d_cellular(
    seed: Seed,
    tree_depth: u8,
    depth: i8,
    position: Vector2<i128>,
) -> Cell2d {
    // Clamped rather than guarded: a cell below one voxel means nothing, and a
    // cell wider than the coordinate range cannot be shifted for.
    let shift: u32 = (tree_depth as i16 - depth as i16)
        .clamp(0, i128::BITS as i16 - 1) as u32;

    let node: Vector2<i128> = position >> shift;

    // Where the sample sits inside its own node, in cell widths.
    let fraction: Vector2<f64> = (position - (node << shift))
        .map(|c| c as f64 / (1u128 << shift) as f64);

    let mut nearest_squared: f64 = f64::INFINITY;
    let mut second_squared: f64 = f64::INFINITY;
    let mut owner: Vector2<i128> = node;

    for x in -SEARCH_RADIUS..=SEARCH_RADIUS {
        for y in -SEARCH_RADIUS..=SEARCH_RADIUS {
            let neighbour: Vector2<i128> = node + Vector2::new(x, y);
            let point: Vector2<f64> = feature_point_2d(seed, depth, neighbour);

            // The neighbour's point, measured from this sample, in cell widths.
            let offset: Vector2<f64> = Vector2::new(x as f64, y as f64);
            let delta: Vector2<f64> = offset + point - fraction;

            // Compared squared, so the loop needs no square roots. Only the two
            // that survive are ever rooted.
            let distance_squared: f64 = delta.norm_squared();

            if distance_squared < nearest_squared {
                second_squared = nearest_squared;
                nearest_squared = distance_squared;
                owner = neighbour;
            } else if distance_squared < second_squared {
                second_squared = distance_squared;
            }
        }
    }

    Cell2d {
        node: owner,
        nearest: nearest_squared.sqrt(),
        second: second_squared.sqrt(),
        value: node_value_2d(seed, depth, owner),
    }
}
