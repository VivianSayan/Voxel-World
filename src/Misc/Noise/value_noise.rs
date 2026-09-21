//! Value noise: a random value at each node corner, interpolated across the
//! node with the same fade curve the gradient noise uses.
//!
//! The cheap relative of `gradient_noise`. It reads one hash per corner instead
//! of a rejection-sampled direction, which measures about 1.6 times faster in
//! 2D and 3 times in 3D, where the corner count doubles. What it gives up is
//! character rather than quality: the extrema of value noise sit on the lattice
//! instead of between it, so at low octave counts the cell grid is easier to
//! pick out. Both fields measure close to isotropic, 0.95 against 1.04 on the
//! ratio of change along an axis to change along the diagonal.
//!
//! Two things it does better here, both from the same cause. A corner carries a
//! value rather than a direction, so a sample landing exactly on a corner
//! returns that corner's value instead of zero:
//!
//! - No dead grid. Gradient noise is zero at every lattice corner, and because
//!   a coarse corner is a corner of every finer octave too, those zeros stack
//!   into a regular lattice of flat spots. This has none.
//! - The deepest level works. Gradient noise contributes nothing at one voxel
//!   per cell, because every sample is then a corner. Here that level degrades
//!   gracefully into white noise, which is the correct limit.
//!
//! The output covers `[-1, 1]` on its own, so unlike the gradient noise there
//! is no scale factor to undo a shortfall.

use crate::misc::linear::{Vector2, Vector3, Vector4};
use crate::misc::mixing::mix128;
use crate::misc::noise::{
    VALUE_DOMAIN, cell_offsets_2d, cell_offsets_3d, cell_offsets_4d, fade, interpolate,
    position_hash,
};
use crate::misc::seed::Seed;

/// The value a lattice corner carries, in `[-1, 1]`.
///
/// The top 53 bits are taken, which is every bit an `f64` holds, so the mapping
/// is exact and the values stay uniform.
#[inline]
fn corner_value<const N: usize>(seed: Seed, depth: i8, corner: [i128; N]) -> f64 {
    let hash: u128 = mix128(position_hash(seed, depth, corner));
    let unit: f64 = ((hash >> 75) as u64) as f64 * (1.0 / (1u64 << 53) as f64);

    unit * 2.0 - 1.0
}

/// Writes out `get_2d_value`, `get_3d_value` and `get_4d_value`.
macro_rules! implement_value_noise {
    ($name:ident, $vector:ident, $offsets:ident, $corner_count:literal) => {
        /// Fractal value noise at a point, summed over consecutive octree
        /// depths.
        ///
        /// Positions arrive at the tree's deepest level, where a node is one
        /// voxel and nothing is shifted, so a node at depth `d` is
        /// `tree_depth - d` bits up from there and `2^(tree_depth - d)` voxels
        /// across.
        ///
        /// `coarsest_depth` is where the first and heaviest octave sits, and
        /// `extra_octaves` is how many further levels down to add, so passing 0
        /// samples `coarsest_depth` on its own. Each level down halves both the
        /// node and the amplitude, so the first octave shapes the terrain and
        /// the rest only add detail. A negative `coarsest_depth` keeps doubling
        /// the node past the root, for features larger than one tree.
        ///
        /// The result is in `[-1, 1]` whatever the octave count. `tree_depth`
        /// is the floor and is itself sampled: at that depth a node is one
        /// voxel, every sample sits on a corner, and the octave reduces to
        /// white noise rather than to nothing. Octaves past the floor stop.
        pub fn $name(
            seed: Seed,
            tree_depth: u8,
            coarsest_depth: i8,
            extra_octaves: u8,
            position: $vector<i128>,
        ) -> f64 {
            // Tagged once, outside the loop, so the per-corner path is the
            // same work it was when this took a bare `u128`.
            let seed: Seed = seed.domain(VALUE_DOMAIN);

            let mut total: f64 = 0.0;
            let mut amplitude: f64 = 1.0;
            let mut total_amplitude: f64 = 0.0;

            for octave in 0..=extra_octaves {
                // Past the deepest level a node would be narrower than a voxel,
                // and depths past what an `i8` holds have no values to draw.
                let depth: i16 = coarsest_depth as i16 + octave as i16;

                if depth > tree_depth as i16 || depth > i8::MAX as i16 {
                    break;
                }

                // Above the root the nodes keep doubling, so a shallow enough
                // `coarsest_depth` asks for one wider than the coordinate
                // range. An `i128` cannot be shifted further than its own
                // width, and by then one node already covers everything.
                let depth: i8 = depth as i8;
                let shift: u32 = (tree_depth as i16 - depth as i16)
                    .min(i128::BITS as i16 - 1) as u32;

                let node: $vector<i128> = position >> shift;
                let fraction: $vector<f64> = (position - (node << shift))
                    .map(|c| c as f64 / (1u128 << shift) as f64);

                // Keyed on the octree depth itself, so every call reaching a
                // depth draws the same values and the levels of detail agree.
                let mut corners: [f64; $corner_count] = $offsets().map(|offset| {
                    corner_value(seed, depth, (node + offset).to_array())
                });

                // A weighted average of values already in `[-1, 1]` stays in
                // `[-1, 1]`, so there is nothing to rescale.
                total += interpolate(&mut corners, &fraction.map(fade).to_array())
                    * amplitude;

                total_amplitude += amplitude;
                amplitude *= 0.5;
            }

            // Without this the result would grow with the octave count rather
            // than staying in `[-1, 1]`.
            if total_amplitude == 0.0 {
                return 0.0;
            }

            total / total_amplitude
        }
    };
}

implement_value_noise!(get_2d_value, Vector2, cell_offsets_2d, 4);
implement_value_noise!(get_3d_value, Vector3, cell_offsets_3d, 8);
implement_value_noise!(get_4d_value, Vector4, cell_offsets_4d, 16);
