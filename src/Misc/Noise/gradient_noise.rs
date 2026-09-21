//! Perlin gradient noise: a random direction at each node corner, dotted
//! against the offset from that corner to the sample, then interpolated.
//!
//! Smoother and less axis-aligned than `value_noise`, at roughly four times the
//! cost. One caveat specific to this field: a gradient dotted against a zero
//! offset is zero, so the value is exactly 0 wherever a sample lands on a node
//! corner. Coarse corners are corners of every finer octave too, so those zeros
//! stack into a regular grid of flat spots, and the deepest level (one voxel
//! per cell, every sample on a corner) contributes nothing at all.
//! `value_noise` has neither problem.

use crate::misc::linear::{Vector2, Vector3, Vector4};
use crate::misc::mixing::mix128;
use crate::misc::noise::{
    GRADIENT_DOMAIN, cell_offsets_2d, cell_offsets_3d, cell_offsets_4d, fade, interpolate,
    position_hash,
};
use crate::misc::seed::Seed;

/// Scales one octave from the range unit gradients actually reach to `[-1, 1]`.
/// In `d` dimensions a corner can contribute at most `sqrt(d) / 2`, so without
/// this the top of the range would never be used.
const NORMALIZE_2D: f64 = 1.414_213_562_373_094_9;
const NORMALIZE_3D: f64 = 1.154_700_538_379_251_7;
const NORMALIZE_4D: f64 = 1.0;

/// Writes out `get_2d_noise`, `get_3d_noise` and `get_4d_noise`.
macro_rules! implement_fractal_noise {
    (
        $name:ident,
        $at_depth:ident,
        $vector:ident,
        $offsets:ident,
        $gradient:ident,
        $corner_count:literal,
        $normalize:ident
    ) => {
        /// Fractal noise at a point, summed over consecutive octree depths.
        ///
        /// Positions arrive at the tree's deepest level, where a cell is one
        /// voxel and nothing is shifted, so a cell at depth `d` is
        /// `tree_depth - d` bits up from there and is `2^(tree_depth - d)`
        /// voxels across.
        ///
        /// `frequency` is the everyday scale control. Each step up halves the
        /// cell and each step down doubles it, which is a bit shift and nothing
        /// more, so a higher number packs more detail into the same ground.
        ///
        /// The first and heaviest octave sits at the root of one tree, depth
        /// 0. To pin it to some other octree level instead, use
        #[doc = concat!("/// [`", stringify!($at_depth), "`].")]
        ///
        /// `extra_octaves` is how many further levels down to add, so passing 0
        /// samples one octave on its own. Each level down halves both the cell
        /// and the amplitude, so the first octave shapes the terrain and the
        /// rest only add detail.
        ///
        /// The result is in `[-1, 1]` whatever the octave count. `tree_depth` is
        /// the floor: nothing deeper is sampled, and the deepest level itself
        /// contributes nothing, because a cell one voxel across puts every
        /// sample on a lattice corner and gradient noise is zero at every corner
        /// by construction. So `coarsest_depth == tree_depth` returns 0.0, and
        /// asking for more octaves than the tree has left stops early rather
        /// than summing zeroes. A `frequency` past the floor returns 0.0 for
        /// the same reason.
        pub fn $name(
            seed: Seed,
            tree_depth: u8,
            frequency: i8,
            extra_octaves: u8,
            position: $vector<i128>,
        ) -> f64 {
            $at_depth(seed, tree_depth, 0, frequency, extra_octaves, position)
        }

        /// As the shorter form, with the first octave pinned to a named octree
        /// level rather than to the root.
        ///
        /// `coarsest_depth` and `frequency` add together and are
        /// interchangeable arithmetically: `(4, 0)` samples exactly the field
        /// `(0, 4)` does. They are separate because they answer different
        /// questions. `coarsest_depth` says which octree level the field is
        /// pinned to, which matters when it has to line up with something
        /// structural; `frequency` says how fine you want it, which is what
        /// gets adjusted while tuning. Reach for the shorter form and
        /// `frequency` unless a specific level is the point.
        ///
        /// A negative `coarsest_depth` keeps doubling past the root, giving
        /// cells that span whole trees for features larger than a single
        /// octree.
        pub fn $at_depth(
            seed: Seed,
            tree_depth: u8,
            coarsest_depth: i8,
            frequency: i8,
            extra_octaves: u8,
            position: $vector<i128>,
        ) -> f64 {
            // Tagged once, outside the loop, so the per-corner path is the
            // same work it was when this took a bare `u128`.
            let seed: Seed = seed.domain(GRADIENT_DOMAIN);

            // Both shift the lattice by whole powers of two, so the first
            // octave sits at their sum.
            let base_depth: i16 = coarsest_depth as i16 + frequency as i16;

            let mut total: f64 = 0.0;
            let mut amplitude: f64 = 1.0;
            let mut total_amplitude: f64 = 0.0;

            for octave in 0..=extra_octaves {
                // Depths past what an `i8` holds have no gradients to draw.
                let depth: i16 = base_depth + octave as i16;
                let shift: i16 = tree_depth as i16 - depth;

                if shift <= 0 || depth > i8::MAX as i16 || depth < i8::MIN as i16 {
                    break;
                }

                // Above the root the cells keep doubling, so a low enough
                // `coarsest_depth` or `frequency` asks for a cell wider than
                // the coordinate range. An `i128` cannot be shifted further than its own
                // width, and by then one cell already covers everything.
                let depth: i8 = depth as i8;
                let shift: u32 = shift.min(i128::BITS as i16 - 1) as u32;

                let cell: $vector<i128> = position >> shift;
                let fraction: $vector<f64> = (position - (cell << shift))
                    .map(|c| c as f64 / (1u128 << shift) as f64);

                // Keyed on the octree depth itself, so every call reaching a
                // depth draws the same gradients and the levels of detail agree.
                let mut corners: [f64; $corner_count] = $offsets().map(|offset| {
                    $gradient(seed, depth, cell + offset)
                        .dot(fraction - offset.map(|c| c as f64))
                });

                total += interpolate(&mut corners, &fraction.map(fade).to_array())
                    * $normalize
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

implement_fractal_noise!(
    get_2d_noise, get_2d_noise_at_depth,
    Vector2, cell_offsets_2d, random_unit_vector_2d, 4, NORMALIZE_2D
);
implement_fractal_noise!(
    get_3d_noise, get_3d_noise_at_depth,
    Vector3, cell_offsets_3d, random_unit_vector_3d, 8, NORMALIZE_3D
);
implement_fractal_noise!(
    get_4d_noise, get_4d_noise_at_depth,
    Vector4, cell_offsets_4d, random_unit_vector_4d, 16, NORMALIZE_4D
);

/// Splits one hash into two independent values in `[-1, 1]`.
#[inline]
fn signed_pair(hash: u128) -> (f64, f64) {
    let high: u64 = (hash >> 64) as u64;
    let low: u64 = hash as u64;

    (
        low as f64 / (u64::MAX as f64) * 2.0 - 1.0,
        high as f64 / (u64::MAX as f64) * 2.0 - 1.0,
    )
}

/// Draws a point uniformly inside the unit disc and returns it with its squared
/// radius. Points are taken from the enclosing square and the corners are
/// thrown away, which keeps about pi/4 of them, so this reads 1.27 hashes per
/// call on average.
///
/// `hash` is advanced past everything the draw consumed, so a second call
/// continues the stream rather than repeating it.
///
/// The origin is rejected along with the corners. Callers that divide by the
/// radius need that, and the ones that do not lose a region of measure zero.
#[inline]
fn unit_disc_point(hash: &mut u128) -> (f64, f64, f64) {
    loop {
        *hash = mix128(*hash);

        let (u, v): (f64, f64) = signed_pair(*hash);
        let squared_radius: f64 = u * u + v * v;

        if squared_radius < 1.0 && squared_radius != 0.0 {
            return (u, v, squared_radius);
        }
    }
}

/// A direction on the unit circle, drawn from the seed, the level and the
/// lattice point alone.
///
/// The obvious way to write this is to draw an angle and take its sine and
/// cosine. That is avoided on purpose: the trigonometric functions are not
/// correctly rounded, so they may differ between platforms and between libm
/// versions, and a world would then generate differently on two machines from
/// the same seed. Everything here is addition, multiplication, division and
/// `sqrt`, all of which IEEE-754 pins to a single result.
fn random_unit_vector_2d(seed: Seed, level: i8, position: Vector2<i128>) -> Vector2<f64> {
    let mut hash: u128 = position_hash(seed, level, position.to_array());

    // A disc is rotationally symmetric, so pushing a point out to the rim
    // leaves the angle uniform.
    let (u, v, squared_radius): (f64, f64, f64) = unit_disc_point(&mut hash);
    let scale: f64 = 1.0 / squared_radius.sqrt();

    Vector2::new(u * scale, v * scale)
}

/// A direction on the unit sphere, by Marsaglia's method: a point from the unit
/// disc lifts to the sphere by way of `z = 1 - 2s`, which spreads the disc over
/// the surface without bunching anything at the poles.
fn random_unit_vector_3d(seed: Seed, level: i8, position: Vector3<i128>) -> Vector3<f64> {
    let mut hash: u128 = position_hash(seed, level, position.to_array());

    let (u, v, squared_radius): (f64, f64, f64) = unit_disc_point(&mut hash);
    let factor: f64 = 2.0 * (1.0 - squared_radius).sqrt();

    Vector3::new(
        u * factor,
        v * factor,
        1.0 - 2.0 * squared_radius,
    )
}

/// A direction on the unit 3-sphere, by Marsaglia's four-dimensional method.
///
/// Two disc points are drawn. The first is used as it stands, and the second is
/// scaled so that the four components together come to length 1: the squared
/// norm is `s + t * (1 - s) / t`, which is 1 whatever the two radii were.
fn random_unit_vector_4d(seed: Seed, level: i8, position: Vector4<i128>) -> Vector4<f64> {
    let mut hash: u128 = position_hash(seed, level, position.to_array());

    let (x, y, first_squared_radius): (f64, f64, f64) = unit_disc_point(&mut hash);
    let (z, w, second_squared_radius): (f64, f64, f64) = unit_disc_point(&mut hash);

    let factor: f64 =
        ((1.0 - first_squared_radius) / second_squared_radius).sqrt();

    Vector4::new(x, y, z * factor, w * factor)
}

