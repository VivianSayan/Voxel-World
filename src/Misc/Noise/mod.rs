//! Noise fields laid out on the octree.
//!
//! Every field here shares the same contract, and this module holds the pieces
//! that make it hold:
//!
//! - **Octree aligned.** A sample's cell at depth `d` is
//!   `position >> (tree_depth - d)`, which is the octree node it falls in.
//!   Positions arrive at the tree's deepest level, where a node is one voxel.
//! - **Stateless.** A sample reads hashes of the nodes around it and nothing
//!   else, so any point evaluates without its neighbours having been
//!   generated, in any order, from any thread.
//! - **Consistent between levels of detail.** Everything random is keyed on the
//!   absolute octree depth, so a node looks the same however a query reaches
//!   it. A coarse pass and a fine pass agree on the depths they share.
//! - **Unbounded.** Coordinates are hashed rather than looked up in a
//!   permutation table, so there is no period to repeat at, and `i128`
//!   coordinates are carried whole.
//! - **Reproducible across machines.** Only `+ - * /`, comparison and `sqrt`,
//!   which IEEE-754 pins to one result. No trigonometry, so no dependence on
//!   the platform's libm.
//!
//! The fields:
//!
//! - [`white_noise`]: one independent value per node. No interpolation, so it
//!   is not continuous; for per-voxel scatter and variant picking.
//! - [`value_noise`]: random values at the node corners, interpolated. Cheap,
//!   and unlike gradient noise it is not zero at the corners.
//! - [`gradient_noise`]: Perlin. Random directions at the node corners, dotted
//!   against the offset to the sample. Smoother and less axis-aligned than
//!   value noise, at four times the cost.
//! - [`cellular`]: one feature point per node, reporting the nearest and the
//!   node that owns it. For regions, biomes and cell walls.
//!
//! Each field mixes its own domain constant into the seed, so that a node's
//! gradient, its value and its feature point are unrelated draws rather than
//! three views of one hash.

use crate::misc::linear::{Vector2, Vector3, Vector4};
use crate::misc::seed::Seed;

pub mod cellular;
pub mod gradient_noise;
pub mod value_noise;
pub mod white_noise;

/// Separates the fields from one another. Each is applied to the seed with
/// [`Seed::domain`] before anything else, so the same node at the same depth
/// hashes differently for each field.
pub(crate) const GRADIENT_DOMAIN: u128 = 0x0FC1_9DC6_8B8C_D5B5_2FFD_72DB_D01A_DFB7;
pub(crate) const VALUE_DOMAIN: u128 = 0x243F_6A88_85A3_08D3_1319_8A2E_0370_7344;
pub(crate) const WHITE_DOMAIN: u128 = 0x4528_21E6_38D0_1377_BE54_66CF_34E9_0C6C;
pub(crate) const CELLULAR_DOMAIN: u128 = 0xA076_1D64_78BD_642F_E703_7ED1_A0B4_28DB;
pub(crate) const CELL_VALUE_DOMAIN: u128 = 0x8EBC_6AF0_9C88_C6E3_5899_65CD_1B3E_9E7F;

/// One odd constant per axis. Giving each input its own multiplier is what
/// makes the key order-sensitive: a plain `x ^ y ^ z` is commutative and
/// self-inverse, so `(1, 2, 3)`, `(3, 2, 1)` and `(0, 0, 0)` would all fold to
/// the same value and the noise would mirror across the diagonals.
pub(crate) const X_PRIME: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835;
pub(crate) const Y_PRIME: u128 = 0xC2B2_AE3D_27D4_EB4F_1656_67B1_9E37_79F9;
pub(crate) const Z_PRIME: u128 = 0x27D4_EB2F_1658_67C5_85EB_CA77_C2B2_AE63;
pub(crate) const W_PRIME: u128 = 0xBF58_476D_1CE4_E5B9_94D0_49BB_1331_11EB;

/// Indexed by axis, so a coordinate keeps the same multiplier whatever the
/// dimension of the lattice it belongs to.
pub(crate) const AXIS_PRIMES: [u128; 4] = [X_PRIME, Y_PRIME, Z_PRIME, W_PRIME];

pub(crate) const LEVEL_PRIME: u128 = 0xFF51_AFD7_ED55_8CCD_D6E8_FEB8_6659_FD93;
pub(crate) const DIMENSION_PRIME: u128 = 0x2545_F491_4F6C_DD1D_A24B_AA9B_4C6D_E44B;

/// Perlin's fade curve, `6t^5 - 15t^4 + 10t^3`.
///
/// Interpolating the corners straight would leave the value continuous but not
/// its slope, and the crease along every cell boundary is plainly visible as a
/// grid in the terrain. This curve is flat to the second derivative at both
/// ends, so the cells meet smoothly and the grid disappears.
#[inline]
pub(crate) fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
pub(crate) fn lerp(from: f64, to: f64, t: f64) -> f64 {
    from + (to - from) * t
}

/// Collapses the corner values of one cell down to a single sample.
///
/// Two corners whose indices differ only in bit `d` are the ends of an edge
/// along axis `d`, and `cell_offsets_*` lays them out so that those two are
/// adjacent once the lower axes are gone. Each pass therefore interpolates
/// neighbouring entries, drops the low bit of the index and halves the array,
/// which walks the axes in order and needs the weights in the same order.
pub(crate) fn interpolate(corners: &mut [f64], weights: &[f64]) -> f64 {
    let mut remaining: usize = corners.len();

    for &weight in weights {
        remaining /= 2;

        // Reading ahead of where it writes, so the source values are always
        // still the previous pass's.
        for i in 0..remaining {
            corners[i] = lerp(corners[2 * i], corners[2 * i + 1], weight);
        }
    }

    corners[0]
}

/// Combines a seed, an octree level and a lattice position into one key for
/// `mix128` to spread.
///
/// `level` is the octree depth the sample belongs to, so that one lattice point
/// draws an unrelated vector at every frequency. Without it a coarse octave and
/// a fine one would reuse the same gradient wherever their lattices meet, and
/// the octaves would reinforce each other instead of adding detail. It is
/// signed because levels run in both directions from the base scale.
///
/// The dimension goes in as well, so that the 2D field at `(x, y)` is not a
/// relative of the 3D field at `(x, y, 0)`. Both draw from the same hash, and
/// without this the plane `z = 0` of one would be a function of the other.
///
/// The scaled inputs are summed rather than xor-ed. Xor is carry-less, and a
/// negative value sign-extends into a run of high one bits, so two negative
/// coordinates would cancel each other's upper half and whole sheets of the
/// lattice below the origin would share a key. Addition carries, so nothing
/// cancels: a collision would need `dl*L + dx*X + dy*Y + dz*Z` to be a multiple
/// of 2^128, and for constants this size the smallest such relation has
/// coefficients far larger than any coordinate the world will ever use.
///
/// The casts to `u128` reinterpret the two's-complement bits rather than
/// saturating, so they are bijective and lose nothing.
#[inline]
pub(crate) fn position_hash<const N: usize>(
    seed: Seed,
    level: i8,
    position: [i128; N],
) -> u128 {
    const { assert!(N <= AXIS_PRIMES.len(), "no multiplier for that many axes") };

    let mut key: u128 = seed
        .value()
        .wrapping_add((level as i128 as u128).wrapping_mul(LEVEL_PRIME))
        .wrapping_add((N as u128).wrapping_mul(DIMENSION_PRIME));

    let mut axis: usize = 0;

    while axis < N {
        key = key
            .wrapping_add((position[axis] as u128).wrapping_mul(AXIS_PRIMES[axis]));

        axis += 1;
    }

    key
}

// ---------------------------------------------------------------------------
// Lattice cells
// ---------------------------------------------------------------------------

/// Writes out the `cell_offsets_*` and `cell_corners_*` pairs.
///
/// Each axis is given the bit of the corner index that carries its offset, so
/// the bodies below are the same shape whatever the dimension.
macro_rules! implement_cell_corners {
    (
        $offsets:ident,
        $corners:ident,
        $vector:ident,
        $dimension:literal,
        [$($axis:ident: $bit:literal),+]
    ) => {
        /// What separates each corner of a lattice cell from the cell's lowest
        /// corner: every combination of `0` and `1` across the axes.
        ///
        /// This is `$corners` with the position left out, for callers that want
        /// the displacement itself rather than where it lands. The two share an
        /// order, so the corner at an index is always that index's offset added
        /// to the position, and a fractional sample position minus the offset at
        /// an index is the vector from that corner to the sample, which is what
        /// a gradient is dotted against.
        ///
        /// The index convention is the one described on `$corners`.
        pub fn $offsets() -> [$vector<i128>; 1 << $dimension] {
            std::array::from_fn(|corner| {
                $vector::new($(((corner >> $bit) & 1) as i128),+)
            })
        }

        /// The corners of the lattice cell whose lowest corner is `position`:
        /// every combination of `+0` and `+1` across the axes.
        ///
        /// Bit `d` of the index is the offset along axis `d`, so the index of a
        /// corner is the set of axes it is displaced on. Index 0 is `position`
        /// itself and the last index is `position` displaced on every axis, and
        /// `index & 1` recovers the x offset, `(index >> 1) & 1` the y offset,
        /// and so on.
        ///
        /// Two corners whose indices differ only in bit `d` are the ends of one
        /// edge along axis `d`, so interpolating one against the other for every
        /// such pair collapses axis `d` and halves the array. Interpolating
        /// adjacent entries pairwise takes axis 0 first, and each pass drops the
        /// low bit and shifts the rest down, so repeating it walks the axes in
        /// order. Folding the array in half instead pairs across the top bit,
        /// which walks them backwards and wants the interpolants reversed.
        ///
        /// The offsets wrap rather than panicking, so that a cell at the very
        /// edge of the coordinate range behaves the same in debug and release
        /// builds. The world does not reach far enough for it to come up.
        pub fn $corners(position: $vector<i128>) -> [$vector<i128>; 1 << $dimension] {
            // Written as the offsets displaced by the position, rather than
            // built from the index again, so that the two cannot drift apart.
            $offsets().map(|offset| {
                $vector::new($(position.$axis.wrapping_add(offset.$axis)),+)
            })
        }
    };
}

implement_cell_corners!(cell_offsets_2d, cell_corners_2d, Vector2, 2, [x: 0, y: 1]);
implement_cell_corners!(cell_offsets_3d, cell_corners_3d, Vector3, 3, [x: 0, y: 1, z: 2]);
implement_cell_corners!(
    cell_offsets_4d,
    cell_corners_4d,
    Vector4,
    4,
    [x: 0, y: 1, z: 2, w: 3]
);
