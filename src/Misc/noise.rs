use crate::misc::linear::{Vector2, Vector3, Vector4};
use crate::misc::mixing::mix128;

/// One odd constant per axis. Giving each input its own multiplier is what
/// makes the key order-sensitive: a plain `x ^ y ^ z` is commutative and
/// self-inverse, so `(1, 2, 3)`, `(3, 2, 1)` and `(0, 0, 0)` would all fold to
/// the same value and the noise would mirror across the diagonals.
const X_PRIME: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835;
const Y_PRIME: u128 = 0xC2B2_AE3D_27D4_EB4F_1656_67B1_9E37_79F9;
const Z_PRIME: u128 = 0x27D4_EB2F_1658_67C5_85EB_CA77_C2B2_AE63;
const W_PRIME: u128 = 0xBF58_476D_1CE4_E5B9_94D0_49BB_1331_11EB;

/// Indexed by axis, so a coordinate keeps the same multiplier whatever the
/// dimension of the lattice it belongs to.
const AXIS_PRIMES: [u128; 4] = [X_PRIME, Y_PRIME, Z_PRIME, W_PRIME];

const LEVEL_PRIME: u128 = 0xFF51_AFD7_ED55_8CCD_D6E8_FEB8_6659_FD93;
const DIMENSION_PRIME: u128 = 0x2545_F491_4F6C_DD1D_A24B_AA9B_4C6D_E44B;

fn get_2d_noise(seed: u128, depth: i8, octaves: u8, position: Vector2<i128>) -> f64 {
    let mut total: f64 = 0.0;
    for i in 0..octaves {
        let octave_level: i8 = depth + i as i8;
        let octave_position: Vector2<i128> = position >> (i as u32);
        let block_internal_position: Vector2<i128> = position - (octave_position << (i as u32));
        let relative_position: Vector2<f64> = block_internal_position.map(|x| x as f64 / (1 << i) as f64);
        let corner_offsets: [Vector2<i128>; 4] = cell_offsets_2d();

        let mut corner_values: [f64; 4] = [0.0; 4];
        for j in 0..4 {
            let corner_position: Vector2<i128> = octave_position + corner_offsets[j];
            let gradient: Vector2<f64> = random_unit_vector_2d(seed, octave_level + depth, corner_position);
            let distance: Vector2<f64> = relative_position - corner_offsets[j].map(|x| x as f64);
            let contribution: f64 = gradient.dot(distance);
            corner_values[j] = contribution;
        }

        // The amplitude of each octave is halved, so the sum converges to a
        // finite value. The first octave is full strength, and the second is
        // half that, and the third is half of that again, and so on.
        //total += octave_contribution * 0.5_f64.powi(i as i32);
    }
    total
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
fn position_hash<const N: usize>(seed: u128, level: i8, position: [i128; N]) -> u128 {
    const { assert!(N <= AXIS_PRIMES.len(), "no multiplier for that many axes") };

    let mut key: u128 = seed
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
fn random_unit_vector_2d(seed: u128, level: i8, position: Vector2<i128>) -> Vector2<f64> {
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
fn random_unit_vector_3d(seed: u128, level: i8, position: Vector3<i128>) -> Vector3<f64> {
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
fn random_unit_vector_4d(seed: u128, level: i8, position: Vector4<i128>) -> Vector4<f64> {
    let mut hash: u128 = position_hash(seed, level, position.to_array());

    let (x, y, first_squared_radius): (f64, f64, f64) = unit_disc_point(&mut hash);
    let (z, w, second_squared_radius): (f64, f64, f64) = unit_disc_point(&mut hash);

    let factor: f64 =
        ((1.0 - first_squared_radius) / second_squared_radius).sqrt();

    Vector4::new(x, y, z * factor, w * factor)
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
