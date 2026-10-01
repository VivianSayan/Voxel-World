//! CORDIC: circular sine, cosine and arctangent in integers alone.
//!
//! # Why one algorithm for four functions
//!
//! CORDIC rotates a vector by a fixed sequence of angles whose tangents are exact
//! powers of two, so each rotation is a shift and an add. Running it *forwards* from
//! a known angle leaves the vector at `(cos, sin)`; running it *backwards*, driving
//! the vector onto the x-axis, leaves behind the angle it started at, which is
//! `atan2`. One loop, one table, one error bound, and no multiplication in the inner
//! step at all.
//!
//! That matters here for a reason beyond speed. Every alternative — a minimax
//! polynomial, a table with interpolation, a series — needs its own coefficients, its
//! own argument range and its own accuracy argument, and each is a separate chance to
//! be subtly wrong. CORDIC gives all four functions one proof: after `n` steps the
//! remaining angle is below `atan(2⁻ⁿ)`, because each step more than halves it.
//!
//! # Precision
//!
//! Everything here works at [`BITS`] fractional bits in an `i128`, well above the 64
//! a [`FixedPoint`](super::FixedPoint) can ask for, and the caller rounds once on the
//! way out. With [`ITERATIONS`] steps the angle residue is below `2⁻⁷²` and the
//! accumulated rounding is under `ITERATIONS × 2⁻⁹⁶`, so both sit far beneath the
//! last bit of any supported layout.
//!
//! # Determinism
//!
//! Integer shifts, adds and comparisons only. No floating point is involved at any
//! point, so the results are bit-identical on every platform, every optimisation
//! level and every compiler version.

use super::fixed::wide::{self, U256};

/// Fractional bits of the internal format.
///
/// Matches the reference constants in [`fixed`](super::fixed), so the π and τ values
/// there are reused here rather than written twice.
pub(crate) const BITS: u32 = 96;

/// Rotation steps taken.
///
/// Chosen so the residual angle `atan(2⁻⁷²)` is below the last bit of the widest
/// layout, with eight bits to spare.
pub(crate) const ITERATIONS: usize = 72;

/// τ at [`BITS`] fractional bits. Exactly four times [`HALF_PI`], which is what lets
/// the quadrant split below be exact.
pub(crate) const TAU: i128 = 497_805_226_624_462_170_461_043_889_244;

/// π/2 at [`BITS`] fractional bits.
pub(crate) const HALF_PI: i128 = 124451306656115542615260972311;

/// 1/τ at [`BITS`] fractional bits, for the argument reduction.
const INVERSE_TAU: i128 = 12609553696233175933233255924;

/// The reciprocal of the CORDIC gain `∏ √(1 + 2⁻²ⁱ)` over [`ITERATIONS`] steps.
///
/// Each rotation stretches the vector by a fixed factor that does not depend on the
/// angle, so starting at `1/K` rather than `1` leaves a unit vector at the end. This
/// is why the loop needs no multiplication: the one scaling it needs is known in
/// advance.
const INVERSE_GAIN: i128 = 48111534222147643976634140972;

/// `atan(2⁻ⁱ)` at [`BITS`] fractional bits.
///
/// Their sum, 1.7432866…, is the convergence range: any angle within it can be
/// reached. π/2 is 1.5707963…, so a quadrant fits with room to spare, which is why
/// the reduction below only needs to reach a quadrant and not an octant.
///
/// From `i = 33` on these are exactly `2⁻ⁱ`: the `x³/3` term of the series for
/// `atan` has fallen below the last bit by then.
const ARCTANGENTS: [i128; ITERATIONS] = [
    62225653328057771307630486156, 36733948115265955625643942646,
    19409209334742409676227471293, 9852417717411270447055696660,
    4945327622306696125935155502, 2475074599931842940039392187,
    1237839310221901317841760601, 618957427126550982503974564,
    309483435713595473912736507, 154742308145852773529447135,
    77371227859691575974375002, 38685623153211227776915241,
    19342812729526912336911253, 9671406508878637801860842,
    4835703272453717209085815, 2417851638478658411936700,
    1208925819520804182482398, 604462909795586463323887,
    302231454902191278172855, 151115727451645394900310,
    75557863725891416926891, 37778931862954298398037,
    18889465931478222940843, 9444732965739245688149,
    4722366482869639621291, 2361183241434821907797,
    1180591620717411216043, 590295810358705640789,
    295147905179352824491, 147573952589676412757,
    73786976294838206443, 36893488147419103229,
    18446744073709551616, 9223372036854775808,
    4611686018427387904, 2305843009213693952,
    1152921504606846976, 576460752303423488,
    288230376151711744, 144115188075855872,
    72057594037927936, 36028797018963968,
    18014398509481984, 9007199254740992,
    4503599627370496, 2251799813685248,
    1125899906842624, 562949953421312,
    281474976710656, 140737488355328,
    70368744177664, 35184372088832,
    17592186044416, 8796093022208,
    4398046511104, 2199023255552,
    1099511627776, 549755813888,
    274877906944, 137438953472,
    68719476736, 34359738368,
    17179869184, 8589934592,
    4294967296, 2147483648,
    1073741824, 536870912,
    268435456, 134217728,
    67108864, 33554432,
];

// ---------------------------------------------------------------------------
// 256-bit helpers
//
// `wide` has most of what is needed, but its right shift is written for shifts
// below 128 and the reduction here needs up to 160. These two fill the gap.
// ---------------------------------------------------------------------------

/// The low 128 bits of a 256-bit value scaled down by `places`, for any `places`.
fn shift_down(value: U256, places: u32) -> u128 {
    if places == 0 {
        value.low
    } else if places < 128 {
        (value.low >> places) | (value.high << (128 - places))
    } else if places < 256 {
        value.high >> (places - 128)
    } else {
        0
    }
}

/// The difference of two 256-bit values, which the caller has shown is not negative.
fn subtract(left: U256, right: U256) -> U256 {
    let (low, borrowed): (u128, bool) = left.low.overflowing_sub(right.low);

    U256 {
        high: left.high - right.high - u128::from(borrowed),
        low,
    }
}

/// Whether `left` is below `right`.
fn is_below(left: U256, right: U256) -> bool {
    (left.high, left.low) < (right.high, right.low)
}

// ---------------------------------------------------------------------------
// Argument reduction
// ---------------------------------------------------------------------------

/// An angle folded into `[0, τ)` at [`BITS`] fractional bits.
///
/// # Why this is not `raw % TAU`
///
/// A remainder taken against τ *rounded to the caller's layout* carries that
/// rounding error once per turn, so an angle of a thousand turns comes back a
/// thousand roundings out. Here the quotient is taken against a 96-bit `1/τ` and the
/// subtraction is done at 256 bits against a 96-bit τ, so the only error is τ's own,
/// multiplied by the number of turns.
///
/// # Accuracy
///
/// The reduced angle is within `|angle| × 2⁻⁹³` of the true one. At a thousand
/// radians that is below `2⁻⁸³`; at a billion, below `2⁻⁶³`. An angle large enough to
/// matter has already lost that precision in its own representation.
fn reduce(raw: i128, fraction_bits: u32) -> i128 {
    let negative: bool = raw < 0;
    let magnitude: u128 = raw.unsigned_abs();

    // The angle at BITS fractional bits, exactly, with room to spare.
    let widened: U256 = wide::shift_left(magnitude, BITS - fraction_bits);

    // How many whole turns it holds. The product is at 2^(fraction_bits + BITS),
    // so shifting that far down leaves the integer count.
    let turns: u128 = shift_down(
        wide::multiply(magnitude, INVERSE_TAU as u128),
        fraction_bits + BITS,
    );

    // 1/τ is rounded, so `turns` can be one out either way. Both are corrected
    // rather than assumed away.
    let mut taken: U256 = wide::multiply(turns, TAU as u128);

    if is_below(widened, taken) {
        taken = wide::multiply(turns - 1, TAU as u128);
    }

    let mut remainder: u128 = subtract(widened, taken).low;

    if remainder >= TAU as u128 {
        remainder -= TAU as u128;
    }

    let remainder: i128 = remainder as i128;

    if negative && remainder != 0 {
        TAU - remainder
    } else {
        remainder
    }
}

// ---------------------------------------------------------------------------
// The two modes
// ---------------------------------------------------------------------------

/// Rotation mode: the unit vector at `angle`, which is `(cos, sin)`.
///
/// `angle` must be within the convergence range, which the quadrant reduction in
/// [`cosine_sine`] guarantees.
fn rotate(angle: i128) -> (i128, i128) {
    let mut x: i128 = INVERSE_GAIN;
    let mut y: i128 = 0;
    let mut residue: i128 = angle;

    let mut step: u32 = 0;

    while (step as usize) < ITERATIONS {
        let (horizontal, vertical): (i128, i128) = (y >> step, x >> step);
        let arctangent: i128 = ARCTANGENTS[step as usize];

        // Turn towards the target angle, by the one amount this step can.
        if residue >= 0 {
            x -= horizontal;
            y += vertical;
            residue -= arctangent;
        } else {
            x += horizontal;
            y -= vertical;
            residue += arctangent;
        }

        step += 1;
    }

    (x, y)
}

/// Vectoring mode: the angle of `(x, y)`, which is `atan2(y, x)`, for `x >= 0`.
///
/// Driving `y` to zero accumulates in `residue` exactly the angle that was there.
fn vector(mut x: i128, mut y: i128) -> i128 {
    let mut residue: i128 = 0;
    let mut step: u32 = 0;

    while (step as usize) < ITERATIONS {
        let (horizontal, vertical): (i128, i128) = (y >> step, x >> step);
        let arctangent: i128 = ARCTANGENTS[step as usize];

        if y < 0 {
            x -= horizontal;
            y += vertical;
            residue -= arctangent;
        } else {
            x += horizontal;
            y -= vertical;
            residue += arctangent;
        }

        step += 1;
    }

    residue
}

// ---------------------------------------------------------------------------
// What the caller uses
// ---------------------------------------------------------------------------

/// The cosine and sine of `raw`, both at [`BITS`] fractional bits.
pub(crate) fn cosine_sine(raw: i128, fraction_bits: u32) -> (i128, i128) {
    let turn: i128 = reduce(raw, fraction_bits);

    // τ is exactly four half-πs here, so a turn below τ lands in quadrant 0..=3 and
    // the remainder is below π/2 — inside the convergence range, with margin.
    let quadrant: i128 = turn / HALF_PI;
    let remainder: i128 = turn - quadrant * HALF_PI;

    let (cosine, sine): (i128, i128) = rotate(remainder);

    match quadrant {
        0 => (cosine, sine),
        1 => (-sine, cosine),
        2 => (-cosine, -sine),
        _ => (sine, -cosine),
    }
}

/// `atan2(y, x)` at [`BITS`] fractional bits, over the full circle `(-π, π]`.
///
/// The arguments may be at any shared scale: an angle depends on the ratio alone, so
/// both are rescaled together to use the full width before the loop runs.
pub(crate) fn arctangent2(y: i128, x: i128) -> i128 {
    if x == 0 && y == 0 {
        return 0;
    }

    // Vectoring converges in the right half-plane, so the other two quadrants are
    // turned into it by a quarter turn, which costs nothing and is exact.
    let (mut across, mut up, offset): (i128, i128, i128) = if x > 0 {
        (x, y, 0)
    } else if y >= 0 {
        (y, -x, HALF_PI)
    } else {
        (-y, x, -HALF_PI)
    };

    // Use the whole width, so a pair of small inputs is as accurate as a large one.
    // The top bit goes to 120, leaving the gain's factor of 1.65 room to grow into.
    let largest: u128 = across.unsigned_abs().max(up.unsigned_abs());
    let headroom: u32 = largest.leading_zeros();

    if headroom > 7 {
        across <<= headroom - 7;
        up <<= headroom - 7;
    } else {
        across >>= 7 - headroom;
        up >>= 7 - headroom;
    }

    vector(across, up) + offset
}
