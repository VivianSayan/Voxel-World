//! Bit mixing primitives shared by the other `Misc` modules.
//!
//! These are the finalizer steps of SplitMix64 (Steele, Lea & Flood, 2014):
//! a multiply-xorshift chain that spreads every input bit across every
//! output bit. None of them is a random number generator on its own. They
//! are the building block that `Random` uses to expand a seed, and that
//! `BitPermuter` uses to derive its multipliers and constants.
//!
//! Every function here is bijective, so distinct inputs always produce
//! distinct outputs: mixing never collapses two seeds into one.

/// Fractional part of the golden ratio, scaled to 64 bits. It is odd, so
/// repeatedly adding it to a counter walks the whole 64-bit range before
/// returning to where it started.
pub const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

const MIX_C1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_C2: u64 = 0x94D0_49BB_1331_11EB;

/// Avalanche step: flipping any single input bit changes about half the
/// output bits. Takes the value as it is, without a gamma step.
#[inline]
pub fn mix64(value: u64) -> u64 {
    let mut x: u64 = value;

    x = (x ^ (x >> 30)).wrapping_mul(MIX_C1);
    x = (x ^ (x >> 27)).wrapping_mul(MIX_C2);

    x ^ (x >> 31)
}

/// SplitMix64 applied to one standalone value: a gamma step, then avalanche.
/// Use this to hash a value that is not part of a running stream.
#[inline]
pub fn splitmix64(value: u64) -> u64 {
    mix64(value.wrapping_add(GOLDEN_GAMMA))
}

/// SplitMix64 as a stream: advances `state` and returns the mixed result.
/// Visits every one of the 2^64 states exactly once before repeating.
#[inline]
pub fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN_GAMMA);
    mix64(*state)
}

/// Folds 128 bits down to 64. The upper half is rotated first, so that a
/// value and its halves swapped cannot fold to the same result.
#[inline]
pub fn fold_u128(value: u128) -> u64 {
    let lower: u64 = value as u64;
    let upper: u64 = (value >> 64) as u64;

    lower ^ upper.rotate_left(32)
}

/// Mixes all 128 input bits into all 128 output bits, for callers that need
/// a full-width mixed value rather than a folded 64-bit one.
#[inline]
pub fn mix128(value: u128) -> u128 {
    let lower: u64 = value as u64;
    let upper: u64 = (value >> 64) as u64;

    let low_out: u64 = splitmix64(lower ^ splitmix64(upper));
    let high_out: u64 = splitmix64(upper ^ low_out.rotate_left(32));

    ((high_out as u128) << 64) | low_out as u128
}

// ---------------------------------------------------------------------------
// Absorption
// ---------------------------------------------------------------------------

/// Initial state for `hash_bytes`. Any odd constant does; this one is the
/// fractional part of pi scaled to 128 bits.
const BYTES_INITIAL: u128 = 0x243F_6A88_85A3_08D3_1319_8A2E_0370_7344;

/// Folds one 128-bit input into a running state.
///
/// Bijective in `input` for any fixed state, so two different inputs can never
/// meet at the same point of a stream. Chaining this is how a value longer than
/// 128 bits is reduced without letting earlier parts cancel later ones, which
/// is what a plain xor-fold would allow.
#[inline]
pub fn absorb128(state: u128, input: u128) -> u128 {
    mix128(state ^ input)
}

/// Widens a 64-bit value into a well-spread 128-bit one, so that a small or
/// structured number still reaches every bit of a seed.
#[inline]
pub fn expand_u64(value: u64) -> u128 {
    mix128(value as u128)
}

/// Hashes a byte string to 128 bits.
///
/// The length is absorbed at the end as well as the bytes, so that inputs
/// differing only in trailing zero bytes, or one being a prefix of another, do
/// not collide. Bytes are read little-endian explicitly rather than through a
/// transmute, so the result is the same on every target.
pub fn hash_bytes(bytes: &[u8]) -> u128 {
    let mut state: u128 = BYTES_INITIAL;

    let (chunks, remainder) = bytes.as_chunks::<16>();

    for chunk in chunks {
        state = absorb128(state, u128::from_le_bytes(*chunk));
    }

    // The tail is zero-padded to a full block; the length absorbed after it is
    // what keeps that padding from being ambiguous.
    let mut tail: [u8; 16] = [0; 16];
    tail[..remainder.len()].copy_from_slice(remainder);

    state = absorb128(state, u128::from_le_bytes(tail));

    absorb128(state, bytes.len() as u128)
}
