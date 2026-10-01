//! Bit mixing primitives shared by the rest of the crate.
//!
//! These are the finalizer steps of SplitMix64 (Steele, Lea & Flood, 2014):
//! a multiply-xorshift chain that spreads every input bit across every
//! output bit. None of them is a random number generator on its own. They
//! are the building block that `Random` uses to expand a seed, and that
//! `BitPermuter` uses to derive its multipliers and constants.
//!
//! [`mix64`], [`splitmix64`], [`mix128`], and [`absorb128`] (for a fixed
//! state) are bijections. Width-reducing functions such as [`fold_u128`] and
//! arbitrary-length hashes such as [`hash_bytes`] necessarily can collide.

/// Fractional part of the golden ratio, scaled to 64 bits. It is odd, so
/// repeatedly adding it to a counter walks the whole 64-bit range before
/// returning to where it started.
pub const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Odd 128-bit Weyl increment used by deterministic seed cursors.
///
/// Adding an odd number modulo `2^128` visits every 128-bit state before
/// repeating. The exact constant is part of the random-data format.
pub(crate) const GOLDEN_GAMMA_128: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835;

const MIX_C1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_C2: u64 = 0x94D0_49BB_1331_11EB;

/// Avalanche step: flipping any single input bit changes about half the
/// output bits. Takes the value as it is, without a gamma step.
#[inline]
pub const fn mix64(value: u64) -> u64 {
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

/// Folds 128 bits down to 64.
///
/// Both halves contribute and swapping them usually changes the result, but
/// this is a lossy reduction: every output necessarily has `2^64` inputs.
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
/// Absorbs the input in 16-byte blocks, zero-padding the last one, and then
/// absorbs the length. The length is what keeps that padding unambiguous: two
/// inputs differing only in trailing zero bytes, or one a prefix of the other,
/// would otherwise reach the same state.
///
/// Bytes are read little-endian explicitly rather than reinterpreted, so the
/// result is the same on a big-endian target. This is the function behind
/// [`Seed::from_text`](crate::random::seed::Seed::from_text) and
/// [`ContentHash`](crate::units::ContentHash), and it is not a cryptographic
/// hash: it resists accident, not forgery.
pub fn hash_bytes(bytes: &[u8]) -> u128 {
    let mut state: u128 = BYTES_INITIAL;

    let (chunks, remainder) = bytes.as_chunks::<16>();

    for chunk in chunks {
        state = absorb128(state, u128::from_le_bytes(*chunk));
    }

    let mut tail: [u8; 16] = [0; 16];
    tail[..remainder.len()].copy_from_slice(remainder);

    state = absorb128(state, u128::from_le_bytes(tail));

    absorb128(state, bytes.len() as u128)
}
