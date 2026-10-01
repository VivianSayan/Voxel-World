//! Hashes and checksums.
//!
//! Two different jobs that both come out as an integer, which is why they are
//! two types.
//!
//! A [`ContentHash`] answers "is this the same content?". It is 128 bits wide,
//! so unrelated content practically never collides, and it is what to key a
//! cache on or compare two chunks with.
//!
//! A [`Checksum`] answers "did this survive being written down?". It is 64 bits
//! and cheap, and it is what to write beside a save file so that a truncated or
//! corrupted one is noticed rather than loaded.
//!
//! Neither resists a deliberate forgery. They are built from the same mixing
//! the rest of the module uses, which is fast and well spread but not a
//! cryptographic hash. For anything an adversary would want to fake, use a real
//! one from outside this crate.

use crate::random::mixing::{absorb128, fold_u128, hash_bytes, mix128};
use crate::random::seed::Seed;
use std::fmt;

/// A 128-bit hash of some content.
///
/// Built from the crate's private byte hashing and 128-bit mixing primitives.
/// They use integer arithmetic only, so a hash of the same content matches on
/// every machine and every run, unlike the standard library's
/// [`Hash`](std::hash::Hash), whose usual map hasher is seeded differently per
/// process.
///
/// The folding methods are order-sensitive on purpose: a chunk's voxels hash
/// differently from the same voxels rearranged.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct ContentHash(u128);

impl ContentHash {
    /// The hash of nothing at all, and the state a builder starts from.
    pub const EMPTY: Self = Self(0);

    /// A hash read back from storage, taken as it stands without mixing.
    pub const fn from_raw(value: u128) -> Self {
        Self(value)
    }

    /// The hash of a block of bytes.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(hash_bytes(bytes))
    }

    /// The hash of a string, which is the hash of its UTF-8 bytes. Equal
    /// strings hash equal on every platform.
    pub fn of_text(text: &str) -> Self {
        Self(hash_bytes(text.as_bytes()))
    }

    /// The hash as a plain `u128`, for writing down.
    pub const fn value(self) -> u128 {
        self.0
    }

    /// Folds another piece of content in, for hashing something built out of
    /// parts.
    ///
    /// The running hash is mixed before the next piece is absorbed, which is
    /// what makes the result depend on the order: `a.and(b)` and `b.and(a)`
    /// differ, so a chunk's voxels hash differently from the same voxels
    /// rearranged. Start from [`ContentHash::EMPTY`] and fold each piece in
    /// turn.
    pub fn and(self, next: Self) -> Self {
        Self(absorb128(mix128(self.0), next.0))
    }

    /// Folds in a value that is already a number, such as a count, an index or
    /// an identifier, without hashing it as bytes first.
    pub fn and_value(self, value: u128) -> Self {
        Self(absorb128(self.0, value))
    }

    /// Folds in another block of bytes: [`ContentHash::of_bytes`] followed by
    /// [`ContentHash::and`].
    pub fn and_bytes(self, bytes: &[u8]) -> Self {
        self.and(Self::of_bytes(bytes))
    }

    /// The short form, for writing beside a file: the same 128 bits folded
    /// down to 64. Two hashes that agree still have their checksums agree.
    pub fn to_checksum(self) -> Checksum {
        Checksum(fold_u128(self.0))
    }

    /// As 32 hexadecimal digits, zero-padded.
    pub fn to_hex(self) -> String {
        format!("{:032x}", self.0)
    }

    /// The top 32 bits as eight hexadecimal digits, which is enough to tell two
    /// apart by eye in a log. Not enough to compare with.
    pub fn to_short_hex(self) -> String {
        format!("{:08x}", (self.0 >> 96) as u32)
    }
}

impl From<Seed> for ContentHash {
    /// A seed is already a well-mixed 128 bits, so it carries across as it is.
    /// Going the other way is deliberately absent: a hash of content is not
    /// something a world should be seeded from without saying so.
    fn from(seed: Seed) -> Self {
        Self(seed.value())
    }
}

impl fmt::Display for ContentHash {
    /// As 32 hexadecimal digits; see [`ContentHash::to_hex`].
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

/// A 64-bit check that data survived storage or transfer.
///
/// Half the width of a [`ContentHash`], which makes an accidental collision
/// about as likely as one in `2^64`: ample for catching a truncated or
/// corrupted file, and not meant for telling two pieces of content apart.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct Checksum(u64);

impl Checksum {
    /// A checksum read back from storage, taken as it stands.
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// The checksum of a block of bytes: its [`ContentHash`], folded to 64
    /// bits.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        ContentHash::of_bytes(bytes).to_checksum()
    }

    /// The checksum as a plain `u64`, for writing down.
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Whether some bytes still check out against this: whether their checksum
    /// is this one.
    pub fn verifies(self, bytes: &[u8]) -> bool {
        Self::of_bytes(bytes) == self
    }

    /// As 16 hexadecimal digits, zero-padded.
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

impl fmt::Display for Checksum {
    /// As 16 hexadecimal digits; see [`Checksum::to_hex`].
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}
