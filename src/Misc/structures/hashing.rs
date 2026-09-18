//! Hashing shared by every structure in this module.
//!
//! `FastHasher` is a multiply-rotate hasher in the style of rustc's FxHash.
//! It is much faster than the standard SipHash for the small keys these
//! structures hold, and it is deterministic: the same operations on the same
//! data always produce the same iteration order. That matters for seeded
//! worlds, where picking a "random" member must replay identically.
//!
//! It is not resistant to deliberately crafted collisions, which is fine for
//! game data but not for untrusted network input.

use crate::misc::mixing::mix64;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};

const MULTIPLIER: u64 = 0xF135_7AEA_2E62_A9C5;

#[derive(Clone, Copy, Debug, Default)]
/// Fast deterministic hasher used by the structures in this module.
///
/// This hasher is intended for trusted game data. It is not collision-attack
/// resistant and should not hash untrusted input exposed to denial-of-service
/// attacks.
pub struct FastHasher {
    hash: u64,
}

impl FastHasher {
    #[inline]
    fn add_word(&mut self, word: u64) {
        self.hash = self.hash.wrapping_add(word).wrapping_mul(MULTIPLIER);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, remainder) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.add_word(u64::from_le_bytes(*chunk));
        }

        if !remainder.is_empty() {
            let mut buffer = [0u8; 8];
            buffer[..remainder.len()].copy_from_slice(remainder);
            // The length keeps "a" and "a\0" apart.
            self.add_word(u64::from_le_bytes(buffer) ^ ((remainder.len() as u64) << 59));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.add_word(value as u64);
    }

    #[inline]
    fn write_u16(&mut self, value: u16) {
        self.add_word(value as u64);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.add_word(value as u64);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add_word(value);
    }

    #[inline]
    fn write_u128(&mut self, value: u128) {
        self.add_word(value as u64);
        self.add_word((value >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add_word(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // The multiply pushes the best-mixed bits to the top; the hash table
        // indexes with the low bits, so rotate the good bits down.
        self.hash.rotate_left(26)
    }
}

/// Build-hasher adapter for [`FastHasher`].
pub type FastBuildHasher = BuildHasherDefault<FastHasher>;
/// A deterministic [`HashMap`] keyed by `K` and storing values of type `V`.
pub type FastHashMap<K, V> = HashMap<K, V, FastBuildHasher>;
/// A deterministic [`HashSet`] storing values of type `T`.
pub type FastHashSet<T> = HashSet<T, FastBuildHasher>;

/// Hashes one value on its own, fully mixed.
#[inline]
pub fn hash_one<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = FastHasher::default();
    value.hash(&mut hasher);
    mix64(hasher.finish())
}

/// A hash of a collection that ignores iteration order, so two sets with
/// the same members hash the same however they were built. Each member is
/// mixed before the commutative sum, so members cannot cancel each other.
pub fn unordered_hash<I>(items: I) -> u64
where
    I: IntoIterator,
    I::Item: Hash,
{
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    for item in items {
        sum = sum.wrapping_add(hash_one(&item));
        count += 1;
    }
    mix64(sum ^ count.wrapping_mul(MULTIPLIER))
}
