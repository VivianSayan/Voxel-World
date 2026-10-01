//! Stable content identity, independent of `std` hasher implementations.

use crate::random::mixing::mix128;
use crate::units::{ContentHash, Id, IdKind};

/// An explicit, versioned-by-code encoding into a stable 128-bit content hash.
///
/// Unlike [`std::hash::Hash`], implementations here promise the same result on
/// every supported architecture. Adding or changing fields may still change
/// the result, so persisted formats must retain their own version number.
pub trait StableHash {
    /// Hashes this value using its canonical representation.
    fn stable_hash(&self) -> ContentHash;
}

macro_rules! stable_unsigned {
    ($($ty:ty),+ $(,)?) => {$ (
        impl StableHash for $ty {
            fn stable_hash(&self) -> ContentHash {
                ContentHash::of_bytes(&(*self as u128).to_le_bytes())
            }
        }
    )+ };
}

macro_rules! stable_signed {
    ($($ty:ty),+ $(,)?) => {$ (
        impl StableHash for $ty {
            fn stable_hash(&self) -> ContentHash {
                ContentHash::of_bytes(&(*self as i128).to_le_bytes())
            }
        }
    )+ };
}

stable_unsigned!(u8, u16, u32, u64, u128, usize);
stable_signed!(i8, i16, i32, i64, i128, isize);

impl StableHash for bool {
    fn stable_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&[u8::from(*self)])
    }
}

impl StableHash for str {
    fn stable_hash(&self) -> ContentHash {
        ContentHash::of_text(self)
    }
}

impl StableHash for String {
    fn stable_hash(&self) -> ContentHash {
        self.as_str().stable_hash()
    }
}

impl<T: StableHash + ?Sized> StableHash for &T {
    fn stable_hash(&self) -> ContentHash {
        (*self).stable_hash()
    }
}

impl<K: IdKind> StableHash for Id<K> {
    fn stable_hash(&self) -> ContentHash {
        ContentHash::EMPTY
            .and_value(K::TAG as u128)
            .and_value(self.value() as u128)
    }
}

impl<A: StableHash, B: StableHash> StableHash for (A, B) {
    fn stable_hash(&self) -> ContentHash {
        self.0.stable_hash().and(self.1.stable_hash())
    }
}

impl<T: StableHash> StableHash for Option<T> {
    fn stable_hash(&self) -> ContentHash {
        match self {
            None => ContentHash::EMPTY.and_value(0),
            Some(value) => ContentHash::EMPTY.and_value(1).and(value.stable_hash()),
        }
    }
}

impl<T: StableHash, const N: usize> StableHash for [T; N] {
    fn stable_hash(&self) -> ContentHash {
        stable_hash_ordered(self.iter())
    }
}

impl<T: StableHash> StableHash for [T] {
    fn stable_hash(&self) -> ContentHash {
        stable_hash_ordered(self.iter())
    }
}

impl<T: StableHash> StableHash for Vec<T> {
    fn stable_hash(&self) -> ContentHash {
        self.as_slice().stable_hash()
    }
}

/// Hashes stable values in iteration order.
pub fn stable_hash_ordered<I>(values: I) -> ContentHash
where
    I: IntoIterator,
    I::Item: StableHash,
{
    values.into_iter().fold(ContentHash::EMPTY, |state, value| {
        state.and(value.stable_hash())
    })
}

/// Hashes stable values without depending on iteration order.
pub fn stable_hash_unordered<I>(values: I) -> ContentHash
where
    I: IntoIterator,
    I::Item: StableHash,
{
    let mut sum = 0u128;
    let mut xor = 0u128;
    let mut count = 0u128;
    for value in values {
        let hash = value.stable_hash().value();
        sum = sum.wrapping_add(hash);
        xor ^= mix128(hash);
        count += 1;
    }
    ContentHash::from_raw(mix128(sum ^ xor.rotate_left(37) ^ mix128(count)))
}
