//! Adaptive storage for a dense sequence of values.
//!
//! [`CompactSequence`] presents one logical sequence, indexed from `0` to
//! `len - 1`, while allowing four different physical representations:
//!
//! - one shared value when the whole sequence is uniform;
//! - a [`Palette`] when there are few distinct values;
//! - a [`RunLengthSequence`] when equal values form long adjacent runs;
//! - a plain [`Vec`] when compression would cost more than it saves.
//!
//! # Choosing a representation
//!
//! Collecting values, converting from a `Vec`, palette, or run-length
//! sequence, and calling [`CompactSequence::compact`] all inspect the complete
//! logical sequence. A uniform sequence always wins. Otherwise the code
//! estimates the bytes needed by palette, run-length, and dense storage and
//! chooses the smallest estimate. The estimate includes values, palette usage
//! counts, packed indices, and run start positions; allocator and alignment
//! overhead are deliberately ignored because they are implementation details.
//!
//! The representation is observable through [`CompactSequence::kind`] for
//! diagnostics, but it is not part of logical equality or content hashing.
//! Two instances compare equal when they yield the same values in the same
//! order, even if one is dense and the other uses runs.
//!
//! # Mutation and recompression
//!
//! A single write does **not** rescan the entire sequence. It modifies the
//! current representation in place; the first differing write to a uniform
//! sequence changes it to palette storage. After a large batch of writes, call
//! [`CompactSequence::compact`] once to reconsider all four representations.
//! This avoids accidentally turning every write into an O(n) operation.
//!
//! # Example
//!
//! ```
//! use voxel_world::structures::storage::{CompactKind, CompactSequence};
//!
//! let mut cells = CompactSequence::filled(4096, 0u16);
//! assert_eq!(cells.kind(), CompactKind::Uniform);
//!
//! cells.set(17, 3);
//! assert_eq!(cells.get(17), Some(&3));
//! assert_eq!(cells.kind(), CompactKind::Palette);
//!
//! // Re-evaluate storage once a batch of edits is finished.
//! cells.compact();
//! ```

use super::Palette;
use crate::random::Random;
use crate::structures::collections::RunLengthSequence;
use crate::structures::traits::{
    CanonicalOrder, Choose, Collection, ContentHashable, DeterministicOrder, Sequence, StableHash,
};
use crate::units::digest::ContentHash;

/// The physical representation currently used by a [`CompactSequence`].
///
/// This is primarily diagnostic information. Generic code should normally use
/// the sequence operations rather than branch on its current kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompactKind {
    /// One stored value is shared by every logical position.
    Uniform,
    /// Each distinct value is stored once and positions use packed indices.
    Palette,
    /// Equal adjacent values are represented by one value and a start index.
    Runs,
    /// A plain vector stores one value directly at every position.
    Dense,
}

/// A dense, indexable sequence that chooses a compact physical representation.
///
/// Every position in `0..len()` has a value; this is not a sparse collection.
/// Reads and iteration have the same logical result for every representation.
/// The main trade-off is that palette and run storage save memory in exchange
/// for more work during some mutations.
///
/// Construct with [`CompactSequence::filled`], collect an iterator, or convert
/// a `Vec`, [`Palette`], or [`RunLengthSequence`]. Call
/// [`CompactSequence::compact`] after a batch of edits when the data's shape
/// may have changed substantially.
#[derive(Clone, Debug)]
pub enum CompactSequence<T> {
    /// Zero or more positions sharing one value.
    ///
    /// `value` is `None` only for the empty sequence created by the provided
    /// constructors.
    Uniform {
        /// Number of represented positions.
        len: usize,
        /// Shared value, absent only for an empty sequence.
        value: Option<T>,
    },
    /// Few distinct values, regardless of where they occur.
    Palette(Palette<T>),
    /// Long adjacent runs of equal values.
    Runs(RunLengthSequence<T>),
    /// Direct storage when either compressed representation would cost more.
    Dense(Vec<T>),
}

impl<T: Clone + PartialEq> PartialEq for CompactSequence<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}
impl<T: Clone + Eq> Eq for CompactSequence<T> {}
impl<T> DeterministicOrder for CompactSequence<T> {}
impl<T> CanonicalOrder for CompactSequence<T> {}

impl<T: Clone + PartialEq> Collection for CompactSequence<T> {
    type Item = T;
    fn len(&self) -> usize {
        CompactSequence::len(self)
    }
    fn contains(&self, item: &T) -> bool {
        self.iter().any(|held| held == item)
    }
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: Clone + PartialEq> Sequence for CompactSequence<T> {
    fn get(&self, index: usize) -> Option<&T> {
        CompactSequence::get(self, index)
    }
    fn index_of(&self, item: &T) -> Option<usize> {
        self.iter().position(|held| held == item)
    }
}

impl<T: Clone + PartialEq> Choose for CompactSequence<T> {
    fn choose(&self, random: &mut Random) -> Option<&T> {
        (!self.is_empty())
            .then(|| self.get(random.uniform_index(self.len())))
            .flatten()
    }
}

impl<T: Clone + PartialEq + StableHash> ContentHashable for CompactSequence<T> {
    fn content_hash(&self) -> ContentHash {
        self.iter().fold(ContentHash::EMPTY, |hash, value| {
            hash.and(value.stable_hash())
        })
    }
}

impl<T: Clone + PartialEq> CompactSequence<T> {
    /// Creates `len` positions that all contain `value`.
    ///
    /// This always uses [`CompactKind::Uniform`] and therefore stores only one
    /// copy of `value`, regardless of `len`. When `len` is zero the supplied
    /// value is discarded.
    ///
    /// O(1) time and storage, apart from moving `value`.
    pub fn filled(len: usize, value: T) -> Self {
        Self::Uniform {
            len,
            value: (len > 0).then_some(value),
        }
    }
    /// Returns the number of logical positions in the sequence.
    ///
    /// This is independent of the number of palette entries or runs and is
    /// O(1) for every representation.
    pub fn len(&self) -> usize {
        match self {
            Self::Uniform { len, .. } => *len,
            Self::Palette(v) => v.len(),
            Self::Runs(v) => v.len(),
            Self::Dense(v) => v.len(),
        }
    }
    /// Returns whether the logical sequence contains no positions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Reports the current physical representation.
    ///
    /// This can change after [`set`](Self::set), [`fill`](Self::fill), or
    /// [`compact`](Self::compact) without changing the sequence's public
    /// meaning.
    pub const fn kind(&self) -> CompactKind {
        match self {
            Self::Uniform { .. } => CompactKind::Uniform,
            Self::Palette(_) => CompactKind::Palette,
            Self::Runs(_) => CompactKind::Runs,
            Self::Dense(_) => CompactKind::Dense,
        }
    }
    /// Borrows the value at `index`, or returns `None` when it is out of range.
    ///
    /// Uniform, palette, and dense reads are O(1). A run-length read is
    /// O(log r), where `r` is the number of runs.
    pub fn get(&self, index: usize) -> Option<&T> {
        match self {
            Self::Uniform { len, value } => {
                if index < *len {
                    value.as_ref()
                } else {
                    None
                }
            }
            Self::Palette(v) => v.get(index),
            Self::Runs(v) => v.get(index),
            Self::Dense(v) => v.get(index),
        }
    }
    /// Replaces the value at one position.
    ///
    /// Returns `true` only when `index` exists and its value changed. An
    /// out-of-range index or an equal value returns `false`.
    ///
    /// The write stays in the current representation, except that the first
    /// differing write to uniform storage creates a palette. Palette writes
    /// may repack all indices when the number of distinct values crosses an
    /// index-width boundary. Run-length writes may split or merge runs. Dense
    /// writes are O(1).
    ///
    /// This method deliberately does not call [`compact`](Self::compact).
    pub fn set(&mut self, index: usize, value: T) -> bool {
        match self {
            Self::Uniform { len, value: held } => {
                if index >= *len || held.as_ref() == Some(&value) {
                    return false;
                }
                let filler = held.take().expect("non-empty uniform sequence has a value");
                let mut palette = Palette::filled(*len, filler);
                palette.set(index, value);
                *self = Self::Palette(palette);
                true
            }
            Self::Palette(v) => v.set(index, value),
            Self::Runs(v) => v.set(index, value),
            Self::Dense(v) => {
                let Some(slot) = v.get_mut(index) else {
                    return false;
                };
                if *slot == value {
                    false
                } else {
                    *slot = value;
                    true
                }
            }
        }
    }
    /// Replaces every value in the half-open range `start..end`.
    ///
    /// `end` is clipped to [`len`](Self::len); an empty, reversed, or wholly
    /// out-of-range interval does nothing. Filling the complete sequence
    /// immediately changes it to uniform storage. Partial fills use the
    /// current representation's native range operation and do not perform a
    /// global representation search.
    pub fn fill(&mut self, start: usize, end: usize, value: T) {
        let end = end.min(self.len());
        if start >= end {
            return;
        }
        if start == 0 && end == self.len() {
            *self = Self::filled(end, value);
            return;
        }
        match self {
            Self::Palette(values) => values.fill(start, end, value),
            Self::Runs(values) => values.fill(start, end, value),
            Self::Dense(values) => values[start..end].fill(value),
            Self::Uniform { .. } => {
                // A partial change from a uniform value is palette-friendly.
                for index in start..end {
                    self.set(index, value.clone());
                }
            }
        }
    }
    /// Iterates over every logical value in position order.
    ///
    /// Repeated values are yielded repeatedly: a uniform sequence of length
    /// 100 yields 100 references to its one stored value. The iterator hides
    /// the current representation so callers see the same sequence either way.
    pub fn iter(&self) -> Box<dyn Iterator<Item = &T> + '_> {
        match self {
            Self::Uniform { len, value } => Box::new(value.iter().cycle().take(*len)),
            Self::Palette(v) => Box::new(v.iter()),
            Self::Runs(v) => Box::new(v.values()),
            Self::Dense(v) => Box::new(v.iter()),
        }
    }
    /// Rebuilds the sequence using the estimated smallest representation.
    ///
    /// This materializes the logical values, counts distinct values and runs,
    /// estimates palette/run/dense storage, and reconstructs the winner.
    /// Consequently it is O(n) plus the cost of finding distinct values. Since
    /// `T` requires only `PartialEq`, distinct-value detection can be O(n²) in
    /// the worst case. Call this once after a batch of edits, not after every
    /// write.
    pub fn compact(&mut self) {
        let values: Vec<T> = self.iter().cloned().collect();
        *self = values.into_iter().collect();
    }
    /// Returns the number of bits per position used by palette indices.
    ///
    /// Returns zero for non-palette representations, including uniform, runs,
    /// and dense storage. This is a palette diagnostic, not a general measure
    /// of the sequence's total memory usage.
    pub fn index_bits(&self) -> u32 {
        match self {
            Self::Palette(v) => v.index_bits(),
            _ => 0,
        }
    }
    /// Returns the bytes occupied by packed palette indices.
    ///
    /// Palette values and usage counts are not included. Non-palette
    /// representations return zero, so use [`kind`](Self::kind) alongside this
    /// method when reporting storage.
    pub fn packed_bytes(&self) -> usize {
        match self {
            Self::Palette(v) => v.packed_bytes(),
            _ => 0,
        }
    }
}

impl<T: Clone + PartialEq> From<Palette<T>> for CompactSequence<T> {
    /// Converts the logical palette contents and re-evaluates all compact
    /// representations; the result is not necessarily palette-backed.
    fn from(value: Palette<T>) -> Self {
        let mut result = Self::Palette(value);
        result.compact();
        result
    }
}

impl<T: Clone + PartialEq> From<RunLengthSequence<T>> for CompactSequence<T> {
    /// Converts the logical run contents and re-evaluates all compact
    /// representations; the result is not necessarily run-backed.
    fn from(value: RunLengthSequence<T>) -> Self {
        let mut result = Self::Runs(value);
        result.compact();
        result
    }
}

impl<T: Clone + PartialEq> From<Vec<T>> for CompactSequence<T> {
    /// Inspects the values and chooses uniform, palette, run, or dense storage.
    fn from(values: Vec<T>) -> Self {
        values.into_iter().collect()
    }
}

impl<T: Clone + PartialEq> FromIterator<T> for CompactSequence<T> {
    /// Collects a dense logical sequence and chooses its initial physical
    /// representation from the value distribution.
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        let values: Vec<T> = items.into_iter().collect();
        if values.is_empty() {
            return Self::Uniform {
                len: 0,
                value: None,
            };
        }
        let mut distinct: Vec<&T> = Vec::new();
        let mut runs = 1usize;
        for (index, value) in values.iter().enumerate() {
            if !distinct.contains(&value) {
                distinct.push(value);
            }
            if index > 0 && values[index - 1] != *value {
                runs += 1;
            }
        }
        if distinct.len() == 1 {
            return Self::filled(values.len(), values[0].clone());
        }
        let value_size = size_of::<T>().max(1);
        let bits = match distinct.len() {
            0 | 1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=16 => 4,
            17..=256 => 8,
            257..=65536 => 16,
            _ => 32,
        };
        let palette_cost = distinct
            .len()
            .saturating_mul(value_size + size_of::<usize>())
            + values.len().saturating_mul(bits).div_ceil(8);
        let run_cost = runs.saturating_mul(value_size + size_of::<usize>());
        let dense_cost = values.len().saturating_mul(value_size);
        if palette_cost <= run_cost && palette_cost < dense_cost {
            Self::Palette(values.into_iter().collect())
        } else if run_cost < dense_cost {
            Self::Runs(values.into_iter().collect())
        } else {
            Self::Dense(values)
        }
    }
}
