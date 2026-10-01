//! A long sequence whose values are drawn from a small set, stored as one copy
//! of each distinct value and a packed index per position.
//!
//! The saving comes from *how few distinct values there are*, not from where
//! they sit: a sequence of a million entries drawn from four values costs four
//! entries and two bits apiece, however thoroughly they are shuffled. That is
//! the opposite of what a tree or a run-length store compresses, both of which
//! need the equal values to be next to one another, and it is why the three are
//! worth having side by side.
//!
//! It fits anything with that shape. A block of terrain drawn from a handful of
//! material types; a tile layer over a map; a per-entity state or team or
//! faction across a crowd; a column of samples quantised to a few levels; any
//! long array of enum-like values.
//!
//! # What it costs
//!
//! Reading and writing a position are both constant time: an index lookup, a
//! shift and a mask. The index width follows the palette, so a sequence of
//! `n` positions over `d` distinct values takes about `n * ceil(log2 d)` bits
//! plus the values themselves:
//!
//! | Distinct values | Bits per position | 4096 positions |
//! |---|---|---|
//! | 1 | 0 | nothing at all |
//! | 2 | 1 | 512 bytes |
//! | 3 to 4 | 2 | 1 KB |
//! | 5 to 16 | 4 | 2 KB |
//! | 17 to 256 | 8 | 4 KB |
//!
//! Against one machine word per position — 32 KB for those 4096 — the win is
//! large until the palette approaches the length of the sequence, at which
//! point the indices cost more than the values they stand in for.
//! [`Palette::distinct_len`] is what to watch if that is in doubt.

use crate::random::Random;
use crate::structures::traits::{
    CanonicalOrder, Choose, Collection, ContentHashable, DeterministicOrder, Sequence, StableHash,
};
use crate::units::digest::ContentHash;

/// Values at `len` positions, held as a palette of distinct values and a packed
/// index per position.
///
/// The index width grows with the palette: adding the third distinct value
/// widens every index from one bit to two, which rewrites the packed store. It
/// never shrinks on its own, since a value removed from every position is still
/// in the palette until [`Palette::compact`] is called; that is what keeps
/// writing a value back and forth from repacking twice.
///
/// Reading and writing a position are both constant time: an index lookup, a
/// shift and a mask.
#[derive(Clone, Debug)]
pub struct Palette<T> {
    /// Each distinct value once, in the order it first appeared.
    values: Vec<T>,
    /// How many positions hold each value, so an unused one can be found.
    counts: Vec<usize>,
    /// The indices, packed `bits` apart, lowest position first.
    packed: Vec<u64>,
    /// How many bits each index takes.
    bits: u32,
    /// How many positions there are.
    len: usize,
}

impl<T: Clone + PartialEq> Palette<T> {
    /// A store of `len` positions, every one holding `value`.
    ///
    /// The cheapest case: one palette entry, and no packed storage at all,
    /// since a palette of one needs no bits to choose between its entries.
    pub fn filled(len: usize, value: T) -> Self {
        Self {
            values: vec![value],
            counts: vec![len],
            packed: Vec::new(),
            bits: 0,
            len,
        }
    }

    /// How many positions the store covers.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether it covers none.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many distinct values the palette holds, including any that no
    /// position uses any more.
    pub fn distinct_len(&self) -> usize {
        self.values.len()
    }

    /// How many bits each position's index takes. Zero while the palette holds
    /// one value, since there is nothing to choose.
    pub const fn index_bits(&self) -> u32 {
        self.bits
    }

    /// How many bytes the packed indices take, not counting the palette itself.
    pub fn packed_bytes(&self) -> usize {
        self.packed.len() * size_of::<u64>()
    }

    /// The palette's values, in the order they first appeared.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Reserves room for additional distinct values without changing the
    /// number of positions represented by this sequence.
    pub fn reserve_distinct(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.counts.reserve(additional);
    }

    /// Releases unused palette and packed-index allocation.
    pub fn shrink_to_fit(&mut self) {
        self.compact();
        self.values.shrink_to_fit();
        self.counts.shrink_to_fit();
        self.packed.shrink_to_fit();
    }

    /// Whether every position holds the same value, which a generator checks
    /// constantly: a uniform chunk needs no storage of its own.
    pub fn is_uniform(&self) -> bool {
        self.values.len() <= 1 || self.counts.iter().filter(|count| **count > 0).count() == 1
    }

    /// The value at a position, or `None` past the end.
    pub fn get(&self, position: usize) -> Option<&T> {
        if position >= self.len {
            return None;
        }

        self.values.get(self.index_at(position))
    }

    /// Sets the value at a position, adding it to the palette if it is new.
    ///
    /// Returns whether the value changed. Widening the indices, which happens
    /// when the palette passes a power of two, repacks the whole store; that is
    /// the one operation here that is not constant time.
    pub fn set(&mut self, position: usize, value: T) -> bool {
        if position >= self.len {
            return false;
        }

        let previous: usize = self.index_at(position);

        if self.values.get(previous) == Some(&value) {
            return false;
        }

        let index: usize = self.intern(value);

        self.counts[previous] -= 1;
        self.counts[index] += 1;
        self.write_index(position, index);

        true
    }

    /// Sets every position in `start..end` to one value.
    pub fn fill(&mut self, start: usize, end: usize, value: T) {
        let end: usize = end.min(self.len);

        if start >= end {
            return;
        }

        let index: usize = self.intern(value);

        for position in start..end {
            let previous: usize = self.index_at(position);

            if previous == index {
                continue;
            }

            self.counts[previous] -= 1;
            self.counts[index] += 1;
            self.write_index(position, index);
        }
    }

    /// Every position's value, in order.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        (0..self.len).map(|position| &self.values[self.index_at(position)])
    }

    /// How many positions hold a value.
    pub fn count_of(&self, value: &T) -> usize {
        self.values
            .iter()
            .position(|held| held == value)
            .map_or(0, |index| self.counts[index])
    }

    /// Drops the palette entries no position uses any more, narrowing the
    /// indices if that brings the palette below a power of two.
    ///
    /// Worth doing when a chunk settles after generation, and not worth doing
    /// after every write: a value written and then written back would otherwise
    /// repack the store twice.
    pub fn compact(&mut self) {
        if self.counts.iter().all(|count| *count > 0) {
            return;
        }

        let kept: Vec<usize> = (0..self.values.len())
            .filter(|index| self.counts[*index] > 0)
            .collect();

        // Where each old index ends up.
        let mut moved: Vec<usize> = vec![0; self.values.len()];

        for (new, old) in kept.iter().enumerate() {
            moved[*old] = new;
        }

        let indices: Vec<usize> = (0..self.len)
            .map(|position| moved[self.index_at(position)])
            .collect();

        self.values = kept
            .iter()
            .map(|index| self.values[*index].clone())
            .collect();
        self.counts = kept.iter().map(|index| self.counts[*index]).collect();
        self.bits = Self::bits_for(self.values.len());
        self.packed = vec![0; self.words_needed(self.bits)];

        for (position, index) in indices.into_iter().enumerate() {
            self.write_index(position, index);
        }
    }

    /// The palette index of a value, adding it if it is new and widening the
    /// packed indices if the palette has outgrown them.
    fn intern(&mut self, value: T) -> usize {
        if let Some(index) = self.values.iter().position(|held| *held == value) {
            return index;
        }

        assert!(
            self.values.len() <= u32::MAX as usize,
            "a palette cannot index more than 2^32 distinct values"
        );
        self.values.push(value);
        self.counts.push(0);

        let needed: u32 = Self::bits_for(self.values.len());

        if needed > self.bits {
            self.widen(needed);
        }

        self.values.len() - 1
    }

    /// Repacks every index at a new width.
    fn widen(&mut self, bits: u32) {
        let indices: Vec<usize> = (0..self.len)
            .map(|position| self.index_at(position))
            .collect();

        self.bits = bits;
        self.packed = vec![0; self.words_needed(bits)];

        for (position, index) in indices.into_iter().enumerate() {
            self.write_index(position, index);
        }
    }

    /// The palette index stored at a position.
    ///
    /// An index never straddles a word: the widths are powers of two, so a
    /// whole number of them fits in 64 bits, which is what keeps a read to one
    /// shift and one mask.
    fn index_at(&self, position: usize) -> usize {
        if self.bits == 0 {
            return 0;
        }

        let per_word: usize = u64::BITS as usize / self.bits as usize;
        let word: u64 = self.packed[position / per_word];
        let offset: u32 = (position % per_word) as u32 * self.bits;

        ((word >> offset) & ((1 << self.bits) - 1)) as usize
    }

    /// Stores a palette index at a position.
    fn write_index(&mut self, position: usize, index: usize) {
        if self.bits == 0 {
            return;
        }

        let per_word: usize = u64::BITS as usize / self.bits as usize;
        let offset: u32 = (position % per_word) as u32 * self.bits;
        let mask: u64 = ((1u64 << self.bits) - 1) << offset;
        let word: &mut u64 = &mut self.packed[position / per_word];

        *word = (*word & !mask) | ((index as u64) << offset);
    }

    /// How many words the packed indices take at a width.
    fn words_needed(&self, bits: u32) -> usize {
        if bits == 0 {
            return 0;
        }

        let per_word: usize = u64::BITS as usize / bits as usize;

        self.len.div_ceil(per_word)
    }

    /// The narrowest power-of-two width that can number `count` values: 0 for
    /// one value, 1 for two, 2 for up to four, 4 for up to sixteen.
    fn bits_for(count: usize) -> u32 {
        match count {
            0 | 1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=16 => 4,
            17..=256 => 8,
            257..=65536 => 16,
            _ => 32,
        }
    }
}

impl<T> DeterministicOrder for Palette<T> {}
impl<T> CanonicalOrder for Palette<T> {}

impl<T: Clone + PartialEq> Collection for Palette<T> {
    type Item = T;
    fn len(&self) -> usize {
        self.len
    }
    fn contains(&self, item: &T) -> bool {
        self.count_of(item) > 0
    }
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: Clone + PartialEq> Sequence for Palette<T> {
    fn get(&self, index: usize) -> Option<&T> {
        Palette::get(self, index)
    }
    fn index_of(&self, item: &T) -> Option<usize> {
        self.iter().position(|held| held == item)
    }
}

impl<T: Clone + PartialEq> Choose for Palette<T> {
    fn choose(&self, random: &mut Random) -> Option<&T> {
        (!self.is_empty())
            .then(|| self.get(random.uniform_index(self.len)))
            .flatten()
    }
}

/// The values in position order, so two stores holding the same run of values
/// agree however their palettes are arranged internally.
impl<T: Clone + PartialEq + StableHash> ContentHashable for Palette<T> {
    fn content_hash(&self) -> ContentHash {
        self.iter().fold(ContentHash::EMPTY, |state, value| {
            state.and(value.stable_hash())
        })
    }
}

impl<T: Clone + PartialEq> FromIterator<T> for Palette<T> {
    /// From one value per position, in order.
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let values: Vec<T> = values.into_iter().collect();

        let Some(first) = values.first().cloned() else {
            return Self {
                values: Vec::new(),
                counts: Vec::new(),
                packed: Vec::new(),
                bits: 0,
                len: 0,
            };
        };

        let mut store: Self = Self::filled(values.len(), first);

        for (position, value) in values.into_iter().enumerate() {
            store.set(position, value);
        }

        store
    }
}

/// As the palette and what it costs, such as
/// `palette of 3 over 4096 positions at 2 bits`, since printing every position
/// is rarely what is wanted.
impl<T: Clone + PartialEq> std::fmt::Display for Palette<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "palette of {} over {} positions at {} bits",
            self.values.len(),
            self.len,
            self.bits
        )
    }
}
