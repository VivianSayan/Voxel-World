//! A set of small non-negative integers, one bit each.
//!
//! Where a [`Set`](super::Set) of `usize` costs a hash table entry per member,
//! this costs one bit per *possible* member, which is the right trade whenever
//! the members are dense, bounded and numerous: which of a chunk's 4096 voxels
//! are solid, which of its faces are exposed, which chunks in view need
//! remeshing. A whole chunk's occupancy is 512 bytes and its union with another
//! is 64 machine words.
//!
//! It is also the natural home for what
//! [`BernoulliMask`](crate::random::BernoulliMask) produces: that sampler hands
//! back 64 independent decisions as a `u64`, and [`BitSet::from_word`] and
//! [`BitSet::insert_word`] take them without unpacking.

use crate::random::source::StochasticSource;
use crate::structures::traits::{
    CanonicalOrder, ContentHashable, DeterministicOrder, ValueCollection, ValueSetAlgebra,
};
use crate::units::digest::ContentHash;
use std::fmt;
use std::hash::{Hash, Hasher};

/// How many members one stored word covers.
const BITS_PER_WORD: usize = u64::BITS as usize;

/// A set of `usize` members held as a bitmap, with the count kept alongside.
///
/// Members are non-negative and bounded only by memory: inserting member `n`
/// grows the map to `n / 64 + 1` words. Removing never shrinks it, so a set
/// that once held a large member keeps the room until
/// [`BitSet::shrink_to_fit`] is called, which also drops the trailing empty
/// words.
///
/// Iteration is in increasing order of member, which makes it
/// [`DeterministicOrder`] without qualification: there is no hashing anywhere
/// and no order to depend on the contents.
#[derive(Clone, Default)]
pub struct BitSet {
    words: Vec<u64>,
    members: usize,
}

impl BitSet {
    /// An empty set holding no words at all.
    pub const fn new() -> Self {
        Self {
            words: Vec::new(),
            members: 0,
        }
    }

    /// An empty set with room for members below `bits`, so that inserting any
    /// of them cannot reallocate.
    pub fn with_bits(bits: usize) -> Self {
        Self {
            words: Vec::with_capacity(bits.div_ceil(BITS_PER_WORD)),
            members: 0,
        }
    }

    /// The members of one word, starting at member zero: bit `n` of `word` is
    /// member `n`.
    ///
    /// The shape [`BernoulliMask`](crate::random::BernoulliMask) hands back, so
    /// sixty-four seeded decisions become a set in one step.
    pub fn from_word(word: u64) -> Self {
        Self {
            words: vec![word],
            members: word.count_ones() as usize,
        }
    }

    /// Whether `member` is in the set.
    pub fn contains(&self, member: usize) -> bool {
        let (word, bit): (usize, u32) = Self::place(member);

        self.words
            .get(word)
            .is_some_and(|value| value & (1 << bit) != 0)
    }

    /// Adds a member, growing the map if it is beyond the current end. Returns
    /// whether it was new.
    pub fn insert(&mut self, member: usize) -> bool {
        let (word, bit): (usize, u32) = Self::place(member);

        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }

        let held: bool = self.words[word] & (1 << bit) != 0;

        if !held {
            self.words[word] |= 1 << bit;
            self.members += 1;
        }

        !held
    }

    /// Removes a member. Returns whether it was there. The map keeps its size.
    pub fn remove(&mut self, member: usize) -> bool {
        let (word, bit): (usize, u32) = Self::place(member);

        let Some(value) = self.words.get_mut(word) else {
            return false;
        };

        let held: bool = *value & (1 << bit) != 0;

        if held {
            *value &= !(1 << bit);
            self.members -= 1;
        }

        held
    }

    /// Adds sixty-four members at once: bit `n` of `word` is member
    /// `index * 64 + n`.
    ///
    /// The word is or-ed in, so members already present stay. Takes a whole
    /// mask from a sampler without a loop over its bits.
    pub fn insert_word(&mut self, index: usize, word: u64) {
        if index >= self.words.len() {
            self.words.resize(index + 1, 0);
        }

        let added: u32 = (word & !self.words[index]).count_ones();

        self.words[index] |= word;
        self.members += added as usize;
    }

    /// The word covering members `index * 64` upwards, or zero past the end.
    pub fn word(&self, index: usize) -> u64 {
        self.words.get(index).copied().unwrap_or(0)
    }

    /// Every stored word, lowest members first.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// How many members the set holds. Kept as the set changes rather than
    /// counted here.
    pub const fn len(&self) -> usize {
        self.members
    }

    /// Whether the set holds nothing.
    pub const fn is_empty(&self) -> bool {
        self.members == 0
    }

    /// Removes every member, keeping the room.
    pub fn clear(&mut self) {
        self.words.clear();
        self.members = 0;
    }

    /// The smallest member, or `None` when empty.
    pub fn first(&self) -> Option<usize> {
        self.words
            .iter()
            .enumerate()
            .find(|(_, word)| **word != 0)
            .map(|(index, word)| index * BITS_PER_WORD + word.trailing_zeros() as usize)
    }

    /// The largest member, or `None` when empty.
    pub fn last(&self) -> Option<usize> {
        self.words
            .iter()
            .enumerate()
            .rev()
            .find(|(_, word)| **word != 0)
            .map(|(index, word)| {
                index * BITS_PER_WORD + (BITS_PER_WORD - 1 - word.leading_zeros() as usize)
            })
    }

    /// Every member, in increasing order.
    ///
    /// Walks whole words and skips empty ones, so an almost-empty set spread
    /// over a wide range costs one step per word rather than one per possible
    /// member.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(index, word)| {
            SetBits::new(*word).map(move |bit| index * BITS_PER_WORD + bit)
        })
    }

    /// Adds every member of `other`.
    pub fn insert_all(&mut self, other: &Self) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }

        self.members = 0;

        for (index, word) in self.words.iter_mut().enumerate() {
            *word |= other.word(index);
            self.members += word.count_ones() as usize;
        }
    }

    /// Removes every member of `other`.
    pub fn remove_all(&mut self, other: &Self) {
        self.members = 0;

        for (index, word) in self.words.iter_mut().enumerate() {
            *word &= !other.word(index);
            self.members += word.count_ones() as usize;
        }
    }

    /// Keeps only the members `other` also holds.
    pub fn retain_all(&mut self, other: &Self) {
        self.members = 0;

        for (index, word) in self.words.iter_mut().enumerate() {
            *word &= other.word(index);
            self.members += word.count_ones() as usize;
        }
    }

    /// Which word holds a member, and which bit of it.
    const fn place(member: usize) -> (usize, u32) {
        (member / BITS_PER_WORD, (member % BITS_PER_WORD) as u32)
    }

    /// Stored words without allocation-only zeroes at the high end.
    fn logical_words(&self) -> &[u64] {
        let len = self
            .words
            .iter()
            .rposition(|word| *word != 0)
            .map_or(0, |index| index + 1);
        &self.words[..len]
    }

    /// Combines two sets word by word, taking the longer length.
    fn combine(&self, other: &Self, mut operation: impl FnMut(u64, u64) -> u64) -> Self {
        let length: usize = self.words.len().max(other.words.len());
        let words: Vec<u64> = (0..length)
            .map(|index| operation(self.word(index), other.word(index)))
            .collect();
        let members: usize = words.iter().map(|word| word.count_ones() as usize).sum();

        Self { words, members }
    }
}

/// The set bits of one word, lowest first, found by counting trailing zeros
/// rather than by testing each bit.
struct SetBits(u64);

impl SetBits {
    const fn new(word: u64) -> Self {
        Self(word)
    }
}

impl Iterator for SetBits {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }

        let bit: usize = self.0.trailing_zeros() as usize;

        self.0 &= self.0 - 1;

        Some(bit)
    }
}

impl DoubleEndedIterator for SetBits {
    fn next_back(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }

        let bit: usize = BITS_PER_WORD - 1 - self.0.leading_zeros() as usize;

        self.0 &= !(1 << bit);

        Some(bit)
    }
}

/// Members are computed from the bits, so they cannot be borrowed and this is a
/// [`ValueCollection`] rather than a [`Collection`](crate::structures::traits::Collection).
/// Everything a set does is here as an inherent method, and the set operators
/// are implemented directly below.
impl ValueCollection for BitSet {
    type Value = usize;

    fn len(&self) -> usize {
        self.members
    }

    fn contains_value(&self, value: usize) -> bool {
        self.contains(value)
    }

    fn values(&self) -> impl Iterator<Item = usize> {
        self.iter()
    }
}

impl ValueSetAlgebra for BitSet {
    fn union(&self, other: &Self) -> Self {
        BitSet::union(self, other)
    }
    fn intersection(&self, other: &Self) -> Self {
        BitSet::intersection(self, other)
    }
    fn difference(&self, other: &Self) -> Self {
        BitSet::difference(self, other)
    }
    fn symmetric_difference(&self, other: &Self) -> Self {
        BitSet::symmetric_difference(self, other)
    }
    fn is_subset(&self, other: &Self) -> bool {
        BitSet::is_subset(self, other)
    }
    fn is_disjoint(&self, other: &Self) -> bool {
        BitSet::is_disjoint(self, other)
    }
}

impl BitSet {
    /// Every member of either set.
    pub fn union(&self, other: &Self) -> Self {
        self.combine(other, |left, right| left | right)
    }

    /// Every member of both sets.
    pub fn intersection(&self, other: &Self) -> Self {
        self.combine(other, |left, right| left & right)
    }

    /// Every member of this set that `other` does not hold.
    pub fn difference(&self, other: &Self) -> Self {
        self.combine(other, |left, right| left & !right)
    }

    /// Every member held by exactly one of the two.
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        self.combine(other, |left, right| left ^ right)
    }

    /// Whether every member of this set is also in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.words
            .iter()
            .enumerate()
            .all(|(index, word)| word & !other.word(index) == 0)
    }

    /// Whether this set holds every member of `other`.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Whether the two share no member.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.words
            .iter()
            .enumerate()
            .all(|(index, word)| word & other.word(index) == 0)
    }

    /// One member, uniformly among those held, or `None` when empty.
    ///
    /// Counts into the set bits: the draw picks a position among the members
    /// and the words are walked until that many have been passed, so the cost
    /// is one step per word rather than one per member.
    pub fn choose_member<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Option<usize> {
        if self.is_empty() {
            return None;
        }

        let mut remaining: usize = source.index_below(self.members);

        for (index, word) in self.words.iter().enumerate() {
            let held: usize = word.count_ones() as usize;

            if remaining < held {
                return SetBits::new(*word)
                    .nth(remaining)
                    .map(|bit| index * BITS_PER_WORD + bit);
            }

            remaining -= held;
        }

        None
    }

    /// How many members fit before the map has to grow, rounded to whole words.
    pub fn capacity_in_bits(&self) -> usize {
        self.words.capacity() * BITS_PER_WORD
    }

    /// Makes room for members below `bits` without reallocating.
    pub fn reserve_bits(&mut self, bits: usize) {
        self.words.reserve(bits.div_ceil(BITS_PER_WORD));
    }

    /// Drops the trailing words that hold nothing, and the spare room with
    /// them, which is the only way a bitmap gives memory back.
    pub fn shrink_to_fit(&mut self) {
        while self.words.last() == Some(&0) {
            self.words.pop();
        }

        self.words.shrink_to_fit();
    }
}

/// Every operation is word-parallel: a union of two chunk-sized sets is
/// sixty-four instructions rather than four thousand.
impl std::ops::BitOr<&BitSet> for &BitSet {
    type Output = BitSet;

    fn bitor(self, other: &BitSet) -> BitSet {
        self.union(other)
    }
}

impl std::ops::BitAnd<&BitSet> for &BitSet {
    type Output = BitSet;

    fn bitand(self, other: &BitSet) -> BitSet {
        self.intersection(other)
    }
}

impl std::ops::Sub<&BitSet> for &BitSet {
    type Output = BitSet;

    fn sub(self, other: &BitSet) -> BitSet {
        self.difference(other)
    }
}

impl std::ops::BitXor<&BitSet> for &BitSet {
    type Output = BitSet;

    fn bitxor(self, other: &BitSet) -> BitSet {
        self.symmetric_difference(other)
    }
}

impl DeterministicOrder for BitSet {}
impl CanonicalOrder for BitSet {}

impl PartialEq for BitSet {
    fn eq(&self, other: &Self) -> bool {
        self.logical_words() == other.logical_words()
    }
}

impl Eq for BitSet {}

impl Hash for BitSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.logical_words().hash(state);
    }
}

/// The words themselves, in order, which is both the contents and the only
/// order they have.
impl ContentHashable for BitSet {
    fn content_hash(&self) -> ContentHash {
        self.logical_words()
            .iter()
            .fold(ContentHash::EMPTY, |state, word| {
                state.and_value(*word as u128)
            })
    }
}

impl FromIterator<usize> for BitSet {
    fn from_iter<I: IntoIterator<Item = usize>>(members: I) -> Self {
        let mut set: Self = Self::new();

        set.extend(members);
        set
    }
}

impl Extend<usize> for BitSet {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, members: I) {
        for member in members {
            BitSet::insert(self, member);
        }
    }
}

impl IntoIterator for &BitSet {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<usize>>().into_iter()
    }
}

impl fmt::Debug for BitSet {
    /// As the members it holds, such as `BitSet{0, 3, 64}`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "BitSet")?;
        formatter.debug_set().entries(self.iter()).finish()
    }
}
