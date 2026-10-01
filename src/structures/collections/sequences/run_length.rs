//! A sequence stored as runs of equal values rather than one entry each.
//!
//! A voxel column is mostly stone, then mostly dirt, then air all the way up.
//! Storing that as one value per voxel spends a word on each of several hundred
//! identical entries; storing it as "stone from 0 to 61, dirt 62 to 63, air 64
//! to 319" spends three. The saving grows with the world: a column through
//! empty sky is one run however tall it is.
//!
//! The trade is the usual one for compression. Reading a position is a binary
//! search over the runs rather than an index, and writing in the middle of a run
//! splits it, so a sequence that is written randomly and densely ends up with a
//! run per position and is worse than a plain list. Reach for this where the
//! data really is banded.

use crate::structures::traits::{
    CanonicalOrder, Capacity, Collection, ContentHashable, DeterministicOrder, RangeQuery,
    Sequence, StableHash,
};
use crate::units::digest::ContentHash;
use std::ops::{Bound, RangeBounds};

/// One run: a value and where it starts.
///
/// The end is implied by the next run's start, which is what keeps a run to one
/// value and one index and makes splitting cheap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Run<T> {
    start: usize,
    value: T,
}

/// A sequence of `length` positions, stored as the runs of equal values that
/// cover it.
///
/// Every position has a value: the sequence is built from a length and a filler
/// and is never sparse, unlike
/// [`SparseSequence`](crate::structures::collections::SparseSequence), which
/// leaves gaps. The two answer different questions — this one compresses a
/// dense sequence, that one addresses a scattered one.
///
/// Runs are kept maximal: writing a value equal to its neighbour's merges with
/// it rather than adding a run, so the storage never grows from a write that
/// changes nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunLengthSequence<T> {
    runs: Vec<Run<T>>,
    length: usize,
}

impl<T: Clone + PartialEq> RunLengthSequence<T> {
    /// A sequence of `length` positions, every one holding `value`: one run
    /// however long it is.
    pub fn filled(length: usize, value: T) -> Self {
        if length == 0 {
            return Self {
                runs: Vec::new(),
                length: 0,
            };
        }

        Self {
            runs: vec![Run { start: 0, value }],
            length,
        }
    }

    /// An empty sequence, holding no positions at all.
    pub const fn new() -> Self {
        Self {
            runs: Vec::new(),
            length: 0,
        }
    }

    /// How many positions the sequence covers.
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Whether it covers none.
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// How many runs the values are stored as, which is what the compression is
    /// worth: equal to [`RunLengthSequence::len`] in the worst case and one in
    /// the best.
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// The value at a position, or `None` past the end.
    ///
    /// A binary search over the runs, so O(log runs) rather than the O(1) of a
    /// list.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.length {
            return None;
        }

        Some(&self.runs[self.run_at(index)].value)
    }

    /// Sets one position, splitting or merging runs as needed. Returns whether
    /// the value changed.
    ///
    /// Writing the value a position already holds costs a search and nothing
    /// else. Writing a different one costs a search and a splice, which is
    /// linear in the runs after it.
    pub fn set(&mut self, index: usize, value: T) -> bool {
        if index >= self.length || self.runs[self.run_at(index)].value == value {
            return false;
        }

        self.write_range(index, index + 1, value);

        true
    }

    /// Sets every position in `start..end` to one value, which is what run-
    /// length storage is good at: one splice however wide the span.
    pub fn fill(&mut self, start: usize, end: usize, value: T) {
        let end: usize = end.min(self.length);

        if start >= end {
            return;
        }

        self.write_range(start, end, value);
    }

    /// Adds a position at the end, joining the last run if the value matches.
    pub fn push(&mut self, value: T) {
        if self.runs.last().is_some_and(|run| run.value == value) {
            self.length += 1;

            return;
        }

        self.runs.push(Run {
            start: self.length,
            value,
        });
        self.length += 1;
    }

    /// Every run, as `(start, end, value)` with `end` exclusive.
    pub fn runs(&self) -> impl DoubleEndedIterator<Item = (usize, usize, &T)> {
        self.runs.iter().enumerate().map(|(index, run)| {
            let end: usize = self
                .runs
                .get(index + 1)
                .map_or(self.length, |next| next.start);

            (run.start, end, &run.value)
        })
    }

    /// Every position's value, one at a time, as a plain sequence would.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.runs()
            .flat_map(|(start, end, value)| (start..end).map(move |_| value))
    }

    /// Whether every position holds the same value, which is the case a voxel
    /// world checks constantly: a uniform chunk needs no storage of its own.
    pub fn is_uniform(&self) -> bool {
        self.runs.len() <= 1
    }

    /// The single value every position holds, or `None` when the sequence is
    /// empty or mixed.
    pub fn uniform_value(&self) -> Option<&T> {
        match self.runs.as_slice() {
            [run] => Some(&run.value),
            _ => None,
        }
    }

    /// Empties the sequence.
    pub fn clear(&mut self) {
        self.runs.clear();
        self.length = 0;
    }

    /// Which run covers a position, by binary search. The position must be
    /// inside the sequence.
    fn run_at(&self, index: usize) -> usize {
        match self.runs.binary_search_by_key(&index, |run| run.start) {
            Ok(exact) => exact,
            Err(after) => after - 1,
        }
    }

    /// Replaces `start..end` with one run of `value`, keeping what the span cut
    /// into on either side and merging with the neighbours where they match.
    fn write_range(&mut self, start: usize, end: usize, value: T) {
        let first: usize = self.run_at(start);
        let last: usize = self.run_at(end - 1);

        // Where the run being overwritten last actually ends, which is the
        // next run's start or the end of the sequence.
        let last_end: usize = self
            .runs
            .get(last + 1)
            .map_or(self.length, |next| next.start);

        // What the span leaves behind on each side, if anything. A span that
        // reaches exactly to the end of the last run it touches leaves no tail:
        // the run after it already starts there.
        let before: Option<Run<T>> = (self.runs[first].start < start).then(|| Run {
            start: self.runs[first].start,
            value: self.runs[first].value.clone(),
        });
        let after: Option<Run<T>> = (end < last_end).then(|| Run {
            start: end,
            value: self.runs[last].value.clone(),
        });

        let mut replacement: Vec<Run<T>> = Vec::with_capacity(3);

        if let Some(run) = before {
            replacement.push(run);
        }

        replacement.push(Run { start, value });

        if let Some(run) = after {
            replacement.push(run);
        }

        self.runs.splice(first..=last, replacement);
        self.merge_around(first);
    }

    /// Joins any neighbouring runs holding equal values, starting one before
    /// the splice and ending one after what it wrote.
    fn merge_around(&mut self, from: usize) {
        let mut index: usize = from.saturating_sub(1);

        while index + 1 < self.runs.len() {
            if self.runs[index].value == self.runs[index + 1].value {
                self.runs.remove(index + 1);
            } else {
                index += 1;
            }
        }
    }
}

impl<T: Clone + PartialEq> Collection for RunLengthSequence<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.length
    }

    fn contains(&self, item: &T) -> bool {
        self.runs.iter().any(|run| run.value == *item)
    }

    /// One entry per position, not per run, so this reads as the sequence it
    /// stands for. [`RunLengthSequence::runs`] is the compressed view.
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.values()
    }
}

impl<T: Clone + PartialEq> Sequence for RunLengthSequence<T> {
    fn get(&self, index: usize) -> Option<&T> {
        RunLengthSequence::get(self, index)
    }

    fn index_of(&self, item: &T) -> Option<usize> {
        self.runs
            .iter()
            .find(|run| run.value == *item)
            .map(|run| run.start)
    }
}

/// Ordered by position, and each match is the run covering it rather than a
/// position at a time, which is the cheap window a run-length store can offer.
impl<T: Clone + PartialEq> RangeQuery for RunLengthSequence<T> {
    type Key = usize;
    type Item<'a>
        = (usize, usize, &'a T)
    where
        Self: 'a;

    fn range<'a, R: RangeBounds<usize>>(
        &'a self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = Self::Item<'a>> {
        let start: usize = match range.start_bound() {
            Bound::Included(index) => *index,
            Bound::Excluded(index) => index.checked_add(1).unwrap_or(self.length),
            Bound::Unbounded => 0,
        }
        .min(self.length);
        let end: usize = match range.end_bound() {
            Bound::Included(index) => index.checked_add(1).unwrap_or(self.length).min(self.length),
            Bound::Excluded(index) => (*index).min(self.length),
            Bound::Unbounded => self.length,
        };
        let (first, last) = if start < end && !self.runs.is_empty() {
            let after_start = self.runs.partition_point(|run| run.start <= start);
            let first = after_start.saturating_sub(1);
            let last = self.runs.partition_point(|run| run.start < end);
            (first, last)
        } else {
            (0, 0)
        };

        self.runs[first..last]
            .iter()
            .enumerate()
            .map(move |(offset, run)| {
                let index = first + offset;
                let run_end = self
                    .runs
                    .get(index + 1)
                    .map_or(self.length, |next| next.start);
                (run.start, run_end, &run.value)
            })
    }
}

/// Room is counted in runs, since that is what is stored.
impl<T: Clone + PartialEq> Capacity for RunLengthSequence<T> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            runs: Vec::with_capacity(capacity),
            length: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.runs.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.runs.reserve(additional);
    }

    fn shrink_to_fit(&mut self) {
        self.runs.shrink_to_fit();
    }
}

impl<T> DeterministicOrder for RunLengthSequence<T> {}
impl<T> CanonicalOrder for RunLengthSequence<T> {}

/// The runs, in order, which settles both the values and where they change.
impl<T: Clone + PartialEq + StableHash> ContentHashable for RunLengthSequence<T> {
    fn content_hash(&self) -> ContentHash {
        self.runs()
            .fold(ContentHash::EMPTY, |state, (start, end, value)| {
                state
                    .and_value(start as u128)
                    .and_value(end as u128)
                    .and(value.stable_hash())
            })
    }
}

impl<T: Clone + PartialEq> FromIterator<T> for RunLengthSequence<T> {
    /// From one value per position, joining equal neighbours as they arrive.
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let mut sequence: Self = Self::new();

        for value in values {
            sequence.push(value);
        }

        sequence
    }
}

impl<T: Clone + PartialEq> Extend<T> for RunLengthSequence<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        for value in values {
            self.push(value);
        }
    }
}

/// As its runs, such as `[0..64 = Stone, 64..80 = Dirt]`, which is the
/// compressed form rather than one entry per position.
impl<T: Clone + PartialEq + std::fmt::Display> std::fmt::Display for RunLengthSequence<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("[")?;

        for (position, (start, end, value)) in self.runs().enumerate() {
            if position > 0 {
                formatter.write_str(", ")?;
            }

            write!(formatter, "{start}..{end} = {value}")?;
        }

        formatter.write_str("]")
    }
}
