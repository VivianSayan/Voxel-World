//! Values attached to spans of an ordered key rather than to single keys.
//!
//! A world has rules that hold over bands: sea level to the snow line is
//! tundra, everything below sixteen is deep stone, this depth range spawns that
//! ore. Storing a value per key would need an entry for every height; storing
//! the boundaries needs one per band, and answering "which band is this?" is a
//! binary search.
//!
//! The spans are half-open, `start..end`, and they never overlap: inserting one
//! that crosses another cuts the old one back, which is what makes a lookup a
//! single answer rather than a list. Gaps are allowed and read as absent.

use crate::structures::traits::{
    CanonicalOrder, Capacity, ContentHashable, DeterministicOrder, RangeQuery, StableHash,
};
use crate::units::digest::ContentHash;
use std::fmt;
use std::ops::{Bound, Range, RangeBounds};

/// One span and what it holds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Span<K, V> {
    start: K,
    end: K,
    value: V,
}

/// A map from half-open spans of `K` to values, kept sorted and non-
/// overlapping.
///
/// Later insertions win: a span laid over an existing one trims or splits it,
/// the way a later rule overrides an earlier one. That makes the order of
/// insertion part of the result, so a map built from the same spans in a
/// different order can differ — which is the behaviour a rule list wants, and
/// worth knowing when building one from unordered data.
///
/// Neighbouring spans holding equal values are merged, so a map that is filled
/// piecewise with one value ends up as one span.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntervalMap<K, V> {
    spans: Vec<Span<K, V>>,
}

impl<K: Ord + Clone, V: Clone + PartialEq> IntervalMap<K, V> {
    /// An empty map, covering nothing.
    pub const fn new() -> Self {
        Self { spans: Vec::new() }
    }

    /// How many spans the map holds, which is not how many keys they cover.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the map covers nothing.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The value covering `key`, or `None` where nothing does.
    ///
    /// A binary search over the spans: O(log n) however wide they are.
    pub fn get(&self, key: &K) -> Option<&V> {
        let index: usize = match self.spans.binary_search_by(|span| span.start.cmp(key)) {
            Ok(exact) => exact,
            Err(0) => return None,
            Err(after) => after - 1,
        };

        let span: &Span<K, V> = &self.spans[index];

        (span.end > *key).then_some(&span.value)
    }

    /// The span covering `key` with its value, or `None` where nothing does.
    pub fn span_of(&self, key: &K) -> Option<(Range<&K>, &V)> {
        let index: usize = match self.spans.binary_search_by(|span| span.start.cmp(key)) {
            Ok(exact) => exact,
            Err(0) => return None,
            Err(after) => after - 1,
        };

        let span: &Span<K, V> = &self.spans[index];

        (span.end > *key).then_some(((&span.start..&span.end), &span.value))
    }

    /// Whether any span covers `key`.
    pub fn covers(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Attaches `value` to `start..end`, cutting back whatever was there.
    ///
    /// An empty or reversed span does nothing. Spans that the new one covers
    /// entirely are dropped, ones it overlaps are trimmed, and one it lands
    /// inside is split in two.
    pub fn insert(&mut self, start: K, end: K, value: V) {
        if start >= end {
            return;
        }

        let mut kept: Vec<Span<K, V>> = Vec::with_capacity(self.spans.len() + 2);

        for span in self.spans.drain(..) {
            // Entirely before or after the new span: keep as it is.
            if span.end <= start || span.start >= end {
                kept.push(span);

                continue;
            }

            // The part sticking out below, and the part sticking out above.
            if span.start < start {
                kept.push(Span {
                    start: span.start.clone(),
                    end: start.clone(),
                    value: span.value.clone(),
                });
            }

            if span.end > end {
                kept.push(Span {
                    start: end.clone(),
                    end: span.end.clone(),
                    value: span.value.clone(),
                });
            }
        }

        kept.push(Span { start, end, value });
        kept.sort_by(|left, right| left.start.cmp(&right.start));

        self.spans = kept;
        self.merge_neighbours();
    }

    /// Removes whatever covers `start..end`, leaving a gap.
    pub fn remove(&mut self, start: K, end: K) {
        if start >= end {
            return;
        }

        let mut kept: Vec<Span<K, V>> = Vec::with_capacity(self.spans.len() + 1);

        for span in self.spans.drain(..) {
            if span.end <= start || span.start >= end {
                kept.push(span);

                continue;
            }

            if span.start < start {
                kept.push(Span {
                    start: span.start.clone(),
                    end: start.clone(),
                    value: span.value.clone(),
                });
            }

            if span.end > end {
                kept.push(Span {
                    start: end.clone(),
                    end: span.end.clone(),
                    value: span.value.clone(),
                });
            }
        }

        self.spans = kept;
    }

    /// Every span with its value, in key order.
    pub fn spans(&self) -> impl DoubleEndedIterator<Item = (Range<&K>, &V)> {
        self.spans
            .iter()
            .map(|span| ((&span.start..&span.end), &span.value))
    }

    /// The values, in key order.
    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> {
        self.spans.iter().map(|span| &span.value)
    }

    /// Empties the map.
    pub fn clear(&mut self) {
        self.spans.clear();
    }

    /// Joins neighbouring spans that touch and hold equal values, so the map
    /// stays as short as its contents allow.
    fn merge_neighbours(&mut self) {
        let mut index: usize = 0;

        while index + 1 < self.spans.len() {
            if self.spans[index].end == self.spans[index + 1].start
                && self.spans[index].value == self.spans[index + 1].value
            {
                self.spans[index].end = self.spans[index + 1].end.clone();
                self.spans.remove(index + 1);
            } else {
                index += 1;
            }
        }
    }
}

/// Ordered by key, and each match is a span that overlaps the range asked for.
impl<K: Ord + Clone, V: Clone + PartialEq> RangeQuery for IntervalMap<K, V> {
    type Key = K;
    type Item<'a>
        = (Range<&'a K>, &'a V)
    where
        Self: 'a;

    fn range<'a, R: RangeBounds<K>>(
        &'a self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = Self::Item<'a>> {
        let first = match range.start_bound() {
            Bound::Included(key) | Bound::Excluded(key) => {
                self.spans.partition_point(|span| span.end <= *key)
            }
            Bound::Unbounded => 0,
        };
        let last = match range.end_bound() {
            Bound::Included(key) => self.spans.partition_point(|span| span.start <= *key),
            Bound::Excluded(key) => self.spans.partition_point(|span| span.start < *key),
            Bound::Unbounded => self.spans.len(),
        };

        self.spans[first.min(last)..last]
            .iter()
            .map(|span| ((&span.start..&span.end), &span.value))
    }
}

/// Room is counted in spans.
impl<K: Ord + Clone, V: Clone + PartialEq> Capacity for IntervalMap<K, V> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            spans: Vec::with_capacity(capacity),
        }
    }

    fn capacity(&self) -> usize {
        self.spans.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.spans.reserve(additional);
    }

    fn shrink_to_fit(&mut self) {
        self.spans.shrink_to_fit();
    }
}

impl<K, V> DeterministicOrder for IntervalMap<K, V> {}
impl<K, V> CanonicalOrder for IntervalMap<K, V> {}

/// The spans in key order, bounds and values alike.
impl<K: Ord + Clone + StableHash, V: Clone + PartialEq + StableHash> ContentHashable
    for IntervalMap<K, V>
{
    fn content_hash(&self) -> ContentHash {
        self.spans.iter().fold(ContentHash::EMPTY, |state, span| {
            state
                .and(span.start.stable_hash())
                .and(span.end.stable_hash())
                .and(span.value.stable_hash())
        })
    }
}

impl<K: Ord + Clone, V: Clone + PartialEq> FromIterator<(Range<K>, V)> for IntervalMap<K, V> {
    /// From spans in order, each overriding what it overlaps.
    fn from_iter<I: IntoIterator<Item = (Range<K>, V)>>(spans: I) -> Self {
        let mut map: Self = Self::new();

        map.extend(spans);
        map
    }
}

impl<K: Ord + Clone, V: Clone + PartialEq> Extend<(Range<K>, V)> for IntervalMap<K, V> {
    fn extend<I: IntoIterator<Item = (Range<K>, V)>>(&mut self, spans: I) {
        for (span, value) in spans {
            self.insert(span.start, span.end, value);
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Display for IntervalMap<K, V> {
    /// As the spans it holds, such as `[0..64 = Stone, 64..80 = Dirt]`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[")?;

        for (index, span) in self.spans.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }

            write!(
                formatter,
                "{:?}..{:?} = {:?}",
                span.start, span.end, span.value
            )?;
        }

        formatter.write_str("]")
    }
}
