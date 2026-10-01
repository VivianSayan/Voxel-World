//! A priority queue for small whole-numbered priorities, where every operation
//! is a constant handful of steps.
//!
//! [`PriorityQueue`](super::PriorityQueue) orders any priority at all, and pays
//! a tree walk for it. When the priorities are a short known range — a light
//! level from 0 to 15, a job class, a distance band — the tree is unnecessary:
//! one list per priority, and a cursor that only ever moves one way finds the
//! next item without comparing anything.
//!
//! That is what a flood fill wants. Propagating light from a source visits
//! cells in falling brightness, pushing neighbours one level dimmer; with
//! sixteen buckets each push is an append and each pop is a step of the cursor,
//! so the whole fill is linear in the cells it touches rather than
//! `n log n`.

use crate::structures::traits::{
    CanonicalOrder, Collection, ContentHashable, DeterministicOrder, Element, Partitioned,
    PriorityQueueLike, StableHash,
};
use crate::units::digest::ContentHash;
use std::collections::VecDeque;

/// A queue of items bucketed by a priority in `0..levels`, served lowest
/// priority first.
///
/// Within a bucket, items come out in the order they went in, so equal
/// priorities keep their arrival order and a replay produces the same sequence.
///
/// The cursor never moves backwards on its own: [`BucketQueue::pop`] leaves it
/// where it found work, and pushing *below* it moves it back, which is what a
/// fill that discovers a brighter route needs. A queue that is drained and
/// refilled should be [`BucketQueue::reset`] rather than trusted to rewind.
#[derive(Clone, Debug)]
pub struct BucketQueue<T> {
    buckets: Vec<VecDeque<T>>,
    cursor: usize,
    items: usize,
}

impl<T> BucketQueue<T> {
    /// An empty queue with `levels` priorities, `0..levels`.
    pub fn new(levels: usize) -> Self {
        Self {
            buckets: (0..levels).map(|_| VecDeque::new()).collect(),
            cursor: 0,
            items: 0,
        }
    }

    /// How many priorities the queue has.
    pub fn levels(&self) -> usize {
        self.buckets.len()
    }

    /// How many items are waiting, across every priority.
    pub const fn len(&self) -> usize {
        self.items
    }

    /// Whether nothing is waiting.
    pub const fn is_empty(&self) -> bool {
        self.items == 0
    }

    /// Adds an item at a priority, behind anything already there.
    ///
    /// Moves the cursor back when the priority is below it, so work discovered
    /// late is still served in order. Ignores a priority at or above
    /// [`BucketQueue::levels`], which is the one thing a bucket queue cannot
    /// represent; check with [`BucketQueue::accepts`] where that is possible.
    pub fn push(&mut self, priority: usize, item: T) -> bool {
        let Some(bucket) = self.buckets.get_mut(priority) else {
            return false;
        };

        bucket.push_back(item);
        self.items += 1;
        self.cursor = self.cursor.min(priority);

        true
    }

    /// Whether a priority is one this queue has a bucket for.
    pub fn accepts(&self, priority: usize) -> bool {
        priority < self.buckets.len()
    }

    /// The next item and its priority, lowest priority first, oldest first
    /// within a priority.
    ///
    /// Advances the cursor past the buckets it finds empty, which is what makes
    /// the whole drain linear: each bucket is passed once, not once per item.
    pub fn pop(&mut self) -> Option<(usize, T)> {
        while self.cursor < self.buckets.len() {
            if let Some(item) = self.buckets[self.cursor].pop_front() {
                self.items -= 1;

                return Some((self.cursor, item));
            }

            self.cursor += 1;
        }

        None
    }

    /// The priority the next item would come from, without taking it.
    pub fn peek_priority(&self) -> Option<usize> {
        self.buckets
            .iter()
            .enumerate()
            .skip(self.cursor)
            .find(|(_, bucket)| !bucket.is_empty())
            .map(|(priority, _)| priority)
    }

    /// The next item without taking it.
    pub fn peek(&self) -> Option<(usize, &T)> {
        self.buckets
            .iter()
            .enumerate()
            .skip(self.cursor)
            .find_map(|(priority, bucket)| Some((priority, bucket.front()?)))
    }

    /// How many items sit at one priority.
    pub fn len_at(&self, priority: usize) -> usize {
        self.buckets.get(priority).map_or(0, VecDeque::len)
    }

    /// The items at one priority, oldest first.
    pub fn at(&self, priority: usize) -> impl DoubleEndedIterator<Item = &T> {
        self.buckets.get(priority).into_iter().flatten()
    }

    /// Reserves room in one particular priority bucket.
    pub fn reserve_for(&mut self, priority: usize, additional: usize) -> bool {
        let Some(bucket) = self.buckets.get_mut(priority) else {
            return false;
        };
        bucket.reserve(additional);
        true
    }

    /// Releases spare allocation in every priority bucket.
    pub fn shrink_to_fit(&mut self) {
        for bucket in &mut self.buckets {
            bucket.shrink_to_fit();
        }
    }

    /// Every item with its priority, lowest first and oldest first within a
    /// priority: the order [`BucketQueue::pop`] would produce.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.buckets
            .iter()
            .enumerate()
            .flat_map(|(priority, bucket)| bucket.iter().map(move |item| (priority, item)))
    }

    /// Empties every bucket and returns the cursor to the start.
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }

        self.cursor = 0;
        self.items = 0;
    }

    /// Returns the cursor to the start without touching the contents.
    ///
    /// For a queue that is refilled after a drain: the cursor is left at the
    /// end by draining, and pushing only ever moves it back as far as the
    /// priority pushed.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

impl<T: PartialEq> Collection for BucketQueue<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items
    }

    fn contains(&self, item: &T) -> bool {
        self.buckets.iter().any(|bucket| bucket.contains(item))
    }

    /// In the order [`BucketQueue::pop`] would serve them.
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.buckets.iter().flatten()
    }
}

impl<T> Default for BucketQueue<T> {
    /// Sixteen levels, which is what light propagation wants.
    fn default() -> Self {
        Self::new(16)
    }
}

impl<T> PriorityQueueLike for BucketQueue<T> {
    type Item = T;
    type Priority = usize;

    fn push_priority(&mut self, priority: usize, item: T) -> bool {
        self.push(priority, item)
    }

    fn peek_priority_item(&self) -> Option<(usize, &T)> {
        self.peek()
    }

    fn pop_priority(&mut self) -> Option<(usize, T)> {
        self.pop()
    }
}

/// The buckets, numbered by priority and in arrival order within each.
///
/// Read-only, and deliberately not a
/// [`BalancedPartition`](crate::structures::traits::BalancedPartition): a
/// bucket queue divides its work by priority, so uneven buckets are the point
/// rather than a fault.
impl<T: PartialEq> Partitioned for BucketQueue<T> {
    type Member = T;

    fn group_count(&self) -> usize {
        self.levels()
    }

    fn group_len(&self, group: usize) -> usize {
        self.len_at(group)
    }

    fn group_members(&self, group: usize) -> impl Iterator<Item = &T> {
        self.at(group)
    }
}

impl<T> DeterministicOrder for BucketQueue<T> {}
impl<T> CanonicalOrder for BucketQueue<T> {}

/// In serving order, so two queues holding the same items at the same
/// priorities agree, and one that would serve them differently does not.
impl<T: Element + StableHash> ContentHashable for BucketQueue<T> {
    fn content_hash(&self) -> ContentHash {
        self.iter()
            .fold(ContentHash::EMPTY, |state, (priority, item)| {
                state.and_value(priority as u128).and(item.stable_hash())
            })
    }
}

impl<T> FromIterator<(usize, T)> for BucketQueue<T> {
    /// From `(priority, item)` pairs, with as many levels as the highest
    /// priority needs.
    fn from_iter<I: IntoIterator<Item = (usize, T)>>(entries: I) -> Self {
        let entries: Vec<(usize, T)> = entries.into_iter().collect();
        let levels: usize = entries
            .iter()
            .map(|(priority, _)| priority + 1)
            .max()
            .unwrap_or(0);

        let mut queue: Self = Self::new(levels);

        for (priority, item) in entries {
            queue.push(priority, item);
        }

        queue
    }
}

impl<T> Extend<(usize, T)> for BucketQueue<T> {
    /// Adds each pair, ignoring any priority the queue has no bucket for.
    fn extend<I: IntoIterator<Item = (usize, T)>>(&mut self, entries: I) {
        for (priority, item) in entries {
            self.push(priority, item);
        }
    }
}

/// As what waits at each priority, such as `[0: 2, 3: 1]`, skipping the empty
/// ones.
impl<T> std::fmt::Display for BucketQueue<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("[")?;

        let mut written: usize = 0;

        for (priority, bucket) in self.buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }

            if written > 0 {
                formatter.write_str(", ")?;
            }

            write!(formatter, "{priority}: {}", bucket.len())?;
            written += 1;
        }

        formatter.write_str("]")
    }
}
