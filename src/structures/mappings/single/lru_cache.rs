//! A map that forgets what has gone longest unused.
//!
//! A world holds more chunks than memory does. Something has to decide which to
//! drop, and "the one nobody has touched for longest" is the decision that
//! survives a player wandering back the way they came, where "the oldest" would
//! drop exactly what they are about to want again.
//!
//! [`RingBuffer`](crate::structures::collections::RingBuffer) evicts too, but by
//! age and without keys: it is a history, this is a cache.

use crate::structures::hashing::FastHashMap;
use crate::structures::traits::{
    Bounded, ContentHashable, DeterministicMapOrder, Element, EvictingInsert, StableHash,
    stable_hash_unordered,
};
use crate::units::digest::ContentHash;
use std::collections::VecDeque;

/// A map of at most `capacity` entries, dropping the least recently used when
/// a new one does not fit.
///
/// Reading an entry counts as using it, which is the whole point and also the
/// reason [`LruCache::get`] takes `&mut self`: the order has to change for the
/// entry that was just wanted. [`LruCache::peek`] looks without touching the
/// order, for a caller that is inspecting rather than using.
///
/// Recency is kept as a queue of keys rather than a linked list, so a use is an
/// append and the queue is compacted when it grows past twice the capacity.
/// That keeps every operation constant on average, at the cost of holding some
/// stale keys between compactions.
#[derive(Clone, Debug)]
pub struct LruCache<K, V> {
    entries: FastHashMap<K, (V, u64)>,
    order: VecDeque<(K, u64)>,
    capacity: usize,
    clock: u64,
}

impl<K: Element, V> LruCache<K, V> {
    /// A cache holding at most `capacity` entries. A capacity of zero holds
    /// nothing and drops every insertion.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: FastHashMap::with_capacity_and_hasher(capacity, Default::default()),
            order: VecDeque::with_capacity(capacity * 2),
            capacity,
            clock: 0,
        }
    }

    /// How many entries the cache holds now.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The most entries it will ever hold.
    pub const fn limit(&self) -> usize {
        self.capacity
    }

    /// Whether it is holding as many as it can, so the next new key evicts one.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// The value for a key, counting as a use so that it becomes the most
    /// recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if !self.entries.contains_key(key) {
            return None;
        }

        self.touch(key);
        self.entries.get(key).map(|(value, _)| value)
    }

    /// The value for a key, leaving the recency order alone.
    ///
    /// For inspecting a cache without changing what it would evict: a debug
    /// view, a statistics pass, a check for whether something is resident.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|(value, _)| value)
    }

    /// Whether a key is held, without counting as a use.
    pub fn contains_key(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Stores a value, evicting the least recently used entry if the cache is
    /// full and the key is new.
    ///
    /// Returns what was evicted, which lets a caller write it back, and the old
    /// value when the key was already held.
    pub fn insert(&mut self, key: K, value: V) -> Evicted<K, V> {
        if self.capacity == 0 {
            return Evicted {
                replaced: None,
                dropped: Some((key, value)),
            };
        }

        if let Some((slot, _)) = self.entries.get_mut(&key) {
            let replaced: V = std::mem::replace(slot, value);

            self.touch(&key);

            return Evicted {
                replaced: Some(replaced),
                dropped: None,
            };
        }

        let dropped: Option<(K, V)> = self.is_full().then(|| self.evict()).flatten();

        self.clock += 1;
        self.entries.insert(key.clone(), (value, self.clock));
        self.order.push_back((key, self.clock));
        self.compact();

        Evicted {
            replaced: None,
            dropped,
        }
    }

    /// Removes a key, returning its value.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.entries.remove(key).map(|(value, _)| value)
    }

    /// Every key and value, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(key, (value, _))| (key, value))
    }

    /// The key that would be evicted next, or `None` when the cache is empty.
    pub fn next_eviction(&self) -> Option<&K> {
        self.order
            .iter()
            .find(|(key, used)| {
                self.entries
                    .get(key)
                    .is_some_and(|(_, current)| current == used)
            })
            .map(|(key, _)| key)
    }

    /// Drops every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.clock = 0;
    }

    /// Records a use of a key, which makes it the most recently used.
    fn touch(&mut self, key: &K) {
        self.clock += 1;

        if let Some((_, used)) = self.entries.get_mut(key) {
            *used = self.clock;
        }

        self.order.push_back((key.clone(), self.clock));
        self.compact();
    }

    /// Removes the least recently used entry and returns it.
    ///
    /// Walks the queue from the front, discarding the entries that have been
    /// used again since, until it finds one whose recorded use is still the
    /// current one.
    fn evict(&mut self) -> Option<(K, V)> {
        while let Some((key, used)) = self.order.pop_front() {
            let current: Option<u64> = self.entries.get(&key).map(|(_, used)| *used);

            if current == Some(used) {
                let (value, _) = self.entries.remove(&key)?;

                return Some((key, value));
            }
        }

        None
    }

    /// Drops the stale entries from the recency queue once it has grown to
    /// twice the number of live entries, which bounds the queue's length
    /// without making a use cost more than an append.
    fn compact(&mut self) {
        if self.order.len() <= self.entries.len().max(self.capacity) * 2 {
            return;
        }

        let live: FastHashMap<K, u64> = self
            .entries
            .iter()
            .map(|(key, (_, used))| (key.clone(), *used))
            .collect();

        self.order.retain(|(key, used)| live.get(key) == Some(used));
    }
}

/// What an insertion displaced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evicted<K, V> {
    /// The value the key held before, when the key was already present.
    pub replaced: Option<V>,
    /// The entry dropped to make room, when the cache was full.
    pub dropped: Option<(K, V)>,
}

impl<K, V> Evicted<K, V> {
    /// Whether the insertion cost another entry its place.
    pub const fn evicted_something(&self) -> bool {
        self.dropped.is_some()
    }
}

impl<K: Element, V> Bounded for LruCache<K, V> {
    fn bounded_len(&self) -> usize {
        self.len()
    }

    fn limit(&self) -> usize {
        self.capacity
    }
}

impl<K: Element, V> EvictingInsert for LruCache<K, V> {
    type Input = (K, V);
    type Output = Evicted<K, V>;

    fn insert_evicting(&mut self, (key, value): Self::Input) -> Self::Output {
        self.insert(key, value)
    }
}

impl<K, V> DeterministicMapOrder for LruCache<K, V> {}

/// The entries, order-independently, so two caches holding the same contents
/// agree however differently they were used.
impl<K: Element + StableHash, V: StableHash> ContentHashable for LruCache<K, V> {
    fn content_hash(&self) -> ContentHash {
        stable_hash_unordered(self.entries.iter().map(|(key, (value, _))| (key, value)))
    }
}

impl<K: Element, V> FromIterator<(K, V)> for LruCache<K, V> {
    /// A cache holding exactly the entries given, with its limit set to their
    /// number, so nothing is evicted on the way in.
    fn from_iter<I: IntoIterator<Item = (K, V)>>(entries: I) -> Self {
        let entries: Vec<(K, V)> = entries.into_iter().collect();
        let mut cache: Self = Self::new(entries.len().max(1));

        for (key, value) in entries {
            cache.insert(key, value);
        }

        cache
    }
}

impl<K: Element, V> Extend<(K, V)> for LruCache<K, V> {
    /// Inserts each entry, evicting as the limit requires.
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, entries: I) {
        for (key, value) in entries {
            self.insert(key, value);
        }
    }
}

/// As how full it is, such as `120/256 cached`.
impl<K: Element, V> std::fmt::Display for LruCache<K, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}/{} cached", self.entries.len(), self.capacity)
    }
}
