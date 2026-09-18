//! Many-to-many: a key maps to a set of values, and a value can belong to
//! many keys. Both directions are indexed, for ownership tables, adjacency,
//! subscription lists and category membership.

use crate::misc::structures::buckets::{add_to_bucket, remove_from_bucket};
use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::traits::{
    Element, Map, MapMut, SetAlgebra, SharedValueMap, ValueIndexed, ValueIndexedMut,
};
use std::collections::hash_map;

#[derive(Clone, Debug)]
/// Many-to-many bidirectional mapping between keys `K` and values `V`.
///
/// Both key-to-values and value-to-keys lookups are expected O(1).
pub struct MultiMap<K, V> {
    forward: FastHashMap<K, Set<V>>,
    backward: FastHashMap<V, Set<K>>,
    pair_count: usize,
}

impl<K, V> Default for MultiMap<K, V> {
    fn default() -> Self {
        Self {
            forward: FastHashMap::default(),
            backward: FastHashMap::default(),
            pair_count: 0,
        }
    }
}

impl<K, V> MultiMap<K, V> {
    /// Creates an empty many-to-many map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    /// Number of distinct values.
    pub fn value_count(&self) -> usize {
        self.backward.len()
    }

    /// Number of key-value pairs.
    pub fn pair_count(&self) -> usize {
        self.pair_count
    }

    /// Returns whether no keys have values.
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Iterates over keys that have at least one value.
    pub fn keys(&self) -> hash_map::Keys<'_, K, Set<V>> {
        self.forward.keys()
    }

    /// Distinct values.
    pub fn values(&self) -> hash_map::Keys<'_, V, Set<K>> {
        self.backward.keys()
    }

    /// Each key with its value set.
    pub fn iter(&self) -> hash_map::Iter<'_, K, Set<V>> {
        self.forward.iter()
    }

    /// Each value with the keys holding it.
    pub fn iter_by_value(&self) -> hash_map::Iter<'_, V, Set<K>> {
        self.backward.iter()
    }

    /// Every key-value pair.
    pub fn pairs(&self) -> impl Iterator<Item = (&K, &V)> {
        self.forward
            .iter()
            .flat_map(|(key, values)| values.iter().map(move |value| (key, value)))
    }

    /// Removes every relationship while retaining allocated storage.
    pub fn clear(&mut self) {
        self.forward.clear();
        self.backward.clear();
        self.pair_count = 0;
    }
}

impl<K: Element, V: Element> MultiMap<K, V> {
    /// Preallocates the two top-level indices. Individual value/key buckets
    /// still grow only when their first pair is inserted.
    pub fn with_capacities(keys: usize, values: usize) -> Self {
        Self {
            forward: FastHashMap::with_capacity_and_hasher(keys, Default::default()),
            backward: FastHashMap::with_capacity_and_hasher(values, Default::default()),
            pair_count: 0,
        }
    }

    /// Reserves the forward index for at least `additional` new keys.
    pub fn reserve_keys(&mut self, additional: usize) {
        self.forward.reserve(additional);
    }

    /// Reserves the reverse index for at least `additional` new values.
    pub fn reserve_values(&mut self, additional: usize) {
        self.backward.reserve(additional);
    }

    /// Releases unused allocation in both top-level indices.
    pub fn shrink_to_fit(&mut self) {
        self.forward.shrink_to_fit();
        self.backward.shrink_to_fit();
    }

    /// Checks pair counts, non-empty buckets, and agreement between the two
    /// directional indices.
    pub fn check_invariants(&self) -> bool {
        self.forward.values().all(|values| !values.is_empty())
            && self.backward.values().all(|keys| !keys.is_empty())
            && self.forward.values().map(Set::len).sum::<usize>() == self.pair_count
            && self.backward.values().map(Set::len).sum::<usize>() == self.pair_count
            && self.forward.iter().all(|(key, values)| {
                values.iter().all(|value| {
                    self.backward
                        .get(value)
                        .is_some_and(|keys| keys.contains(key))
                })
            })
    }

    /// Returns whether `key` maps to at least one value.
    pub fn contains_key(&self, key: &K) -> bool {
        self.forward.contains_key(key)
    }

    /// Returns whether at least one key maps to `value`.
    pub fn contains_value(&self, value: &V) -> bool {
        self.backward.contains_key(value)
    }

    /// Returns whether the exact `(key, value)` relationship exists.
    pub fn contains_pair(&self, key: &K, value: &V) -> bool {
        self.forward
            .get(key)
            .is_some_and(|values| values.contains(value))
    }

    /// The values of `key`.
    pub fn get(&self, key: &K) -> Option<&Set<V>> {
        self.forward.get(key)
    }

    /// The keys holding `value`.
    pub fn keys_of(&self, value: &V) -> Option<&Set<K>> {
        self.backward.get(value)
    }

    /// Adds a pair. Returns whether it was new.
    pub fn insert(&mut self, key: K, value: V) -> bool {
        if self.contains_pair(&key, &value) {
            return false;
        }
        add_to_bucket(&mut self.backward, &value, key.clone());
        add_to_bucket(&mut self.forward, &key, value);
        self.pair_count += 1;
        true
    }

    /// Adds every value in `values` to `key`.
    pub fn insert_all(&mut self, key: K, values: impl IntoIterator<Item = V>) {
        for value in values {
            self.insert(key.clone(), value);
        }
    }

    /// Removes exactly `(key, value)`, returning whether it existed.
    pub fn remove_pair(&mut self, key: &K, value: &V) -> bool {
        if !remove_from_bucket(&mut self.forward, key, value) {
            return false;
        }
        remove_from_bucket(&mut self.backward, value, key);
        self.pair_count -= 1;
        true
    }

    /// Removes a key with all its values.
    pub fn remove(&mut self, key: &K) -> Option<Set<V>> {
        let (key, values) = self.forward.remove_entry(key)?;
        for value in &values {
            remove_from_bucket(&mut self.backward, value, &key);
        }
        self.pair_count -= values.len();
        Some(values)
    }

    /// Removes a value from every key holding it.
    pub fn remove_value(&mut self, value: &V) -> Option<Set<K>> {
        let (value, keys) = self.backward.remove_entry(value)?;
        for key in &keys {
            remove_from_bucket(&mut self.forward, key, &value);
        }
        self.pair_count -= keys.len();
        Some(keys)
    }

    /// Keeps only the pairs `keep` accepts.
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &V) -> bool) {
        let removed: Vec<(K, V)> = self
            .pairs()
            .filter(|(key, value)| !keep(key, value))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        for (key, value) in &removed {
            self.remove_pair(key, value);
        }
    }
}

impl<K: Element, V: Element> Map for MultiMap<K, V> {
    type Key = K;
    type Value = V;
    type Mapped = Set<V>;

    fn len(&self) -> usize {
        self.forward.len()
    }

    fn pair_count(&self) -> usize {
        self.pair_count
    }

    fn contains_key(&self, key: &K) -> bool {
        Self::contains_key(self, key)
    }

    fn get(&self, key: &K) -> Option<&Set<V>> {
        Self::get(self, key)
    }

    fn contains_pair(&self, key: &K, value: &V) -> bool {
        Self::contains_pair(self, key, value)
    }

    fn keys(&self) -> impl Iterator<Item = &K> {
        self.forward.keys()
    }

    fn pairs(&self) -> impl Iterator<Item = (&K, &V)> {
        Self::pairs(self)
    }

    fn is_submap(&self, other: &Self) -> bool {
        self.pair_count <= other.pair_count
            && self.iter().all(|(key, values)| {
                other
                    .get(key)
                    .is_some_and(|theirs| values.is_subset(theirs))
            })
    }
}

impl<K: Element, V: Element> MapMut for MultiMap<K, V> {
    fn insert_pair(&mut self, key: K, value: V) -> bool {
        self.insert(key, value)
    }

    fn remove(&mut self, key: &K) -> Option<Set<V>> {
        Self::remove(self, key)
    }

    fn remove_pair(&mut self, key: &K, value: &V) -> bool {
        Self::remove_pair(self, key, value)
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&K, &V) -> bool>(&mut self, keep: F) {
        Self::retain(self, keep)
    }
}

impl<K: Element, V: Element> ValueIndexed for MultiMap<K, V> {
    fn contains_value(&self, value: &V) -> bool {
        Self::contains_value(self, value)
    }

    fn value_count(&self) -> usize {
        self.backward.len()
    }

    fn values(&self) -> impl Iterator<Item = &V> {
        self.backward.keys()
    }
}

impl<K: Element, V: Element> ValueIndexedMut for MultiMap<K, V> {
    fn remove_value(&mut self, value: &V) -> bool {
        Self::remove_value(self, value).is_some()
    }
}

impl<K: Element, V: Element> SharedValueMap for MultiMap<K, V> {
    fn keys_of(&self, value: &V) -> Option<&Set<K>> {
        self.backward.get(value)
    }
}

impl<K: Element, V: Element> PartialEq for MultiMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.pair_count == other.pair_count && self.forward == other.forward
    }
}

impl<K: Element, V: Element> Eq for MultiMap<K, V> {}

impl<K: Element, V: Element> FromIterator<(K, V)> for MultiMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(pairs: I) -> Self {
        let mut map = Self::new();
        map.extend(pairs);
        map
    }
}

impl<K: Element, V: Element> Extend<(K, V)> for MultiMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, pairs: I) {
        for (key, value) in pairs {
            self.insert(key, value);
        }
    }
}

impl<K, V> IntoIterator for MultiMap<K, V> {
    type Item = (K, Set<V>);
    type IntoIter = hash_map::IntoIter<K, Set<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.forward.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a MultiMap<K, V> {
    type Item = (&'a K, &'a Set<V>);
    type IntoIter = hash_map::Iter<'a, K, Set<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.forward.iter()
    }
}
