//! One-to-many: a key owns a set of values, and each value has exactly one
//! owner. For ownership, exclusive assignment and slot occupancy.

use crate::misc::structures::buckets::{add_to_bucket, remove_from_bucket};
use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::traits::{
    Element, Map, MapMut, UniqueValueMap, ValueIndexed, ValueIndexedMut,
};
use std::collections::hash_map;

#[derive(Clone, Debug)]
/// One-to-many map from keys `K` to values `V`, where every value has exactly
/// one owner while assigned.
pub struct UniqueMultiMap<K, V> {
    forward: FastHashMap<K, Set<V>>,
    owners: FastHashMap<V, K>,
}

impl<K, V> Default for UniqueMultiMap<K, V> {
    fn default() -> Self {
        Self {
            forward: FastHashMap::default(),
            owners: FastHashMap::default(),
        }
    }
}

impl<K, V> UniqueMultiMap<K, V> {
    /// Creates an empty ownership map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    /// Number of values, which is also the number of pairs.
    pub fn value_count(&self) -> usize {
        self.owners.len()
    }

    /// Returns whether no keys own values.
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Iterates over keys that own at least one value.
    pub fn keys(&self) -> hash_map::Keys<'_, K, Set<V>> {
        self.forward.keys()
    }

    /// Iterates over all assigned values.
    pub fn values(&self) -> hash_map::Keys<'_, V, K> {
        self.owners.keys()
    }

    /// Each key with its value set.
    pub fn iter(&self) -> hash_map::Iter<'_, K, Set<V>> {
        self.forward.iter()
    }

    /// Each value with its owner.
    pub fn iter_by_value(&self) -> hash_map::Iter<'_, V, K> {
        self.owners.iter()
    }

    /// Every key-value pair.
    pub fn pairs(&self) -> impl Iterator<Item = (&K, &V)> {
        self.owners.iter().map(|(value, key)| (key, value))
    }

    /// Removes every assignment while retaining allocated storage.
    pub fn clear(&mut self) {
        self.forward.clear();
        self.owners.clear();
    }
}

impl<K: Element, V: Element> UniqueMultiMap<K, V> {
    /// Preallocates the key and owner indices. Individual value buckets grow
    /// only when values are assigned.
    pub fn with_capacities(keys: usize, values: usize) -> Self {
        Self {
            forward: FastHashMap::with_capacity_and_hasher(keys, Default::default()),
            owners: FastHashMap::with_capacity_and_hasher(values, Default::default()),
        }
    }

    /// Reserves the forward index for at least `additional` new keys.
    pub fn reserve_keys(&mut self, additional: usize) {
        self.forward.reserve(additional);
    }

    /// Reserves the owner index for at least `additional` new values.
    pub fn reserve_values(&mut self, additional: usize) {
        self.owners.reserve(additional);
    }

    /// Releases unused allocation in both top-level indices.
    pub fn shrink_to_fit(&mut self) {
        self.forward.shrink_to_fit();
        self.owners.shrink_to_fit();
    }

    /// Checks that every value has exactly one owner and both indices agree.
    pub fn check_invariants(&self) -> bool {
        self.forward.values().all(|values| !values.is_empty())
            && self.forward.values().map(Set::len).sum::<usize>() == self.owners.len()
            && self.forward.iter().all(|(key, values)| {
                values
                    .iter()
                    .all(|value| self.owners.get(value) == Some(key))
            })
    }

    /// Returns whether `key` owns at least one value.
    pub fn contains_key(&self, key: &K) -> bool {
        self.forward.contains_key(key)
    }

    /// Returns whether `value` currently has an owner.
    pub fn contains_value(&self, value: &V) -> bool {
        self.owners.contains_key(value)
    }

    /// Returns whether `key` is the owner of `value`.
    pub fn contains_pair(&self, key: &K, value: &V) -> bool {
        self.owners.get(value) == Some(key)
    }

    /// Returns all values owned by `key`.
    pub fn get(&self, key: &K) -> Option<&Set<V>> {
        self.forward.get(key)
    }

    /// The owner of `value`.
    pub fn key_of(&self, value: &V) -> Option<&K> {
        self.owners.get(value)
    }

    /// Assigns an unowned value to `key`. Fails if the value has an owner.
    pub fn insert(&mut self, key: K, value: V) -> bool {
        if self.owners.contains_key(&value) {
            return false;
        }
        add_to_bucket(&mut self.forward, &key, value.clone());
        self.owners.insert(value, key);
        true
    }

    /// Assigns every unowned value from `values` to `key`.
    pub fn insert_all(&mut self, key: K, values: impl IntoIterator<Item = V>) {
        for value in values {
            self.insert(key.clone(), value);
        }
    }

    /// Assigns `value` to `key`, taking it from its current owner. Returns
    /// the previous owner.
    pub fn insert_or_move(&mut self, key: K, value: V) -> Option<K> {
        if self.contains_pair(&key, &value) {
            return Some(key);
        }
        let previous = self.remove_value(&value);
        self.insert(key, value);
        previous
    }

    /// Removes exactly the assignment `(key, value)`.
    pub fn remove_pair(&mut self, key: &K, value: &V) -> bool {
        self.contains_pair(key, value) && self.remove_value(value).is_some()
    }

    /// Unassigns `value`, returning its owner.
    pub fn remove_value(&mut self, value: &V) -> Option<K> {
        let (value, key) = self.owners.remove_entry(value)?;
        remove_from_bucket(&mut self.forward, &key, &value);
        Some(key)
    }

    /// Removes a key and unassigns all its values.
    pub fn remove(&mut self, key: &K) -> Option<Set<V>> {
        let values = self.forward.remove(key)?;
        for value in &values {
            self.owners.remove(value);
        }
        Some(values)
    }

    /// Keeps only the pairs `keep` accepts.
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &V) -> bool) {
        let forward = &mut self.forward;
        self.owners.retain(|value, key| {
            let kept = keep(key, value);
            if !kept {
                remove_from_bucket(forward, key, value);
            }
            kept
        });
    }
}

impl<K: Element, V: Element> Map for UniqueMultiMap<K, V> {
    type Key = K;
    type Value = V;
    type Mapped = Set<V>;

    fn len(&self) -> usize {
        self.forward.len()
    }

    fn pair_count(&self) -> usize {
        self.owners.len()
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
}

/// `insert_pair` moves the value from its previous owner.
impl<K: Element, V: Element> MapMut for UniqueMultiMap<K, V> {
    fn insert_pair(&mut self, key: K, value: V) -> bool {
        if self.contains_pair(&key, &value) {
            return false;
        }
        self.insert_or_move(key, value);
        true
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

impl<K: Element, V: Element> ValueIndexed for UniqueMultiMap<K, V> {
    fn contains_value(&self, value: &V) -> bool {
        self.owners.contains_key(value)
    }

    fn value_count(&self) -> usize {
        self.owners.len()
    }

    fn values(&self) -> impl Iterator<Item = &V> {
        self.owners.keys()
    }
}

impl<K: Element, V: Element> ValueIndexedMut for UniqueMultiMap<K, V> {
    fn remove_value(&mut self, value: &V) -> bool {
        Self::remove_value(self, value).is_some()
    }
}

impl<K: Element, V: Element> UniqueValueMap for UniqueMultiMap<K, V> {
    fn key_of(&self, value: &V) -> Option<&K> {
        self.owners.get(value)
    }
}

impl<K: Element, V: Element> PartialEq for UniqueMultiMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.owners == other.owners
    }
}

impl<K: Element, V: Element> Eq for UniqueMultiMap<K, V> {}

impl<K: Element, V: Element> FromIterator<(K, V)> for UniqueMultiMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(pairs: I) -> Self {
        let mut map = Self::new();
        map.extend(pairs);
        map
    }
}

/// Later pairs move values away from earlier owners.
impl<K: Element, V: Element> Extend<(K, V)> for UniqueMultiMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, pairs: I) {
        for (key, value) in pairs {
            self.insert_or_move(key, value);
        }
    }
}

impl<K, V> IntoIterator for UniqueMultiMap<K, V> {
    type Item = (K, Set<V>);
    type IntoIter = hash_map::IntoIter<K, Set<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.forward.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a UniqueMultiMap<K, V> {
    type Item = (&'a K, &'a Set<V>);
    type IntoIter = hash_map::Iter<'a, K, Set<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.forward.iter()
    }
}
