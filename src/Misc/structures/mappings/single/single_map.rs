//! Each key maps to exactly one value. A thin wrapper over `HashMap` with
//! the same API (including `entry`), the crate's fast hasher, and the
//! `Map` traits.
//!
//! The Godot `SetKeyMap` needed string keys to compare structures by content.
//! Rust sets can be hashed by content, so it is represented by the alias
//! below. Tuples already work directly as `SingleMap` keys and need no alias.

use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::traits::{Key, Map, MapMut};
use std::borrow::Borrow;
use std::collections::hash_map;
use std::hash::Hash;

/// A map keyed by unordered combinations: `{a, b}` and `{b, a}` are the
/// same key.
pub type SetKeyMap<K, V> = SingleMap<Set<K>, V>;

#[derive(Clone, Debug)]
/// Deterministic hash map from keys of type `K` to values of type `V`.
///
/// Every key has at most one value. `K` only needs [`Key`], so keys do not
/// need to be cloneable.
pub struct SingleMap<K, V> {
    map: FastHashMap<K, V>,
}

impl<K, V> Default for SingleMap<K, V> {
    fn default() -> Self {
        Self {
            map: FastHashMap::default(),
        }
    }
}

impl<K, V> SingleMap<K, V> {
    /// Creates an empty map without allocating.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of key-value pairs.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns whether the map has no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterates over borrowed key-value pairs.
    pub fn iter(&self) -> hash_map::Iter<'_, K, V> {
        self.map.iter()
    }

    /// Iterates over keys and mutable values.
    pub fn iter_mut(&mut self) -> hash_map::IterMut<'_, K, V> {
        self.map.iter_mut()
    }

    /// Iterates over keys.
    pub fn keys(&self) -> hash_map::Keys<'_, K, V> {
        self.map.keys()
    }

    /// Iterates over values.
    pub fn values(&self) -> hash_map::Values<'_, K, V> {
        self.map.values()
    }

    /// Iterates over mutable values.
    pub fn values_mut(&mut self) -> hash_map::ValuesMut<'_, K, V> {
        self.map.values_mut()
    }

    /// Removes every entry while retaining allocated storage.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Retains entries for which `keep(key, value)` returns `true`.
    pub fn retain(&mut self, keep: impl FnMut(&K, &mut V) -> bool) {
        self.map.retain(keep);
    }

    /// Borrows the underlying deterministic hash map.
    pub fn as_hash_map(&self) -> &FastHashMap<K, V> {
        &self.map
    }

    /// Consumes this wrapper and returns its underlying hash map.
    pub fn into_hash_map(self) -> FastHashMap<K, V> {
        self.map
    }
}

impl<K: Key, V> SingleMap<K, V> {
    /// Creates an empty map sized for at least `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: FastHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Reserves capacity for at least `additional` more entries.
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional);
    }

    /// Releases unused allocation as far as the allocator permits.
    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit();
    }

    /// Returns whether a key equivalent to borrowed `key` exists.
    ///
    /// `Q` is a borrowed lookup form of `K`, such as `str` for `String`.
    pub fn contains_key<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.map.contains_key(key)
    }

    /// Returns the value associated with borrowed `key`.
    pub fn get<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.map.get(key)
    }

    /// Returns mutable access to the value associated with borrowed `key`.
    pub fn get_mut<Q: Hash + Eq + ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        self.map.get_mut(key)
    }

    /// Returns the stored key and value equivalent to borrowed `key`.
    pub fn get_key_value<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
    {
        self.map.get_key_value(key)
    }

    /// Returns whether borrowed `key` maps to exactly `value`.
    pub fn contains_pair<Q: Hash + Eq + ?Sized>(&self, key: &Q, value: &V) -> bool
    where
        K: Borrow<Q>,
        V: PartialEq,
    {
        self.map.get(key) == Some(value)
    }

    /// Sets the value, returning the one it replaced.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.map.insert(key, value)
    }

    /// Adds the mapping only if the key is new. Returns whether it was added.
    pub fn try_insert(&mut self, key: K, value: V) -> bool {
        match self.map.entry(key) {
            hash_map::Entry::Occupied(_) => false,
            hash_map::Entry::Vacant(slot) => {
                slot.insert(value);
                true
            }
        }
    }

    /// Returns the entry API for inserting or modifying `key` in place.
    pub fn entry(&mut self, key: K) -> hash_map::Entry<'_, K, V> {
        self.map.entry(key)
    }

    /// Removes borrowed `key` and returns its value.
    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        self.map.remove(key)
    }

    /// Removes borrowed `key` and returns both the stored key and value.
    pub fn remove_entry<Q: Hash + Eq + ?Sized>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
    {
        self.map.remove_entry(key)
    }

    /// Removes `key` only if it maps to `value`.
    pub fn remove_pair<Q: Hash + Eq + ?Sized>(&mut self, key: &Q, value: &V) -> bool
    where
        K: Borrow<Q>,
        V: PartialEq,
    {
        self.contains_pair(key, value) && self.map.remove(key).is_some()
    }
}

impl<K: Key, V: PartialEq> Map for SingleMap<K, V> {
    type Key = K;
    type Value = V;
    type Mapped = V;

    fn len(&self) -> usize {
        self.map.len()
    }

    fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    fn contains_pair(&self, key: &K, value: &V) -> bool {
        self.map.get(key) == Some(value)
    }

    fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }

    fn pairs(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter()
    }
}

/// `insert_pair` replaces the key's old value.
impl<K: Key, V: PartialEq> MapMut for SingleMap<K, V> {
    fn insert_pair(&mut self, key: K, value: V) -> bool {
        if self.map.get(&key) == Some(&value) {
            return false;
        }
        self.map.insert(key, value);
        true
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        self.map.remove(key)
    }

    fn remove_pair(&mut self, key: &K, value: &V) -> bool {
        Self::remove_pair(self, key, value)
    }

    fn clear(&mut self) {
        self.map.clear();
    }

    fn retain<F: FnMut(&K, &V) -> bool>(&mut self, mut keep: F) {
        self.map.retain(|key, value| keep(key, value));
    }
}

impl<K: Key, V: PartialEq> PartialEq for SingleMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
    }
}

impl<K: Key, V: Eq> Eq for SingleMap<K, V> {}

impl<K: Key + Borrow<Q>, Q: Hash + Eq + ?Sized, V> std::ops::Index<&Q> for SingleMap<K, V> {
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        &self.map[key]
    }
}

/// Later pairs overwrite earlier ones with the same key, as in `HashMap`.
impl<K: Key, V> FromIterator<(K, V)> for SingleMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(pairs: I) -> Self {
        Self {
            map: pairs.into_iter().collect(),
        }
    }
}

impl<K: Key, V, const N: usize> From<[(K, V); N]> for SingleMap<K, V> {
    fn from(pairs: [(K, V); N]) -> Self {
        pairs.into_iter().collect()
    }
}

impl<K: Key, V> Extend<(K, V)> for SingleMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, pairs: I) {
        self.map.extend(pairs);
    }
}

impl<K, V> IntoIterator for SingleMap<K, V> {
    type Item = (K, V);
    type IntoIter = hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a SingleMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = hash_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut SingleMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = hash_map::IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.iter_mut()
    }
}
