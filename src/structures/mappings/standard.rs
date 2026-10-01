//! The mapping traits on the standard hash map, including `FastHashMap`.

use crate::structures::traits::{Map, MapMut};
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

impl<K, V, S> Map for HashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    type Key = K;
    type Value = V;
    type Mapped = V;

    fn len(&self) -> usize {
        HashMap::len(self)
    }

    fn contains_key(&self, key: &K) -> bool {
        HashMap::contains_key(self, key)
    }

    fn get(&self, key: &K) -> Option<&V> {
        HashMap::get(self, key)
    }

    fn contains_pair(&self, key: &K, value: &V) -> bool {
        self.get(key) == Some(value)
    }

    fn keys(&self) -> impl Iterator<Item = &K> {
        HashMap::keys(self)
    }

    fn pairs(&self) -> impl Iterator<Item = (&K, &V)> {
        self.iter()
    }
}

impl<K, V, S> MapMut for HashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn insert_pair(&mut self, key: K, value: V) -> bool {
        if self.get(&key) == Some(&value) {
            return false;
        }
        self.insert(key, value);
        true
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        HashMap::remove(self, key)
    }

    fn remove_pair(&mut self, key: &K, value: &V) -> bool {
        if self.get(key) != Some(value) {
            return false;
        }
        self.remove(key).is_some()
    }

    fn clear(&mut self) {
        HashMap::clear(self);
    }

    fn get_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, default: F) -> &V
    where
        K: Clone,
    {
        self.entry(key).or_insert_with(default)
    }

    fn retain<F: FnMut(&K, &V) -> bool>(&mut self, mut keep: F) {
        HashMap::retain(self, |key, value| keep(key, value));
    }
}
