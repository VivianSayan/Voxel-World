//! Traits for key -> value structures.
//!
//! | Trait            | Adds                                         |
//! |------------------|----------------------------------------------|
//! | `Map`            | lookup by key                                |
//! | `MapMut`         | adding and removing pairs                    |
//! | `ValueIndexed`   | lookup by value                              |
//! | `UniqueValueMap` | each value has one key (`key_of`)            |
//! | `SharedValueMap` | a value can have many keys (`keys_of`)       |

use crate::structures::collections::sets::set::Set;

/// Read-only capability shared by key-to-value relationship structures.
pub trait Map {
    /// Type used to look up mappings.
    type Key;
    /// Type on the value side of each individual pair.
    type Value;
    /// What a key looks up: the value itself, or a set of values.
    type Mapped;

    /// Number of keys.
    fn len(&self) -> usize;

    /// Returns `true` when the map contains no keys.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of key-value pairs.
    fn pair_count(&self) -> usize {
        self.len()
    }

    /// Returns whether `key` has at least one mapping.
    fn contains_key(&self, key: &Self::Key) -> bool;

    /// Returns the value or value-set associated with `key`.
    fn get(&self, key: &Self::Key) -> Option<&Self::Mapped>;

    /// Returns whether the exact `(key, value)` relationship exists.
    fn contains_pair(&self, key: &Self::Key, value: &Self::Value) -> bool;

    /// Iterates over all keys that currently have mappings.
    fn keys(&self) -> impl Iterator<Item = &Self::Key>;

    /// Every pair, once each.
    fn pairs(&self) -> impl Iterator<Item = (&Self::Key, &Self::Value)>;

    /// True when every pair here is also in `other`.
    fn is_submap(&self, other: &Self) -> bool {
        self.pair_count() <= other.pair_count()
            && self
                .pairs()
                .all(|(key, value)| other.contains_pair(key, value))
    }

    /// Returns whether every pair in `other` is also present in `self`.
    fn is_supermap(&self, other: &Self) -> bool {
        other.is_submap(self)
    }
}

/// Mutation operations shared by key-to-value relationship structures.
pub trait MapMut: Map {
    /// Makes sure the pair is present. Returns whether the map changed.
    /// Maps with one value per key (or per value) replace the pairs that
    /// clash with it.
    fn insert_pair(&mut self, key: Self::Key, value: Self::Value) -> bool;

    /// Removes a key and everything it maps to.
    fn remove(&mut self, key: &Self::Key) -> Option<Self::Mapped>;

    /// Removes only the exact `(key, value)` pair, returning whether it existed.
    fn remove_pair(&mut self, key: &Self::Key, value: &Self::Value) -> bool;

    /// Removes all mappings while retaining allocation where practical.
    fn clear(&mut self);

    /// The value for `key`, inserting what `default` returns first if the key
    /// is absent.
    ///
    /// Replaces the look-up, insert-on-miss, read-back dance that every caller
    /// otherwise writes by hand, and calls `default` only when the key is
    /// missing.
    ///
    /// The default implementation is written in terms of the methods above and
    /// so looks the key up twice; a map whose storage has an entry API of its
    /// own should override it and do the work once.
    fn get_or_insert_with<F: FnOnce() -> Self::Value>(
        &mut self,
        key: Self::Key,
        default: F,
    ) -> &Self::Mapped
    where
        Self::Key: Clone,
    {
        if !self.contains_key(&key) {
            self.insert_pair(key.clone(), default());
        }

        self.get(&key).expect("the key is present after insertion")
    }

    /// The value for `key`, inserting `default` first if the key is absent.
    ///
    /// [`get_or_insert_with`](MapMut::get_or_insert_with) where the value is
    /// already to hand and costs nothing to make.
    fn get_or_insert(&mut self, key: Self::Key, default: Self::Value) -> &Self::Mapped
    where
        Self::Key: Clone,
    {
        self.get_or_insert_with(key, || default)
    }

    /// Keeps only the pairs `keep` accepts.
    fn retain<F: FnMut(&Self::Key, &Self::Value) -> bool>(&mut self, keep: F);
}

/// Maps that index their values as well as their keys.
pub trait ValueIndexed: Map {
    /// Returns whether at least one key maps to `value`.
    fn contains_value(&self, value: &Self::Value) -> bool;

    /// Number of distinct values.
    fn value_count(&self) -> usize;

    /// Distinct values.
    fn values(&self) -> impl Iterator<Item = &Self::Value>;
}

/// Mutation operations for maps that maintain a reverse value index.
pub trait ValueIndexedMut: ValueIndexed + MapMut {
    /// Removes a value from every key holding it. Returns whether it was there.
    fn remove_value(&mut self, value: &Self::Value) -> bool;
}

/// Each value belongs to at most one key.
pub trait UniqueValueMap: ValueIndexed {
    /// Returns the sole key that owns `value`, or `None` when it is unassigned.
    fn key_of(&self, value: &Self::Value) -> Option<&Self::Key>;
}

/// A value can belong to many keys.
pub trait SharedValueMap: ValueIndexed {
    /// Returns all keys mapped to `value`, or `None` when it is unused.
    fn keys_of(&self, value: &Self::Value) -> Option<&Set<Self::Key>>;
}
