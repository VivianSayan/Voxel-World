//! Internals shared by the one-to-many structures: set-valued indices whose
//! empty sets are dropped, and an index of groups by size.

use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::hashing::{FastHashMap, FastHashSet};
use crate::misc::structures::traits::Element;
use std::collections::BTreeMap;

/// Adds `value` to `key`'s set, creating the set if needed. Returns whether
/// the value was new.
pub fn add_to_bucket<K: Element, V: Element>(
    map: &mut FastHashMap<K, Set<V>>,
    key: &K,
    value: V,
) -> bool {
    match map.get_mut(key) {
        Some(bucket) => bucket.insert(value),
        None => {
            map.insert(key.clone(), Set::from([value]));
            true
        }
    }
}

/// Removes `value` from `key`'s set, dropping the set once empty. Returns
/// whether the value was there.
pub fn remove_from_bucket<K: Element, V: Element>(
    map: &mut FastHashMap<K, Set<V>>,
    key: &K,
    value: &V,
) -> bool {
    let Some(bucket) = map.get_mut(key) else {
        return false;
    };
    if !bucket.remove(value) {
        return false;
    }
    if bucket.is_empty() {
        map.remove(key);
    }
    true
}

/// Labels indexed by the size of their group, so the largest and smallest
/// groups are found in O(log n) instead of re-sorting on every change.
#[derive(Clone, Debug)]
pub struct SizeIndex<L> {
    by_size: BTreeMap<usize, FastHashSet<L>>,
}

impl<L> Default for SizeIndex<L> {
    fn default() -> Self {
        Self {
            by_size: BTreeMap::new(),
        }
    }
}

impl<L> SizeIndex<L> {
    pub fn clear(&mut self) {
        self.by_size.clear();
    }
}

impl<L: Element> SizeIndex<L> {
    /// Records that `label`'s group went from `old` to `new` members. A
    /// size of zero means the group does not exist.
    pub fn resize(&mut self, label: &L, old: usize, new: usize) {
        if old == new {
            return;
        }
        let owned = if old > 0 { self.take(label, old) } else { None };
        if new > 0 {
            let owned = owned.unwrap_or_else(|| label.clone());
            self.by_size.entry(new).or_default().insert(owned);
        }
    }

    fn take(&mut self, label: &L, size: usize) -> Option<L> {
        let labels = self.by_size.get_mut(&size)?;
        let owned = labels.take(label);
        if labels.is_empty() {
            self.by_size.remove(&size);
        }
        owned
    }

    pub fn largest(&self) -> Option<&L> {
        self.by_size.values().next_back()?.iter().next()
    }

    pub fn smallest(&self) -> Option<&L> {
        self.by_size.values().next()?.iter().next()
    }

    pub fn with_size(&self, size: usize) -> impl Iterator<Item = &L> {
        self.by_size.get(&size).into_iter().flatten()
    }
}
