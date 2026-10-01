//! Internals shared by the one-to-many structures: set-valued indices whose
//! empty sets are dropped, and an index of groups by size.

use crate::structures::collections::sets::set::Set;
use crate::structures::hashing::{FastHashMap, FastHashSet};
use crate::structures::traits::Element;
use std::collections::BTreeMap;

/// Adds `value` to `key`'s set, creating the set if the key has none yet.
/// Returns whether the value was new to that set.
///
/// The key is cloned only when a set has to be created, which is why this takes
/// it by reference.
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

/// Removes `value` from `key`'s set, dropping the set itself once it is empty.
/// Returns whether the value was there.
///
/// Dropping empty sets is what keeps "the key is present" and "the key has
/// values" the same question for every one-to-many structure built on this.
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
///
/// A sorted map from size to the labels of the groups at that size. Sizes with
/// no groups left are dropped, so the first and last entries are always real
/// answers, and a group of zero members is not recorded at all.
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
    /// Forgets every group.
    pub fn clear(&mut self) {
        self.by_size.clear();
    }
}

impl<L: Element> SizeIndex<L> {
    /// Records that `label`'s group went from `old` to `new` members, moving it
    /// between the two size entries. A size of zero means the group does not
    /// exist, so growing from zero adds it and shrinking to zero drops it.
    ///
    /// The label is taken back out of the old entry and reused where possible,
    /// so a resize usually clones nothing.
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

    /// Removes a label from one size entry and hands back the owned label,
    /// dropping the entry once no group is left at that size.
    fn take(&mut self, label: &L, size: usize) -> Option<L> {
        let labels = self.by_size.get_mut(&size)?;
        let owned = labels.take(label);
        if labels.is_empty() {
            self.by_size.remove(&size);
        }
        owned
    }

    /// A label of one of the largest groups, or `None` when there are no
    /// groups. Which one, among equals, is not defined.
    pub fn largest(&self) -> Option<&L> {
        self.by_size.values().next_back()?.iter().next()
    }

    /// A label of one of the smallest non-empty groups, or `None` when there
    /// are no groups.
    pub fn smallest(&self) -> Option<&L> {
        self.by_size.values().next()?.iter().next()
    }

    /// Every label whose group has exactly this many members, and nothing at
    /// all for a size no group has.
    pub fn with_size(&self, size: usize) -> impl Iterator<Item = &L> {
        self.by_size.get(&size).into_iter().flatten()
    }
}
