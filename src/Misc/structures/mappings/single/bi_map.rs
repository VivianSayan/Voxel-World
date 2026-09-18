//! A one-to-one pairing between left and right values, with O(1) lookup in
//! both directions. Each value belongs to at most one pair.
//!
//! Method names follow the `bimap` crate. Left and right remain distinct
//! domains even when they have the same Rust type; use `PairMap<T>` for
//! globally exclusive, unordered symmetric pairs.

use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::traits::{
    Element, Map, MapMut, UniqueValueMap, ValueIndexed, ValueIndexedMut,
};
use std::borrow::Borrow;
use std::collections::hash_map;
use std::hash::Hash;

/// The pairs `BiMap::insert` removed to make room.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Overwritten<L, R> {
    /// Nothing clashed.
    Neither,
    /// The left value was paired with another right value.
    Left(L, R),
    /// The right value was paired with another left value.
    Right(L, R),
    /// Exactly this pair already existed.
    Pair(L, R),
    /// Both values were in different pairs.
    Both((L, R), (L, R)),
}

impl<L, R> Overwritten<L, R> {
    /// Returns whether insertion displaced or replaced any existing pair.
    pub fn did_overwrite(&self) -> bool {
        !matches!(self, Self::Neither)
    }
}

#[derive(Clone, Debug)]
/// One-to-one bidirectional map between left values `L` and right values `R`.
///
/// Every value appears in at most one pair and can be looked up from either
/// side in expected O(1) time.
pub struct BiMap<L, R> {
    forward: FastHashMap<L, R>,
    backward: FastHashMap<R, L>,
}

impl<L, R> Default for BiMap<L, R> {
    fn default() -> Self {
        Self {
            forward: FastHashMap::default(),
            backward: FastHashMap::default(),
        }
    }
}

impl<L, R> BiMap<L, R> {
    /// Creates an empty bidirectional map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of pairs.
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    /// Returns whether no pairs are stored.
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Iterates over `(left, right)` pairs.
    pub fn iter(&self) -> hash_map::Iter<'_, L, R> {
        self.forward.iter()
    }

    /// Iterates over all left-side values.
    pub fn left_values(&self) -> hash_map::Keys<'_, L, R> {
        self.forward.keys()
    }

    /// Iterates over all right-side values.
    pub fn right_values(&self) -> hash_map::Keys<'_, R, L> {
        self.backward.keys()
    }

    /// Removes every pair while retaining allocated storage.
    pub fn clear(&mut self) {
        self.forward.clear();
        self.backward.clear();
    }
}

impl<L: Element, R: Element> BiMap<L, R> {
    /// Creates an empty map sized for at least `capacity` pairs in both indices.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            forward: FastHashMap::with_capacity_and_hasher(capacity, Default::default()),
            backward: FastHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Reserves both indices for at least `additional` more pairs.
    pub fn reserve(&mut self, additional: usize) {
        self.forward.reserve(additional);
        self.backward.reserve(additional);
    }

    /// Releases unused allocation from both directional indices.
    pub fn shrink_to_fit(&mut self) {
        self.forward.shrink_to_fit();
        self.backward.shrink_to_fit();
    }

    /// Checks that both directional indices contain exactly the same pairs.
    pub fn check_invariants(&self) -> bool {
        self.forward.len() == self.backward.len()
            && self
                .forward
                .iter()
                .all(|(left, right)| self.backward.get(right) == Some(left))
    }

    /// Returns whether a left value equivalent to borrowed `left` is paired.
    pub fn contains_left<Q: Hash + Eq + ?Sized>(&self, left: &Q) -> bool
    where
        L: Borrow<Q>,
    {
        self.forward.contains_key(left)
    }

    /// Returns whether a right value equivalent to borrowed `right` is paired.
    pub fn contains_right<Q: Hash + Eq + ?Sized>(&self, right: &Q) -> bool
    where
        R: Borrow<Q>,
    {
        self.backward.contains_key(right)
    }

    /// Returns whether `left` and `right` form an exact pair.
    pub fn contains_pair(&self, left: &L, right: &R) -> bool {
        self.forward.get(left) == Some(right)
    }

    /// The right value paired with `left`.
    pub fn get_by_left<Q: Hash + Eq + ?Sized>(&self, left: &Q) -> Option<&R>
    where
        L: Borrow<Q>,
    {
        self.forward.get(left)
    }

    /// The left value paired with `right`.
    pub fn get_by_right<Q: Hash + Eq + ?Sized>(&self, right: &Q) -> Option<&L>
    where
        R: Borrow<Q>,
    {
        self.backward.get(right)
    }

    /// Pairs two values, first breaking any pairs they were in.
    pub fn insert(&mut self, left: L, right: R) -> Overwritten<L, R> {
        let overwritten = if self.contains_pair(&left, &right) {
            let pair = self.remove_by_left(&left).unwrap();
            Overwritten::Pair(pair.0, pair.1)
        } else {
            match (self.remove_by_left(&left), self.remove_by_right(&right)) {
                (None, None) => Overwritten::Neither,
                (Some((l, r)), None) => Overwritten::Left(l, r),
                (None, Some((l, r))) => Overwritten::Right(l, r),
                (Some(by_left), Some(by_right)) => Overwritten::Both(by_left, by_right),
            }
        };
        self.forward.insert(left.clone(), right.clone());
        self.backward.insert(right, left);
        overwritten
    }

    /// Pairs two values only if both are free; otherwise hands them back.
    pub fn insert_no_overwrite(&mut self, left: L, right: R) -> Result<(), (L, R)> {
        if self.contains_left(&left) || self.contains_right(&right) {
            return Err((left, right));
        }
        self.forward.insert(left.clone(), right.clone());
        self.backward.insert(right, left);
        Ok(())
    }

    /// Removes by borrowed left value and returns the owned pair.
    pub fn remove_by_left<Q: Hash + Eq + ?Sized>(&mut self, left: &Q) -> Option<(L, R)>
    where
        L: Borrow<Q>,
    {
        let (left, right) = self.forward.remove_entry(left)?;
        self.backward.remove(&right);
        Some((left, right))
    }

    /// Removes by borrowed right value and returns the owned pair.
    pub fn remove_by_right<Q: Hash + Eq + ?Sized>(&mut self, right: &Q) -> Option<(L, R)>
    where
        R: Borrow<Q>,
    {
        let (right, left) = self.backward.remove_entry(right)?;
        self.forward.remove(&left);
        Some((left, right))
    }

    /// Removes exactly `(left, right)`, returning whether it existed.
    pub fn remove_pair(&mut self, left: &L, right: &R) -> bool {
        self.contains_pair(left, right) && self.remove_by_left(left).is_some()
    }

    /// Retains pairs for which `keep(left, right)` returns `true`.
    pub fn retain(&mut self, mut keep: impl FnMut(&L, &R) -> bool) {
        let backward = &mut self.backward;
        self.forward.retain(|left, right| {
            let kept = keep(left, right);
            if !kept {
                backward.remove(right);
            }
            kept
        });
    }
}

impl<L: Element, R: Element> Map for BiMap<L, R> {
    type Key = L;
    type Value = R;
    type Mapped = R;

    fn len(&self) -> usize {
        self.forward.len()
    }

    fn contains_key(&self, left: &L) -> bool {
        self.forward.contains_key(left)
    }

    fn get(&self, left: &L) -> Option<&R> {
        self.forward.get(left)
    }

    fn contains_pair(&self, left: &L, right: &R) -> bool {
        Self::contains_pair(self, left, right)
    }

    fn keys(&self) -> impl Iterator<Item = &L> {
        self.forward.keys()
    }

    fn pairs(&self) -> impl Iterator<Item = (&L, &R)> {
        self.forward.iter()
    }
}

/// `insert_pair` breaks any pairs that clash with the new one.
impl<L: Element, R: Element> MapMut for BiMap<L, R> {
    fn insert_pair(&mut self, left: L, right: R) -> bool {
        !matches!(self.insert(left, right), Overwritten::Pair(..))
    }

    fn remove(&mut self, left: &L) -> Option<R> {
        self.remove_by_left(left).map(|(_, right)| right)
    }

    fn remove_pair(&mut self, left: &L, right: &R) -> bool {
        Self::remove_pair(self, left, right)
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&L, &R) -> bool>(&mut self, keep: F) {
        Self::retain(self, keep)
    }
}

impl<L: Element, R: Element> ValueIndexed for BiMap<L, R> {
    fn contains_value(&self, right: &R) -> bool {
        self.backward.contains_key(right)
    }

    fn value_count(&self) -> usize {
        self.backward.len()
    }

    fn values(&self) -> impl Iterator<Item = &R> {
        self.backward.keys()
    }
}

impl<L: Element, R: Element> ValueIndexedMut for BiMap<L, R> {
    fn remove_value(&mut self, right: &R) -> bool {
        self.remove_by_right(right).is_some()
    }
}

impl<L: Element, R: Element> UniqueValueMap for BiMap<L, R> {
    fn key_of(&self, right: &R) -> Option<&L> {
        self.backward.get(right)
    }
}

impl<L: Element, R: Element> PartialEq for BiMap<L, R> {
    fn eq(&self, other: &Self) -> bool {
        self.forward == other.forward
    }
}

impl<L: Element, R: Element> Eq for BiMap<L, R> {}

/// Later pairs overwrite clashing earlier ones.
impl<L: Element, R: Element> FromIterator<(L, R)> for BiMap<L, R> {
    fn from_iter<I: IntoIterator<Item = (L, R)>>(pairs: I) -> Self {
        let mut map = Self::new();
        map.extend(pairs);
        map
    }
}

impl<L: Element, R: Element> Extend<(L, R)> for BiMap<L, R> {
    fn extend<I: IntoIterator<Item = (L, R)>>(&mut self, pairs: I) {
        for (left, right) in pairs {
            self.insert(left, right);
        }
    }
}

impl<L, R> IntoIterator for BiMap<L, R> {
    type Item = (L, R);
    type IntoIter = hash_map::IntoIter<L, R>;

    fn into_iter(self) -> Self::IntoIter {
        self.forward.into_iter()
    }
}

impl<'a, L, R> IntoIterator for &'a BiMap<L, R> {
    type Item = (&'a L, &'a R);
    type IntoIter = hash_map::Iter<'a, L, R>;

    fn into_iter(self) -> Self::IntoIter {
        self.forward.iter()
    }
}
