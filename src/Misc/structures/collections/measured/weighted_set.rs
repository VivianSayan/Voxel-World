//! Elements with continuous weights: relative strength, importance or
//! likelihood. Like `MultiSet`, but the amount is a real number.
//!
//! Set algebra keeps the larger weight (union) or the smaller
//! (intersection); `+` adds weights.

use crate::misc::random::Random;
use crate::misc::structures::collections::measured::multi_set::MultiSet;
use crate::misc::structures::collections::measured::tally::Tally;
use crate::misc::structures::traits::operators::impl_set_operators;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, Measured, MeasuredMut,
    SetAlgebra, UniqueCollection,
};
use std::borrow::Borrow;
use std::hash::Hash;

fn assert_valid_weight(weight: f64) {
    assert!(
        weight.is_finite() && weight >= 0.0,
        "weights must be finite and non-negative"
    );
}

#[derive(Clone, Debug)]
/// Unordered elements of type `T` associated with finite, non-negative
/// floating-point weights.
pub struct WeightedSet<T> {
    tally: Tally<T, f64>,
}

impl<T> Default for WeightedSet<T> {
    fn default() -> Self {
        Self {
            tally: Tally::default(),
        }
    }
}

impl<T> WeightedSet<T> {
    /// Creates an empty weighted set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of distinct weighted elements.
    pub fn len(&self) -> usize {
        self.tally.len()
    }

    /// Returns whether no elements have positive weight.
    pub fn is_empty(&self) -> bool {
        self.tally.is_empty()
    }

    /// Sum of all weights.
    pub fn total_weight(&self) -> f64 {
        self.tally.total()
    }

    /// Each element with its stored value.
    pub fn iter(&self) -> impl Iterator<Item = (&T, f64)> {
        self.tally.iter()
    }

    /// Iterates over each distinct element once.
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, T, f64> {
        self.tally.keys()
    }

    /// Removes every element and weight while retaining allocated storage.
    pub fn clear(&mut self) {
        self.tally.clear();
    }

    fn empty_like(&self) -> Self {
        Self::new()
    }
}

impl<T: Element> WeightedSet<T> {
    /// Returns whether a value equivalent to borrowed `item` has positive weight.
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.tally.contains(item)
    }

    /// The element's weight, or zero when absent.
    pub fn weight_of<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> f64
    where
        T: Borrow<Q>,
    {
        self.tally.get(item).unwrap_or(0.0)
    }

    /// Alias retained for code that wants to emphasize linear weights.
    pub fn linear_weight_of<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> f64
    where
        T: Borrow<Q>,
    {
        self.weight_of(item)
    }

    /// Adds a weight of one.
    pub fn insert(&mut self, item: T) {
        self.add_weight(item, 1.0);
    }

    /// Adds a finite, non-negative weight.
    pub fn add_weight(&mut self, item: T, value: f64) {
        assert_valid_weight(value);
        if value == 0.0 {
            return;
        }
        let current = self.weight_of(&item);
        let combined = current + value;
        assert!(
            combined.is_finite(),
            "the combined weight must remain finite"
        );
        self.tally.set(item, combined);
    }

    /// Alias for `add_weight`.
    pub fn add_linear_weight(&mut self, item: T, weight: f64) {
        self.add_weight(item, weight);
    }

    /// Sets a finite, non-negative weight outright. Zero removes the item.
    pub fn set_weight(&mut self, item: T, value: f64) {
        assert_valid_weight(value);
        if value == 0.0 {
            self.tally.remove(&item);
        } else {
            self.tally.set(item, value);
        }
    }

    /// Subtracts from the stored value. In linear mode the weight cannot go
    /// below zero, and an element whose weight reaches zero is removed.
    pub fn subtract_weight<Q: Hash + Eq + ?Sized>(&mut self, item: &Q, value: f64)
    where
        T: Borrow<Q>,
    {
        let Some(current) = self.tally.get(item) else {
            return;
        };
        assert_valid_weight(value);
        let remaining = current - value.min(current);
        self.tally
            .update(item, (remaining > 0.0).then_some(remaining));
    }

    /// Removes the element whatever its weight, returning the stored value.
    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> Option<f64>
    where
        T: Borrow<Q>,
    {
        self.tally.remove(item)
    }

    /// Retains entries for which `keep(item, weight)` returns `true`.
    pub fn retain(&mut self, keep: impl FnMut(&T, f64) -> bool) {
        self.tally.retain(keep);
    }

    /// Adds the weights of both sets.
    pub fn sum(&self, other: &Self) -> Self {
        let mut output = self.clone();
        for (item, value) in other.iter() {
            output.add_weight(item.clone(), value);
        }
        output
    }

    /// Rounds each linear weight to a count.
    pub fn to_multi_set(&self) -> MultiSet<T> {
        self.iter()
            .map(|(item, _)| {
                (
                    item.clone(),
                    self.linear_weight_of(item).round().max(0.0) as usize,
                )
            })
            .collect()
    }

    fn combined(
        &self,
        other: &Self,
        keys: &Self,
        combine: impl Fn(Option<f64>, Option<f64>) -> Option<f64>,
    ) -> Self {
        let mut output = self.empty_like();
        for (item, _) in keys.iter() {
            if let Some(value) = combine(self.tally.get(item), other.tally.get(item)) {
                output.set_weight(item.clone(), value);
            }
        }
        output
    }
}

impl<T: Element> Collection for WeightedSet<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.tally.len()
    }

    fn contains(&self, item: &T) -> bool {
        self.tally.contains(item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.tally.keys()
    }
}

impl<T: Element> CollectionInsert for WeightedSet<T> {
    /// Adds a weight of one; always changes the set.
    fn insert(&mut self, item: T) -> bool {
        Self::insert(self, item);
        true
    }
}

impl<T: Element> CollectionRemove for WeightedSet<T> {
    fn remove(&mut self, item: &T) -> bool {
        Self::remove(self, item).is_some()
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, mut keep: F) {
        self.tally.retain(|item, _| keep(item));
    }
}

impl<T: Element> UniqueCollection for WeightedSet<T> {}

impl<T: Element> Measured for WeightedSet<T> {
    type Measure = f64;

    fn measure_of(&self, item: &T) -> f64 {
        self.weight_of(item)
    }

    fn total_measure(&self) -> f64 {
        self.tally.total()
    }

    fn distinct_len(&self) -> usize {
        self.tally.len()
    }

    fn measures(&self) -> impl Iterator<Item = (&T, f64)> {
        self.tally.iter()
    }
}

impl<T: Element> MeasuredMut for WeightedSet<T> {
    fn set_measure(&mut self, item: T, value: f64) {
        self.set_weight(item, value);
    }

    fn add_measure(&mut self, item: T, value: f64) {
        self.add_weight(item, value);
    }

    fn subtract_measure(&mut self, item: &T, value: f64) {
        self.subtract_weight(item, value);
    }
}

/// Picks elements in proportion to their linear weight. Non-positive
/// weights are never picked.
impl<T: Element> Choose for WeightedSet<T> {
    fn choose_multiple(&self, random: &mut Random, amount: usize) -> Vec<&T> {
        self.tally
            .choose_weighted(random, amount, |value| value, |_| {})
    }
}

impl<T: Element> SetAlgebra for WeightedSet<T> {
    /// The larger weight of each element.
    fn union(&self, other: &Self) -> Self {
        let mut output = self.clone();
        for (item, value) in other.iter() {
            if self.tally.get(item).is_none_or(|mine| value > mine) {
                output.set_weight(item.clone(), value);
            }
        }
        output
    }

    /// The smaller weight of each element found in both.
    fn intersection(&self, other: &Self) -> Self {
        self.combined(other, self, |mine, theirs| Some(mine?.min(theirs?)))
    }

    /// Weights minus `other`'s, dropping elements that would go negative.
    fn difference(&self, other: &Self) -> Self {
        self.combined(other, self, |mine, theirs| {
            let remaining = mine? - theirs.unwrap_or(0.0);
            (remaining >= 0.0).then_some(remaining)
        })
    }

    fn is_subset(&self, other: &Self) -> bool {
        self.len() <= other.len()
            && self
                .iter()
                .all(|(item, value)| other.tally.get(item).is_some_and(|theirs| value <= theirs))
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        !self.keys().any(|item| other.contains(item))
    }
}

impl_set_operators!([T: Element] WeightedSet<T>);

impl<T: Element> std::ops::Add<&WeightedSet<T>> for &WeightedSet<T> {
    type Output = WeightedSet<T>;

    fn add(self, other: &WeightedSet<T>) -> WeightedSet<T> {
        self.sum(other)
    }
}

impl<T: Element> PartialEq for WeightedSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.tally.map() == other.tally.map()
    }
}

/// A linear set where every element weighs one per occurrence.
impl<T: Element> FromIterator<T> for WeightedSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        let mut set = Self::new();
        for item in items {
            set.insert(item);
        }
        set
    }
}

/// A linear set from `(element, weight)` pairs; repeats add up.
impl<T: Element> FromIterator<(T, f64)> for WeightedSet<T> {
    fn from_iter<I: IntoIterator<Item = (T, f64)>>(pairs: I) -> Self {
        let mut set = Self::new();
        set.extend(pairs);
        set
    }
}

/// Adds each linear weight.
impl<T: Element> Extend<(T, f64)> for WeightedSet<T> {
    fn extend<I: IntoIterator<Item = (T, f64)>>(&mut self, pairs: I) {
        for (item, value) in pairs {
            self.add_weight(item, value);
        }
    }
}

impl<'a, T> IntoIterator for &'a WeightedSet<T> {
    type Item = (&'a T, &'a f64);
    type IntoIter = std::collections::hash_map::Iter<'a, T, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.tally.map().iter()
    }
}
