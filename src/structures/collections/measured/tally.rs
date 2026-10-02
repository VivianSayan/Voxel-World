//! The storage behind [`MultiSet`], [`WeightedSet`] and [`FuzzySet`]: a map
//! from element to a number, with a running total.
//!
//! The three collections differ only in what the number means and how it
//! combines. A multiset counts copies and adds them; a weighted set holds a
//! weight and replaces it; a fuzzy set holds a membership in `[0, 1]` and
//! combines by the fuzzy operators. What they share, the map, the total and the
//! weighted picking, is here.
//!
//! [`MultiSet`]: super::multi_set::MultiSet
//! [`WeightedSet`]: super::weighted_set::WeightedSet
//! [`FuzzySet`]: super::fuzzy_set::FuzzySet

use crate::random::source::StochasticSource;
use crate::structures::hashing::FastHashMap;
use crate::structures::sampling;
use crate::structures::traits::Element;
use std::borrow::Borrow;
use std::hash::Hash;
use std::ops::{Add, Sub};

/// What a tally can count in: anything that has a zero, adds, subtracts and
/// compares.
///
/// Implemented for every such type at once by the blanket below, so a tally
/// counts in `usize` for a multiset, `f64` for a weighted or fuzzy set, or
/// anything else that behaves like a number.
pub trait TallyValue:
    Copy + Default + PartialOrd + Add<Output = Self> + Sub<Output = Self>
{
}

impl<N: Copy + Default + PartialOrd + Add<Output = N> + Sub<Output = N>> TallyValue for N {}

/// A map from element to a number, with the running total kept beside it.
///
/// The total is updated on every change rather than summed on demand, which is
/// what lets the collections built on this answer "what share does this take?"
/// and pick weighted elements without walking everything first. Every method
/// that changes a value goes through this type for that reason.
///
/// The map is a [`FastHashMap`], so iteration order is the same on every run
/// for the same insertions: a weighted pick from a tally is reproducible.
#[derive(Clone, Debug)]
pub struct Tally<T, N> {
    values: FastHashMap<T, N>,
    total: N,
}

impl<T, N: Default> Default for Tally<T, N> {
    fn default() -> Self {
        Self {
            values: FastHashMap::default(),
            total: N::default(),
        }
    }
}

impl<T, N: TallyValue> Tally<T, N> {
    /// How many distinct elements are counted, whatever their values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether nothing is counted at all.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Every value added together, kept up to date as the tally changes rather
    /// than summed here.
    pub fn total(&self) -> N {
        self.total
    }

    /// The underlying map, for the collections built on this to read directly.
    pub fn map(&self) -> &FastHashMap<T, N> {
        &self.values
    }

    /// The elements and their values, by value, consuming the tally.
    ///
    /// What the collections built on this hand out when they are themselves
    /// consumed, since a borrowed view cannot outlive them.
    pub fn into_map(self) -> FastHashMap<T, N> {
        self.values
    }

    /// Every element with its value, in the map's own order.
    pub fn iter(&self) -> impl Iterator<Item = (&T, N)> {
        self.values.iter().map(|(item, value)| (item, *value))
    }

    /// Every element, in the map's own order.
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, T, N> {
        self.values.keys()
    }

    /// Removes everything and resets the total to zero.
    pub fn clear(&mut self) {
        self.values.clear();
        self.total = N::default();
    }

    /// Picks up to `amount` distinct elements, each in proportion to its
    /// weight among those not yet picked.
    ///
    /// `to_weight` turns a value into a weight, which is what lets one
    /// implementation serve counts, weights and memberships alike; `adjust`
    /// then sees all the weights together, for a collection that has to rescale
    /// them before drawing. Allocates two lists, so
    /// [`Tally::choose_one_weighted`] is the better choice for a single
    /// pick.
    pub fn choose_weighted<S: StochasticSource + ?Sized>(
        &self,
        source: &mut S,
        amount: usize,
        to_weight: impl Fn(N) -> f64,
        adjust: impl FnOnce(&mut [f64]),
    ) -> Vec<&T> {
        if amount == 0 {
            return Vec::new();
        }
        let (items, mut values): (Vec<&T>, Vec<f64>) = self
            .values
            .iter()
            .map(|(item, value)| (item, to_weight(*value)))
            .unzip();
        adjust(&mut values);
        sampling::weighted_indices(&values, amount, source)
            .into_iter()
            .map(|index| items[index])
            .collect()
    }

    /// One element picked in proportion to its weight, without building any
    /// temporary lists.
    ///
    /// Every weight is divided by the largest before the walk, which keeps the
    /// sum finite even when the weights are enormous, and costs two passes
    /// rather than one. `None` when nothing has a positive finite weight.
    /// Should rounding carry the target past the end, the last element that
    /// could have been chosen is returned.
    pub fn choose_one_weighted<S: StochasticSource + ?Sized>(
        &self,
        source: &mut S,
        to_weight: impl Fn(N) -> f64,
    ) -> Option<&T> {
        let maximum = self
            .iter()
            .map(|(_, value)| to_weight(value))
            .fold(0.0, f64::max);
        if maximum <= 0.0 || !maximum.is_finite() {
            return None;
        }
        let total: f64 = self
            .iter()
            .map(|(_, value)| to_weight(value) / maximum)
            .sum();
        let mut target = source.unit_f64() * total;
        let mut last = None;
        for (item, value) in self.iter() {
            let weight = to_weight(value) / maximum;
            if weight > 0.0 {
                if target < weight {
                    return Some(item);
                }
                target -= weight;
                last = Some(item);
            }
        }
        last
    }
}

impl<T: Element, N: TallyValue> Tally<T, N> {
    /// Whether the element is counted at all, whatever its value.
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.values.contains_key(item)
    }

    /// An element's value, or `None` when it is not counted. Borrowed lookup,
    /// so a `String` key can be found by `&str`.
    pub fn get<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> Option<N>
    where
        T: Borrow<Q>,
    {
        self.values.get(item).copied()
    }

    /// Sets an element's value, returning the previous one, and moves the
    /// total by the difference. Takes an owned element, since it may be a new
    /// key.
    pub fn set(&mut self, item: T, value: N) -> Option<N> {
        let previous = self.values.insert(item, value);
        self.total = self.total - previous.unwrap_or_default() + value;
        previous
    }

    /// Changes an element that is already counted, without needing an owned
    /// key, and moves the total by the difference. A value of `None` removes
    /// the element instead.
    ///
    /// `None` is returned, and nothing changes, when the element is not
    /// counted: this cannot add one.
    pub fn update<Q: Hash + Eq + ?Sized>(&mut self, item: &Q, value: Option<N>) -> Option<N>
    where
        T: Borrow<Q>,
    {
        let Some(value) = value else {
            return self.remove(item);
        };
        let slot = self.values.get_mut(item)?;
        let previous = std::mem::replace(slot, value);
        self.total = self.total - previous + value;
        Some(previous)
    }

    /// Stops counting an element and returns its value, taking that value out
    /// of the total. `None` when it was not counted.
    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> Option<N>
    where
        T: Borrow<Q>,
    {
        let previous = self.values.remove(item)?;
        self.total = self.total - previous;
        Some(previous)
    }

    /// Keeps only the elements `keep` accepts, subtracting the values of the
    /// rest from the total in one pass.
    pub fn retain(&mut self, mut keep: impl FnMut(&T, N) -> bool) {
        let mut removed = N::default();
        self.values.retain(|item, value| {
            let kept = keep(item, *value);
            if !kept {
                removed = removed + *value;
            }
            kept
        });
        self.total = self.total - removed;
    }

    /// Replaces the whole map and its total in one step, for a caller that has
    /// rebuilt both together.
    ///
    /// The total is taken on trust rather than recomputed, so it must be the
    /// sum of the values given.
    pub fn replace_all(&mut self, values: FastHashMap<T, N>, total: N) {
        self.values = values;
        self.total = total;
    }
}
