//! The storage behind `MultiSet`, `WeightedSet` and `FuzzySet`: a map from
//! element to a number with a running total. The three collections differ
//! only in what the number means and how it combines.

use crate::misc::random::Random;
use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::sampling;
use crate::misc::structures::traits::Element;
use std::borrow::Borrow;
use std::hash::Hash;
use std::ops::{Add, Sub};

pub trait TallyValue:
    Copy + Default + PartialOrd + Add<Output = Self> + Sub<Output = Self>
{
}

impl<N: Copy + Default + PartialOrd + Add<Output = N> + Sub<Output = N>> TallyValue for N {}

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
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn total(&self) -> N {
        self.total
    }

    pub fn map(&self) -> &FastHashMap<T, N> {
        &self.values
    }

    pub fn iter(&self) -> impl Iterator<Item = (&T, N)> {
        self.values.iter().map(|(item, value)| (item, *value))
    }

    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, T, N> {
        self.values.keys()
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.total = N::default();
    }

    /// Picks up to `amount` distinct elements, each in proportion to its
    /// weight among those not yet picked. `to_weight` turns values into
    /// weights; `adjust` can then rescale them all.
    pub fn choose_weighted(
        &self,
        random: &mut Random,
        amount: usize,
        to_weight: impl Fn(N) -> f64,
        adjust: impl FnOnce(&mut [f64]),
    ) -> Vec<&T> {
        let (items, mut values): (Vec<&T>, Vec<f64>) = self
            .values
            .iter()
            .map(|(item, value)| (item, to_weight(*value)))
            .unzip();
        adjust(&mut values);
        sampling::weighted_indices(&values, amount, random)
            .into_iter()
            .map(|index| items[index])
            .collect()
    }
}

impl<T: Element, N: TallyValue> Tally<T, N> {
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.values.contains_key(item)
    }

    pub fn get<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> Option<N>
    where
        T: Borrow<Q>,
    {
        self.values.get(item).copied()
    }

    /// Sets an element's value, returning the previous one.
    pub fn set(&mut self, item: T, value: N) -> Option<N> {
        let previous = self.values.insert(item, value);
        self.total = self.total - previous.unwrap_or_default() + value;
        previous
    }

    /// Changes an existing element in place, without needing an owned key.
    /// `None` removes it.
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

    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> Option<N>
    where
        T: Borrow<Q>,
    {
        let previous = self.values.remove(item)?;
        self.total = self.total - previous;
        Some(previous)
    }

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

    /// Replaces every value at once.
    pub fn replace_all(&mut self, values: FastHashMap<T, N>, total: N) {
        self.values = values;
        self.total = total;
    }
}
