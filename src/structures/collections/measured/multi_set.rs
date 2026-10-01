//! A bag: each element has a whole-number count. Use it when duplicates
//! matter as counts rather than as repeated ordered entries, such as
//! inventories, token bags and frequency tables.
//!
//! Set algebra follows multiset rules: union keeps the larger count,
//! intersection the smaller, difference subtracts. `+` adds counts.

use crate::random::Random;
use crate::structures::collections::measured::tally::Tally;
use crate::structures::hashing::unordered_hash;
use crate::structures::traits::operators::impl_set_operators;
use crate::structures::traits::{
    Choose, ChooseByMeasure, Collection, CollectionInsert, CollectionRemove, Element, Measured,
    MeasuredMut, SetAlgebra, WeightedChoose,
};
use crate::units::Probability;
use std::borrow::Borrow;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
/// Unordered bag of elements of type `T`, storing a positive `usize` count
/// for every distinct element.
pub struct MultiSet<T> {
    tally: Tally<T, usize>,
}

impl<T> Default for MultiSet<T> {
    fn default() -> Self {
        Self {
            tally: Tally::default(),
        }
    }
}

impl<T> MultiSet<T> {
    /// Creates an empty multiset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct elements.
    pub fn len(&self) -> usize {
        self.tally.len()
    }

    /// Number of distinct elements. Prefer `len`; retained as a descriptive
    /// alias for measured-collection code.
    pub fn distinct_len(&self) -> usize {
        self.tally.len()
    }

    /// Total number of occurrences across all elements.
    pub fn total_count(&self) -> usize {
        self.tally.total()
    }

    /// Returns whether there are no distinct elements.
    pub fn is_empty(&self) -> bool {
        self.tally.is_empty()
    }

    /// Each distinct element with its count.
    pub fn iter(&self) -> impl Iterator<Item = (&T, usize)> {
        self.tally.iter()
    }

    /// Each distinct element once.
    pub fn distinct(&self) -> std::collections::hash_map::Keys<'_, T, usize> {
        self.tally.keys()
    }

    /// Every occurrence: an element with count 3 appears three times.
    pub fn occurrences(&self) -> impl Iterator<Item = &T> {
        self.tally
            .iter()
            .flat_map(|(item, count)| std::iter::repeat_n(item, count))
    }

    /// Removes all elements and counts while retaining allocated storage.
    pub fn clear(&mut self) {
        self.tally.clear();
    }
}

impl<T: Element> MultiSet<T> {
    /// Returns whether a value equivalent to borrowed `item` has a positive count.
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.tally.contains(item)
    }

    /// Returns the count of borrowed `item`, or zero when absent.
    pub fn count_of<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> usize
    where
        T: Borrow<Q>,
    {
        self.tally.get(item).unwrap_or(0)
    }

    /// Adds one occurrence.
    pub fn insert(&mut self, item: T) {
        self.insert_times(item, 1);
    }

    /// Adds `times` occurrences of `item`; zero leaves the multiset unchanged.
    pub fn insert_times(&mut self, item: T, times: usize) {
        if times == 0 {
            return;
        }
        let current = self.count_of(&item);
        self.tally.set(item, current + times);
    }

    /// Sets the count outright; zero removes the element.
    pub fn set_count(&mut self, item: T, count: usize) {
        if count == 0 {
            self.tally.remove(&item);
        } else {
            self.tally.set(item, count);
        }
    }

    /// Removes one occurrence. Returns whether anything was removed.
    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.remove_times(item, 1) > 0
    }

    /// Removes up to `times` occurrences and returns how many were removed.
    pub fn remove_times<Q: Hash + Eq + ?Sized>(&mut self, item: &Q, times: usize) -> usize
    where
        T: Borrow<Q>,
    {
        let current = self.count_of(item);
        let removed = times.min(current);
        if removed > 0 {
            let remaining = current - removed;
            self.tally
                .update(item, (remaining > 0).then_some(remaining));
        }
        removed
    }

    /// Removes every occurrence and returns how many there were.
    pub fn remove_all<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> usize
    where
        T: Borrow<Q>,
    {
        self.tally.remove(item).unwrap_or(0)
    }

    /// Retains entries for which `keep(item, count)` returns `true`.
    pub fn retain(&mut self, keep: impl FnMut(&T, usize) -> bool) {
        self.tally.retain(keep);
    }

    /// Adds the counts of both sets.
    pub fn sum(&self, other: &Self) -> Self {
        let mut output = self.clone();
        for (item, count) in other.iter() {
            output.insert_times(item.clone(), count);
        }
        output
    }

    fn from_counts<'a>(counts: impl Iterator<Item = (&'a T, usize)>) -> Self
    where
        T: 'a,
    {
        counts
            .filter(|(_, count)| *count > 0)
            .map(|(item, count)| (item.clone(), count))
            .collect()
    }
}

impl<T: Element> Collection for MultiSet<T> {
    type Item = T;

    /// Number of distinct elements.
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

impl<T: Element> CollectionInsert for MultiSet<T> {
    /// Adds one occurrence; always changes the set.
    fn insert(&mut self, item: T) -> bool {
        Self::insert(self, item);
        true
    }
}

impl<T: Element> CollectionRemove for MultiSet<T> {
    fn remove(&mut self, item: &T) -> bool {
        Self::remove(self, item)
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, mut keep: F) {
        self.tally.retain(|item, _| keep(item));
    }
}

impl<T: Element> Measured for MultiSet<T> {
    type Measure = usize;
    type Total = usize;

    fn measure_of(&self, item: &T) -> usize {
        self.count_of(item)
    }

    fn total_measure(&self) -> usize {
        self.tally.total()
    }

    fn distinct_len(&self) -> usize {
        self.tally.len()
    }

    fn measures(&self) -> impl Iterator<Item = (&T, usize)> {
        self.tally.iter()
    }
}

impl<T: Element> MeasuredMut for MultiSet<T> {
    fn set_measure(&mut self, item: T, count: usize) {
        self.set_count(item, count);
    }

    fn add_measure(&mut self, item: T, count: usize) {
        self.insert_times(item, count);
    }

    fn subtract_measure(&mut self, item: &T, count: usize) {
        self.remove_times(item, count);
    }
}

/// Picks elements in proportion to their counts.
impl<T: Element> Choose for MultiSet<T> {
    fn choose(&self, random: &mut Random) -> Option<&T> {
        let total = self.total_count();
        if total == 0 {
            return None;
        }
        let mut target = random.uniform_index(total);
        for (item, count) in self.iter() {
            if target < count {
                return Some(item);
            }
            target -= count;
        }
        None
    }

    fn choose_multiple(&self, random: &mut Random, amount: usize) -> Vec<&T> {
        self.tally
            .choose_weighted(random, amount, |count| count as f64, |_| {})
    }
}

/// Weighted by count: an element held three times is three times as likely.
impl<T: Element> WeightedChoose for MultiSet<T> {
    fn choose_weighted(&self, random: &mut Random) -> Option<&T> {
        self.choose(random)
    }

    fn choose_multiple_weighted(&self, random: &mut Random, amount: usize) -> Vec<&T> {
        self.choose_multiple(random, amount)
    }

    fn weighted_chance_of(&self, item: &T) -> Probability {
        Probability::ratio(self.count_of(item) as u64, self.total_count() as u64)
    }
}

impl<T: Element> ChooseByMeasure for MultiSet<T> {}

impl<T: Element> SetAlgebra for MultiSet<T> {
    /// The larger count of each element.
    fn union(&self, other: &Self) -> Self {
        let mut output = self.clone();
        for (item, count) in other.iter() {
            if count > output.count_of(item) {
                output.tally.set(item.clone(), count);
            }
        }
        output
    }

    /// The smaller count of each element.
    fn intersection(&self, other: &Self) -> Self {
        let (smaller, larger) = if self.distinct_len() <= other.distinct_len() {
            (self, other)
        } else {
            (other, self)
        };
        Self::from_counts(
            smaller
                .iter()
                .map(|(item, count)| (item, count.min(larger.count_of(item)))),
        )
    }

    /// Counts minus `other`'s counts.
    fn difference(&self, other: &Self) -> Self {
        Self::from_counts(
            self.iter()
                .map(|(item, count)| (item, count.saturating_sub(other.count_of(item)))),
        )
    }

    /// How far apart the counts are.
    fn symmetric_difference(&self, other: &Self) -> Self {
        let mine = self
            .iter()
            .map(|(item, count)| (item, count.abs_diff(other.count_of(item))));
        let only_theirs = other.iter().filter(|(item, _)| !self.contains(*item));
        Self::from_counts(mine.chain(only_theirs))
    }

    fn is_subset(&self, other: &Self) -> bool {
        self.distinct_len() <= other.distinct_len()
            && self
                .iter()
                .all(|(item, count)| count <= other.count_of(item))
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        !self.distinct().any(|item| other.contains(item))
    }
}

impl_set_operators!([T: Element] MultiSet<T>);

impl<T: Element> std::ops::Add<&MultiSet<T>> for &MultiSet<T> {
    type Output = MultiSet<T>;

    fn add(self, other: &MultiSet<T>) -> MultiSet<T> {
        self.sum(other)
    }
}

impl<T: Element> PartialEq for MultiSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.tally.map() == other.tally.map()
    }
}

impl<T: Element> Eq for MultiSet<T> {}

impl<T: Element> Hash for MultiSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(unordered_hash(self.tally.map()));
    }
}

/// One occurrence per item.
impl<T: Element> FromIterator<T> for MultiSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        let mut set = Self::new();
        set.extend(items);
        set
    }
}

/// `(element, count)` pairs; repeated elements add up.
impl<T: Element> FromIterator<(T, usize)> for MultiSet<T> {
    fn from_iter<I: IntoIterator<Item = (T, usize)>>(pairs: I) -> Self {
        let mut set = Self::new();
        set.extend(pairs);
        set
    }
}

impl<T: Element> Extend<T> for MultiSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        for item in items {
            self.insert(item);
        }
    }
}

impl<T: Element> Extend<(T, usize)> for MultiSet<T> {
    fn extend<I: IntoIterator<Item = (T, usize)>>(&mut self, pairs: I) {
        for (item, count) in pairs {
            self.insert_times(item, count);
        }
    }
}

/// Each element with its count, by value. The order is the map's, which is
/// settled by the contents rather than by the run.
impl<T: Element> IntoIterator for MultiSet<T> {
    type Item = (T, usize);
    type IntoIter = std::collections::hash_map::IntoIter<T, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.tally.into_map().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a MultiSet<T> {
    type Item = (&'a T, &'a usize);
    type IntoIter = std::collections::hash_map::Iter<'a, T, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.tally.map().iter()
    }
}

/// As each element with its count, such as `{stone x3, dirt x1}`.
impl<T: Element + std::fmt::Display> std::fmt::Display for MultiSet<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("{")?;

        for (position, (item, count)) in self.iter().enumerate() {
            if position > 0 {
                formatter.write_str(", ")?;
            }

            write!(formatter, "{item} x{count}")?;
        }

        formatter.write_str("}")
    }
}
