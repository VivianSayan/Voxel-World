//! Graded membership: each element belongs to the set to a degree in
//! (0, 1]. This is partial belonging, not a weight or a count, for soft
//! classification, uncertain state and heuristic tagging.
//!
//! Union (`|`) is the probabilistic sum `a + b - ab`, intersection (`&`)
//! the product `ab`, difference (`-`) is `min(a, 1 - b)` and complement
//! (`!`) is `1 - a`, so memberships behave like independent probabilities.

use crate::misc::random::Random;
use crate::misc::structures::collections::measured::tally::Tally;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, Measured, MeasuredMut,
    UniqueCollection,
};
use std::borrow::Borrow;
use std::hash::Hash;

/// Suggested tolerance for calls to [`FuzzySet::approx_eq`].
pub const MEMBERSHIP_TOLERANCE: f64 = 1e-5;

#[inline]
fn probabilistic_sum(a: f64, b: f64) -> f64 {
    a + b - a * b
}

#[derive(Clone, Debug)]
/// Unordered fuzzy set whose elements of type `T` have memberships in `(0, 1]`.
///
/// A zero membership is represented by absence. Non-finite inputs are rejected.
pub struct FuzzySet<T> {
    tally: Tally<T, f64>,
}

impl<T> Default for FuzzySet<T> {
    fn default() -> Self {
        Self {
            tally: Tally::default(),
        }
    }
}

impl<T> FuzzySet<T> {
    /// Creates an empty fuzzy set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of elements with non-zero membership.
    pub fn len(&self) -> usize {
        self.tally.len()
    }

    /// Returns whether no element has non-zero membership.
    pub fn is_empty(&self) -> bool {
        self.tally.is_empty()
    }

    /// Sum of all memberships (the scalar cardinality).
    pub fn cardinality(&self) -> f64 {
        self.tally.total()
    }

    /// Each element with its membership.
    pub fn iter(&self) -> impl Iterator<Item = (&T, f64)> {
        self.tally.iter()
    }

    /// Iterates over every element with non-zero membership.
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, T, f64> {
        self.tally.keys()
    }

    /// Removes every membership while retaining allocated storage.
    pub fn clear(&mut self) {
        self.tally.clear();
    }

    /// The element with the highest membership.
    pub fn max_member(&self) -> Option<&T> {
        self.iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(item, _)| item)
    }

    /// The element with the lowest membership.
    pub fn min_member(&self) -> Option<&T> {
        self.iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(item, _)| item)
    }

    /// Resolves the fuzziness: each element is kept with probability equal
    /// to its membership. Collect into whichever collection is needed.
    pub fn realize<'a>(&'a self, random: &mut Random) -> impl Iterator<Item = &'a T> {
        self.iter()
            .filter(move |(_, membership)| *membership >= 1.0 || random.bernoulli(*membership))
            .map(|(item, _)| item)
    }
}

impl<T: Element> FuzzySet<T> {
    /// True when `item` belongs to the set at all.
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.tally.contains(item)
    }

    /// True when `item` belongs with membership above `threshold`.
    pub fn contains_above<Q: Hash + Eq + ?Sized>(&self, item: &Q, threshold: f64) -> bool
    where
        T: Borrow<Q>,
    {
        self.membership_of(item) > threshold
    }

    /// Returns the membership of borrowed `item`, or zero when absent.
    pub fn membership_of<Q: Hash + Eq + ?Sized>(&self, item: &Q) -> f64
    where
        T: Borrow<Q>,
    {
        self.tally.get(item).unwrap_or(0.0)
    }

    /// Adds `item` with full membership.
    pub fn insert(&mut self, item: T) {
        self.set_membership(item, 1.0);
    }

    /// Sets the membership, clamped to [0, 1]. Zero removes the element.
    pub fn set_membership(&mut self, item: T, membership: f64) {
        assert!(membership.is_finite(), "membership must be finite");
        let membership = membership.clamp(0.0, 1.0);
        if membership == 0.0 {
            self.tally.remove(&item);
        } else {
            self.tally.set(item, membership);
        }
    }

    /// Combines `amount` into the membership with the probabilistic sum,
    /// so repeated evidence approaches but never passes 1.
    pub fn reinforce(&mut self, item: T, amount: f64) {
        assert!(
            amount.is_finite(),
            "membership reinforcement must be finite"
        );
        let combined = probabilistic_sum(self.membership_of(&item), amount.clamp(0.0, 1.0));
        self.set_membership(item, combined);
    }

    /// Lowers the membership by `amount`, removing the element at zero.
    pub fn weaken<Q: Hash + Eq + ?Sized>(&mut self, item: &Q, amount: f64)
    where
        T: Borrow<Q>,
    {
        assert!(amount.is_finite(), "membership weakening must be finite");
        if let Some(current) = self.tally.get(item) {
            let lowered = (current - amount).clamp(0.0, 1.0);
            self.tally.update(item, (lowered > 0.0).then_some(lowered));
        }
    }

    /// Removes the element, returning its membership.
    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, item: &Q) -> Option<f64>
    where
        T: Borrow<Q>,
    {
        self.tally.remove(item)
    }

    /// Retains entries for which `keep(item, membership)` returns `true`.
    pub fn retain(&mut self, keep: impl FnMut(&T, f64) -> bool) {
        self.tally.retain(keep);
    }

    /// Scales memberships so they sum to one.
    pub fn normalise(&mut self) {
        let total = self.cardinality();
        if total.abs() < f64::EPSILON {
            return;
        }
        let scaled = self
            .iter()
            .map(|(item, membership)| (item.clone(), membership / total))
            .collect();
        self.tally.replace_all(scaled, 1.0);
    }

    /// `1 - membership` over the currently stored support. Elements absent
    /// from this finite support are not materialized; full members drop out.
    pub fn complement(&self) -> Self {
        self.iter()
            .map(|(item, membership)| (item.clone(), 1.0 - membership))
            .collect()
    }

    fn mapped(&self, mut membership: impl FnMut(&T, f64) -> f64) -> Self {
        let mut output = Self::new();
        for (item, current) in self.iter() {
            output.set_membership(item.clone(), membership(item, current));
        }
        output
    }

    /// Approximate comparison kept explicit so `PartialEq` remains an
    /// equivalence relation.
    pub fn approx_eq(&self, other: &Self, tolerance: f64) -> bool {
        assert!(
            tolerance.is_finite() && tolerance >= 0.0,
            "tolerance must be finite and non-negative"
        );
        self.len() == other.len()
            && self.iter().all(|(item, membership)| {
                other
                    .tally
                    .get(item)
                    .is_some_and(|theirs| (membership - theirs).abs() <= tolerance)
            })
    }
}

impl<T: Element> Collection for FuzzySet<T> {
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

impl<T: Element> CollectionInsert for FuzzySet<T> {
    /// Raises the element to full membership.
    fn insert(&mut self, item: T) -> bool {
        let changed = self.membership_of(&item) < 1.0;
        Self::insert(self, item);
        changed
    }
}

impl<T: Element> CollectionRemove for FuzzySet<T> {
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

impl<T: Element> UniqueCollection for FuzzySet<T> {}

impl<T: Element> Measured for FuzzySet<T> {
    type Measure = f64;

    fn measure_of(&self, item: &T) -> f64 {
        self.membership_of(item)
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

/// `add_measure` reinforces and `subtract_measure` weakens.
impl<T: Element> MeasuredMut for FuzzySet<T> {
    fn set_measure(&mut self, item: T, membership: f64) {
        self.set_membership(item, membership);
    }

    fn add_measure(&mut self, item: T, amount: f64) {
        self.reinforce(item, amount);
    }

    fn subtract_measure(&mut self, item: &T, amount: f64) {
        self.weaken(item, amount);
    }
}

/// Picks elements in proportion to their membership.
impl<T: Element> Choose for FuzzySet<T> {
    fn choose_multiple(&self, random: &mut Random, amount: usize) -> Vec<&T> {
        self.tally
            .choose_weighted(random, amount, |membership| membership, |_| {})
    }
}

/// Fuzzy operations are inherent rather than `SetAlgebra`: probabilistic
/// union is intentionally not idempotent and therefore does not obey the
/// laws generic crisp-set algorithms commonly assume.
impl<T: Element> FuzzySet<T> {
    /// Returns the probabilistic union, using `a + b - a*b` per element.
    pub fn union(&self, other: &Self) -> Self {
        let mut output = self.clone();
        for (item, membership) in other.iter() {
            output.reinforce(item.clone(), membership);
        }
        output
    }

    /// Returns the probabilistic intersection, using `a*b` per element.
    pub fn intersection(&self, other: &Self) -> Self {
        let (smaller, larger) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        smaller.mapped(|item, membership| membership * larger.membership_of(item))
    }

    /// Returns the fuzzy difference, using `min(a, 1-b)` per element.
    pub fn difference(&self, other: &Self) -> Self {
        self.mapped(|item, membership| membership.min(1.0 - other.membership_of(item)))
    }

    /// True when no element belongs more strongly here than in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.len() <= other.len()
            && self
                .iter()
                .all(|(item, membership)| membership <= other.membership_of(item))
    }

    /// Returns whether the sets have no element with positive membership in common.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        !self.keys().any(|item| other.contains(item))
    }
}

impl<T: Element> std::ops::BitOr<&FuzzySet<T>> for &FuzzySet<T> {
    type Output = FuzzySet<T>;

    fn bitor(self, other: &FuzzySet<T>) -> FuzzySet<T> {
        self.union(other)
    }
}

impl<T: Element> std::ops::BitAnd<&FuzzySet<T>> for &FuzzySet<T> {
    type Output = FuzzySet<T>;

    fn bitand(self, other: &FuzzySet<T>) -> FuzzySet<T> {
        self.intersection(other)
    }
}

impl<T: Element> std::ops::Sub<&FuzzySet<T>> for &FuzzySet<T> {
    type Output = FuzzySet<T>;

    fn sub(self, other: &FuzzySet<T>) -> FuzzySet<T> {
        self.difference(other)
    }
}

impl<T: Element> std::ops::Not for &FuzzySet<T> {
    type Output = FuzzySet<T>;

    fn not(self) -> FuzzySet<T> {
        self.complement()
    }
}

impl<T: Element> PartialEq for FuzzySet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.tally.map() == other.tally.map()
    }
}

/// Every element gets full membership.
impl<T: Element> FromIterator<T> for FuzzySet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        let mut set = Self::new();
        for item in items {
            set.insert(item);
        }
        set
    }
}

/// `(element, membership)` pairs; repeats are combined with the
/// probabilistic sum.
impl<T: Element> FromIterator<(T, f64)> for FuzzySet<T> {
    fn from_iter<I: IntoIterator<Item = (T, f64)>>(pairs: I) -> Self {
        let mut set = Self::new();
        set.extend(pairs);
        set
    }
}

impl<T: Element> Extend<(T, f64)> for FuzzySet<T> {
    fn extend<I: IntoIterator<Item = (T, f64)>>(&mut self, pairs: I) {
        for (item, membership) in pairs {
            self.reinforce(item, membership);
        }
    }
}

impl<'a, T> IntoIterator for &'a FuzzySet<T> {
    type Item = (&'a T, &'a f64);
    type IntoIter = std::collections::hash_map::Iter<'a, T, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.tally.map().iter()
    }
}
