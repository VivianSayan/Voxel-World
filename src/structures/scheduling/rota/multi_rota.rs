//! Elements spread evenly over a fixed number of groups, repeats allowed, with
//! no order kept inside a group.

use super::balance::Balance;
use crate::random::source::StochasticSource;
use crate::structures::hashing::FastHashMap;
use crate::structures::indices::Gate;
use crate::structures::traits::{
    BalancedPartition, Capacity, Choose, Collection, CollectionInsert, CollectionRemove,
    ContentHashable, DeterministicOrder, Element, Kinded, Measured, MeasuredMut, Partitioned,
    StableHash, WeightedChoose,
};
use crate::units::Probability;
use crate::units::digest::ContentHash;
use std::fmt;

/// A bag of elements divided into `GROUPS` groups of near-equal size, where the
/// same element may be held several times.
///
/// # Type parameters
///
/// - `T` is the element type, an [`Element`] so that occurrences can be found
///   by value.
/// - `GROUPS` is how many groups there are, fixed at compile time and at least
///   one.
///
/// # How it differs from [`Rota`](super::Rota)
///
/// Inserting an element that is already held adds another copy rather than
/// doing nothing, and each copy is placed independently, so two copies of the
/// same element usually land in different groups. Removal takes one copy at a
/// time.
///
/// The index records which groups hold copies of a value, but not where in
/// those groups, so removing a copy scans the group it lives in: `O(n /
/// GROUPS)` rather than the constant time [`Rota`](super::Rota) manages.
/// Everything else, including the balance promise, is the same.
///
/// # What it is for
///
/// The same rotation as [`Rota`](super::Rota), where the population is counted
/// rather than enumerated: several pending jobs of the same kind, several
/// instances of one template, repeated work for one owner. Each copy takes its
/// own turn.
#[derive(Clone)]
pub struct MultiRota<T, const GROUPS: usize> {
    groups: [Vec<T>; GROUPS],
    /// For each value, the group of every copy held. One entry per copy, so its
    /// length is that value's count.
    homes: FastHashMap<T, Vec<usize>>,
    balance: Balance<GROUPS>,
}

impl<T: Element, const GROUPS: usize> MultiRota<T, GROUPS> {
    /// An empty rota whose groups are all empty.
    pub fn new() -> Self {
        Self {
            groups: std::array::from_fn(|_| Vec::new()),
            homes: FastHashMap::default(),
            balance: Balance::new(),
        }
    }

    /// An empty rota with room for `capacity` copies in total, divided evenly
    /// between the groups.
    pub fn with_capacity(capacity: usize) -> Self {
        let per_group: usize = capacity.div_ceil(GROUPS);

        Self {
            groups: std::array::from_fn(|_| Vec::with_capacity(per_group)),
            homes: FastHashMap::default(),
            balance: Balance::new(),
        }
    }

    /// How many groups the rota divides into, which is `GROUPS`.
    pub const fn group_count(&self) -> usize {
        GROUPS
    }

    /// How many copies it holds across every group, counting repeats.
    pub const fn len(&self) -> usize {
        self.balance.len()
    }

    /// Whether it holds none.
    pub const fn is_empty(&self) -> bool {
        self.balance.len() == 0
    }

    /// How many distinct values it holds, however many copies each has.
    pub fn distinct_len(&self) -> usize {
        self.homes.len()
    }

    /// How many copies of one value are held, across every group.
    pub fn count_of(&self, value: &T) -> usize {
        self.homes.get(value).map_or(0, Vec::len)
    }

    /// Whether at least one copy is held.
    pub fn contains(&self, value: &T) -> bool {
        self.homes.contains_key(value)
    }

    /// How many copies one group holds.
    pub const fn group_len(&self, group: usize) -> usize {
        self.balance.size(group)
    }

    /// Every group's size, in group order.
    pub const fn group_sizes(&self) -> &[usize] {
        self.balance.sizes()
    }

    /// The copies in one group, in no particular order. Panics for a group that
    /// does not exist.
    pub fn group(&self, group: usize) -> &[T] {
        assert!(group < GROUPS, "group {group} is outside 0..{GROUPS}");

        &self.groups[group]
    }

    /// The copies in one group, or `None` for a group that does not exist.
    pub fn get_group(&self, group: usize) -> Option<&[T]> {
        self.groups.get(group).map(Vec::as_slice)
    }

    /// Every group in turn, each as its own slice.
    pub fn groups(&self) -> impl ExactSizeIterator<Item = &[T]> {
        self.groups.iter().map(Vec::as_slice)
    }

    /// The groups holding a copy of this value, one entry per copy and in no
    /// particular order.
    pub fn groups_of(&self, value: &T) -> &[usize] {
        self.homes.get(value).map_or(&[], Vec::as_slice)
    }

    /// Adds one copy to whichever group is emptiest, and returns that group.
    pub fn insert(&mut self, value: T) -> usize {
        let group: usize = self.balance.next_group();

        self.homes.entry(value.clone()).or_default().push(group);
        self.groups[group].push(value);
        self.balance.record_insert(group);

        group
    }

    /// Adds `count` copies, each placed independently.
    pub fn insert_times(&mut self, value: T, count: usize) {
        for _ in 0..count {
            self.insert(value.clone());
        }
    }

    /// Removes one copy and returns whether any was held.
    ///
    /// Costs a scan of the group the copy lives in, which holds about
    /// `len / GROUPS` elements.
    pub fn remove_one(&mut self, value: &T) -> bool {
        let Some(homes) = self.homes.get_mut(value) else {
            return false;
        };

        let Some(group) = homes.pop() else {
            return false;
        };

        if homes.is_empty() {
            self.homes.remove(value);
        }

        if let Some(slot) = self.groups[group].iter().position(|held| held == value) {
            self.groups[group].swap_remove(slot);
            self.balance.record_remove(group);
        }

        self.restore_balance();

        true
    }

    /// Removes every copy of a value and returns how many went.
    pub fn remove_all(&mut self, value: &T) -> usize {
        let Some(homes) = self.homes.remove(value) else {
            return 0;
        };

        let removed: usize = homes.len();

        for group in homes {
            if let Some(slot) = self.groups[group].iter().position(|held| held == value) {
                self.groups[group].swap_remove(slot);
                self.balance.record_remove(group);
            }
        }

        self.restore_balance();

        removed
    }

    /// Every copy, group by group.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.groups.iter().flatten()
    }

    /// Keeps only the copies `keep` accepts, then spreads what is left evenly
    /// again.
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut keep: F) {
        let kept: Vec<T> = std::mem::replace(&mut self.groups, std::array::from_fn(|_| Vec::new()))
            .into_iter()
            .flatten()
            .filter(|value| keep(value))
            .collect();

        self.homes.clear();
        self.balance.clear();

        for value in kept {
            self.insert(value);
        }
    }

    /// Removes everything.
    pub fn clear(&mut self) {
        for group in &mut self.groups {
            group.clear();
        }

        self.homes.clear();
        self.balance.clear();
    }

    /// Moves copies from the fullest group to the emptiest until no two differ
    /// by more than one.
    fn restore_balance(&mut self) {
        while let Some((from, to)) = self.balance.transfer() {
            let Self {
                groups,
                homes,
                balance,
            } = self;

            let Some(value) = groups[from].pop() else {
                break;
            };

            if let Some(entries) = homes.get_mut(&value)
                && let Some(entry) = entries.iter_mut().find(|entry| **entry == from)
            {
                *entry = to;
            }

            groups[to].push(value);
            balance.record_move(from, to);
        }
    }
}

impl<T: Element, const GROUPS: usize> Default for MultiRota<T, GROUPS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Element, const GROUPS: usize> Collection for MultiRota<T, GROUPS> {
    type Item = T;

    /// Counting repeats; [`MultiRota::distinct_len`] counts values.
    fn len(&self) -> usize {
        self.balance.len()
    }

    fn contains(&self, item: &T) -> bool {
        MultiRota::contains(self, item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: Element, const GROUPS: usize> CollectionInsert for MultiRota<T, GROUPS> {
    /// Always adds a copy, so this always reports `true`.
    fn insert(&mut self, item: T) -> bool {
        MultiRota::insert(self, item);

        true
    }
}

impl<T: Element, const GROUPS: usize> CollectionRemove for MultiRota<T, GROUPS> {
    /// Removes one copy; [`MultiRota::remove_all`] removes them all.
    fn remove(&mut self, item: &T) -> bool {
        MultiRota::remove_one(self, item)
    }

    fn clear(&mut self) {
        MultiRota::clear(self);
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        MultiRota::retain(self, keep);
    }
}

/// Uniform over the copies held, so a value with more copies is likelier.
impl<T: Element, const GROUPS: usize> Choose for MultiRota<T, GROUPS> {
    fn choose<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let mut remaining: usize = source.index_below(self.balance.len());

        for group in &self.groups {
            if remaining < group.len() {
                return group.get(remaining);
            }

            remaining -= group.len();
        }

        None
    }
}

/// Room is counted in copies and spread evenly over the groups.
impl<T: Element, const GROUPS: usize> Capacity for MultiRota<T, GROUPS> {
    fn with_capacity(capacity: usize) -> Self {
        MultiRota::with_capacity(capacity)
    }

    fn capacity(&self) -> usize {
        self.groups.iter().map(Vec::capacity).sum()
    }

    fn reserve(&mut self, additional: usize) {
        let per_group: usize = additional.div_ceil(GROUPS);

        for group in &mut self.groups {
            group.reserve(per_group);
        }
    }

    fn shrink_to_fit(&mut self) {
        for group in &mut self.groups {
            group.shrink_to_fit();
        }

        self.homes.shrink_to_fit();
    }
}

/// The groups as the rota keeps them, in no particular order within each.
impl<T: Element, const GROUPS: usize> Partitioned for MultiRota<T, GROUPS> {
    type Member = T;

    fn group_count(&self) -> usize {
        GROUPS
    }

    fn group_len(&self, group: usize) -> usize {
        self.groups.get(group).map_or(0, Vec::len)
    }

    fn group_members(&self, group: usize) -> impl Iterator<Item = &T> {
        self.groups.get(group).into_iter().flatten()
    }
}

/// Sizes stay within one of each other after every insertion and removal.
impl<T: Element, const GROUPS: usize> BalancedPartition for MultiRota<T, GROUPS> {}

/// Measured by how many copies of each value are held, as a
/// [`MultiSet`](crate::structures::collections::MultiSet) is.
impl<T: Element, const GROUPS: usize> Measured for MultiRota<T, GROUPS> {
    type Measure = usize;
    type Total = usize;

    fn measure_of(&self, item: &T) -> usize {
        self.count_of(item)
    }

    fn total_measure(&self) -> usize {
        self.balance.len()
    }

    fn distinct_len(&self) -> usize {
        MultiRota::distinct_len(self)
    }

    fn measures(&self) -> impl Iterator<Item = (&T, usize)> {
        self.homes.iter().map(|(value, homes)| (value, homes.len()))
    }
}

impl<T: Element, const GROUPS: usize> MeasuredMut for MultiRota<T, GROUPS> {
    /// Adds or removes copies until exactly `measure` are held. A measure of
    /// zero removes the value.
    fn set_measure(&mut self, item: T, measure: usize) {
        let held: usize = self.count_of(&item);

        match measure.cmp(&held) {
            std::cmp::Ordering::Greater => self.insert_times(item, measure - held),
            std::cmp::Ordering::Less => {
                for _ in 0..held - measure {
                    self.remove_one(&item);
                }
            }
            std::cmp::Ordering::Equal => {}
        }
    }

    fn add_measure(&mut self, item: T, amount: usize) {
        self.insert_times(item, amount);
    }

    fn subtract_measure(&mut self, item: &T, amount: usize) {
        for _ in 0..amount {
            if !self.remove_one(item) {
                break;
            }
        }
    }
}

/// Weighted by how many copies a value has, which is what
/// [`Choose`] already does here; this states it in the type.
impl<T: Element, const GROUPS: usize> WeightedChoose for MultiRota<T, GROUPS> {
    fn choose_weighted<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Option<&T> {
        self.choose(source)
    }

    fn choose_multiple_weighted<S: StochasticSource + ?Sized>(
        &self,
        source: &mut S,
        amount: usize,
    ) -> Vec<&T> {
        self.choose_multiple(source, amount)
    }

    fn weighted_chance_of(&self, item: &T) -> Probability {
        Probability::ratio(self.count_of(item) as u64, self.balance.len() as u64)
    }
}

impl<T: Element, const GROUPS: usize> DeterministicOrder for MultiRota<T, GROUPS> {}

/// Each copy with the group it sits in, combined so that the order within a
/// group does not matter but the division does; see
/// [`Rota`](super::Rota)'s implementation for what that means.
impl<T: Element + StableHash, const GROUPS: usize> ContentHashable for MultiRota<T, GROUPS> {
    fn content_hash(&self) -> ContentHash {
        let combined: u128 = self
            .groups
            .iter()
            .enumerate()
            .flat_map(|(group, values)| values.iter().map(move |value| (group, value)))
            .fold(0, |state, (group, value)| {
                state.wrapping_add(
                    value
                        .stable_hash()
                        .and_value(group as u128)
                        .value()
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15),
                )
            });

        ContentHash::EMPTY.and_value(combined)
    }
}

impl<T: Element, const GROUPS: usize> FromIterator<T> for MultiRota<T, GROUPS> {
    /// Spreads the copies as they arrive, keeping repeats.
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let mut rota: Self = Self::new();

        rota.extend(values);
        rota
    }
}

impl<T: Element, const GROUPS: usize> Extend<T> for MultiRota<T, GROUPS> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        for value in values {
            self.insert(value);
        }
    }
}

impl<T: Element, const GROUPS: usize> IntoIterator for MultiRota<T, GROUPS> {
    type Item = T;
    type IntoIter = std::iter::Flatten<std::array::IntoIter<Vec<T>, GROUPS>>;

    fn into_iter(self) -> Self::IntoIter {
        self.groups.into_iter().flatten()
    }
}

impl<'a, T: Element, const GROUPS: usize> IntoIterator for &'a MultiRota<T, GROUPS> {
    type Item = &'a T;
    type IntoIter = std::iter::Flatten<std::slice::Iter<'a, Vec<T>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.groups.iter().flatten()
    }
}

impl<T: Element + fmt::Debug, const GROUPS: usize> fmt::Debug for MultiRota<T, GROUPS> {
    /// As the groups and their contents, such as `MultiRota[[a, a], [b]]`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MultiRota")?;
        formatter.debug_list().entries(self.groups.iter()).finish()
    }
}

impl<T: Element, const GROUPS: usize> fmt::Display for MultiRota<T, GROUPS> {
    /// As the share each group carries, such as `3 over [2, 1]`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} over {:?}",
            self.balance.len(),
            self.balance.sizes()
        )
    }
}

impl<T: Element, const GROUPS: usize> MultiRota<T, GROUPS> {
    /// The members of a group that the gate allows, in the group's order.
    ///
    /// Nothing is rescheduled or removed: a rota turn is a read, so a member
    /// the gate refuses is simply left out of this turn and comes round again
    /// with its group next round. Panics for a group outside `0..GROUPS`, as
    /// [`MultiRota::group`] does.
    pub fn group_with<'a, P: Element, V: Element + Kinded + PartialOrd>(
        &'a self,
        group: usize,
        gate: &'a Gate<T, P, V>,
    ) -> impl Iterator<Item = &'a T> {
        self.group(group)
            .iter()
            .filter(move |member| gate.allows(member))
    }
}
