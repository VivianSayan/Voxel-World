//! Elements spread evenly over a fixed number of groups, repeats allowed, with
//! the order kept both inside each group and across the rota as a whole.

use super::balance::Balance;
use crate::random::source::StochasticSource;
use crate::structures::hashing::FastHashMap;
use crate::structures::indices::Gate;
use crate::structures::traits::{
    BalancedPartition, Choose, Collection, CollectionInsert, CollectionRemove, ContentHashable,
    DeterministicOrder, Element, Kinded, Measured, MeasuredMut, Partitioned, StableHash,
    WeightedChoose,
};
use crate::units::Probability;
use crate::units::digest::ContentHash;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One copy: its group, and the value itself.
#[derive(Clone, Debug)]
struct Entry<T> {
    group: usize,
    value: T,
}

/// A bag of elements divided into `GROUPS` groups of near-equal size, where the
/// same element may be held several times and the order is kept both inside a
/// group and across the whole.
///
/// # Type parameters
///
/// - `T` is the element type, an [`Element`] so that copies can be found by
///   value.
/// - `GROUPS` is how many groups there are, fixed at compile time and at least
///   one.
///
/// # How it relates to the others
///
/// It is [`OrderedRota`](super::OrderedRota) without the uniqueness check:
/// each copy takes its own position in the order and its own place in a group,
/// so two copies of one value usually sit in different groups and always at
/// different positions. Removal takes the *earliest* copy, which is what makes
/// a queue of repeated work behave as a queue.
///
/// Every other property is shared with the rest of the family: the groups stay
/// within one element of each other, a group reads back as a subsequence of the
/// whole, and balancing moves the latest copy of the fullest group rather than
/// an early one.
#[derive(Clone)]
pub struct OrderedMultiRota<T, const GROUPS: usize> {
    /// Every copy by position, which is the order of the whole.
    entries: BTreeMap<u64, Entry<T>>,
    /// The positions each value occupies, in order, so a removal can take the
    /// earliest copy.
    positions: FastHashMap<T, BTreeSet<u64>>,
    /// The positions each group holds, in order.
    groups: [BTreeSet<u64>; GROUPS],
    /// The position the next copy will take.
    next: u64,
    balance: Balance<GROUPS>,
}

impl<T: Element, const GROUPS: usize> OrderedMultiRota<T, GROUPS> {
    /// An empty rota whose groups are all empty.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            positions: FastHashMap::default(),
            groups: std::array::from_fn(|_| BTreeSet::new()),
            next: 0,
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
        self.positions.len()
    }

    /// How many copies of one value are held.
    pub fn count_of(&self, value: &T) -> usize {
        self.positions.get(value).map_or(0, BTreeSet::len)
    }

    /// Whether at least one copy is held.
    pub fn contains(&self, value: &T) -> bool {
        self.positions.contains_key(value)
    }

    /// How many copies one group holds.
    pub const fn group_len(&self, group: usize) -> usize {
        self.balance.size(group)
    }

    /// Every group's size, in group order.
    pub const fn group_sizes(&self) -> &[usize] {
        self.balance.sizes()
    }

    /// The group holding the earliest copy of a value, or `None` when none is
    /// held.
    pub fn group_of(&self, value: &T) -> Option<usize> {
        let position: u64 = *self.positions.get(value)?.first()?;

        self.entries.get(&position).map(|entry| entry.group)
    }

    /// Adds one copy at the end of the order, in whichever group is emptiest,
    /// and returns that group.
    pub fn insert(&mut self, value: T) -> usize {
        let group: usize = self.balance.next_group();
        let position: u64 = self.next;

        self.next += 1;
        self.positions
            .entry(value.clone())
            .or_default()
            .insert(position);
        self.entries.insert(position, Entry { group, value });
        self.groups[group].insert(position);
        self.balance.record_insert(group);

        group
    }

    /// Adds `count` copies, each taking its own position and group.
    pub fn insert_times(&mut self, value: T, count: usize) {
        for _ in 0..count {
            self.insert(value.clone());
        }
    }

    /// Removes the earliest copy of a value and returns whether any was held.
    pub fn remove_one(&mut self, value: &T) -> bool {
        let Some(positions) = self.positions.get_mut(value) else {
            return false;
        };

        let Some(position) = positions.pop_first() else {
            return false;
        };

        if positions.is_empty() {
            self.positions.remove(value);
        }

        if let Some(entry) = self.entries.remove(&position) {
            self.groups[entry.group].remove(&position);
            self.balance.record_remove(entry.group);
        }

        self.restore_balance();

        true
    }

    /// Removes every copy of a value and returns how many went.
    pub fn remove_all(&mut self, value: &T) -> usize {
        let Some(positions) = self.positions.remove(value) else {
            return 0;
        };

        let removed: usize = positions.len();

        for position in positions {
            if let Some(entry) = self.entries.remove(&position) {
                self.groups[entry.group].remove(&position);
                self.balance.record_remove(entry.group);
            }
        }

        self.restore_balance();

        removed
    }

    /// Every copy, in the order they were added.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.entries.values().map(|entry| &entry.value)
    }

    /// The copies in one group, in the order they were added. Panics for a
    /// group index at or above `GROUPS`.
    pub fn group(&self, group: usize) -> impl DoubleEndedIterator<Item = &T> {
        assert!(group < GROUPS, "group {group} is outside 0..{GROUPS}");

        self.groups[group]
            .iter()
            .filter_map(|position| Some(&self.entries.get(position)?.value))
    }

    /// The copies in one group, or `None` for a group that does not exist.
    pub fn get_group(&self, group: usize) -> Option<impl DoubleEndedIterator<Item = &T>> {
        let positions: &BTreeSet<u64> = self.groups.get(group)?;

        Some(
            positions
                .iter()
                .filter_map(|position| Some(&self.entries.get(position)?.value)),
        )
    }

    /// The earliest copy in the order, or `None` when the rota is empty.
    pub fn first(&self) -> Option<&T> {
        self.entries.values().next().map(|entry| &entry.value)
    }

    /// The latest copy in the order, or `None` when the rota is empty.
    pub fn last(&self) -> Option<&T> {
        self.entries.values().next_back().map(|entry| &entry.value)
    }

    /// Keeps only the copies `keep` accepts, in order, then spreads what is
    /// left evenly again.
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut keep: F) {
        let kept: Vec<T> = std::mem::take(&mut self.entries)
            .into_values()
            .map(|entry| entry.value)
            .filter(|value| keep(value))
            .collect();

        self.clear();

        for value in kept {
            self.insert(value);
        }
    }

    /// Removes everything, and restarts the order.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.positions.clear();

        for group in &mut self.groups {
            group.clear();
        }

        self.next = 0;
        self.balance.clear();
    }

    /// Moves the latest copy of the fullest group to the emptiest, until no two
    /// groups differ by more than one.
    fn restore_balance(&mut self) {
        while let Some((from, to)) = self.balance.transfer() {
            let Some(position) = self.groups[from].pop_last() else {
                break;
            };

            if let Some(entry) = self.entries.get_mut(&position) {
                entry.group = to;
            }

            self.groups[to].insert(position);
            self.balance.record_move(from, to);
        }
    }
}

impl<T: Element, const GROUPS: usize> Default for OrderedMultiRota<T, GROUPS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Element, const GROUPS: usize> Collection for OrderedMultiRota<T, GROUPS> {
    type Item = T;

    /// Counting repeats; [`OrderedMultiRota::distinct_len`] counts values.
    fn len(&self) -> usize {
        self.balance.len()
    }

    fn contains(&self, item: &T) -> bool {
        OrderedMultiRota::contains(self, item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: Element, const GROUPS: usize> CollectionInsert for OrderedMultiRota<T, GROUPS> {
    /// Always adds a copy, so this always reports `true`.
    fn insert(&mut self, item: T) -> bool {
        OrderedMultiRota::insert(self, item);

        true
    }
}

impl<T: Element, const GROUPS: usize> CollectionRemove for OrderedMultiRota<T, GROUPS> {
    /// Removes the earliest copy; [`OrderedMultiRota::remove_all`] removes them
    /// all.
    fn remove(&mut self, item: &T) -> bool {
        OrderedMultiRota::remove_one(self, item)
    }

    fn clear(&mut self) {
        OrderedMultiRota::clear(self);
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        OrderedMultiRota::retain(self, keep);
    }
}

/// Uniform over the copies held, so a value with more copies is likelier.
impl<T: Element, const GROUPS: usize> Choose for OrderedMultiRota<T, GROUPS> {
    fn choose<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        self.iter().nth(source.index_below(self.balance.len()))
    }
}

/// The groups as the rota keeps them, each in the order its copies were added.
impl<T: Element, const GROUPS: usize> Partitioned for OrderedMultiRota<T, GROUPS> {
    type Member = T;

    fn group_count(&self) -> usize {
        GROUPS
    }

    fn group_len(&self, group: usize) -> usize {
        self.groups.get(group).map_or(0, BTreeSet::len)
    }

    fn group_members(&self, group: usize) -> impl Iterator<Item = &T> {
        self.groups
            .get(group)
            .into_iter()
            .flatten()
            .filter_map(|position| Some(&self.entries.get(position)?.value))
    }
}

/// Sizes stay within one of each other after every insertion and removal.
impl<T: Element, const GROUPS: usize> BalancedPartition for OrderedMultiRota<T, GROUPS> {}

/// Measured by how many copies of each value are held.
impl<T: Element, const GROUPS: usize> Measured for OrderedMultiRota<T, GROUPS> {
    type Measure = usize;
    type Total = usize;

    fn measure_of(&self, item: &T) -> usize {
        self.count_of(item)
    }

    fn total_measure(&self) -> usize {
        self.balance.len()
    }

    fn distinct_len(&self) -> usize {
        OrderedMultiRota::distinct_len(self)
    }

    fn measures(&self) -> impl Iterator<Item = (&T, usize)> {
        self.positions
            .iter()
            .map(|(value, positions)| (value, positions.len()))
    }
}

impl<T: Element, const GROUPS: usize> MeasuredMut for OrderedMultiRota<T, GROUPS> {
    /// Adds or removes copies until exactly `measure` are held, taking the
    /// earliest copies away first. A measure of zero removes the value.
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

/// Weighted by how many copies a value has.
impl<T: Element, const GROUPS: usize> WeightedChoose for OrderedMultiRota<T, GROUPS> {
    fn choose_weighted<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Option<&T> {
        self.choose(source)
    }

    fn choose_multiple_weighted<S: StochasticSource + ?Sized>(&self, source: &mut S, amount: usize) -> Vec<&T> {
        self.choose_multiple(source, amount)
    }

    fn weighted_chance_of(&self, item: &T) -> Probability {
        Probability::ratio(self.count_of(item) as u64, self.balance.len() as u64)
    }
}

impl<T: Element, const GROUPS: usize> DeterministicOrder for OrderedMultiRota<T, GROUPS> {}

/// The copies in order, each with the group it sits in.
impl<T: Element + StableHash, const GROUPS: usize> ContentHashable for OrderedMultiRota<T, GROUPS> {
    fn content_hash(&self) -> ContentHash {
        self.entries
            .values()
            .fold(ContentHash::EMPTY, |state, entry| {
                state
                    .and(entry.value.stable_hash())
                    .and_value(entry.group as u128)
            })
    }
}

impl<T: Element, const GROUPS: usize> FromIterator<T> for OrderedMultiRota<T, GROUPS> {
    /// In the order given, keeping repeats.
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let mut rota: Self = Self::new();

        rota.extend(values);
        rota
    }
}

impl<T: Element, const GROUPS: usize> Extend<T> for OrderedMultiRota<T, GROUPS> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        for value in values {
            self.insert(value);
        }
    }
}

impl<T: Element, const GROUPS: usize> IntoIterator for OrderedMultiRota<T, GROUPS> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries
            .into_values()
            .map(|entry| entry.value)
            .collect::<Vec<T>>()
            .into_iter()
    }
}

impl<'a, T: Element, const GROUPS: usize> IntoIterator for &'a OrderedMultiRota<T, GROUPS> {
    type Item = &'a T;
    type IntoIter = std::vec::IntoIter<&'a T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<&'a T>>().into_iter()
    }
}

impl<T: Element + fmt::Debug, const GROUPS: usize> fmt::Debug for OrderedMultiRota<T, GROUPS> {
    /// As the groups and their contents in order.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OrderedMultiRota")?;
        formatter
            .debug_list()
            .entries((0..GROUPS).map(|group| self.group(group).collect::<Vec<&T>>()))
            .finish()
    }
}

impl<T: Element, const GROUPS: usize> fmt::Display for OrderedMultiRota<T, GROUPS> {
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

impl<T: Element, const GROUPS: usize> OrderedMultiRota<T, GROUPS> {
    /// The members of a group that the gate allows, in the group's order.
    ///
    /// Nothing is rescheduled or removed: a rota turn is a read, so a member
    /// the gate refuses is simply left out of this turn and comes round again
    /// with its group next round. Panics for a group outside `0..GROUPS`, as
    /// [`OrderedMultiRota::group`] does.
    pub fn group_with<'a, P: Element, V: Element + Kinded + PartialOrd>(
        &'a self,
        group: usize,
        gate: &'a Gate<T, P, V>,
    ) -> impl DoubleEndedIterator<Item = &'a T> {
        self.group(group).filter(move |member| gate.allows(member))
    }
}
