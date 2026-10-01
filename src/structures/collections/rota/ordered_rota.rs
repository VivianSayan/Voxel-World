//! Unique elements spread evenly over a fixed number of groups, with the order
//! kept both inside each group and across the rota as a whole.

use super::balance::Balance;
use crate::random::Random;
use crate::structures::hashing::FastHashMap;
use crate::structures::indices::Gate;
use crate::structures::traits::{
    BalancedPartition, Choose, Collection, CollectionInsert, CollectionRemove, ContentHashable,
    DeterministicOrder, Element, Kinded, Partitioned, StableHash, UniqueCollection,
};
use crate::units::digest::ContentHash;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One element: its group, and the value itself.
#[derive(Clone, Debug)]
struct Entry<T> {
    group: usize,
    value: T,
}

/// A set of unique elements divided into `GROUPS` groups of near-equal size,
/// where both the order within a group and the order of the whole are kept.
///
/// # Type parameters
///
/// - `T` is the element type, an [`Element`] so that an element can be found
///   and removed by value.
/// - `GROUPS` is how many groups there are, fixed at compile time and at least
///   one.
///
/// # The two orders, and how they agree
///
/// Every element is stamped with a position when it is added, and it keeps that
/// position for as long as it is held. Reading the whole rota with
/// [`OrderedRota::iter`] gives the elements in that order; reading one group
/// gives that group's elements in the same order, skipping the ones that belong
/// to other groups. A group is therefore a subsequence of the whole, and the
/// two orders can never disagree.
///
/// Balancing moves the *latest* element of the fullest group when it has to
/// move one, so the elements that have waited longest stay where they are.
///
/// # What it costs
///
/// A position is a `u64` counter and the groups are ordered sets of positions,
/// so insertion and removal are a few tree operations rather than the constant
/// time [`Rota`](super::Rota) manages, and reading a group costs a lookup per
/// element. That is the price of the order; where it is not needed,
/// [`Rota`](super::Rota) is the cheaper structure.
///
/// # What it is for
///
/// The same even rotation as [`Rota`](super::Rota), for work whose order is
/// part of its meaning: a queue of pending changes applied a slice per tick,
/// where earlier changes must still be applied before later ones.
#[derive(Clone)]
pub struct OrderedRota<T, const GROUPS: usize> {
    /// Every element by position, which is the order of the whole.
    entries: BTreeMap<u64, Entry<T>>,
    /// Where each value sits, for lookup and removal by value.
    positions: FastHashMap<T, u64>,
    /// The positions each group holds, in order.
    groups: [BTreeSet<u64>; GROUPS],
    /// The position the next element will take.
    next: u64,
    balance: Balance<GROUPS>,
}

impl<T: Element, const GROUPS: usize> OrderedRota<T, GROUPS> {
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

    /// How many elements it holds across every group.
    pub const fn len(&self) -> usize {
        self.balance.len()
    }

    /// Whether it holds none.
    pub const fn is_empty(&self) -> bool {
        self.balance.len() == 0
    }

    /// How many elements one group holds.
    pub const fn group_len(&self, group: usize) -> usize {
        self.balance.size(group)
    }

    /// Every group's size, in group order.
    pub const fn group_sizes(&self) -> &[usize] {
        self.balance.sizes()
    }

    /// Whether an element is held.
    pub fn contains(&self, value: &T) -> bool {
        self.positions.contains_key(value)
    }

    /// Which group an element is in, or `None` when it is not held.
    ///
    /// Worth reading fresh rather than caching: an element moves group when a
    /// removal elsewhere would otherwise unbalance the rota.
    pub fn group_of(&self, value: &T) -> Option<usize> {
        let position: u64 = *self.positions.get(value)?;

        self.entries.get(&position).map(|entry| entry.group)
    }

    /// Adds an element at the end of the order, in whichever group is
    /// emptiest, and returns whether it was new.
    pub fn insert(&mut self, value: T) -> bool {
        if self.positions.contains_key(&value) {
            return false;
        }

        let group: usize = self.balance.next_group();
        let position: u64 = self.next;

        self.next += 1;
        self.positions.insert(value.clone(), position);
        self.entries.insert(position, Entry { group, value });
        self.groups[group].insert(position);
        self.balance.record_insert(group);

        true
    }

    /// Removes an element and returns whether it was held.
    ///
    /// The elements around it keep their order, and one element may move
    /// between groups to keep the sizes within one of each other.
    pub fn remove(&mut self, value: &T) -> bool {
        let Some(position) = self.positions.remove(value) else {
            return false;
        };

        let Some(entry) = self.entries.remove(&position) else {
            return false;
        };

        self.groups[entry.group].remove(&position);
        self.balance.record_remove(entry.group);
        self.restore_balance();

        true
    }

    /// Every element, in the order they were added.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.entries.values().map(|entry| &entry.value)
    }

    /// The elements of one group, in the order they were added.
    ///
    /// Panics for a group index at or above `GROUPS`;
    /// [`OrderedRota::get_group`] is the checked form.
    pub fn group(&self, group: usize) -> impl DoubleEndedIterator<Item = &T> {
        assert!(group < GROUPS, "group {group} is outside 0..{GROUPS}");

        self.groups[group]
            .iter()
            .filter_map(|position| Some(&self.entries.get(position)?.value))
    }

    /// The elements of one group, or `None` for a group that does not exist.
    pub fn get_group(&self, group: usize) -> Option<impl DoubleEndedIterator<Item = &T>> {
        let positions: &BTreeSet<u64> = self.groups.get(group)?;

        Some(
            positions
                .iter()
                .filter_map(|position| Some(&self.entries.get(position)?.value)),
        )
    }

    /// The first element in the order, or `None` when the rota is empty.
    pub fn first(&self) -> Option<&T> {
        self.entries.values().next().map(|entry| &entry.value)
    }

    /// The last element in the order, or `None` when the rota is empty.
    pub fn last(&self) -> Option<&T> {
        self.entries.values().next_back().map(|entry| &entry.value)
    }

    /// Keeps only the elements `keep` accepts, in order, then spreads what is
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

    /// Moves the latest element of the fullest group to the emptiest, until no
    /// two groups differ by more than one.
    ///
    /// Taking the latest rather than the earliest is deliberate: the elements
    /// that have waited longest keep the group they were promised.
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

impl<T: Element, const GROUPS: usize> Default for OrderedRota<T, GROUPS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Element, const GROUPS: usize> Collection for OrderedRota<T, GROUPS> {
    type Item = T;

    fn len(&self) -> usize {
        self.balance.len()
    }

    fn contains(&self, item: &T) -> bool {
        OrderedRota::contains(self, item)
    }

    /// In the order they were added, across every group.
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: Element, const GROUPS: usize> CollectionInsert for OrderedRota<T, GROUPS> {
    fn insert(&mut self, item: T) -> bool {
        OrderedRota::insert(self, item)
    }
}

impl<T: Element, const GROUPS: usize> CollectionRemove for OrderedRota<T, GROUPS> {
    fn remove(&mut self, item: &T) -> bool {
        OrderedRota::remove(self, item)
    }

    fn clear(&mut self) {
        OrderedRota::clear(self);
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        OrderedRota::retain(self, keep);
    }
}

impl<T: Element, const GROUPS: usize> UniqueCollection for OrderedRota<T, GROUPS> {}

/// Uniform over the elements held, in the order they were added.
impl<T: Element, const GROUPS: usize> Choose for OrderedRota<T, GROUPS> {
    fn choose(&self, random: &mut Random) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        self.iter().nth(random.uniform_index(self.balance.len()))
    }
}

/// The groups as the rota keeps them, each in the order its elements were
/// added.
impl<T: Element, const GROUPS: usize> Partitioned for OrderedRota<T, GROUPS> {
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
impl<T: Element, const GROUPS: usize> BalancedPartition for OrderedRota<T, GROUPS> {}

impl<T: Element, const GROUPS: usize> DeterministicOrder for OrderedRota<T, GROUPS> {}

/// The elements in order, each with the group it sits in, so that both the
/// order and the division are part of the identity.
impl<T: Element + StableHash, const GROUPS: usize> ContentHashable for OrderedRota<T, GROUPS> {
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

impl<T: Element, const GROUPS: usize> FromIterator<T> for OrderedRota<T, GROUPS> {
    /// In the order given, skipping repeats.
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let mut rota: Self = Self::new();

        rota.extend(values);
        rota
    }
}

impl<T: Element, const GROUPS: usize> Extend<T> for OrderedRota<T, GROUPS> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        for value in values {
            self.insert(value);
        }
    }
}

/// Every element, in the order they were added.
impl<T: Element, const GROUPS: usize> IntoIterator for OrderedRota<T, GROUPS> {
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

impl<'a, T: Element, const GROUPS: usize> IntoIterator for &'a OrderedRota<T, GROUPS> {
    type Item = &'a T;
    type IntoIter = std::vec::IntoIter<&'a T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<&'a T>>().into_iter()
    }
}

impl<T: Element + fmt::Debug, const GROUPS: usize> fmt::Debug for OrderedRota<T, GROUPS> {
    /// As the groups and their contents in order, such as
    /// `OrderedRota[[a, c], [b]]`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OrderedRota")?;
        formatter
            .debug_list()
            .entries((0..GROUPS).map(|group| self.group(group).collect::<Vec<&T>>()))
            .finish()
    }
}

impl<T: Element, const GROUPS: usize> fmt::Display for OrderedRota<T, GROUPS> {
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

impl<T: Element, const GROUPS: usize> OrderedRota<T, GROUPS> {
    /// The members of a group that the gate allows, in the group's order.
    ///
    /// Nothing is rescheduled or removed: a rota turn is a read, so a member
    /// the gate refuses is simply left out of this turn and comes round again
    /// with its group next round. Panics for a group outside `0..GROUPS`, as
    /// [`OrderedRota::group`] does.
    pub fn group_with<'a, P: Element, V: Element + Kinded + PartialOrd>(
        &'a self,
        group: usize,
        gate: &'a Gate<T, P, V>,
    ) -> impl DoubleEndedIterator<Item = &'a T> {
        self.group(group).filter(move |member| gate.allows(member))
    }
}
