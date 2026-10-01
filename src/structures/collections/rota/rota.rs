//! Unique elements spread evenly over a fixed number of groups, with no order
//! kept inside a group.

use super::balance::Balance;
use crate::random::Random;
use crate::structures::hashing::FastHashMap;
use crate::structures::indices::Gate;
use crate::structures::traits::{
    BalancedPartition, Capacity, Choose, Collection, CollectionInsert, CollectionRemove,
    ContentHashable, DeterministicOrder, Element, Kinded, Partitioned, SetAlgebra, StableHash,
    UniqueCollection,
};
use crate::units::digest::ContentHash;
use std::fmt;

/// Where one element lives: which group, and where in that group's list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Place {
    group: usize,
    slot: usize,
}

/// A set of unique elements divided into `GROUPS` groups of near-equal size.
///
/// # Type parameters
///
/// - `T` is the element type. It must be an [`Element`], meaning hashable,
///   comparable and cloneable, because each element is also recorded in an
///   index that says where it lives.
/// - `GROUPS` is how many groups there are, fixed at compile time and at least
///   one. It is the number of turns the work is spread over, so a rota that
///   feeds "one group per tick" over eight ticks is a `Rota<T, 8>`.
///
/// # What it promises
///
/// No two group sizes ever differ by more than one, whatever sequence of
/// insertions and removals produced them. With `n` elements over `g` groups,
/// every group holds either `n / g` or `n / g + 1`.
///
/// Keeping that promise sometimes means moving an element from one group to
/// another: removing from an already-small group would otherwise open a gap of
/// two. An element is therefore not guaranteed to stay in the group it first
/// landed in, which matters if a caller has cached where something was.
///
/// # What it does not promise
///
/// Order, inside a group or across them. Removal is a swap with the group's
/// last element, which is what keeps it constant time and what scrambles the
/// order as a side effect. Use [`OrderedRota`](super::OrderedRota) where the
/// order matters, and [`MultiRota`](super::MultiRota) where an element may
/// appear more than once.
///
/// # What it is for
///
/// Any population that has to be visited in even instalments rather than all
/// at once: entities whose expensive check runs on one tick in eight, chunks
/// revalidated a slice at a time, listeners polled in rotation. The caller
/// decides what a group means; the rota only keeps the shares even.
///
/// ```ignore
/// let mut due: Rota<EntityId, 8> = Rota::new();
/// due.insert(entity);
///
/// // Each tick, take one group's worth. Every element is visited once per
/// // eight ticks, and no tick carries more than one more than another.
/// for entity in due.group(tick % 8) {
///     check(entity);
/// }
/// ```
#[derive(Clone)]
pub struct Rota<T, const GROUPS: usize> {
    groups: [Vec<T>; GROUPS],
    places: FastHashMap<T, Place>,
    balance: Balance<GROUPS>,
}

impl<T: Element, const GROUPS: usize> Rota<T, GROUPS> {
    /// An empty rota whose groups are all empty.
    pub fn new() -> Self {
        Self {
            groups: std::array::from_fn(|_| Vec::new()),
            places: FastHashMap::default(),
            balance: Balance::new(),
        }
    }

    /// An empty rota with room for `capacity` elements in total, divided
    /// evenly between the groups.
    pub fn with_capacity(capacity: usize) -> Self {
        let per_group: usize = capacity.div_ceil(GROUPS);

        Self {
            groups: std::array::from_fn(|_| Vec::with_capacity(per_group)),
            places: FastHashMap::with_capacity_and_hasher(capacity, Default::default()),
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

    /// How many elements one group holds. Panics for a group that does not
    /// exist.
    pub const fn group_len(&self, group: usize) -> usize {
        self.balance.size(group)
    }

    /// Every group's size, in group order.
    pub const fn group_sizes(&self) -> &[usize] {
        self.balance.sizes()
    }

    /// The elements of one group, in no particular order.
    ///
    /// Panics for a group index at or above `GROUPS`, which is a mistake the
    /// caller can always avoid; [`Rota::get_group`] is the checked form.
    pub fn group(&self, group: usize) -> &[T] {
        assert!(group < GROUPS, "group {group} is outside 0..{GROUPS}");

        &self.groups[group]
    }

    /// The elements of one group, or `None` for a group that does not exist.
    pub fn get_group(&self, group: usize) -> Option<&[T]> {
        self.groups.get(group).map(Vec::as_slice)
    }

    /// Every group in turn, each as its own slice.
    pub fn groups(&self) -> impl ExactSizeIterator<Item = &[T]> {
        self.groups.iter().map(Vec::as_slice)
    }

    /// Which group an element is in, or `None` when it is not held.
    ///
    /// Worth reading fresh rather than caching: an element moves group when a
    /// removal elsewhere would otherwise unbalance the rota.
    pub fn group_of(&self, value: &T) -> Option<usize> {
        self.places.get(value).map(|place| place.group)
    }

    /// Whether an element is held.
    pub fn contains(&self, value: &T) -> bool {
        self.places.contains_key(value)
    }

    /// Adds an element to whichever group is emptiest, and returns whether it
    /// was new. An element already held is left where it is.
    ///
    /// Constant time, apart from the scan over group counts that picks the
    /// emptiest.
    pub fn insert(&mut self, value: T) -> bool {
        if self.places.contains_key(&value) {
            return false;
        }

        let group: usize = self.balance.next_group();

        self.places.insert(
            value.clone(),
            Place {
                group,
                slot: self.groups[group].len(),
            },
        );
        self.groups[group].push(value);
        self.balance.record_insert(group);

        true
    }

    /// Removes an element and returns whether it was held.
    ///
    /// Constant time. The group's last element takes the vacated slot, and one
    /// element may move between groups to keep the sizes within one of each
    /// other.
    pub fn remove(&mut self, value: &T) -> bool {
        let Some(place) = self.places.remove(value) else {
            return false;
        };

        self.detach(place);
        self.balance.record_remove(place.group);
        self.restore_balance();

        true
    }

    /// Every element, group by group.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.groups.iter().flatten()
    }

    /// Keeps only the elements `keep` accepts, then spreads what is left
    /// evenly again.
    ///
    /// Rebuilds rather than removing one at a time, so it costs one pass over
    /// the rota however many elements go.
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut keep: F) {
        let kept: Vec<T> = std::mem::replace(&mut self.groups, std::array::from_fn(|_| Vec::new()))
            .into_iter()
            .flatten()
            .filter(|value| keep(value))
            .collect();

        self.places.clear();
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

        self.places.clear();
        self.balance.clear();
    }

    /// Takes an element out of its slot, moving the group's last element into
    /// the gap and recording where that one went.
    fn detach(&mut self, place: Place) {
        let Self { groups, places, .. } = self;
        let group: &mut Vec<T> = &mut groups[place.group];

        group.swap_remove(place.slot);

        if let Some(moved) = group.get(place.slot)
            && let Some(entry) = places.get_mut(moved)
        {
            entry.slot = place.slot;
        }
    }

    /// Moves elements from the fullest group to the emptiest until no two
    /// differ by more than one.
    fn restore_balance(&mut self) {
        while let Some((from, to)) = self.balance.transfer() {
            let Self {
                groups,
                places,
                balance,
            } = self;

            let Some(value) = groups[from].pop() else {
                break;
            };

            if let Some(entry) = places.get_mut(&value) {
                *entry = Place {
                    group: to,
                    slot: groups[to].len(),
                };
            }

            groups[to].push(value);
            balance.record_move(from, to);
        }
    }
}

impl<T: Element, const GROUPS: usize> Default for Rota<T, GROUPS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Element, const GROUPS: usize> Collection for Rota<T, GROUPS> {
    type Item = T;

    fn len(&self) -> usize {
        self.balance.len()
    }

    fn contains(&self, item: &T) -> bool {
        Rota::contains(self, item)
    }

    /// Group by group, and within a group in whatever order removals have left
    /// behind.
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: Element, const GROUPS: usize> CollectionInsert for Rota<T, GROUPS> {
    fn insert(&mut self, item: T) -> bool {
        Rota::insert(self, item)
    }
}

impl<T: Element, const GROUPS: usize> CollectionRemove for Rota<T, GROUPS> {
    fn remove(&mut self, item: &T) -> bool {
        Rota::remove(self, item)
    }

    fn clear(&mut self) {
        Rota::clear(self);
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        Rota::retain(self, keep);
    }
}

impl<T: Element, const GROUPS: usize> UniqueCollection for Rota<T, GROUPS> {}

/// Uniform over the elements held, not over the groups: a group with more
/// elements is more likely to be drawn from, which is what keeps every element
/// equally likely.
impl<T: Element, const GROUPS: usize> Choose for Rota<T, GROUPS> {
    fn choose(&self, random: &mut Random) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let mut remaining: usize = random.uniform_index(self.balance.len());

        for group in &self.groups {
            if remaining < group.len() {
                return group.get(remaining);
            }

            remaining -= group.len();
        }

        None
    }
}

/// Room is counted in elements and spread evenly over the groups.
impl<T: Element, const GROUPS: usize> Capacity for Rota<T, GROUPS> {
    fn with_capacity(capacity: usize) -> Self {
        Rota::with_capacity(capacity)
    }

    fn capacity(&self) -> usize {
        self.groups.iter().map(Vec::capacity).sum()
    }

    fn reserve(&mut self, additional: usize) {
        let per_group: usize = additional.div_ceil(GROUPS);

        for group in &mut self.groups {
            group.reserve(per_group);
        }

        self.places.reserve(additional);
    }

    fn shrink_to_fit(&mut self) {
        for group in &mut self.groups {
            group.shrink_to_fit();
        }

        self.places.shrink_to_fit();
    }
}

/// The groups as the rota keeps them, in no particular order within each.
impl<T: Element, const GROUPS: usize> Partitioned for Rota<T, GROUPS> {
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
impl<T: Element, const GROUPS: usize> BalancedPartition for Rota<T, GROUPS> {}

/// Set operations over the elements, ignoring how either rota had divided
/// them: the result is a fresh rota, spread evenly as it is built. Two rotas
/// holding the same elements are equal as sets however their groups differ.
impl<T: Element, const GROUPS: usize> SetAlgebra for Rota<T, GROUPS> {
    fn union(&self, other: &Self) -> Self {
        self.iter().chain(other.iter()).cloned().collect()
    }

    fn intersection(&self, other: &Self) -> Self {
        self.iter()
            .filter(|value| other.contains(value))
            .cloned()
            .collect()
    }

    fn difference(&self, other: &Self) -> Self {
        self.iter()
            .filter(|value| !other.contains(value))
            .cloned()
            .collect()
    }

    fn is_subset(&self, other: &Self) -> bool {
        self.iter().all(|value| other.contains(value))
    }
}

impl<T: Element, const GROUPS: usize> DeterministicOrder for Rota<T, GROUPS> {}

/// Each element with the group it sits in, combined so that the order within a
/// group does not matter but the division does.
///
/// Two rotas agree only if they hold the same elements *and* have them divided
/// the same way, which is the contents of this structure: the division is the
/// thing it exists to maintain. Two rotas built from the same elements in a
/// different order will usually disagree.
impl<T: Element + StableHash, const GROUPS: usize> ContentHashable for Rota<T, GROUPS> {
    fn content_hash(&self) -> ContentHash {
        let combined: u128 = self
            .groups
            .iter()
            .enumerate()
            .flat_map(|(group, values)| values.iter().map(move |value| (group, value)))
            .fold(0, |state, (group, value)| {
                state
                    ^ value
                        .stable_hash()
                        .and_value(group as u128)
                        .value()
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            });

        ContentHash::EMPTY.and_value(combined)
    }
}

impl<T: Element, const GROUPS: usize> FromIterator<T> for Rota<T, GROUPS> {
    /// Spreads the elements as they arrive, skipping repeats.
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let mut rota: Self = Self::new();

        rota.extend(values);
        rota
    }
}

impl<T: Element, const GROUPS: usize> Extend<T> for Rota<T, GROUPS> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        for value in values {
            self.insert(value);
        }
    }
}

/// Every element, group by group.
impl<T: Element, const GROUPS: usize> IntoIterator for Rota<T, GROUPS> {
    type Item = T;
    type IntoIter = std::iter::Flatten<std::array::IntoIter<Vec<T>, GROUPS>>;

    fn into_iter(self) -> Self::IntoIter {
        self.groups.into_iter().flatten()
    }
}

impl<'a, T: Element, const GROUPS: usize> IntoIterator for &'a Rota<T, GROUPS> {
    type Item = &'a T;
    type IntoIter = std::iter::Flatten<std::slice::Iter<'a, Vec<T>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.groups.iter().flatten()
    }
}

impl<T: Element + fmt::Debug, const GROUPS: usize> fmt::Debug for Rota<T, GROUPS> {
    /// As the groups and their contents, such as `Rota[[a, c], [b]]`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Rota")?;
        formatter.debug_list().entries(self.groups.iter()).finish()
    }
}

impl<T: Element, const GROUPS: usize> fmt::Display for Rota<T, GROUPS> {
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

impl<T: Element, const GROUPS: usize> Rota<T, GROUPS> {
    /// The members of a group that the gate allows, in the group's order.
    ///
    /// Nothing is rescheduled or removed: a rota turn is a read, so a member
    /// the gate refuses is simply left out of this turn and comes round again
    /// with its group next round. Panics for a group outside `0..GROUPS`, as
    /// [`Rota::group`] does.
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
