//! Values at reusable slots, addressed by handles that expire with them.
//!
//! An [`Id`] is a number, and a number that named something removed a while ago
//! still names its slot. Whatever took that slot next answers to it, which is
//! the bug where an arrow follows the entity that replaced the one it was
//! chasing.
//!
//! A slot map closes that by numbering the reuses. The handle carries both the
//! slot and which occupant of it, so a handle from a previous occupant fails to
//! resolve instead of resolving to the wrong value.

use crate::structures::traits::{
    CanonicalOrder, Capacity, ContentHashable, DeterministicOrder, HandleStore, StableHash,
};
use crate::units::digest::ContentHash;
use crate::units::{Id, IdKind};
use std::marker::PhantomData;

/// How many of an id's bits name the slot; the rest count its reuses.
const SLOT_BITS: u32 = 40;

/// The largest slot an id can name, which is about a million million.
const MAX_SLOT: u64 = (1 << SLOT_BITS) - 1;

/// One slot: what it holds, and how many times it has been filled.
#[derive(Clone, Debug)]
struct Slot<T> {
    value: Option<T>,
    generation: u64,
}

/// Values addressed by [`Id<K>`], where an id stops working once its value is
/// removed.
///
/// An id packs a slot number in its low 40 bits and a generation in the rest.
/// Removing a value bumps the slot's generation, so the id that pointed at it
/// no longer matches and [`SlotMap::get`] answers `None` even after something
/// else has taken the slot. A slot can be reused about sixteen million times
/// before the generation wraps, at which point an ancient handle could match
/// again; nothing in a world's lifetime comes close.
///
/// Iteration is by slot, so it is [`DeterministicOrder`]: the same sequence of
/// insertions and removals gives the same order on every run.
#[derive(Clone, Debug)]
pub struct SlotMap<K: IdKind, T> {
    slots: Vec<Slot<T>>,
    free: Vec<u64>,
    occupied: usize,
    kind: PhantomData<K>,
}

impl<K: IdKind, T> SlotMap<K, T> {
    /// An empty map.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            occupied: 0,
            kind: PhantomData,
        }
    }

    /// How many values are stored.
    pub const fn len(&self) -> usize {
        self.occupied
    }

    /// Whether none are.
    pub const fn is_empty(&self) -> bool {
        self.occupied == 0
    }

    /// How many slots exist, including the free ones waiting to be reused.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Stores a value and returns the id that reaches it.
    ///
    /// Reuses a free slot where there is one, so a map that is filled and
    /// emptied repeatedly does not grow. Panics only if every one of the `2^40`
    /// slots is in use at once.
    pub fn insert(&mut self, value: T) -> Id<K> {
        if let Some(slot) = self.free.pop() {
            let entry: &mut Slot<T> = &mut self.slots[slot as usize];

            entry.value = Some(value);
            self.occupied += 1;

            return Self::identify(slot, entry.generation);
        }

        let slot: u64 = self.slots.len() as u64;

        assert!(
            slot <= MAX_SLOT,
            "a slot map cannot hold more than 2^40 values"
        );

        self.slots.push(Slot {
            value: Some(value),
            generation: 0,
        });
        self.occupied += 1;

        Self::identify(slot, 0)
    }

    /// The value an id reaches, or `None` if it was removed or the id came from
    /// another map.
    pub fn get(&self, id: Id<K>) -> Option<&T> {
        let (slot, generation): (u64, u64) = Self::split(id)?;
        let entry: &Slot<T> = self.slots.get(slot as usize)?;

        (entry.generation == generation).then_some(entry.value.as_ref()?)
    }

    /// The value an id reaches, for changing.
    pub fn get_mut(&mut self, id: Id<K>) -> Option<&mut T> {
        let (slot, generation): (u64, u64) = Self::split(id)?;
        let entry: &mut Slot<T> = self.slots.get_mut(slot as usize)?;

        (entry.generation == generation).then_some(entry.value.as_mut()?)
    }

    /// Whether an id still reaches a value.
    pub fn contains(&self, id: Id<K>) -> bool {
        self.get(id).is_some()
    }

    /// Removes the value an id reaches and returns it, freeing the slot and
    /// retiring every id that pointed at it.
    pub fn remove(&mut self, id: Id<K>) -> Option<T> {
        let (slot, generation): (u64, u64) = Self::split(id)?;
        let entry: &mut Slot<T> = self.slots.get_mut(slot as usize)?;

        if entry.generation != generation {
            return None;
        }

        let value: T = entry.value.take()?;

        entry.generation += 1;
        self.occupied -= 1;
        self.free.push(slot);

        Some(value)
    }

    /// Every id and value, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (Id<K>, &T)> {
        self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            Some((
                Self::identify(slot as u64, entry.generation),
                entry.value.as_ref()?,
            ))
        })
    }

    /// Every value, in slot order.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|entry| entry.value.as_ref())
    }

    /// Every value, for changing, in slot order.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots
            .iter_mut()
            .filter_map(|entry| entry.value.as_mut())
    }

    /// Every id, in slot order.
    pub fn ids(&self) -> impl Iterator<Item = Id<K>> + '_ {
        self.iter().map(|(id, _)| id)
    }

    /// Keeps only the values `keep` accepts, retiring the ids of the rest.
    pub fn retain<F: FnMut(Id<K>, &T) -> bool>(&mut self, mut keep: F) {
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            let Some(value) = entry.value.as_ref() else {
                continue;
            };

            if keep(Self::identify(slot as u64, entry.generation), value) {
                continue;
            }

            entry.value = None;
            entry.generation += 1;
            self.occupied -= 1;
            self.free.push(slot as u64);
        }
    }

    /// Removes every value, retiring every id.
    ///
    /// Keeps the slots, and bumps the generation of each so that no id from
    /// before the clear can resolve again.
    pub fn clear(&mut self) {
        self.free.clear();

        for (slot, entry) in self.slots.iter_mut().enumerate() {
            if entry.value.take().is_some() {
                entry.generation += 1;
            }

            self.free.push(slot as u64);
        }

        self.occupied = 0;
    }

    /// An id from a slot and a generation.
    fn identify(slot: u64, generation: u64) -> Id<K> {
        Id::new((generation << SLOT_BITS) | slot)
    }

    /// The slot and generation an id names, or `None` for [`Id::NONE`].
    fn split(id: Id<K>) -> Option<(u64, u64)> {
        if id.is_none() {
            return None;
        }

        Some((id.value() & MAX_SLOT, id.value() >> SLOT_BITS))
    }
}

impl<K: IdKind, T> Default for SlotMap<K, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: IdKind, T> HandleStore for SlotMap<K, T> {
    type Handle = Id<K>;
    type Value = T;

    fn len(&self) -> usize {
        SlotMap::len(self)
    }
    fn insert(&mut self, value: T) -> Id<K> {
        SlotMap::insert(self, value)
    }
    fn get(&self, handle: Id<K>) -> Option<&T> {
        SlotMap::get(self, handle)
    }
    fn get_mut(&mut self, handle: Id<K>) -> Option<&mut T> {
        SlotMap::get_mut(self, handle)
    }
    fn remove(&mut self, handle: Id<K>) -> Option<T> {
        SlotMap::remove(self, handle)
    }
}

/// Room is counted in slots.
impl<K: IdKind, T> Capacity for SlotMap<K, T> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
            occupied: 0,
            kind: PhantomData,
        }
    }

    fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
    }

    /// Drops the spare room, and the free list with it where the free slots sit
    /// at the end.
    fn shrink_to_fit(&mut self) {
        while self.slots.last().is_some_and(|entry| entry.value.is_none()) {
            self.slots.pop();
        }

        self.free.retain(|slot| (*slot as usize) < self.slots.len());
        self.slots.shrink_to_fit();
        self.free.shrink_to_fit();
    }
}

impl<K: IdKind, T> DeterministicOrder for SlotMap<K, T> {}
impl<K: IdKind, T> CanonicalOrder for SlotMap<K, T> {}

/// The values in slot order, with the slot and generation each sits at, so a
/// map that has reused a slot hashes differently from one that never emptied
/// it.
impl<K: IdKind, T: StableHash> ContentHashable for SlotMap<K, T> {
    fn content_hash(&self) -> ContentHash {
        self.slots
            .iter()
            .enumerate()
            .fold(ContentHash::EMPTY, |state, (slot, entry)| {
                match &entry.value {
                    Some(value) => state
                        .and_value(slot as u128)
                        .and_value(entry.generation as u128)
                        .and(value.stable_hash()),
                    None => state,
                }
            })
    }
}

impl<K: IdKind, T> FromIterator<T> for SlotMap<K, T> {
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let mut map: Self = Self::new();

        map.extend(values);
        map
    }
}

impl<K: IdKind, T> Extend<T> for SlotMap<K, T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        for value in values {
            self.insert(value);
        }
    }
}

/// As how much is stored and how many slots it occupies, such as
/// `7 values in 10 slots`.
impl<K: IdKind, T> std::fmt::Display for SlotMap<K, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} values in {} slots",
            self.occupied,
            self.slots.len()
        )
    }
}
