//! The module's traits, implemented on the standard library's sequences.
//!
//! There is no `Stack` or `Queue` here, because `Vec` and `VecDeque` already
//! are those and wrapping them only hid their own methods behind new names. A
//! stack is a `Vec` used with `push` and `pop`; a queue is a `VecDeque` used
//! with `push_back` and `pop_front`. What the wrappers really contributed was
//! the traits, so the traits are here instead and the storage stays standard.
//!
//! Everything in this module that is not in the standard library earns its
//! place by doing something the standard library does not: `RingBuffer` evicts,
//! `OrderedSet` keeps positions beside uniqueness, `PriorityQueue` can remove
//! and re-prioritise any element, `Scheduler` holds work by the step it is due
//! on. Renaming a `Vec` is not that.
//!
//! One thing to know when reaching for these: `HashMap` and `HashSet` from the
//! standard library seed their hasher randomly for each process, so their
//! iteration order changes between runs. Anything that has to replay from a
//! seed wants `FastHashMap` and `FastHashSet` from `hashing` instead, which are
//! the same types with a deterministic hasher.

use crate::random::Random;
use crate::structures::sampling;
use crate::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, InsertAt, Reorder, Sequence,
    SequenceMut,
};
use std::cmp::Ordering;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Vec: the stack
// ---------------------------------------------------------------------------

impl<T: PartialEq> Collection for Vec<T> {
    type Item = T;

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn contains(&self, item: &T) -> bool {
        <[T]>::contains(self, item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: PartialEq> CollectionInsert for Vec<T> {
    /// Pushes onto the end, which for a stack is the top.
    fn insert(&mut self, item: T) -> bool {
        self.push(item);
        true
    }
}

impl<T: PartialEq> CollectionRemove for Vec<T> {
    /// Removes the first match, keeping the order of everything else.
    fn remove(&mut self, item: &T) -> bool {
        let Some(position) = self.iter().position(|held| held == item) else {
            return false;
        };

        Vec::remove(self, position);
        true
    }

    fn clear(&mut self) {
        Vec::clear(self);
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        Vec::retain(self, keep);
    }
}

impl<T: PartialEq> Sequence for Vec<T> {
    fn get(&self, index: usize) -> Option<&T> {
        <[T]>::get(self, index)
    }

    fn index_of(&self, item: &T) -> Option<usize> {
        self.iter().position(|held| held == item)
    }
}

impl<T: PartialEq> SequenceMut for Vec<T> {
    fn remove_at(&mut self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }

        Some(Vec::remove(self, index))
    }
}

impl<T: PartialEq> InsertAt for Vec<T> {
    fn insert_at(&mut self, index: usize, item: T) -> bool {
        if index > self.len() {
            return false;
        }

        Vec::insert(self, index, item);
        true
    }
}

impl<T: PartialEq> Reorder for Vec<T> {
    fn sort_by<F: FnMut(&T, &T) -> Ordering>(&mut self, compare: F) {
        <[T]>::sort_by(self, compare);
    }

    fn reverse(&mut self) {
        <[T]>::reverse(self);
    }

    fn shuffle(&mut self, random: &mut Random) {
        sampling::shuffle(self, random);
    }
}

/// Indexed directly: one draw for one pick, rather than the trait's default of
/// walking every element with a draw each. The default is right for
/// collections that cannot be indexed, and wrong here twice over: it is O(n)
/// where this is O(1), and it consumes a different stream of draws, so a world
/// seeded through it would pick differently. `OrderedSet` and `RingBuffer` pick
/// the same way, which is what keeps every indexable collection in step.
///
/// `ChooseMut` arrives on its own: the module blanket-implements it for any
/// `Choose + CollectionRemove` whose items clone.
impl<T: PartialEq> Choose for Vec<T> {
    fn choose(&self, random: &mut Random) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        <[T]>::get(self, random.uniform_index(self.len()))
    }

    fn choose_multiple(&self, random: &mut Random, amount: usize) -> Vec<&T> {
        sampling::uniform_indices(self.len(), amount, Some(random))
            .into_iter()
            .map(|index| &self[index])
            .collect()
    }
}

// ---------------------------------------------------------------------------
// VecDeque: the queue
// ---------------------------------------------------------------------------

impl<T: PartialEq> Collection for VecDeque<T> {
    type Item = T;

    fn len(&self) -> usize {
        VecDeque::len(self)
    }

    fn contains(&self, item: &T) -> bool {
        VecDeque::contains(self, item)
    }

    /// Oldest first, which is the order a queue hands them back.
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: PartialEq> CollectionInsert for VecDeque<T> {
    /// Joins the back of the queue, behind everything already waiting.
    fn insert(&mut self, item: T) -> bool {
        self.push_back(item);
        true
    }
}

impl<T: PartialEq> CollectionRemove for VecDeque<T> {
    fn remove(&mut self, item: &T) -> bool {
        let Some(position) = self.iter().position(|held| held == item) else {
            return false;
        };

        VecDeque::remove(self, position);
        true
    }

    fn clear(&mut self) {
        VecDeque::clear(self);
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        VecDeque::retain(self, keep);
    }
}

impl<T: PartialEq> Sequence for VecDeque<T> {
    fn get(&self, index: usize) -> Option<&T> {
        VecDeque::get(self, index)
    }

    fn index_of(&self, item: &T) -> Option<usize> {
        self.iter().position(|held| held == item)
    }
}

impl<T: PartialEq> SequenceMut for VecDeque<T> {
    fn remove_at(&mut self, index: usize) -> Option<T> {
        VecDeque::remove(self, index)
    }
}

impl<T: PartialEq> InsertAt for VecDeque<T> {
    fn insert_at(&mut self, index: usize, item: T) -> bool {
        if index > self.len() {
            return false;
        }

        VecDeque::insert(self, index, item);
        true
    }
}

impl<T: PartialEq> Reorder for VecDeque<T> {
    fn sort_by<F: FnMut(&T, &T) -> Ordering>(&mut self, compare: F) {
        self.make_contiguous().sort_by(compare);
    }

    fn reverse(&mut self) {
        self.make_contiguous().reverse();
    }

    fn shuffle(&mut self, random: &mut Random) {
        sampling::shuffle(self.make_contiguous(), random);
    }
}

/// Indexed directly, for the same reasons as `Vec`.
impl<T: PartialEq> Choose for VecDeque<T> {
    fn choose(&self, random: &mut Random) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        VecDeque::get(self, random.uniform_index(self.len()))
    }

    fn choose_multiple(&self, random: &mut Random, amount: usize) -> Vec<&T> {
        sampling::uniform_indices(self.len(), amount, Some(random))
            .into_iter()
            .map(|index| &self[index])
            .collect()
    }
}
