//! Work put aside for later, taken out a step at a time.
//!
//! Put something in with a delay and it stays out of the way until that many
//! steps have been taken, then comes back out with everything else due at the
//! same moment. The obvious use is a tick scheduler, but the step is whatever
//! the caller advances it by: a turn, a generation pass, a retry round.
//!
//! ```ignore
//! let mut schedule = Scheduler::new();
//! schedule.schedule(9, "mine");
//!
//! for _ in 0..8 { assert!(schedule.advance().is_empty()); }
//! assert_eq!(schedule.advance(), vec!["mine"]);
//! ```
//!
//! # Storage
//!
//! Nothing is kept for a step that has nothing in it. Entries live in a
//! `BTreeMap` from the step they are due on to the values due then, so the map
//! holds exactly as many entries as there are distinct occupied steps, however
//! far apart they are. Scheduling something a million steps out costs one
//! entry, not a million, and the step counter is a plain integer that no amount
//! of advancing allocates against.
//!
//! That makes the empty case genuinely free rather than merely cheap:
//! [`Scheduler::advance`] over a stretch with nothing due never touches the map
//! at all. Where a whole stretch can be skipped, [`Scheduler::skip_to_due`]
//! jumps straight to the next occupied step without walking the ones between.
//!
//! The cost of that choice is a `Vec` per occupied step, which is three words
//! of overhead even when only one value is due. It is the right trade while
//! occupied steps are sparse and the values on them are few; a dense wheel
//! would win only if nearly every step were occupied.

use crate::structures::indices::Gate;
use crate::structures::traits::{Element, Kinded, Pending, RangeQuery};
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ops::RangeBounds;

#[derive(Clone, Debug)]
/// Values of type `T` held until the step they are due on.
pub struct Scheduler<T> {
    /// How many steps have been taken. Absolute, so that advancing never has
    /// to touch what is already scheduled.
    now: u64,

    /// Due step to the values due then. Only occupied steps appear.
    pending: BTreeMap<u64, Vec<T>>,

    /// Kept alongside so that `len` does not have to walk every step.
    len: usize,
}

impl<T> Default for Scheduler<T> {
    fn default() -> Self {
        Self {
            now: 0,
            pending: BTreeMap::new(),
            len: 0,
        }
    }
}

impl<T> Scheduler<T> {
    /// Creates an empty scheduler standing at step 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty scheduler standing at `start`, for resuming a world
    /// part way through rather than replaying it.
    pub fn starting_at(start: u64) -> Self {
        Self {
            now: start,
            pending: BTreeMap::new(),
            len: 0,
        }
    }

    /// Puts a value in, due after `delay` more steps.
    ///
    /// A delay of 1 comes out of the next [`Scheduler::advance`]. A delay of 0
    /// is already due and comes out of the next advance as well, rather than
    /// being stranded behind the step counter.
    pub fn schedule(&mut self, delay: u64, value: T) {
        let due: u64 = self.now.saturating_add(delay);

        self.schedule_at(due, value);
    }

    /// Puts a value in, due on an exact step.
    ///
    /// A step that has already passed is not an error; the value comes out of
    /// the next advance, along with anything else overdue.
    pub fn schedule_at(&mut self, step: u64, value: T) {
        self.pending.entry(step).or_default().push(value);
        self.len += 1;
    }

    /// Puts several values in at once, all due after the same delay.
    ///
    /// Nothing is stored for a step that turns out to have no values, so an
    /// empty call leaves no empty entry behind to be walked over later.
    pub fn schedule_all(&mut self, delay: u64, values: impl IntoIterator<Item = T>) {
        let due: u64 = self.now.saturating_add(delay);

        match self.pending.entry(due) {
            Entry::Occupied(mut occupied) => {
                let before: usize = occupied.get().len();

                occupied.get_mut().extend(values);
                self.len += occupied.get().len() - before;
            }
            Entry::Vacant(vacant) => {
                let held: Vec<T> = values.into_iter().collect();

                self.len += held.len();

                if !held.is_empty() {
                    vacant.insert(held);
                }
            }
        }
    }

    /// Takes one step and returns everything that falls due, oldest first.
    ///
    /// Anything overdue comes out too, so nothing scheduled into the past can
    /// be left behind. An empty step costs one integer addition and one lookup
    /// at the front of the map.
    pub fn advance(&mut self) -> Vec<T> {
        self.now = self.now.saturating_add(1);

        self.take_due()
    }

    /// Takes `steps` steps at once and returns everything that falls due
    /// across all of them, in step order.
    ///
    /// The same result as calling [`Scheduler::advance`] that many times and
    /// joining the results, but it walks only the occupied steps in between
    /// rather than every step.
    pub fn advance_by(&mut self, steps: u64) -> Vec<T> {
        self.now = self.now.saturating_add(steps);

        self.take_due()
    }

    /// Everything due at or before the current step, oldest step first.
    ///
    /// Useful on its own after [`Scheduler::schedule_at`] has put something in
    /// the past, or to drain what is owed without taking another step.
    ///
    /// The map is split at the first step still in the future: that part
    /// becomes the new map and what is left is drained in step order. Steps
    /// that come out are removed with their entries, so nothing emptied stays
    /// behind.
    pub fn take_due(&mut self) -> Vec<T> {
        if !self.has_due() {
            return Vec::new();
        }

        let future: BTreeMap<u64, Vec<T>> = match self.now.checked_add(1) {
            Some(next) => self.pending.split_off(&next),
            None => BTreeMap::new(),
        };
        let due: BTreeMap<u64, Vec<T>> = std::mem::replace(&mut self.pending, future);

        let taken: Vec<T> = due.into_values().flatten().collect();

        self.len -= taken.len();
        taken
    }

    /// Jumps to the next step that has anything on it, taking those values.
    ///
    /// Returns the step landed on and what was due there, or `None` when
    /// nothing is scheduled at all. Steps with nothing on them are skipped
    /// rather than walked, so an idle stretch of any length costs the same.
    pub fn skip_to_due(&mut self) -> Option<(u64, Vec<T>)> {
        let next: u64 = self.next_due()?;

        self.now = self.now.max(next);

        Some((next, self.take_due()))
    }

    /// The step the scheduler is standing on.
    pub fn now(&self) -> u64 {
        self.now
    }

    /// The next step with anything on it, which may be at or before `now` when
    /// something is overdue.
    pub fn next_due(&self) -> Option<u64> {
        self.pending.keys().next().copied()
    }

    /// How many steps until the next thing comes out, or `None` when nothing
    /// is scheduled. Zero when something is already due.
    pub fn steps_until_due(&self) -> Option<u64> {
        Some(self.next_due()?.saturating_sub(self.now))
    }

    /// Whether anything is due at or before the current step.
    pub fn has_due(&self) -> bool {
        self.next_due().is_some_and(|step| step <= self.now)
    }

    /// The values due next, without taking them or moving.
    pub fn peek(&self) -> Option<(u64, &[T])> {
        self.pending
            .iter()
            .next()
            .map(|(step, values)| (*step, values.as_slice()))
    }

    /// The values waiting on one exact step.
    pub fn at(&self, step: u64) -> &[T] {
        self.pending.get(&step).map_or(&[], Vec::as_slice)
    }

    /// Takes everything waiting on one exact step, wherever it is in time.
    pub fn take_at(&mut self, step: u64) -> Vec<T> {
        let taken: Vec<T> = self.pending.remove(&step).unwrap_or_default();

        self.len -= taken.len();
        taken
    }

    /// How many values are waiting, across every step.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no pending values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many distinct steps hold anything, which is what the map costs.
    pub fn occupied_steps(&self) -> usize {
        self.pending.len()
    }

    /// Throws away everything scheduled, leaving the step counter where it is.
    pub fn clear(&mut self) {
        self.pending.clear();
        self.len = 0;
    }

    /// Throws away everything and returns to step 0.
    pub fn reset(&mut self) {
        self.clear();
        self.now = 0;
    }

    /// Every value waiting, with the step it is due on, in step order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &T)> {
        self.pending
            .iter()
            .flat_map(|(step, values)| values.iter().map(move |value| (*step, value)))
    }

    /// Every occupied step and what is on it, in step order.
    pub fn steps(&self) -> impl Iterator<Item = (u64, &[T])> {
        self.pending
            .iter()
            .map(|(step, values)| (*step, values.as_slice()))
    }

    /// Keeps only the values a predicate accepts, told which step each is on.
    ///
    /// A step left with nothing on it is removed rather than kept as an empty
    /// entry, so dropping work never leaves the map holding gaps.
    pub fn retain(&mut self, mut keep: impl FnMut(u64, &T) -> bool) {
        let mut remaining: usize = 0;

        self.pending.retain(|step, values| {
            values.retain(|value| keep(*step, value));
            remaining += values.len();

            !values.is_empty()
        });

        self.len = remaining;
    }

    /// Moves everything on one step to another, merging with whatever is
    /// already there. Does nothing when the step is empty.
    pub fn reschedule(&mut self, from: u64, to: u64) {
        let Some(moved) = self.pending.remove(&from) else {
            return;
        };

        if moved.is_empty() {
            return;
        }

        self.pending.entry(to).or_default().extend(moved);
    }
}

impl<T: PartialEq> Scheduler<T> {
    /// Whether a value is waiting on any step.
    pub fn contains(&self, value: &T) -> bool {
        self.pending.values().any(|held| held.contains(value))
    }

    /// The step a value is waiting on, if any. The earliest, when it is
    /// scheduled more than once.
    pub fn step_of(&self, value: &T) -> Option<u64> {
        self.pending
            .iter()
            .find(|(_, held)| held.contains(value))
            .map(|(step, _)| *step)
    }

    /// Removes every copy of a value, wherever it is waiting. Returns how many
    /// were taken out.
    pub fn cancel(&mut self, value: &T) -> usize {
        let before: usize = self.len;

        self.retain(|_, held| held != value);

        before - self.len
    }
}

impl<T> RangeQuery for Scheduler<T> {
    type Key = u64;
    type Item<'a>
        = (u64, &'a T)
    where
        Self: 'a;

    fn range<'a, R: RangeBounds<u64>>(
        &'a self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = Self::Item<'a>> {
        self.pending
            .range(range)
            .flat_map(|(step, values)| values.iter().map(move |value| (*step, value)))
    }
}

impl<T> IntoIterator for Scheduler<T> {
    type Item = (u64, T);
    type IntoIter = std::vec::IntoIter<(u64, T)>;

    /// In step order.
    fn into_iter(self) -> Self::IntoIter {
        self.pending
            .into_iter()
            .flat_map(|(step, values)| values.into_iter().map(move |value| (step, value)))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<T> Extend<(u64, T)> for Scheduler<T> {
    /// Each value is scheduled at its exact step, not after a delay.
    fn extend<I: IntoIterator<Item = (u64, T)>>(&mut self, values: I) {
        for (step, value) in values {
            self.schedule_at(step, value);
        }
    }
}

impl<T> FromIterator<(u64, T)> for Scheduler<T> {
    /// Each value is scheduled at its exact step, starting from step 0.
    fn from_iter<I: IntoIterator<Item = (u64, T)>>(values: I) -> Self {
        let mut schedule: Self = Self::new();

        schedule.extend(values);
        schedule
    }
}

/// Ready means due at or before the current step, which is what
/// [`Scheduler::take_due`] hands over.
impl<T> Pending for Scheduler<T> {
    type Ready = T;

    fn pending_len(&self) -> usize {
        self.len()
    }

    fn ready_len(&self) -> usize {
        self.pending
            .range(..=self.now)
            .map(|(_, values)| values.len())
            .sum()
    }

    fn has_ready(&self) -> bool {
        self.has_due()
    }

    fn take_ready(&mut self) -> Vec<T> {
        self.take_due()
    }
}

impl<T: Element> Scheduler<T> {
    /// Takes one step and returns what falls due and the gate allows.
    ///
    /// Anything the gate refuses is **dropped**: this scheduler fires a value
    /// once at a step the caller named, so a refused occurrence has no next
    /// occurrence to wait for. Schedule it again if it should come back.
    pub fn advance_with<P: Element, V: Element + Kinded + PartialOrd>(
        &mut self,
        gate: &Gate<T, P, V>,
    ) -> Vec<T> {
        self.advance_by_with(1, gate)
    }

    /// Takes `steps` steps at once and returns what falls due and the gate
    /// allows, dropping the rest.
    pub fn advance_by_with<P: Element, V: Element + Kinded + PartialOrd>(
        &mut self,
        steps: u64,
        gate: &Gate<T, P, V>,
    ) -> Vec<T> {
        self.advance_by(steps)
            .into_iter()
            .filter(|value| gate.allows(value))
            .collect()
    }
}
