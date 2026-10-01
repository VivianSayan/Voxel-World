//! A scheduler where each value may be pending at only one step.

use crate::structures::hashing::FastHashMap;
use crate::structures::indices::Gate;
use crate::structures::traits::{Element, Kinded, Pending, RangeQuery};
use std::collections::BTreeMap;
use std::ops::RangeBounds;

/// Unique work scheduled by absolute step. Scheduling an existing value moves
/// it rather than creating a second copy.
#[derive(Clone, Debug)]
pub struct UniqueScheduler<T> {
    now: u64,
    pending: BTreeMap<u64, Vec<T>>,
    due: FastHashMap<T, u64>,
}

impl<T> Default for UniqueScheduler<T> {
    fn default() -> Self {
        Self {
            now: 0,
            pending: BTreeMap::new(),
            due: FastHashMap::default(),
        }
    }
}

impl<T: Element> UniqueScheduler<T> {
    /// Creates an empty scheduler at step zero.
    pub fn new() -> Self {
        Self::default()
    }
    /// Current step.
    pub const fn now(&self) -> u64 {
        self.now
    }
    /// Number of pending values.
    pub fn len(&self) -> usize {
        self.due.len()
    }
    /// Whether no values are pending.
    pub fn is_empty(&self) -> bool {
        self.due.is_empty()
    }
    /// Exact step for a value.
    pub fn step_of(&self, value: &T) -> Option<u64> {
        self.due.get(value).copied()
    }
    /// Schedules after a delay, returning the previous step if the value moved.
    pub fn schedule(&mut self, delay: u64, value: T) -> Option<u64> {
        self.schedule_at(self.now.saturating_add(delay), value)
    }
    /// Schedules on an exact step, moving an existing value.
    pub fn schedule_at(&mut self, step: u64, value: T) -> Option<u64> {
        let old = self.cancel(&value);
        self.pending.entry(step).or_default().push(value.clone());
        self.due.insert(value, step);
        old
    }
    /// Cancels a value and returns its former step.
    pub fn cancel(&mut self, value: &T) -> Option<u64> {
        let step = self.due.remove(value)?;
        if let Some(values) = self.pending.get_mut(&step) {
            values.retain(|held| held != value);
            if values.is_empty() {
                self.pending.remove(&step);
            }
        }
        Some(step)
    }
    /// Advances and returns all values now due, oldest step first.
    pub fn advance_by(&mut self, steps: u64) -> Vec<T> {
        self.now = self.now.saturating_add(steps);
        let future = match self.now.checked_add(1) {
            Some(next) => self.pending.split_off(&next),
            None => BTreeMap::new(),
        };
        let due = std::mem::replace(&mut self.pending, future);
        let values: Vec<T> = due.into_values().flatten().collect();
        for value in &values {
            self.due.remove(value);
        }
        values
    }
    /// Advances one step.
    pub fn advance(&mut self) -> Vec<T> {
        self.advance_by(1)
    }
    /// Pending values in step order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (u64, &T)> {
        self.pending
            .iter()
            .flat_map(|(step, values)| values.iter().map(move |value| (*step, value)))
    }
}

impl<T: Element> RangeQuery for UniqueScheduler<T> {
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

/// Ready means due at or before the current step.
impl<T: Element> Pending for UniqueScheduler<T> {
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

    /// Advancing by nothing drains what is due without moving the clock, which
    /// is what this scheduler has in place of a separate `take_due`.
    fn take_ready(&mut self) -> Vec<T> {
        self.advance_by(0)
    }
}

impl<T: Element> UniqueScheduler<T> {
    /// Takes one step and returns what falls due and the gate allows.
    ///
    /// Anything the gate refuses is dropped, as in
    /// [`Scheduler::advance_with`](super::Scheduler::advance_with),
    /// leaving the value with nothing pending.
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
