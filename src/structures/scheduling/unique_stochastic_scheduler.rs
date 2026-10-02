//! Recurring drawn work where each value is pending at only one step.

use super::cadence::Cadence;
use super::stochastic_scheduler::{CadenceId, Firing, StochasticScheduler};
use crate::random::{Seed, SeedCursor, StochasticStream};
use crate::structures::hashing::FastHashMap;
use crate::structures::indices::Gate;
use crate::structures::traits::{
    Collection, DeterministicOrder, Element, Kinded, Pending, RangeQuery, UniqueCollection,
};
use std::fmt;
use std::ops::RangeBounds;

/// Recurring work drawn from a [`Cadence`], with one pending occurrence per
/// value.
///
/// # How it differs from [`StochasticScheduler`]
///
/// Scheduling a value that is already pending **moves** it rather than adding a
/// second occurrence, so a value cannot pile up behind itself. That is what a
/// caller wants when the value stands for a thing rather than a job: a plot of
/// land has one next growth, not four, however many times it was registered.
///
/// Keeping that promise costs an index from value to the step it is due on,
/// which also makes [`UniqueStochasticScheduler::due_at`] and cancellation
/// direct lookups rather than searches.
///
/// Everything else is [`StochasticScheduler`]: cadences are registered once and
/// shared, entries re-arm themselves from a drawn wait, placement evens the
/// load within the cadence's spread, and a backlog is handled by the cadence's
/// [`OnBacklog`](super::OnBacklog).
///
/// # Type parameters
///
/// - `T` is what is scheduled, an [`Element`] so it can be indexed and
///   cancelled by value.
/// - `R` is where each entry's randomness comes from; see
///   [`StochasticStream`].
#[derive(Clone)]
pub struct UniqueStochasticScheduler<T, R = SeedCursor> {
    inner: StochasticScheduler<T, R>,
    due: FastHashMap<T, u64>,
}

impl<T: Element, R: StochasticStream> UniqueStochasticScheduler<T, R> {
    /// An empty scheduler whose entries derive their randomness from `domain`.
    pub fn new(domain: Seed) -> Self {
        Self {
            inner: StochasticScheduler::new(domain),
            due: FastHashMap::default(),
        }
    }

    /// An empty scheduler whose clock starts at `start`.
    pub fn starting_at(domain: Seed, start: u64) -> Self {
        Self {
            inner: StochasticScheduler::starting_at(domain, start),
            due: FastHashMap::default(),
        }
    }

    /// Registers a cadence and returns the id entries refer to it by.
    pub fn register(&mut self, cadence: Cadence) -> CadenceId {
        self.inner.register(cadence)
    }

    /// A registered cadence, or `None` for an id this scheduler did not issue.
    pub fn cadence(&self, id: CadenceId) -> Option<&Cadence> {
        self.inner.cadence(id)
    }

    /// Schedules a value, replacing whatever it had pending.
    ///
    /// Returns `false` for a cadence this scheduler did not issue, in which
    /// case nothing changes. A value already pending is re-drawn rather than duplicated, so
    /// this is also how a value's cadence is changed.
    pub fn insert(&mut self, value: T, cadence: CadenceId) -> bool {
        self.cancel(&value);

        let Some(due) = self.inner.insert(value.clone(), cadence) else {
            return false;
        };

        self.due.insert(value, due);

        true
    }

    /// Schedules a value with randomness the caller supplies, replacing
    /// whatever it had pending.
    pub fn insert_with(&mut self, value: T, cadence: CadenceId, source: R) -> bool {
        self.cancel(&value);

        let Some(due) = self.inner.insert_with(value.clone(), cadence, source) else {
            return false;
        };

        self.due.insert(value, due);

        true
    }

    /// The step a value is due on, or `None` when it is not pending.
    pub fn due_at(&self, value: &T) -> Option<u64> {
        self.due.get(value).copied()
    }

    /// Whether a value is pending.
    pub fn contains(&self, value: &T) -> bool {
        self.due.contains_key(value)
    }

    /// Removes a value and returns whether it was pending.
    ///
    /// The index says which step to look at, so this costs the length of that
    /// step rather than a walk over the schedule.
    pub fn cancel(&mut self, value: &T) -> bool {
        let Some(step) = self.due.remove(value) else {
            return false;
        };

        self.inner.cancel_at(step, value);

        true
    }

    /// The step the clock reads.
    pub const fn now(&self) -> u64 {
        self.inner.now()
    }

    /// How many values are pending.
    pub fn len(&self) -> usize {
        self.due.len()
    }

    /// Whether nothing is pending.
    pub fn is_empty(&self) -> bool {
        self.due.is_empty()
    }

    /// The next step anything is due, or `None` when nothing is pending.
    pub fn next_due(&self) -> Option<u64> {
        self.inner.next_due()
    }

    /// How many values are due at one step.
    pub fn len_at(&self, step: u64) -> usize {
        self.inner.len_at(step)
    }

    /// The values due at one step.
    pub fn at(&self, step: u64) -> impl Iterator<Item = &T> {
        self.inner.at(step)
    }

    /// Every pending value, earliest step first.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter()
    }

    /// Takes one step and returns what falls due, each value re-armed.
    pub fn advance(&mut self) -> Vec<Firing<T>> {
        self.advance_by(1)
    }

    /// Takes `steps` steps at once and returns what falls due across all of
    /// them.
    pub fn advance_by(&mut self, steps: u64) -> Vec<Firing<T>> {
        let fired: Vec<Firing<T>> = self.inner.advance_by(steps);

        self.follow(&fired);

        fired
    }

    /// Everything due at or before the current step, without moving the clock.
    pub fn take_due(&mut self) -> Vec<Firing<T>> {
        let fired: Vec<Firing<T>> = self.inner.take_due();

        self.follow(&fired);

        fired
    }

    /// Removes everything, keeping the registered cadences and the clock.
    pub fn clear(&mut self) {
        self.inner.clear();
        self.due.clear();
    }

    /// Moves the index on to where each entry was re-armed to.
    ///
    /// Every firing carries the step its entry went to, so this is a write per
    /// firing and never a search. A value that fired under
    /// [`OnBacklog::FireAll`](super::OnBacklog) is reported once per owed
    /// occurrence; each of those firings names the same step, so repeating the
    /// write is harmless.
    fn follow(&mut self, fired: &[Firing<T>]) {
        for firing in fired {
            self.due.insert(firing.value.clone(), firing.next);
        }
    }
}

impl<T: Element, R: StochasticStream> Collection for UniqueStochasticScheduler<T, R> {
    type Item = T;

    fn len(&self) -> usize {
        self.due.len()
    }

    fn contains(&self, item: &T) -> bool {
        UniqueStochasticScheduler::contains(self, item)
    }

    /// Earliest step first.
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: Element, R: StochasticStream> UniqueCollection for UniqueStochasticScheduler<T, R> {}

/// Ordered by the step work is due on, as [`StochasticScheduler`] is.
impl<T: Element, R: StochasticStream> RangeQuery for UniqueStochasticScheduler<T, R> {
    type Key = u64;
    type Item<'a>
        = (u64, &'a T)
    where
        Self: 'a;

    fn range<'a, B: RangeBounds<u64>>(
        &'a self,
        range: B,
    ) -> impl DoubleEndedIterator<Item = Self::Item<'a>> {
        self.inner.range(range)
    }
}

impl<T: Element, R: StochasticStream> DeterministicOrder for UniqueStochasticScheduler<T, R> {}

impl<T: Element + fmt::Debug, R: StochasticStream> fmt::Debug for UniqueStochasticScheduler<T, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, formatter)
    }
}

impl<T: Element, R: StochasticStream> fmt::Display for UniqueStochasticScheduler<T, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

/// As [`StochasticScheduler`]'s, with the index kept in step.
impl<T: Element, R: StochasticStream> Pending for UniqueStochasticScheduler<T, R> {
    type Ready = Firing<T>;

    fn pending_len(&self) -> usize {
        self.due.len()
    }

    fn ready_len(&self) -> usize {
        self.inner.ready_len()
    }

    fn take_ready(&mut self) -> Vec<Firing<T>> {
        self.take_due()
    }
}

impl<T: Element, R: StochasticStream> UniqueStochasticScheduler<T, R> {
    /// Takes one step and returns what falls due and the gate allows.
    ///
    /// As [`StochasticScheduler::advance_with`]: a refused entry has already
    /// re-armed itself and comes round again, and the index follows it.
    pub fn advance_with<P: Element, V: Element + Kinded + PartialOrd>(
        &mut self,
        gate: &Gate<T, P, V>,
    ) -> Vec<Firing<T>> {
        self.advance_by_with(1, gate)
    }

    /// Takes `steps` steps at once and returns what falls due and the gate
    /// allows, re-arming the rest.
    pub fn advance_by_with<P: Element, V: Element + Kinded + PartialOrd>(
        &mut self,
        steps: u64,
        gate: &Gate<T, P, V>,
    ) -> Vec<Firing<T>> {
        self.advance_by(steps)
            .into_iter()
            .filter(|firing| gate.allows(&firing.value))
            .collect()
    }
}
