//! World time laid over the generic step scheduler.
//!
//! Lives here rather than with the time quantities because it *is* a scheduler —
//! it holds a [`Scheduler`](super::Scheduler) and does nothing but keep the types
//! honest at its edges. Keeping it beside the quantities made `time` depend on
//! `structures`, which was a dependency in the wrong direction: a unit of time
//! should not need to know that collections exist.

use crate::units::time::{Tick, TickDuration};

/// World-time adapter over the generic step scheduler.
///
/// Holds a [`Scheduler`](crate::structures::collections::Scheduler) and does
/// nothing but keep the types honest at the edges: deadlines are [`Tick`]s,
/// delays are [`TickDuration`]s, and what comes back out is whatever was put
/// in. Everything about the storage, including the sparse map of pending steps
/// and the cost of an empty step, is documented there.
#[derive(Clone, Debug)]
pub struct TickScheduler<T>(crate::structures::collections::Scheduler<T>);

/// A scheduler starting at [`Tick::ORIGIN`].
impl<T> Default for TickScheduler<T> {
    fn default() -> Self {
        Self::starting_at(Tick::ORIGIN)
    }
}

impl<T> TickScheduler<T> {
    /// An empty scheduler whose clock reads `now`.
    pub fn starting_at(now: Tick) -> Self {
        Self(crate::structures::collections::Scheduler::starting_at(
            now.count(),
        ))
    }

    /// The moment the scheduler's clock currently reads.
    pub fn now(&self) -> Tick {
        Tick::new(self.0.now())
    }

    /// Puts an item in, due `delay` ticks from now.
    pub fn schedule(&mut self, delay: TickDuration, item: T) {
        self.0.schedule(delay.count(), item);
    }

    /// Puts an item in, due at a given moment. A deadline already past comes
    /// out at the next take.
    pub fn schedule_at(&mut self, deadline: Tick, item: T) {
        self.0.schedule_at(deadline.count(), item);
    }

    /// Advances one tick and returns everything that has fallen due, oldest
    /// first.
    pub fn advance(&mut self) -> Vec<T> {
        self.0.advance()
    }

    /// Advances several ticks at once, returning everything due across all of
    /// them in order, without walking the ticks that hold nothing.
    pub fn advance_by(&mut self, duration: TickDuration) -> Vec<T> {
        self.0.advance_by(duration.count())
    }

    /// Everything due at or before the current moment, without advancing.
    pub fn take_due(&mut self) -> Vec<T> {
        self.0.take_due()
    }

    /// The next moment anything is due, or `None` when nothing is pending.
    pub fn next_due(&self) -> Option<Tick> {
        self.0.next_due().map(Tick::new)
    }

    /// Jumps the clock straight to the next moment anything is due and takes
    /// it, or `None` when nothing is pending.
    pub fn skip_to_due(&mut self) -> Option<(Tick, Vec<T>)> {
        self.0.skip_to_due().map(|(_, values)| (self.now(), values))
    }

    /// How many items are waiting, across all moments.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether nothing is waiting.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
