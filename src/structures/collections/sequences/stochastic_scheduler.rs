//! Recurring work whose next occurrence is drawn rather than fixed, spread so
//! that no one step carries the crowd.

use super::cadence::{Cadence, OnBacklog};
use crate::random::{StochasticStream, Seed, SeedCursor, UniformU64};
use crate::structures::hashing::unordered_hash;
use crate::structures::indices::Gate;
use crate::structures::traits::{
    Collection, DeterministicOrder, Element, Kinded, Pending, RangeQuery,
};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Bound, RangeBounds};

/// Which registered cadence an entry follows.
///
/// An id carries the scheduler that issued it as well as the position it was
/// registered at, so one scheduler will not accept another's. Without that tag
/// the first cadence of every scheduler would be id zero, and handing the wrong
/// one over would silently bind work to whatever recurrence happened to sit at
/// that position — a job meant to fire every other step scheduled once a
/// century, with nothing reporting it.
///
/// The tag comes from the scheduler's domain seed, so it replays and survives a
/// clone: ids keep working on a scheduler cloned from the one that issued them.
/// Two schedulers built from the *same* domain seed therefore share a tag and
/// cannot be told apart. That is already a mistake for a different reason —
/// they would derive identical randomness for identical values — so the
/// remaining gap is one nothing correct can be in.
///
/// There is deliberately no `Default`: an id that named position zero of no
/// particular scheduler is exactly the hazard the tag exists to remove.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CadenceId {
    /// Which scheduler issued this id.
    origin: u32,
    /// Where the cadence sits in the order they were registered.
    index: u32,
}

impl CadenceId {
    /// The cadence's position in the order they were registered.
    pub const fn index(self) -> usize {
        self.index as usize
    }
}

/// One occurrence coming due.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Firing<T> {
    /// What was scheduled.
    pub value: T,
    /// How many occurrences this firing stands for.
    ///
    /// One in the ordinary case. More only under
    /// [`OnBacklog::Coalesce`], where the clock ran
    /// past several owed occurrences and they arrive as one.
    pub occurrences: u32,
    /// The step this entry is next due on, having already been re-armed.
    ///
    /// Under [`OnBacklog::Cap`] with occurrences still owed this is the
    /// following step, where the remainder comes out, rather than a freshly
    /// drawn wait. It saves a holder that indexes entries by due step from
    /// having to search for where each one went.
    pub next: u64,
}

/// One scheduled entry.
#[derive(Clone, Debug)]
struct Entry<T, R> {
    value: T,
    cadence: CadenceId,
    source: R,
    /// Occurrences owed but not yet delivered, under
    /// [`OnBacklog::Cap`](super::OnBacklog::Cap).
    owed: u32,
}

/// Recurring work that re-arms itself with a drawn wait, and is placed to keep
/// the load across steps even.
///
/// # Type parameters
///
/// - `T` is what is scheduled, usually an id. It is [`Element`] so that entries
///   can be found and cancelled by value, and it is cloned when it fires.
/// - `R` is where each entry's randomness comes from: a [`SeedCursor`], which
///   is compact and replayable from the world seed, or a
///   [`Random`](crate::random::Random), which is a stream of its own, or
///   [`EventRandom`](crate::random::EventRandom) to mix the two. See
///   [`StochasticStream`].
///
/// # What it does
///
/// Every entry follows a [`Cadence`]: when it fires, it draws its next wait
/// from that cadence's mean time to happen and schedules itself again. The
/// caller never re-arms anything.
///
/// Placement then evens the load. With a cadence spread above zero, two
/// candidate steps are drawn within that many steps of where the wait fell and
/// the emptier one is taken. Sampling two and keeping the better is what turns
/// a distribution with occasional pile-ups into one without: it costs a draw
/// and two lookups, and it cuts the tallest step from roughly `log n / log log
/// n` entries to about `log log n`.
///
/// # Relationship to the other schedulers
///
/// [`Scheduler`](super::Scheduler) fires once at a step the caller names.
/// [`Rota`](crate::structures::collections::Rota) recurs on an exact period
/// with perfectly even turns. This one recurs on a random period and evens the
/// turns as well as a random period allows. A value may be scheduled more than
/// once; [`UniqueStochasticScheduler`](super::UniqueStochasticScheduler) keeps
/// one pending occurrence per value.
///
/// ```ignore
/// let mut events = StochasticScheduler::new(world.child("growth"));
/// let slow = events.register(Cadence::mtth(600).spread(4));
///
/// events.insert(plot, slow);
///
/// for firing in events.advance() {
///     grow(firing.value, firing.occurrences);
/// }
/// ```
#[derive(Clone)]
pub struct StochasticScheduler<T, R = SeedCursor> {
    now: u64,
    pending: BTreeMap<u64, Vec<Entry<T, R>>>,
    cadences: Vec<Cadence>,
    domain: Seed,
    len: usize,
    /// Identifies this scheduler in the ids it hands out, so that another
    /// scheduler's are recognisable rather than merely in range.
    origin: u32,
    /// Distinguishes entries that would otherwise derive the same randomness,
    /// which is every repeat of one value.
    serial: u64,
}

impl<T: Element, R: StochasticStream> StochasticScheduler<T, R> {
    /// An empty scheduler whose entries derive their randomness from `domain`.
    ///
    /// Derive that seed once, from the world seed, and let the scheduler branch
    /// it per entry: nothing here hashes a string after this call.
    pub fn new(domain: Seed) -> Self {
        Self {
            now: 0,
            pending: BTreeMap::new(),
            cadences: Vec::new(),
            domain,
            len: 0,
            origin: Self::origin_of(domain),
            serial: 0,
        }
    }

    /// An empty scheduler whose clock starts at `start`.
    pub fn starting_at(domain: Seed, start: u64) -> Self {
        Self {
            now: start,
            ..Self::new(domain)
        }
    }

    /// Registers a cadence and returns the id entries refer to it by.
    ///
    /// Register once and share the id: the cadence owns the sampler its mean
    /// needs, including the table a rare chance builds, so a cadence per entry
    /// would build that table per entry.
    pub fn register(&mut self, cadence: Cadence) -> CadenceId {
        self.cadences.push(cadence);

        CadenceId {
            origin: self.origin,
            index: self.cadences.len() as u32 - 1,
        }
    }

    /// The tag this scheduler stamps its ids with, folded from its domain.
    fn origin_of(domain: Seed) -> u32 {
        let raw: u64 = domain.as_u64();

        (raw ^ (raw >> 32)) as u32
    }

    /// The cadence an id names, if this scheduler issued it.
    fn policy(&self, id: CadenceId) -> Option<&Cadence> {
        if id.origin != self.origin {
            return None;
        }

        self.cadences.get(id.index())
    }

    /// A registered cadence, or `None` for an id this scheduler did not issue.
    ///
    /// An id carries its issuer, so this is `None` for another scheduler's id
    /// whether or not that position happens to be in range here.
    pub fn cadence(&self, id: CadenceId) -> Option<&Cadence> {
        self.policy(id)
    }

    /// How many cadences are registered.
    pub fn cadence_count(&self) -> usize {
        self.cadences.len()
    }

    /// Schedules a value on a cadence, deriving its randomness from the
    /// scheduler's domain.
    ///
    /// The derivation takes in the cadence, the value itself and a serial
    /// number, so two copies of one value get different waits and the whole
    /// schedule is a function of the seed rather than of when anything was
    /// inserted.
    ///
    /// Returns the step the value is due on, or `None` for a cadence this
    /// scheduler did not issue, in which case nothing is scheduled.
    pub fn insert(&mut self, value: T, cadence: CadenceId) -> Option<u64> {
        let serial: u64 = self.serial;
        let source: R = R::from_seed(
            self.domain
                .index(cadence.index() as u64)
                .index(unordered_hash([&value]))
                .index(serial),
        );

        self.insert_with(value, cadence, source)
    }

    /// Schedules a value on a cadence with randomness the caller supplies.
    ///
    /// For an entry that should follow a particular stream or seed rather than
    /// one derived here. Returns the step the value is due on, or `None` for a
    /// cadence this scheduler did not issue.
    pub fn insert_with(&mut self, value: T, cadence: CadenceId, mut source: R) -> Option<u64> {
        let policy: Cadence = self.policy(cadence).cloned()?;

        self.serial += 1;

        let due: u64 = self.place(&mut source, &policy);

        self.pending.entry(due).or_default().push(Entry {
            value,
            cadence,
            source,
            owed: 0,
        });
        self.len += 1;

        Some(due)
    }

    /// The step the clock reads.
    pub const fn now(&self) -> u64 {
        self.now
    }

    /// How many entries are scheduled.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing is scheduled.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The next step anything is due, or `None` when nothing is scheduled.
    pub fn next_due(&self) -> Option<u64> {
        self.pending.keys().next().copied()
    }

    /// How many entries are due at one step.
    pub fn len_at(&self, step: u64) -> usize {
        self.pending.get(&step).map_or(0, Vec::len)
    }

    /// The values due at one step, in the order they were placed there.
    pub fn at(&self, step: u64) -> impl Iterator<Item = &T> {
        self.pending
            .get(&step)
            .into_iter()
            .flatten()
            .map(|entry| &entry.value)
    }

    /// Every scheduled value, earliest step first.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.pending.values().flatten().map(|entry| &entry.value)
    }

    /// Takes one step and returns what falls due, each entry re-armed for its
    /// next occurrence.
    pub fn advance(&mut self) -> Vec<Firing<T>> {
        self.advance_by(1)
    }

    /// Takes `steps` steps at once and returns what falls due across all of
    /// them.
    ///
    /// Occurrences the clock passed are handled by each entry's cadence: see
    /// [`OnBacklog`]. The waits are drawn forward until one lands beyond the
    /// new step, which is exactly what ticking through them would have done,
    /// so a long skip costs draws rather than accuracy.
    pub fn advance_by(&mut self, steps: u64) -> Vec<Firing<T>> {
        self.now = self.now.saturating_add(steps);

        self.take_due()
    }

    /// Everything due at or before the current step, without moving the clock.
    pub fn take_due(&mut self) -> Vec<Firing<T>> {
        let mut fired: Vec<Firing<T>> = Vec::new();

        while let Some(step) = self.next_due().filter(|step| *step <= self.now) {
            let Some(entries) = self.pending.remove(&step) else {
                break;
            };

            for entry in entries {
                self.len -= 1;
                self.fire(entry, step, &mut fired);
            }
        }

        fired
    }

    /// Removes the occurrences of a value that are due at one step, and returns
    /// how many went.
    ///
    /// For a caller that already knows where the value sits, which costs the
    /// length of that one step rather than a walk over the whole schedule.
    pub fn cancel_at(&mut self, step: u64, value: &T) -> usize {
        let Some(entries) = self.pending.get_mut(&step) else {
            return 0;
        };

        let before: usize = entries.len();

        entries.retain(|entry| entry.value != *value);

        let removed: usize = before - entries.len();

        if entries.is_empty() {
            self.pending.remove(&step);
        }

        self.len -= removed;

        removed
    }

    /// Removes every occurrence of a value and returns how many went.
    ///
    /// This walks the whole schedule. Prefer [`StochasticScheduler::cancel_at`]
    /// where the step is known.
    pub fn cancel(&mut self, value: &T) -> usize {
        let mut removed: usize = 0;

        self.pending.retain(|_, entries| {
            let before: usize = entries.len();

            entries.retain(|entry| entry.value != *value);
            removed += before - entries.len();

            !entries.is_empty()
        });

        self.len -= removed;

        removed
    }

    /// Whether a value has any occurrence scheduled.
    pub fn contains(&self, value: &T) -> bool {
        self.pending
            .values()
            .flatten()
            .any(|entry| entry.value == *value)
    }

    /// Removes everything, keeping the registered cadences and the clock.
    pub fn clear(&mut self) {
        self.pending.clear();
        self.len = 0;
    }

    /// Delivers one entry's due occurrences and schedules it again.
    fn fire(&mut self, mut entry: Entry<T, R>, due: u64, fired: &mut Vec<Firing<T>>) {
        let Some(policy) = self.policy(entry.cadence).cloned() else {
            return;
        };

        // Walk the waits forward until one lands past the clock, which is what
        // ticking through the skipped steps would have done. The last draw
        // becomes the next occurrence.
        let mut owed: u32 = entry.owed.saturating_add(1);
        let mut next: u64 = due;

        loop {
            next = self.draw_next(&mut entry.source, &policy, next);

            if next > self.now {
                break;
            }

            owed = owed.saturating_add(1);
        }

        let backlog: OnBacklog = policy.backlog();
        let delivered: u32 = match backlog {
            OnBacklog::Coalesce | OnBacklog::FireAll => owed,
            OnBacklog::Cap(limit) => owed.min(limit),
        };

        entry.owed = owed - delivered;

        // Anything still owed comes out on the next step rather than waiting
        // for the cadence again. Settling this before the firings are built is
        // what lets each one report where the entry went.
        let due: u64 = if entry.owed > 0 {
            self.now.saturating_add(1)
        } else {
            next
        };

        match backlog {
            OnBacklog::Coalesce => fired.push(Firing {
                value: entry.value.clone(),
                occurrences: owed,
                next: due,
            }),
            OnBacklog::FireAll | OnBacklog::Cap(_) => {
                for _ in 0..delivered {
                    fired.push(Firing {
                        value: entry.value.clone(),
                        occurrences: 1,
                        next: due,
                    });
                }
            }
        }

        self.pending.entry(due).or_default().push(entry);
        self.len += 1;
    }

    /// Where the occurrence after the one at `from` lands, which is always at
    /// least one step later.
    ///
    /// The floor is `from`, not the clock: walking a backlog forward asks where
    /// each missed occurrence fell, and answering with the current step would
    /// collapse the whole backlog into one.
    fn draw_next(&self, source: &mut R, policy: &Cadence, from: u64) -> u64 {
        let wait: u64 = source.draw(policy.wait()).saturating_add(1);
        let base: u64 = from.saturating_add(wait);

        self.smooth(source, policy, base, from)
    }

    /// Where a fresh entry's first occurrence lands.
    fn place(&self, source: &mut R, policy: &Cadence) -> u64 {
        self.draw_next(source, policy, self.now)
    }

    /// Nudges a step within the cadence's spread towards whichever of two
    /// candidates is carrying less.
    ///
    /// Both candidates are drawn from the same symmetric window, so the mean
    /// stays where the cadence put it; taking the emptier of the two is what
    /// keeps a crowd from forming on one step. Ties go to the earlier step.
    fn smooth(&self, source: &mut R, policy: &Cadence, base: u64, after: u64) -> u64 {
        let spread: u32 = policy.allowed_spread();

        if spread == 0 {
            return base.max(after.saturating_add(1));
        }

        let window: UniformU64 = UniformU64::new(0, u64::from(spread) * 2)
            .expect("the window is never empty when the spread is above zero");

        let first: u64 = Self::offset(base, source.draw(&window), spread, after);
        let second: u64 = Self::offset(base, source.draw(&window), spread, after);

        let (earlier, later): (u64, u64) = if first <= second {
            (first, second)
        } else {
            (second, first)
        };

        if self.len_at(later) < self.len_at(earlier) {
            later
        } else {
            earlier
        }
    }

    /// One candidate step: the base moved by a drawn offset within the window,
    /// never earlier than the step after `after`.
    fn offset(base: u64, drawn: u64, spread: u32, after: u64) -> u64 {
        let shifted: i128 = base as i128 + drawn as i128 - i128::from(spread);

        (shifted.max(0) as u64).max(after.saturating_add(1))
    }
}

impl<T: Element, R: StochasticStream> Collection for StochasticScheduler<T, R> {
    type Item = T;

    fn len(&self) -> usize {
        self.len
    }

    fn contains(&self, item: &T) -> bool {
        StochasticScheduler::contains(self, item)
    }

    /// Earliest step first, and within a step in the order entries were placed.
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

/// Ordered by the step work is due on, so a range is a span of steps.
impl<T: Element, R: StochasticStream> RangeQuery for StochasticScheduler<T, R> {
    type Key = u64;
    type Item<'a>
        = (u64, &'a T)
    where
        Self: 'a;

    fn range<'a, B: RangeBounds<u64>>(
        &'a self,
        range: B,
    ) -> impl DoubleEndedIterator<Item = Self::Item<'a>> {
        let bounds: (Bound<u64>, Bound<u64>) =
            (range.start_bound().cloned(), range.end_bound().cloned());

        self.pending
            .range(bounds)
            .flat_map(|(step, entries)| entries.iter().map(move |entry| (*step, &entry.value)))
    }
}

impl<T: Element, R: StochasticStream> DeterministicOrder for StochasticScheduler<T, R> {}

impl<T: Element + fmt::Debug, R: StochasticStream> fmt::Debug for StochasticScheduler<T, R> {
    /// As the steps and what is due on them.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_map()
            .entries(self.pending.iter().map(|(step, entries)| {
                (
                    step,
                    entries
                        .iter()
                        .map(|entry| &entry.value)
                        .collect::<Vec<&T>>(),
                )
            }))
            .finish()
    }
}

impl<T: Element, R: StochasticStream> fmt::Display for StochasticScheduler<T, R> {
    /// As what is waiting and when the next of it is due.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.next_due() {
            Some(step) => write!(formatter, "{} pending, next at {step}", self.len),
            None => write!(formatter, "nothing pending"),
        }
    }
}

/// Ready means due at or before the current step. For this scheduler that is
/// only ever so part-way through an advance, since entries are always placed
/// after the clock; the trait is implemented so that generic code can drain it
/// alongside the others.
impl<T: Element, R: StochasticStream> Pending for StochasticScheduler<T, R> {
    type Ready = Firing<T>;

    fn pending_len(&self) -> usize {
        self.len
    }

    fn ready_len(&self) -> usize {
        self.pending
            .range(..=self.now)
            .map(|(_, entries)| entries.len())
            .sum()
    }

    fn take_ready(&mut self) -> Vec<Firing<T>> {
        self.take_due()
    }
}

impl<T: Element, R: StochasticStream> StochasticScheduler<T, R> {
    /// Takes one step and returns what falls due and the gate allows.
    ///
    /// An entry the gate refuses is **not** dropped: it has already drawn its
    /// next wait and been placed, exactly as it would have been had it fired,
    /// so it simply comes round again later. The occurrence is spent, not the
    /// entry. That is what a recurrence wants — a plot that could not grow this
    /// time tries again on its own schedule.
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
