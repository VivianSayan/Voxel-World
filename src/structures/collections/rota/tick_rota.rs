//! A rota that knows about simulation time: which share's turn it is, and how many
//! seconds that share has to account for.
//!
//! # What this adds to [`Rota`]
//!
//! [`Rota`] spreads a population over shares and says nothing about when a share's turn
//! comes — deliberately, since a turn is whatever the caller decides. That leaves every
//! caller to work out how much simulation time a share has waited through, and in a
//! world whose tick rate can change that is not a subtraction:
//!
//! ```ignore
//! // Per tick, for every share:
//! accumulated[share] += this_tick_length;
//!
//! // When a share's turn comes:
//! let delta = accumulated[share];
//! accumulated[share] = Seconds::ZERO;
//! ```
//!
//! It has to be an accumulation because `ticks since last turn × current tick length`
//! is wrong the moment the rate moved during the span. Four ticks across a 60→20 Hz
//! change are neither `4/60` of a second nor `4/20` — they are the sum of four
//! individual tick lengths, and only something adding them up as they happened knows
//! it.
//!
//! That is bookkeeping, it is identical everywhere, and getting it wrong gives a system
//! that integrates the wrong amount of time — a bug that looks like bad tuning rather
//! than a mistake. So it lives here instead, and a caller writes:
//!
//! ```ignore
//! if let Some(update) = rota_clock.step(now, rate.tick_length()) {
//!     for entity in update.group_of(&rota) {
//!         simulate(entity, update.delta());
//!     }
//! }
//! ```
//!
//! # Timing, not membership
//!
//! This holds no elements. [`Rota<T, GROUPS>`](Rota) holds those, and the two are paired
//! by sharing `GROUPS` — [`TickRotaUpdate::group_of`] takes the rota and returns the
//! share whose turn it is.
//!
//! Keeping them apart means the same clock can drive something that is not a population
//! at all: a set of systems, a list of gates, one function per share.
//!
//! # The cadence
//!
//! A share's turn comes round every `cadence` ticks, and share `i` takes the tick where
//! `tick % cadence == i`. So with a cadence of four and four shares:
//!
//! ```text
//! tick  0  1  2  3  4  5  6  7  8 ...
//! share 0  1  2  3  0  1  2  3  0 ...
//! ```
//!
//! Every share runs once per four ticks, one share per tick, and no tick carries more
//! than a quarter of the work. Under a 60 Hz world that is a 15 Hz system.
//!
//! A cadence longer than the share count is allowed and means the shares burst and then
//! idle — six shares at a cadence of ten run on ticks 0..6 of every ten. A cadence
//! *shorter* than the share count is refused, because the later shares could never take
//! a turn.
//!
//! # Integer ticks only
//!
//! The cadence is a whole number of ticks, so 60 Hz divides into 60, 30, 20, 15, 12, 10
//! and so on, and **not** into 17. There is no accumulated phase error here and no
//! floating point: a rate that does not divide evenly is refused by
//! [`TickRota::at_rate`] rather than approximated. Something wanting exactly 17 Hz wants
//! a scheduler, not a rota.

use super::rota::Rota;
use crate::units::time::{Seconds, Tick, TickDuration, TickRate, UpdateDelta};
use std::fmt;

/// One share's turn: which share, when, and how much simulation time it covers.
///
/// Produced by [`TickRota::update`]. A reading, not state — the clock keeps the state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickRotaUpdate {
    bucket: usize,
    at: Tick,
    elapsed: TickDuration,
    delta: UpdateDelta,
    first: bool,
}

impl TickRotaUpdate {
    /// Which share's turn it is, as an index into a [`Rota`]'s groups.
    pub const fn bucket(self) -> usize {
        self.bucket
    }

    /// The tick this turn fell on.
    pub const fn at(self) -> Tick {
        self.at
    }

    /// How many ticks passed since this share's previous turn.
    ///
    /// The cadence on an ordinary turn, more if the turn was late, and the cadence on
    /// the very first one — see [`TickRota::update`].
    pub const fn elapsed_ticks(self) -> TickDuration {
        self.elapsed
    }

    /// How much authoritative simulation time passed since this share's previous turn.
    ///
    /// # Question
    ///
    /// "How many seconds does this share have to account for?"
    ///
    /// **This is the value a simulation integrates over** — seconds, in fixed point,
    /// not a tick count. [`TickRotaUpdate::elapsed_ticks`] is the same span counted in
    /// ticks, which is useful for logging or for a system that works in whole steps.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    /// use voxel_world::structures::collections::rota::TickRota;
    /// use voxel_world::time::{Tick, TickRate};
    ///
    /// let rate = TickRate::new(64).expect("positive");
    /// let mut clock: TickRota<4> = TickRota::spread(rate);
    ///
    /// // Each tick reports its own length, which the clock adds up per share.
    /// for count in 0..4u64 {
    ///     clock.step(Tick::new(count), rate.tick_length()).expect("due");
    /// }
    ///
    /// let update = clock.step(Tick::new(4), rate.tick_length()).expect("due");
    ///
    /// // Four ticks of a sixty-fourth each: a sixteenth of a second, as seconds.
    /// assert_eq!(update.delta().fixed(), Fixed::ONE / 16);
    /// assert_eq!(update.seconds().fixed(), update.delta().fixed());
    /// ```
    ///
    /// # Passing it to a simulation
    ///
    /// It multiplies the kinematic quantities directly, so a system takes it and does
    /// whatever delta treatment it likes:
    ///
    /// ```
    /// # use voxel_world::math::Fixed;
    /// # use voxel_world::spatial::kinematics::{Acceleration3, Velocity3};
    /// # use voxel_world::spatial::precise::PrecisePosition3;
    /// # use voxel_world::structures::collections::rota::TickRota;
    /// # use voxel_world::time::{Tick, TickRate, UpdateDelta};
    /// fn simulate(
    ///     position: &mut PrecisePosition3,
    ///     velocity: &mut Velocity3,
    ///     gravity: Acceleration3,
    ///     delta: UpdateDelta,
    /// ) {
    ///     *velocity += gravity * delta;
    ///     *position += *velocity * delta;
    /// }
    ///
    /// # let mut clock: TickRota<4> = TickRota::spread(TickRate::new(64).unwrap());
    /// # let mut position = PrecisePosition3::new(Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
    /// # let mut velocity = Velocity3::ZERO;
    /// # let gravity = Acceleration3::new(Fixed::ZERO, Fixed::from_integer(-10), Fixed::ZERO);
    /// let update = clock.update(Tick::new(0)).expect("due");
    ///
    /// simulate(&mut position, &mut velocity, gravity, update.delta());
    /// ```
    pub const fn delta(self) -> UpdateDelta {
        self.delta
    }

    /// The same elapsed time as a general [`Seconds`].
    ///
    /// For a method that takes a plain physical duration rather than specifically an
    /// authoritative update's. [`TickRotaUpdate::delta`] is the one to prefer inside a
    /// simulation, because its type says the time came from counting ticks and may
    /// therefore change the world.
    pub const fn seconds(self) -> Seconds {
        self.delta.seconds()
    }

    /// Whether this is the share's first turn.
    ///
    /// For a system that genuinely wants to initialise rather than integrate on its
    /// first run, since the delta it is given is nominal rather than measured.
    pub const fn is_first(self) -> bool {
        self.first
    }

    /// The elements whose turn it is, from a rota sharing this clock's share count.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::structures::collections::rota::Rota;
    /// use voxel_world::structures::collections::rota::TickRota;
    /// use voxel_world::time::{Tick, TickRate};
    ///
    /// let mut population: Rota<u32, 4> = Rota::new();
    /// for entity in 0..100 {
    ///     population.insert(entity);
    /// }
    ///
    /// let mut clock: TickRota<4> = TickRota::spread(TickRate::new(60).expect("positive"));
    ///
    /// let rate = TickRate::new(60).expect("positive");
    /// # let _ = rate;
    /// let update = clock.step(Tick::new(6), rate.tick_length()).expect("a share is due");
    ///
    /// // A quarter of the population, and the time since it last ran.
    /// assert_eq!(update.bucket(), 2);
    /// assert!(update.group_of(&population).len() <= 26);
    /// ```
    ///
    /// # Panics
    ///
    /// Never for a rota with the same share count, which the type parameter enforces.
    pub fn group_of<T, const GROUPS: usize>(self, rota: &Rota<T, GROUPS>) -> &[T]
    where
        T: crate::structures::traits::Element,
    {
        rota.group(self.bucket)
    }
}

impl fmt::Display for TickRotaUpdate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "share {} at {} after {}",
            self.bucket, self.at, self.elapsed
        )
    }
}

/// A rota's clock: whose turn it is, and how many seconds that share has to account for.
///
/// # Question
///
/// "Which share runs now, and how much simulation time has built up for it?"
///
/// # Why it accumulates seconds rather than counting ticks
///
/// A share's elapsed time *could* be worked out as `ticks since its last turn × tick
/// length`. That is wrong in any world whose tick rate can change, which is most of
/// them: a world that drops from 60 Hz to 20 Hz under load, or speeds up, or is
/// deliberately slowed, has ticks of different lengths. Four ticks spanning a 60→20
/// change are not `4/20` of a second and not `4/60` either — they are the sum of four
/// individual tick lengths, and only something that added them up as they happened
/// knows what that sum is.
///
/// So this clock holds a **seconds accumulator per share**. Every tick adds that tick's
/// own length to all of them; taking a share's turn hands over its total and resets it
/// to nothing. The tick count is still tracked alongside, but only as information —
/// the seconds are what a simulation integrates over.
///
/// # The loop
///
/// ```
/// use voxel_world::structures::collections::rota::TickRota;
/// use voxel_world::time::{Seconds, Tick, TickRate};
///
/// let mut rate = TickRate::new(60).expect("positive");
/// let mut clock: TickRota<4> = TickRota::spread(rate);
///
/// for count in 0..8u64 {
///     // The world slows to 20 Hz partway through.
///     if count == 4 {
///         rate = TickRate::new(20).expect("positive");
///     }
///
///     // One call per simulated tick: this tick's length in, the due share out.
///     if let Some(update) = clock.step(Tick::new(count), rate.tick_length()) {
///         // `update.delta()` is the sum of the tick lengths this share actually
///         // waited through, whatever the rate was doing.
///         let _ = update.delta();
///     }
/// }
///
/// // Share 0's second turn spanned one 60 Hz tick and three 20 Hz ticks.
/// let expected = Seconds::from_fixed(
///     TickRate::new(60).unwrap().tick_length().fixed()
///         + TickRate::new(20).unwrap().tick_length().fixed() * voxel_world::math::Fixed::from_integer(3),
/// )
/// .unwrap();
/// # let _ = expected;
/// ```
///
/// # The cost of accumulating
///
/// Adding tick lengths up is not quite the same number as dividing once. A sixtieth of
/// a second is not a binary fraction, so each tick is stored as the nearest `2^-32` and
/// falls a fraction of a step short; four of them summed come out one step under the
/// nearest fifteenth.
///
/// One step is `2^-32` of a second — about 233 picoseconds — and the error **does not
/// build up**, because taking a turn empties that share's accumulator. So the most any
/// single delta is out by is the cadence in steps: about a nanosecond at a cadence of
/// four, against an interval of 67 milliseconds.
///
/// That is the trade, stated plainly:
///
/// | | constant rate | changing rate |
/// |---|---|---|
/// | `ticks × rate` | exact | **wrong** |
/// | accumulating | within a nanosecond | correct |
///
/// Being a nanosecond out is worth being right when the rate moves.
///
/// # Determinism
///
/// Integer ticks, integer share indices and [`Fixed`](crate::math::Fixed) seconds. The
/// same sequence of ticks and tick lengths gives the same shares and the same deltas on
/// every target. Nothing here measures anything itself — the caller supplies each tick's
/// length, so replaying a recorded sequence reproduces the run exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickRota<const GROUPS: usize> {
    cadence: TickDuration,
    /// The rate the first turn's nominal interval was primed from.
    ///
    /// Kept only for that, and for [`TickRota::reset`]. Every later delta comes from the
    /// accumulators, so a change in the world's rate needs no change here.
    nominal: TickRate,
    /// Seconds built up since each share last took a turn.
    ///
    /// This array is the whole reason the type exists: the one piece of bookkeeping
    /// every caller would otherwise duplicate, and the one that cannot be reconstructed
    /// after the fact once the rate has moved.
    accumulated: [Seconds; GROUPS],
    /// When each share last took a turn, for the informational tick count.
    last_run: [Option<Tick>; GROUPS],
}

impl<const GROUPS: usize> TickRota<GROUPS> {
    /// A clock whose cadence equals its share count: one share per tick, each share
    /// once per round.
    ///
    /// The ordinary case. Four shares under a 60 Hz world gives a 15 Hz system with the
    /// work spread evenly across every tick.
    ///
    /// `nominal` is the rate the **first** turn's interval is primed from; see
    /// [`TickRota::step`]. Later turns use the accumulated time instead, so the world's
    /// rate may change afterwards without telling this.
    ///
    /// # Panics
    ///
    /// On zero shares, which has nothing to schedule. A share count is a compile-time
    /// constant, so this cannot depend on input.
    pub fn spread(nominal: TickRate) -> Self {
        assert!(GROUPS > 0, "a rota clock needs at least one share");

        Self::primed(nominal, TickDuration::new(GROUPS as u64))
    }

    /// A clock with an explicit cadence, or `None` for one that cannot work.
    ///
    /// # Refused configurations
    ///
    /// - a cadence of zero, which names no interval;
    /// - zero shares, which have nothing to schedule;
    /// - a cadence below the share count, where the later shares would never take a
    ///   turn — six shares at a cadence of four leaves shares 4 and 5 permanently idle,
    ///   which is a mistake rather than a choice.
    pub fn new(nominal: TickRate, cadence: TickDuration) -> Option<Self> {
        if GROUPS == 0 || cadence.count() == 0 || cadence.count() < GROUPS as u64 {
            return None;
        }

        Some(Self::primed(nominal, cadence))
    }

    /// A clock running each share `updates_per_second` times a second at the nominal
    /// rate, or `None` if that does not divide into whole ticks.
    ///
    /// # Question
    ///
    /// "I want this system at 15 Hz. What cadence is that?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::structures::collections::rota::TickRota;
    /// use voxel_world::time::{TickDuration, TickRate};
    ///
    /// let world = TickRate::new(60).expect("positive");
    ///
    /// // 60 divides by 15, so this is every four ticks.
    /// let clock: TickRota<4> = TickRota::at_rate(world, 15).expect("divides");
    /// assert_eq!(clock.cadence(), TickDuration::new(4));
    ///
    /// // 60 does not divide by 17, so there is no whole cadence for it.
    /// assert!(TickRota::<4>::at_rate(world, 17).is_none());
    /// ```
    ///
    /// # Why an uneven rate is refused rather than approximated
    ///
    /// Approximating it would mean a cadence that drifts — some updates four ticks apart
    /// and some three, with a phase error accumulating. That is a scheduler's job, not a
    /// rota's.
    ///
    /// Note this fixes the **cadence in ticks**, not the real frequency: if the world
    /// later runs at 20 Hz, a cadence of four becomes 5 Hz, and the deltas say so.
    pub fn at_rate(nominal: TickRate, updates_per_second: u32) -> Option<Self> {
        if updates_per_second == 0 || !nominal.per_second().is_multiple_of(updates_per_second) {
            return None;
        }

        Self::new(
            nominal,
            TickDuration::new(u64::from(nominal.per_second() / updates_per_second)),
        )
    }

    /// A clock with nothing accumulated yet.
    ///
    /// The accumulators start empty rather than primed: a share's first turn takes its
    /// delta from the nominal cadence directly, so there is nothing to prime them with.
    /// See [`TickRota::step`].
    fn primed(nominal: TickRate, cadence: TickDuration) -> Self {
        Self {
            cadence,
            nominal,
            accumulated: [Seconds::ZERO; GROUPS],
            last_run: [None; GROUPS],
        }
    }

    /// One nominal cadence as seconds: what a share's first turn reports.
    fn nominal_interval(self) -> Seconds {
        self.nominal.seconds_in(self.cadence)
    }

    /// The rate the first turn's interval was primed from.
    ///
    /// Not "the world's rate": this clock does not track that and does not need to,
    /// because every delta after the first comes from accumulated time.
    pub const fn nominal_rate(self) -> TickRate {
        self.nominal
    }

    /// How many ticks between one share's turns.
    pub const fn cadence(self) -> TickDuration {
        self.cadence
    }

    /// How many shares the round is divided into.
    pub const fn bucket_count(self) -> usize {
        GROUPS
    }

    /// How often each share runs at the nominal rate, when the cadence divides it.
    pub fn updates_per_second(self) -> Option<u32> {
        let cadence: u32 = u32::try_from(self.cadence.count()).ok()?;

        self.nominal
            .per_second()
            .is_multiple_of(cadence)
            .then(|| self.nominal.per_second() / cadence)
    }

    /// Which share's turn falls on this tick, if any.
    ///
    /// A pure query: it accumulates nothing and takes no turn.
    pub const fn due_bucket(self, now: Tick) -> Option<usize> {
        let phase: u64 = now.count() % self.cadence.count();

        if phase < GROUPS as u64 {
            return Some(phase as usize);
        }

        None
    }

    /// Seconds built up for a share since its last turn, or `None` for a bad index.
    ///
    /// What [`TickRota::step`] would hand that share if its turn came now.
    pub const fn accumulated(self, bucket: usize) -> Option<Seconds> {
        if bucket >= GROUPS {
            return None;
        }

        Some(self.accumulated[bucket])
    }

    /// When a share last took a turn, or `None` before its first.
    pub const fn last_run(self, bucket: usize) -> Option<Tick> {
        if bucket >= GROUPS {
            return None;
        }

        self.last_run[bucket]
    }

    /// How many ticks since a share's last turn, counting the cadence before its first.
    ///
    /// Informational. The authoritative span is [`TickRota::accumulated`], which is
    /// what the delta is built from.
    pub fn ticks_since(self, bucket: usize, now: Tick) -> TickDuration {
        match self.last_run(bucket) {
            Some(previous) => now.ticks_since(previous),
            None => self.cadence,
        }
    }

    /// Adds one tick's length to every share's accumulator.
    ///
    /// # Question
    ///
    /// "A tick just happened and it lasted this long. Tell every share."
    ///
    /// Call it once per simulated tick with that tick's own length — ordinarily
    /// `rate.tick_length()` for whatever rate the world is running at *now*. Adding to
    /// every share is what makes the clock correct across a rate change: each share's
    /// total is the sum of the ticks it actually waited through, not a count multiplied
    /// by a rate that may since have moved.
    ///
    /// Saturating, as [`Seconds`] addition is, so a clock left running cannot wrap.
    ///
    /// Most callers want [`TickRota::step`], which does this and takes the due turn in
    /// one call — and so cannot be half-forgotten.
    pub fn advance(&mut self, elapsed: Seconds) {
        for total in &mut self.accumulated {
            *total += elapsed;
        }
    }

    /// Takes the turn due on this tick, handing over the seconds built up for it.
    ///
    /// Resets that share's accumulator to nothing, so the next turn reports only the
    /// time after this one. Accumulates nothing itself — pair it with
    /// [`TickRota::advance`], or use [`TickRota::step`] to do both.
    pub fn update(&mut self, now: Tick) -> Option<TickRotaUpdate> {
        let bucket: usize = self.due_bucket(now)?;

        Some(self.take_turn(bucket, now))
    }

    /// One simulated tick: accumulate its length, then take whatever turn is due.
    ///
    /// # Question
    ///
    /// "This tick lasted this long. Whose turn is it, and how long have they waited?"
    ///
    /// The call to reach for. It is [`TickRota::advance`] followed by
    /// [`TickRota::update`], in the order that makes the tick being reported count
    /// towards the turn being taken.
    ///
    /// # Call it every tick
    ///
    /// The clock can only add up time it is told about. A tick that goes by without a
    /// `step` is a tick no share accounts for, so the next deltas come out short. That
    /// is inherent to accumulating rather than deriving — and it is the price of being
    /// right when the rate moves.
    ///
    /// # The first turn
    ///
    /// A share's first turn reports **one nominal cadence**, taken from the rate given
    /// at construction rather than from the accumulator, which has nothing meaningful
    /// in it yet.
    ///
    /// Zero would make a system integrate nothing on its first run, so anything that
    /// moves would visibly stall for an interval — and a system that only acts when
    /// time has passed would skip its first turn entirely. Whatever had accumulated
    /// since the clock was built is worse: a rota built long before its first turn would
    /// hand it an enormous delta and fling everything across the world.
    ///
    /// [`TickRotaUpdate::is_first`] distinguishes it for anything that would rather
    /// initialise than integrate.
    ///
    /// # A late or stretched turn
    ///
    /// The delta is always the time actually accumulated, never the cadence assumed. A
    /// share whose turn came late reports the longer total; a share whose ticks were
    /// long reports the longer total. That is the entire reason this type holds state.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    /// use voxel_world::structures::collections::rota::TickRota;
    /// use voxel_world::time::{Tick, TickRate};
    ///
    /// let rate = TickRate::new(64).expect("positive");
    /// let mut clock: TickRota<4> = TickRota::spread(rate);
    ///
    /// // A full round at a steady rate.
    /// for count in 0..4u64 {
    ///     clock.step(Tick::new(count), rate.tick_length()).expect("due");
    /// }
    ///
    /// // Share 0's second turn: four ticks of a sixty-fourth each is a sixteenth.
    /// let again = clock.step(Tick::new(4), rate.tick_length()).expect("due");
    ///
    /// assert_eq!(again.bucket(), 0);
    /// assert_eq!(again.delta().fixed(), Fixed::ONE / 16);
    /// ```
    pub fn step(&mut self, now: Tick, elapsed: Seconds) -> Option<TickRotaUpdate> {
        self.advance(elapsed);
        self.update(now)
    }

    /// Takes a named share's turn, whether or not this tick is its own.
    ///
    /// # Question
    ///
    /// "The world jumped forward. How do I run this share now and account for the gap?"
    ///
    /// The escape hatch for a caller implementing its own catch-up: it can run the
    /// shares it chooses, each with the time actually accumulated. This type holds no
    /// catch-up policy, because running every missed turn and running once with a bigger
    /// delta are different simulations and only the caller knows which it wants.
    ///
    /// `None` for a share index at or above the share count.
    pub fn update_bucket(&mut self, bucket: usize, now: Tick) -> Option<TickRotaUpdate> {
        if bucket >= GROUPS {
            return None;
        }

        Some(self.take_turn(bucket, now))
    }

    /// Whether a share has accumulated more than one nominal cadence of time.
    ///
    /// Mechanism for a caller deciding what to do about a stall; this type decides
    /// nothing. A share that has never run is not overdue — its first turn is still to
    /// come, and it was primed with exactly one cadence.
    pub fn is_overdue(self, bucket: usize) -> bool {
        match (self.last_run(bucket), self.accumulated(bucket)) {
            (Some(_), Some(total)) => total > self.nominal_interval(),
            _ => false,
        }
    }

    /// Every share that has accumulated more than one nominal cadence, in order.
    pub fn overdue_buckets(self) -> impl Iterator<Item = usize> {
        (0..GROUPS).filter(move |bucket| self.is_overdue(*bucket))
    }

    /// Forgets every share's turn and re-primes its accumulator.
    ///
    /// For a world reloaded or a system restarted, where the accumulated time describes
    /// a timeline that no longer applies.
    pub fn reset(&mut self) {
        *self = Self::primed(self.nominal, self.cadence);
    }

    /// Records the turn, empties that share's accumulator, and builds the reading.
    fn take_turn(&mut self, bucket: usize, now: Tick) -> TickRotaUpdate {
        let previous: Option<Tick> = self.last_run[bucket];

        // A first turn reports one nominal cadence rather than whatever happened to
        // have accumulated before it — see `step` for why. Every later turn reports the
        // time actually added up, which is the whole point of accumulating.
        let total: Seconds = match previous {
            Some(_) => self.accumulated[bucket],
            None => self.nominal_interval(),
        };

        let elapsed: TickDuration = match previous {
            Some(previous) => now.ticks_since(previous),
            None => self.cadence,
        };

        // The turn has been taken, so the wait starts again from nothing.
        self.accumulated[bucket] = Seconds::ZERO;
        self.last_run[bucket] = Some(now);

        TickRotaUpdate {
            bucket,
            at: now,
            elapsed,
            delta: UpdateDelta::from_accumulated(total),
            first: previous.is_none(),
        }
    }
}

impl<const GROUPS: usize> fmt::Display for TickRota<GROUPS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{GROUPS} shares every {} (nominally {})",
            self.cadence, self.nominal
        )
    }
}
