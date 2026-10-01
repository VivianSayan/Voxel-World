//! Time points, tick durations, and elapsed seconds are different quantities.
//!
//! A [`Tick`] is a moment measured from the world's origin, a [`TickDuration`]
//! is a count of ticks between two moments, and [`Seconds`] is wall-clock time.
//! Mixing them up is the usual source of off-by-one scheduling and of simulation
//! speed changing with frame rate, so each is its own type: adding two moments
//! is meaningless and will not compile, while a moment plus a duration is
//! another moment, and subtracting two moments gives a duration.
//!
//! A [`TickRate`] is the one place the three meet: it converts between ticks and
//! seconds, and nothing else does. Simulation counts ticks, so the world runs
//! the same however the frame rate varies; seconds appear only at the edges,
//! where something is displayed or a real clock is read.
//!
//! [`TickScheduler`](crate::structures::collections::TickScheduler) is the
//! world-time view of
//! [`Scheduler`](crate::structures::collections::Scheduler): the same sparse
//! map of pending work, with deadlines given as [`Tick`]s and delays as
//! [`TickDuration`]s.

use std::fmt;
use std::num::NonZeroU32;
use std::ops::{Add, Sub};

/// A non-negative, finite elapsed duration in seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Seconds(f64);

impl Seconds {
    /// No time at all.
    pub const ZERO: Self = Self(0.0);

    /// A duration in seconds, or `None` unless it is finite and not negative.
    pub fn new(value: f64) -> Option<Self> {
        (value.is_finite() && value >= 0.0).then_some(Self(value))
    }

    /// The duration as a plain `f64`, in seconds.
    pub const fn value(self) -> f64 {
        self.0
    }
}

/// A count of ticks: how long something takes or how long to wait, never when
/// it happens.
///
/// [`Tick`] is the moment; this is the distance between two of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TickDuration(u64);

impl TickDuration {
    /// No ticks at all: something due now.
    pub const ZERO: Self = Self(0);
    /// A single tick, which is the step the world advances by.
    pub const ONE: Self = Self(1);

    /// A duration from a count of ticks.
    pub const fn new(count: u64) -> Self {
        Self(count)
    }

    /// The duration as a count of ticks.
    pub const fn count(self) -> u64 {
        self.0
    }
}

/// A point in world time, counted in ticks from the world's origin.
///
/// Saturating rather than wrapping at the end of representable time: a world
/// that somehow reached tick `u64::MAX` would stop advancing rather than jump
/// back to its own beginning. At 20 ticks a second that is about 29 billion
/// years away.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tick(u64);

impl Tick {
    /// The world's first tick, where its time starts.
    pub const ORIGIN: Self = Self(0);

    /// A moment from a count of ticks since the origin.
    pub const fn new(count: u64) -> Self {
        Self(count)
    }

    /// How many ticks have passed since the origin.
    pub const fn count(self) -> u64 {
        self.0
    }

    /// The moment one tick later.
    pub const fn next(self) -> Self {
        self.advanced_by(TickDuration::ONE)
    }

    /// The moment `duration` ticks later, saturating rather than wrapping at
    /// the end of representable time.
    pub const fn advanced_by(self, duration: TickDuration) -> Self {
        Self(self.0.saturating_add(duration.0))
    }

    /// How many ticks have passed since `earlier`, or zero when `earlier` is
    /// actually in the future. Never negative, since a duration is unsigned.
    pub const fn ticks_since(self, earlier: Self) -> TickDuration {
        TickDuration(self.0.saturating_sub(earlier.0))
    }

    /// Whether this moment falls on a multiple of `interval`, which is how a
    /// job that runs every so many ticks decides to act.
    ///
    /// Always false for an interval of zero, which names no schedule.
    pub const fn is_every(self, interval: TickDuration) -> bool {
        interval.0 != 0 && self.0.is_multiple_of(interval.0)
    }
}

/// A moment plus a duration is a later moment; see [`Tick::advanced_by`].
impl Add<TickDuration> for Tick {
    type Output = Self;
    fn add(self, duration: TickDuration) -> Self {
        self.advanced_by(duration)
    }
}

/// Two moments subtract to the duration between them; see
/// [`Tick::ticks_since`].
impl Sub for Tick {
    type Output = TickDuration;
    fn sub(self, earlier: Self) -> TickDuration {
        self.ticks_since(earlier)
    }
}

/// How many ticks the world runs per second: the one converter between tick
/// counts and wall-clock time.
///
/// Positive by construction, since a rate of zero would make a tick take
/// forever and every conversion infinite. Pausing is not a rate of zero; it is
/// simply not advancing the clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TickRate(NonZeroU32);

impl TickRate {
    /// A rate in ticks per second, or `None` for zero.
    pub const fn new(per_second: u32) -> Option<Self> {
        match NonZeroU32::new(per_second) {
            Some(rate) => Some(Self(rate)),
            None => None,
        }
    }

    /// The rate as a plain count of ticks per second.
    pub const fn per_second(self) -> u32 {
        self.0.get()
    }

    /// How long one tick lasts, which is the reciprocal of the rate.
    pub fn tick_length(self) -> Seconds {
        Seconds(1.0 / self.per_second() as f64)
    }

    /// How much wall-clock time has passed at a given moment, counting from
    /// the world's origin.
    pub fn seconds_at(self, tick: Tick) -> Seconds {
        self.seconds_in(TickDuration::new(tick.count()))
    }

    /// How long a duration of ticks lasts in seconds.
    pub fn seconds_in(self, duration: TickDuration) -> Seconds {
        Seconds(duration.count() as f64 / self.per_second() as f64)
    }

    /// How many whole ticks fit in a stretch of seconds.
    ///
    /// Rounds down, so a partial tick is not counted, and saturates rather than
    /// wrapping if the answer would exceed the tick counter.
    pub fn ticks_in(self, seconds: Seconds) -> TickDuration {
        TickDuration::new((seconds.value() * self.per_second() as f64) as u64)
    }
}

/// Twenty ticks a second, the rate the world assumes unless told otherwise.
impl Default for TickRate {
    fn default() -> Self {
        Self::new(20).unwrap()
    }
}

/// As `tick <n>`.
impl fmt::Display for Tick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tick {}", self.0)
    }
}
/// As `<n> ticks`.
impl fmt::Display for TickDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ticks", self.0)
    }
}
/// As `<n> s`.
impl fmt::Display for Seconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} s", self.0)
    }
}
/// As `<n> ticks/s`.
impl fmt::Display for TickRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ticks/s", self.per_second())
    }
}
