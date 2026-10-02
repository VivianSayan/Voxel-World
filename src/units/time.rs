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

use crate::math::Fixed;
use std::fmt;
use std::num::NonZeroU32;
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// A non-negative elapsed duration of real time, in seconds.
///
/// # Question
///
/// "*How long* did that take, in real time?"
///
/// # Why this is fixed-point and not an `f64`
///
/// It used to be an `f64`, which made every use of it a conversion: the platform hands
/// over integer nanoseconds, the engine computes in [`Fixed`], and a float in between
/// turned one exact value into two roundings. Backed by `Fixed` it is the same
/// representation the rest of the world mathematics uses, so a frame time reaches a
/// velocity without passing through anything lossy.
///
/// The resolution is `2^-32` seconds — about 233 picoseconds, four times finer than a
/// nanosecond — and the range runs to `2^95` seconds, which is longer than the universe
/// has existed. Neither is a practical limit on a frame timer.
///
/// # Simulation time against real time
///
/// This is **real** time, measured. [`TickDuration`] is **simulation** time, counted.
/// They are not interchangeable and [`TickRate`] is the only bridge between them: a
/// count of ticks does not intrinsically mean a number of seconds.
///
/// # Example
///
/// ```
/// use voxel_world::time::Seconds;
///
/// // A half and a quarter of a second are dyadic, so these are exact.
/// let half = Seconds::from_millis(500).expect("in range");
///
/// assert_eq!(half + half, Seconds::from_seconds(1).unwrap());
/// assert!(Seconds::from_millis(16).unwrap() < Seconds::from_millis(17).unwrap());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Seconds(Fixed);

impl Seconds {
    /// No time at all.
    pub const ZERO: Self = Self(Fixed::ZERO);

    /// One second.
    pub const SECOND: Self = Self(Fixed::ONE);

    /// The longest duration this can hold.
    pub const MAX: Self = Self(Fixed::MAX);

    /// A duration from a fixed-point count of seconds, or `None` if negative.
    ///
    /// Negative time is not a duration, so it is refused rather than clamped — a
    /// negative value here means the measurement went wrong upstream, and saying so is
    /// more use than a silent zero.
    pub const fn from_fixed(value: Fixed) -> Option<Self> {
        if value.to_bits() < 0 {
            return None;
        }

        Some(Self(value))
    }

    /// A duration from a value already known to be non-negative.
    ///
    /// # Why this exists beside [`Seconds::from_fixed`]
    ///
    /// For a caller that has *already* established the invariant and would otherwise
    /// have to handle an impossible `None`. [`FrameDelta`](crate::units::frame::FrameDelta)
    /// is the case: it refuses a negative value at construction, so converting one to a
    /// duration cannot fail — and writing a fallback for that would mean inventing a
    /// wrong answer for a branch that is never taken.
    ///
    /// Crate-internal, so the public checked constructor stays the only way in from
    /// outside. The debug assertion is there to catch a future caller that is mistaken
    /// about its own invariant.
    pub(crate) const fn from_non_negative(value: Fixed) -> Self {
        debug_assert!(
            value.to_bits() >= 0,
            "from_non_negative was given a negative duration"
        );

        Self(value)
    }

    /// A whole number of seconds, or `None` if it is negative or too large.
    pub const fn from_seconds(whole: i64) -> Option<Self> {
        if whole < 0 {
            return None;
        }

        match Fixed::from_integer_i128(whole as i128) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// A duration in whole milliseconds, or `None` if too large.
    pub const fn from_millis(millis: u64) -> Option<Self> {
        Self::from_scaled(millis, 1_000)
    }

    /// A duration in whole microseconds, or `None` if too large.
    pub const fn from_micros(micros: u64) -> Option<Self> {
        Self::from_scaled(micros, 1_000_000)
    }

    /// A duration in whole nanoseconds, or `None` if too large.
    ///
    /// A nanosecond is not a whole number of `2^-32` seconds, so this rounds to the
    /// nearest step — within about 117 picoseconds.
    pub const fn from_nanos(nanos: u64) -> Option<Self> {
        Self::from_scaled(nanos, 1_000_000_000)
    }

    /// `count / per_second` seconds, in integers throughout.
    ///
    /// The whole part scales exactly; only the remainder is divided, once, and rounded
    /// to the nearest step. No float appears and nothing is rounded twice.
    ///
    /// # Which values are exact
    ///
    /// Those whose fraction is dyadic: halves, quarters, eighths and so on. A
    /// millisecond is `1/1000` of a second, and 1000 has a factor of five, so
    /// `from_millis(20)` is **not** exact — it is the nearest `2^-32` to a fiftieth,
    /// within about 117 picoseconds. `from_micros(15_625)` is a sixty-fourth and is
    /// exact.
    const fn from_scaled(count: u64, per_second: u64) -> Option<Self> {
        let whole: u64 = count / per_second;
        let remainder: u64 = count % per_second;

        let Some(seconds) = Fixed::from_integer_i128(whole as i128) else {
            return None;
        };

        // Nearest rather than truncated: adding half the divisor before dividing
        // halves the worst-case error for the cost of one addition. `remainder` is
        // below `per_second`, so the result stays under one whole unit.
        let scaled: u128 = ((remainder as u128) << Fixed::FRACTION_BITS) + (per_second as u128) / 2;
        let fraction: i128 = (scaled / per_second as u128) as i128;

        match seconds.to_bits().checked_add(fraction) {
            Some(bits) => Some(Self(Fixed::from_bits(bits))),
            None => None,
        }
    }

    /// A duration from the standard library's, without touching a float.
    ///
    /// # Question
    ///
    /// "The platform measured a `Duration`. How does it enter the engine?"
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    /// use voxel_world::time::Seconds;
    ///
    /// let measured = Seconds::from_duration(Duration::new(2, 500_000_000)).unwrap();
    ///
    /// // Two and a half seconds, exactly: a half is a dyadic rational.
    /// assert_eq!(measured, Seconds::from_millis(2_500).unwrap());
    /// ```
    ///
    /// # Why not `as_secs_f64`
    ///
    /// A [`Duration`](std::time::Duration) already holds integer seconds and integer
    /// nanoseconds, so it is exact. Going through `as_secs_f64` rounds it to a float
    /// and then rounds that to [`Fixed`], and two roundings do not always compose to
    /// the nearest value. Here the seconds scale exactly and only the nanoseconds are
    /// divided, once.
    ///
    /// # Range
    ///
    /// Every [`Duration`](std::time::Duration) fits, so this never actually returns
    /// [`None`]: the standard library's longest is about `1.8e19` seconds and this
    /// reaches `4.0e28`. The [`Option`] is the type's contract rather than a case to
    /// plan for.
    ///
    /// The nanoseconds round to the nearest step, as [`Seconds::from_nanos`] does.
    pub const fn from_duration(duration: std::time::Duration) -> Option<Self> {
        let Some(whole) = Fixed::from_integer_i128(duration.as_secs() as i128) else {
            return None;
        };

        let scaled: u128 =
            ((duration.subsec_nanos() as u128) << Fixed::FRACTION_BITS) + 500_000_000;
        let fraction: i128 = (scaled / 1_000_000_000) as i128;

        match whole.to_bits().checked_add(fraction) {
            Some(bits) => Some(Self(Fixed::from_bits(bits))),
            None => None,
        }
    }

    /// The duration as the engine's own scalar, in seconds.
    ///
    /// This is the value, not a conversion: the type is a [`Fixed`] with a meaning
    /// attached.
    pub const fn fixed(self) -> Fixed {
        self.0
    }

    /// The duration as an `f64`, for a boundary that needs one.
    ///
    /// For logging, a graphics API or anything outside the engine's own mathematics.
    /// Inside it, prefer [`Seconds::fixed`] and keep the arithmetic exact.
    pub fn to_f64(self) -> f64 {
        self.0.to_f64()
    }

    /// Whether no time at all has passed.
    pub const fn is_zero(self) -> bool {
        self.0.to_bits() == 0
    }

    /// How many times a second this duration would fit, or `None` at zero.
    ///
    /// # Question
    ///
    /// "That frame took this long. What frame rate is that?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    /// use voxel_world::time::Seconds;
    ///
    /// // A sixty-fourth of a second, which is dyadic and so exact both ways.
    /// let frame = Seconds::from_micros(15_625).expect("in range");
    ///
    /// assert_eq!(frame.frequency(), Some(Fixed::from_integer(64)));
    /// assert_eq!(Seconds::ZERO.frequency(), None, "no rate without a duration");
    /// ```
    ///
    /// Zero is reported rather than divided by: a frame that took no time has no frame
    /// rate, and an infinity would only travel further before causing trouble.
    pub fn frequency(self) -> Option<Fixed> {
        Fixed::ONE.checked_div(self.0)
    }

    /// The shorter of the two.
    ///
    /// # Why clamping is a method and not part of construction
    ///
    /// A measured duration should say what was measured. A debugger pause, a window
    /// drag or a suspended machine produce a genuinely enormous frame, and whether to
    /// cap it is the game loop's policy, not the measurement's. So nothing is clamped
    /// on the way in and this is here for the caller to apply deliberately.
    pub fn min(self, other: Self) -> Self {
        Self(if self.0 <= other.0 { self.0 } else { other.0 })
    }

    /// The longer of the two.
    pub fn max(self, other: Self) -> Self {
        Self(if self.0 >= other.0 { self.0 } else { other.0 })
    }

    /// This duration, or `maximum` if it is longer.
    ///
    /// The deliberate cap described on [`Seconds::min`].
    pub fn clamped(self, maximum: Self) -> Self {
        self.min(maximum)
    }

    /// Both durations together, or `None` if the sum leaves the range.
    ///
    /// Never wraps. [`Add`] is the saturating form, matching how [`TickDuration`]
    /// behaves; this is for a caller who would rather hear about it.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0
            .to_bits()
            .checked_add(other.0.to_bits())
            .map(|bits| Self(Fixed::from_bits(bits)))
    }

    /// What is left after taking `other` away, never below zero.
    ///
    /// Clamped at zero because a duration is non-negative by definition, so the
    /// alternative is not a negative duration but no answer at all.
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0).max(Fixed::ZERO))
    }

    /// This duration scaled by a factor, or `None` if the factor is negative or the
    /// result leaves the range.
    pub fn checked_scale(self, factor: Fixed) -> Option<Self> {
        if factor.to_bits() < 0 {
            return None;
        }

        self.0.checked_mul(factor).map(Self)
    }

    /// How many whole times `other` fits into this, with what is left over.
    ///
    /// `None` for a zero divisor. The mechanism a fixed-step loop is built on: the
    /// quotient is how many ticks are owed, the remainder is how far into the next one
    /// the clock already is.
    pub fn divide_whole(self, other: Self) -> Option<(u64, Self)> {
        if other.is_zero() {
            return None;
        }

        let quotient: Fixed = self.0.checked_div(other.0)?;
        // An arithmetic shift floors, and both sides are non-negative.
        let whole: i128 = quotient.to_bits() >> Fixed::FRACTION_BITS;
        let whole: u64 = u64::try_from(whole).unwrap_or(u64::MAX);

        let consumed: Fixed = other.0.saturating_mul(Fixed::from_integer_i128(whole as i128)?);

        Some((whole, self.saturating_sub(Self(consumed))))
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
/// # Question
///
/// "*When* is this?"
///
/// # Why it is not a `u64`
///
/// Because almost every number in a simulation is a `u64`, and a moment in time is
/// interchangeable with none of them. The type is mostly here to stop that:
///
/// ```compile_fail
/// use voxel_world::time::Tick;
/// use voxel_world::random::seed::Seed;
///
/// // A bare count is not a moment, so this will not compile.
/// Seed::from_integer(1u64).at_tick(1_000);
/// ```
///
/// ```compile_fail
/// use voxel_world::time::Tick;
///
/// // Two moments do not add. The span between them is a TickDuration, and a
/// // moment plus a span is what makes a later moment.
/// let _ = Tick::new(10) + Tick::new(2);
/// ```
///
/// A moment and a duration are the distinction worth drawing: tick 1000 and "a
/// thousand ticks" are different things, and only one of them can be multiplied by a
/// velocity or compared against an interval.
///
/// # Companions
///
/// - [`TickDuration`] — *how long*, the difference between two moments.
/// - [`TickRate`] — the only bridge to wall-clock [`Seconds`].
///
/// # Example
///
/// ```
/// use voxel_world::time::{Tick, TickDuration};
///
/// let mut now = Tick::ORIGIN;
///
/// for _ in 0..120 {
///     now.increment();
/// }
///
/// assert_eq!(now, Tick::new(120));
/// assert_eq!(now - Tick::new(100), TickDuration::new(20));
/// assert!(now.is_every(TickDuration::new(60)), "a round second at 60 a second");
/// ```
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

    /// Moves this moment on by one tick, in place.
    ///
    /// # Question
    ///
    /// "The simulation just ran a step. How do I say so?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::time::Tick;
    ///
    /// let mut now = Tick::ORIGIN;
    /// now.increment();
    ///
    /// assert_eq!(now, Tick::new(1));
    /// ```
    ///
    /// # Why this exists beside [`Tick::next`]
    ///
    /// `next` returns a later moment and leaves this one alone, which is what reading
    /// code wants. A loop that *owns* the clock wants to advance it, and writing
    /// `now = now.next()` invites the mistake of forgetting the assignment — a bug that
    /// compiles and leaves the world frozen at tick zero. This cannot be written
    /// wrongly.
    ///
    /// Saturating, as every other advance here is: see the type's own documentation.
    pub const fn increment(&mut self) {
        *self = self.next();
    }

    /// Moves this moment on by `duration`, in place.
    pub const fn advance_by(&mut self, duration: TickDuration) {
        *self = self.advanced_by(duration);
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

/// Advances a moment in place; see [`Tick::advance_by`].
impl AddAssign<TickDuration> for Tick {
    fn add_assign(&mut self, duration: TickDuration) {
        self.advance_by(duration);
    }
}

/// Two durations add to a longer one, saturating as the rest of this module does.
impl Add for TickDuration {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

/// Accumulates one duration into another; see [`Add`].
impl AddAssign for TickDuration {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

/// Two real durations add to a longer one, saturating rather than wrapping — the same
/// philosophy as [`TickDuration`], and for the same reason: time running backwards is
/// never the answer the caller wanted.
impl Add for Seconds {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

/// Accumulates one real duration into another; see [`Add`].
impl AddAssign for Seconds {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

/// The time between two durations, clamped at zero; see [`Seconds::saturating_sub`].
impl Sub for Seconds {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
}

/// Removes one real duration from another; see [`Sub`].
impl SubAssign for Seconds {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

/// Authoritative simulation time elapsed since this system last ran, in seconds.
///
/// # Question
///
/// "How much simulation time passed since this update last happened?"
///
/// # Why this is not [`FrameDelta`](crate::units::frame::FrameDelta)
///
/// They can hold the identical number of seconds and still must not be interchanged,
/// because they come from different clocks with different authority:
///
/// | | measured by | authority |
/// |---|---|---|
/// | [`UpdateDelta`] | counting ticks, through a [`TickRate`] | **authoritative** — may change the world |
/// | [`FrameDelta`](crate::units::frame::FrameDelta) | the platform's clock | presentation only — must not |
///
/// A system that integrated authoritative physics over a frame delta would make the
/// world's contents depend on the frame rate, and two players would diverge. There is
/// deliberately no conversion either way; both reach [`Seconds`] if something genuinely
/// needs a plain duration.
///
/// # Why this is not [`TickDuration`]
///
/// A tick count does not know how long it is. `TickDuration::new(4)` is a fifteenth of
/// a second at 60 ticks a second and a fifth at 20 — so it becomes a duration only once
/// a [`TickRate`] says which. `UpdateDelta` is the result of that conversion, which is
/// why it can be multiplied by a velocity and `TickDuration` cannot.
///
/// # Example
///
/// ```
/// use voxel_world::math::Fixed;
/// use voxel_world::spatial::kinematics::Velocity3;
/// use voxel_world::time::{TickDuration, TickRate, UpdateDelta};
///
/// // A system running at 15 Hz under a 60 Hz world: four ticks per update.
/// let rate = TickRate::new(60).expect("positive");
/// let delta = UpdateDelta::from_ticks(TickDuration::new(4), rate);
///
/// assert_eq!(delta.fixed(), Fixed::from_ratio(1, 15).unwrap());
///
/// // And it integrates like any other duration. Half a second is a binary fraction, so
/// // this one is exact; a fifteenth is not, and lands a hair under.
/// let half = UpdateDelta::from_ticks(TickDuration::new(30), rate);
/// let velocity = Velocity3::new(Fixed::from_integer(30), Fixed::ZERO, Fixed::ZERO);
///
/// assert_eq!((velocity * half).as_vector().x, Fixed::from_integer(15));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct UpdateDelta(Fixed);

impl UpdateDelta {
    /// No simulation time at all.
    pub const ZERO: Self = Self(Fixed::ZERO);

    /// The simulation time `duration` represents at `rate`.
    ///
    /// # Question
    ///
    /// "Four ticks went by. How long is that?"
    ///
    /// The ordinary way to make one. Exact and integer throughout: `count / per_second`
    /// as a [`Fixed`] ratio, with no float anywhere, so the same ticks give the same
    /// delta on every target.
    ///
    /// Cannot fail: the longest `u64` of ticks is far inside the range [`Fixed`] holds.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    /// use voxel_world::time::{TickDuration, TickRate, UpdateDelta};
    ///
    /// let rate = TickRate::new(60).expect("positive");
    ///
    /// assert_eq!(
    ///     UpdateDelta::from_ticks(TickDuration::new(1), rate).fixed(),
    ///     Fixed::from_ratio(1, 60).unwrap(),
    /// );
    /// assert_eq!(
    ///     UpdateDelta::from_ticks(TickDuration::new(2), rate).fixed(),
    ///     Fixed::from_ratio(1, 30).unwrap(),
    /// );
    /// assert_eq!(UpdateDelta::from_ticks(TickDuration::ZERO, rate), UpdateDelta::ZERO);
    /// ```
    pub fn from_ticks(duration: TickDuration, rate: TickRate) -> Self {
        Self(rate.seconds_in(duration).fixed())
    }

    /// The simulation time between two moments at `rate`.
    ///
    /// Zero when `earlier` is not actually earlier, since [`Tick::ticks_since`] is
    /// never negative.
    pub fn between(earlier: Tick, later: Tick, rate: TickRate) -> Self {
        Self::from_ticks(later.ticks_since(earlier), rate)
    }

    /// The simulation time an accumulated duration represents.
    ///
    /// # Question
    ///
    /// "I have been adding up each tick's length. How do I hand that to a system?"
    ///
    /// Infallible, since [`Seconds`] is already non-negative. This is the constructor
    /// for a clock that **accumulates** real time rather than deriving it from a tick
    /// count — which is what any world whose tick rate can vary has to do, because
    /// `ticks × current rate` is wrong the moment the rate has changed during the span.
    pub const fn from_accumulated(elapsed: Seconds) -> Self {
        Self(elapsed.fixed())
    }

    /// A delta from a fixed-point count of seconds, or `None` if negative.
    ///
    /// The escape hatch for a system whose interval is not a whole number of ticks.
    /// Prefer [`UpdateDelta::from_ticks`] or [`UpdateDelta::from_accumulated`], neither
    /// of which can be given a value the simulation did not actually take.
    pub const fn new(seconds: Fixed) -> Option<Self> {
        if seconds.to_bits() < 0 {
            return None;
        }

        Some(Self(seconds))
    }

    /// The delta as the engine's own scalar, in seconds.
    pub const fn fixed(self) -> Fixed {
        self.0
    }

    /// The delta as a general real duration.
    ///
    /// One way only, as with [`FrameDelta`](crate::units::frame::FrameDelta): an
    /// arbitrary duration is not what this update actually took.
    pub const fn seconds(self) -> Seconds {
        Seconds::from_non_negative(self.0)
    }

    /// Whether no simulation time passed.
    pub const fn is_zero(self) -> bool {
        self.0.to_bits() == 0
    }

    /// Both together, saturating rather than wrapping.
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

/// Saturating, as the rest of this module is.
impl Add for UpdateDelta {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl AddAssign for UpdateDelta {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

/// An update's elapsed time is a real duration; the reverse is not provided.
impl From<UpdateDelta> for Seconds {
    fn from(delta: UpdateDelta) -> Self {
        delta.seconds()
    }
}

impl fmt::Display for UpdateDelta {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}s", self.0)
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
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::time::{Seconds, TickRate};
    ///
    /// let rate = TickRate::new(64).expect("positive");
    ///
    /// // A sixty-fourth of a second, exactly: the denominator is a power of two.
    /// assert_eq!(rate.tick_length(), Seconds::from_micros(15_625).unwrap());
    /// ```
    ///
    /// A rate whose reciprocal is not a dyadic rational — 60, say — rounds to the
    /// nearest `2^-32` of a second, deterministically and identically on every target.
    pub fn tick_length(self) -> Seconds {
        Seconds(
            Fixed::from_ratio(1, i128::from(self.per_second()))
                .expect("a reciprocal of a positive rate is within range"),
        )
    }

    /// How much wall-clock time has passed at a given moment, counting from
    /// the world's origin.
    pub fn seconds_at(self, tick: Tick) -> Seconds {
        self.seconds_in(TickDuration::new(tick.count()))
    }

    /// How long a duration of ticks lasts in seconds.
    ///
    /// Cannot overflow: the longest tick count a `u64` holds is under `2^64` seconds
    /// even at one tick a second, and [`Seconds`] reaches `2^95`.
    pub fn seconds_in(self, duration: TickDuration) -> Seconds {
        Seconds(
            Fixed::from_ratio(
                i128::from(duration.count()),
                i128::from(self.per_second()),
            )
            .expect("a u64 tick count of seconds is within range"),
        )
    }

    /// How many whole ticks fit in a stretch of seconds.
    ///
    /// Rounds down, so a partial tick is not counted, and saturates rather than
    /// wrapping if the answer would exceed the tick counter.
    pub fn ticks_in(self, seconds: Seconds) -> TickDuration {
        let Some(scaled) = seconds
            .fixed()
            .checked_mul(Fixed::from_integer(i64::from(self.per_second())))
        else {
            return TickDuration::new(u64::MAX);
        };

        // An arithmetic shift floors, and the value is non-negative.
        let whole: i128 = scaled.to_bits() >> Fixed::FRACTION_BITS;

        TickDuration::new(u64::try_from(whole).unwrap_or(u64::MAX))
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
        write!(f, "{}s", self.0)
    }
}
/// As `<n> ticks/s`.
impl fmt::Display for TickRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ticks/s", self.per_second())
    }
}
