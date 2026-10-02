//! Presentation timing: how long the last frame took, and where drawing sits between
//! two simulation states.
//!
//! # Two clocks, and why they are not the same clock
//!
//! | | what it measures | role |
//! |---|---|---|
//! | [`Tick`], [`TickDuration`] | simulation steps | **authoritative** world chronology, chosen by the simulation |
//! | [`FrameDelta`] | real seconds, for one frame | presentation: what the platform measured |
//! | [`Seconds`] | real seconds, in general | either, wherever a physical duration is wanted |
//! | [`FrameIndex`] | nothing — it is a counter | presentation: *which* frame, not how long |
//! | [`TickAlpha`] | nothing — it is a ratio | presentation: *where* to draw between two states |
//!
//! Only the first two rows are durations. A [`FrameIndex`] is a sequence number and a
//! [`TickAlpha`] is dimensionless; neither is a quantity of time, and neither can be
//! integrated over.
//!
//! The world's state advances in whole ticks and nothing else. A frame time is a
//! *measurement* of how long the machine took, which varies with the window being
//! dragged, a shader compiling or a laptop throttling — none of which should change
//! what the world becomes. Driving physics from it would make the simulation depend on
//! the frame rate, and two players would diverge.
//!
//! So frame time is for **presentation**: camera motion, visual interpolation, UI
//! animation, diagnostics — and for feeding the accumulator that decides *how many*
//! authoritative ticks are now owed.
//!
//! # The loop this exists for
//!
//! ```text
//! accumulate the measured frame time
//!         ↓
//! while a whole tick is owed: simulate one tick
//!         ↓
//! what is left over, as a fraction of a tick, is the interpolation alpha
//!         ↓
//! draw the world between the last two states at that alpha
//! ```
//!
//! [`FrameClock`] is that mechanism. It deliberately holds no policy: it will report
//! a thousand owed ticks after a thousand-second stall and let the caller decide
//! whether to run them, cap them or discard them, because that choice differs between
//! a client that wants to be current and a server that wants to miss nothing.
//!
//! # Three kinds of time, and the question each answers
//!
//! All three are about time or progress, all three are small and cheap, and none is
//! interchangeable with another:
//!
//! | type | question | backed by |
//! |---|---|---|
//! | [`FrameDelta`] | "how much real time passed since the last frame?" | [`Fixed`] seconds |
//! | [`TickAlpha`] | "where do I draw between two simulation states?" | [`Unit`], a ratio |
//! | [`TickDuration`] | "how many simulation steps passed?" | a count of ticks |
//!
//! ```text
//! FrameDelta  != TickAlpha      one is seconds, the other is dimensionless
//! FrameDelta  != TickDuration   one is measured real time, the other is counted ticks
//! TickAlpha   != TickDuration   one is a fraction of a tick, the other is whole ticks
//! ```
//!
//! The compiler enforces that. A velocity multiplied by a `FrameDelta` is a
//! displacement; multiplied by a `TickAlpha` or a `TickDuration` it does not compile,
//! because neither of those is a number of seconds.

use crate::math::{Fixed, Unit};
use crate::units::time::{Seconds, Tick, TickDuration, TickRate};
use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

// ---------------------------------------------------------------------------
// How much real time this frame took
// ---------------------------------------------------------------------------

/// Elapsed real time between the previous presented frame and this one, in seconds.
///
/// # Question
///
/// "How long did the last frame take?"
///
/// The quantity engine code calls *delta*, given its own type so that nothing else can
/// be passed where it belongs and it cannot be passed anywhere else.
///
/// # What it is not
///
/// - Not a [`TickDuration`]. That counts authoritative simulation steps; this measures
///   real time, which varies with the machine and must never decide what the world
///   becomes.
/// - Not a [`TickAlpha`]. That is a dimensionless fraction of one tick; this is seconds.
///
/// # Storage
///
/// A [`Fixed`] directly, not a wrapped [`Seconds`]: it is a first-class quantity, and
/// `frame.delta()` should reach its value without a second unwrapping. [`Seconds`] is
/// still the general real duration, and [`FrameDelta::seconds`] converts to it —
/// deliberately one-way, since an arbitrary duration is not "this frame's delta" until
/// somebody says it is.
///
/// # Example
///
/// ```
/// use voxel_world::math::Fixed;
/// use voxel_world::spatial::kinematics::Velocity3;
/// use voxel_world::units::frame::FrameDelta;
///
/// let delta = FrameDelta::from_millis(16).expect("in range");
/// let velocity = Velocity3::new(Fixed::from_integer(60), Fixed::ZERO, Fixed::ZERO);
///
/// // Sixty units a second, for a sixteen-millisecond frame.
/// let step = velocity * delta;
///
/// assert!(step.as_vector().x > Fixed::ZERO);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct FrameDelta(Fixed);

impl FrameDelta {
    /// No time at all, which is what the frame before the first one took.
    pub const ZERO: Self = Self(Fixed::ZERO);

    /// A delta from a fixed-point count of seconds, or `None` if negative.
    ///
    /// Negative time is not a duration, so it is refused rather than clamped: it means
    /// the measurement went wrong, and a silent zero would hide that.
    pub const fn new(seconds: Fixed) -> Option<Self> {
        if seconds.to_bits() < 0 {
            return None;
        }

        Some(Self(seconds))
    }

    /// A delta of whole seconds, or `None` if negative or too large.
    pub const fn from_seconds(whole: i64) -> Option<Self> {
        match Seconds::from_seconds(whole) {
            Some(value) => Some(Self(value.fixed())),
            None => None,
        }
    }

    /// A delta in whole milliseconds, or `None` if too large.
    pub const fn from_millis(millis: u64) -> Option<Self> {
        match Seconds::from_millis(millis) {
            Some(value) => Some(Self(value.fixed())),
            None => None,
        }
    }

    /// A delta in whole microseconds, or `None` if too large.
    pub const fn from_micros(micros: u64) -> Option<Self> {
        match Seconds::from_micros(micros) {
            Some(value) => Some(Self(value.fixed())),
            None => None,
        }
    }

    /// A delta in whole nanoseconds, or `None` if too large.
    pub const fn from_nanos(nanos: u64) -> Option<Self> {
        match Seconds::from_nanos(nanos) {
            Some(value) => Some(Self(value.fixed())),
            None => None,
        }
    }

    /// A delta from the platform's own measurement, without touching a float.
    ///
    /// # Question
    ///
    /// "The window system measured a `Duration`. How does it become a delta?"
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    /// use voxel_world::units::frame::FrameDelta;
    ///
    /// let measured = FrameDelta::from_duration(Duration::new(0, 16_666_667)).unwrap();
    ///
    /// assert!(!measured.is_zero());
    /// ```
    ///
    /// A [`Duration`](std::time::Duration) holds integer seconds and integer
    /// nanoseconds, so the whole seconds scale exactly and only the nanoseconds are
    /// divided — once, to the nearest step. No `as_secs_f64`, so nothing rounds twice.
    ///
    /// Every `Duration` fits, so this never returns [`None`] in practice; the
    /// [`Option`] is the type's contract.
    pub const fn from_duration(duration: std::time::Duration) -> Option<Self> {
        match Seconds::from_duration(duration) {
            Some(value) => Some(Self(value.fixed())),
            None => None,
        }
    }

    /// The delta as the engine's own scalar, in seconds.
    ///
    /// The value, not a conversion.
    pub const fn fixed(self) -> Fixed {
        self.0
    }

    /// The delta as a general real duration.
    ///
    /// One way only: an arbitrary [`Seconds`] is not this frame's delta until something
    /// measures it, so there is no implicit conversion back.
    pub const fn seconds(self) -> Seconds {
        // Infallible: every constructor refuses a negative value, so the invariant
        // `Seconds` needs is already established. There is deliberately no fallback —
        // quietly turning an impossible failure into a zero duration would be a worse
        // answer than any, and would hide whatever broke the invariant.
        Seconds::from_non_negative(self.0)
    }

    /// The delta as an `f64`, for a boundary outside the engine.
    pub fn to_f64(self) -> f64 {
        self.0.to_f64()
    }

    /// Whether no time at all passed.
    pub const fn is_zero(self) -> bool {
        self.0.to_bits() == 0
    }

    /// How many frames a second this delta would be, or `None` at zero.
    pub fn frequency(self) -> Option<Fixed> {
        self.seconds().frequency()
    }

    /// The shorter of the two.
    ///
    /// Capping a pathological frame — a debugger pause, a dragged window, a suspended
    /// machine — is the game loop's policy, so nothing is clamped on the way in.
    pub fn min(self, other: Self) -> Self {
        Self(if self.0 <= other.0 { self.0 } else { other.0 })
    }

    /// This delta, or `maximum` if it is longer.
    pub fn clamped(self, maximum: Self) -> Self {
        self.min(maximum)
    }

    /// Both together, or `None` if the sum leaves the range.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0
            .to_bits()
            .checked_add(other.0.to_bits())
            .map(|bits| Self(Fixed::from_bits(bits)))
    }

    /// What is left after taking `other` away, never below zero.
    ///
    /// Clamped rather than checked because the alternative is not a negative delta but
    /// no answer at all — a duration is non-negative by definition.
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0).max(Fixed::ZERO))
    }

    /// Scaled by a factor, or `None` if the factor is negative or the result overflows.
    ///
    /// # Why there is no `Mul<Fixed>` operator
    ///
    /// A factor is an unrestricted [`Fixed`], and a negative one would produce a
    /// negative delta — which this type exists to make impossible. An operator cannot
    /// report that, so scaling is a checked method and the caller sees the refusal.
    pub fn checked_scale(self, factor: Fixed) -> Option<Self> {
        if factor.to_bits() < 0 {
            return None;
        }

        self.0.checked_mul(factor).map(Self)
    }
}

/// Saturating, as the rest of the time module is: a wrapped duration is never the
/// answer anybody wanted.
impl Add for FrameDelta {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

impl AddAssign for FrameDelta {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

/// Clamped at zero; see [`FrameDelta::saturating_sub`].
impl Sub for FrameDelta {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
}

impl SubAssign for FrameDelta {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

/// A frame's delta is a real duration; the reverse is deliberately not provided.
impl From<FrameDelta> for Seconds {
    fn from(delta: FrameDelta) -> Self {
        delta.seconds()
    }
}

/// From the platform's measurement; see [`FrameDelta::from_duration`].
impl TryFrom<std::time::Duration> for FrameDelta {
    type Error = DeltaOutOfRange;

    fn try_from(duration: std::time::Duration) -> Result<Self, Self::Error> {
        Self::from_duration(duration).ok_or(DeltaOutOfRange)
    }
}

/// A measured duration longer than a [`FrameDelta`] can hold.
///
/// Unreachable with a `std::time::Duration`, whose longest is about `1.8e19` seconds
/// against this type's `4.0e28`. It exists so [`TryFrom`] has an error type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeltaOutOfRange;

impl fmt::Display for DeltaOutOfRange {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the measured duration is longer than a frame delta can hold")
    }
}

impl std::error::Error for DeltaOutOfRange {}

impl fmt::Display for FrameDelta {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}s", self.0)
    }
}

// ---------------------------------------------------------------------------
// Which frame
// ---------------------------------------------------------------------------

/// Which presented frame this is, counted from the first.
///
/// # Question
///
/// "*Which* frame is this?"
///
/// # Not a [`Tick`]
///
/// A tick is a moment in the world's own chronology, and the world has exactly as many
/// of them as it has simulated. A frame index counts *pictures*, which arrive at
/// whatever rate the machine manages — more than one per tick on fast hardware, fewer
/// on slow. Two worlds that simulated identically will share every tick and agree about
/// no frame index at all, which is why they are separate types.
///
/// Useful for diagnostics, profiling, statistics and temporal rendering that needs to
/// know whether it has already run this frame.
///
/// # Example
///
/// ```
/// use voxel_world::units::frame::FrameIndex;
///
/// let mut frame = FrameIndex::FIRST;
/// frame.increment();
///
/// assert_eq!(frame.count(), 1);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameIndex(u64);

impl FrameIndex {
    /// The first frame presented.
    ///
    /// # The convention
    ///
    /// Zero-based: the first [`Frame`] [`FrameClock::begin_frame`] returns carries this
    /// index, the second carries one, and so on. A frame index is therefore *which*
    /// frame, not *how many* frames — [`FrameClock::frames_presented`] answers that.
    ///
    /// This used to disagree with itself: `FIRST` was zero while the clock incremented
    /// before returning, so the first frame came back as one. Either convention would
    /// do; having both was the bug.
    pub const FIRST: Self = Self(0);

    /// A frame index from a count.
    pub const fn new(count: u64) -> Self {
        Self(count)
    }

    /// How many frames have been presented before this one.
    pub const fn count(self) -> u64 {
        self.0
    }

    /// The next frame.
    ///
    /// Saturating, as [`Tick`] is: a counter that wrapped would make a later frame
    /// compare as earlier, and at sixty a second `u64` lasts nine billion years.
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }

    /// Moves to the next frame, in place.
    pub const fn increment(&mut self) {
        *self = self.next();
    }
}

impl fmt::Display for FrameIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "frame {}", self.0)
    }
}

// ---------------------------------------------------------------------------
// How far between ticks
// ---------------------------------------------------------------------------

/// How far presentation lies between the last completed tick and the next one.
///
/// # Question
///
/// "The world is at tick 400 and a bit. How much of a bit?"
///
/// # What it is for
///
/// Drawing the world exactly at its last tick makes motion stutter, because the picture
/// is up to one tick stale and by a varying amount. Interpolating between the previous
/// and current states by this fraction removes that, without the simulation itself
/// knowing anything about it.
///
/// ```text
/// 0.0   draw the previous state exactly
/// 0.5   halfway between the two
/// →1.0  arbitrarily close to the current state
/// ```
///
/// # The convention: `[0, 1)`, never exactly one
///
/// Because of what it is made from. The accumulator has every whole tick taken out of
/// it before this is read, so what remains is strictly less than one tick's worth —
/// reaching one would mean a tick was owed and had not been run. An alpha of exactly
/// one would also be a second name for an alpha of zero one tick later, and two names
/// for one moment is how off-by-one errors start.
///
/// [`Unit`] is the representation because it is the crate's exact `[0, 1]` type; this
/// simply never uses its top value. [`TickAlpha::ALMOST_ONE`] is as far as it goes.
///
/// One caveat, because it would otherwise be a surprise: [`TickAlpha::fixed`] converts
/// onto [`Fixed`]'s coarser grid of `2^-32`, and a `Unit` within `2^-63` of one rounds
/// to exactly one there. So the half-open guarantee holds in the `Unit` — which is the
/// value — and not necessarily in that conversion. It is harmless: an interpolation
/// weight of one yields the current state, which is precisely the limit the alpha was
/// approaching.
///
/// # It is not a duration
///
/// It is a ratio. Multiplying a velocity by it would be meaningless — that needs
/// [`Seconds`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TickAlpha(Unit);

impl TickAlpha {
    /// Exactly at the previous tick.
    pub const ZERO: Self = Self(Unit::ZERO);

    /// Halfway between the two states.
    pub const HALF: Self = Self(Unit::HALF);

    /// As close to the next tick as the convention allows.
    pub const ALMOST_ONE: Self = Self(Unit::ALMOST_ONE);

    /// An alpha from a unit value, which cannot be out of range by construction.
    ///
    /// A value of exactly one is brought down to [`TickAlpha::ALMOST_ONE`], keeping the
    /// half-open convention the type documents.
    pub const fn new(progress: Unit) -> Self {
        if progress.to_bits() >= Unit::ONE.to_bits() {
            return Self::ALMOST_ONE;
        }

        Self(progress)
    }

    /// The alpha for an accumulator that has had its whole ticks taken out.
    ///
    /// `None` for a tick length of zero, which names no interval to be partway through.
    pub fn from_remainder(remaining: Seconds, tick_length: Seconds) -> Option<Self> {
        if tick_length.is_zero() {
            return None;
        }

        let ratio: Fixed = remaining.fixed().checked_div(tick_length.fixed())?;

        Some(Self::new(Unit::from_fixed_clamped(ratio)))
    }

    /// The progress as the crate's exact unit value.
    pub const fn unit(self) -> Unit {
        self.0
    }

    /// The progress as the engine's scalar, for feeding an interpolation.
    ///
    /// Pairs with [`Vector3::lerp`](crate::math::Vector3::lerp) and
    /// [`PrecisePosition3::lerp`](crate::spatial::precise::PrecisePosition3::lerp),
    /// which both take a [`Fixed`] weight — so interpolation needs no new machinery
    /// here.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::{Fixed, Vector3};
    /// use voxel_world::units::frame::TickAlpha;
    ///
    /// let previous = Vector3::new(Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
    /// let current = Vector3::new(Fixed::from_integer(10), Fixed::ZERO, Fixed::ZERO);
    ///
    /// let drawn = previous.lerp(current, TickAlpha::HALF.fixed()).expect("in range");
    ///
    /// assert_eq!(drawn.x, Fixed::from_integer(5));
    /// ```
    pub fn fixed(self) -> Fixed {
        self.0.to_fixed()
    }
}

impl fmt::Display for TickAlpha {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// One frame's worth of timing
// ---------------------------------------------------------------------------

/// Everything worth knowing about the frame being presented.
///
/// # Question
///
/// "What should this frame draw, and how long did the last one take?"
///
/// An immutable snapshot of one presented frame's timing, produced by
/// [`FrameClock::begin_frame`]. It answers questions about that frame and changes
/// nothing: the clock keeps the state, this is the reading taken from it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Hash)]
pub struct Frame {
    index: FrameIndex,
    delta: FrameDelta,
    elapsed: Seconds,
    alpha: TickAlpha,
}

impl Frame {
    /// Which frame this is.
    pub const fn index(self) -> FrameIndex {
        self.index
    }

    /// How much real time passed since the previous frame.
    ///
    /// The canonical name, because `frame.delta()` is what engine code expects to mean
    /// exactly this. The measurement is uncapped; see [`FrameDelta::clamped`] for why
    /// capping it is the caller's decision.
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::math::Fixed;
    /// # use voxel_world::spatial::kinematics::{Acceleration3, Velocity3};
    /// # use voxel_world::spatial::precise::PrecisePosition3;
    /// # use voxel_world::time::TickRate;
    /// # use voxel_world::units::frame::{FrameClock, FrameDelta};
    /// # let mut clock = FrameClock::new(TickRate::new(60).unwrap());
    /// # let mut camera_position = PrecisePosition3::new(Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
    /// # let mut camera_velocity = Velocity3::ZERO;
    /// # let camera_acceleration = Acceleration3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);
    /// let frame = clock.begin_frame(FrameDelta::from_millis(16).unwrap());
    ///
    /// camera_velocity += camera_acceleration * frame.delta();
    /// camera_position += camera_velocity * frame.delta();
    /// ```
    pub const fn delta(self) -> FrameDelta {
        self.delta
    }

    /// How much real time passed since the previous frame, as a general duration.
    ///
    /// The same value as [`Frame::delta`]. Kept because an accumulator or a timeout
    /// wants a [`Seconds`] rather than a delta, and converting at every such call site
    /// reads worse than asking for the one that fits.
    pub const fn duration(self) -> Seconds {
        self.delta.seconds()
    }

    /// How much real time has passed since the clock started.
    pub const fn elapsed(self) -> Seconds {
        self.elapsed
    }

    /// How far between simulation states to draw.
    ///
    /// Only meaningful once the frame's owed ticks have been run; see
    /// [`FrameClock::take_tick`].
    pub const fn tick_alpha(self) -> TickAlpha {
        self.alpha
    }
}

// ---------------------------------------------------------------------------
// The accumulator
// ---------------------------------------------------------------------------

/// Turns measured real time into whole simulation ticks, and the leftover into an alpha.
///
/// # Question
///
/// "The frame took 17 milliseconds. How many ticks do I owe, and where do I draw?"
///
/// # Mechanism, not policy
///
/// This will happily report that two thousand ticks are owed. It will not decide what
/// to do about that, because the right answer differs: a client would rather drop the
/// backlog and be current, a server replaying a world would rather run every one. So
/// the loop stays with the caller:
///
/// ```
/// use voxel_world::time::TickRate;
/// use voxel_world::units::frame::{FrameClock, FrameDelta};
///
/// const MOST_PER_FRAME: u32 = 5;
///
/// let mut clock = FrameClock::new(TickRate::new(60).expect("positive"));
/// let frame = clock.begin_frame(FrameDelta::from_millis(100).expect("in range"));
///
/// // Zero-based: the first frame returned is FrameIndex::FIRST.
/// assert_eq!(frame.index(), voxel_world::units::frame::FrameIndex::FIRST);
///
/// // Six ticks are owed by a hundred milliseconds at sixty a second.
/// assert_eq!(clock.ready_ticks().count(), 6);
///
/// let mut run = 0;
/// while run < MOST_PER_FRAME && clock.take_tick().is_some() {
///     run += 1;
/// }
///
/// // The caller's cap, not the clock's.
/// assert_eq!(run, 5);
/// assert_eq!(clock.ready_ticks().count(), 1);
///
/// clock.discard_backlog();
/// assert_eq!(clock.ready_ticks().count(), 0);
/// # let _ = frame;
/// ```
///
/// # Determinism
///
/// Every step is integer and fixed-point: no `f64` takes part, so a recorded sequence
/// of frame durations replays to the same ticks on every target. What the *platform*
/// measures is of course not reproducible, which is exactly why the simulation advances
/// in ticks and not in frame times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameClock {
    rate: TickRate,
    tick_length: Seconds,
    accumulated: Seconds,
    elapsed: Seconds,
    now: Tick,
    /// The index the next frame will carry, so the first one carries
    /// [`FrameIndex::FIRST`].
    next_index: FrameIndex,
    presented: u64,
}

impl FrameClock {
    /// A clock at the world's origin, running at `rate`.
    pub fn new(rate: TickRate) -> Self {
        Self::starting_at(rate, Tick::ORIGIN)
    }

    /// A clock at `start`, for a world resumed rather than begun.
    pub fn starting_at(rate: TickRate, start: Tick) -> Self {
        Self {
            rate,
            tick_length: rate.tick_length(),
            accumulated: Seconds::ZERO,
            elapsed: Seconds::ZERO,
            now: start,
            next_index: FrameIndex::FIRST,
            presented: 0,
        }
    }

    /// The simulation rate this clock accumulates against.
    pub const fn rate(self) -> TickRate {
        self.rate
    }

    /// How long one tick lasts, which the clock holds rather than recomputing.
    pub const fn tick_length(self) -> Seconds {
        self.tick_length
    }

    /// The last tick the simulation completed.
    pub const fn now(self) -> Tick {
        self.now
    }

    /// Which frame was most recently presented, or [`None`] before the first.
    ///
    /// [`None`] rather than a zero that would be indistinguishable from the real first
    /// frame — which is the ambiguity the old contradictory convention had.
    pub const fn frame_index(self) -> Option<FrameIndex> {
        if self.presented == 0 {
            return None;
        }

        Some(FrameIndex::new(self.presented - 1))
    }

    /// How many frames have been presented.
    pub const fn frames_presented(self) -> u64 {
        self.presented
    }

    /// How much real time the clock has seen.
    pub const fn elapsed(self) -> Seconds {
        self.elapsed
    }

    /// Real time taken in but not yet spent on a tick.
    pub const fn accumulated(self) -> Seconds {
        self.accumulated
    }

    /// Takes in this frame's measured [`FrameDelta`] and reports the frame.
    ///
    /// Advances the frame index and the accumulator. Does **not** run any ticks — that
    /// is [`FrameClock::take_tick`], so the caller keeps the loop.
    ///
    /// The returned [`Frame`] carries [`FrameIndex::FIRST`] on the first call, one on
    /// the second, and so on.
    pub fn begin_frame(&mut self, delta: FrameDelta) -> Frame {
        let index: FrameIndex = self.next_index;

        self.next_index.increment();
        self.presented = self.presented.saturating_add(1);

        // The accumulator is a general real duration, since it holds time from many
        // frames rather than one frame's measurement.
        self.accumulated += delta.seconds();
        self.elapsed += delta.seconds();

        Frame {
            index,
            delta,
            elapsed: self.elapsed,
            alpha: self.tick_alpha(),
        }
    }

    /// How many whole ticks the accumulator can pay for right now.
    pub fn ready_ticks(self) -> TickDuration {
        TickDuration::new(
            self.accumulated
                .divide_whole(self.tick_length)
                .map_or(0, |(whole, _)| whole),
        )
    }

    /// Spends one tick's worth of accumulated time and reports the tick now reached.
    ///
    /// [`None`] when less than a whole tick is accumulated, which is what ends the
    /// caller's loop.
    pub fn take_tick(&mut self) -> Option<Tick> {
        if self.accumulated < self.tick_length || self.tick_length.is_zero() {
            return None;
        }

        self.accumulated -= self.tick_length;
        self.now.increment();

        Some(self.now)
    }

    /// Throws away every whole tick still owed, keeping the part-tick remainder.
    ///
    /// # Question
    ///
    /// "The machine was asleep for a minute. How do I stop the world chasing it?"
    ///
    /// The resynchronisation a client wants after a stall: the backlog is abandoned
    /// rather than simulated, so being current costs one frame instead of a minute. The
    /// remainder is kept so the interpolation alpha stays continuous.
    ///
    /// Returns how many ticks were dropped, which is worth logging — silently losing
    /// time is the sort of thing that should show up somewhere.
    pub fn discard_backlog(&mut self) -> TickDuration {
        let Some((whole, remainder)) = self.accumulated.divide_whole(self.tick_length) else {
            return TickDuration::ZERO;
        };

        self.accumulated = remainder;

        TickDuration::new(whole)
    }

    /// How far between the last completed tick and the next one the clock sits.
    ///
    /// Read it after draining the owed ticks; before that it saturates just below one,
    /// since a full tick's worth means a tick is owed rather than partly elapsed.
    pub fn tick_alpha(self) -> TickAlpha {
        TickAlpha::from_remainder(self.accumulated, self.tick_length).unwrap_or(TickAlpha::ZERO)
    }
}
