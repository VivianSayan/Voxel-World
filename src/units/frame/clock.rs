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
