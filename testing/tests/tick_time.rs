//! The tick types: a moment, a duration, and the rate that converts either to seconds.
//!
//! # What these are really testing
//!
//! Mostly that the arithmetic is right. But the reason the types exist is the mistakes
//! they make impossible, and those cannot be tested from here — a test that adds a
//! `Tick` to a `Tick` does not fail, it fails to compile. Those live as `compile_fail`
//! examples in the crate's own documentation.

use voxel_world::time::{Seconds, Tick, TickDuration, TickRate};

// ---------------------------------------------------------------------------
// Moments
// ---------------------------------------------------------------------------

#[test]
fn a_moment_starts_at_the_origin_and_counts_from_there() {
    assert_eq!(Tick::ORIGIN.count(), 0);
    assert_eq!(Tick::new(7).count(), 7);
    assert_eq!(Tick::default(), Tick::ORIGIN);
}

#[test]
fn incrementing_advances_exactly_one_tick() {
    let mut now = Tick::ORIGIN;

    for expected in 1..=1_000u64 {
        now.increment();

        assert_eq!(now, Tick::new(expected));
    }

    // And agrees with the value-returning form it is built on.
    let mut counted = Tick::new(41);
    counted.increment();

    assert_eq!(counted, Tick::new(41).next());
}

#[test]
fn advancing_in_place_matches_advancing_by_value() {
    for step in [0u64, 1, 2, 100, 1_000_000] {
        let duration = TickDuration::new(step);

        let mut moved = Tick::new(500);
        moved.advance_by(duration);

        assert_eq!(moved, Tick::new(500).advanced_by(duration));

        // And through the operator, which is the same thing again.
        let mut assigned = Tick::new(500);
        assigned += duration;

        assert_eq!(assigned, moved);
    }
}

#[test]
fn time_saturates_rather_than_wrapping_round() {
    // A world that somehow reached the end of representable time should stop, not
    // find itself back at its own beginning.
    let last = Tick::new(u64::MAX);

    let mut edge = last;
    edge.increment();

    assert_eq!(edge, last, "a tick wrapped past the end of time");
    assert_eq!(last.advanced_by(TickDuration::new(1_000)), last);

    let mut duration = TickDuration::new(u64::MAX);
    duration += TickDuration::ONE;

    assert_eq!(duration, TickDuration::new(u64::MAX), "a duration wrapped");
}

#[test]
fn the_duration_between_two_moments_is_never_negative() {
    let early = Tick::new(10);
    let late = Tick::new(25);

    assert_eq!(late.ticks_since(early), TickDuration::new(15));
    assert_eq!(late - early, TickDuration::new(15));

    // The other way round is zero rather than a wrapped enormity, since a duration
    // here is unsigned.
    assert_eq!(early.ticks_since(late), TickDuration::ZERO);
    assert_eq!(early - late, TickDuration::ZERO);
}

// ---------------------------------------------------------------------------
// Schedules
// ---------------------------------------------------------------------------

#[test]
fn is_every_marks_the_multiples_and_nothing_else() {
    let every_three = TickDuration::new(3);

    for count in 0..30u64 {
        assert_eq!(
            Tick::new(count).is_every(every_three),
            count % 3 == 0,
            "tick {count} against every three",
        );
    }

    // An interval of zero names no schedule, so nothing is ever on it — including
    // the origin, which every other interval does match.
    for count in 0..10u64 {
        assert!(!Tick::new(count).is_every(TickDuration::ZERO));
    }
}

#[test]
fn a_once_a_second_schedule_fires_once_a_second() {
    // What the game loop uses it for, checked over a simulated minute.
    let rate = TickRate::new(60).expect("positive");
    let second = TickDuration::new(u64::from(rate.per_second()));

    let mut now = Tick::ORIGIN;
    let mut fired: u32 = 0;

    for _ in 0..3_600u32 {
        now.increment();

        if now.is_every(second) {
            fired += 1;
        }
    }

    assert_eq!(fired, 60, "sixty seconds of ticks should fire sixty times");
}

// ---------------------------------------------------------------------------
// Durations
// ---------------------------------------------------------------------------

#[test]
fn durations_accumulate() {
    let mut total = TickDuration::ZERO;

    for _ in 0..10 {
        total += TickDuration::new(7);
    }

    assert_eq!(total, TickDuration::new(70));
    assert_eq!(TickDuration::new(4) + TickDuration::new(5), TickDuration::new(9));
    assert_eq!(TickDuration::ONE.count(), 1);
}

// ---------------------------------------------------------------------------
// The rate, which is the only bridge to wall-clock time
// ---------------------------------------------------------------------------

#[test]
fn a_rate_converts_both_ways_consistently() {
    let rate = TickRate::new(60).expect("positive");

    assert_eq!(rate.per_second(), 60);

    // A sixtieth of a second is not a dyadic rational, so it rounds to the nearest
    // `2^-32` — within one step, and the same step on every target.
    let expected = voxel_world::math::Fixed::from_ratio(1, 60).expect("in range");
    assert_eq!(rate.tick_length().fixed(), expected);

    // A moment's wall-clock time, and the round trip back, with no float involved.
    let one_minute = Tick::new(3_600);

    assert_eq!(rate.seconds_at(one_minute), Seconds::from_seconds(60).unwrap());
    assert_eq!(
        rate.ticks_in(Seconds::from_seconds(60).expect("in range")),
        TickDuration::new(3_600),
    );
}

#[test]
fn a_rate_must_be_positive() {
    // A rate of zero would make a tick take forever and every conversion infinite.
    // Pausing is not a rate of zero; it is not advancing the clock.
    assert!(TickRate::new(0).is_none());
    assert!(TickRate::new(1).is_some());
}

// ---------------------------------------------------------------------------
// Where the types are used
// ---------------------------------------------------------------------------

#[test]
fn a_seed_derives_from_a_moment() {
    use voxel_world::random::seed::Seed;

    let world = Seed::from_integer(1u64).child("weather");

    // Different moments, different weather; the same moment, the same weather.
    let now = world.at_tick(Tick::new(1_000));
    let later = world.at_tick(Tick::new(1_001));

    assert_ne!(now, later);
    assert_eq!(now, world.at_tick(Tick::new(1_000)));

    // And a moment reached by incrementing is the same moment as one named outright,
    // so a loop's clock and a saved number agree.
    let mut walked = Tick::new(999);
    walked.increment();

    assert_eq!(world.at_tick(walked), now);
}

#[test]
fn a_position_moves_by_a_velocity_over_real_seconds() {
    use voxel_world::math::Fixed;
    use voxel_world::spatial::kinematics::{Displacement3, Velocity3};
    use voxel_world::spatial::precise::PrecisePosition3;

    let start = PrecisePosition3::new(Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
    let velocity = Velocity3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);

    // A velocity is per *second*, so the duration has to be seconds. A TickDuration
    // only becomes seconds once a TickRate says so, which is the whole invariant.
    let rate = TickRate::new(10).expect("positive");
    let over = rate.seconds_in(TickDuration::new(10));

    let moved = start.moved_by(velocity, over).expect("in range");

    assert_eq!(
        moved.displacement_from(start),
        Some(Displacement3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO)),
        "one unit a second for one second is one unit",
    );

    // Moving for no time stays put.
    assert_eq!(start.moved_by(velocity, Seconds::ZERO), Some(start));
}
