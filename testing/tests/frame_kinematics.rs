//! Fixed-point real time, frame timing, and the kinematic quantities built on them.

use std::time::Duration;
use voxel_world::math::Fixed;
use voxel_world::spatial::kinematics::{Acceleration3, Displacement3, Force3, Mass, Velocity3};
use voxel_world::spatial::precise::PrecisePosition3;
use voxel_world::time::{Seconds, Tick, TickDuration, TickRate};
use voxel_world::units::frame::{FrameClock, FrameDelta, FrameIndex, TickAlpha};

fn fixed(whole: i64) -> Fixed {
    Fixed::from_integer(whole)
}

// ---------------------------------------------------------------------------
// Real duration, in fixed point
// ---------------------------------------------------------------------------

#[test]
fn a_duration_of_nothing_is_zero_and_stays_zero() {
    assert!(Seconds::ZERO.is_zero());
    assert_eq!(Seconds::ZERO + Seconds::ZERO, Seconds::ZERO);
    assert_eq!(Seconds::from_millis(0), Some(Seconds::ZERO));
    assert_eq!(Seconds::from_duration(Duration::ZERO), Some(Seconds::ZERO));

    // No frame rate without a frame, rather than a division by zero.
    assert_eq!(Seconds::ZERO.frequency(), None);
}

#[test]
fn whole_and_fractional_seconds_are_exact_where_they_can_be() {
    assert_eq!(Seconds::from_seconds(3).unwrap().fixed(), fixed(3));

    // Halves, quarters and eighths of a second are dyadic, so exact.
    assert_eq!(Seconds::from_millis(500).unwrap().fixed(), Fixed::ONE / 2);
    assert_eq!(Seconds::from_millis(250).unwrap().fixed(), Fixed::ONE / 4);
    assert_eq!(Seconds::from_micros(125_000).unwrap().fixed(), Fixed::ONE / 8);

    // And the scales agree with one another.
    assert_eq!(Seconds::from_millis(1_500), Seconds::from_micros(1_500_000));
    assert_eq!(Seconds::from_micros(2_000), Seconds::from_nanos(2_000_000));
    assert_eq!(Seconds::from_seconds(2), Seconds::from_millis(2_000));
}

#[test]
fn the_smallest_units_land_within_one_step() {
    // The step is `2^-32` seconds, about 233 picoseconds — *finer* than a nanosecond,
    // which is therefore about 4.29 steps and rounds to 4.
    let step = Fixed::from_bits(1);
    let one_nano = Seconds::from_nanos(1).expect("in range");

    assert_eq!(one_nano.fixed(), step * fixed(4), "a nanosecond is four steps");
    assert!(!one_nano.is_zero(), "a nanosecond does not vanish");

    // So the smallest representable duration is under a nanosecond, which is as fine
    // as any frame timer needs.
    assert!(Seconds::from_fixed(step).unwrap() < one_nano);

    // A microsecond is a thousand of those, comfortably exact in scale.
    let micro = Seconds::from_micros(1).expect("in range");
    assert!(micro.fixed() > step * fixed(4_000));
}

#[test]
fn a_platform_duration_enters_without_a_float() {
    // `Duration` holds integer seconds and nanoseconds, so the conversion is exact
    // wherever the value is dyadic — no `as_secs_f64` in the path.
    assert_eq!(
        Seconds::from_duration(Duration::new(2, 500_000_000)),
        Seconds::from_millis(2_500),
    );
    assert_eq!(
        Seconds::from_duration(Duration::new(0, 250_000_000)).unwrap().fixed(),
        Fixed::ONE / 4,
    );

    // A whole second of nanoseconds is a whole second.
    assert_eq!(
        Seconds::from_duration(Duration::from_nanos(1_000_000_000)),
        Seconds::from_seconds(1),
    );

    // Agreement with the named constructors across a spread of values.
    for millis in [0u64, 1, 7, 16, 17, 33, 999, 1_000, 1_001, 60_000] {
        assert_eq!(
            Seconds::from_duration(Duration::from_millis(millis)),
            Seconds::from_millis(millis),
            "{millis} ms disagreed",
        );
    }
}

#[test]
fn negative_time_cannot_be_constructed() {
    assert_eq!(Seconds::from_fixed(fixed(-1)), None);
    assert_eq!(Seconds::from_fixed(Fixed::from_bits(-1)), None);
    assert_eq!(Seconds::from_seconds(-1), None);

    assert!(Seconds::from_fixed(Fixed::ZERO).is_some(), "zero is a duration");

    // Subtraction clamps rather than going negative, since a duration cannot.
    let small = Seconds::from_millis(10).unwrap();
    let large = Seconds::from_millis(50).unwrap();

    assert_eq!(small - large, Seconds::ZERO);
    assert_eq!(large - small, Seconds::from_millis(40).unwrap());
}

#[test]
fn a_very_long_duration_is_handled_rather_than_wrapped() {
    // `Fixed` reaches about `2^95` seconds, so a `u64` of seconds fits easily.
    assert!(Seconds::from_seconds(i64::MAX / 2).is_some());

    // Addition saturates instead of wrapping round to nothing.
    assert_eq!(Seconds::MAX + Seconds::from_seconds(1).unwrap(), Seconds::MAX);
    assert_eq!(Seconds::MAX.checked_add(Seconds::MAX), None, "reported, not wrapped");

    // Every `std::time::Duration` fits: its longest is about 1.8e19 seconds and this
    // reaches 4.0e28. So there is no platform duration to refuse.
    assert!(Seconds::from_duration(Duration::new(u64::MAX, 0)).is_some());
    assert!(Seconds::from_duration(Duration::MAX).is_some());
}

#[test]
fn a_frame_duration_reports_its_frame_rate() {
    // A sixty-fourth of a second is dyadic, so this one is exact.
    assert_eq!(
        Seconds::from_micros(15_625).unwrap().frequency(),
        Some(fixed(64)),
    );
    assert_eq!(Seconds::from_millis(500).unwrap().frequency(), Some(fixed(2)));

    // Twenty milliseconds is a fiftieth, and a fifth is not dyadic, so the frame time
    // is the nearest `2^-32` to it and the rate inherits that. Near, not equal — which
    // is a fact about binary fractions, not about the division.
    for (millis, rate) in [(20u64, 50i64), (10, 100), (100, 10)] {
        let measured = Seconds::from_millis(millis).unwrap().frequency().expect("non-zero");

        assert!(
            (measured - fixed(rate)).abs() < Fixed::ONE / 1_000,
            "{millis} ms reported {measured}, expected about {rate}",
        );
    }
}

#[test]
fn clamping_is_the_callers_decision_not_the_measurements() {
    // A stall produces a genuinely enormous frame; the measurement keeps it.
    let stall = Seconds::from_seconds(30).unwrap();
    let cap = Seconds::from_millis(100).unwrap();

    assert_eq!(stall.clamped(cap), cap);
    assert_eq!(Seconds::from_millis(16).unwrap().clamped(cap).fixed(), Seconds::from_millis(16).unwrap().fixed());
}

// ---------------------------------------------------------------------------
// Rate conversion
// ---------------------------------------------------------------------------

#[test]
fn a_rate_converts_ticks_and_seconds_both_ways() {
    let rate = TickRate::new(64).expect("positive");

    // A sixty-fourth is a power of two, so exactly representable.
    assert_eq!(rate.tick_length(), Seconds::from_micros(15_625).unwrap());
    assert_eq!(rate.seconds_in(TickDuration::new(64)), Seconds::from_seconds(1).unwrap());
    assert_eq!(rate.seconds_at(Tick::new(128)), Seconds::from_seconds(2).unwrap());
    assert_eq!(rate.ticks_in(Seconds::from_seconds(3).unwrap()), TickDuration::new(192));

    // Partial ticks are not counted.
    let almost = rate.tick_length() - Seconds::from_nanos(1).unwrap();
    assert_eq!(rate.ticks_in(almost), TickDuration::ZERO);
}

#[test]
fn ticks_in_saturates_rather_than_wrapping() {
    let rate = TickRate::new(1_000).expect("positive");

    assert_eq!(rate.ticks_in(Seconds::MAX).count(), u64::MAX);
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

#[test]
fn a_frame_index_counts_and_saturates() {
    let mut frame = FrameIndex::FIRST;

    assert_eq!(frame.count(), 0);

    for expected in 1..=100u64 {
        frame.increment();
        assert_eq!(frame.count(), expected);
    }

    let last = FrameIndex::new(u64::MAX);
    assert_eq!(last.next(), last, "a frame index wrapped");
}

#[test]
fn an_alpha_stays_below_one() {
    assert_eq!(TickAlpha::ZERO.fixed(), Fixed::ZERO);
    assert_eq!(TickAlpha::HALF.fixed(), Fixed::ONE / 2);

    // Exactly one is brought down, keeping the half-open convention.
    assert_eq!(TickAlpha::new(voxel_world::math::Unit::ONE), TickAlpha::ALMOST_ONE);
    assert!(
        TickAlpha::ALMOST_ONE.unit() < voxel_world::math::Unit::ONE,
        "the guarantee is in the Unit, which is the value",
    );

    // Converting to `Fixed` moves onto a grid of `2^-32`, where a `Unit` within
    // `2^-63` of one rounds to exactly one. Documented, and harmless: an interpolation
    // weight of one gives the current state, the limit the alpha approaches.
    assert_eq!(TickAlpha::ALMOST_ONE.fixed(), Fixed::ONE);

    // From a remainder, which is how the clock makes one.
    let tick = Seconds::from_millis(20).unwrap();

    // A quarter of the way through, near enough: neither 5 ms nor 20 ms is dyadic, so
    // their ratio carries both representations.
    let quarter = TickAlpha::from_remainder(Seconds::from_millis(5).unwrap(), tick)
        .expect("non-zero tick")
        .fixed();

    assert!((quarter - Fixed::ONE / 4).abs() < Fixed::ONE / 100_000, "was {quarter}");

    // With dyadic values it is exact.
    assert_eq!(
        TickAlpha::from_remainder(
            Seconds::from_micros(7_812).unwrap() + Seconds::from_nanos(500).unwrap(),
            Seconds::from_micros(15_625).unwrap(),
        )
        .map(TickAlpha::fixed),
        Some(Fixed::ONE / 2),
    );
    assert_eq!(
        TickAlpha::from_remainder(Seconds::ZERO, tick),
        Some(TickAlpha::ZERO),
    );
    assert_eq!(
        TickAlpha::from_remainder(Seconds::from_millis(5).unwrap(), Seconds::ZERO),
        None,
        "no interval to be partway through",
    );
}

#[test]
fn a_clock_turns_frame_time_into_whole_ticks() {
    let rate = TickRate::new(100).expect("positive");
    let mut clock = FrameClock::new(rate);

    assert_eq!(clock.now(), Tick::ORIGIN);
    assert_eq!(clock.ready_ticks(), TickDuration::ZERO);

    // Twenty-five milliseconds at a hundred a second pays for two ticks and leaves
    // half of one over.
    let frame = clock.begin_frame(FrameDelta::from_millis(25).unwrap());


    assert_eq!(frame.index(), FrameIndex::FIRST, "zero-based: the first frame is FIRST");
    assert_eq!(frame.delta(), FrameDelta::from_millis(25).unwrap());
    assert_eq!(frame.duration(), FrameDelta::from_millis(25).unwrap().seconds());
    assert_eq!(clock.ready_ticks(), TickDuration::new(2));

    assert_eq!(clock.take_tick(), Some(Tick::new(1)));
    assert_eq!(clock.take_tick(), Some(Tick::new(2)));
    assert_eq!(clock.take_tick(), None, "only two were paid for");

    // Half a tick remains, so drawing sits halfway between states. A hundredth of a
    // second is not dyadic, so this is near a half rather than exactly one — the
    // ratio inherits the representation of the values it came from.
    let alpha = clock.tick_alpha().fixed();

    assert!((alpha - Fixed::ONE / 2).abs() < Fixed::ONE / 10_000, "alpha was {alpha}");
}

#[test]
fn a_dyadic_rate_gives_an_exact_alpha() {
    // At a rate whose tick length *is* representable, the arithmetic is exact
    // throughout — which is what to choose if exactness matters more than 60 Hz.
    let mut clock = FrameClock::new(TickRate::new(64).expect("positive"));

    // Two and a half ticks' worth: 2.5 / 64 of a second.
    clock.begin_frame(
        FrameDelta::from_micros(15_625 * 2 + 7_812).unwrap() + FrameDelta::from_nanos(500).unwrap(),
    );

    assert_eq!(clock.ready_ticks(), TickDuration::new(2));
    assert_eq!(clock.take_tick(), Some(Tick::new(1)));
    assert_eq!(clock.take_tick(), Some(Tick::new(2)));

    let alpha = clock.tick_alpha().fixed();

    assert!((alpha - Fixed::ONE / 2).abs() < Fixed::ONE / 1_000_000, "alpha was {alpha}");
}

#[test]
fn a_clock_carries_its_remainder_between_frames() {
    // Frames that are not a whole number of ticks must not lose or gain time: after
    // a hundred frames of 7ms at 100 ticks a second, exactly 70 ticks have run.
    let mut clock = FrameClock::new(TickRate::new(100).expect("positive"));
    let frame = FrameDelta::from_millis(7).unwrap();

    let mut run: u64 = 0;

    for _ in 0..100 {
        clock.begin_frame(frame);

        while clock.take_tick().is_some() {
            run += 1;
        }
    }

    assert_eq!(run, 70, "700ms of frames is 70 ticks at 100 a second");
    assert_eq!(clock.now(), Tick::new(70));

    // Seven milliseconds is not dyadic, so a hundred of them accumulate to very near
    // seven hundred rather than exactly — the point is that nothing was lost or
    // gained, which the tick count above already proves.
    let elapsed = clock.elapsed().fixed();
    let expected = Seconds::from_millis(700).unwrap().fixed();

    assert!((elapsed - expected).abs() < Fixed::ONE / 100_000, "elapsed was {elapsed}");
}

#[test]
fn a_clock_holds_no_catch_up_policy() {
    let mut clock = FrameClock::new(TickRate::new(60).expect("positive"));

    // A one-second stall owes sixty ticks, and the clock says so rather than
    // deciding anything about it.
    clock.begin_frame(FrameDelta::from_seconds(1).unwrap());

    assert_eq!(clock.ready_ticks(), TickDuration::new(60));

    // The caller's cap.
    for _ in 0..5 {
        assert!(clock.take_tick().is_some());
    }

    assert_eq!(clock.ready_ticks(), TickDuration::new(55));

    // And the caller's resynchronisation, which reports what it threw away.
    assert_eq!(clock.discard_backlog(), TickDuration::new(55));
    assert_eq!(clock.ready_ticks(), TickDuration::ZERO);
    assert_eq!(clock.now(), Tick::new(5), "discarding does not advance the world");
}

#[test]
fn a_clock_can_resume_a_world_in_progress() {
    let clock = FrameClock::starting_at(TickRate::default(), Tick::new(5_000));

    assert_eq!(clock.now(), Tick::new(5_000));
    assert_eq!(clock.frame_index(), None, "no frame presented yet");
    assert_eq!(clock.frames_presented(), 0);
}

// ---------------------------------------------------------------------------
// Kinematics
// ---------------------------------------------------------------------------

#[test]
fn a_velocity_over_a_duration_is_a_displacement() {
    let velocity = Velocity3::new(fixed(10), Fixed::ZERO, Fixed::ZERO);
    let over = Seconds::from_seconds(2).unwrap();

    assert_eq!(
        velocity * over,
        Displacement3::new(fixed(20), Fixed::ZERO, Fixed::ZERO),
        "ten units a second for two seconds is twenty units",
    );
    assert_eq!(velocity.checked_over(over), Some(velocity * over));

    // And in every direction at once, including backwards.
    let diagonal = Velocity3::new(fixed(3), fixed(-4), fixed(5));

    assert_eq!(
        diagonal * Seconds::from_seconds(3).unwrap(),
        Displacement3::new(fixed(9), fixed(-12), fixed(15)),
    );

    // No time, no movement.
    assert_eq!(velocity * Seconds::ZERO, Displacement3::ZERO);
}

#[test]
fn an_acceleration_over_a_duration_is_a_change_in_velocity() {
    let acceleration = Acceleration3::new(fixed(3), Fixed::ZERO, Fixed::ZERO);
    let over = Seconds::from_seconds(4).unwrap();

    assert_eq!(
        acceleration * over,
        Velocity3::new(fixed(12), Fixed::ZERO, Fixed::ZERO),
        "three a second squared for four seconds is twelve a second",
    );

    // Which then adds to an existing velocity, because both are velocities.
    let mut velocity = Velocity3::new(fixed(5), Fixed::ZERO, Fixed::ZERO);
    velocity += acceleration * over;

    assert_eq!(velocity, Velocity3::new(fixed(17), Fixed::ZERO, Fixed::ZERO));

    // Deceleration is an acceleration with the sign the other way.
    let braking = Acceleration3::new(fixed(-2), Fixed::ZERO, Fixed::ZERO);

    assert_eq!(
        braking * Seconds::from_seconds(3).unwrap(),
        Velocity3::new(fixed(-6), Fixed::ZERO, Fixed::ZERO),
    );
}

#[test]
fn constant_acceleration_gives_the_textbook_displacement() {
    // s = ut + at^2/2. With u = 0, a = 10, t = 2: s = 20.
    let from_rest = Acceleration3::new(Fixed::ZERO, fixed(-10), Fixed::ZERO)
        .checked_displacement(Velocity3::ZERO, Seconds::from_seconds(2).unwrap())
        .expect("in range");

    assert_eq!(from_rest, Displacement3::new(Fixed::ZERO, fixed(-20), Fixed::ZERO));

    // With u = 5, a = 4, t = 3: s = 15 + 18 = 33.
    let moving = Acceleration3::new(fixed(4), Fixed::ZERO, Fixed::ZERO)
        .checked_displacement(
            Velocity3::new(fixed(5), Fixed::ZERO, Fixed::ZERO),
            Seconds::from_seconds(3).unwrap(),
        )
        .expect("in range");

    assert_eq!(moving, Displacement3::new(fixed(33), Fixed::ZERO, Fixed::ZERO));
}

#[test]
fn mass_and_acceleration_make_force_both_ways_round() {
    let mass = Mass::from_integer(2).expect("positive");
    let acceleration = Acceleration3::new(fixed(5), Fixed::ZERO, Fixed::ZERO);
    let expected = Force3::new(fixed(10), Fixed::ZERO, Fixed::ZERO);

    assert_eq!(acceleration * mass, expected);
    assert_eq!(mass * acceleration, expected, "multiplication is commutative here");

    // And back again.
    assert_eq!(expected / mass, acceleration);
    assert_eq!(expected.checked_divide(mass), Some(acceleration));
}

#[test]
fn zero_mass_is_reported_rather_than_divided_by() {
    let force = Force3::new(fixed(10), Fixed::ZERO, Fixed::ZERO);

    assert_eq!(force.checked_divide(Mass::ZERO), None);

    // Mass itself can be zero; it just cannot be a divisor.
    assert!(Mass::new(Fixed::ZERO).is_some());
    assert_eq!(Mass::new(fixed(-1)), None, "negative mass is not a mass");
    assert_eq!(Mass::from_integer(-5), None);
}

#[test]
fn quantities_add_and_negate_within_their_own_kind() {
    let one = Velocity3::new(fixed(1), fixed(2), fixed(3));
    let two = Velocity3::new(fixed(10), fixed(20), fixed(30));

    assert_eq!(one + two, Velocity3::new(fixed(11), fixed(22), fixed(33)));
    assert_eq!(two - one, Velocity3::new(fixed(9), fixed(18), fixed(27)));
    assert_eq!(-one, Velocity3::new(fixed(-1), fixed(-2), fixed(-3)));
    assert_eq!(one * fixed(3), Velocity3::new(fixed(3), fixed(6), fixed(9)));

    // The same for every quantity, from the same macro.
    let push = Force3::new(fixed(1), Fixed::ZERO, Fixed::ZERO);
    assert_eq!(push + push, Force3::new(fixed(2), Fixed::ZERO, Fixed::ZERO));

    let speed_up = Acceleration3::new(fixed(1), Fixed::ZERO, Fixed::ZERO);
    assert_eq!(speed_up + speed_up, Acceleration3::new(fixed(2), Fixed::ZERO, Fixed::ZERO));

    let step = Displacement3::new(fixed(1), Fixed::ZERO, Fixed::ZERO);
    assert_eq!(step + step, Displacement3::new(fixed(2), Fixed::ZERO, Fixed::ZERO));
}

#[test]
fn a_position_moves_by_a_displacement_and_differences_back() {
    let start = PrecisePosition3::new(fixed(100), fixed(64), fixed(100));
    let step = Displacement3::new(fixed(5), fixed(-2), Fixed::ZERO);

    let mut position = start;
    position += step;

    assert_eq!(position, PrecisePosition3::new(fixed(105), fixed(62), fixed(100)));
    assert_eq!(position.displacement_from(start), Some(step));

    // And back.
    position -= step;
    assert_eq!(position, start);
    assert_eq!(position.displacement_from(start), Some(Displacement3::ZERO));
}

#[test]
fn integration_reads_as_the_physics_it_is() {
    // The whole point: every quantity says what it is, and the expression could not
    // be written with the wrong one in it.
    // A dyadic rate, so a tick length is exactly representable and the integration is
    // exact rather than near. A rate of 60 would work and drift by a part in 10^9.
    let rate = TickRate::new(64).expect("positive");
    let step: Seconds = rate.tick_length();

    let mut position = PrecisePosition3::new(Fixed::ZERO, fixed(100), Fixed::ZERO);
    let mut velocity = Velocity3::ZERO;
    let gravity = Acceleration3::new(Fixed::ZERO, fixed(-10), Fixed::ZERO);

    // One second of simulation, a sixty-fourth at a time.
    for _ in 0..64 {
        velocity += gravity * step;
        position += velocity * step;
    }

    // After a second of ten-a-second-squared gravity, falling at ten a second.
    assert_eq!(velocity, Velocity3::new(Fixed::ZERO, fixed(-10), Fixed::ZERO));

    // And somewhere near five units below where it started: the discrete sum is
    // `a*h^2*n(n+1)/2`, which is 5.1 rather than the continuous 5.
    let fallen = position.displacement_from(PrecisePosition3::new(Fixed::ZERO, fixed(100), Fixed::ZERO))
        .expect("in range");
    let down = fallen.as_vector().y;

    assert!(down < fixed(-5), "fell at least five units: {down}");
    assert!(down > fixed(-6), "but not six: {down}");
}

#[test]
fn two_dimensional_quantities_work_the_same_way() {
    use voxel_world::spatial::kinematics::{Acceleration2, Displacement2, Velocity2};

    let velocity = Velocity2::new(fixed(3), fixed(4));

    assert_eq!(
        velocity * Seconds::from_seconds(2).unwrap(),
        Displacement2::new(fixed(6), fixed(8)),
    );
    assert_eq!(
        Acceleration2::new(fixed(1), fixed(2)) * Seconds::from_seconds(3).unwrap(),
        Velocity2::new(fixed(3), fixed(6)),
    );
}

// ---------------------------------------------------------------------------
// FrameDelta: the frame's own measured time
// ---------------------------------------------------------------------------

#[test]
fn a_delta_of_nothing_is_zero() {
    assert!(FrameDelta::ZERO.is_zero());
    assert_eq!(FrameDelta::ZERO.fixed(), Fixed::ZERO);
    assert_eq!(FrameDelta::from_millis(0), Some(FrameDelta::ZERO));
    assert_eq!(FrameDelta::from_duration(Duration::ZERO), Some(FrameDelta::ZERO));

    // A frame that took no time has no frame rate, rather than an infinite one.
    assert_eq!(FrameDelta::ZERO.frequency(), None);
}

#[test]
fn a_delta_is_built_from_a_fixed_value_directly() {
    assert_eq!(FrameDelta::new(fixed(2)).map(FrameDelta::fixed), Some(fixed(2)));
    assert_eq!(FrameDelta::new(Fixed::ONE / 4).map(FrameDelta::fixed), Some(Fixed::ONE / 4));

    // The storage is the `Fixed`, so this is the value and not a conversion.
    let delta = FrameDelta::from_millis(500).expect("in range");
    assert_eq!(delta.fixed(), Fixed::ONE / 2);
}

#[test]
fn a_negative_delta_cannot_be_constructed() {
    assert_eq!(FrameDelta::new(fixed(-1)), None);
    assert_eq!(FrameDelta::new(Fixed::from_bits(-1)), None, "not even one step below zero");
    assert_eq!(FrameDelta::from_seconds(-1), None);

    assert!(FrameDelta::new(Fixed::ZERO).is_some(), "zero is a duration");

    // Subtraction clamps rather than going negative, since a negative delta is not a
    // thing a frame can have taken.
    let small = FrameDelta::from_millis(4).unwrap();
    let large = FrameDelta::from_millis(20).unwrap();

    assert_eq!(small - large, FrameDelta::ZERO);
    assert_eq!(large - small, FrameDelta::from_millis(16).unwrap());

    // And scaling refuses a negative factor rather than producing one.
    assert_eq!(large.checked_scale(fixed(-1)), None);
    assert_eq!(large.checked_scale(fixed(2)), FrameDelta::from_millis(40));
}

#[test]
fn the_integer_constructors_agree_with_one_another() {
    assert_eq!(FrameDelta::from_seconds(2), FrameDelta::from_millis(2_000));
    assert_eq!(FrameDelta::from_millis(3), FrameDelta::from_micros(3_000));
    assert_eq!(FrameDelta::from_micros(5), FrameDelta::from_nanos(5_000));

    // Dyadic fractions are exact.
    assert_eq!(FrameDelta::from_micros(15_625).unwrap().fixed(), Fixed::ONE / 64);
    assert_eq!(FrameDelta::from_millis(250).unwrap().fixed(), Fixed::ONE / 4);

    // And a sixty-fourth of a second really is sixty-four frames a second.
    assert_eq!(FrameDelta::from_micros(15_625).unwrap().frequency(), Some(fixed(64)));
}

#[test]
fn a_platform_duration_becomes_a_delta_without_a_float() {
    // Whole seconds scale exactly; only the nanoseconds divide, once.
    assert_eq!(
        FrameDelta::from_duration(Duration::new(2, 500_000_000)),
        FrameDelta::from_millis(2_500),
    );
    assert_eq!(
        FrameDelta::from_duration(Duration::new(0, 250_000_000)).unwrap().fixed(),
        Fixed::ONE / 4,
    );

    // A typical sixty-hertz frame, which is not dyadic and so lands nearby.
    let sixty = FrameDelta::from_duration(Duration::new(0, 16_666_667)).expect("in range");
    let rate = sixty.frequency().expect("non-zero");

    assert!((rate - fixed(60)).abs() < Fixed::ONE / 1_000, "reported {rate}");

    // Through TryFrom as well, which is the same path.
    let converted: FrameDelta = Duration::from_millis(8).try_into().expect("in range");
    assert_eq!(converted, FrameDelta::from_millis(8).unwrap());

    // Every platform duration fits, so there is nothing to refuse.
    assert!(FrameDelta::from_duration(Duration::MAX).is_some());
}

#[test]
fn a_delta_converts_to_a_general_duration_one_way() {
    let delta = FrameDelta::from_millis(250).expect("in range");

    assert_eq!(delta.seconds(), Seconds::from_millis(250).unwrap());
    assert_eq!(Seconds::from(delta), Seconds::from_millis(250).unwrap());

    // The reverse is deliberately absent: an arbitrary duration is not this frame's
    // delta until something measured it. Constructing one is an explicit choice.
    assert_eq!(
        FrameDelta::new(Seconds::from_millis(250).unwrap().fixed()),
        Some(delta),
    );
}

#[test]
fn deltas_accumulate_and_saturate() {
    let mut total = FrameDelta::ZERO;

    for _ in 0..8 {
        total += FrameDelta::from_millis(125).unwrap();
    }

    assert_eq!(total, FrameDelta::from_seconds(1).unwrap(), "eight eighths of a second");

    // Saturating rather than wrapping, as the rest of the time module is.
    let huge = FrameDelta::new(Fixed::MAX).expect("non-negative");

    assert_eq!(huge + huge, huge);
    assert_eq!(huge.checked_add(huge), None, "reported, not wrapped");
}

#[test]
fn capping_a_pathological_delta_is_the_callers_choice() {
    // A debugger pause produces a genuinely enormous frame; the measurement keeps it.
    let stall = FrameDelta::from_seconds(30).unwrap();
    let cap = FrameDelta::from_millis(100).unwrap();

    assert_eq!(stall.clamped(cap), cap);
    assert_eq!(FrameDelta::from_millis(16).unwrap().clamped(cap), FrameDelta::from_millis(16).unwrap());
}

// ---------------------------------------------------------------------------
// The frame, and the clock that makes one
// ---------------------------------------------------------------------------

#[test]
fn the_first_frame_is_frame_zero() {
    // One convention, stated and checked: zero-based, so the first frame returned
    // carries FrameIndex::FIRST.
    let mut clock = FrameClock::new(TickRate::new(60).expect("positive"));

    assert_eq!(clock.frame_index(), None, "nothing presented yet");
    assert_eq!(clock.frames_presented(), 0);

    let first = clock.begin_frame(FrameDelta::from_millis(16).unwrap());

    assert_eq!(first.index(), FrameIndex::FIRST);
    assert_eq!(first.index().count(), 0);
    assert_eq!(clock.frame_index(), Some(FrameIndex::FIRST));
    assert_eq!(clock.frames_presented(), 1);

    let second = clock.begin_frame(FrameDelta::from_millis(16).unwrap());

    assert_eq!(second.index(), FrameIndex::new(1));
    assert_eq!(clock.frames_presented(), 2);
}

#[test]
fn a_frame_reports_the_delta_it_was_given() {
    let mut clock = FrameClock::new(TickRate::new(64).expect("positive"));
    let delta = FrameDelta::from_micros(15_625).expect("in range");

    let frame = clock.begin_frame(delta);

    assert_eq!(frame.delta(), delta, "the measurement comes back unchanged");
    assert_eq!(frame.duration(), delta.seconds(), "and as a general duration");
    assert_eq!(frame.elapsed(), delta.seconds(), "one frame in, so elapsed is the delta");
}

// ---------------------------------------------------------------------------
// Integration over a delta
// ---------------------------------------------------------------------------

#[test]
fn a_velocity_over_a_delta_is_a_displacement() {
    let velocity = Velocity3::new(fixed(10), fixed(-4), Fixed::ZERO);
    let delta = FrameDelta::from_millis(500).expect("in range");

    assert_eq!(
        velocity * delta,
        Displacement3::new(fixed(5), fixed(-2), Fixed::ZERO),
        "half a second at ten and minus four",
    );

    // The checked form agrees, and the general-duration form agrees with both.
    assert_eq!(velocity.checked_over_delta(delta), Some(velocity * delta));
    assert_eq!(velocity * delta, velocity * delta.seconds());

    assert_eq!(velocity * FrameDelta::ZERO, Displacement3::ZERO);
}

#[test]
fn an_acceleration_over_a_delta_is_a_change_in_velocity() {
    let acceleration = Acceleration3::new(Fixed::ZERO, fixed(-10), Fixed::ZERO);
    let delta = FrameDelta::from_millis(250).expect("in range");

    assert_eq!(
        acceleration * delta,
        Velocity3::new(Fixed::ZERO, Fixed::ZERO - fixed(2) - Fixed::ONE / 2, Fixed::ZERO),
        "ten a second squared for a quarter second is two and a half a second",
    );
    assert_eq!(acceleration.checked_over_delta(delta), Some(acceleration * delta));
    assert_eq!(acceleration * delta, acceleration * delta.seconds());

    // And the constant-acceleration displacement over the same delta.
    assert_eq!(
        acceleration.checked_displacement_delta(Velocity3::ZERO, delta),
        acceleration.checked_displacement(Velocity3::ZERO, delta.seconds()),
    );
}

#[test]
fn checked_integration_reports_overflow_rather_than_panicking() {
    let enormous = Velocity3::new(Fixed::MAX, Fixed::ZERO, Fixed::ZERO);
    let long = FrameDelta::from_seconds(1_000).expect("in range");

    assert_eq!(enormous.checked_over_delta(long), None);
    assert_eq!(enormous.checked_over(long.seconds()), None);

    let push = Acceleration3::new(Fixed::MAX, Fixed::ZERO, Fixed::ZERO);
    assert_eq!(push.checked_over_delta(long), None);
    assert_eq!(push.checked_displacement_delta(Velocity3::ZERO, long), None);
}

#[test]
fn frame_code_reads_as_the_physics_it_is() {
    // The intended shape of ordinary engine code.
    let mut clock = FrameClock::new(TickRate::new(64).expect("positive"));

    let mut camera_position = PrecisePosition3::new(Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
    let mut camera_velocity = Velocity3::ZERO;
    let camera_acceleration = Acceleration3::new(fixed(8), Fixed::ZERO, Fixed::ZERO);

    for _ in 0..64 {
        let frame = clock.begin_frame(FrameDelta::from_micros(15_625).unwrap());

        camera_velocity += camera_acceleration * frame.delta();
        camera_position += camera_velocity * frame.delta();
    }

    // One second of eight-a-second-squared acceleration.
    assert_eq!(camera_velocity, Velocity3::new(fixed(8), Fixed::ZERO, Fixed::ZERO));

    let travelled = camera_position
        .displacement_from(PrecisePosition3::new(Fixed::ZERO, Fixed::ZERO, Fixed::ZERO))
        .expect("in range")
        .as_vector()
        .x;

    // Near four units: the discrete sum is a little over the continuous `at^2/2`.
    assert!(travelled > fixed(4), "travelled {travelled}");
    assert!(travelled < fixed(5), "travelled {travelled}");
}

// ---------------------------------------------------------------------------
// The delta-to-duration conversion is infallible
// ---------------------------------------------------------------------------

#[test]
fn a_delta_converts_to_seconds_without_losing_its_value() {
    // There is no fallback path that could turn an awkward value into zero: the
    // conversion is infallible because the invariant is established at construction.
    // Checked across the whole range, including values with no dyadic form.
    let cases: [FrameDelta; 8] = [
        FrameDelta::ZERO,
        FrameDelta::new(Fixed::from_bits(1)).expect("one step"),
        FrameDelta::from_nanos(1).expect("in range"),
        FrameDelta::from_micros(16_667).expect("in range"),
        FrameDelta::from_millis(7).expect("in range"),
        FrameDelta::from_seconds(1).expect("in range"),
        FrameDelta::from_seconds(86_400).expect("in range"),
        FrameDelta::new(Fixed::MAX).expect("non-negative"),
    ];

    for delta in cases {
        assert_eq!(
            delta.seconds().fixed(),
            delta.fixed(),
            "the duration changed value on the way across",
        );
        assert_eq!(Seconds::from(delta).fixed(), delta.fixed());

        // And only zero converts to zero.
        assert_eq!(delta.seconds().is_zero(), delta.is_zero());
    }
}

#[test]
fn one_step_above_zero_survives_the_conversion() {
    // The value most likely to be lost by a careless fallback: the smallest positive
    // duration there is.
    let smallest = FrameDelta::new(Fixed::from_bits(1)).expect("non-negative");

    assert!(!smallest.is_zero());
    assert!(!smallest.seconds().is_zero(), "a fallback would have zeroed this");
    assert_eq!(smallest.seconds().fixed().to_bits(), 1);
}

// ---------------------------------------------------------------------------
// Four dimensions
// ---------------------------------------------------------------------------

#[test]
fn four_dimensional_velocity_integrates_over_a_delta() {
    use voxel_world::spatial::kinematics::{Displacement4, Velocity4};

    // Every component participates, including a negative one.
    let velocity = Velocity4::new(fixed(10), fixed(-4), fixed(2), fixed(-8));
    let delta = FrameDelta::from_millis(500).expect("in range");

    assert_eq!(
        velocity * delta,
        Displacement4::new(fixed(5), fixed(-2), Fixed::ONE, fixed(-4)),
    );

    // The checked form and the general-duration form agree with it.
    assert_eq!(velocity.checked_over_delta(delta), Some(velocity * delta));
    assert_eq!(velocity * delta, velocity * delta.seconds());
    assert_eq!(velocity * FrameDelta::ZERO, Displacement4::ZERO);
}

#[test]
fn four_dimensional_acceleration_integrates_over_a_delta() {
    use voxel_world::spatial::kinematics::{Acceleration4, Velocity4};

    let acceleration = Acceleration4::new(fixed(4), fixed(-8), fixed(16), fixed(-2));
    let delta = FrameDelta::from_millis(250).expect("in range");

    assert_eq!(
        acceleration * delta,
        Velocity4::new(Fixed::ONE, fixed(-2), fixed(4), Fixed::ZERO - Fixed::ONE / 2),
    );
    assert_eq!(acceleration.checked_over_delta(delta), Some(acceleration * delta));
    assert_eq!(acceleration * delta, acceleration * delta.seconds());

    // And the constant-acceleration displacement agrees between the two durations.
    assert_eq!(
        acceleration.checked_displacement_delta(Velocity4::ZERO, delta),
        acceleration.checked_displacement(Velocity4::ZERO, delta.seconds()),
    );
}

#[test]
fn four_dimensional_mass_and_force_relate_as_they_should() {
    use voxel_world::spatial::kinematics::{Acceleration4, Force4};

    let mass = Mass::from_integer(3).expect("positive");
    let acceleration = Acceleration4::new(fixed(2), fixed(-4), Fixed::ZERO, fixed(10));
    let expected = Force4::new(fixed(6), fixed(-12), Fixed::ZERO, fixed(30));

    assert_eq!(acceleration * mass, expected);
    assert_eq!(mass * acceleration, expected);
    assert_eq!(expected / mass, acceleration, "and back again");
    assert_eq!(expected.checked_divide(mass), Some(acceleration));

    // Zero mass is reported, not divided by, in four dimensions as in three.
    assert_eq!(expected.checked_divide(Mass::ZERO), None);
}

#[test]
fn four_dimensional_quantities_add_subtract_and_negate() {
    use voxel_world::spatial::kinematics::{Acceleration4, Displacement4, Force4, Velocity4};

    let one = Velocity4::new(fixed(1), fixed(2), fixed(3), fixed(4));
    let two = Velocity4::new(fixed(10), fixed(-20), fixed(30), fixed(-40));

    assert_eq!(one + two, Velocity4::new(fixed(11), fixed(-18), fixed(33), fixed(-36)));
    assert_eq!(two - one, Velocity4::new(fixed(9), fixed(-22), fixed(27), fixed(-44)));
    assert_eq!(-one, Velocity4::new(fixed(-1), fixed(-2), fixed(-3), fixed(-4)));
    assert_eq!(one * fixed(2), Velocity4::new(fixed(2), fixed(4), fixed(6), fixed(8)));

    assert_eq!(one.checked_add(two), Some(one + two));
    assert_eq!(two.checked_sub(one), Some(two - one));
    assert_eq!(one.checked_scale(fixed(2)), Some(one * fixed(2)));

    // The same for the other three quantities, from the same macro.
    let step = Displacement4::new(Fixed::ONE, Fixed::ONE, Fixed::ONE, Fixed::ONE);
    assert_eq!(step + step, Displacement4::new(fixed(2), fixed(2), fixed(2), fixed(2)));

    let push = Force4::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
    assert_eq!(push + push, Force4::new(fixed(2), Fixed::ZERO, Fixed::ZERO, Fixed::ZERO));

    let speed_up = Acceleration4::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
    assert_eq!(
        speed_up + speed_up,
        Acceleration4::new(fixed(2), Fixed::ZERO, Fixed::ZERO, Fixed::ZERO),
    );
}

#[test]
fn four_dimensional_types_round_trip_through_their_vector() {
    use voxel_world::math::Vector4;
    use voxel_world::spatial::kinematics::Velocity4;

    let vector = Vector4 {
        x: fixed(1),
        y: fixed(2),
        z: fixed(3),
        w: fixed(4),
    };

    let velocity = Velocity4::from_vector(vector);

    assert_eq!(velocity.as_vector(), vector);
    assert_eq!(velocity.into_vector(), vector);
    assert_eq!(velocity, Velocity4::new(fixed(1), fixed(2), fixed(3), fixed(4)));
    assert_eq!(Velocity4::ZERO.as_vector().w, Fixed::ZERO);
}

#[test]
fn the_kinematic_types_are_reachable_from_the_spatial_module() {
    // Exported alongside the position types rather than only through the submodule.
    use voxel_world::spatial::{
        Acceleration2, Acceleration3, Acceleration4, Displacement2, Displacement3, Displacement4,
        Force2, Force3, Force4, Velocity2, Velocity3, Velocity4,
    };

    assert_eq!(Velocity2::ZERO.as_vector().x, Fixed::ZERO);
    assert_eq!(Velocity3::ZERO.as_vector().x, Fixed::ZERO);
    assert_eq!(Velocity4::ZERO.as_vector().w, Fixed::ZERO);

    let _ = (
        Acceleration2::ZERO,
        Acceleration3::ZERO,
        Acceleration4::ZERO,
        Displacement2::ZERO,
        Displacement3::ZERO,
        Displacement4::ZERO,
        Force2::ZERO,
        Force3::ZERO,
        Force4::ZERO,
    );
}
