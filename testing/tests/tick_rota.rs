//! Simulation-time deltas, and the rota clock that works them out for you.

use std::collections::HashSet;
use voxel_world::math::Fixed;
use voxel_world::spatial::kinematics::{
    Acceleration2, Acceleration3, Acceleration4, Displacement2, Displacement3, Displacement4,
    Velocity2, Velocity3, Velocity4,
};
use voxel_world::spatial::precise::PrecisePosition3;
use voxel_world::structures::collections::rota::{Rota, TickRota};
use voxel_world::time::{Seconds, Tick, TickDuration, TickRate, UpdateDelta};

fn fixed(whole: i64) -> Fixed {
    Fixed::from_integer(whole)
}

fn world_rate() -> TickRate {
    TickRate::new(60).expect("positive")
}

/// One tick at the world's ordinary rate.
fn tick_length() -> Seconds {
    world_rate().tick_length()
}

// ---------------------------------------------------------------------------
// UpdateDelta
// ---------------------------------------------------------------------------

#[test]
fn ticks_become_simulation_time_through_the_rate() {
    let rate = world_rate();

    // The cases a 60 Hz world actually produces.
    let cases: [(u64, i128); 6] = [
        (1, 60),
        (2, 30),
        (3, 20),
        (4, 15),
        (5, 12),
        (6, 10),
    ];

    for (ticks, divisor) in cases {
        assert_eq!(
            UpdateDelta::from_ticks(TickDuration::new(ticks), rate).fixed(),
            Fixed::from_ratio(1, divisor).expect("in range"),
            "{ticks} ticks at 60 a second should be a {divisor}th",
        );
    }
}

#[test]
fn the_same_tick_count_means_different_time_at_different_rates() {
    // The reason `TickDuration` is not itself a duration: four ticks is a fifteenth of
    // a second at 60 a second and a fifth at 20.
    let four = TickDuration::new(4);

    assert_eq!(
        UpdateDelta::from_ticks(four, TickRate::new(60).unwrap()).fixed(),
        Fixed::from_ratio(1, 15).unwrap(),
    );
    assert_eq!(
        UpdateDelta::from_ticks(four, TickRate::new(20).unwrap()).fixed(),
        Fixed::from_ratio(1, 5).unwrap(),
    );
    assert_eq!(
        UpdateDelta::from_ticks(four, TickRate::new(4).unwrap()),
        UpdateDelta::new(Fixed::ONE).unwrap(),
    );
}

#[test]
fn no_ticks_is_no_simulation_time() {
    assert_eq!(
        UpdateDelta::from_ticks(TickDuration::ZERO, world_rate()),
        UpdateDelta::ZERO,
    );
    assert!(UpdateDelta::ZERO.is_zero());
    assert_eq!(UpdateDelta::ZERO.fixed(), Fixed::ZERO);
}

#[test]
fn an_update_delta_cannot_be_negative() {
    assert_eq!(UpdateDelta::new(fixed(-1)), None);
    assert_eq!(UpdateDelta::new(Fixed::from_bits(-1)), None);
    assert!(UpdateDelta::new(Fixed::ZERO).is_some());

    // And `between` never goes backwards, because `ticks_since` does not.
    let delta = UpdateDelta::between(Tick::new(100), Tick::new(50), world_rate());

    assert_eq!(delta, UpdateDelta::ZERO, "a reversed pair is zero, not negative");
}

#[test]
fn an_update_delta_converts_to_seconds_without_changing_value() {
    for ticks in [0u64, 1, 4, 60, 3_600, 1_000_000] {
        let delta = UpdateDelta::from_ticks(TickDuration::new(ticks), world_rate());

        assert_eq!(delta.seconds().fixed(), delta.fixed());
        assert_eq!(Seconds::from(delta).fixed(), delta.fixed());
    }

    // One second of ticks really is one second.
    assert_eq!(
        UpdateDelta::from_ticks(TickDuration::new(60), world_rate()).seconds(),
        Seconds::from_seconds(1).unwrap(),
    );
}

#[test]
fn update_deltas_accumulate_and_saturate() {
    let quarter = UpdateDelta::from_ticks(TickDuration::new(15), world_rate());
    let mut total = UpdateDelta::ZERO;

    for _ in 0..4 {
        total += quarter;
    }

    assert_eq!(total.seconds(), Seconds::from_seconds(1).unwrap());

    let huge = UpdateDelta::new(Fixed::MAX).expect("non-negative");
    assert_eq!(huge + huge, huge, "saturating, never wrapping");
}

// ---------------------------------------------------------------------------
// Kinematics over simulation time
// ---------------------------------------------------------------------------

#[test]
fn velocity_integrates_over_simulation_time_in_every_dimension() {
    // Half a second: thirty ticks at 60 a second, and a half is dyadic so the
    // arithmetic is exact. A tenth of a second would not be — see the test below.
    let delta = UpdateDelta::from_ticks(TickDuration::new(30), world_rate());

    assert_eq!(
        Velocity2::new(fixed(20), fixed(-40)) * delta,
        Displacement2::new(fixed(10), fixed(-20)),
    );
    assert_eq!(
        Velocity3::new(fixed(20), fixed(-40), fixed(10)) * delta,
        Displacement3::new(fixed(10), fixed(-20), fixed(5)),
    );
    assert_eq!(
        Velocity4::new(fixed(20), fixed(-40), fixed(10), fixed(0)) * delta,
        Displacement4::new(fixed(10), fixed(-20), fixed(5), Fixed::ZERO),
    );

    // The checked forms agree, and so does the general-duration route.
    let velocity = Velocity3::new(fixed(20), fixed(-40), fixed(10));

    assert_eq!(velocity.checked_over_update(delta), Some(velocity * delta));
    assert_eq!(velocity * delta, velocity * delta.seconds());
}

#[test]
fn a_non_dyadic_cadence_lands_near_rather_than_on() {
    // A tenth and a fifteenth of a second are not binary fractions, so a delta built
    // from them is the nearest `2^-32` and the product inherits that. Near, not equal —
    // a fact about binary fractions, not about the conversion.
    // 6 ticks is a tenth of a second (20 x 1/10 = 2); 12 ticks is a fifth (20 x 1/5 = 4).
    for (ticks, expected) in [(6u64, 2i64), (12, 4)] {
        let delta = UpdateDelta::from_ticks(TickDuration::new(ticks), world_rate());
        let moved = Velocity3::new(fixed(20), Fixed::ZERO, Fixed::ZERO) * delta;
        let want = fixed(expected);

        assert!(
            (moved.as_vector().x - want).abs() < Fixed::ONE / 100_000,
            "{ticks} ticks gave {}, expected about {want}",
            moved.as_vector().x,
        );
    }
}

#[test]
fn acceleration_integrates_over_simulation_time_in_every_dimension() {
    // A quarter of a second: fifteen ticks at 60 a second.
    let delta = UpdateDelta::from_ticks(TickDuration::new(15), world_rate());

    assert_eq!(
        Acceleration2::new(fixed(8), fixed(-4)) * delta,
        Velocity2::new(fixed(2), fixed(-1)),
    );
    assert_eq!(
        Acceleration3::new(fixed(8), fixed(-4), fixed(40)) * delta,
        Velocity3::new(fixed(2), fixed(-1), fixed(10)),
    );
    assert_eq!(
        Acceleration4::new(fixed(8), fixed(-4), fixed(40), fixed(-80)) * delta,
        Velocity4::new(fixed(2), fixed(-1), fixed(10), fixed(-20)),
    );

    let acceleration = Acceleration3::new(fixed(8), fixed(-4), fixed(40));

    assert_eq!(acceleration.checked_over_update(delta), Some(acceleration * delta));
    assert_eq!(
        acceleration.checked_displacement_update(Velocity3::ZERO, delta),
        acceleration.checked_displacement(Velocity3::ZERO, delta.seconds()),
    );
}

// ---------------------------------------------------------------------------
// TickRota: whose turn, and how long since
// ---------------------------------------------------------------------------

#[test]
fn shares_take_their_turns_in_order() {
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    // One share per tick, round and round.
    for tick in 0..20u64 {
        let update = clock.step(Tick::new(tick), tick_length()).expect("a share is always due");

        assert_eq!(
            update.bucket(),
            (tick % 4) as usize,
            "tick {tick} went to the wrong share",
        );
        assert_eq!(update.at(), Tick::new(tick));
    }
}

#[test]
fn a_returning_share_knows_how_long_it_has_been() {
    let mut clock: TickRota<4> = TickRota::spread(world_rate());
    let expected = Fixed::from_ratio(1, 15).expect("in range");

    // Two full rounds, so every share returns.
    for tick in 0..12u64 {
        let update = clock.step(Tick::new(tick), tick_length()).expect("due");

        assert_eq!(
            update.elapsed_ticks(),
            TickDuration::new(4),
            "share {} at tick {tick}",
            update.bucket(),
        );
        // Four accumulated sixtieths, which lands one raw step under the nearest
        // fifteenth: each tick rounds down a fraction. See the type's own note.
        assert!(
            (update.delta().fixed() - expected).abs() <= Fixed::from_bits(4),
            "share {} gave {}, expected about {expected}",
            update.bucket(),
            update.delta().fixed(),
        );
    }
}

#[test]
fn every_share_accumulates_on_its_own() {
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    // Three full rounds, every tick reported.
    for count in 0..12u64 {
        clock.step(Tick::new(count), tick_length()).expect("due");
    }

    // Each share has its own last turn, one tick apart.
    for bucket in 0..4usize {
        assert_eq!(clock.last_run(bucket), Some(Tick::new(8 + bucket as u64)));
    }

    // And its own accumulator, holding the ticks since that turn: share 0 ran at tick 8
    // and three ticks have been reported since, share 3 ran at tick 11 and none have.
    for (bucket, ticks) in [(0usize, 3u64), (1, 2), (2, 1), (3, 0)] {
        assert_eq!(
            clock.accumulated(bucket),
            Some(world_rate().seconds_in(TickDuration::new(ticks))),
            "share {bucket} should hold {ticks} ticks",
        );
    }
}

#[test]
fn a_shares_first_turn_is_nominal_however_late_it_comes() {
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    // Twenty ticks go by before share 3 takes its first turn, and it still reports one
    // cadence rather than twenty ticks — a first turn is nominal by design.
    for count in 0..19u64 {
        clock.step(Tick::new(count), tick_length());
    }

    let mut fresh: TickRota<4> = TickRota::spread(world_rate());

    for count in 0..19u64 {
        if count % 4 != 3 {
            fresh.step(Tick::new(count), tick_length());
        } else {
            fresh.advance(tick_length());
        }
    }

    let late = fresh.step(Tick::new(19), tick_length()).expect("due");

    assert_eq!(late.bucket(), 3);
    assert!(late.is_first());
    assert_eq!(
        late.delta().fixed(),
        world_rate().seconds_in(TickDuration::new(4)).fixed(),
        "one nominal cadence, not nineteen ticks",
    );
}

#[test]
fn a_first_turn_reports_one_cadence() {
    // The documented convention: nominal, so a system behaves on its first turn as it
    // will on every later one. Zero would make it stall for an interval; the time since
    // the world began would fling it across the map.
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    let first = clock.step(Tick::new(0), tick_length()).expect("due");

    assert!(first.is_first());
    assert_eq!(first.elapsed_ticks(), clock.cadence());
    assert_eq!(first.delta().fixed(), Fixed::from_ratio(1, 15).unwrap());

    // Even when the rota starts long after the world did.
    let mut late: TickRota<4> = TickRota::spread(world_rate());
    let started = late.update(Tick::new(100_000)).expect("due");

    assert!(started.is_first());
    assert_eq!(
        started.elapsed_ticks(),
        TickDuration::new(4),
        "not a hundred thousand ticks",
    );

    // The second turn is measured, not nominal — and here it matches anyway.
    let second = late.update(Tick::new(100_004)).expect("due");

    assert!(!second.is_first());
    assert_eq!(second.elapsed_ticks(), TickDuration::new(4));
}

#[test]
fn a_late_turn_reports_the_time_actually_accumulated() {
    // The whole reason the clock holds state: a share that ran late must account for
    // the simulation time it really waited through, not the cadence it hoped for.
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    clock.step(Tick::new(0), tick_length()).expect("due");

    // Every tick is still reported — the clock can only add up what it is told — but
    // share 0's turns at ticks 4 and 8 are passed over, so it next runs at tick 12.
    for count in 1..12u64 {
        if count % 4 == 0 {
            clock.advance(tick_length());
        } else {
            clock.step(Tick::new(count), tick_length());
        }
    }

    let late = clock.step(Tick::new(12), tick_length()).expect("due");

    assert_eq!(late.bucket(), 0);
    assert_eq!(late.elapsed_ticks(), TickDuration::new(12), "three rounds passed over");
    // A fifth of a second, not a fifteenth — within the accumulation's own resolution.
    assert!(
        (late.delta().fixed() - Fixed::from_ratio(12, 60).unwrap()).abs() <= Fixed::from_bits(12),
        "a fifth of a second, not a fifteenth: got {}",
        late.delta().fixed(),
    );
    assert!(!late.is_first());
}

#[test]
fn a_changing_tick_rate_is_accounted_for_exactly() {
    // The case a tick count cannot answer. Share 0 runs at tick 0, then the world slows
    // from 60 Hz to 20 Hz, so its next four ticks are not four sixtieths.
    let fast = TickRate::new(60).expect("positive");
    let slow = TickRate::new(20).expect("positive");

    let mut clock: TickRota<4> = TickRota::spread(fast);

    clock.step(Tick::new(0), fast.tick_length()).expect("due");

    // Tick 1 at the old rate, then the rate drops for ticks 2, 3 and 4.
    clock.step(Tick::new(1), fast.tick_length()).expect("due");
    clock.step(Tick::new(2), slow.tick_length()).expect("due");
    clock.step(Tick::new(3), slow.tick_length()).expect("due");

    let update = clock.step(Tick::new(4), slow.tick_length()).expect("due");

    // One 60 Hz tick plus three 20 Hz ticks.
    let expected = fast.tick_length().fixed() + slow.tick_length().fixed() * fixed(3);

    assert_eq!(update.bucket(), 0);
    assert_eq!(update.delta().fixed(), expected);

    // Which is emphatically not what four ticks at either rate would give.
    assert_ne!(update.delta().fixed(), fast.seconds_in(TickDuration::new(4)).fixed());
    assert_ne!(update.delta().fixed(), slow.seconds_in(TickDuration::new(4)).fixed());

    // The tick count is unchanged by any of it — which is exactly why it is not enough.
    assert_eq!(update.elapsed_ticks(), TickDuration::new(4));
}

#[test]
fn taking_a_turn_empties_that_shares_accumulator_only() {
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    for count in 0..4u64 {
        clock.step(Tick::new(count), tick_length()).expect("due");
    }

    // Share 0 ran at tick 0 and has had three ticks since; the others less.
    clock.advance(tick_length());

    let before: Vec<Seconds> = (0..4).map(|b| clock.accumulated(b).unwrap()).collect();

    let update = clock.step(Tick::new(4), tick_length()).expect("due");

    assert_eq!(update.bucket(), 0);
    assert_eq!(
        clock.accumulated(0),
        Some(Seconds::ZERO),
        "the share that ran is reset to nothing",
    );

    // Everybody else kept their total, plus the tick that was just reported.
    for (bucket, held) in before.iter().enumerate().skip(1) {
        assert_eq!(
            clock.accumulated(bucket),
            Some(*held + tick_length()),
            "share {bucket} should have kept accumulating",
        );
    }
}

#[test]
fn overdue_shares_are_reported_without_a_policy_being_applied() {
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    // One full round so every share has run once.
    for tick in 0..4u64 {
        clock.step(Tick::new(tick), tick_length()).expect("due");
    }

    assert_eq!(clock.overdue_buckets().count(), 0, "nobody is behind yet");

    // Then a stall: the time is reported, but no turns are taken.
    for _ in 0..96 {
        clock.advance(tick_length());
    }

    assert_eq!(
        clock.overdue_buckets().collect::<Vec<usize>>(),
        vec![0, 1, 2, 3],
        "all four have more than a cadence banked up",
    );

    // The caller decides what to do. Here: run each once with what it accumulated.
    let jumped = Tick::new(100);

    for bucket in 0..4usize {
        let banked = clock.accumulated(bucket).expect("valid share");
        let update = clock.update_bucket(bucket, jumped).expect("valid share");

        assert_eq!(update.bucket(), bucket);
        assert_eq!(update.delta().fixed(), banked.fixed(), "hands over what it held");
        assert!(update.delta() > UpdateDelta::from_ticks(TickDuration::new(4), world_rate()));
    }

    // Nothing is overdue now; every accumulator was emptied.
    assert_eq!(clock.overdue_buckets().count(), 0);

    for bucket in 0..4usize {
        assert_eq!(clock.accumulated(bucket), Some(Seconds::ZERO));
    }

    // A share that has never run is not overdue; its first turn is still to come.
    let fresh: TickRota<4> = TickRota::spread(world_rate());
    assert_eq!(fresh.overdue_buckets().count(), 0);
}

#[test]
fn a_cadence_longer_than_the_share_count_leaves_idle_ticks() {
    // Two shares at a cadence of four: they burst on ticks 0 and 1 of every four, and
    // ticks 2 and 3 carry nothing.
    let mut clock: TickRota<2> =
        TickRota::new(world_rate(), TickDuration::new(4)).expect("valid");

    let mut ran: Vec<Option<usize>> = Vec::new();

    for tick in 0..8u64 {
        ran.push(clock.step(Tick::new(tick), tick_length()).map(|u| u.bucket()));
    }

    assert_eq!(
        ran,
        vec![Some(0), Some(1), None, None, Some(0), Some(1), None, None],
    );

    // And each share still reports its cadence, not the gap to the next share.
    let update = clock.step(Tick::new(8), tick_length()).expect("due");
    assert_eq!(update.elapsed_ticks(), TickDuration::new(4));
}

#[test]
fn invalid_configurations_are_refused() {
    let rate = world_rate();

    // A cadence of zero names no interval.
    assert!(TickRota::<4>::new(rate, TickDuration::ZERO).is_none());

    // A cadence below the share count would leave the later shares permanently idle.
    assert!(TickRota::<4>::new(rate, TickDuration::new(3)).is_none());
    assert!(TickRota::<4>::new(rate, TickDuration::new(4)).is_some());
    assert!(TickRota::<4>::new(rate, TickDuration::new(100)).is_some());
}

#[test]
fn a_requested_rate_must_divide_the_worlds() {
    let rate = world_rate();

    // The divisors of sixty that a four-share rota can express.
    assert_eq!(
        TickRota::<4>::at_rate(rate, 15).map(|c| c.cadence()),
        Some(TickDuration::new(4)),
    );
    assert_eq!(
        TickRota::<2>::at_rate(rate, 30).map(|c| c.cadence()),
        Some(TickDuration::new(2)),
    );
    assert_eq!(
        TickRota::<6>::at_rate(rate, 10).map(|c| c.cadence()),
        Some(TickDuration::new(6)),
    );

    // Seventeen does not divide sixty, so there is no whole cadence for it — refused
    // rather than approximated with a drifting phase.
    assert!(TickRota::<4>::at_rate(rate, 17).is_none());
    assert!(TickRota::<4>::at_rate(rate, 7).is_none());
    assert!(TickRota::<4>::at_rate(rate, 0).is_none());

    // And a rate too fast for the share count: 60 Hz would be a cadence of one, which
    // four shares cannot divide.
    assert!(TickRota::<4>::at_rate(rate, 60).is_none());

    // The round trip reports what was asked for.
    assert_eq!(
        TickRota::<4>::at_rate(rate, 15).and_then(|c| c.updates_per_second()),
        Some(15),
    );
}

#[test]
fn the_clock_keeps_the_rate_it_converts_with() {
    let clock: TickRota<4> = TickRota::spread(TickRate::new(20).expect("positive"));

    assert_eq!(clock.nominal_rate(), TickRate::new(20).unwrap());
    assert_eq!(clock.cadence(), TickDuration::new(4));
    assert_eq!(clock.bucket_count(), 4);
    assert_eq!(clock.updates_per_second(), Some(5), "20 Hz over four shares");
}

#[test]
fn a_due_query_records_nothing() {
    let clock: TickRota<4> = TickRota::spread(world_rate());

    // Asking is free and repeatable; taking the turn is what records it.
    for _ in 0..3 {
        assert_eq!(clock.due_bucket(Tick::new(6)), Some(2));
    }

    assert_eq!(clock.last_run(2), None, "asking did not take the turn");
    assert_eq!(clock.due_bucket(Tick::new(0)), Some(0));
    assert_eq!(clock.last_run(0), None);
}

#[test]
fn resetting_forgets_every_last_turn() {
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    for tick in 0..8u64 {
        clock.step(Tick::new(tick), tick_length()).expect("due");
    }

    assert!(clock.last_run(0).is_some());

    clock.reset();

    for bucket in 0..4usize {
        assert_eq!(clock.last_run(bucket), None);
    }

    // So the next turn is a first turn again, with the nominal delta.
    let after = clock.step(Tick::new(8), tick_length()).expect("due");

    assert!(after.is_first());
    assert_eq!(after.elapsed_ticks(), clock.cadence());
}

#[test]
fn large_tick_values_are_handled() {
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    // Far into a world's life, with the phase still correct.
    let base: u64 = 1_000_000_000_000;

    for offset in 0..8u64 {
        let tick = Tick::new(base + offset);
        let update = clock.update(tick).expect("due");

        assert_eq!(update.bucket(), ((base + offset) % 4) as usize);
    }

    // And right at the end of representable time, where nothing may wrap.
    let mut edge: TickRota<4> = TickRota::spread(world_rate());
    let last = Tick::new(u64::MAX);

    assert!(edge.update(last).is_some() || edge.due_bucket(last).is_none());
}

#[test]
fn out_of_range_shares_are_refused() {
    let mut clock: TickRota<4> = TickRota::spread(world_rate());

    assert_eq!(clock.update_bucket(4, Tick::new(0)), None);
    assert_eq!(clock.update_bucket(usize::MAX, Tick::new(0)), None);
    assert_eq!(clock.last_run(4), None);
    assert!(!clock.is_overdue(4));
}

// ---------------------------------------------------------------------------
// Phase spreading and determinism
// ---------------------------------------------------------------------------

#[test]
fn the_work_is_spread_rather_than_bunched() {
    // The reason a rota exists: a thousand entities at 15 Hz under a 60 Hz world wake
    // two hundred and fifty at a time, not a thousand every fourth tick.
    let mut population: Rota<u32, 4> = Rota::new();

    for entity in 0..1_000u32 {
        population.insert(entity);
    }

    let mut clock: TickRota<4> = TickRota::spread(world_rate());
    let mut visited: HashSet<u32> = HashSet::new();

    for tick in 0..4u64 {
        let update = clock.step(Tick::new(tick), tick_length()).expect("due");
        let share = update.group_of(&population);

        assert_eq!(share.len(), 250, "each tick should carry a quarter");

        for entity in share {
            assert!(visited.insert(*entity), "{entity} was visited twice in one round");
        }
    }

    assert_eq!(visited.len(), 1_000, "one round covers everybody exactly once");
}

#[test]
fn the_same_tick_sequence_replays_identically() {
    let run = || {
        let mut clock: TickRota<4> = TickRota::spread(world_rate());
        let mut seen: Vec<(usize, u64, i128)> = Vec::new();

        // An irregular sequence, including skips, so the bookkeeping is exercised.
        for tick in [0u64, 1, 2, 3, 4, 9, 12, 13, 20, 21, 100, 101, 102, 103] {
            if let Some(update) = clock.step(Tick::new(tick), tick_length()) {
                seen.push((
                    update.bucket(),
                    update.elapsed_ticks().count(),
                    update.delta().fixed().to_bits(),
                ));
            }
        }

        seen
    };

    assert_eq!(run(), run(), "the clock is not deterministic");
    assert!(!run().is_empty());
}

#[test]
fn an_authoritative_system_reads_as_the_physics_it_is() {
    // The shape this whole exercise exists for: no manual elapsed-tick arithmetic, no
    // rate conversion, no last-run array in the caller.
    //
    // A 64 Hz world over four shares is 16 Hz, and a sixteenth of a second is dyadic,
    // so the integration is exact rather than near.
    let rate = TickRate::new(64).expect("positive");
    let mut clock: TickRota<4> = TickRota::at_rate(rate, 16).expect("divides");

    let mut position = PrecisePosition3::new(Fixed::ZERO, fixed(100), Fixed::ZERO);
    let mut velocity = Velocity3::ZERO;
    let gravity = Acceleration3::new(Fixed::ZERO, fixed(-10), Fixed::ZERO);

    // One second of world time: sixty-four ticks, of which share 0 takes sixteen turns.
    for tick in 0..64u64 {
        let Some(update) = clock.step(Tick::new(tick), rate.tick_length()) else {
            continue;
        };

        if update.bucket() != 0 {
            continue;
        }

        let delta: UpdateDelta = update.delta();

        velocity += gravity * delta;
        position += velocity * delta;
    }

    // Sixteen updates of a sixteenth of a second each is one second of falling.
    assert_eq!(velocity, Velocity3::new(Fixed::ZERO, fixed(-10), Fixed::ZERO));

    let fallen = position
        .displacement_from(PrecisePosition3::new(Fixed::ZERO, fixed(100), Fixed::ZERO))
        .expect("in range")
        .as_vector()
        .y;

    assert!(fallen < fixed(-5), "fell at least five units: {fallen}");
    assert!(fallen > fixed(-6), "but under six: {fallen}");
}
