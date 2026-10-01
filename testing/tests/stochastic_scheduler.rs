//! Drawn recurring work: mean interval, load spreading, backlog policy, and
//! replay determinism under both kinds of randomness.

use voxel_world::random::seed::Seed;
use voxel_world::random::{EventRandom, Random};
use voxel_world::structures::collections::{
    Cadence, Firing, OnBacklog, StochasticScheduler, UniqueStochasticScheduler,
};
use voxel_world::structures::traits::{Collection, RangeQuery};

fn domain() -> Seed {
    Seed::from_raw(0x5EED).child("events")
}

#[test]
fn waits_average_out_to_the_mean_time_to_happen() {
    let mut events: StochasticScheduler<u32> = StochasticScheduler::new(domain());
    let cadence = events.register(Cadence::mtth(50));
    assert!(events.insert(1, cadence).is_some());

    // Count firings over a long run: with MTTH 50 the count should be within a
    // few percent of steps / 50.
    let steps = 200_000;
    let mut fired = 0;
    for _ in 0..steps {
        fired += events.advance().len();
    }

    let mean = steps as f64 / fired as f64;
    assert!(
        (mean - 50.0).abs() < 2.0,
        "mean interval {mean} should be near 50"
    );

    // And the wait really is memoryless: the gaps vary widely rather than
    // clustering at the mean.
    let mut gaps: Vec<u64> = Vec::new();
    let mut solo: StochasticScheduler<u32> = StochasticScheduler::new(domain());
    let cadence = solo.register(Cadence::mtth(50));
    assert!(solo.insert(1, cadence).is_some());
    let mut previous = solo.now();
    for _ in 0..20_000 {
        if !solo.advance().is_empty() {
            gaps.push(solo.now() - previous);
            previous = solo.now();
        }
    }
    assert!(
        gaps.iter().any(|gap| *gap > 120),
        "a geometric tail is expected"
    );
    assert!(
        gaps.iter().any(|gap| *gap < 10),
        "short waits are expected too"
    );
}

#[test]
fn spread_evens_the_load_across_steps() {
    // Same population and cadence, with and without spread: the busiest step
    // should be lower when the scheduler is allowed to nudge placements.
    fn busiest(spread: u32) -> usize {
        let mut events: StochasticScheduler<u32> = StochasticScheduler::new(domain());
        let cadence = events.register(Cadence::mtth(40).spread(spread));

        for entity in 0..2_000 {
            assert!(events.insert(entity, cadence).is_some());
        }

        // Let it settle, then look at the load over the next stretch.
        for _ in 0..200 {
            events.advance();
        }

        let now = events.now();
        (now + 1..now + 60)
            .map(|step| events.len_at(step))
            .max()
            .unwrap_or(0)
    }

    let unsmoothed = busiest(0);
    let smoothed = busiest(4);

    assert!(
        smoothed < unsmoothed,
        "spread {smoothed} should beat no spread {unsmoothed}"
    );
}

#[test]
fn every_entry_keeps_firing_and_nothing_is_lost() {
    let mut events: StochasticScheduler<u32> = StochasticScheduler::new(domain());
    let cadence = events.register(Cadence::mtth(10).spread(2));

    for entity in 0..50 {
        assert!(events.insert(entity, cadence).is_some());
    }
    assert_eq!(events.len(), 50);

    let mut seen = vec![0u32; 50];
    for _ in 0..500 {
        for firing in events.advance() {
            seen[firing.value as usize] += firing.occurrences;
        }
    }

    // Still fifty entries pending, and each has fired roughly 50 times.
    assert_eq!(events.len(), 50);
    assert_eq!(Collection::len(&events), 50);
    assert!(
        seen.iter().all(|count| (20..90).contains(count)),
        "{seen:?}"
    );
}

#[test]
fn a_backlog_is_handled_the_way_the_cadence_says() {
    // Coalesce: one firing that reports how many occurrences it stands for.
    let mut coalescing: StochasticScheduler<&str> = StochasticScheduler::new(domain());
    let cadence = coalescing.register(Cadence::mtth(10).on_backlog(OnBacklog::Coalesce));
    assert!(coalescing.insert("plant", cadence).is_some());

    let fired: Vec<Firing<&str>> = coalescing.advance_by(1000);
    assert_eq!(fired.len(), 1, "one firing however many were owed");
    assert!(fired[0].occurrences > 50, "about a hundred were owed");
    assert_eq!(coalescing.len(), 1, "still scheduled afterwards");
    assert!(coalescing.next_due().unwrap() > coalescing.now());

    // FireAll: one firing per owed occurrence.
    let mut all: StochasticScheduler<&str> = StochasticScheduler::new(domain());
    let cadence = all.register(Cadence::mtth(10).on_backlog(OnBacklog::FireAll));
    assert!(all.insert("plant", cadence).is_some());

    let fired = all.advance_by(1000);
    assert!(fired.len() > 50, "one per occurrence");
    assert!(fired.iter().all(|firing| firing.occurrences == 1));

    // Cap: at most n now, the rest still owed and delivered as the clock runs.
    let mut capped: StochasticScheduler<&str> = StochasticScheduler::new(domain());
    let cadence = capped.register(Cadence::mtth(10).on_backlog(OnBacklog::Cap(3)));
    assert!(capped.insert("plant", cadence).is_some());

    let fired = capped.advance_by(1000);
    assert_eq!(fired.len(), 3, "capped at three");
    let next = capped.advance();
    assert!(!next.is_empty(), "the rest drains on the following steps");
    assert!(next.len() <= 3);
}

#[test]
fn the_unique_variant_keeps_one_occurrence_per_value() {
    let mut events: UniqueStochasticScheduler<&str> = UniqueStochasticScheduler::new(domain());
    let slow = events.register(Cadence::mtth(100));
    let fast = events.register(Cadence::mtth(3));

    assert!(events.insert("plot", slow));
    let first = events.due_at(&"plot").unwrap();
    assert_eq!(events.len(), 1);

    // Re-inserting moves it rather than adding a second occurrence, which is
    // also how a cadence is changed.
    assert!(events.insert("plot", fast));
    assert_eq!(events.len(), 1);
    assert_ne!(events.due_at(&"plot"), Some(first));
    assert!(events.contains(&"plot"));

    // It keeps re-arming itself, one pending occurrence at a time.
    for _ in 0..50 {
        events.advance();
        assert_eq!(events.len(), 1);
        assert!(events.due_at(&"plot").unwrap() > events.now());
    }

    assert!(events.cancel(&"plot"));
    assert!(!events.cancel(&"plot"));
    assert!(events.is_empty());
    assert_eq!(events.due_at(&"plot"), None);

    // An id no cadence was registered for is refused rather than scheduled.
    let mut fresh: UniqueStochasticScheduler<&str> = UniqueStochasticScheduler::new(domain());
    assert!(!fresh.insert("plot", fast));
    assert!(fresh.is_empty());
}

#[test]
fn the_same_seed_replays_exactly_whichever_source_is_used() {
    fn run_seeded() -> Vec<(u64, u32, u32)> {
        let mut events: StochasticScheduler<u32> = StochasticScheduler::new(domain());
        let cadence = events.register(Cadence::mtth(20).spread(3));
        for entity in 0..20 {
            assert!(events.insert(entity, cadence).is_some());
        }

        let mut log = Vec::new();
        for _ in 0..300 {
            for firing in events.advance() {
                log.push((events.now(), firing.value, firing.occurrences));
            }
        }
        log
    }

    // A Seed-backed scheduler replays bit for bit.
    assert_eq!(run_seeded(), run_seeded());
    assert!(!run_seeded().is_empty());

    // So does one whose entries own streams, since those streams are derived
    // from the same domain.
    fn run_streamed() -> Vec<(u64, u32)> {
        let mut events: StochasticScheduler<u32, Random> = StochasticScheduler::new(domain());
        let cadence = events.register(Cadence::mtth(20).spread(3));
        for entity in 0..20 {
            assert!(events.insert(entity, cadence).is_some());
        }

        let mut log = Vec::new();
        for _ in 0..300 {
            for firing in events.advance() {
                log.push((events.now(), firing.value));
            }
        }
        log
    }

    assert_eq!(run_streamed(), run_streamed());

    // The two disagree with each other, which is the point of choosing.
    let seeded: Vec<(u64, u32)> = run_seeded()
        .into_iter()
        .map(|(step, value, _)| (step, value))
        .collect();
    assert_ne!(seeded, run_streamed());

    // A different world seed gives a different schedule.
    let mut other: StochasticScheduler<u32> = StochasticScheduler::new(Seed::from_raw(7));
    let cadence = other.register(Cadence::mtth(20).spread(3));
    assert!(other.insert(0, cadence).is_some());
    let mut same: StochasticScheduler<u32> = StochasticScheduler::new(domain());
    let cadence = same.register(Cadence::mtth(20).spread(3));
    assert!(same.insert(0, cadence).is_some());
    assert_ne!(other.next_due(), same.next_due());
}

#[test]
fn entries_may_carry_their_own_kind_of_randomness() {
    let mut events: StochasticScheduler<&str, EventRandom> = StochasticScheduler::new(domain());
    let cadence = events.register(Cadence::mtth(25));

    // One entry follows a derived seed, another its own stream.
    assert!(events.insert("seeded", cadence).is_some());
    assert!(
        events
            .insert_with(
                "streamed",
                cadence,
                EventRandom::Stream(Box::new(Random::new(Seed::from_raw(3)))),
            )
            .is_some()
    );
    assert_eq!(events.len(), 2);

    let mut fired = 0;
    for _ in 0..500 {
        fired += events.advance().len();
    }
    assert!(fired > 20, "both kinds keep firing");
    assert_eq!(events.len(), 2);
}

#[test]
fn scheduled_work_can_be_inspected_and_cancelled() {
    let mut events: StochasticScheduler<u32> = StochasticScheduler::new(domain());
    let cadence = events.register(Cadence::mtth(30));
    for entity in 0..10 {
        assert!(events.insert(entity, cadence).is_some());
    }

    assert_eq!(events.cadence(cadence).unwrap().mean(), 30);
    assert_eq!(events.cadence_count(), 1);
    assert!(events.next_due().unwrap() > events.now());
    assert_eq!(events.iter().count(), 10);

    // Range over the steps work is due on.
    let horizon = events.now() + 1000;
    assert_eq!(events.range(..horizon).count(), 10);
    assert_eq!(events.range(horizon..).count(), 0);

    assert_eq!(events.cancel(&3), 1);
    assert_eq!(events.cancel(&3), 0);
    assert_eq!(events.len(), 9);
    assert!(!events.contains(&3));

    events.clear();
    assert!(events.is_empty());
    assert_eq!(events.cadence_count(), 1, "cadences survive a clear");
}

#[test]
fn a_firing_says_where_its_entry_went() {
    let mut events: StochasticScheduler<u32> = StochasticScheduler::new(domain());
    let cadence = events.register(Cadence::mtth(15).spread(2));
    for entity in 0..40 {
        assert!(events.insert(entity, cadence).is_some());
    }

    for _ in 0..300 {
        for firing in events.advance() {
            assert!(firing.next > events.now(), "re-armed into the future");
            assert!(
                events.at(firing.next).any(|value| *value == firing.value),
                "the entry is where its firing said"
            );
        }
    }
}

#[test]
fn a_capped_backlog_reports_the_step_the_remainder_comes_on() {
    let mut events: StochasticScheduler<&str> = StochasticScheduler::new(domain());
    let cadence = events.register(Cadence::mtth(10).on_backlog(OnBacklog::Cap(2)));
    assert!(events.insert("plant", cadence).is_some());

    let fired = events.advance_by(500);
    assert_eq!(fired.len(), 2);
    assert!(
        fired.iter().all(|firing| firing.next == events.now() + 1),
        "still owed, so the remainder is due next step"
    );
    assert!(events.at(events.now() + 1).any(|value| *value == "plant"));
}

#[test]
fn cancelling_at_a_known_step_touches_only_that_step() {
    let mut events: StochasticScheduler<u32> = StochasticScheduler::new(domain());
    let cadence = events.register(Cadence::mtth(40));

    let first = events.insert(1, cadence).unwrap();
    let second = events.insert(1, cadence).unwrap();
    assert_ne!(first, second, "two copies land apart");
    assert!(events.insert(2, cadence).is_some());
    assert_eq!(events.len(), 3);

    // Only the copy at that step goes, and only that value.
    assert_eq!(events.cancel_at(second, &1), 1);
    assert_eq!(events.len(), 2);
    assert!(events.at(first).any(|value| *value == 1));
    assert_eq!(events.cancel_at(second, &1), 0);

    // A step nothing is due on is a no-op.
    assert_eq!(events.cancel_at(events.now() + 100_000, &1), 0);
    assert_eq!(events.len(), 2);
}

#[test]
fn the_unique_index_tracks_the_schedule_through_every_operation() {
    let mut events: UniqueStochasticScheduler<u32> = UniqueStochasticScheduler::new(domain());
    let quick = events.register(Cadence::mtth(8).spread(2));
    let slow = events.register(Cadence::mtth(60).on_backlog(OnBacklog::Cap(2)));

    for entity in 0..60 {
        assert!(events.insert(entity, if entity % 2 == 0 { quick } else { slow }));
    }

    // Every value's recorded step is where the schedule actually holds it, and
    // the schedule holds nothing the index has forgotten.
    let agrees = |events: &UniqueStochasticScheduler<u32>| {
        let mut seen = 0;
        for (step, value) in events.range(..) {
            assert_eq!(
                events.due_at(value),
                Some(step),
                "index disagrees for {value}"
            );
            seen += 1;
        }
        assert_eq!(seen, events.len(), "index and schedule differ in size");
    };

    agrees(&events);

    for round in 0..200 {
        events.advance();
        agrees(&events);

        // Churn: re-insert, cancel, and re-add while it is running.
        if round % 7 == 0 {
            assert!(events.insert(round % 60, quick));
            events.cancel(&((round + 1) % 60));
            agrees(&events);
        }
    }

    // A long skip that leaves work owed keeps the index honest too.
    events.advance_by(5_000);
    agrees(&events);
}

#[test]
fn a_cadence_id_is_only_good_at_the_scheduler_that_issued_it() {
    let mut fast: StochasticScheduler<u32> = StochasticScheduler::new(Seed::from_raw(1));
    let mut slow: StochasticScheduler<u32> = StochasticScheduler::new(Seed::from_raw(2));

    let quick = fast.register(Cadence::mtth(2));
    let crawl = slow.register(Cadence::mtth(100_000));

    // Both sit at position zero, and are still distinguishable.
    assert_eq!(quick.index(), 0);
    assert_eq!(crawl.index(), 0);
    assert_ne!(quick, crawl);

    // Neither scheduler resolves the other's id, though the position is in
    // range at both.
    assert!(slow.cadence(quick).is_none());
    assert!(fast.cadence(crawl).is_none());
    assert_eq!(slow.cadence(crawl).unwrap().mean(), 100_000);
    assert_eq!(fast.cadence(quick).unwrap().mean(), 2);

    // And nothing is scheduled on a cadence that was never meant for it.
    assert_eq!(slow.insert(1, quick), None);
    assert!(slow.is_empty());
    assert_eq!(
        slow.insert_with(1, quick, Seed::from_raw(9).cursor()),
        None,
        "the same check on the explicit path"
    );
    assert!(slow.is_empty());

    // Its own id still works.
    assert!(slow.insert(1, crawl).is_some());
    assert_eq!(slow.len(), 1);
}

#[test]
fn ids_keep_working_on_a_clone_of_the_scheduler_that_issued_them() {
    let mut events: StochasticScheduler<u32> = StochasticScheduler::new(domain());
    let cadence = events.register(Cadence::mtth(30));
    assert!(events.insert(1, cadence).is_some());

    // A clone carries the issuer's tag, so ids taken before it was cloned are
    // still good, and both sides go on agreeing.
    let mut copy = events.clone();
    assert_eq!(copy.cadence(cadence).unwrap().mean(), 30);
    assert!(copy.insert(2, cadence).is_some());
    assert_eq!(copy.len(), 2);
    assert_eq!(events.len(), 1);
}

#[test]
fn the_unique_variant_refuses_a_foreign_cadence_too() {
    let mut first: UniqueStochasticScheduler<u32> =
        UniqueStochasticScheduler::new(Seed::from_raw(1));
    let mut second: UniqueStochasticScheduler<u32> =
        UniqueStochasticScheduler::new(Seed::from_raw(2));

    let theirs = first.register(Cadence::mtth(5));
    let ours = second.register(Cadence::mtth(5));

    assert!(second.cadence(theirs).is_none());
    assert!(!second.insert(1, theirs));
    assert!(second.is_empty());
    assert!(second.insert(1, ours));
    assert_eq!(second.len(), 1);
}
