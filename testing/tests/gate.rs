//! Level-triggered gating: what each family does when a condition is false.

use voxel_world::random::seed::Seed;
use voxel_world::structures::collections::{
    Cadence, MultiRota, OrderedMultiRota, OrderedRota, Rota, Scheduler, StochasticScheduler,
    UniqueScheduler, UniqueStochasticScheduler,
};
use voxel_world::structures::indices::{Expr, Gate, PropertyKind, SchemaError};
use voxel_world::structures::traits::{Collection, Map};

type Gates = Gate<u32, &'static str, &'static str>;

fn gate() -> Gates {
    let mut gate = Gates::new();
    gate.register_categorical_property("ground", PropertyKind::Single, ())
        .unwrap();
    gate.register_categorical_fact("season", ()).unwrap();
    gate
}

/// Restricts `element` to thawed ground, and sets the ground to `state`.
fn restrict(gate: &mut Gates, element: u32, state: &'static str) {
    gate.require(element, Expr::is("ground", "thawed")).unwrap();
    gate.set(element, "ground", state).unwrap();
}

#[test]
fn a_gate_answers_about_now_and_has_no_memory_of_edges() {
    let mut gate = gate();
    restrict(&mut gate, 1, "frozen");

    assert!(gate.refuses(&1));
    assert!(!gate.allows(&1));

    gate.set(1, "ground", "thawed").unwrap();
    assert!(gate.allows(&1));

    // Asking again says the same. There is no crossing to consume, which is the
    // whole difference from a condition index.
    assert!(gate.allows(&1));
    assert!(gate.allows(&1));

    gate.set(1, "ground", "frozen").unwrap();
    assert!(gate.refuses(&1));
}

#[test]
fn an_element_with_no_condition_is_allowed() {
    let mut gate = gate();

    assert!(gate.allows(&99), "never heard of it");
    assert!(!gate.is_restricted(&99));
    assert!(gate.is_empty());

    restrict(&mut gate, 99, "frozen");
    assert!(gate.refuses(&99));
    assert!(gate.is_restricted(&99));
    assert_eq!(gate.len(), 1);

    // Lifting the condition lets it through again.
    assert!(gate.clear_condition(&99).is_some());
    assert!(gate.allows(&99));
    assert!(gate.clear_condition(&99).is_none());
}

#[test]
fn a_facts_change_moves_every_gated_element_at_once() {
    let mut gate = gate();
    for element in 0..5u32 {
        gate.require(element, Expr::is("season", "spring")).unwrap();
    }

    assert!((0..5).all(|element| gate.refuses(&element)));

    gate.set_fact("season", "spring").unwrap();
    assert!((0..5).all(|element| gate.allows(&element)));

    gate.clear_fact(&"season");
    assert!((0..5).all(|element| gate.refuses(&element)));
}

#[test]
fn a_condition_that_could_never_hold_is_refused_when_it_is_required() {
    let mut gate = gate();

    assert_eq!(
        gate.require(1, Expr::has("nowhere")),
        Err(SchemaError::Unregistered {
            property: "nowhere"
        })
    );
    assert!(!gate.is_restricted(&1), "nothing was registered");
    assert!(gate.allows(&1));

    // A sound one is kept, and a later bad one leaves it standing.
    gate.require(1, Expr::is("ground", "thawed")).unwrap();
    assert!(gate.require(1, Expr::has("nowhere")).is_err());
    assert_eq!(gate.condition(&1), Some(&Expr::is("ground", "thawed")));
}

#[test]
fn a_plain_scheduler_drops_what_the_gate_refuses() {
    let mut gate = gate();
    restrict(&mut gate, 1, "frozen");
    restrict(&mut gate, 2, "thawed");

    let mut work: Scheduler<u32> = Scheduler::new();
    work.schedule(1, 1);
    work.schedule(1, 2);
    work.schedule(1, 3);

    // Two passes its condition, three has none; one is refused and is gone.
    let mut fired = work.advance_with(&gate);
    fired.sort_unstable();
    assert_eq!(fired, vec![2, 3]);

    assert!(work.is_empty(), "a refused value is not rescheduled");

    // Thawing it later brings nothing back: the occurrence was spent.
    gate.set(1, "ground", "thawed").unwrap();
    assert!(work.advance_by_with(100, &gate).is_empty());
}

#[test]
fn a_unique_scheduler_drops_what_the_gate_refuses_too() {
    let mut gate = gate();
    restrict(&mut gate, 1, "frozen");

    let mut work: UniqueScheduler<u32> = UniqueScheduler::new();
    work.schedule(1, 1);
    work.schedule(1, 2);

    assert_eq!(work.advance_with(&gate), vec![2]);
    assert!(work.is_empty());
    assert_eq!(work.step_of(&1), None, "no pending occurrence left");
}

#[test]
fn a_drawn_scheduler_re_arms_what_the_gate_refuses() {
    let mut gate = gate();
    restrict(&mut gate, 1, "frozen");

    let mut growth: StochasticScheduler<u32> = StochasticScheduler::new(Seed::from_raw(7));
    let cadence = growth.register(Cadence::mtth(4));
    assert!(growth.insert(1, cadence).is_some());

    // Nothing comes through while it is frozen, but the entry is never lost:
    // it keeps drawing its next occurrence as it would have anyway.
    let mut fired = 0;
    for _ in 0..400 {
        fired += growth.advance_with(&gate).len();
        assert_eq!(growth.len(), 1, "still scheduled");
    }
    assert_eq!(fired, 0, "every occurrence was refused");

    // Thawing it makes the next occurrence count, without rescheduling by hand.
    gate.set(1, "ground", "thawed").unwrap();
    let mut fired = 0;
    for _ in 0..400 {
        fired += growth.advance_with(&gate).len();
    }
    assert!(fired > 50, "about a hundred, on its own cadence");
    assert_eq!(growth.len(), 1);
}

#[test]
fn a_refused_occurrence_costs_the_same_as_a_taken_one() {
    // The entry keeps its own schedule whether or not the gate lets it act, so
    // occurrences are not saved up while it is barred.
    let open = gate();
    let mut shut = gate();
    shut.require(1, Expr::is("ground", "thawed")).unwrap();
    shut.set(1, "ground", "frozen").unwrap();

    let domain = Seed::from_raw(11);
    let mut counted: StochasticScheduler<u32> = StochasticScheduler::new(domain);
    let cadence = counted.register(Cadence::mtth(10));
    counted.insert(1, cadence).unwrap();

    let mut barred: StochasticScheduler<u32> = StochasticScheduler::new(domain);
    let cadence = barred.register(Cadence::mtth(10));
    barred.insert(1, cadence).unwrap();

    for _ in 0..1_000 {
        counted.advance_with(&open);
        barred.advance_with(&shut);
    }

    // Both schedulers sit at the same place, having drawn the same waits.
    assert_eq!(counted.next_due(), barred.next_due());
    assert!(open.allows(&1) && shut.refuses(&1));
}

#[test]
fn a_unique_drawn_scheduler_keeps_its_index_through_refusals() {
    let mut gate = gate();
    restrict(&mut gate, 1, "frozen");

    let mut growth: UniqueStochasticScheduler<u32> =
        UniqueStochasticScheduler::new(Seed::from_raw(3));
    let cadence = growth.register(Cadence::mtth(5));
    assert!(growth.insert(1, cadence));

    for _ in 0..200 {
        assert!(growth.advance_with(&gate).is_empty());
        assert_eq!(growth.len(), 1);
        // The index follows the re-armed entry even though nothing was reported.
        assert!(growth.due_at(&1).unwrap() > growth.now());
    }

    gate.set(1, "ground", "thawed").unwrap();
    let mut fired = 0;
    for _ in 0..200 {
        fired += growth.advance_with(&gate).len();
    }
    assert!(fired > 10);
}

#[test]
fn a_rota_hands_out_only_the_members_a_gate_allows() {
    let mut gate = gate();
    for element in 0..8u32 {
        gate.require(element, Expr::is("ground", "thawed")).unwrap();
        let ground = if element % 2 == 0 { "thawed" } else { "frozen" };
        gate.set(element, "ground", ground).unwrap();
    }

    let turns: Rota<u32, 4> = (0..8).collect();

    let mut allowed: Vec<u32> = Vec::new();
    for turn in 0..4 {
        // Nothing is removed: the refused members are still in their group.
        let whole = turns.group(turn).len();
        let passing: Vec<u32> = turns.group_with(turn, &gate).copied().collect();
        assert!(passing.len() <= whole);
        allowed.extend(passing);
    }
    allowed.sort_unstable();
    assert_eq!(allowed, vec![0, 2, 4, 6], "the odd ones are frozen");

    // A later thaw lets them through on their next turn, unprompted.
    gate.set(1, "ground", "thawed").unwrap();
    let group = turns.group_of(&1).expect("a member");
    assert!(turns.group_with(group, &gate).any(|member| *member == 1));
    assert_eq!(turns.len(), 8, "membership never changed");
}

#[test]
fn every_rota_variant_can_be_gated() {
    let mut gate = gate();
    restrict(&mut gate, 1, "frozen");
    restrict(&mut gate, 2, "thawed");

    let multi: MultiRota<u32, 2> = [1u32, 1, 2, 3].into_iter().collect();
    let ordered: OrderedRota<u32, 2> = [1u32, 2, 3].into_iter().collect();
    let ordered_multi: OrderedMultiRota<u32, 2> = [1u32, 1, 2, 3].into_iter().collect();

    let seen: Vec<u32> = (0..2)
        .flat_map(|turn| multi.group_with(turn, &gate).copied().collect::<Vec<u32>>())
        .collect();
    assert!(!seen.contains(&1), "frozen, in every copy");
    assert!(seen.contains(&2) && seen.contains(&3));

    // Each group's own order survives the filter: the members a gate allows
    // come out as a subsequence of that group, with the refused ones missing.
    for turn in 0..2 {
        let whole: Vec<u32> = ordered.group(turn).copied().collect();
        let passing: Vec<u32> = ordered.group_with(turn, &gate).copied().collect();
        assert_eq!(
            passing,
            whole
                .iter()
                .copied()
                .filter(|member| *member != 1)
                .collect::<Vec<u32>>(),
            "order kept, the frozen one dropped"
        );
    }

    for turn in 0..2 {
        let whole: Vec<u32> = ordered_multi.group(turn).copied().collect();
        let passing: Vec<u32> = ordered_multi.group_with(turn, &gate).copied().collect();
        assert_eq!(
            passing,
            whole
                .iter()
                .copied()
                .filter(|member| *member != 1)
                .collect::<Vec<u32>>()
        );
    }

    // And between them the two ordered rotas still show everything allowed.
    let mut seen: Vec<u32> = (0..2)
        .flat_map(|turn| {
            ordered
                .group_with(turn, &gate)
                .copied()
                .collect::<Vec<u32>>()
        })
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, vec![2, 3]);
}

#[test]
fn one_gate_can_serve_several_structures_at_once() {
    let mut gate = gate();
    for element in 0..4u32 {
        gate.require(element, Expr::is("season", "spring")).unwrap();
    }

    let mut once: Scheduler<u32> = Scheduler::new();
    let mut growth: StochasticScheduler<u32> = StochasticScheduler::new(Seed::from_raw(5));
    let cadence = growth.register(Cadence::mtth(3));
    let turns: Rota<u32, 2> = (0..4).collect();

    for element in 0..4u32 {
        once.schedule(1, element);
        growth.insert(element, cadence).unwrap();
    }

    // Out of season: the one-shot work is dropped, the recurring work waits,
    // and the rota hands out nothing.
    assert!(once.advance_with(&gate).is_empty());
    assert!(growth.advance_with(&gate).is_empty());
    assert_eq!((0..2).flat_map(|t| turns.group_with(t, &gate)).count(), 0);
    assert_eq!(growth.len(), 4, "the recurring work survived");

    gate.set_fact("season", "spring").unwrap();
    assert_eq!((0..2).flat_map(|t| turns.group_with(t, &gate)).count(), 4);

    let mut fired = 0;
    for _ in 0..100 {
        fired += growth.advance_with(&gate).len();
    }
    assert!(fired > 50, "and resumes without being rescheduled");
}

#[test]
fn the_capability_traits_describe_what_is_restricted() {
    use voxel_world::structures::traits::CollectionRemove;

    let mut gate = gate();
    for element in 0..4u32 {
        gate.require(element, Expr::is("ground", "thawed")).unwrap();
    }

    assert_eq!(Collection::len(&gate), 4);
    assert!(Collection::contains(&gate, &2));
    assert!(!Collection::contains(&gate, &9), "unrestricted, not held");
    assert_eq!(Map::len(&gate), 4);
    assert_eq!(Map::pairs(&gate).count(), 4);
    assert!(Map::get(&gate, &2).is_some());

    // Removing lifts the restriction and leaves the state alone.
    gate.set(2, "ground", "thawed").unwrap();
    assert!(CollectionRemove::remove(&mut gate, &2));
    assert!(gate.allows(&2));
    assert!(gate.state().has_value(&2, &"ground", &"thawed"));

    CollectionRemove::retain(&mut gate, |element| *element == 0);
    assert_eq!(Collection::len(&gate), 1);
    CollectionRemove::clear(&mut gate);
    assert!(gate.is_empty());
}

#[test]
fn the_state_can_be_reached_directly_because_nothing_is_cached() {
    let mut gate = gate();
    restrict(&mut gate, 1, "frozen");

    // A gate keeps nothing derived, so the store is safe to change directly.
    gate.store_mut().set(1, "ground", "thawed").unwrap();
    assert!(gate.allows(&1), "seen at once, with no re-check to run");

    assert!(gate.store().is_fact(&"season"));
    assert!(!gate.store().is_fact(&"ground"));
    assert_eq!(gate.store().value_kind(&"ground"), Some(&()));

    gate.reset();
    assert!(gate.is_empty());
    assert!(gate.allows(&1));
}
