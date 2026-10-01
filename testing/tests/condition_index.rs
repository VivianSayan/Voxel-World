//! Conditions registered over element state: when they fire, what wakes them,
//! and that the single-element evaluator agrees with the set-building one.

use std::sync::Arc;
use voxel_world::structures::indices::{
    Comparability, ConditionIndex, Expr, OnSatisfied, PropertyKind, PropertyQuery, SchemaError,
};
use voxel_world::structures::traits::{Collection, Kinded};

/// A value type with several logical kinds, which is the case the schema is
/// there for: every variant has the same Rust type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Value {
    Ground(&'static str),
    Light(&'static str),
    Season(&'static str),
    Depth(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    Ground,
    Light,
    Season,
    Depth,
}

/// Written by hand, never derived: deriving would order `Ground("a")` against
/// `Depth(1)` by variant position, which means nothing. `None` says "these do
/// not compare", which is what a categorical kind and a cross-kind pair both
/// are.
impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            // The only kind here with a real order.
            (Self::Depth(a), Self::Depth(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl Kinded for Value {
    type Kind = Kind;

    fn kind(&self) -> Kind {
        match self {
            Self::Ground(_) => Kind::Ground,
            Self::Light(_) => Kind::Light,
            Self::Season(_) => Kind::Season,
            Self::Depth(_) => Kind::Depth,
        }
    }
}

const THAWED: Value = Value::Ground("thawed");
const FROZEN: Value = Value::Ground("frozen");
const FULL: Value = Value::Light("full");
const SPRING: Value = Value::Season("spring");

type Gates = ConditionIndex<u32, &'static str, Value>;

fn gates() -> Gates {
    let mut gates = Gates::new();
    gates
        .register_categorical_property("ground", PropertyKind::Single, Kind::Ground)
        .expect("a fresh schema");
    gates
        .register_categorical_property("light", PropertyKind::Single, Kind::Light)
        .expect("a fresh schema");
    gates.register_flag("tended").expect("a fresh schema");
    gates
        .register_categorical_fact("season", Kind::Season)
        .expect("a fresh schema");
    gates
}

#[test]
fn a_condition_fires_when_it_becomes_true_and_not_while_it_stays_true() {
    let mut gates = gates();
    gates.watch(1, Expr::is("ground", THAWED)).unwrap();

    assert!(!gates.has_ready());
    assert!(gates.is_watching(&1));

    gates.set(1, "ground", FROZEN).unwrap();
    assert!(!gates.has_ready(), "the wrong value is not a crossing");

    gates.set(1, "ground", THAWED).unwrap();
    assert_eq!(gates.take_ready(), vec![1]);

    // Writing it again is no crossing, and the condition is spent anyway.
    gates.set(1, "ground", THAWED).unwrap();
    assert!(!gates.has_ready());
    assert!(!gates.is_watching(&1), "Once retires the condition");
}

#[test]
fn a_condition_that_already_holds_fires_at_once() {
    let mut gates = gates();
    gates.set(1, "ground", THAWED).unwrap();

    assert!(
        gates.watch(1, Expr::is("ground", THAWED)).unwrap(),
        "registration reports that it was already satisfied"
    );
    assert_eq!(gates.take_ready(), vec![1]);
    assert!(!gates.is_watching(&1));
}

#[test]
fn a_rearming_condition_waits_for_the_next_crossing() {
    let mut gates = gates();
    gates
        .watch_with(1, Expr::is("ground", THAWED), OnSatisfied::Rearm)
        .unwrap();

    gates.set(1, "ground", THAWED).unwrap();
    assert_eq!(gates.take_ready(), vec![1]);
    assert!(gates.is_watching(&1), "Rearm keeps the condition");
    assert!(gates.is_satisfied(&1));

    // Still true, so there is nothing to report.
    gates.set(1, "ground", THAWED).unwrap();
    assert!(!gates.has_ready());

    // It has to go false before it can come true again.
    gates.set(1, "ground", FROZEN).unwrap();
    assert!(!gates.has_ready());
    assert!(!gates.is_satisfied(&1));

    gates.set(1, "ground", THAWED).unwrap();
    assert_eq!(gates.take_ready(), vec![1]);
}

#[test]
fn a_conjunction_waits_for_every_part() {
    let mut gates = gates();
    gates
        .watch(
            1,
            Expr::and([
                Expr::is("ground", THAWED),
                Expr::is("light", FULL),
                Expr::has("tended"),
            ]),
        )
        .unwrap();

    gates.set(1, "ground", THAWED).unwrap();
    gates.set(1, "light", FULL).unwrap();
    assert!(!gates.has_ready(), "two of three");

    gates.set_flag(1, "tended").unwrap();
    assert_eq!(gates.take_ready(), vec![1]);
}

#[test]
fn a_fact_wakes_everything_that_reads_it_and_nothing_else() {
    let mut gates = gates();

    // Half wait on spring, half on the ground alone.
    for element in 0..10u32 {
        if element % 2 == 0 {
            gates
                .watch(
                    element,
                    Expr::and([Expr::is("ground", THAWED), Expr::is("season", SPRING)]),
                )
                .unwrap();
        } else {
            gates.watch(element, Expr::is("ground", THAWED)).unwrap();
        }
        gates.set(element, "ground", THAWED).unwrap();
    }

    // The odd ones went through on their own state.
    assert_eq!(gates.take_ready(), vec![1, 3, 5, 7, 9]);
    assert_eq!(gates.len(), 5);

    // One fact releases the rest, in element order.
    gates.set_fact("season", SPRING).unwrap();
    let mut released = gates.take_ready();
    released.sort_unstable();
    assert_eq!(released, vec![0, 2, 4, 6, 8]);
    assert!(gates.is_empty());

    // A fact nothing reads wakes nothing.
    gates.watch(11, Expr::is("ground", THAWED)).unwrap();
    gates.set_fact("season", Value::Season("winter")).unwrap();
    assert!(!gates.has_ready());
}

#[test]
fn facts_and_element_properties_do_not_collide() {
    let mut gates = gates();
    assert!(
        gates
            .register_categorical_property("season", PropertyKind::Single, Kind::Season)
            .is_err(),
        "already a fact"
    );
    assert!(
        gates
            .register_categorical_fact("ground", Kind::Ground)
            .is_err(),
        "already a property"
    );
    assert!(gates.is_fact(&"season"));
    assert!(!gates.is_fact(&"ground"));

    // Writing a fact through the element API is refused, not quietly ignored:
    // to the element half of the schema that name does not exist.
    assert_eq!(
        gates.set(1, "season", SPRING),
        Err(SchemaError::Unregistered { property: "season" })
    );
    assert!(!gates.fact_holds(&"season"));

    // A valued fact cannot be raised as though it were a flag.
    assert!(gates.set_fact_flag("season").is_err());

    gates.set_fact("season", SPRING).unwrap();
    assert!(gates.fact_holds(&"season"));
    assert!(gates.fact_is(&"season", &SPRING));
    assert!(gates.clear_fact(&"season"));
    assert!(!gates.clear_fact(&"season"));

    // A fact that carries no value is raised and dropped instead.
    gates.register_fact_flag("war").unwrap();
    assert_eq!(gates.value_kind(&"war"), None);
    assert!(gates.set_fact("war", SPRING).is_err(), "it holds no value");
    gates.set_fact_flag("war").unwrap();
    assert!(gates.fact_holds(&"war"));

    // And the declared kinds are readable.
    assert_eq!(gates.value_kind(&"ground"), Some(&Kind::Ground));
    assert_eq!(gates.value_kind(&"tended"), None, "a flag has none");
}

#[test]
fn negation_is_rechecked_when_the_universe_changes() {
    let mut gates = gates();

    // "In the world, and not frozen."
    gates
        .watch(1, Expr::negate(Expr::is("ground", FROZEN)))
        .unwrap();
    assert!(!gates.has_ready(), "element 1 is not in the universe yet");

    gates.insert_element(1);
    assert_eq!(
        gates.take_ready(),
        vec![1],
        "joining the universe satisfies it"
    );

    // The same condition, re-armed, goes false when the element leaves.
    gates
        .watch_with(
            2,
            Expr::negate(Expr::is("ground", FROZEN)),
            OnSatisfied::Rearm,
        )
        .unwrap();
    gates.insert_element(2);
    assert_eq!(gates.take_ready(), vec![2]);

    gates.remove_element(&2);
    assert!(
        !gates.is_satisfied(&2),
        "outside the universe satisfies nothing"
    );

    gates.insert_element(2);
    assert_eq!(gates.take_ready(), vec![2], "and fires again on return");
}

#[test]
fn re_registering_replaces_and_unwatching_stops_the_wake_ups() {
    let mut gates = gates();
    gates.watch(1, Expr::is("ground", THAWED)).unwrap();
    gates.watch(1, Expr::is("light", FULL)).unwrap();
    assert_eq!(gates.len(), 1);

    // The old condition is gone, so its property no longer wakes anything.
    gates.set(1, "ground", THAWED).unwrap();
    assert!(!gates.has_ready());

    gates.set(1, "light", FULL).unwrap();
    assert_eq!(gates.take_ready(), vec![1]);

    // Unwatching leaves no trace behind in the reverse index.
    gates.watch(2, Expr::is("ground", FROZEN)).unwrap();
    assert!(gates.unwatch(&2));
    assert!(!gates.unwatch(&2));
    gates.set(2, "ground", FROZEN).unwrap();
    assert!(!gates.has_ready());
    assert!(gates.is_empty());
}

#[test]
fn the_ready_list_is_taken_once_and_keeps_its_order() {
    let mut gates = gates();
    for element in [5u32, 3, 9, 1] {
        gates.watch(element, Expr::is("ground", THAWED)).unwrap();
    }

    for element in [9u32, 1, 5, 3] {
        gates.set(element, "ground", THAWED).unwrap();
    }

    assert_eq!(gates.ready_len(), 4);
    assert_eq!(
        gates.take_ready(),
        vec![9, 1, 5, 3],
        "the order they came true in"
    );
    assert!(!gates.has_ready(), "taking drains it");
    assert_eq!(gates.take_ready(), Vec::<u32>::new());
}

#[test]
fn the_single_element_evaluator_agrees_with_the_set_query() {
    // The same state and expressions, evaluated both ways round.
    let mut gates = Gates::new();
    let mut query: PropertyQuery<u32, &str, Value> = PropertyQuery::new();

    for (property, kind, value_kind) in [
        ("ground", PropertyKind::Single, Kind::Ground),
        ("light", PropertyKind::Single, Kind::Light),
    ] {
        gates
            .register_categorical_property(property, kind, value_kind)
            .unwrap();
        query
            .register_categorical_property(property, kind, value_kind)
            .unwrap();
    }
    gates.register_flag("tended").unwrap();
    query.register_flag("tended").unwrap();

    for element in 0..12u32 {
        let ground = if element % 3 == 0 { THAWED } else { FROZEN };
        gates.set(element, "ground", ground.clone()).unwrap();
        query.set(element, "ground", ground).unwrap();

        if element % 2 == 0 {
            gates.set(element, "light", FULL).unwrap();
            query.set(element, "light", FULL).unwrap();
        }
        if element % 4 == 0 {
            gates.set_flag(element, "tended").unwrap();
            query.set_flag(element, "tended").unwrap();
        }
    }
    // One element in the universe with nothing on it at all.
    gates.insert_element(99);
    query.insert_element(99);

    let expressions: Vec<Arc<Expr<&str, Value>>> = vec![
        Expr::has("tended"),
        Expr::is("ground", THAWED),
        Expr::and([Expr::is("ground", THAWED), Expr::is("light", FULL)]),
        Expr::or([Expr::has("tended"), Expr::is("ground", FROZEN)]),
        Expr::negate(Expr::is("ground", THAWED)),
        Expr::and([
            Expr::negate(Expr::has("tended")),
            Expr::or([Expr::is("light", FULL), Expr::is("ground", THAWED)]),
        ]),
        Expr::and([]),
        Expr::or([]),
    ];

    for expression in &expressions {
        // Both agree it is a legal expression to begin with.
        assert!(query.validate(expression).is_ok());
        assert!(gates.validate(expression).is_ok());

        let matching = query.query_uncached(expression).unwrap();
        for element in 0..12u32 {
            assert_eq!(
                gates.satisfies(&element, expression).unwrap(),
                matching.contains(&element),
                "element {element} against {expression:?}"
            );
        }
        assert_eq!(
            gates.satisfies(&99, expression).unwrap(),
            matching.contains(&99),
            "bare element against {expression:?}"
        );
    }
}

#[test]
fn the_schema_rejects_values_of_the_wrong_kind() {
    let mut gates = gates();

    // A depth is a perfectly good Value, and a nonsense ground.
    let wrong = gates.set(1, "ground", Value::Depth(3));
    assert_eq!(
        wrong,
        Err(SchemaError::WrongValueKind {
            property: "ground",
            expected: Kind::Ground,
            found: Kind::Depth,
        })
    );
    assert!(!gates.state().has(&1, &"ground"), "nothing was stored");

    // The same rule on every writing path.
    assert!(
        gates
            .add_value(1, "ground", Value::Season("spring"))
            .is_err()
    );
    assert!(gates.set_fact("season", THAWED).is_err());

    // And on a flag, which holds no value at all.
    assert_eq!(
        gates.set(1, "tended", THAWED),
        Err(SchemaError::FlagTakesNoValue { property: "tended" })
    );

    // An unregistered property is not invented from the first write.
    assert_eq!(
        gates.set(1, "rumour", THAWED),
        Err(SchemaError::Unregistered { property: "rumour" })
    );

    // The right kind still goes in.
    assert!(gates.set(1, "ground", THAWED).is_ok());
    assert!(gates.state().has_value(&1, &"ground", &THAWED));
}

#[test]
fn a_condition_of_the_wrong_kind_is_refused_when_it_is_registered() {
    let mut gates = gates();
    gates.watch(1, Expr::is("ground", THAWED)).unwrap();

    // A condition that could never match is caught at registration, not left
    // to never fire.
    let refused = gates.watch(2, Expr::is("ground", Value::Depth(3)));
    assert_eq!(
        refused,
        Err(SchemaError::WrongValueKind {
            property: "ground",
            expected: Kind::Ground,
            found: Kind::Depth,
        })
    );
    assert!(!gates.is_watching(&2), "nothing was registered");

    // Nested just as deeply.
    assert!(
        gates
            .watch(
                2,
                Expr::and([
                    Expr::is("ground", THAWED),
                    Expr::negate(Expr::is("light", Value::Season("spring"))),
                ]),
            )
            .is_err()
    );
    assert!(
        gates.watch(2, Expr::has("nowhere")).is_err(),
        "unregistered"
    );

    // A refused registration leaves an existing condition standing.
    assert!(gates.watch(1, Expr::is("ground", Value::Depth(1))).is_err());
    assert!(gates.is_watching(&1));
    gates.set(1, "ground", THAWED).unwrap();
    assert_eq!(gates.take_ready(), vec![1], "the old condition still works");
}

#[test]
fn the_error_says_what_was_wrong() {
    let mut gates = gates();
    let message = gates
        .set(1, "ground", Value::Depth(3))
        .expect_err("the kinds differ")
        .to_string();

    assert!(message.contains("ground"), "{message}");
    assert!(message.contains("Ground"), "{message}");
    assert!(message.contains("Depth"), "{message}");
}

#[test]
fn conditions_survive_state_churn_without_leaking() {
    let mut gates = gates();

    for round in 0..200u32 {
        let element = round % 20;

        gates
            .watch_with(
                element,
                Expr::and([Expr::is("ground", THAWED), Expr::has("tended")]),
                OnSatisfied::Rearm,
            )
            .unwrap();
        gates.set(element, "ground", THAWED).unwrap();
        gates.set_flag(element, "tended").unwrap();
        gates.clear_property(&element, &"tended");

        if round % 5 == 0 {
            gates.remove_element(&element);
        }
        gates.take_ready();
    }

    // One condition per element, and the reverse index has not grown a tail.
    assert_eq!(gates.len(), 20);
    for element in 0..20u32 {
        assert!(gates.is_watching(&element));
        assert!(!gates.is_satisfied(&element), "tending was cleared");
    }

    // Everything still fires.
    for element in 0..20u32 {
        gates.set(element, "ground", THAWED).unwrap();
        gates.set_flag(element, "tended").unwrap();
    }
    let mut fired = gates.take_ready();
    fired.sort_unstable();
    assert_eq!(fired, (0..20).collect::<Vec<u32>>());
}

#[test]
fn the_capability_traits_describe_it_the_same_way_the_inherent_methods_do() {
    use voxel_world::structures::traits::{CollectionRemove, Map, Pending};

    let mut gates = gates();
    for element in 0..6u32 {
        gates.watch(element, Expr::is("ground", THAWED)).unwrap();
    }

    // Collection and Map agree on what is held.
    assert_eq!(Collection::len(&gates), 6);
    assert_eq!(Map::len(&gates), 6);
    assert!(Collection::contains(&gates, &3));
    assert!(Map::contains_key(&gates, &3));
    assert_eq!(Map::pairs(&gates).count(), 6);

    let held = Map::get(&gates, &3).expect("a condition is registered");
    assert!(gates.contains_pair(&3, &held.clone()));
    assert!(!gates.contains_pair(&3, &Expr::has("tended")));

    // Pending separates what is held from what is ready.
    assert_eq!(Pending::pending_len(&gates), 6);
    assert_eq!(Pending::ready_len(&gates), 0);
    assert!(!Pending::has_ready(&gates));
    assert!(Pending::has_pending(&gates));

    gates.set(3, "ground", THAWED).unwrap();
    assert_eq!(Pending::ready_len(&gates), 1);
    assert_eq!(
        Pending::pending_len(&gates),
        5,
        "Once retires the condition as it becomes ready"
    );
    assert_eq!(Pending::take_ready(&mut gates), vec![3]);

    // Removal through the trait drops conditions, not state.
    assert!(CollectionRemove::remove(&mut gates, &4));
    assert!(!CollectionRemove::remove(&mut gates, &4));
    CollectionRemove::retain(&mut gates, |element| element % 2 == 0);
    assert_eq!(Collection::len(&gates), 2, "0 and 2 remain");

    CollectionRemove::clear(&mut gates);
    assert!(Collection::is_empty(&gates));
    assert!(
        gates.state().universe().contains(&3),
        "state outlives the conditions judged against it"
    );
}

#[test]
fn every_holder_of_deferred_work_drains_the_same_way() {
    use voxel_world::random::seed::Seed;
    use voxel_world::structures::collections::{
        Cadence, Scheduler, StochasticScheduler, UniqueScheduler, UniqueStochasticScheduler,
    };
    use voxel_world::structures::traits::Pending;

    // A plain scheduler: ready is what the clock has reached.
    let mut once: Scheduler<&str> = Scheduler::starting_at(10);
    once.schedule_at(5, "overdue");
    once.schedule_at(50, "later");
    assert_eq!(once.pending_len(), 2);
    assert_eq!(once.ready_len(), 1);
    assert_eq!(once.take_ready(), vec!["overdue"]);
    assert_eq!(once.ready_len(), 0);
    assert_eq!(once.pending_len(), 1);

    let mut unique: UniqueScheduler<&str> = UniqueScheduler::new();
    unique.schedule(0, "now");
    unique.schedule(9, "later");
    assert_eq!(unique.pending_len(), 2);
    assert_eq!(unique.ready_len(), 1);
    assert_eq!(unique.take_ready(), vec!["now"]);
    assert_eq!(unique.pending_len(), 1);

    // The drawn schedulers place entries after the clock, so nothing is ever
    // ready between advances; they still report what they hold.
    let domain = Seed::from_raw(3).child("events");
    let mut drawn: StochasticScheduler<u32> = StochasticScheduler::new(domain);
    let cadence = drawn.register(Cadence::mtth(20));
    assert!(drawn.insert(1, cadence).is_some());
    assert_eq!(drawn.pending_len(), 1);
    assert_eq!(drawn.ready_len(), 0);
    assert!(drawn.take_ready().is_empty());

    let mut drawn_unique: UniqueStochasticScheduler<u32> = UniqueStochasticScheduler::new(domain);
    let cadence = drawn_unique.register(Cadence::mtth(20));
    assert!(drawn_unique.insert(1, cadence));
    assert_eq!(drawn_unique.pending_len(), 1);
    assert_eq!(drawn_unique.ready_len(), 0);

    // A condition index: ready is what state changes have satisfied.
    let mut gates = gates();
    gates.watch(1, Expr::is("ground", THAWED)).unwrap();
    assert_eq!(gates.pending_len(), 1);
    assert_eq!(gates.ready_len(), 0);
    gates.set(1, "ground", THAWED).unwrap();
    assert_eq!(gates.ready_len(), 1);
    assert_eq!(gates.take_ready(), vec![1]);
}

#[test]
fn a_property_cannot_be_re_registered_for_a_different_value_kind() {
    let mut gates = gates();

    // The same terms twice is a repeat, not a redefinition.
    assert!(
        gates
            .register_categorical_property("ground", PropertyKind::Single, Kind::Ground)
            .is_ok()
    );

    // A different value kind under the same cardinality is refused, so values
    // already stored cannot end up under a contract nothing checked them
    // against.
    gates.set(1, "ground", THAWED).unwrap();
    assert_eq!(
        gates.register_categorical_property("ground", PropertyKind::Single, Kind::Depth),
        Err(SchemaError::ValueKindConflict {
            property: "ground",
            expected: Kind::Ground,
            incoming: Kind::Depth,
        })
    );
    assert_eq!(gates.value_kind(&"ground"), Some(&Kind::Ground));
    assert!(
        gates.state().has_value(&1, &"ground", &THAWED),
        "still there"
    );
    assert!(
        gates.set(1, "ground", Value::Depth(2)).is_err(),
        "still typed"
    );

    // A different cardinality stays an error too.
    assert!(
        gates
            .register_categorical_property("ground", PropertyKind::Multi, Kind::Ground)
            .is_err()
    );
    assert!(gates.register_flag("ground").is_err());

    // Unregistering is the deliberate migration, and then it may be re-typed.
    let mut query: PropertyQuery<u32, &str, Value> = PropertyQuery::new();
    query
        .register_categorical_property("ground", PropertyKind::Single, Kind::Ground)
        .unwrap();
    query.set(1, "ground", THAWED).unwrap();
    assert!(query.unregister_property(&"ground"));
    assert!(
        query
            .register_ordered_property("ground", PropertyKind::Multi, Kind::Depth)
            .is_ok()
    );
    assert!(query.set(1, "ground", Value::Depth(2)).is_ok());
}

#[test]
fn a_fact_cannot_be_re_registered_for_a_different_value_kind() {
    let mut gates = gates();

    assert!(
        gates
            .register_categorical_fact("season", Kind::Season)
            .is_ok(),
        "a repeat"
    );
    assert_eq!(
        gates.register_categorical_fact("season", Kind::Depth),
        Err(SchemaError::ValueKindConflict {
            property: "season",
            expected: Kind::Season,
            incoming: Kind::Depth,
        })
    );
    assert_eq!(gates.value_kind(&"season"), Some(&Kind::Season));

    // A valued fact cannot become a flag fact, nor the other way round.
    assert!(gates.register_fact_flag("season").is_err());
    gates.register_fact_flag("war").unwrap();
    assert!(gates.register_fact_flag("war").is_ok(), "a repeat");
    assert!(
        gates
            .register_categorical_fact("war", Kind::Season)
            .is_err()
    );
}

#[test]
fn queries_refuse_malformed_expressions_instead_of_answering_nothing() {
    let mut query: PropertyQuery<u32, &str, Value> = PropertyQuery::new();
    query
        .register_categorical_property("ground", PropertyKind::Single, Kind::Ground)
        .unwrap();
    query.register_flag("tended").unwrap();
    query.set(1, "ground", THAWED).unwrap();
    query.set_flag(1, "tended").unwrap();

    // A well-formed query still answers.
    let matching = query.query(&Expr::is("ground", THAWED)).unwrap();
    assert!(matching.contains(&1));

    // The wrong kind is malformed, not merely unmatched.
    let wrong = Expr::is("ground", Value::Depth(3));
    assert_eq!(
        query.query(&wrong),
        Err(SchemaError::WrongValueKind {
            property: "ground",
            expected: Kind::Ground,
            found: Kind::Depth,
        })
    );
    assert!(query.query_uncached(&wrong).is_err());

    // A value compared against a flag, and an unknown property.
    assert_eq!(
        query.query(&Expr::is("tended", THAWED)),
        Err(SchemaError::FlagTakesNoValue { property: "tended" })
    );
    assert_eq!(
        query.query(&Expr::has("rumour")),
        Err(SchemaError::Unregistered { property: "rumour" })
    );

    // Nested anywhere in the tree, through every combinator.
    assert!(
        query
            .query(&Expr::and([
                Expr::has("tended"),
                Expr::negate(Expr::is("ground", Value::Depth(1))),
            ]))
            .is_err()
    );
    assert!(
        query
            .query(&Expr::or([Expr::is("ground", Value::Season("spring"))]))
            .is_err()
    );

    // The pair helpers go through the same check.
    assert!(query.query_all([("ground", Value::Depth(1))]).is_err());
    assert!(query.query_any([("ground", Value::Depth(1))]).is_err());
    assert!(query.query_all([("ground", THAWED)]).is_ok());

    // A refused query caches nothing, so the next good one is unaffected.
    assert_eq!(query.query(&Expr::is("ground", THAWED)).unwrap().len(), 1);

    // The single-element evaluator agrees about what is malformed.
    let mut gates = gates();
    assert!(
        gates
            .satisfies(&1, &Expr::is("ground", Value::Depth(3)))
            .is_err()
    );
    assert!(gates.satisfies(&1, &Expr::is("ground", THAWED)).is_ok());
    gates.set(1, "ground", THAWED).unwrap();
    assert!(gates.satisfies(&1, &Expr::is("ground", THAWED)).unwrap());
}

#[test]
fn equality_on_a_value_type_is_total_and_agrees_with_its_kind() {
    use voxel_world::structures::traits::{KindContract, check_kinds};

    // The value type this file uses throughout derives everything, so it
    // satisfies the contract by construction.
    assert_eq!(
        check_kinds(&[
            THAWED,
            THAWED,
            FROZEN,
            FULL,
            SPRING,
            Value::Depth(1),
            Value::Depth(1),
        ]),
        Ok(())
    );

    // Equal values are equal all the way down, which is what the store relies
    // on when it looks a value up.
    assert_eq!(Value::Depth(1), Value::Depth(1));
    assert_ne!(Value::Depth(1), Value::Depth(2));
    assert_ne!(THAWED, FROZEN);
    assert_ne!(
        Value::Ground("spring"),
        Value::Season("spring"),
        "same payload, different kind"
    );

    // A hand-written equality that ignores the payload and the variant is
    // caught: two "equal" values reporting different kinds is exactly the state
    // that would let a value be stored under one kind and found under another.
    // Clippy objects to this pairing for the same reason `check_kinds` does;
    // it is broken on purpose, to show that the check catches it.
    #[allow(clippy::derived_hash_with_manual_eq)]
    #[derive(Clone, Debug, Hash)]
    enum Sloppy {
        Count(u64),
        Name(&'static str),
    }

    impl PartialEq for Sloppy {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }

    impl Eq for Sloppy {}

    impl Kinded for Sloppy {
        type Kind = bool;

        fn kind(&self) -> bool {
            matches!(self, Self::Count(_))
        }
    }

    assert_eq!(
        check_kinds(&[Sloppy::Count(1), Sloppy::Name("a")]),
        Err(KindContract::KindDiffers { left: 0, right: 1 })
    );

    // Equality that claims more than the hash agrees with is caught too.
    #[allow(clippy::derived_hash_with_manual_eq)]
    #[derive(Clone, Debug, Hash)]
    struct Loose(u64);

    impl PartialEq for Loose {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }

    impl Eq for Loose {}

    impl Kinded for Loose {
        type Kind = ();

        fn kind(&self) {}
    }

    assert_eq!(
        check_kinds(&[Loose(1), Loose(2)]),
        Err(KindContract::HashDiffers { left: 0, right: 1 })
    );
}

#[test]
fn a_value_type_with_sound_equality_behaves_in_the_store() {
    // The payloads are compared, not just the variants: two grounds with
    // different text are different values, and both can be stored and found.
    let mut gates = gates();
    gates
        .register_categorical_property("soil", PropertyKind::Multi, Kind::Ground)
        .unwrap();

    gates.add_value(1, "soil", Value::Ground("loam")).unwrap();
    gates.add_value(1, "soil", Value::Ground("clay")).unwrap();

    assert!(gates.state().has_value(&1, &"soil", &Value::Ground("loam")));
    assert!(gates.state().has_value(&1, &"soil", &Value::Ground("clay")));
    assert!(
        !gates
            .state()
            .has_value(&1, &"soil", &Value::Ground("chalk"))
    );
    assert_eq!(gates.state().values(&1, &"soil").count(), 2);

    // And a condition distinguishes them: the clay is there, the chalk is not.
    assert!(
        gates
            .watch(1, Expr::is("soil", Value::Ground("clay")))
            .unwrap(),
        "already satisfied, so it fires at once"
    );
    assert_eq!(gates.take_ready(), vec![1]);

    assert!(
        !gates
            .watch(1, Expr::is("soil", Value::Ground("chalk")))
            .unwrap()
    );
    assert!(!gates.has_ready());

    gates.add_value(1, "soil", Value::Ground("chalk")).unwrap();
    assert_eq!(gates.take_ready(), vec![1]);
}

#[test]
fn comparisons_answer_about_order_within_a_kind() {
    let mut gates = gates();
    gates
        .register_ordered_property("depth", PropertyKind::Single, Kind::Depth)
        .unwrap();

    for (element, depth) in [(1u32, 2u32), (2, 5), (3, 9)] {
        gates.set(element, "depth", Value::Depth(depth)).unwrap();
    }

    let deep = Expr::greater("depth", Value::Depth(4));
    assert!(
        !gates.satisfies(&1, &deep).unwrap(),
        "two is not above four"
    );
    assert!(gates.satisfies(&2, &deep).unwrap());
    assert!(gates.satisfies(&3, &deep).unwrap());

    // Every operator, and the inclusive edges.
    assert!(
        gates
            .satisfies(&2, &Expr::at_least("depth", Value::Depth(5)))
            .unwrap()
    );
    assert!(
        !gates
            .satisfies(&2, &Expr::greater("depth", Value::Depth(5)))
            .unwrap()
    );
    assert!(
        gates
            .satisfies(&2, &Expr::at_most("depth", Value::Depth(5)))
            .unwrap()
    );
    assert!(
        !gates
            .satisfies(&2, &Expr::less("depth", Value::Depth(5)))
            .unwrap()
    );
    assert!(
        gates
            .satisfies(&1, &Expr::less("depth", Value::Depth(5)))
            .unwrap()
    );

    // A range, and an element with no value for the property at all.
    let middling = Expr::between("depth", Value::Depth(3), Value::Depth(8));
    assert!(!gates.satisfies(&1, &middling).unwrap());
    assert!(gates.satisfies(&2, &middling).unwrap());
    assert!(!gates.satisfies(&3, &middling).unwrap());
    assert!(!gates.satisfies(&99, &middling).unwrap(), "holds no depth");
}

#[test]
fn a_comparison_against_a_kind_with_no_order_is_refused() {
    let mut gates = gates();
    gates
        .register_ordered_property("depth", PropertyKind::Single, Kind::Depth)
        .unwrap();

    // "ground" is categorical: thawed and frozen have no order between them,
    // which the value type says by returning None from partial_cmp.
    assert_eq!(
        gates.validate(&Expr::greater("ground", THAWED)),
        Err(SchemaError::NotOrdered {
            property: "ground",
            kind: Kind::Ground,
        })
    );

    // So a condition built on one is refused when it is registered, rather than
    // quietly never holding.
    assert!(gates.watch(1, Expr::less("ground", FROZEN)).is_err());
    assert!(!gates.is_watching(&1));

    // The wrong-kind check still comes first: a season is not a depth.
    assert_eq!(
        gates.validate(&Expr::greater("depth", SPRING)),
        Err(SchemaError::WrongValueKind {
            property: "depth",
            expected: Kind::Depth,
            found: Kind::Season,
        })
    );

    // And an ordered kind passes both.
    assert!(
        gates
            .validate(&Expr::greater("depth", Value::Depth(1)))
            .is_ok()
    );
}

#[test]
fn a_comparison_drives_a_condition_like_any_other_expression() {
    let mut gates = gates();
    gates
        .register_ordered_property("depth", PropertyKind::Single, Kind::Depth)
        .unwrap();
    gates.set(1, "depth", Value::Depth(1)).unwrap();

    gates
        .watch(1, Expr::at_least("depth", Value::Depth(5)))
        .unwrap();
    assert!(!gates.has_ready());

    // Crossing the threshold fires it, exactly as an equality condition would.
    gates.set(1, "depth", Value::Depth(4)).unwrap();
    assert!(!gates.has_ready());
    gates.set(1, "depth", Value::Depth(5)).unwrap();
    assert_eq!(gates.take_ready(), vec![1]);
}

#[test]
fn one_of_matches_any_listed_value() {
    let mut gates = gates();
    gates.set(1, "ground", THAWED).unwrap();
    gates.set(2, "ground", FROZEN).unwrap();
    gates.set(3, "ground", Value::Ground("flooded")).unwrap();

    let workable = Expr::one_of("ground", [THAWED, Value::Ground("flooded")]);
    assert!(gates.satisfies(&1, &workable).unwrap());
    assert!(!gates.satisfies(&2, &workable).unwrap());
    assert!(gates.satisfies(&3, &workable).unwrap());

    // It needs no order, so it works on a categorical kind where a comparison
    // would be refused.
    assert!(gates.validate(&workable).is_ok());
    assert!(gates.validate(&Expr::greater("ground", THAWED)).is_err());

    // The same answer as the Or it stands for.
    let spelled_out = Expr::or([
        Expr::is("ground", THAWED),
        Expr::is("ground", Value::Ground("flooded")),
    ]);
    for element in 1..=3u32 {
        assert_eq!(
            gates.satisfies(&element, &workable).unwrap(),
            gates.satisfies(&element, &spelled_out).unwrap()
        );
    }

    // An empty list matches nothing, as an empty Or does.
    let nothing = Expr::one_of("ground", []);
    assert!(!gates.satisfies(&1, &nothing).unwrap());

    // And a value of the wrong kind in the list is still caught.
    assert!(gates.validate(&Expr::one_of("ground", [SPRING])).is_err());
}

#[test]
fn the_set_query_agrees_with_the_single_element_evaluator_on_the_new_operators() {
    let mut query: PropertyQuery<u32, &str, Value> = PropertyQuery::new();
    query
        .register_ordered_property("depth", PropertyKind::Single, Kind::Depth)
        .unwrap();
    query
        .register_categorical_property("ground", PropertyKind::Single, Kind::Ground)
        .unwrap();

    let mut gates = gates();
    gates
        .register_ordered_property("depth", PropertyKind::Single, Kind::Depth)
        .unwrap();

    for element in 0..12u32 {
        let depth = Value::Depth(element);
        query.set(element, "depth", depth.clone()).unwrap();
        gates.set(element, "depth", depth).unwrap();

        let ground = if element % 2 == 0 { THAWED } else { FROZEN };
        query.set(element, "ground", ground.clone()).unwrap();
        gates.set(element, "ground", ground).unwrap();
    }

    let expressions = [
        Expr::greater("depth", Value::Depth(6)),
        Expr::at_most("depth", Value::Depth(3)),
        Expr::between("depth", Value::Depth(4), Value::Depth(8)),
        Expr::one_of("ground", [THAWED]),
        Expr::and([
            Expr::less("depth", Value::Depth(10)),
            Expr::one_of("ground", [FROZEN]),
        ]),
        Expr::negate(Expr::at_least("depth", Value::Depth(2))),
    ];

    for expression in &expressions {
        let matching = query.query_uncached(expression).unwrap();
        for element in 0..12u32 {
            assert_eq!(
                gates.satisfies(&element, expression).unwrap(),
                matching.contains(&element),
                "element {element} against {expression:?}"
            );
        }

        // The cached path agrees with the uncached one, twice over.
        assert_eq!(*query.query(expression).unwrap(), matching);
        assert_eq!(*query.query(expression).unwrap(), matching);
    }
}

#[test]
fn a_comparison_matches_when_any_held_value_does() {
    // For a multi property, `Is` means "holds this value", so a comparison
    // means "holds a value standing in this relation".
    let mut gates = gates();
    gates
        .register_ordered_property("depth", PropertyKind::Multi, Kind::Depth)
        .unwrap();

    gates.add_value(1, "depth", Value::Depth(1)).unwrap();
    gates.add_value(1, "depth", Value::Depth(9)).unwrap();

    assert!(
        gates
            .satisfies(&1, &Expr::greater("depth", Value::Depth(5)))
            .unwrap()
    );
    assert!(
        gates
            .satisfies(&1, &Expr::less("depth", Value::Depth(5)))
            .unwrap()
    );
    assert!(
        !gates
            .satisfies(&1, &Expr::greater("depth", Value::Depth(9)))
            .unwrap()
    );
}

/// A value type whose `PartialOrd` is **derived**, so every kind reports an
/// order whether it has one or not. This is the case the schema has to catch.
#[derive(Clone, PartialEq, Eq, PartialOrd, Hash, Debug)]
enum Derived {
    Weather(&'static str),
    Depth(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum DerivedKind {
    Weather,
    Depth,
}

impl Kinded for Derived {
    type Kind = DerivedKind;

    fn kind(&self) -> DerivedKind {
        match self {
            Self::Weather(_) => DerivedKind::Weather,
            Self::Depth(_) => DerivedKind::Depth,
        }
    }
}

#[test]
fn a_categorical_property_refuses_comparison_even_with_a_derived_partial_ord() {
    // Before the schema declared it, this passed validation and then compared
    // labels by the order they happened to be written in.
    let mut gates: ConditionIndex<u32, &str, Derived> = ConditionIndex::new();

    gates
        .register_categorical_property("weather", PropertyKind::Single, DerivedKind::Weather)
        .unwrap();
    gates
        .register_ordered_property("depth", PropertyKind::Single, DerivedKind::Depth)
        .unwrap();

    // A comparison on the categorical property is refused, though `Derived`
    // reports an order for it.
    assert!(
        Derived::Weather("rain") < Derived::Weather("storm"),
        "the derived order exists, and means nothing"
    );
    assert_eq!(
        gates.validate(&Expr::greater("weather", Derived::Weather("rain"))),
        Err(SchemaError::NotOrdered {
            property: "weather",
            kind: DerivedKind::Weather,
        })
    );

    // And the same expression on the ordered property is allowed.
    assert!(
        gates
            .validate(&Expr::greater("depth", Derived::Depth(3)))
            .is_ok()
    );

    // So a condition built on a categorical comparison is refused outright.
    assert!(
        gates
            .watch(1, Expr::greater("weather", Derived::Weather("rain")))
            .is_err()
    );
    assert!(!gates.is_watching(&1));

    // The declaration is readable, and a flag has none.
    assert_eq!(
        gates.comparability(&"weather"),
        Some(Comparability::Categorical)
    );
    assert_eq!(gates.comparability(&"depth"), Some(Comparability::Ordered));
    assert_eq!(gates.comparability(&"tended"), None, "unregistered");
}

#[test]
fn a_property_cannot_change_from_ordered_to_categorical() {
    let mut gates: ConditionIndex<u32, &str, Derived> = ConditionIndex::new();

    gates
        .register_ordered_property("depth", PropertyKind::Single, DerivedKind::Depth)
        .unwrap();

    // The same terms again is a repeat.
    assert!(
        gates
            .register_ordered_property("depth", PropertyKind::Single, DerivedKind::Depth)
            .is_ok()
    );

    // Changing its comparability is not, since conditions already built against
    // it were validated under the old declaration.
    assert_eq!(
        gates.register_categorical_property("depth", PropertyKind::Single, DerivedKind::Depth),
        Err(SchemaError::ComparabilityConflict {
            property: "depth",
            current: Comparability::Ordered,
            incoming: Comparability::Categorical,
        })
    );
    assert_eq!(gates.comparability(&"depth"), Some(Comparability::Ordered));

    // The message names both.
    let message = gates
        .register_categorical_property("depth", PropertyKind::Single, DerivedKind::Depth)
        .expect_err("a conflict")
        .to_string();
    assert!(message.contains("Ordered"), "{message}");
    assert!(message.contains("Categorical"), "{message}");
}

#[test]
fn facts_declare_their_comparability_too() {
    let mut gates: ConditionIndex<u32, &str, Derived> = ConditionIndex::new();

    gates
        .register_categorical_fact("weather", DerivedKind::Weather)
        .unwrap();
    gates
        .register_ordered_fact("tide", DerivedKind::Depth)
        .unwrap();

    assert!(
        gates
            .validate(&Expr::greater("weather", Derived::Weather("rain")))
            .is_err(),
        "a categorical fact compares no better than a categorical property"
    );
    assert!(
        gates
            .validate(&Expr::greater("tide", Derived::Depth(2)))
            .is_ok()
    );

    // Equality against a categorical fact is still fine — it is only the order
    // that is refused.
    assert!(
        gates
            .validate(&Expr::is("weather", Derived::Weather("rain")))
            .is_ok()
    );
    assert!(
        gates
            .validate(&Expr::one_of("weather", [Derived::Weather("rain")]))
            .is_ok()
    );
}
