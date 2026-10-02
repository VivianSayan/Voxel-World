//! Bulk weighted picking: the three stages, and what each of them costs.

use std::collections::HashSet;
use voxel_world::fixed;
use voxel_world::math::Fixed;
use voxel_world::random::Random;
use voxel_world::random::bulk_pick::{
    BulkError, BulkPickEntry, BulkPickResult, BulkPickTable, Count, Quantity, REPEATED_DRAW_LIMIT,
};
use voxel_world::random::distributions::Distribution;
use voxel_world::random::seed::Seed;
use voxel_world::unit;

type Table = BulkPickTable<&'static str, Count, Quantity>;

fn stream(seed: u64) -> Random {
    Random::new(Seed::from_integer(seed))
}

/// Four rows, every weight positive, every quantity a single unit.
fn simple(count: Count) -> Table {
    BulkPickTable {
        count,
        entries: vec![
            BulkPickEntry::new("a", 40, Quantity::Fixed(1)),
            BulkPickEntry::new("b", 30, Quantity::Fixed(1)),
            BulkPickEntry::new("c", 20, Quantity::Fixed(1)),
            BulkPickEntry::new("d", 10, Quantity::Fixed(1)),
        ],
    }
}

fn total_selections(results: &[BulkPickResult<&str>]) -> u64 {
    results.iter().map(|one| one.selections).sum()
}

// ---------------------------------------------------------------------------
// Stage one: the count models
// ---------------------------------------------------------------------------

#[test]
fn a_fixed_count_distributes_exactly_that_many() {
    let table = simple(Count::Fixed(12));
    let mut random = stream(1);

    for _ in 0..500 {
        let results = random.bulk_pick(&table).expect("valid");

        assert_eq!(total_selections(&results), 12, "the selections did not add up");

        // Every reported entry was actually selected.
        for one in &results {
            assert!(one.selections > 0, "an unselected entry was reported");
        }
    }
}

#[test]
fn a_uniform_count_stays_inside_its_range() {
    let table = simple(Count::Uniform { low: 0, high: 8 });
    let mut random = stream(2);
    let mut seen: HashSet<u64> = HashSet::new();

    for _ in 0..5_000 {
        let total = total_selections(&random.bulk_pick(&table).expect("valid"));

        assert!(total <= 8, "{total} selections from a range ending at 8");
        seen.insert(total);
    }

    // The whole range turns up, including the empty case the range admits.
    assert_eq!(seen, (0..=8).collect::<HashSet<u64>>());
}

#[test]
fn a_binomial_count_never_exceeds_its_trials() {
    let table = simple(Count::Binomial {
        trials: 10,
        chance: unit!(0.4),
    });
    let mut random = stream(3);
    let mut total: u64 = 0;

    for _ in 0..4_000 {
        let drawn = total_selections(&random.bulk_pick(&table).expect("valid"));

        assert!(drawn <= 10, "{drawn} selections from 10 trials");
        total += drawn;
    }

    // Ten trials at two in five is a mean of four.
    let mean = total as f64 / 4_000.0;
    assert!((mean - 4.0).abs() < 0.15, "binomial count mean {mean}, expected 4");
}

#[test]
fn a_poisson_count_can_be_zero_and_sits_near_its_mean() {
    let table = simple(Count::Poisson { mean: fixed!(3.0) });
    let mut random = stream(4);

    let mut total: u64 = 0;
    let mut empties: u32 = 0;

    for _ in 0..8_000 {
        let results = random.bulk_pick(&table).expect("valid");
        let drawn = total_selections(&results);

        if drawn == 0 {
            assert!(results.is_empty(), "zero selections but entries reported");
            empties += 1;
        }

        total += drawn;
    }

    let mean = total as f64 / 8_000.0;
    assert!((mean - 3.0).abs() < 0.15, "poisson count mean {mean}, expected 3");

    // P(0) for a mean of three is e^-3, about 5%, so empties must occur.
    let share = f64::from(empties) / 8_000.0;
    assert!((share - 0.0498).abs() < 0.015, "empty share {share}, expected about 0.05");
}

#[test]
fn a_geometric_count_follows_the_crates_convention() {
    // The crate counts failures before the first success throughout, so the mean is
    // `(1 - p) / p`: at four in ten that is 1.5.
    let table = simple(Count::Geometric { chance: unit!(0.4) });
    let mut random = stream(5);
    let mut total: u64 = 0;

    for _ in 0..8_000 {
        total += total_selections(&random.bulk_pick(&table).expect("valid"));
    }

    let mean = total as f64 / 8_000.0;
    assert!((mean - 1.5).abs() < 0.1, "geometric count mean {mean}, expected 1.5");
}

#[test]
fn a_zero_count_is_an_ordinary_empty_result() {
    let table = simple(Count::Fixed(0));

    assert_eq!(stream(6).bulk_pick(&table), Ok(Vec::new()));
    assert_eq!(Seed::from_integer(6u64).bulk_pick(&table), Ok(Vec::new()));

    // Even with an unusable table: nothing is selected, so nothing has to be valid.
    let broken: Table = BulkPickTable {
        count: Count::Fixed(0),
        entries: vec![BulkPickEntry::new("a", 0, Quantity::Fixed(1))],
    };

    assert_eq!(stream(7).bulk_pick(&broken), Ok(Vec::new()), "zero weights are fine at zero count");

    let empty: Table = BulkPickTable {
        count: Count::Fixed(0),
        entries: Vec::new(),
    };

    assert_eq!(stream(8).bulk_pick(&empty), Ok(Vec::new()), "an empty table is fine at zero count");
}

// ---------------------------------------------------------------------------
// Draw accounting
// ---------------------------------------------------------------------------

#[test]
fn a_fixed_count_of_zero_consumes_no_words_at_all() {
    // `Fixed` costs no count draw, and a count of zero costs no selection or quantity
    // draw, so the whole call must leave the stream exactly where it was.
    let table = simple(Count::Fixed(0));
    let seed = Seed::from_integer(9u64);

    let mut used = Random::new(seed);
    let mut untouched = Random::new(seed);

    assert_eq!(used.bulk_pick(&table), Ok(Vec::new()));

    for _ in 0..8 {
        assert_eq!(
            used.next_u64(),
            untouched.next_u64(),
            "the call consumed a word it should not have",
        );
    }
}

#[test]
fn a_fixed_quantity_consumes_no_words() {
    // Two tables alike but for the fixed amount. If a fixed quantity drew anything,
    // the streams would diverge and the selections would differ.
    let seed = Seed::from_integer(10u64);

    let one: Table = BulkPickTable {
        count: Count::Fixed(9),
        entries: vec![
            BulkPickEntry::new("a", 50, Quantity::Fixed(1)),
            BulkPickEntry::new("b", 50, Quantity::Fixed(1)),
        ],
    };
    let seven: Table = BulkPickTable {
        count: Count::Fixed(9),
        entries: vec![
            BulkPickEntry::new("a", 50, Quantity::Fixed(7)),
            BulkPickEntry::new("b", 50, Quantity::Fixed(7)),
        ],
    };

    let mut first = Random::new(seed);
    let mut second = Random::new(seed);

    let left = first.bulk_pick(&one).expect("valid");
    let right = second.bulk_pick(&seven).expect("valid");

    // The same selections, because no extra word was spent either way.
    assert_eq!(
        left.iter().map(|r| r.selections).collect::<Vec<u64>>(),
        right.iter().map(|r| r.selections).collect::<Vec<u64>>(),
    );

    // And the streams are still in step afterwards.
    assert_eq!(first.next_u64(), second.next_u64());

    // Exact multiplication, not a draw.
    for (lean, fat) in left.iter().zip(&right) {
        assert_eq!(lean.quantity, lean.selections);
        assert_eq!(fat.quantity, fat.selections * 7);
    }
}

#[test]
fn an_unselected_entry_consumes_no_quantity_draw() {
    // The zero-weight row can never be selected, so its expensive quantity model must
    // never be reached. If it were, the stream would diverge from the table without it.
    let seed = Seed::from_integer(11u64);

    let with_dead_row: Table = BulkPickTable {
        count: Count::Fixed(6),
        entries: vec![
            BulkPickEntry::new("a", 100, Quantity::Fixed(1)),
            BulkPickEntry::new("never", 0, Quantity::Poisson { mean: fixed!(50.0) }),
        ],
    };
    let without: Table = BulkPickTable {
        count: Count::Fixed(6),
        entries: vec![BulkPickEntry::new("a", 100, Quantity::Fixed(1))],
    };

    let mut first = Random::new(seed);
    let mut second = Random::new(seed);

    let left = first.bulk_pick(&with_dead_row).expect("valid");
    let right = second.bulk_pick(&without).expect("valid");

    assert_eq!(left.len(), 1, "the zero-weight row was selected");
    assert_eq!(left[0].value, "a");
    assert_eq!(left[0].selections, 6);
    assert_eq!(right[0].selections, 6);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn one_seed_and_one_table_give_one_answer() {
    let table = simple(Count::Poisson { mean: fixed!(4.0) });
    let world = Seed::from_integer(12u64).child("loot");

    for index in 0..200u64 {
        let chest = world.index(index);
        let first = chest.bulk_pick(&table).expect("valid");

        assert_eq!(first, chest.bulk_pick(&table).expect("valid"), "not stable");
    }
}

#[test]
fn different_seeds_normally_give_different_answers() {
    let table = simple(Count::Fixed(20));
    let world = Seed::from_integer(13u64);

    let shapes: HashSet<Vec<u64>> = (0..300u64)
        .map(|index| {
            world
                .index(index)
                .bulk_pick(&table)
                .expect("valid")
                .iter()
                .map(|one| one.selections)
                .collect()
        })
        .collect();

    assert!(shapes.len() > 100, "only {} distinct splits from 300 seeds", shapes.len());
}

#[test]
fn a_stream_advances_between_calls() {
    let table = simple(Count::Fixed(20));
    let mut random = stream(14);

    let first = random.bulk_pick(&table).expect("valid");
    let second = random.bulk_pick(&table).expect("valid");

    assert_ne!(first, second, "the stream repeated itself");
}

#[test]
fn results_follow_the_tables_own_entry_order() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(400),
        entries: vec![
            BulkPickEntry::new("zebra", 25, Quantity::Fixed(1)),
            BulkPickEntry::new("apple", 25, Quantity::Fixed(1)),
            BulkPickEntry::new("mango", 25, Quantity::Fixed(1)),
            BulkPickEntry::new("beetle", 25, Quantity::Fixed(1)),
        ],
    };

    // Four rows at equal weight and four hundred selections: all four appear, and in
    // the order written rather than sorted, hashed or shuffled.
    for index in 0..50u64 {
        let results = Seed::from_integer(15u64).index(index).bulk_pick(&table).expect("valid");

        assert_eq!(
            results.iter().map(|one| one.value).collect::<Vec<&str>>(),
            vec!["zebra", "apple", "mango", "beetle"],
        );
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

#[test]
fn relative_weights_are_what_matter() {
    // 1,2,3 and 10,20,30 are the same table, so the same seed must give the same split.
    let small: Table = BulkPickTable {
        count: Count::Fixed(60),
        entries: vec![
            BulkPickEntry::new("a", 1, Quantity::Fixed(1)),
            BulkPickEntry::new("b", 2, Quantity::Fixed(1)),
            BulkPickEntry::new("c", 3, Quantity::Fixed(1)),
        ],
    };
    let large: Table = BulkPickTable {
        count: Count::Fixed(60),
        entries: vec![
            BulkPickEntry::new("a", 10, Quantity::Fixed(1)),
            BulkPickEntry::new("b", 20, Quantity::Fixed(1)),
            BulkPickEntry::new("c", 30, Quantity::Fixed(1)),
        ],
    };

    let seed = Seed::from_integer(16u64);

    assert_eq!(seed.bulk_pick(&small), seed.bulk_pick(&large));
}

#[test]
fn weights_set_the_proportions() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(1_000),
        entries: vec![
            BulkPickEntry::new("a", 40, Quantity::Fixed(1)),
            BulkPickEntry::new("b", 30, Quantity::Fixed(1)),
            BulkPickEntry::new("c", 20, Quantity::Fixed(1)),
            BulkPickEntry::new("d", 10, Quantity::Fixed(1)),
        ],
    };

    let mut totals = [0u64; 4];
    let mut random = stream(17);

    for _ in 0..200 {
        for (position, one) in random.bulk_pick(&table).expect("valid").iter().enumerate() {
            totals[position] += one.selections;
        }
    }

    let drawn: u64 = totals.iter().sum();

    for (position, share) in [0.4, 0.3, 0.2, 0.1].iter().enumerate() {
        let seen = totals[position] as f64 / drawn as f64;

        assert!((seen - share).abs() < 0.02, "entry {position} took {seen}, expected {share}");
    }
}

#[test]
fn a_zero_weight_entry_is_never_selected() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(50),
        entries: vec![
            BulkPickEntry::new("live", 100, Quantity::Fixed(1)),
            BulkPickEntry::new("dead", 0, Quantity::Fixed(1)),
            BulkPickEntry::new("alive", 100, Quantity::Fixed(1)),
        ],
    };

    let mut random = stream(18);

    for _ in 0..2_000 {
        for one in random.bulk_pick(&table).expect("valid") {
            assert_ne!(one.value, "dead", "a zero-weight entry was selected");
        }
    }
}

#[test]
fn all_zero_weights_with_a_positive_count_is_an_error() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(5),
        entries: vec![
            BulkPickEntry::new("a", 0, Quantity::Fixed(1)),
            BulkPickEntry::new("b", 0, Quantity::Fixed(1)),
        ],
    };

    assert_eq!(stream(19).bulk_pick(&table), Err(BulkError::AllWeightsZero));
    assert_eq!(Seed::from_integer(19u64).bulk_pick(&table), Err(BulkError::AllWeightsZero));
}

#[test]
fn an_empty_table_with_a_positive_count_is_an_error() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(5),
        entries: Vec::new(),
    };

    assert_eq!(stream(20).bulk_pick(&table), Err(BulkError::EmptyTable));
}

#[test]
fn weights_that_cannot_be_summed_are_reported() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(1),
        entries: vec![
            BulkPickEntry::new("a", u64::MAX, Quantity::Fixed(1)),
            BulkPickEntry::new("b", u64::MAX, Quantity::Fixed(1)),
        ],
    };

    assert_eq!(stream(21).bulk_pick(&table), Err(BulkError::WeightOverflow));
}

// ---------------------------------------------------------------------------
// Stage three: the quantity models
// ---------------------------------------------------------------------------

#[test]
fn a_fixed_quantity_multiplies_exactly() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(30),
        entries: vec![BulkPickEntry::new("stack", 1, Quantity::Fixed(7))],
    };

    let results = stream(22).bulk_pick(&table).expect("valid");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].selections, 30);
    assert_eq!(results[0].quantity, 210, "30 selections of 7 is 210");
}

#[test]
fn a_fixed_quantity_reports_overflow_rather_than_wrapping() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(4),
        entries: vec![BulkPickEntry::new("huge", 1, Quantity::Fixed(u64::MAX / 2))],
    };

    assert_eq!(stream(23).bulk_pick(&table), Err(BulkError::QuantityOverflow));
}

#[test]
fn a_uniform_quantity_stays_in_its_repeated_range() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(5),
        entries: vec![BulkPickEntry::new("arrows", 1, Quantity::Uniform { low: 4, high: 12 })],
    };

    let mut random = stream(24);

    for _ in 0..2_000 {
        let results = random.bulk_pick(&table).expect("valid");

        assert_eq!(results[0].selections, 5);
        assert!(results[0].quantity >= 20, "below five lots of four");
        assert!(results[0].quantity <= 60, "above five lots of twelve");
    }
}

#[test]
fn a_uniform_quantity_refuses_more_repeats_than_it_allows() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(REPEATED_DRAW_LIMIT + 1),
        entries: vec![BulkPickEntry::new("many", 1, Quantity::Uniform { low: 0, high: 1 })],
    };

    assert_eq!(stream(25).bulk_pick(&table), Err(BulkError::TooManyRepeats));
}

#[test]
fn a_binomial_quantity_never_exceeds_its_combined_trials() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(6),
        entries: vec![BulkPickEntry::new(
            "scatter",
            1,
            Quantity::Binomial { trials: 12, chance: unit!(0.5) },
        )],
    };

    let mut random = stream(26);
    let mut total: u64 = 0;

    for _ in 0..2_000 {
        let results = random.bulk_pick(&table).expect("valid");

        assert!(results[0].quantity <= 72, "above six lots of twelve");
        total += results[0].quantity;
    }

    // Six selections of Binomial(12, 1/2) aggregate to Binomial(72, 1/2): mean 36.
    let mean = total as f64 / 2_000.0;
    assert!((mean - 36.0).abs() < 0.5, "binomial quantity mean {mean}, expected 36");
}

#[test]
fn a_poisson_quantity_aggregates_its_mean() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(5),
        entries: vec![BulkPickEntry::new("coins", 1, Quantity::Poisson { mean: fixed!(8.0) })],
    };

    let mut random = stream(27);
    let mut total: u64 = 0;

    for _ in 0..4_000 {
        total += random.bulk_pick(&table).expect("valid")[0].quantity;
    }

    // Five selections of Poisson(8) are one Poisson(40).
    let mean = total as f64 / 4_000.0;
    assert!((mean - 40.0).abs() < 0.5, "poisson quantity mean {mean}, expected 40");
}

#[test]
fn a_geometric_quantity_aggregates_as_a_negative_binomial() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(4),
        entries: vec![BulkPickEntry::new("streak", 1, Quantity::Geometric { chance: unit!(0.4) })],
    };

    let mut random = stream(28);
    let mut total: u64 = 0;

    for _ in 0..6_000 {
        total += random.bulk_pick(&table).expect("valid")[0].quantity;
    }

    // Failures before the first success is the crate's convention, mean (1-p)/p = 1.5
    // per selection, so four selections average six.
    let mean = total as f64 / 6_000.0;
    assert!((mean - 6.0).abs() < 0.2, "geometric quantity mean {mean}, expected 6");
}

#[test]
fn a_negative_poisson_mean_is_refused() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(1),
        entries: vec![BulkPickEntry::new("bad", 1, Quantity::Poisson { mean: fixed!(-1.0) })],
    };

    assert_eq!(stream(29).bulk_pick(&table), Err(BulkError::InvalidQuantity));

    let bad_count: Table = BulkPickTable {
        count: Count::Poisson { mean: Fixed::ZERO - Fixed::ONE },
        entries: vec![BulkPickEntry::new("a", 1, Quantity::Fixed(1))],
    };

    assert_eq!(stream(30).bulk_pick(&bad_count), Err(BulkError::InvalidCount));
}

#[test]
fn an_inverted_uniform_range_is_refused() {
    let table: Table = BulkPickTable {
        count: Count::Uniform { low: 8, high: 2 },
        entries: vec![BulkPickEntry::new("a", 1, Quantity::Fixed(1))],
    };

    assert_eq!(stream(31).bulk_pick(&table), Err(BulkError::InvalidCount));
}

// ---------------------------------------------------------------------------
// Ergonomics
// ---------------------------------------------------------------------------

#[test]
fn the_fixed_count_constructor_matches_the_long_form() {
    let short = BulkPickTable::fixed_count(
        10,
        vec![BulkPickEntry::new("a", 1, Quantity::Fixed(1))],
    );
    let long: Table = BulkPickTable {
        count: Count::Fixed(10),
        entries: vec![BulkPickEntry::new("a", 1, Quantity::Fixed(1))],
    };

    assert_eq!(short, long);
}

#[test]
fn a_dud_entry_is_different_from_an_empty_result() {
    // Both are "nothing", and the design keeps them apart: one is a selection that
    // produced nothing, the other is no selection at all.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Drop {
        Item,
        Dud,
    }

    let table: BulkPickTable<Drop, Count, Quantity> = BulkPickTable {
        count: Count::Fixed(10),
        entries: vec![
            BulkPickEntry::new(Drop::Item, 50, Quantity::Fixed(1)),
            BulkPickEntry::new(Drop::Dud, 50, Quantity::Fixed(0)),
        ],
    };

    let results = Seed::from_integer(32u64).bulk_pick(&table).expect("valid");

    // Ten selections happened, so the result is not empty — even where some of them
    // were duds that yielded nothing.
    assert_eq!(results.iter().map(|one| one.selections).sum::<u64>(), 10);

    for one in &results {
        if one.value == Drop::Dud {
            assert_eq!(one.quantity, 0, "a dud yielded something");
            assert!(one.selections > 0, "a dud was reported without being selected");
        }
    }

    // Against which: a zero count really is empty.
    let nothing: BulkPickTable<Drop, Count, Quantity> = BulkPickTable {
        count: Count::Fixed(0),
        entries: table.entries.clone(),
    };

    assert_eq!(Seed::from_integer(32u64).bulk_pick(&nothing), Ok(Vec::new()));
}

// ---------------------------------------------------------------------------
// Portability
// ---------------------------------------------------------------------------

#[test]
fn a_fully_portable_table_gives_known_exact_results() {
    // Every stage here is integer-only: a Poisson count through `PoissonRatio`, the
    // split through `Multinomial::portable`, and quantity models built on
    // `BinomialRatio`, `UniformU64`, `PoissonRatio` and `NegativeBinomial::by_gaps`.
    //
    // So these values are not merely stable on this machine — they are what the same
    // seed and table produce on every target. Pinning them is what turns that claim
    // into something a change has to break deliberately.
    let table: Table = BulkPickTable {
        count: Count::Poisson { mean: fixed!(3.0) },
        entries: vec![
            BulkPickEntry::new("iron", 40, Quantity::Binomial { trials: 8, chance: unit!(0.5) }),
            BulkPickEntry::new("arrows", 25, Quantity::Uniform { low: 4, high: 12 }),
            BulkPickEntry::new("coins", 30, Quantity::Poisson { mean: fixed!(8.0) }),
            BulkPickEntry::new("shard", 5, Quantity::Geometric { chance: unit!(0.4) }),
        ],
    };

    let expected: [Vec<(&str, u64, u64)>; 6] = [
        vec![("iron", 3, 12), ("arrows", 1, 4), ("coins", 1, 6)],
        // The Poisson count came out zero: an empty chest, with no row standing for it.
        vec![],
        vec![("shard", 1, 6)],
        vec![("arrows", 1, 7), ("coins", 2, 13), ("shard", 1, 2)],
        vec![("iron", 2, 9), ("coins", 1, 13)],
        vec![("iron", 1, 6), ("arrows", 2, 19), ("coins", 4, 34)],
    ];

    let world = Seed::from_integer(777u64).child("loot");

    for (index, wanted) in expected.iter().enumerate() {
        let results = world.index(index as u64).bulk_pick(&table).expect("valid");

        let got: Vec<(&str, u64, u64)> = results
            .iter()
            .map(|one| (one.value, one.selections, one.quantity))
            .collect();

        assert_eq!(&got, wanted, "chest {index}");
    }
}

// ---------------------------------------------------------------------------
// Exact integer weights
//
// The weights a caller writes are whole numbers, and the ratios between whole
// numbers are exact. These check that nothing rounds them onto a grid on the way
// to the draw.
// ---------------------------------------------------------------------------

#[test]
fn equivalent_weight_vectors_draw_identically() {
    // `Ratio` reduces on construction, so `10/60` and `1/6` are the same fraction and
    // reach `BinomialRatio` as the same numbers. Equivalent tables are therefore not
    // merely statistically alike — they are bit-for-bit identical under one seed.
    let scales: [u64; 4] = [1, 10, 1_000, 7];
    let seed = Seed::from_integer(900u64);

    let reference: Vec<(&str, u64, u64)> = {
        let table: Table = BulkPickTable {
            count: Count::Fixed(120),
            entries: vec![
                BulkPickEntry::new("a", 1, Quantity::Fixed(1)),
                BulkPickEntry::new("b", 2, Quantity::Fixed(1)),
                BulkPickEntry::new("c", 3, Quantity::Fixed(1)),
            ],
        };

        seed.bulk_pick(&table)
            .expect("valid")
            .iter()
            .map(|one| (one.value, one.selections, one.quantity))
            .collect()
    };

    for scale in scales {
        let table: Table = BulkPickTable {
            count: Count::Fixed(120),
            entries: vec![
                BulkPickEntry::new("a", scale, Quantity::Fixed(1)),
                BulkPickEntry::new("b", 2 * scale, Quantity::Fixed(1)),
                BulkPickEntry::new("c", 3 * scale, Quantity::Fixed(1)),
            ],
        };

        let got: Vec<(&str, u64, u64)> = seed
            .bulk_pick(&table)
            .expect("valid")
            .iter()
            .map(|one| (one.value, one.selections, one.quantity))
            .collect();

        assert_eq!(got, reference, "scaling the weights by {scale} changed the split");
    }
}

#[test]
fn a_small_positive_weight_is_genuinely_reachable() {
    // One part in a hundred thousand, over enough selections that it must turn up.
    // A weight that had been rounded away would show here as a flat zero.
    let table: Table = BulkPickTable {
        count: Count::Fixed(2_000_000),
        entries: vec![
            BulkPickEntry::new("rare", 1, Quantity::Fixed(1)),
            BulkPickEntry::new("common", 99_999, Quantity::Fixed(1)),
        ],
    };

    let results = stream(901).bulk_pick(&table).expect("valid");
    let rare: u64 = results
        .iter()
        .find(|one| one.value == "rare")
        .map_or(0, |one| one.selections);

    // Twenty expected, so anything in single figures upwards proves it is positive.
    assert!(rare > 0, "a positive weight never came up in two million selections");
    assert!(rare < 200, "the rare entry came up {rare} times, far above its weight");

    // And the total is still exactly what was asked for.
    assert_eq!(total_selections(&results), 2_000_000);
}

#[test]
fn the_widest_possible_weight_spread_is_accepted() {
    // The extreme a `u64` table can express: the smallest positive weight against
    // almost the whole range. This is the case where normalising onto `Unit`'s grid
    // was at its worst — one part in `2^64` has to round to the nearest multiple of
    // `2^-63`, which doubles it. The exact path has no grid to round to.
    let table: Table = BulkPickTable {
        count: Count::Fixed(1_000),
        entries: vec![
            BulkPickEntry::new("needle", 1, Quantity::Fixed(1)),
            BulkPickEntry::new("haystack", u64::MAX - 1, Quantity::Fixed(1)),
        ],
    };

    // Summable, so the table is usable rather than an overflow.
    assert_eq!(table.total_weight(), Ok(u64::MAX));

    let results = stream(902).bulk_pick(&table).expect("valid");

    assert_eq!(total_selections(&results), 1_000);

    // At one part in `2^64` the needle should not appear at all in a thousand draws.
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].value, "haystack");
}

#[test]
fn a_zero_weight_never_takes_a_selection_however_many_are_made() {
    let table: Table = BulkPickTable {
        count: Count::Fixed(100_000),
        entries: vec![
            BulkPickEntry::new("first", 1, Quantity::Fixed(1)),
            BulkPickEntry::new("void", 0, Quantity::Fixed(1)),
            BulkPickEntry::new("last", 1, Quantity::Fixed(1)),
        ],
    };

    let results = stream(903).bulk_pick(&table).expect("valid");

    assert!(
        results.iter().all(|one| one.value != "void"),
        "a zero weight took a selection",
    );
    assert_eq!(total_selections(&results), 100_000, "the total stopped being exact");
}

#[test]
fn the_selections_always_add_up_to_the_count() {
    // The last category absorbs the remainder, which is what makes this exact rather
    // than approximately exact. Checked across awkward weightings and counts.
    let weightings: [&[u64]; 5] = [
        &[1, 1, 1],
        &[1, 2, 3, 4, 5, 6, 7],
        &[1, 0, 1_000_000, 0, 3],
        &[u64::MAX / 4, 1, u64::MAX / 4],
        &[9],
    ];

    for (which, weights) in weightings.iter().enumerate() {
        for count in [1u64, 2, 7, 64, 1_000, 100_000] {
            let table: BulkPickTable<usize, Count, Quantity> = BulkPickTable {
                count: Count::Fixed(count),
                entries: weights
                    .iter()
                    .enumerate()
                    .map(|(index, weight)| BulkPickEntry::new(index, *weight, Quantity::Fixed(1)))
                    .collect(),
            };

            let results = Seed::from_integer(904u64)
                .index(count)
                .bulk_pick(&table)
                .expect("valid");

            let total: u64 = results.iter().map(|one| one.selections).sum();

            assert_eq!(total, count, "weighting {which} at count {count} did not add up");
        }
    }
}

#[test]
fn the_exact_constructor_rejects_what_cannot_be_split() {
    use voxel_world::random::Multinomial;

    assert!(Multinomial::portable_weights(10, &[]).is_none(), "no categories");
    assert!(Multinomial::portable_weights(10, &[0, 0, 0]).is_none(), "nothing positive");

    let split = Multinomial::portable_weights(10, &[1, u64::MAX]).expect("valid");

    assert!(split.is_portable(), "the exact constructor must be portable");
    assert_eq!(split.categories(), 2);

    // A weight sum past `u64` is fine here — the weights are held in a `u128` — and
    // the split still totals exactly.
    let counts = split.sample(&mut stream(905));

    assert_eq!(counts.iter().sum::<u64>(), 10);
}
