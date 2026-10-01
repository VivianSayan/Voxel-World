//! Bulk stochastic sampling: invariants, boundaries, and algorithm selection.
//!
//! Kept in two halves, deliberately:
//!
//! - **Exact correctness** — invariants that must hold for every draw. Counts in
//!   range, indices valid, totals exact, no duplicates. These would catch a real
//!   bug and cannot be flaky.
//! - **Statistical sanity** — large samples with generous tolerances, marked as
//!   such. A small sample asserting a particular frequency would be flaky, so
//!   there are none of those.

use voxel_world::random::approximation::{
    Approximation, BinomialAlgorithm, HypergeometricAlgorithm, PoissonAlgorithm,
    binomial_odds_drift_for_hypergeometric, normal_error_for_binomial, normal_error_for_poisson,
    poisson_error_for_binomial,
};
use voxel_world::random::Random;
use voxel_world::random::seed::Seed;
use voxel_world::units::{Probability, Rate, Ratio, Unit};

fn generator(seed: u64) -> Random {
    Random::new(Seed::from_integer(seed))
}

// ===========================================================================
// Success indices and masks
// ===========================================================================

#[test]
fn success_indices_are_valid_and_ascending() {
    let mut random = generator(1);

    for chance in [Unit::ZERO, Unit::one_in(1_000), Unit::percent(5).unwrap(), Unit::HALF] {
        for count in [0u64, 1, 2, 100, 10_000] {
            let found: Vec<u64> = random.success_indices(count, chance);

            assert!(
                found.iter().all(|index| *index < count),
                "an index escaped the range at chance {chance}"
            );
            assert!(
                found.windows(2).all(|pair| pair[0] < pair[1]),
                "indices must be strictly ascending"
            );
            assert!(found.len() as u64 <= count, "more successes than positions");
        }
    }
}

#[test]
fn success_indices_handle_both_certainties() {
    let mut random = generator(2);

    // Never: nothing at all, however many positions.
    assert!(random.success_indices(10_000, Unit::ZERO).is_empty());

    // Always: every position, and no infinite loop looking for gaps of zero.
    let all: Vec<u64> = random.success_indices(1_000, Unit::ONE);
    assert_eq!(all.len(), 1_000);
    assert_eq!(all, (0..1_000).collect::<Vec<u64>>());

    // No positions at all.
    assert!(random.success_indices(0, Unit::HALF).is_empty());
    assert!(random.success_indices(0, Unit::ONE).is_empty());
}

#[test]
fn a_success_mask_is_sixty_four_coins() {
    let mut random = generator(3);

    assert_eq!(random.success_mask(Ratio::new(0, 1).unwrap()), 0, "never");
    assert_eq!(
        random.success_mask(Ratio::new(1, 1).unwrap()),
        u64::MAX,
        "always"
    );

    // And it agrees with the older name it replaces.
    let mut left = generator(99);
    let mut right = generator(99);
    let chance = Ratio::one_in(4).unwrap();
    assert_eq!(left.success_mask(chance), right.chance_mask(chance));
}

// ===========================================================================
// Multinomial
// ===========================================================================

#[test]
fn multinomial_counts_sum_to_the_total() {
    let mut random = generator(4);

    let shares = [
        Unit::percent(40).unwrap(),
        Unit::percent(30).unwrap(),
        Unit::percent(20).unwrap(),
        Unit::percent(10).unwrap(),
    ];

    for count in [0u64, 1, 2, 7, 1_000, 100_000] {
        let counts: Vec<u64> = random.multinomial(count, &shares).expect("valid shares");

        assert_eq!(counts.len(), 4, "one count per category");
        assert_eq!(
            counts.iter().sum::<u64>(),
            count,
            "every item must be placed exactly once"
        );
    }
}

#[test]
fn multinomial_handles_degenerate_shares() {
    let mut random = generator(5);

    // A single category takes everything.
    let counts = random.multinomial(500, &[Unit::ONE]).expect("valid");
    assert_eq!(counts, vec![500]);

    // A zero share never receives anything.
    let counts = random
        .multinomial(1_000, &[Unit::ZERO, Unit::ONE, Unit::ZERO])
        .expect("valid");
    assert_eq!(counts[0], 0, "a zero share takes nothing");
    assert_eq!(counts[2], 0, "even as the last category");
    assert_eq!(counts[1], 1_000);
    assert_eq!(counts.iter().sum::<u64>(), 1_000);

    // Weights need not sum to one: they are normalised.
    let counts = random
        .multinomial(600, &[Unit::ONE, Unit::ONE, Unit::ONE])
        .expect("valid");
    assert_eq!(counts.iter().sum::<u64>(), 600);

    // Rejected: nothing to distribute among.
    assert_eq!(random.multinomial(10, &[]), None, "no categories");
    assert_eq!(
        random.multinomial(10, &[Unit::ZERO, Unit::ZERO]),
        None,
        "no weight anywhere"
    );
}

// ===========================================================================
// Hypergeometric
// ===========================================================================

#[test]
fn hypergeometric_respects_the_population() {
    let mut random = generator(6);

    for (population, successes, draws) in [
        (100u64, 30u64, 10u64),
        (10, 10, 10),
        (10, 0, 10),
        (1, 1, 1),
        (1000, 1, 999),
        (50, 25, 50),
    ] {
        let found: u64 = random
            .hypergeometric(population, successes, draws)
            .expect("valid");

        assert!(found <= draws, "cannot find more than were drawn");
        assert!(found <= successes, "cannot find more than exist");
        // Nor fewer than the draw forces: if the failures run out, the rest are hits.
        let failures: u64 = population - successes;
        assert!(found >= draws.saturating_sub(failures), "pigeonhole");
    }
}

#[test]
fn hypergeometric_boundaries_are_exact() {
    let mut random = generator(7);

    // Every item is a success, so every draw is one.
    assert_eq!(random.hypergeometric(20, 20, 7), Some(7));
    // None are, so none are found.
    assert_eq!(random.hypergeometric(20, 0, 7), Some(0));
    // Drawing the whole population finds all the successes.
    assert_eq!(random.hypergeometric(20, 8, 20), Some(8));
    // Drawing more than exists takes the population and no more.
    assert_eq!(random.hypergeometric(20, 8, 1_000), Some(8));
    // No draws, no successes.
    assert_eq!(random.hypergeometric(20, 8, 0), Some(0));

    // Rejected: more successes than population is not a population.
    assert_eq!(random.hypergeometric(10, 11, 5), None);
}

// ===========================================================================
// Negative binomial
// ===========================================================================

#[test]
fn negative_binomial_counts_failures_only() {
    let mut random = generator(8);

    // Zero successes wanted: nothing to wait for.
    assert_eq!(random.negative_binomial(0, Unit::HALF), 0);
    // Certain success: no failures at all, however many are wanted.
    assert_eq!(random.negative_binomial(100, Unit::ONE), 0);
    // Impossible: saturates rather than looping for ever.
    assert_eq!(random.negative_binomial(1, Unit::ZERO), u64::MAX);

    // One success is exactly the geometric, which is the documented convention.
    let mut left = generator(555);
    let mut right = generator(555);
    assert_eq!(
        left.negative_binomial(1, Unit::one_in(8)),
        right.geometric(Unit::one_in(8).to_probability()),
        "R = 1 is the geometric"
    );
}

// ===========================================================================
// Poisson binomial
// ===========================================================================

#[test]
fn poisson_binomial_stays_within_its_trials() {
    let mut random = generator(9);

    assert_eq!(random.poisson_binomial(&[]), 0, "no trials, no successes");
    assert_eq!(
        random.poisson_binomial(&[Unit::ONE, Unit::ONE, Unit::ONE]),
        3,
        "all certain"
    );
    assert_eq!(
        random.poisson_binomial(&[Unit::ZERO, Unit::ZERO]),
        0,
        "all impossible"
    );

    // Mixed certainty: the certain ones always land, the impossible never do.
    for _ in 0..200 {
        let count = random.poisson_binomial(&[Unit::ONE, Unit::ZERO, Unit::HALF]);
        assert!((1..=2).contains(&count), "got {count}");
    }
}

// ===========================================================================
// Selection without replacement
// ===========================================================================

#[test]
fn weighted_choice_never_repeats_an_item() {
    let mut random = generator(10);
    let weights = [10.0, 5.0, 1.0, 0.5, 3.0, 7.0];

    for count in 0..=weights.len() {
        let chosen: Vec<usize> = random.choose_weighted_n(&weights, count);

        assert_eq!(chosen.len(), count, "asked for {count}");

        let mut sorted = chosen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), chosen.len(), "duplicates in {chosen:?}");
        assert!(chosen.iter().all(|index| *index < weights.len()));
    }

    // Asking for more than exists gives everything, once each.
    let all = random.choose_weighted_n(&weights, 100);
    assert_eq!(all.len(), weights.len());

    // Unusable weights are never chosen.
    let sparse = [0.0, 5.0, -1.0, f64::NAN, 2.0];
    let chosen = random.choose_weighted_n(&sparse, 5);
    assert_eq!(chosen.len(), 2, "only the two usable weights");
    assert!(chosen.iter().all(|index| *index == 1 || *index == 4));
}

#[test]
fn reservoir_sampling_keeps_at_most_what_was_asked() {
    let mut random = generator(11);

    for (length, count) in [(0usize, 5usize), (3, 5), (5, 5), (1_000, 5), (1_000, 0)] {
        let kept: Vec<usize> = random.reservoir_sample(0..length, count);

        assert_eq!(kept.len(), count.min(length), "length {length}, want {count}");
        assert!(kept.iter().all(|item| *item < length), "outside the stream");

        let mut sorted = kept.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), kept.len(), "an item was kept twice");
    }

    // The point of it: a stream that is never materialised.
    let kept: Vec<u64> = random.reservoir_sample((0..10_000_000u64).filter(|n| n % 3 == 0), 4);
    assert_eq!(kept.len(), 4);
    assert!(kept.iter().all(|n| n % 3 == 0));
}

#[test]
fn shuffling_keeps_every_item() {
    let mut random = generator(12);

    for length in [0usize, 1, 2, 10, 1_000] {
        let mut items: Vec<usize> = (0..length).collect();
        random.shuffle(&mut items);

        items.sort_unstable();
        assert_eq!(items, (0..length).collect::<Vec<usize>>(), "an item was lost");
    }

    // A partial shuffle fills the prefix and keeps the multiset whole.
    for (length, count) in [(10usize, 3usize), (10, 10), (10, 20), (0, 3)] {
        let mut items: Vec<usize> = (0..length).collect();
        let taken: usize = random.partial_shuffle(&mut items, count);

        assert_eq!(taken, count.min(length));

        let prefix: Vec<usize> = items[..taken].to_vec();
        let mut unique = prefix.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), prefix.len(), "the prefix repeated an item");

        items.sort_unstable();
        assert_eq!(items, (0..length).collect::<Vec<usize>>(), "an item was lost");
    }
}

// ===========================================================================
// Approximation policy
// ===========================================================================

#[test]
fn exact_mode_never_approximates() {
    let mut random = generator(13);

    // Parameters where every approximation would otherwise be permitted.
    for (trials, chance) in [
        (10_000_000u64, Unit::one_in(10_000_000)),
        (1_000_000, Unit::percent(50).unwrap()),
        (100_000, Unit::percent(1).unwrap()),
    ] {
        let (_, algorithm) = random.binomial_with_info(trials, chance, Approximation::Exact);

        assert_eq!(
            algorithm,
            BinomialAlgorithm::Exact,
            "Exact must never approximate"
        );
    }

    let (_, algorithm) = random.poisson_with_info(Rate::new(1e9).unwrap(), Approximation::Exact);
    assert_eq!(algorithm, PoissonAlgorithm::Exact);

    let (_, algorithm) = random
        .hypergeometric_with_info(1_000_000_000, 500_000_000, 2, Approximation::Exact)
        .expect("valid");
    assert_eq!(algorithm, HypergeometricAlgorithm::Exact);

    // A zero error budget is the same as Exact, since no approximation proves zero.
    let (_, algorithm) = random.binomial_with_info(
        1_000_000,
        Unit::one_in(1_000_000),
        Approximation::AllowError(Unit::ZERO),
    );
    assert_eq!(algorithm, BinomialAlgorithm::Exact);
}

#[test]
fn the_error_bounds_are_the_ones_the_documentation_names() {
    // Le Cam: n * p^2.
    assert!((poisson_error_for_binomial(1_000, 0.001) - 0.001).abs() < 1e-12);
    assert!((poisson_error_for_binomial(100, 0.5) - 25.0).abs() < 1e-12);

    // Berry-Esseen for a binomial: 0.4748 (p^2 + q^2) / sqrt(n p q).
    let bound: f64 = normal_error_for_binomial(10_000, 0.5);
    let expected: f64 = 0.4748 * 0.5 / (10_000.0f64 * 0.25).sqrt();
    assert!((bound - expected).abs() < 1e-12);

    // A point mass has no normal to approximate it.
    assert!(normal_error_for_binomial(100, 0.0).is_infinite());
    assert!(normal_error_for_binomial(100, 1.0).is_infinite());

    // Berry-Esseen for a Poisson: 0.4748 / sqrt(lambda).
    assert!((normal_error_for_poisson(10_000.0) - 0.004_748).abs() < 1e-12);
    assert!(normal_error_for_poisson(0.0).is_infinite());

    // Hypergeometric drift: the familiar five-percent advice, as a number.
    let drift: f64 = binomial_odds_drift_for_hypergeometric(1_000, 50);
    assert!((drift - 50.0 / 950.0).abs() < 1e-12);
    assert!(drift > 0.05 && drift < 0.06, "about five percent");
    assert!(binomial_odds_drift_for_hypergeometric(10, 10).is_infinite());
}

#[test]
fn the_policy_permits_exactly_what_it_says() {
    assert_eq!(Approximation::Exact.tolerance(), None);
    assert_eq!(Approximation::Auto.tolerance(), Some(Unit::one_in(10_000)));
    assert_eq!(Approximation::Fast.tolerance(), Some(Unit::one_in(100)));
    assert_eq!(
        Approximation::AllowError(Unit::ZERO).tolerance(),
        None,
        "a zero budget is Exact"
    );

    // On the boundary, inclusive.
    let budget = Approximation::AllowError(Unit::one_in(1_000));
    assert!(budget.permits(0.001), "exactly at the budget is allowed");
    assert!(!budget.permits(0.002));
    assert!(budget.permits(0.0));

    // A bound that is not a proof of anything is never allowed.
    assert!(!Approximation::Fast.permits(f64::INFINITY));
    assert!(!Approximation::Fast.permits(f64::NAN));
    assert!(!Approximation::Fast.permits(-1.0));
    assert!(!Approximation::Exact.permits(0.0), "Exact permits nothing");
}

#[test]
fn approximations_are_chosen_at_the_boundaries_the_bounds_set() {
    let mut random = generator(14);

    // Many trials, tiny chance: Le Cam is small, so the Poisson swap is allowed.
    let (_, algorithm) =
        random.binomial_with_info(1_000_000, Unit::one_in(10_000_000), Approximation::Auto);
    assert_eq!(algorithm, BinomialAlgorithm::Poisson);

    // A fair coin: Le Cam is hopeless (it is n/4), so only the normal swap is ever
    // available. Berry-Esseen puts the boundary for `Auto` at about 22.5 million
    // trials — `0.4748 * 0.5 / sqrt(0.25n) <= 1e-4`. Bracketing it checks the rule
    // rather than one lucky value.
    let (_, below) = random.binomial_with_info(20_000_000, Unit::HALF, Approximation::Auto);
    let (_, above) = random.binomial_with_info(30_000_000, Unit::HALF, Approximation::Auto);

    assert_eq!(below, BinomialAlgorithm::Exact, "under the bound, stay exact");
    assert_eq!(above, BinomialAlgorithm::Normal, "over it, approximate");

    // And the bound itself is what decided: confirm directly.
    assert!(normal_error_for_binomial(20_000_000, 0.5) > 1e-4);
    assert!(normal_error_for_binomial(30_000_000, 0.5) < 1e-4);

    // A looser policy takes the same parameters the tighter one refused.
    let (_, loose) = random.binomial_with_info(20_000_000, Unit::HALF, Approximation::Fast);
    assert_eq!(loose, BinomialAlgorithm::Normal);

    // Few trials: neither bound is good enough, so it stays exact.
    let (_, algorithm) = random.binomial_with_info(10, Unit::HALF, Approximation::Auto);
    assert_eq!(algorithm, BinomialAlgorithm::Exact);

    // Poisson: a large mean permits the normal, a small one does not.
    let (_, algorithm) = random.poisson_with_info(Rate::new(1e9).unwrap(), Approximation::Auto);
    assert_eq!(algorithm, PoissonAlgorithm::Normal);
    let (_, algorithm) = random.poisson_with_info(Rate::new(4.2).unwrap(), Approximation::Auto);
    assert_eq!(algorithm, PoissonAlgorithm::Exact);

    // Hypergeometric: a small bite of a big population permits the binomial.
    let (_, algorithm) = random
        .hypergeometric_with_info(1_000_000_000, 300_000_000, 10, Approximation::Auto)
        .expect("valid");
    assert_eq!(algorithm, HypergeometricAlgorithm::Binomial);
    // Half the deck does not.
    let (_, algorithm) = random
        .hypergeometric_with_info(100, 30, 50, Approximation::Auto)
        .expect("valid");
    assert_eq!(algorithm, HypergeometricAlgorithm::Exact);
}

#[test]
fn approximate_results_stay_inside_the_valid_range() {
    let mut random = generator(15);

    // A normal is continuous and unbounded; a binomial count is neither. The
    // rounding and clamping must hold at the edges.
    for _ in 0..2_000 {
        let count = random.binomial_with(100, Unit::HALF, Approximation::Fast);
        assert!(count <= 100, "a binomial cannot exceed its trials");
    }

    // A Poisson count cannot go negative however far the normal wanders.
    for _ in 0..2_000 {
        let count = random.poisson_with(Rate::new(2.0).unwrap(), Approximation::Fast);
        // Just needs to exist as a u64 — the type forbids negatives, so what is
        // being tested is that the cast did not wrap.
        assert!(count < 1_000, "an absurd count suggests a wrapped cast: {count}");
    }

    // And a hypergeometric approximation cannot exceed the draw.
    for _ in 0..500 {
        let found = random
            .hypergeometric_with(1_000_000, 500_000, 20, Approximation::Fast)
            .expect("valid");
        assert!(found <= 20);
    }
}

#[test]
fn boundary_parameters_need_no_approximation() {
    let mut random = generator(16);

    for policy in [Approximation::Exact, Approximation::Auto, Approximation::Fast] {
        assert_eq!(
            random.binomial_with_info(0, Unit::HALF, policy),
            (0, BinomialAlgorithm::Exact),
            "no trials"
        );
        assert_eq!(
            random.binomial_with_info(1_000_000, Unit::ZERO, policy),
            (0, BinomialAlgorithm::Exact),
            "impossible"
        );
        assert_eq!(
            random.binomial_with_info(1_000_000, Unit::ONE, policy),
            (1_000_000, BinomialAlgorithm::Exact),
            "certain"
        );
        assert_eq!(
            random.binomial_with_info(1, Unit::HALF, policy).1,
            BinomialAlgorithm::Exact,
            "one trial"
        );
    }
}

// ===========================================================================
// Determinism
// ===========================================================================

#[test]
fn the_same_seed_gives_the_same_bulk_answers() {
    let shares = [Unit::percent(50).unwrap(), Unit::percent(50).unwrap()];
    let weights = [3.0, 1.0, 4.0, 1.0, 5.0];

    for seed in [1u64, 2, 12345] {
        let mut left = generator(seed);
        let mut right = generator(seed);

        assert_eq!(
            left.success_indices(10_000, Unit::one_in(100)),
            right.success_indices(10_000, Unit::one_in(100))
        );
        assert_eq!(left.multinomial(1_000, &shares), right.multinomial(1_000, &shares));
        assert_eq!(
            left.hypergeometric(100, 30, 10),
            right.hypergeometric(100, 30, 10)
        );
        assert_eq!(
            left.negative_binomial(3, Unit::one_in(4)),
            right.negative_binomial(3, Unit::one_in(4))
        );
        assert_eq!(
            left.choose_weighted_n(&weights, 3),
            right.choose_weighted_n(&weights, 3)
        );
        assert_eq!(
            left.reservoir_sample(0..1_000, 5),
            right.reservoir_sample(0..1_000, 5)
        );

        let mut one: Vec<u32> = (0..50).collect();
        let mut two: Vec<u32> = (0..50).collect();
        left.shuffle(&mut one);
        right.shuffle(&mut two);
        assert_eq!(one, two);
    }
}

// ===========================================================================
// Statistical sanity — large samples, generous tolerances, kept separate
// ===========================================================================

#[test]
fn statistical_sanity_of_the_bulk_samplers() {
    let mut random = generator(2024);

    // Multinomial shares land near their weights over many items.
    let shares = [
        Unit::percent(40).unwrap(),
        Unit::percent(30).unwrap(),
        Unit::percent(20).unwrap(),
        Unit::percent(10).unwrap(),
    ];
    let counts = random.multinomial(1_000_000, &shares).expect("valid");
    for (index, expected) in [0.4, 0.3, 0.2, 0.1].iter().enumerate() {
        let share: f64 = counts[index] as f64 / 1_000_000.0;
        assert!(
            (share - expected).abs() < 0.01,
            "category {index} took {share}, expected about {expected}"
        );
    }

    // Success indices arrive at about the stated rate.
    let found = random.success_indices(1_000_000, Unit::one_in(1_000));
    let rate: f64 = found.len() as f64 / 1_000_000.0;
    assert!((rate - 0.001).abs() < 0.0002, "rate was {rate}");

    // Hypergeometric averages the population's success fraction.
    let mut total: u64 = 0;
    for _ in 0..20_000 {
        total += random.hypergeometric(100, 30, 10).expect("valid");
    }
    let mean: f64 = total as f64 / 20_000.0;
    assert!((mean - 3.0).abs() < 0.1, "mean was {mean}, expected about 3");

    // The negative binomial's mean is r(1-p)/p: for r=3, p=1/4 that is 9.
    let mut total: u64 = 0;
    for _ in 0..20_000 {
        total += random.negative_binomial(3, Unit::one_in(4));
    }
    let mean: f64 = total as f64 / 20_000.0;
    assert!((mean - 9.0).abs() < 0.5, "mean was {mean}, expected about 9");

    // An approximate binomial still has about the right mean.
    let mut total: u64 = 0;
    for _ in 0..2_000 {
        total += random.binomial_with(100_000, Unit::one_in(1_000), Approximation::Fast);
    }
    let mean: f64 = total as f64 / 2_000.0;
    assert!((mean - 100.0).abs() < 3.0, "mean was {mean}, expected about 100");
}

// ===========================================================================
// Poisson-binomial approximation
// ===========================================================================

#[test]
fn the_lyapunov_bound_is_what_the_documentation_names() {
    use voxel_world::random::approximation::{
        normal_error_for_poisson_binomial, poisson_binomial_moments,
    };

    // With every trial the same, the Lyapunov form must reduce to the identical
    // case's shape — the only difference being the larger constant it is forced to
    // use, 0.5600 against 0.4748.
    let same: Vec<f64> = vec![0.3; 10_000];
    let lyapunov: f64 = normal_error_for_poisson_binomial(&same);
    let identical: f64 = normal_error_for_binomial(10_000, 0.3);

    let ratio: f64 = lyapunov / identical;
    assert!(
        (ratio - 0.5600 / 0.4748).abs() < 1e-9,
        "the two should differ only by their constants, got {ratio}"
    );
    assert!(lyapunov > identical, "the general bound is never the tighter one");

    // No spread, nothing to approximate.
    assert!(normal_error_for_poisson_binomial(&[]).is_infinite());
    assert!(normal_error_for_poisson_binomial(&[1.0, 1.0]).is_infinite());
    assert!(normal_error_for_poisson_binomial(&[0.0, 0.0]).is_infinite());

    // The moments are the textbook ones: for n identical trials, mean np and
    // variance np(1-p).
    let (mean, deviation, skewness) = poisson_binomial_moments(&same).expect("has spread");
    assert!((mean - 3_000.0).abs() < 1e-9);
    assert!((deviation - (10_000.0 * 0.3 * 0.7f64).sqrt()).abs() < 1e-9);
    // Skew of a binomial is (1-2p)/sqrt(np(1-p)), positive when p is below a half.
    assert!(skewness > 0.0, "chances below a half skew right");
    assert!(poisson_binomial_moments(&[1.0, 1.0]).is_none());

    // A symmetric set has no skew to correct.
    let (_, _, skewness) = poisson_binomial_moments(&vec![0.5; 100]).expect("has spread");
    assert!(skewness.abs() < 1e-12, "a fair set is symmetric");
}

#[test]
fn poisson_binomial_approximation_is_chosen_only_when_proven() {
    use voxel_world::random::PoissonBinomialAlgorithm;
    use voxel_world::random::approximation::normal_error_for_poisson_binomial;

    let mut random = generator(20);

    // A handful of trials proves nothing, whatever the policy.
    let few = [Unit::HALF, Unit::one_in(4), Unit::percent(80).unwrap()];
    for policy in [Approximation::Exact, Approximation::Auto, Approximation::Fast] {
        let (count, how) = random.poisson_binomial_with_info(&few, policy);

        assert_eq!(how, PoissonBinomialAlgorithm::Exact, "too few to approximate");
        assert!(count <= 3);
    }

    // Many varied trials: the bound becomes small enough for a loose policy.
    let many: Vec<Unit> = (0..200_000)
        .map(|n| Unit::out_of(n % 90 + 5, 100))
        .collect();
    let values: Vec<f64> = many.iter().map(|chance| chance.to_f64()).collect();
    let bound: f64 = normal_error_for_poisson_binomial(&values);

    assert!(bound < 1e-2, "the bound should permit Fast, got {bound}");

    let (count, how) = random.poisson_binomial_with_info(&many, Approximation::Fast);
    assert_eq!(how, PoissonBinomialAlgorithm::Normal);
    assert!(count <= 200_000, "clamped to the trial count");

    // Exact never takes it, however good the bound.
    let (_, how) = random.poisson_binomial_with_info(&many, Approximation::Exact);
    assert_eq!(how, PoissonBinomialAlgorithm::Exact);

    // All-certain trials have no variance, so there is nothing to approximate and
    // the exact path answers correctly.
    let certain = vec![Unit::ONE; 50_000];
    let (count, how) = random.poisson_binomial_with_info(&certain, Approximation::Fast);
    assert_eq!(how, PoissonBinomialAlgorithm::Exact);
    assert_eq!(count, 50_000);
}

/// The approximate path must centre where the exact one does. There is no skew
/// correction any more — see `approximation` for why one would break the bound and
/// buy nothing in the regime that permits the swap — so this checks the plain
/// rounded normal tracks the truth.
#[test]
fn the_approximate_poisson_binomial_centres_correctly() {
    let mut random = generator(21);

    // Chances well below a half skew the true distribution right. The rounding and
    // clamping must not shift the centre: a bias at the boundary would show here.
    let skewed: Vec<Unit> = vec![Unit::one_in(20); 100_000];
    let expected: f64 = 100_000.0 / 20.0;

    let mut total: u64 = 0;
    const ROUNDS: u32 = 300;

    for _ in 0..ROUNDS {
        total += random.poisson_binomial_with(&skewed, Approximation::Fast);
    }

    let mean: f64 = total as f64 / f64::from(ROUNDS);
    assert!(
        (mean - expected).abs() < expected * 0.01,
        "mean was {mean}, expected about {expected}"
    );

    // And the approximate answer tracks the exact one's mean.
    let mut exact_total: u64 = 0;
    for _ in 0..ROUNDS {
        exact_total += random.poisson_binomial_with(&skewed, Approximation::Exact);
    }
    let exact_mean: f64 = exact_total as f64 / f64::from(ROUNDS);

    assert!(
        (mean - exact_mean).abs() < expected * 0.02,
        "approximate {mean} strayed from exact {exact_mean}"
    );
}

// ===========================================================================
// The symmetry reduction in Hypergeometric
// ===========================================================================

/// The exact probability of drawing `hits` successes, by counting hands.
///
/// `C(K, k) * C(N-K, n-k) / C(N, n)`, in `u128` so the small cases below are exact.
fn hypergeometric_probability(population: u64, successes: u64, draws: u64, hits: u64) -> f64 {
    fn choose(n: u64, k: u64) -> u128 {
        if k > n {
            return 0;
        }

        let k: u64 = k.min(n - k);
        let mut result: u128 = 1;

        for step in 0..k {
            result = result * u128::from(n - step) / u128::from(step + 1);
        }

        result
    }

    let failures: u64 = population - successes;

    if hits > successes || draws < hits || draws - hits > failures {
        return 0.0;
    }

    let ways: u128 = choose(successes, hits) * choose(failures, draws - hits);

    ways as f64 / choose(population, draws) as f64
}

/// Every branch of the four-way symmetry must give the same distribution as the
/// plain definition. This is the test that reduction has to pass: it changes which
/// draws happen, so it cannot be checked by comparing sequences.
#[test]
fn the_hypergeometric_symmetries_preserve_the_distribution() {
    const SAMPLES: u32 = 120_000;

    // Chosen to exercise each branch: small draws, draws above the successes,
    // draws above half the population, successes above half, and both at once.
    let cases: [(u64, u64, u64); 8] = [
        (20, 8, 3),   // nothing to swap
        (20, 3, 8),   // draws exceed successes  -> swap
        (20, 8, 15),  // draws exceed half       -> complement draws
        (20, 15, 8),  // successes exceed half   -> complement successes
        (20, 15, 16), // both exceed half        -> both complements
        (20, 19, 19), // extreme
        (10, 5, 5),   // perfectly balanced, the case no symmetry helps
        (12, 1, 11),  // a single success, drawn nearly everything
    ];

    for (population, successes, draws) in cases {
        let mut random = generator(population * 1000 + successes * 10 + draws);
        let mut counts: Vec<u32> = vec![0; (draws + 1) as usize];

        for _ in 0..SAMPLES {
            let found: u64 = random
                .hypergeometric(population, successes, draws)
                .expect("valid");

            assert!(found <= draws && found <= successes, "outside the support");
            counts[found as usize] += 1;
        }

        for hits in 0..=draws {
            let expected: f64 =
                hypergeometric_probability(population, successes, draws, hits);
            let seen: f64 = f64::from(counts[hits as usize]) / f64::from(SAMPLES);

            // Four standard errors of a binomial proportion, plus a floor so the
            // near-impossible outcomes do not trip on their own rarity.
            let error: f64 =
                4.0 * (expected * (1.0 - expected) / f64::from(SAMPLES)).sqrt() + 0.002;

            assert!(
                (seen - expected).abs() < error,
                "N={population} K={successes} n={draws}: P(X={hits}) was {seen}, \
                 exact is {expected}"
            );
        }
    }
}

/// The reduction exists to make the loop short. This checks it actually is.
#[test]
fn a_lopsided_hypergeometric_is_cheap() {
    let mut random = generator(31);

    // Half a million drawn from a million, but only thirty successes exist. The
    // naive walk would take 500,000 draws; the reduction takes about thirty.
    // If the symmetry were missing this test would take minutes rather than
    // milliseconds, so the assertion is really the runtime.
    for _ in 0..2_000 {
        let found: u64 = random
            .hypergeometric(1_000_000, 30, 500_000)
            .expect("valid");

        assert!(found <= 30, "cannot exceed the successes that exist");
    }

    // And the mean is right: n * K / N = 500_000 * 30 / 1_000_000 = 15.
    let mut total: u64 = 0;
    for _ in 0..5_000 {
        total += random.hypergeometric(1_000_000, 30, 500_000).expect("valid");
    }
    let mean: f64 = total as f64 / 5_000.0;
    assert!((mean - 15.0).abs() < 0.4, "mean was {mean}, expected about 15");
}

// ===========================================================================
// The Gamma-Poisson path in NegativeBinomial
// ===========================================================================

#[test]
fn the_negative_binomial_mixture_matches_the_gap_sum() {
    let mut random = generator(32);

    // Either side of the crossover at twenty. The mean is r(1-p)/p in both cases,
    // and the mixture is an identity rather than an approximation, so they must
    // agree on it.
    for (successes, chance, expected) in [
        (5u64, Unit::one_in(4), 15.0),       // gap sum
        (20, Unit::one_in(4), 60.0),         // gap sum, at the boundary
        (21, Unit::one_in(4), 63.0),         // mixture, just over
        (1_000, Unit::one_in(4), 3_000.0),   // mixture
        (10_000, Unit::HALF, 10_000.0),      // mixture, large
    ] {
        let mut total: u64 = 0;
        const ROUNDS: u32 = 4_000;

        for _ in 0..ROUNDS {
            total += random.negative_binomial(successes, chance);
        }

        let mean: f64 = total as f64 / f64::from(ROUNDS);
        // Variance is r(1-p)/p^2, so the standard error of the mean is
        // sqrt(r(1-p))/p / sqrt(ROUNDS). Four of those, with a small floor.
        let probability: f64 = chance.to_f64();
        let deviation: f64 =
            (successes as f64 * (1.0 - probability)).sqrt() / probability / f64::from(ROUNDS).sqrt();

        assert!(
            (mean - expected).abs() < 4.0 * deviation + 0.5,
            "r={successes}: mean was {mean}, expected about {expected}"
        );
    }

    // A huge r no longer costs a draw per success — this would be ten million
    // geometric draws without the mixture.
    let failures: u64 = random.negative_binomial(10_000_000, Unit::HALF);
    assert!(failures > 9_000_000 && failures < 11_000_000, "got {failures}");
}

/// The constant-time path, checked the same way as the walk.
///
/// The cases above all sit under the 32-draw limit and so exercise the walk. These
/// are chosen to clear it — the reduced draw count is 50 — while staying small
/// enough that `C(100, 50)` and the terms above it still fit in a `u128`, so the
/// reference probabilities are exact integer arithmetic rather than a float
/// approximation of the thing being tested.
#[test]
fn the_constant_time_hypergeometric_matches_the_exact_distribution() {
    const SAMPLES: u32 = 150_000;

    let cases: [(u64, u64, u64); 4] = [
        (100, 50, 50), // perfectly balanced: no symmetry helps, so HRUA must
        (100, 40, 45), // off-centre
        (100, 50, 60), // draws above half -> complement, still over the limit
        (90, 45, 45),  // balanced again, odd population
    ];

    for (population, successes, draws) in cases {
        let mut random = generator(population * 7 + successes * 3 + draws);
        let mut counts: Vec<u32> = vec![0; (draws + 1) as usize];

        for _ in 0..SAMPLES {
            let found: u64 = random
                .hypergeometric(population, successes, draws)
                .expect("valid");

            assert!(found <= draws && found <= successes, "outside the support");
            counts[found as usize] += 1;
        }

        for hits in 0..=draws {
            let expected: f64 =
                hypergeometric_probability(population, successes, draws, hits);
            let seen: f64 = f64::from(counts[hits as usize]) / f64::from(SAMPLES);
            let error: f64 =
                4.0 * (expected * (1.0 - expected) / f64::from(SAMPLES)).sqrt() + 0.002;

            assert!(
                (seen - expected).abs() < error,
                "N={population} K={successes} n={draws}: P(X={hits}) was {seen}, \
                 exact is {expected}"
            );
        }
    }
}

/// The balanced case is the one no symmetry can shrink, and the one the walk would
/// choke on. Half a million drawn from a million, half of them successes.
#[test]
fn a_balanced_hypergeometric_is_still_cheap() {
    let mut random = generator(41);

    // Without the constant-time path this is 500,000 draws per sample and the test
    // would never finish. The runtime is the assertion.
    let mut total: u64 = 0;
    const ROUNDS: u32 = 3_000;

    for _ in 0..ROUNDS {
        let found: u64 = random
            .hypergeometric(1_000_000, 500_000, 500_000)
            .expect("valid");

        assert!(found <= 500_000);
        total += found;
    }

    // Mean is n * K / N = 250,000.
    let mean: f64 = total as f64 / f64::from(ROUNDS);
    assert!(
        (mean - 250_000.0).abs() < 1_000.0,
        "mean was {mean}, expected about 250,000"
    );

    // And the spread is right too: the finite-population correction makes this much
    // tighter than the binomial it would be with replacement. Standard deviation is
    // sqrt(n K (N-K) (N-n) / (N^2 (N-1))) which is about 250 here, against a
    // binomial's 354.
    let mut squared: f64 = 0.0;
    for _ in 0..ROUNDS {
        let found = random.hypergeometric(1_000_000, 500_000, 500_000).unwrap() as f64;
        squared += (found - 250_000.0) * (found - 250_000.0);
    }
    let deviation: f64 = (squared / f64::from(ROUNDS)).sqrt();

    assert!(
        (deviation - 250.0).abs() < 40.0,
        "deviation was {deviation}, expected about 250 — a binomial would give 354"
    );
}

// ===========================================================================
// Against slow reference samplers
// ===========================================================================

/// `SparseSuccesses` must be indistinguishable from asking every position.
///
/// Not the same *sequence* — it consumes words differently — but the same
/// distribution. Compared here against an explicit Bernoulli loop over the same
/// positions, on both the count and the per-position frequency, since a convention
/// slip in the gap arithmetic would shift one or the other.
#[test]
fn sparse_successes_matches_asking_every_position() {
    const ROUNDS: u32 = 4_000;
    const COUNT: u64 = 64;

    for chance in [Unit::percent(5).unwrap(), Unit::percent(25).unwrap(), Unit::HALF] {
        // The reference: one coin per position, the obvious way.
        let mut plain = generator(700);
        let mut plain_total: u64 = 0;
        let mut plain_hits: Vec<u32> = vec![0; COUNT as usize];

        for _ in 0..ROUNDS {
            for index in 0..COUNT {
                if chance.decide_from(&mut plain) {
                    plain_total += 1;
                    plain_hits[index as usize] += 1;
                }
            }
        }

        // The sparse sampler over the same positions.
        let mut sparse = generator(701);
        let mut sparse_total: u64 = 0;
        let mut sparse_hits: Vec<u32> = vec![0; COUNT as usize];

        for _ in 0..ROUNDS {
            for index in sparse.success_indices(COUNT, chance) {
                sparse_total += 1;
                sparse_hits[index as usize] += 1;
            }
        }

        // The expected count per round is `count * chance`.
        let expected: f64 = COUNT as f64 * chance.to_f64();
        let plain_mean: f64 = plain_total as f64 / f64::from(ROUNDS);
        let sparse_mean: f64 = sparse_total as f64 / f64::from(ROUNDS);

        assert!(
            (plain_mean - expected).abs() < 0.2,
            "the reference itself is off: {plain_mean} vs {expected}"
        );
        assert!(
            (sparse_mean - expected).abs() < 0.2,
            "sparse mean {sparse_mean} vs expected {expected} at chance {chance}"
        );

        // And no position is favoured: a gap-arithmetic slip would bias index zero or
        // the last index in particular.
        let per_position: f64 = f64::from(ROUNDS) * chance.to_f64();
        for (index, hits) in sparse_hits.iter().enumerate().take(COUNT as usize) {
            let seen: f64 = f64::from(*hits);
            let allowed: f64 = 5.0 * per_position.sqrt() + 5.0;

            assert!(
                (seen - per_position).abs() < allowed,
                "position {index} was hit {seen} times, expected about {per_position}"
            );
        }

        // Index zero specifically: a geometric that counted trials instead of failures
        // would never land there.
        assert!(sparse_hits[0] > 0, "index zero is reachable");
        assert!(
            sparse_hits[COUNT as usize - 1] > 0,
            "the last index is reachable"
        );
    }
}

/// The negative binomial, against counting Bernoulli trials by hand for small `r`.
#[test]
fn negative_binomial_matches_counting_trials() {
    use voxel_world::random::distributions::NegativeBinomial;

    const ROUNDS: u32 = 20_000;

    for (successes, chance) in [(1u64, Unit::HALF), (3, Unit::one_in(4)), (5, Unit::percent(30).unwrap())] {
        // Reference: flip until `successes` successes, counting the failures.
        let mut plain = generator(710);
        let mut reference_total: u64 = 0;

        for _ in 0..ROUNDS {
            let mut wins: u64 = 0;
            let mut losses: u64 = 0;

            while wins < successes {
                if chance.decide_from(&mut plain) {
                    wins += 1;
                } else {
                    losses += 1;
                }
            }

            reference_total += losses;
        }

        let mut sampled = generator(711);
        let coin = NegativeBinomial::by_gaps(successes, chance);
        let mut sampled_total: u64 = 0;

        for _ in 0..ROUNDS {
            sampled_total += sampled.sample(&coin);
        }

        let expected: f64 = successes as f64 * (1.0 - chance.to_f64()) / chance.to_f64();
        let reference_mean: f64 = reference_total as f64 / f64::from(ROUNDS);
        let sampled_mean: f64 = sampled_total as f64 / f64::from(ROUNDS);

        // The reference pins the convention: failures, not trials.
        assert!(
            (reference_mean - expected).abs() < expected * 0.05 + 0.1,
            "reference {reference_mean} vs theory {expected}"
        );
        assert!(
            (sampled_mean - reference_mean).abs() < expected * 0.08 + 0.2,
            "sampled {sampled_mean} strayed from reference {reference_mean}"
        );
    }
}

/// The multinomial, against rolling each item through a categorical by hand.
#[test]
fn multinomial_matches_rolling_each_item() {
    const ROUNDS: u32 = 2_000;
    const ITEMS: u64 = 40;

    let shares = [
        Unit::percent(50).unwrap(),
        Unit::percent(30).unwrap(),
        Unit::percent(20).unwrap(),
    ];

    // Reference: pick a category per item, by walking the cumulative shares.
    let mut plain = generator(720);
    let mut reference: [u64; 3] = [0; 3];

    for _ in 0..ROUNDS {
        for _ in 0..ITEMS {
            let draw: Unit = plain.unit();
            let mut running: f64 = 0.0;
            let total: f64 = shares.iter().map(|s| s.to_f64()).sum();

            for (index, share) in shares.iter().enumerate() {
                running += share.to_f64() / total;
                if draw.to_f64() < running || index == 2 {
                    reference[index] += 1;
                    break;
                }
            }
        }
    }

    let mut sampled = generator(721);
    let mut counts: [u64; 3] = [0; 3];

    for _ in 0..ROUNDS {
        let round = sampled.multinomial(ITEMS, &shares).expect("valid");
        assert_eq!(round.iter().sum::<u64>(), ITEMS, "every item placed");

        for index in 0..3 {
            counts[index] += round[index];
        }
    }

    let total_items: f64 = f64::from(ROUNDS) * ITEMS as f64;
    for index in 0..3 {
        let reference_share: f64 = reference[index] as f64 / total_items;
        let sampled_share: f64 = counts[index] as f64 / total_items;

        assert!(
            (sampled_share - reference_share).abs() < 0.02,
            "category {index}: sampled {sampled_share} vs reference {reference_share}"
        );
    }
}

// ===========================================================================
// Variances, not just means
// ===========================================================================

/// A mean can be right while the spread is wrong — that is exactly what averaging
/// probabilities and calling it a binomial gets you — so the second moment is checked
/// too.
#[test]
fn the_spreads_match_their_theoretical_variances() {
    const ROUNDS: u32 = 40_000;

    fn spread(samples: &[u64]) -> (f64, f64) {
        let n: f64 = samples.len() as f64;
        let mean: f64 = samples.iter().map(|v| *v as f64).sum::<f64>() / n;
        let variance: f64 =
            samples.iter().map(|v| (*v as f64 - mean) * (*v as f64 - mean)).sum::<f64>() / n;

        (mean, variance)
    }

    // Binomial: mean np, variance np(1-p).
    let mut random = generator(730);
    let p: f64 = 0.25;
    let samples: Vec<u64> = (0..ROUNDS)
        .map(|_| random.binomial(100, Probability::new(p).unwrap()))
        .collect();
    let (mean, variance) = spread(&samples);
    assert!((mean - 25.0).abs() < 0.3, "binomial mean {mean}");
    assert!((variance - 100.0 * p * (1.0 - p)).abs() < 1.0, "binomial variance {variance}");

    // Poisson: mean and variance both lambda.
    let mut random = generator(731);
    let samples: Vec<u64> = (0..ROUNDS)
        .map(|_| random.poisson(Rate::new(9.0).unwrap()))
        .collect();
    let (mean, variance) = spread(&samples);
    assert!((mean - 9.0).abs() < 0.1, "poisson mean {mean}");
    assert!((variance - 9.0).abs() < 0.4, "poisson variance {variance}");

    // Geometric (failures before success): mean (1-p)/p, variance (1-p)/p^2.
    let mut random = generator(732);
    let chance: Unit = Unit::one_in(5);
    let samples: Vec<u64> = (0..ROUNDS)
        .map(|_| random.geometric(chance.to_probability()))
        .collect();
    let (mean, variance) = spread(&samples);
    assert!((mean - 4.0).abs() < 0.1, "geometric mean {mean}");
    assert!((variance - 20.0).abs() < 1.5, "geometric variance {variance}");

    // Negative binomial: mean r(1-p)/p, variance r(1-p)/p^2.
    let mut random = generator(733);
    let samples: Vec<u64> = (0..ROUNDS)
        .map(|_| random.negative_binomial(4, Unit::one_in(4)))
        .collect();
    let (mean, variance) = spread(&samples);
    assert!((mean - 12.0).abs() < 0.3, "negative binomial mean {mean}");
    assert!((variance - 48.0).abs() < 3.0, "negative binomial variance {variance}");

    // Hypergeometric: mean nK/N, variance nK(N-K)(N-n) / (N^2 (N-1)).
    let mut random = generator(734);
    let samples: Vec<u64> = (0..ROUNDS)
        .map(|_| random.hypergeometric(100, 30, 10).unwrap())
        .collect();
    let (mean, variance) = spread(&samples);
    let theory: f64 = 10.0 * 30.0 * 70.0 * 90.0 / (100.0 * 100.0 * 99.0);
    assert!((mean - 3.0).abs() < 0.05, "hypergeometric mean {mean}");
    assert!((variance - theory).abs() < 0.08, "hypergeometric variance {variance} vs {theory}");

    // Poisson-binomial with identical chances must agree with the binomial it is.
    let mut random = generator(735);
    let chances: Vec<Unit> = vec![Unit::one_in(4); 100];
    let samples: Vec<u64> = (0..ROUNDS).map(|_| random.poisson_binomial(&chances)).collect();
    let (mean, variance) = spread(&samples);
    assert!((mean - 25.0).abs() < 0.3, "poisson-binomial mean {mean}");
    assert!(
        (variance - 100.0 * 0.25 * 0.75).abs() < 1.0,
        "poisson-binomial variance {variance} should match a binomial's"
    );
}

// ===========================================================================
// Portability claims
// ===========================================================================

/// The claims have to be checkable, not just written down.
#[test]
fn the_portable_paths_are_the_ones_advertised() {
    use voxel_world::random::distributions::{Hypergeometric, NegativeBinomial};

    // Hypergeometric: the walk is portable, HRUA is not.
    let small = Hypergeometric::new(100, 30, 10).unwrap();
    let large = Hypergeometric::new(1_000_000, 500_000, 500_000).unwrap();

    assert!(small.is_portable(), "a small draw takes the walk");
    assert!(!large.is_portable(), "a balanced large draw reaches HRUA");
    assert!(
        Hypergeometric::walking(1_000_000, 500_000, 500_000).unwrap().is_portable(),
        "forcing the walk makes it portable"
    );

    // And forcing the walk must not change the distribution, only the cost.
    let mut forced = generator(740);
    let deck = Hypergeometric::walking(200, 60, 80).unwrap();
    let mut total: u64 = 0;
    for _ in 0..4_000 {
        let found = forced.sample(&deck);
        // The support is bounded by whichever of the successes and the draws is
        // smaller, which here is the 60 successes rather than the 80 draws.
        assert!(found <= 60, "drew {found} successes from a deck holding 60");
        total += found;
    }
    let mean: f64 = total as f64 / 4_000.0;
    assert!((mean - 24.0).abs() < 0.5, "forced walk mean {mean}, expected 80*60/200 = 24");

    // NegativeBinomial: the gap sum is portable, the mixture is not.
    assert!(NegativeBinomial::new(5, Unit::HALF).is_portable());
    assert!(!NegativeBinomial::new(5_000, Unit::HALF).is_portable());
    assert!(NegativeBinomial::by_gaps(5_000, Unit::HALF).is_portable());
}
