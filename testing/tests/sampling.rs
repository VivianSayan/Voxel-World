use std::collections::{BTreeSet, VecDeque};
use voxel_world::random::seed::Seed;
use voxel_world::random::{
    Beta, Categorical, Cauchy, Distribution, Gamma, GeometricRatio, LogNormal, Normal, Pareto,
    Power, Random, Triangular, TruncatedNormal, Uniform, Zipf,
};
use voxel_world::structures::{
    collections::{FuzzySet, MultiSet, OrderedSet, RingBuffer},
    sampling::uniform_indices,
    traits::{Choose, Measured},
};
use voxel_world::units::{Probability, Ratio, Weights};

fn rng(seed: u128) -> Random {
    Random::new(Seed::from_raw(seed))
}

#[test]
fn distinct_sampling_matches_a_full_shuffle_reference() {
    for len in 0..25 {
        for count in 0..=len + 2 {
            for seed in 0..16 {
                let mut reference_random = rng(seed);
                let mut expected: Vec<_> = (0..len).collect();
                for position in 0..count.min(len) {
                    let other = position + reference_random.uniform_index(len - position);
                    expected.swap(position, other);
                }
                expected.truncate(count.min(len));
                let actual = uniform_indices(len, count, Some(&mut rng(seed)));
                assert_eq!(actual, expected, "len={len}, count={count}, seed={seed}");
                assert_eq!(actual.iter().collect::<BTreeSet<_>>().len(), actual.len());
            }
        }
    }
    // This would be impossible with an allocation proportional to population.
    let tiny = rng(42).sample_distinct(3, usize::MAX);
    assert_eq!(tiny.len(), 3);
    assert_eq!(tiny.iter().collect::<BTreeSet<_>>().len(), 3);
    assert!(tiny.iter().all(|index| *index < usize::MAX));
}

fn check_weighted<C: Choose<Item = &'static str>>(collection: &C) {
    let mut random = rng(42);
    let mut singles = 0;
    let mut multiples = 0;
    for _ in 0..20_000 {
        singles += usize::from(collection.choose(&mut random) == Some(&"common"));
        multiples += usize::from(collection.choose_multiple(&mut random, 1)[0] == &"common");
    }
    assert!(singles > 19_900, "single weighted choices: {singles}");
    assert!(multiples > 19_900, "multiple weighted choices: {multiples}");
}

#[test]
fn all_measured_collections_weight_single_and_multiple_choices() {
    let weights = [("common", 1000.0), ("rare", 1.0)]
        .into_iter()
        .collect::<voxel_world::structures::collections::WeightedSet<_>>();
    let mut counts = MultiSet::new();
    counts.insert_times("common", 1000);
    counts.insert("rare");
    let memberships = [
        ("common", Probability::ALWAYS),
        ("rare", Probability::new(0.001).unwrap()),
    ]
    .into_iter()
    .collect::<FuzzySet<_>>();
    check_weighted(&weights);
    check_weighted(&counts);
    check_weighted(&memberships);
    assert!(FuzzySet::<u8>::new().choose(&mut rng(1)).is_none());
}

#[test]
fn fuzzy_operations_preserve_memberships_but_totals_can_exceed_one() {
    let mut fuzzy = FuzzySet::new();
    fuzzy.set_membership(1, Probability::ALWAYS);
    fuzzy.set_membership(2, Probability::ALWAYS);
    assert_eq!(fuzzy.total_measure(), 2.0);
    assert_eq!(fuzzy.measure_of(&1), Probability::ALWAYS);
    for _ in 0..100 {
        fuzzy.reinforce(3, Probability::new(0.1).unwrap());
    }
    fuzzy.weaken(&3, Probability::new(0.25).unwrap());
    fuzzy.normalise();
    assert!((fuzzy.iter().map(|(_, p)| p.value()).sum::<f64>() - 1.0).abs() < 1e-14);
    for result in [
        fuzzy.clone(),
        fuzzy.complement(),
        fuzzy.union(&fuzzy),
        fuzzy.intersection(&fuzzy),
        fuzzy.difference(&fuzzy),
    ] {
        for (_, p) in &result {
            assert!(Probability::new(p.value()).is_some());
        }
    }
}

fn check_indexed<C: Choose<Item = usize>>(collection: &C) {
    let mut random = rng(3);
    let mut reference = rng(3);
    for _ in 0..100 {
        assert_eq!(
            collection.choose(&mut random),
            Some(&reference.uniform_index(100))
        );
    }
    let chosen = collection.choose_multiple(&mut random, 20);
    assert_eq!(chosen.iter().collect::<BTreeSet<_>>().len(), 20);
}

#[test]
fn indexed_collections_use_one_random_index_per_pick() {
    // Stack and Queue were removed as renamed Vec and VecDeque; the std types
    // now carry `Choose` themselves, and must pick the same way.
    check_indexed(&(0..100).collect::<VecDeque<_>>());
    check_indexed(&(0..100).collect::<Vec<_>>());
    check_indexed(&(0..100).collect::<OrderedSet<_>>());
    let mut ring = RingBuffer::new(100);
    for i in 0..100 {
        ring.push_back(i);
    }
    check_indexed(&ring);
    assert!(VecDeque::<u8>::new().choose(&mut rng(0)).is_none());
    assert!(Vec::<u8>::new().choose(&mut rng(0)).is_none());
}

#[test]
fn cached_geometric_matches_one_shot_at_every_strategy_boundary() {
    for count in [
        0,
        1,
        2,
        7,
        64,
        256,
        257,
        1000,
        1_000_000,
        10_000_000_000,
        u64::MAX,
    ] {
        let sampler = GeometricRatio::one_in(count);
        let mut cached = rng(19);
        let mut fresh = rng(19);
        for _ in 0..1000 {
            assert_eq!(sampler.sample(&mut cached), fresh.geometric_one_in(count));
        }
    }
    assert!(GeometricRatio::new(Ratio::new(3, 2).unwrap()).is_none());
}

#[test]
fn geometric_mean_and_survival_match_the_distribution() {
    for count in [2u64, 7, 256, 257, 1000, 10_000_000_000] {
        let sampler = GeometricRatio::one_in(count);
        let mut random = rng(17);
        let n = 30_000;
        let mut total = 0.0;
        let mut survived = 0;
        for _ in 0..n {
            let sample = sampler.sample(&mut random);
            total += sample as f64;
            survived += usize::from(sample >= count);
        }
        let p = 1.0 / count as f64;
        assert!(
            (total / n as f64 * p - (1.0 - p)).abs() < 0.04,
            "mean, count={count}"
        );
        let expected_tail = (count as f64 * (-p).ln_1p()).exp();
        assert!(
            (survived as f64 / n as f64 - expected_tail).abs() < 0.02,
            "tail, count={count}"
        );
    }
}

#[test]
fn all_distribution_constructors_validate_in_release_too() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(Gamma::new(invalid, 1.0).is_none());
        assert!(Beta::new(1.0, invalid).is_none());
        assert!(Power::new(invalid).is_none());
        assert!(Cauchy::new(0.0, invalid).is_none());
        assert!(Pareto::new(1.0, invalid).is_none());
        assert!(Zipf::new(10, invalid).is_none());
    }
    assert!(Normal::new(f64::NAN, 1.0).is_none());
    assert!(Normal::new(0.0, -1.0).is_none());
    assert!(LogNormal::new(0.0, -1.0).is_none());
    assert!(Uniform::new(1.0, 0.0).is_none());
    assert!(Uniform::new(-f64::MAX, f64::MAX).is_none());
    assert!(Triangular::new(0.0, 2.0, 1.0).is_none());
    assert!(TruncatedNormal::new(0.0, 0.0, -1.0, 1.0).is_none());
    assert!(TruncatedNormal::new(0.0, 1.0, 2.0, 1.0).is_none());
    assert!(Zipf::new(0, 1.0).is_none());
    assert!(std::panic::catch_unwind(|| rng(0).gamma(f64::NAN, 1.0)).is_err());
}

#[test]
fn normal_gamma_beta_and_truncation_have_plausible_samples() {
    let mut random = rng(99);
    let normal = Normal::new(5.0, 2.0).unwrap();
    let gamma = Gamma::new(2.0, 3.0).unwrap();
    let beta = Beta::new(2.0, 3.0).unwrap();
    let truncated = TruncatedNormal::new(0.0, 1.0, 8.0, 9.0).unwrap();
    let (mut normal_sum, mut normal_square, mut gamma_sum, mut beta_sum) = (0.0, 0.0, 0.0, 0.0);
    let n = 20_000;
    for _ in 0..n {
        let value = normal.sample(&mut random);
        normal_sum += value;
        normal_square += (value - 5.0).powi(2);
        gamma_sum += gamma.sample(&mut random);
        let value = beta.sample(&mut random);
        assert!((0.0..=1.0).contains(&value));
        beta_sum += value;
        assert!((8.0..=9.0).contains(&truncated.sample(&mut random)));
    }
    assert!((normal_sum / n as f64 - 5.0).abs() < 0.08);
    assert!((normal_square / n as f64 - 4.0).abs() < 0.2);
    assert!((gamma_sum / n as f64 - 6.0).abs() < 0.2);
    assert!((beta_sum / n as f64 - 0.4).abs() < 0.02);
}

#[test]
fn uniform_floats_stay_half_open_even_for_adjacent_bounds() {
    let mut random = rng(90);
    for _ in 0..20_000 {
        assert!((0.0..1.0).contains(&random.next_f64()));
        assert!((0.0..1.0).contains(&random.next_f32()));
        assert_eq!(random.uniform_f64(1.0, 1.0f64.next_up()), 1.0);
        assert_eq!(random.uniform_f32(1.0, 1.0f32.next_up()), 1.0);
    }
}

#[test]
fn alias_preparation_does_not_overflow_large_finite_weights() {
    let weights = Weights::new([f64::MAX / 2.0, f64::MAX / 4.0, 0.0]).unwrap();
    let categorical = Categorical::new(&weights);
    let mut random = rng(72);
    let mut counts = [0; 3];
    for _ in 0..20_000 {
        counts[categorical.sample(&mut random)] += 1;
    }
    assert_eq!(counts[2], 0);
    assert!((counts[0] as f64 / 20_000.0 - 2.0 / 3.0).abs() < 0.02);
}
