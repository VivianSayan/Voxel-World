use voxel_world::random::{
    Bernoulli, BernoulliMask, BernoulliRatio, BinomialRatio, Categorical, DiscreteGaussian,
    DiscreteLaplace, Distribution, GeometricRatio, IntegerCategorical, Normal, PoissonRatio,
    PortableDistribution, RANDOM_ALGORITHM_VERSION, Random, RandomSource, RandomState, Seed,
    StochasticRound, Triangular, Uniform, UniformU64, UnitCircle, UnitDisc, UnitHypersphere,
    UnitSphere, WeightedDiscrete,
};
use voxel_world::units::{Probability, Ratio, Weights};

fn rng(seed: u128) -> Random {
    Random::new(Seed::from_raw(seed))
}

fn ratio(numerator: u64, denominator: u64) -> Ratio {
    Ratio::new(numerator, denominator).unwrap()
}

/// Asserts a frequency lies within five standard errors of `expected`.
fn assert_frequency(hits: u64, draws: u64, expected: f64, label: &str) {
    let observed = hits as f64 / draws as f64;
    let error = (expected * (1.0 - expected) / draws as f64).sqrt();
    assert!(
        (observed - expected).abs() <= 5.0 * error + 1e-12,
        "{label}: observed {observed}, expected {expected}"
    );
}

/// Sample mean and variance.
fn moments(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let values: Vec<f64> = values.collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance)
}

#[test]
fn seed_samples_are_pure_and_match_its_one_shot_draws() {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let triangle = Triangular::new(0.0, 2.0, 5.0).unwrap();
    for index in 0..200 {
        let seed = Seed::from_raw(7).index(index);
        assert_eq!(seed.sample(&normal), seed.sample(&normal));
        assert_eq!(seed.sample(&triangle), seed.sample(&triangle));
        let mut cursor = seed.cursor();
        assert_eq!(
            seed.sample(&UniformU64::new(0, u64::MAX).unwrap()),
            cursor.next_u64()
        );
        let mut cursor = seed.cursor();
        assert_eq!(seed.unit_f64(), cursor.unit_f64());
    }
    // Different seeds spread out like independent draws.
    let (mean, variance) =
        moments((0..20_000).map(|index| Seed::from_raw(3).index(index).sample(&normal)));
    assert!(
        mean.abs() < 0.05 && (variance - 1.0).abs() < 0.05,
        "{mean} {variance}"
    );
}

#[test]
fn seed_cursor_advances_and_round_trips() {
    let seed = Seed::from_raw(0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210);
    let full = UniformU64::new(0, u64::MAX).unwrap();
    let mut cursor = seed.cursor();

    assert_eq!(cursor.seed(), seed);
    assert_eq!(cursor.position(), seed);
    assert_eq!(cursor.sample(&full), seed.sample(&full));
    assert_eq!(cursor.position(), seed.advance());

    let encoded = cursor.to_bytes();
    let mut restored = voxel_world::random::SeedCursor::from_bytes(&encoded).unwrap();
    assert_eq!(restored, cursor);
    assert_eq!(restored.sample(&full), cursor.sample(&full));
    assert!(voxel_world::random::SeedCursor::from_bytes(&encoded[..31]).is_none());
}

#[test]
fn all_seed_bits_reach_random_state() {
    // These collide under the old xor-and-rotate 128-to-64 fold.
    let low = Seed::from_raw(0);
    let high = Seed::from_raw((1u128 << 64) | (1u128 << 32));
    assert_eq!(low.as_u64(), high.as_u64());

    let low_state = Random::new(low).snapshot();
    let high_state = Random::new(high).snapshot();
    assert_ne!(low_state.words(), high_state.words());
}

#[test]
fn random_state_round_trips_integer_and_cached_normal_state() {
    let mut random = rng(0xA11C_E5EED);
    for _ in 0..7 {
        random.next_u64();
    }
    random.standard_normal();
    assert!(random.snapshot().spare_normal().is_some());

    let bytes = random.snapshot().to_bytes();
    let state = RandomState::from_bytes(&bytes).unwrap();
    let mut restored = Random::from_state(state).unwrap();
    assert_eq!(random.standard_normal(), restored.standard_normal());
    for _ in 0..32 {
        assert_eq!(random.next_u64(), restored.next_u64());
    }

    let mut invalid = bytes;
    invalid[..4].copy_from_slice(&(RANDOM_ALGORITHM_VERSION + 1).to_le_bytes());
    assert!(RandomState::from_bytes(&invalid).is_none());

    let mut invalid = bytes;
    invalid[52] = 2;
    assert!(RandomState::from_bytes(&invalid).is_none());

    let mut invalid = Random::new(Seed::default()).snapshot().to_bytes();
    invalid[60] = 1;
    assert!(RandomState::from_bytes(&invalid).is_none());

    assert!(RandomState::from_bytes(&bytes[..bytes.len() - 1]).is_none());
    assert!(
        RandomState::from_parts(RANDOM_ALGORITHM_VERSION, Seed::default(), [0; 4], None).is_none()
    );
    assert!(
        RandomState::from_parts(
            RANDOM_ALGORITHM_VERSION,
            Seed::default(),
            [1, 0, 0, 0],
            Some(f64::NAN),
        )
        .is_none()
    );
}

#[test]
fn portable_distribution_marker_covers_replay_types() {
    fn assert_portable<D: PortableDistribution>(_distribution: &D) {}

    assert_portable(&UniformU64::new(0, 10).unwrap());
    assert_portable(&BernoulliRatio::new(ratio(1, 3)).unwrap());
    assert_portable(&BinomialRatio::new(10, ratio(1, 3)).unwrap());
    assert_portable(&PoissonRatio::new(ratio(7, 3)));
    assert_portable(&DiscreteLaplace::new(ratio(3, 2)).unwrap());
    assert_portable(&DiscreteGaussian::new(0, ratio(3, 2)).unwrap());
    assert_portable(&IntegerCategorical::new(&[1, 2, 3]).unwrap());

    // These floating-point samplers use only reproducible IEEE-754 operations
    // (including correctly rounded sqrt), never platform libm transcendentals.
    assert_portable(&Uniform::new(-1.0, 2.0).unwrap());
    assert_portable(&Triangular::new(-1.0, 0.0, 2.0).unwrap());
    assert_portable(&Bernoulli::new(Probability::EVEN));
    assert_portable(&StochasticRound::new(1.25).unwrap());
    assert_portable(&UnitDisc);
    assert_portable(&UnitCircle);
    assert_portable(&UnitSphere);
    assert_portable(&UnitHypersphere);

    let weights = Weights::new([1.0, 2.0, 3.0]).unwrap();
    assert_portable(&weights);
    assert_portable(&Categorical::new(&weights));
    assert_portable(&GeometricRatio::one_in(10_000));
}

#[test]
fn ratio_coins_are_exact_fractions() {
    let mut random = rng(1);
    let draws = 300_000;
    for (numerator, denominator) in [(1, 3), (2, 7), (1, 1000), (999, 1000)] {
        let coin = BernoulliRatio::new(ratio(numerator, denominator)).unwrap();
        let hits = (0..draws).filter(|_| coin.sample(&mut random)).count() as u64;
        assert_frequency(hits, draws, numerator as f64 / denominator as f64, "coin");
    }
    assert!(!rng(0).sample(&BernoulliRatio::one_in(0)));
    assert!(rng(0).sample(&BernoulliRatio::one_in(1)));
    assert!(BernoulliRatio::new(ratio(3, 2)).is_none());
    assert_eq!(Seed::from_raw(9).chance_ratio(ratio(3, 2)), None);
}

#[test]
fn masks_set_each_lane_independently_at_the_ratio() {
    for (numerator, denominator) in [(3, 7), (5, 8), (1, 100)] {
        let mask = BernoulliMask::new(ratio(numerator, denominator)).unwrap();
        let p = numerator as f64 / denominator as f64;
        let mut random = rng(numerator as u128);
        let words = 20_000;
        let mut lane_hits = [0u64; 64];
        let counts: Vec<f64> = (0..words)
            .map(|_| {
                let bits = mask.sample(&mut random);
                for (lane, hits) in lane_hits.iter_mut().enumerate() {
                    *hits += (bits >> lane) & 1;
                }
                bits.count_ones() as f64
            })
            .collect();
        for hits in lane_hits {
            assert_frequency(hits, words, p, "lane");
        }
        // Independent lanes make the count binomial, variance 64 p (1 - p).
        let (mean, variance) = moments(counts.into_iter());
        assert!((mean - 64.0 * p).abs() < 0.1, "mean {mean}");
        assert!(
            (variance / (64.0 * p * (1.0 - p)) - 1.0).abs() < 0.05,
            "variance {variance}"
        );
    }
    // One half is decided by a single word: every lane is its bit, inverted.
    let mut random = rng(5);
    let mut replay = random.clone();
    assert_eq!(random.chance_mask(Ratio::HALF), !replay.next_u64());
    assert_eq!(random.next_u64(), replay.next_u64());
}

#[test]
fn exact_binomial_has_binomial_moments() {
    let mut random = rng(11);
    for (trials, numerator, denominator) in [(1000, 3, 10), (70, 1, 2), (4096, 1, 64)] {
        let binomial = BinomialRatio::new(trials, ratio(numerator, denominator)).unwrap();
        let p = numerator as f64 / denominator as f64;
        let (mean, variance) = moments((0..20_000).map(|_| binomial.sample(&mut random) as f64));
        let expected_variance = trials as f64 * p * (1.0 - p);
        assert!((mean - trials as f64 * p).abs() < 5.0 * (expected_variance / 20_000.0).sqrt());
        assert!(
            (variance / expected_variance - 1.0).abs() < 0.05,
            "{trials}: {variance}"
        );
    }
    assert_eq!(random.binomial_ratio(0, Ratio::HALF), 0);
    assert_eq!(random.binomial_ratio(77, Ratio::ZERO), 0);
    assert_eq!(random.binomial_ratio(77, Ratio::ONE), 77);
}

#[test]
fn exact_poisson_matches_its_probabilities() {
    let mut random = rng(21);
    for (numerator, denominator) in [(1, 3), (7, 2), (25, 1)] {
        let lambda = numerator as f64 / denominator as f64;
        let poisson = PoissonRatio::new(ratio(numerator, denominator));
        let draws = 100_000;
        let samples: Vec<u64> = (0..draws).map(|_| poisson.sample(&mut random)).collect();
        let zeros = samples.iter().filter(|&&count| count == 0).count() as u64;
        assert_frequency(zeros, draws, (-lambda).exp(), "P(0)");
        let (mean, variance) = moments(samples.iter().map(|&count| count as f64));
        assert!(
            (mean - lambda).abs() < 5.0 * (lambda / draws as f64).sqrt(),
            "mean {mean}"
        );
        assert!(
            (variance / lambda - 1.0).abs() < 0.03,
            "variance {variance}"
        );
    }
    assert_eq!(random.poisson_ratio(Ratio::ZERO), 0);
}

#[test]
fn discrete_laplace_decays_by_its_scale() {
    let mut random = rng(31);
    let laplace = DiscreteLaplace::new(ratio(2, 1)).unwrap();
    let draws = 200_000;
    let samples: Vec<i64> = (0..draws).map(|_| laplace.sample(&mut random)).collect();
    // P(k) = tanh(1 / (2 * scale)) * exp(-|k| / scale).
    let at_zero = (0.25f64).tanh();
    for k in [-2i64, -1, 0, 1, 3] {
        let hits = samples.iter().filter(|&&value| value == k).count() as u64;
        assert_frequency(
            hits,
            draws,
            at_zero * (-(k.abs() as f64) / 2.0).exp(),
            "laplace",
        );
    }
    assert!(DiscreteLaplace::new(Ratio::ZERO).is_none());
}

#[test]
fn discrete_gaussian_matches_its_probabilities() {
    for (numerator, denominator, mean) in [(1, 1, 0), (25, 4, -40), (100, 1, 1000)] {
        let variance = numerator as f64 / denominator as f64;
        let gaussian = DiscreteGaussian::new(mean, ratio(numerator, denominator)).unwrap();
        let mut random = rng(numerator as u128);
        let draws = 100_000;
        let samples: Vec<i64> = (0..draws).map(|_| gaussian.sample(&mut random)).collect();
        let weight = |k: i64| (-((k - mean) as f64).powi(2) / (2.0 * variance)).exp();
        let total: f64 = (mean - 200..=mean + 200).map(weight).sum();
        for offset in [0, 1, -2] {
            let hits = samples
                .iter()
                .filter(|&&value| value == mean + offset)
                .count() as u64;
            assert_frequency(hits, draws, weight(mean + offset) / total, "gaussian");
        }
        let (sample_mean, _) = moments(samples.iter().map(|&value| value as f64));
        assert!((sample_mean - mean as f64).abs() < 5.0 * (variance / draws as f64).sqrt());
    }
    assert_eq!(rng(0).discrete_gaussian(17, Ratio::ZERO), 17);
    assert!(DiscreteGaussian::new(0, ratio(u64::MAX, 1)).is_none());
}

#[test]
fn integer_categorical_hits_its_weights_exactly() {
    let table = IntegerCategorical::new(&[70, 25, 5, 0]).unwrap();
    let mut random = rng(41);
    let draws = 200_000;
    let mut counts = [0u64; 4];
    for _ in 0..draws {
        counts[table.sample(&mut random)] += 1;
    }
    assert_eq!(counts[3], 0);
    for (count, share) in counts.iter().zip([0.70, 0.25, 0.05]) {
        assert_frequency(*count, draws, share, "categorical");
    }
    // Weights near the top of u64 must not overflow the integer table.
    let wide = IntegerCategorical::new(&[u64::MAX, u64::MAX / 3, 1]).unwrap();
    let hits = (0..draws).filter(|_| wide.sample(&mut random) == 0).count() as u64;
    assert_frequency(hits, draws, 0.75, "wide");
    assert!(IntegerCategorical::new(&[]).is_none());
    assert!(IntegerCategorical::new(&[0, 0]).is_none());
    // The same seed picks the same entry every time.
    let seed = Seed::from_raw(5).child("palette");
    assert_eq!(seed.sample(&table), seed.sample(&table));
    let palette = WeightedDiscrete::new([("stone", 7.0), ("ore", 1.0)]).unwrap();
    for index in 0..100 {
        let voxel = seed.index(index);
        assert_eq!(
            voxel.sample_weighted(&palette),
            voxel.sample_weighted(&palette)
        );
    }
    assert!(Seed::from_raw(1).chance(Probability::ALWAYS));
}
