//! Drawing a [`Fixed`] from a stream or a seed.

use std::collections::HashSet;
use voxel_world::math::{Fixed, FixedPoint, Unit};
use voxel_world::random::seed::Seed;
use voxel_world::random::{Random, UniformFixed, UniformFixedRange};

fn stream(seed: u64) -> Random {
    Random::new(Seed::from_integer(seed))
}

// ---------------------------------------------------------------------------
// The half-open contract
// ---------------------------------------------------------------------------

#[test]
fn a_drawn_fraction_never_reaches_one() {
    let mut random = stream(1);

    for _ in 0..200_000 {
        let value = random.fixed();

        assert!(value >= Fixed::ZERO, "a fraction went below zero: {value}");
        assert!(value < Fixed::ONE, "a fraction reached one: {value}");
    }
}

#[test]
fn the_extreme_words_map_to_the_extreme_fractions() {
    assert_eq!(Fixed::fraction_from_word(0), Fixed::ZERO);
    assert_eq!(
        Fixed::fraction_from_word(u64::MAX),
        Fixed::ONE - Fixed::from_bits(1),
        "all ones is one step below one, not one"
    );
}

#[test]
fn truncating_is_why_the_range_stays_half_open() {
    // The implementation keeps the word's top bits rather than converting through
    // `Unit`. This is the reason: that conversion rounds, and the roundest possible
    // word carries all the way to one — which would break the half-open contract and,
    // with it, the exactness of every comparison against a chance.
    let roundest: u64 = u64::MAX;

    assert_eq!(
        Unit::from_word(roundest).to_fixed(),
        Fixed::ONE,
        "the Unit route reaches one, which is the trap"
    );
    assert!(
        Fixed::fraction_from_word(roundest) < Fixed::ONE,
        "taking bits cannot carry, so it cannot reach one"
    );
}

#[test]
fn a_fraction_is_one_draw_at_two_precisions() {
    // `fixed` and `unit` read the same word, so the coarser is the finer with its low
    // bits dropped. Two unrelated draws would not line up like this.
    for word in [0u64, 1, 7, 1 << 40, u64::MAX / 3, u64::MAX] {
        let fine: Unit = Unit::from_word(word);
        let coarse: Fixed = Fixed::fraction_from_word(word);

        assert_eq!(
            coarse.to_bits(),
            (fine.to_bits() >> 31) as i128,
            "the two disagree about the word {word}"
        );
    }
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn a_seed_always_gives_the_same_fraction() {
    let world = Seed::from_integer(5u64).child("moisture");

    for index in 0..500u64 {
        let place = world.index(index);

        assert_eq!(place.fixed(), place.fixed(), "a seed changed its mind");
        assert!(place.fixed() < Fixed::ONE);
    }

    // And different places disagree, which is the other half of being useful.
    let values: HashSet<i128> = (0..500u64)
        .map(|index| world.index(index).fixed().to_bits())
        .collect();

    assert!(values.len() > 450, "only {} distinct values in 500", values.len());
}

#[test]
fn a_seeds_fraction_is_its_unit_at_lower_precision() {
    let world = Seed::from_integer(9u64);

    for index in 0..200u64 {
        let place = world.index(index);

        assert_eq!(
            place.fixed().to_bits(),
            (place.unit().to_bits() >> 31) as i128,
            "the seed's two fractions came from different words"
        );
    }
}

#[test]
fn two_streams_from_one_seed_agree() {
    let seed = Seed::from_integer(21u64);
    let (mut left, mut right) = (Random::new(seed), Random::new(seed));

    for _ in 0..1_000 {
        assert_eq!(left.fixed(), right.fixed());
    }
}

// ---------------------------------------------------------------------------
// Uniformity
// ---------------------------------------------------------------------------

#[test]
fn fractions_fill_the_interval_evenly() {
    const BUCKETS: usize = 16;
    const DRAWS: u32 = 320_000;

    let mut counts = [0u32; BUCKETS];
    let mut random = stream(11);

    for _ in 0..DRAWS {
        let value = random.fixed();
        let bucket = ((value.to_bits() * BUCKETS as i128) >> 32) as usize;

        counts[bucket.min(BUCKETS - 1)] += 1;
    }

    let expected: f64 = f64::from(DRAWS) / BUCKETS as f64;
    let allowed: f64 = 5.0 * expected.sqrt();

    for (bucket, count) in counts.iter().enumerate() {
        assert!(
            (f64::from(*count) - expected).abs() < allowed,
            "bucket {bucket} held {count}, expected about {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// Ranges
// ---------------------------------------------------------------------------

#[test]
fn a_range_draw_stays_inside_its_bounds() {
    let (low, high) = (Fixed::from_integer(60), Fixed::from_integer(80));
    let mut random = stream(13);

    for _ in 0..100_000 {
        let value = random.fixed_range(low, high);

        assert!(value >= low && value < high, "{value} left [60, 80)");
    }
}

#[test]
fn a_range_draw_covers_its_bounds_evenly() {
    const BUCKETS: usize = 20;
    const DRAWS: u32 = 200_000;

    let (low, high) = (Fixed::from_integer(0), Fixed::from_integer(BUCKETS as i64));
    let mut counts = [0u32; BUCKETS];
    let mut random = stream(17);

    for _ in 0..DRAWS {
        let value = random.fixed_range(low, high);
        let bucket = (value.to_bits() >> 32) as usize;

        counts[bucket.min(BUCKETS - 1)] += 1;
    }

    let expected: f64 = f64::from(DRAWS) / BUCKETS as f64;
    let allowed: f64 = 5.0 * expected.sqrt();

    for (bucket, count) in counts.iter().enumerate() {
        assert!(
            (f64::from(*count) - expected).abs() < allowed,
            "bucket {bucket} held {count}, expected about {expected}"
        );
    }
}

#[test]
fn a_narrow_range_reaches_every_step_in_it() {
    // Five steps wide, so every value must turn up and none outside.
    let low = Fixed::from_bits(1_000);
    let high = Fixed::from_bits(1_005);

    let range = UniformFixedRange::new(low, high).expect("non-empty");
    let mut random = stream(19);

    let seen: HashSet<i128> = (0..5_000).map(|_| random.sample(&range).to_bits()).collect();

    assert_eq!(
        seen,
        (1_000..1_005).collect::<HashSet<i128>>(),
        "a five-step range should produce exactly those five values"
    );
}

#[test]
fn an_empty_or_inverted_range_is_refused() {
    let one = Fixed::ONE;

    assert!(UniformFixedRange::new(one, one).is_none(), "empty");
    assert!(UniformFixedRange::new(one, Fixed::ZERO).is_none(), "inverted");
    assert!(UniformFixedRange::new(Fixed::ZERO, one).is_some());

    // A single step is the narrowest range that holds anything.
    let range = UniformFixedRange::new(Fixed::ZERO, Fixed::from_bits(1)).expect("one step");

    assert_eq!(stream(23).sample(&range), Fixed::ZERO);
}

#[test]
fn the_widest_possible_range_works() {
    // MAX - MIN overflows an i128, which is why the span is counted as a u128. If it
    // were not, this would wrap and the draw would be nonsense.
    let range = UniformFixedRange::new(Fixed::MIN, Fixed::MAX).expect("non-empty");
    let mut random = stream(29);

    for _ in 0..10_000 {
        let value = random.sample(&range);

        assert!(value >= Fixed::MIN && value < Fixed::MAX);
    }

    assert_eq!(range.low(), Fixed::MIN);
    assert_eq!(range.high(), Fixed::MAX);
}

#[test]
fn a_seed_gives_one_value_in_a_range_for_ever() {
    let world = Seed::from_integer(31u64).child("trees");
    let (low, high) = (Fixed::from_integer(4), Fixed::from_integer(9));

    for index in 0..300u64 {
        let place = world.index(index);
        let height = place.fixed_range(low, high);

        assert_eq!(height, place.fixed_range(low, high), "not stable");
        assert!(height >= low && height < high);
    }
}

// ---------------------------------------------------------------------------
// Other layouts
// ---------------------------------------------------------------------------

#[test]
fn every_layout_can_be_drawn() {
    fn check<const N: u32>(seed: u64) {
        let mut random = stream(seed);

        for _ in 0..20_000 {
            let value: FixedPoint<N> = random.sample(&UniformFixed::<N>);

            assert!(value >= FixedPoint::<N>::ZERO, "below zero at {N} bits");
            assert!(value < FixedPoint::<N>::ONE, "reached one at {N} bits");
        }

        // The narrowest layout has two values, so both must appear.
        let range = UniformFixedRange::<N>::new(FixedPoint::<N>::ZERO, FixedPoint::<N>::ONE)
            .expect("non-empty");
        let mut counts = 0u32;

        for _ in 0..1_000 {
            if random.sample(&range) != FixedPoint::<N>::ZERO {
                counts += 1;
            }
        }

        assert!(counts > 0, "a range at {N} bits produced only its lowest value");
    }

    check::<1>(101);
    check::<16>(102);
    check::<32>(103);
    check::<64>(104);
}

#[test]
fn a_single_bit_layout_draws_only_zero_and_a_half() {
    // One fractional bit means exactly two fractions below one: 0 and 1/2.
    let mut random = stream(37);
    let seen: HashSet<i128> = (0..1_000)
        .map(|_| random.sample(&UniformFixed::<1>).to_bits())
        .collect();

    assert_eq!(seen, HashSet::from([0, 1]));
}

// ---------------------------------------------------------------------------
// Word cost
// ---------------------------------------------------------------------------

#[test]
fn the_plain_fraction_costs_exactly_one_word() {
    // Documented, and worth pinning: a sequence of these stays aligned with a
    // sequence of any other one-word draw, which rejection sampling would break.
    let seed = Seed::from_integer(41u64);

    let mut by_fraction = Random::new(seed);
    let mut by_word = Random::new(seed);

    for _ in 0..1_000 {
        let drawn = by_fraction.fixed();
        let expected = Fixed::fraction_from_word(by_word.next_u64());

        assert_eq!(drawn, expected);
    }
}
