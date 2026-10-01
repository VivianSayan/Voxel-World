//! The integer-backed unit interval.
//!
//! The claims under test, in order of how much they matter:
//!
//! 1. **The complement is exact** for every value — the reason the type exists.
//! 2. **`decide` lands on the asked-for probability exactly**, including at both
//!    ends, which is what the half-open draw buys.
//! 3. **Dyadic rationals are exact**, which the all-ones convention could not have
//!    given.
//! 4. The combining rules agree with probability theory, checked against exact
//!    rational arithmetic rather than against themselves.

use std::collections::BTreeMap;
use voxel_world::random::distributions::Distribution;
use voxel_world::random::seed::Seed;
use voxel_world::random::Random;
use voxel_world::math::BigUint;
use std::str::FromStr;
use voxel_world::units::{Probability, Ratio, UniformUnit, Unit};

/// A spread of values on the awkward boundaries, plus a fixed pseudorandom set.
fn probes() -> Vec<Unit> {
    let mut values: Vec<Unit> = vec![
        Unit::ZERO,
        Unit::STEP,
        Unit::HALF,
        Unit::ALMOST_ONE,
        Unit::ONE,
        Unit::one_in(3),
        Unit::one_in(10),
        Unit::one_in(u64::MAX),
        Unit::out_of(1, 4),
        Unit::out_of(3, 4),
        Unit::out_of(999_999, 1_000_000),
    ];

    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for _ in 0..40 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(Unit::from_bits(state >> 1).expect("63 bits is in range"));
    }

    values
}

// ===========================================================================
// The headline: an exact complement
// ===========================================================================

#[test]
fn the_complement_is_exact_for_every_value() {
    for value in probes() {
        assert_eq!(
            value.complement().complement(),
            value,
            "complementing {value} twice should return it"
        );
        assert_eq!(
            value.to_bits() + value.complement().to_bits(),
            Unit::STEPS,
            "a value and its complement should sum to exactly one"
        );
    }

    assert_eq!(Unit::ZERO.complement(), Unit::ONE);
    assert_eq!(Unit::ONE.complement(), Unit::ZERO);
    assert_eq!(Unit::HALF.complement(), Unit::HALF);
}

/// The same round trip in `f64`, which is what this type is for.
#[test]
fn f64_loses_what_the_integer_keeps() {
    // A probability far below an f64's resolution *next to one*.
    let tiny: Unit = Unit::one_in(1_000_000_000_000_000_000);

    assert!(!tiny.is_zero(), "the integer holds it");
    assert_eq!(tiny.complement().complement(), tiny, "and gets it back");

    // f64 cannot: 1 - 1e-18 rounds to 1, and the value is gone for good.
    let as_float: f64 = 1e-18;
    assert_eq!(1.0 - as_float, 1.0);
    assert_eq!(1.0 - (1.0 - as_float), 0.0);

    // Through Probability, the same loss.
    let lost: Probability = tiny.to_probability();
    assert_eq!(lost.complement().complement().value(), 0.0);
}

// ===========================================================================
// Dyadic exactness, which the all-ones convention could not give
// ===========================================================================

#[test]
fn dyadic_rationals_are_exact() {
    // A half, a quarter, an eighth — all the way down to the step.
    for power in 0..63u32 {
        let divisor: u64 = 1 << power;
        let value: Unit = Unit::one_in(divisor);

        assert_eq!(
            value.to_bits() * divisor,
            Unit::STEPS,
            "1/2^{power} should be exact"
        );

        // And it round-trips through Ratio without moving.
        assert_eq!(
            Unit::from_ratio_capped(Ratio::new(1, divisor).expect("valid")),
            value,
            "1/2^{power} through Ratio"
        );
    }

    // Halving repeatedly is exact, and lands on the step after 63 halvings.
    let mut value: Unit = Unit::ONE;
    for _ in 0..63 {
        value = value.and(Unit::HALF);
    }
    assert_eq!(value, Unit::STEP);
    assert_eq!(value.and(Unit::HALF), Unit::ZERO, "and then underflows to zero");
}

#[test]
fn a_half_is_exactly_a_half() {
    assert_eq!(Unit::HALF.to_bits() * 2, Unit::STEPS);
    assert_eq!(Unit::HALF.to_f64(), 0.5);
    assert_eq!(Unit::HALF.and(Unit::HALF), Unit::one_in(4));
    assert_eq!(Unit::HALF.or(Unit::HALF), Unit::out_of(3, 4));
    assert_eq!(Unit::HALF.complement(), Unit::HALF);
}

// ===========================================================================
// Deciding
// ===========================================================================

#[test]
fn a_drawn_value_is_never_exactly_one() {
    // What makes `decide` exact: the draw is half-open even though the type is not.
    let mut seed: Seed = Seed::from_integer(12345u64);

    for _ in 0..100_000 {
        let drawn: Unit = Unit::from_seed(seed);

        assert!(!drawn.is_one(), "a draw reached one: {drawn}");
        assert!(drawn <= Unit::ALMOST_ONE);
        seed = seed.advance();
    }
}

#[test]
fn the_certain_and_the_impossible_are_exactly_that() {
    let mut seed: Seed = Seed::from_integer(777u64);

    for _ in 0..50_000 {
        assert!(Unit::ONE.decide(seed), "a certainty must always happen");
        assert!(!Unit::ZERO.decide(seed), "an impossibility never can");
        seed = seed.advance();
    }
}

#[test]
fn deciding_lands_on_the_asked_for_frequency() {
    const DRAWS: u32 = 200_000;

    for (chance, expected) in [
        (Unit::HALF, 0.5),
        (Unit::one_in(4), 0.25),
        (Unit::one_in(10), 0.1),
        (Unit::out_of(9, 10), 0.9),
        (Unit::out_of(1, 100), 0.01),
    ] {
        let mut seed: Seed = Seed::from_integer(0xABCDEFu64);
        let mut hits: u32 = 0;

        for _ in 0..DRAWS {
            if chance.decide(seed) {
                hits += 1;
            }
            seed = seed.advance();
        }

        let frequency: f64 = f64::from(hits) / f64::from(DRAWS);
        // Four standard deviations of a binomial, which is generous but still
        // catches any real bias.
        let deviation: f64 = 4.0 * (expected * (1.0 - expected) / f64::from(DRAWS)).sqrt();

        assert!(
            (frequency - expected).abs() < deviation,
            "chance {expected}: saw {frequency}, outside {deviation}"
        );
    }
}

#[test]
fn a_uniform_draw_covers_the_range_evenly() {
    let mut seed: Seed = Seed::from_integer(31337u64);
    let mut buckets: [u32; 8] = [0; 8];
    const DRAWS: u32 = 160_000;

    for _ in 0..DRAWS {
        let drawn: Unit = Unit::from_seed(seed);
        buckets[drawn.index_of(8) as usize] += 1;
        seed = seed.advance();
    }

    let expected: f64 = f64::from(DRAWS) / 8.0;
    for (index, count) in buckets.iter().enumerate() {
        let deviation: f64 = (f64::from(*count) - expected).abs();
        assert!(
            deviation < 5.0 * expected.sqrt(),
            "bucket {index} held {count}, expected about {expected}"
        );
    }
}

#[test]
fn the_distribution_draws_uniformly_too() {
    let mut source = Random::new(Seed::from_integer(2024u64));

    // One word per draw, with no rejection, so a sequence consumes a predictable
    // number of words whatever it draws.
    let drawn: Vec<Unit> = (0..20_000).map(|_| UniformUnit.sample(&mut source)).collect();

    assert!(drawn.iter().all(|value| !value.is_one()), "a draw is half-open");
    assert!(
        drawn.iter().any(|value| *value != drawn[0]),
        "the draws should vary"
    );

    // And the mean sits where a uniform draw's should.
    let mean: f64 = drawn.iter().map(|value| value.to_f64()).sum::<f64>() / drawn.len() as f64;
    assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");

    // Deciding through a source agrees with deciding through a seed.
    let mut source = Random::new(Seed::from_integer(99u64));
    let hits: usize = (0..20_000)
        .filter(|_| Unit::HALF.decide_from(&mut source))
        .count();
    assert!((hits as f64 / 20_000.0 - 0.5).abs() < 0.02, "{hits} hits");
}

// ===========================================================================
// Probability theory, against exact rational arithmetic
// ===========================================================================

/// The correctly-rounded product, worked out here in `u128` so it shares no code
/// with the implementation.
fn exact_product(left: Unit, right: Unit) -> u64 {
    let product: u128 = left.to_bits() as u128 * right.to_bits() as u128;
    let quotient: u128 = product >> 63;
    let remainder: u128 = product & ((1u128 << 63) - 1);
    let half: u128 = 1u128 << 62;

    if remainder > half || (remainder == half && quotient & 1 == 1) {
        (quotient + 1) as u64
    } else {
        quotient as u64
    }
}

#[test]
fn and_is_the_product() {
    // Against the product worked out independently, for every probe pair.
    for left in probes() {
        for right in probes() {
            assert_eq!(
                left.and(right).to_bits(),
                exact_product(left, right),
                "{left} and {right}"
            );
        }
    }

    // Where both inputs are dyadic the whole thing is exact, with no rounding at
    // any step — which is what the power-of-two denominator buys.
    assert_eq!(Unit::HALF.and(Unit::HALF), Unit::one_in(4));
    assert_eq!(Unit::one_in(4).and(Unit::one_in(8)), Unit::one_in(32));
    assert_eq!(Unit::out_of(3, 4).and(Unit::HALF), Unit::out_of(3, 8));

    // Where an input is not dyadic, `and` rounds its already-rounded inputs, so it
    // lands within one step of the directly-built fraction rather than on it. That
    // is double rounding, and it is inherent rather than a defect.
    let sixth: Unit = Unit::HALF.and(Unit::one_in(3));
    let difference: u64 = sixth.to_bits().abs_diff(Unit::out_of(1, 6).to_bits());
    assert!(difference <= 1, "off by {difference} steps");

    // The identities.
    for value in probes() {
        assert_eq!(value.and(Unit::ONE), value, "anding with certainty");
        assert_eq!(value.and(Unit::ZERO), Unit::ZERO, "anding with impossibility");
        assert!(value.and(Unit::HALF) <= value, "and never increases");
    }
}

#[test]
fn or_is_the_inclusive_union() {
    // Dyadic inputs are exact: 1/4 or 1/4 = 7/16, 1/2 or 1/2 = 3/4.
    assert_eq!(Unit::one_in(4).or(Unit::one_in(4)), Unit::out_of(7, 16));
    assert_eq!(Unit::HALF.or(Unit::HALF), Unit::out_of(3, 4));

    // A non-dyadic input brings double rounding, so 1/2 or 1/3 lands within a step
    // of 2/3 rather than exactly on it.
    let two_thirds: Unit = Unit::HALF.or(Unit::one_in(3));
    assert!(two_thirds.to_bits().abs_diff(Unit::out_of(2, 3).to_bits()) <= 1);

    for value in probes() {
        assert_eq!(value.or(Unit::ZERO), value, "oring with impossibility");
        assert_eq!(value.or(Unit::ONE), Unit::ONE, "oring with certainty");
        assert!(value.or(Unit::HALF) >= value, "or never decreases");
    }
}

/// De Morgan, which ties `and`, `or` and `complement` together. It holds here as an
/// exact equality, which it would not in floating point.
#[test]
fn de_morgan_holds_exactly() {
    for left in probes() {
        for right in probes() {
            assert_eq!(
                left.or(right),
                left.complement().and(right.complement()).complement(),
                "not(not a and not b) should be a or b"
            );
            assert_eq!(
                left.and(right),
                left.complement().or(right.complement()).complement(),
                "not(not a or not b) should be a and b"
            );
            assert_eq!(left.and(right), right.and(left), "and commutes");
            assert_eq!(left.or(right), right.or(left), "or commutes");
        }
    }
}

#[test]
fn in_any_of_is_the_repeated_or() {
    for chance in [Unit::HALF, Unit::one_in(6), Unit::out_of(2, 7)] {
        let mut accumulated: Unit = Unit::ZERO;

        for tries in 0..12u32 {
            // `in_any_of` squares the complement, where the loop multiplies it one
            // at a time, so the two accumulate rounding differently. A step per
            // multiplication is the bound.
            let drift: u64 = chance
                .in_any_of(tries)
                .to_bits()
                .abs_diff(accumulated.to_bits());

            assert!(
                drift <= u64::from(tries),
                "{chance} in any of {tries} drifted {drift} steps"
            );
            accumulated = accumulated.or(chance);
        }
    }

    // Two coin flips give three quarters, exactly.
    assert_eq!(Unit::HALF.in_any_of(2), Unit::out_of(3, 4));
    assert_eq!(Unit::HALF.in_any_of(0), Unit::ZERO, "no tries, no chance");
    assert_eq!(Unit::ONE.in_any_of(5), Unit::ONE);
    assert_eq!(Unit::ZERO.in_any_of(5), Unit::ZERO);
}

#[test]
fn addition_is_for_disjoint_events_and_says_when_it_is_not() {
    // Two disjoint quarters make a half, exactly, because both are dyadic.
    assert_eq!(
        Unit::one_in(4).checked_add(Unit::one_in(4)),
        Some(Unit::HALF)
    );

    // Two thirds is within a step: each third was rounded before being added.
    let two_thirds: Unit = Unit::one_in(3).checked_add(Unit::one_in(3)).expect("under one");
    assert!(two_thirds.to_bits().abs_diff(Unit::out_of(2, 3).to_bits()) <= 1);

    // Three halves is not a probability, and it refuses rather than saturating.
    assert_eq!(Unit::HALF.checked_add(Unit::out_of(3, 4)), None);
    assert_eq!(Unit::ONE.checked_add(Unit::STEP), None);
    assert_eq!(Unit::ONE.checked_add(Unit::ZERO), Some(Unit::ONE));

    // Saturating is there when a cap is genuinely wanted.
    assert_eq!(Unit::HALF.saturating_add(Unit::out_of(3, 4)), Unit::ONE);
    assert_eq!(Unit::ONE.saturating_add(Unit::ONE), Unit::ONE);

    // And note it differs from `or`, which is the whole reason they are named.
    assert_ne!(
        Unit::HALF.checked_add(Unit::HALF).unwrap(),
        Unit::HALF.or(Unit::HALF),
        "disjoint and independent are not the same combination"
    );
    assert_eq!(Unit::HALF.checked_add(Unit::HALF).unwrap(), Unit::ONE);
    assert_eq!(Unit::HALF.or(Unit::HALF), Unit::out_of(3, 4));
}

#[test]
fn subtraction_and_conditional_division() {
    assert_eq!(Unit::out_of(3, 4).checked_sub(Unit::HALF), Some(Unit::one_in(4)));
    assert_eq!(Unit::HALF.checked_sub(Unit::out_of(3, 4)), None);
    assert_eq!(Unit::HALF.saturating_sub(Unit::ONE), Unit::ZERO);

    // P(A | B) = P(A and B) / P(B). A quarter given a half is a half.
    assert_eq!(Unit::one_in(4).checked_div(Unit::HALF), Some(Unit::HALF));
    assert_eq!(Unit::HALF.checked_div(Unit::HALF), Some(Unit::ONE));

    // Not a probability: the event is not contained in the condition.
    assert_eq!(Unit::HALF.checked_div(Unit::one_in(4)), None);
    assert_eq!(Unit::HALF.checked_div(Unit::ZERO), None);
}

#[test]
fn powers_agree_with_repeated_and() {
    for value in [Unit::HALF, Unit::one_in(3), Unit::out_of(7, 8)] {
        let mut accumulated: Unit = Unit::ONE;

        for exponent in 0..10u32 {
            assert_eq!(value.pow(exponent), accumulated, "{value}^{exponent}");
            accumulated = accumulated.and(value);
        }
    }

    assert_eq!(Unit::HALF.pow(63), Unit::STEP);
    assert_eq!(Unit::HALF.pow(64), Unit::ZERO, "below the step is zero");
}

// ===========================================================================
// Scaling to integers
// ===========================================================================

#[test]
fn scaling_rounds_sensibly() {
    assert_eq!(Unit::HALF.scale_rounded(10), 5);
    assert_eq!(Unit::ONE.scale_rounded(10), 10);
    assert_eq!(Unit::ZERO.scale_rounded(10), 0);
    assert_eq!(Unit::one_in(3).scale_rounded(10), 3);
    assert_eq!(Unit::out_of(2, 3).scale_rounded(10), 7);

    // Floor and ceiling bracket it.
    for value in probes() {
        for count in [1u64, 2, 7, 100, 1_000_000, u32::MAX as u64] {
            let floor: u64 = value.scale_floor(count);
            let rounded: u64 = value.scale_rounded(count);
            let ceiling: u64 = value.scale_ceil(count);

            assert!(floor <= rounded && rounded <= ceiling, "{value} x {count}");
            assert!(ceiling <= count, "scaling cannot exceed the count");
            assert!(ceiling - floor <= 1, "floor and ceiling differ by at most one");
        }
    }

    // An exact multiple has floor, round and ceiling all equal.
    assert_eq!(Unit::HALF.scale_floor(8), 4);
    assert_eq!(Unit::HALF.scale_ceil(8), 4);
}

#[test]
fn ties_round_to_even() {
    // 1/2 of an odd count is an exact tie. 5/2 = 2.5 -> 2 (even), 7/2 = 3.5 -> 4.
    assert_eq!(Unit::HALF.scale_rounded(5), 2, "2.5 rounds to even 2");
    assert_eq!(Unit::HALF.scale_rounded(7), 4, "3.5 rounds to even 4");
    assert_eq!(Unit::HALF.scale_rounded(9), 4, "4.5 rounds to even 4");
    assert_eq!(Unit::HALF.scale_rounded(11), 6, "5.5 rounds to even 6");
}

/// The classic off-by-one a closed unit interval invites: `floor(1.0 × count)` is
/// `count`, one past the end of a table.
#[test]
fn indexing_is_always_in_range() {
    for value in probes() {
        for count in [1u64, 2, 3, 8, 1000, u32::MAX as u64] {
            let index: u64 = value.index_of(count);

            assert!(index < count, "{value}.index_of({count}) gave {index}");
        }
    }

    // Specifically at one, where a plain floor would overrun.
    assert_eq!(Unit::ONE.index_of(4), 3);
    assert_eq!(Unit::ONE.scale_floor(4), 4, "the unclamped floor does overrun");
    assert_eq!(Unit::ZERO.index_of(4), 0);
    assert_eq!(Unit::HALF.index_of(4), 2);
    assert_eq!(Unit::ZERO.index_of(0), 0, "a zero count gives zero");
}

#[test]
fn between_is_closed_at_both_ends() {
    assert_eq!(Unit::ZERO.between(-5.0, 5.0), -5.0);
    assert_eq!(Unit::ONE.between(-5.0, 5.0), 5.0);
    assert_eq!(Unit::HALF.between(0.0, 10.0), 5.0);
    assert_eq!(Unit::HALF.between(3.0, 3.0), 3.0, "equal bounds");

    for value in probes() {
        let spread: f64 = value.between(-2.5, 7.5);
        assert!((-2.5..=7.5).contains(&spread), "{value} gave {spread}");
    }
}

#[test]
fn blending_stays_in_range_and_hits_both_ends() {
    for a in probes() {
        for b in probes() {
            assert_eq!(a.blend(b, Unit::ZERO), a, "no weight keeps the first");
            assert_eq!(a.blend(b, Unit::ONE), b, "full weight takes the second");

            let middle: Unit = a.blend(b, Unit::HALF);
            assert!(middle >= a.min(b) && middle <= a.max(b), "a blend lies between");
            assert_eq!(a.blend(a, Unit::HALF), a, "blending a value with itself");
        }
    }

    assert_eq!(Unit::ZERO.blend(Unit::ONE, Unit::HALF), Unit::HALF);
}

// ===========================================================================
// Conversions
// ===========================================================================

#[test]
fn round_trips_through_f64() {
    // An f64 above about `2^-11` has all its significant bits at or above `2^-63`,
    // so it converts and returns exactly. Below that its own bits run finer than
    // this type's step and the low ones are lost — which is the documented trade,
    // not a defect: `1e-10` as an f64 carries bits down to `2^-86`.
    for value in [0.0, 1.0, 0.5, 0.25, 0.1, 0.9, 0.999_999_999, 0.000_976_562_5] {
        let unit: Unit = Unit::new(value).expect("in range");

        assert_eq!(unit.to_f64(), value, "round trip of {value}");
    }

    // Below the step's reach, the conversion rounds rather than round-trips.
    let small: f64 = 1e-10;
    let rounded: Unit = Unit::new(small).expect("in range");
    // The error is bounded by half a step, which is `2^-64`, about `5.4e-20`.
    assert!(
        (rounded.to_f64() - small).abs() < 2f64.powi(-64),
        "the rounding must stay within half a step"
    );
    assert_ne!(rounded.to_f64(), small, "and it is genuinely not exact");

    assert_eq!(Unit::new(-0.1), None);
    assert_eq!(Unit::new(1.1), None);
    assert_eq!(Unit::new(f64::NAN), None);
    assert_eq!(Unit::clamped(-5.0), Unit::ZERO);
    assert_eq!(Unit::clamped(5.0), Unit::ONE);
    assert_eq!(Unit::clamped(f64::NAN), Unit::ZERO);
}

#[test]
fn round_trips_through_probability() {
    for value in [0.0, 1.0, 0.5, 0.25, 0.75, 0.125] {
        let chance: Probability = Probability::new(value).expect("in range");
        let unit: Unit = Unit::from_probability(chance);

        assert_eq!(unit.to_probability().value(), value);
        assert_eq!(Unit::from(chance), unit, "the From impl agrees");
    }
}

#[test]
fn converts_from_ratio_with_a_cap() {
    assert_eq!(
        Unit::from_ratio_capped(Ratio::new(1, 2).unwrap()),
        Unit::HALF
    );
    assert_eq!(
        Unit::from_ratio_capped(Ratio::new(3, 4).unwrap()),
        Unit::out_of(3, 4)
    );

    // Above one, capped rather than refused — the name says so.
    assert_eq!(
        Unit::from_ratio_capped(Ratio::new(5, 2).unwrap()),
        Unit::ONE
    );
    assert_eq!(
        Unit::from_ratio_capped(Ratio::new(1, 1).unwrap()),
        Unit::ONE
    );
    assert_eq!(
        Unit::try_from(Ratio::new(1, 3).unwrap()),
        Ok(Unit::one_in(3)),
        "the default conversion is fallible now"
    );
}

#[test]
fn builds_from_counts() {
    assert_eq!(Unit::one_in(1), Unit::ONE);
    assert_eq!(Unit::one_in(2), Unit::HALF);
    assert_eq!(Unit::one_in(0), Unit::ZERO, "one in nothing is nothing");
    assert_eq!(Unit::out_of(0, 5), Unit::ZERO);
    assert_eq!(Unit::out_of(5, 5), Unit::ONE);
    assert_eq!(Unit::out_of(7, 5), Unit::ONE, "more hits than trials caps");
    assert_eq!(Unit::out_of(1, 0), Unit::ZERO, "no trials gives nothing");

    // The smallest and largest non-trivial values.
    assert_eq!(Unit::one_in(u64::MAX).to_bits(), 1);
    assert!(!Unit::one_in(u64::MAX).is_zero());
}

// ===========================================================================
// Structure
// ===========================================================================

/// A total order, which `Probability` cannot offer because an `f64` carries NaN.
#[test]
fn the_order_is_total() {
    let mut map: BTreeMap<Unit, &str> = BTreeMap::new();

    map.insert(Unit::HALF, "half");
    map.insert(Unit::ZERO, "never");
    map.insert(Unit::ONE, "always");
    map.insert(Unit::one_in(3), "a third");

    assert_eq!(
        map.values().copied().collect::<Vec<_>>(),
        vec!["never", "a third", "half", "always"]
    );

    for value in probes() {
        assert_eq!(value, value, "every value equals itself");
        assert_eq!(value.cmp(&value), std::cmp::Ordering::Equal);
    }

    // Sorting agrees with the f64 order.
    let mut values: Vec<Unit> = probes();
    values.sort();
    for pair in values.windows(2) {
        assert!(pair[0].to_f64() <= pair[1].to_f64());
    }
}

#[test]
fn raw_bits_are_validated() {
    assert_eq!(Unit::from_bits(0), Some(Unit::ZERO));
    assert_eq!(Unit::from_bits(Unit::STEPS), Some(Unit::ONE));
    assert_eq!(Unit::from_bits(Unit::STEPS + 1), None, "past one is refused");
    assert_eq!(Unit::from_bits(u64::MAX), None, "all ones is not a value");

    assert_eq!(Unit::from_bits_clamped(u64::MAX), Unit::ONE);
    assert_eq!(Unit::from_bits_clamped(Unit::STEPS / 2), Unit::HALF);

    // Round trip through the raw count.
    for value in probes() {
        assert_eq!(Unit::from_bits(value.to_bits()), Some(value));
    }
}

// ===========================================================================
// Conversion is lossy, and the documentation now says so
// ===========================================================================

/// A `Probability` far below the `2^-63` grid has nowhere to land but zero.
///
/// The old documentation claimed this conversion was exact because an `f64` has 53
/// significant bits and a `Unit` has 63 fractional ones. That reasoning confuses
/// *significant* bits with *absolute* position: an `f64`'s bits sit wherever its
/// exponent puts them, and for `1e-30` all of them are beneath the grid.
#[test]
fn a_tiny_probability_rounds_away_to_zero() {
    let tiny: Probability = Probability::clamped(1e-30);
    let unit: Unit = Unit::from_probability(tiny);

    assert_eq!(unit, Unit::ZERO, "1e-30 is far below the 2^-63 step");
    assert!(tiny.value() > 0.0, "but it was genuinely non-zero going in");

    // The boundary: half a step is the largest value that still rounds to zero.
    let half_step: f64 = 2f64.powi(-64);
    assert_eq!(Unit::clamped(half_step * 0.9), Unit::ZERO);
    assert_eq!(Unit::clamped(half_step * 1.1), Unit::STEP);

    // And at or above 2^-11 an f64's last bit sits on the grid, so it is exact.
    for exponent in -11..=0 {
        let value: f64 = 2f64.powi(exponent);
        assert_eq!(
            Unit::clamped(value).to_f64(),
            value,
            "2^{exponent} should round-trip exactly"
        );
    }

    // Just below that, it need not.
    let fine: f64 = 1e-10;
    assert_ne!(Unit::clamped(fine).to_f64(), fine);
}

/// Distinct `Unit`s near one share an `f64`, so `to_f64` cannot be a round trip.
#[test]
fn conversion_to_f64_loses_neighbours_near_one() {
    let top: Unit = Unit::ALMOST_ONE;
    let below: Unit = Unit::from_bits(top.to_bits() - 1).expect("in range");

    assert_ne!(top, below, "they are different values");
    assert_eq!(top.to_f64(), below.to_f64(), "but the same f64");
    assert_eq!(top.to_f64(), 1.0, "and that f64 is exactly one");

    // Exactly 512 of them collapse to `1.0`: an f64's last step below one is
    // `2^-53`, so everything within half of that — `2^-54`, or `2^63 x 2^-54 = 512`
    // of this type's steps — rounds up to it.
    let shared: usize = (0..3000)
        .map(|offset| Unit::from_bits(Unit::STEPS - 1 - offset).unwrap().to_f64())
        .filter(|value| *value == 1.0)
        .count();
    assert_eq!(shared, 512, "512 distinct Units share the f64 1.0");

    // The same loss through Probability.
    assert_eq!(top.to_probability().value(), below.to_probability().value());

    // What survives: the raw count, and the exact decimal.
    assert_ne!(top.to_bits(), below.to_bits());
    assert_ne!(top.to_string(), below.to_string());
}

// ===========================================================================
// Exact decimal display
// ===========================================================================

#[test]
fn displays_the_exact_decimal_value() {
    assert_eq!(Unit::ZERO.to_string(), "0");
    assert_eq!(Unit::ONE.to_string(), "1");
    assert_eq!(Unit::HALF.to_string(), "0.5");
    assert_eq!(Unit::one_in(4).to_string(), "0.25");
    assert_eq!(Unit::one_in(8).to_string(), "0.125");
    assert_eq!(Unit::out_of(3, 4).to_string(), "0.75");

    // The step is 5^63 / 10^63, so it has exactly 63 decimal places and every one
    // of them is printed.
    let step: String = Unit::STEP.to_string();
    assert_eq!(
        step,
        "0.000000000000000000108420217248550443400745280086994171142578125"
    );
    assert_eq!(step.len() - 2, 63, "63 fractional digits");
    assert_ne!(step, Unit::ZERO.to_string());

    // And the value just below one, which an f64 could not have told from one.
    assert!(Unit::ALMOST_ONE.to_string().starts_with("0.99999999999999999989"));
    assert_ne!(Unit::ALMOST_ONE.to_string(), Unit::ONE.to_string());
}

/// The property the old implementation could not deliver: distinct values print
/// distinctly, everywhere, including where `to_f64` collapses them.
#[test]
fn neighbouring_values_never_print_the_same() {
    let interesting: [u64; 7] = [
        0,
        1,
        2,
        Unit::STEPS / 2,
        Unit::STEPS - 2,
        Unit::STEPS - 1,
        Unit::STEPS,
    ];

    for count in interesting {
        for offset in 0..3u64 {
            let Some(lower) = count.checked_sub(offset).and_then(Unit::from_bits) else {
                continue;
            };
            let Some(upper) = count
                .checked_sub(offset)
                .and_then(|value| value.checked_add(1))
                .and_then(Unit::from_bits)
            else {
                continue;
            };

            assert_ne!(
                lower.to_string(),
                upper.to_string(),
                "{} and {} printed the same",
                lower.to_bits(),
                upper.to_bits()
            );
        }
    }
}

#[test]
fn a_precision_gives_a_short_form() {
    assert_eq!(format!("{:.4}", Unit::one_in(3)), "0.3333");
    assert_eq!(format!("{:.2}", Unit::HALF), "0.50");
    assert_eq!(format!("{:.0}", Unit::HALF), "0", "a tie goes to even");
    assert_eq!(format!("{:.0}", Unit::out_of(3, 4)), "1", "0.75 rounds up");
    assert_eq!(format!("{:.3}", Unit::ZERO), "0.000");
    assert_eq!(format!("{:.3}", Unit::ONE), "1.000");

    // Rounding up cascades out of the fraction entirely.
    assert_eq!(format!("{:.3}", Unit::ALMOST_ONE), "1.000");
    // Not one: `1 - 2^-63` differs from one by about `1.08e-19`, which is more
    // than half of the last place at 19 digits, so it stays below.
    assert_eq!(format!("{:.19}", Unit::ALMOST_ONE), "0.9999999999999999999");

    // Asking for more places than the value has pads rather than inventing digits.
    assert_eq!(format!("{:.5}", Unit::HALF), "0.50000");
    assert_eq!(format!("{:.70}", Unit::STEP).len(), 72);
}

/// Every printed value must read back as the number it came from, which is what
/// "exact" has to mean.
#[test]
fn the_printed_decimal_is_the_true_value() {
    for value in probes() {
        let text: String = value.to_string();

        // Rebuild the fraction from the digits: 0.d1d2... = (sum of d_i x 10^-i),
        // and compare against k / 2^63 by cross-multiplying in exact integers.
        let Some((whole, fraction)) = text.split_once('.') else {
            // No point means zero or one, both of which are exact by inspection.
            assert!(text == "0" || text == "1");
            assert_eq!(value.to_bits(), if text == "1" { Unit::STEPS } else { 0 });
            continue;
        };

        assert_eq!(whole, "0", "only one has a whole part, and it has no fraction");
        assert!(fraction.len() <= 63, "at most 63 places");
        assert!(!fraction.ends_with('0'), "trailing zeros should be trimmed");

        // value = k / 2^63, and the text is n / 10^len. They agree exactly when
        // k * 10^len == n * 2^63. Both sides are built with big integers so nothing
        // overflows on the way.
        let digits: BigUint = BigUint::from_str(fraction).expect("digits");
        let scale: BigUint = BigUint::from(10u64).pow(fraction.len() as u64);
        let left: BigUint = BigUint::from(value.to_bits()).product(&scale);
        let right: BigUint = digits.product(&BigUint::from(Unit::STEPS));

        assert_eq!(left, right, "{value} does not print its own value");
    }
}

// ===========================================================================
// Integration with Seed, Random and the samplers
// ===========================================================================

/// Every path from a random word to a `Unit` goes through `Unit::from_word`, so
/// they all agree. If they did not, a replay would depend on which one the caller
/// happened to use.
#[test]
fn every_draw_path_agrees() {
    use voxel_world::random::Random;

    let seed: Seed = Seed::from_integer(4242u64);

    // Seed's one-off draw, the distribution, and the trait method — same word, same
    // value.
    let mut source = Random::new(seed);
    let by_trait: Unit = source.unit();

    let mut source = Random::new(seed);
    let by_distribution: Unit = UniformUnit.sample(&mut source);

    assert_eq!(by_trait, by_distribution, "trait and distribution agree");

    // And `Random::unit` is the same as the trait method.
    let mut source = Random::new(seed);
    assert_eq!(source.unit(), by_trait);
}

/// `Seed::unit` and `Seed::unit_f64` read the *same word*, so one is the other at
/// full precision. Before this, `Unit::from_seed` folded the seed directly and
/// produced an unrelated draw.
#[test]
fn the_integer_and_float_fractions_are_the_same_draw() {
    let mut seed: Seed = Seed::from_integer(90210u64);

    for _ in 0..10_000 {
        let integer: Unit = seed.unit();
        let float: f64 = seed.unit_f64();

        // `unit_f64` keeps the top 53 bits; `unit` keeps the top 63. Dropping ten
        // more bits from the second must land exactly on the first.
        let truncated: f64 = (integer.to_bits() >> 10) as f64 / (1u64 << 53) as f64;

        assert_eq!(truncated, float, "the two fractions disagree");
        assert_eq!(Unit::from_seed(seed), integer, "and `from_seed` agrees too");

        seed = seed.advance();
    }
}

#[test]
fn deciding_agrees_across_every_entry_point() {
    use voxel_world::random::distributions::Bernoulli;

    let mut seed: Seed = Seed::from_integer(5150u64);

    for chance in [Unit::ZERO, Unit::HALF, Unit::one_in(3), Unit::out_of(9, 10), Unit::ONE] {
        for _ in 0..2_000 {
            let direct: bool = chance.decide(seed);

            assert_eq!(seed.chance_unit(chance), direct, "Seed::chance_unit");
            assert_eq!(
                seed.sample(&Bernoulli::at(chance)),
                direct,
                "Bernoulli::at"
            );

            seed = seed.advance();
        }
    }
}

/// A `Bernoulli` built from a `Unit` keeps the chance exactly; built from a
/// `Probability` it keeps whatever the `f64` could hold.
#[test]
fn bernoulli_keeps_a_unit_chance_exactly() {
    use voxel_world::random::distributions::Bernoulli;

    // A chance an f64 cannot separate from certainty.
    let almost: Unit = Unit::ALMOST_ONE;
    assert_eq!(almost.to_f64(), 1.0, "an f64 sees this as one");

    assert_eq!(Bernoulli::at(almost).chance(), almost, "the Unit form keeps it");
    assert_eq!(
        Bernoulli::new(almost.to_probability()).chance(),
        Unit::ONE,
        "the Probability form rounds it to certainty"
    );

    // The two agree wherever the f64 can hold the value.
    for chance in [Unit::ZERO, Unit::HALF, Unit::one_in(4), Unit::ONE] {
        assert_eq!(
            Bernoulli::new(chance.to_probability()).chance(),
            chance,
            "{chance} survives the round trip"
        );
    }
}

/// `Probability::decide` now routes through `Unit`, so it must agree with
/// `Bernoulli`, which is what `Seed::chance` uses.
#[test]
fn probability_decide_agrees_with_bernoulli() {
    let mut seed: Seed = Seed::from_integer(1848u64);

    for value in [0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9] {
        let chance: Probability = Probability::new(value).expect("in range");

        for _ in 0..1_000 {
            assert_eq!(
                chance.decide(seed),
                seed.chance(chance),
                "the two coins disagree at {value}"
            );
            seed = seed.advance();
        }
    }
}

#[test]
fn bernoulli_still_lands_on_its_frequency() {
    use voxel_world::random::distributions::Bernoulli;
    use voxel_world::random::Random;

    const DRAWS: u32 = 200_000;

    for (chance, expected) in [
        (Unit::HALF, 0.5),
        (Unit::one_in(4), 0.25),
        (Unit::out_of(9, 10), 0.9),
    ] {
        let mut source = Random::new(Seed::from_integer(11u64));
        let coin = Bernoulli::at(chance);
        let hits: u32 = (0..DRAWS).filter(|_| source.sample(&coin)).count() as u32;

        let frequency: f64 = f64::from(hits) / f64::from(DRAWS);
        let deviation: f64 = 4.0 * (expected * (1.0 - expected) / f64::from(DRAWS)).sqrt();

        assert!(
            (frequency - expected).abs() < deviation,
            "chance {expected}: saw {frequency}"
        );
    }

    // And the two ends stay exact through the sampler.
    let mut source = Random::new(Seed::from_integer(12u64));
    assert!((0..5_000).all(|_| source.sample(&Bernoulli::at(Unit::ONE))));
    assert!((0..5_000).all(|_| !source.sample(&Bernoulli::at(Unit::ZERO))));
}

#[test]
fn converts_between_unit_and_unit_value() {
    use voxel_world::units::UnitValue;

    for value in [0.0, 0.25, 0.5, 0.75, 0.999] {
        let half_open: UnitValue = UnitValue::new(value).expect("in range");
        let unit: Unit = half_open.to_unit();

        assert_eq!(unit.to_f64(), value);
        assert_eq!(unit.to_unit_value().map(|back| back.value()), Some(value));
    }

    // Certainty has nowhere to go in a half-open type, and it says so rather than
    // quietly becoming the largest value below one.
    assert_eq!(Unit::ONE.to_unit_value(), None);
    // Just below one: it cannot be told from one as an `f64`, so it lands on the
    // largest value the half-open type has rather than being refused.
    let nearly = Unit::ALMOST_ONE.to_unit_value().expect("below one");
    assert!(nearly.value() < 1.0);
    assert_eq!(nearly.value(), f64::from_bits(0x3FEF_FFFF_FFFF_FFFF));
}

// ===========================================================================
// What the f64 range does and does not buy
// ===========================================================================

/// A chance below half a step cannot be realised by *any* comparison against a
/// uniform draw, however small the `f64` holding it. This is the claim the
/// `Probability` documentation now makes, so it should be tested rather than
/// asserted in prose.
#[test]
fn a_chance_below_the_grid_never_fires() {
    // Half a step is 2^-64, about 5.42e-20.
    let floor: f64 = 2f64.powi(-64);

    // Below it: rounds to zero, and so never happens.
    let too_small: Probability = Probability::clamped(1e-20);
    assert!(too_small.value() > 0.0, "the f64 holds it happily");
    assert_eq!(
        Unit::from_probability(too_small),
        Unit::ZERO,
        "but there is no representable chance for it"
    );

    let mut seed: Seed = Seed::from_integer(60221023u64);
    for _ in 0..100_000 {
        assert!(!too_small.decide(seed), "a sub-grid chance must never fire");
        assert!(!seed.chance(too_small));
        seed = seed.advance();
    }

    // Just above it: representable, and distinct from never.
    assert_eq!(Unit::clamped(1e-19), Unit::STEP);
    assert_ne!(Unit::clamped(floor * 1.1), Unit::ZERO);
    assert_eq!(Unit::clamped(floor * 0.9), Unit::ZERO);
}

/// Where a genuinely tiny chance has to be sampled, the exact fraction samplers
/// have no floor at all — which is what the documentation now points at instead of
/// implying the `f64` range would do it.
#[test]
fn an_exact_fraction_has_no_floor() {
    let one_in_a_quintillion: Ratio = Ratio::one_in(1_000_000_000_000_000_000).expect("valid");

    // Far below the Unit grid, and below what any f64 comparison could realise.
    assert!(one_in_a_quintillion.to_f64() < 2f64.powi(-59));

    // Accepted, deterministic, and not a rounding of anything.
    let mut seed: Seed = Seed::from_integer(1729u64);
    for _ in 0..1_000 {
        assert_eq!(seed.chance_ratio(one_in_a_quintillion), Some(false));
        assert_eq!(
            seed.chance_ratio(one_in_a_quintillion),
            seed.chance_ratio(one_in_a_quintillion),
            "the same seed decides the same way"
        );
        seed = seed.advance();
    }

    // And it keeps the fraction exactly, unlike either float-backed type.
    assert_eq!(one_in_a_quintillion.numerator(), 1);
    assert_eq!(one_in_a_quintillion.denominator(), 1_000_000_000_000_000_000);
}

/// Interpolation factors are `Unit`s now, not `Probability`s. They are positions
/// along an interval, not chances, and the type says so.
#[test]
fn blend_factors_are_units() {
    use voxel_world::math::UnitQuaternion;
    use voxel_world::units::NoiseValue;

    // Both ends are exact.
    assert_eq!(
        NoiseValue::LOWEST.blend(NoiseValue::HIGHEST, Unit::ZERO),
        NoiseValue::LOWEST
    );
    assert_eq!(
        NoiseValue::LOWEST.blend(NoiseValue::HIGHEST, Unit::ONE),
        NoiseValue::HIGHEST
    );

    // And the midpoint lands between them.
    let middle: NoiseValue = NoiseValue::LOWEST.blend(NoiseValue::HIGHEST, Unit::HALF);
    assert!(middle.value() > NoiseValue::LOWEST.value());
    assert!(middle.value() < NoiseValue::HIGHEST.value());

    // Scaling an amplitude only ever shrinks, since a Unit is at most one.
    let sample: NoiseValue = NoiseValue::clamped(0.8);
    assert_eq!(sample.scaled(Unit::ONE), sample);
    assert_eq!(sample.scaled(Unit::ZERO).value(), 0.0);
    assert!(sample.scaled(Unit::HALF).value().abs() < sample.value().abs());

    // Slerp likewise takes a position along the arc.
    let identity: UnitQuaternion = UnitQuaternion::IDENTITY;
    assert_eq!(identity.slerp(identity, Unit::HALF), identity);
}

// ===========================================================================
// Drawing the crate's own unit-interval types
// ===========================================================================

#[test]
fn seed_and_random_produce_probabilities() {
    use voxel_world::random::Random;

    let seed: Seed = Seed::from_integer(8080u64);

    // A Seed's one-off draw reads a fresh cursor; a Random is a sequential
    // generator seeded from it. They are deliberately different streams, so what
    // should agree is each one against the `Unit` draw it is built on.
    assert_eq!(
        seed.probability().value(),
        seed.unit().to_f64(),
        "a seed's chance is its own unit draw, narrowed to an f64"
    );

    let mut source = Random::new(seed);
    let drawn: Probability = source.probability();
    let mut source = Random::new(seed);
    assert_eq!(
        source.unit().to_probability(),
        drawn,
        "and a generator's is its own unit draw"
    );

    // Half-open: a drawn chance is never certainty.
    let mut seed: Seed = Seed::from_integer(1u64);
    let mut total: f64 = 0.0;
    const DRAWS: u32 = 50_000;

    for _ in 0..DRAWS {
        let drawn: Probability = seed.probability();

        assert!(drawn.value() >= 0.0 && drawn.value() < 1.0, "{drawn}");
        total += drawn.value();
        seed = seed.advance();
    }

    let mean: f64 = total / f64::from(DRAWS);
    assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");
}

#[test]
fn seed_and_random_produce_noise_values() {
    use voxel_world::random::Random;
    use voxel_world::units::NoiseValue;

    let seed: Seed = Seed::from_integer(3141u64);

    // Each is its own unit draw, stretched to [-1, 1) — not each other's, since a
    // cursor and a generator are different streams.
    assert_eq!(
        seed.noise_value().value(),
        seed.unit().to_unit_value().expect("below one").to_noise_value().value(),
        "a seed's noise value is its own unit draw, stretched"
    );

    let mut source = Random::new(seed);
    let drawn: NoiseValue = source.noise_value();
    let mut source = Random::new(seed);
    assert_eq!(
        source.unit().to_unit_value().expect("below one").to_noise_value(),
        drawn,
        "and a generator's likewise"
    );

    let mut seed: Seed = Seed::from_integer(271828u64);
    let mut total: f64 = 0.0;
    const DRAWS: u32 = 50_000;

    for _ in 0..DRAWS {
        let drawn: NoiseValue = seed.noise_value();

        // Half-open at the top, inherited from the unit draw.
        assert!(drawn.value() >= -1.0 && drawn.value() < 1.0, "{drawn}");
        total += drawn.value();
        seed = seed.advance();
    }

    // Centred on zero, since the range is symmetric.
    let mean: f64 = total / f64::from(DRAWS);
    assert!(mean.abs() < 0.02, "mean was {mean}");
}

/// A uniform orientation, which is the one people most often get wrong.
#[test]
fn seed_and_random_produce_uniform_rotations() {
    use voxel_world::math::{UnitQuaternion, Vector3};
    use voxel_world::random::Random;

    let seed: Seed = Seed::from_integer(1066u64);

    // Each entry point is stable for its own stream. A cursor and a generator are
    // different streams, so they give different — equally valid — orientations.
    assert_eq!(seed.rotation(), seed.rotation(), "a seed is deterministic");

    let mut source = Random::new(seed);
    let drawn: UnitQuaternion = source.rotation();
    let mut source = Random::new(seed);
    assert_eq!(source.rotation(), drawn, "a generator is too");

    // Every draw is a genuine rotation: unit length, and it preserves lengths.
    let mut seed: Seed = Seed::from_integer(1215u64);
    let probe: Vector3<f64> = Vector3::new(1.0, 2.0, -3.0);
    let length: f64 = probe.norm();

    for _ in 0..5_000 {
        let rotation: UnitQuaternion = seed.rotation();
        let turned: Vector3<f64> = rotation.rotate(probe);

        assert!(
            (turned.norm() - length).abs() < 1e-9,
            "a rotation must preserve length"
        );
        seed = seed.advance();
    }

    // Uniformity: a fixed probe turned by a uniform rotation lands uniformly on the
    // sphere, so each of the eight octants should take about an eighth of them.
    // Sampling Euler angles uniformly would visibly fail this.
    let mut seed: Seed = Seed::from_integer(1453u64);
    let mut octants: [u32; 8] = [0; 8];
    const DRAWS: u32 = 80_000;

    for _ in 0..DRAWS {
        let turned: Vector3<f64> = seed.rotation().rotate(Vector3::new(0.0, 0.0, 1.0));
        let index: usize = usize::from(turned.x > 0.0)
            | usize::from(turned.y > 0.0) << 1
            | usize::from(turned.z > 0.0) << 2;

        octants[index] += 1;
        seed = seed.advance();
    }

    let expected: f64 = f64::from(DRAWS) / 8.0;
    for (index, count) in octants.iter().enumerate() {
        assert!(
            (f64::from(*count) - expected).abs() < 5.0 * expected.sqrt(),
            "octant {index} held {count}, expected about {expected}"
        );
    }
}

#[test]
fn the_distributions_compose_like_the_others() {
    use voxel_world::random::Random;
    use voxel_world::random::distributions::UniformRotation;
    use voxel_world::units::{UniformNoise, UniformProbability};

    let mut source = Random::new(Seed::from_integer(2718u64));

    // Usable through `sample`, like every other distribution.
    let chance: Probability = source.sample(&UniformProbability);
    assert!(chance.value() >= 0.0 && chance.value() < 1.0);

    let mut source = Random::new(Seed::from_integer(2718u64));
    assert_eq!(
        source.probability(),
        chance,
        "the method and the distribution agree"
    );

    let mut source = Random::new(Seed::from_integer(5u64));
    let noise = source.sample(&UniformNoise);
    assert!(noise.value() >= -1.0 && noise.value() < 1.0);

    let mut source = Random::new(Seed::from_integer(6u64));
    let rotation = source.sample(&UniformRotation);
    assert!((rotation.quaternion().norm() - 1.0).abs() < 1e-12);
}

// ===========================================================================
// Exact relationships with Ratio and Fixed
// ===========================================================================

#[test]
fn converts_to_a_ratio_exactly() {
    assert_eq!(Unit::ZERO.to_ratio(), Ratio::new(0, 1).unwrap());
    assert_eq!(Unit::HALF.to_ratio(), Ratio::new(1, 2).unwrap());
    assert_eq!(Unit::ONE.to_ratio(), Ratio::new(1, 1).unwrap());

    // The step is one over the scale, and the scale is the largest power of two a
    // u64 holds.
    let step: Ratio = Unit::STEP.to_ratio();
    assert_eq!(step.numerator(), 1);
    assert_eq!(step.denominator(), Unit::STEPS);
    assert_eq!(Unit::STEPS, 1 << 63);

    // Reduced, so a dyadic value comes back in lowest terms.
    assert_eq!(Unit::one_in(4).to_ratio(), Ratio::new(1, 4).unwrap());
    assert_eq!(Unit::out_of(3, 8).to_ratio(), Ratio::new(3, 8).unwrap());

    // And the round trip is exact for every probe, since the ratio *is* the value.
    for value in probes() {
        let ratio: Ratio = value.to_ratio();

        assert_eq!(
            Unit::from_ratio_exact(ratio),
            Some(value),
            "{value} did not survive the round trip"
        );
        assert_eq!(Ratio::from(value), ratio, "the From impl agrees");
    }
}

#[test]
fn the_checked_ratio_conversion_refuses_what_the_capped_one_folds() {
    assert_eq!(Unit::try_from_ratio(Ratio::new(0, 1).unwrap()), Some(Unit::ZERO));
    assert_eq!(Unit::try_from_ratio(Ratio::new(1, 2).unwrap()), Some(Unit::HALF));
    assert_eq!(Unit::try_from_ratio(Ratio::new(1, 1).unwrap()), Some(Unit::ONE));

    // Above one: refused, where the capped form saturates. Same input, different
    // answer, because the two methods promise different things.
    let improper: Ratio = Ratio::new(7, 2).unwrap();

    assert_eq!(Unit::try_from_ratio(improper), None);
    assert_eq!(Unit::from_ratio_capped(improper), Unit::ONE);

    // The fallible `TryFrom` is the default conversion now.
    assert_eq!(Unit::try_from(improper), Err(voxel_world::units::RatioOutOfRange));
    assert_eq!(Unit::try_from(Ratio::new(3, 4).unwrap()), Ok(Unit::out_of(3, 4)));

    // Inside the range the two agree exactly, since they do the same arithmetic.
    for (numerator, denominator) in [(0u64, 1u64), (1, 3), (2, 7), (99, 100), (1, 1)] {
        let ratio: Ratio = Ratio::new(numerator, denominator).unwrap();

        assert_eq!(
            Unit::try_from_ratio(ratio),
            Some(Unit::from_ratio_capped(ratio)),
            "{numerator}/{denominator}"
        );
    }
}

#[test]
fn only_dyadic_ratios_convert_exactly() {
    // Powers of two in the denominator: exact.
    for (numerator, denominator, expected) in [
        (1u64, 2u64, Unit::HALF),
        (3, 8, Unit::out_of(3, 8)),
        (17, 64, Unit::out_of(17, 64)),
        (1, 1, Unit::ONE),
        (0, 1, Unit::ZERO),
    ] {
        let ratio: Ratio = Ratio::new(numerator, denominator).unwrap();

        assert_eq!(
            Unit::from_ratio_exact(ratio),
            Some(expected),
            "{numerator}/{denominator} should be exact"
        );
    }

    // Anything with an odd factor above one in the reduced denominator: never.
    for (numerator, denominator) in [(1u64, 3u64), (1, 10), (2, 7), (5, 6), (99, 100)] {
        assert_eq!(
            Unit::from_ratio_exact(Ratio::new(numerator, denominator).unwrap()),
            None,
            "{numerator}/{denominator} is not dyadic"
        );
    }

    // `Ratio` reduces on construction, so an unreduced dyadic fraction is still
    // recognised: 3/12 arrives as 1/4.
    assert_eq!(
        Unit::from_ratio_exact(Ratio::new(3, 12).unwrap()),
        Some(Unit::one_in(4)),
        "the test is on the reduced denominator"
    );

    // And above one it refuses, dyadic or not.
    assert_eq!(Unit::from_ratio_exact(Ratio::new(3, 2).unwrap()), None);

    // Right down to the step.
    assert_eq!(
        Unit::from_ratio_exact(Ratio::new(1, 1 << 63).unwrap()),
        Some(Unit::STEP)
    );
}

#[test]
fn converts_to_and_from_fixed_without_floating_point() {
    use voxel_world::math::Fixed;

    // Fixed -> Unit is exact: 32 fractional bits into 63.
    for (value, expected) in [
        (Fixed::ZERO, Unit::ZERO),
        (Fixed::ONE, Unit::ONE),
        (Fixed::HALF, Unit::HALF),
        (Fixed::ONE / Fixed::from_integer(4), Unit::one_in(4)),
    ] {
        assert_eq!(Unit::try_from_fixed(value), Some(expected), "{value}");

        // And it round-trips back, since nothing was lost going in.
        assert_eq!(expected.to_fixed(), value, "round trip of {value}");
    }

    // Every Fixed step in range converts exactly and returns unchanged.
    let step: i128 = 1;
    for count in [0i128, 1, 2, 1000, (1 << 32) - 1, 1 << 32] {
        let value: Fixed = Fixed::from_bits(count * step);
        let unit: Unit = Unit::try_from_fixed(value).expect("in range");

        assert_eq!(unit.to_fixed(), value, "bits {count}");
        // 31 bits of headroom, so the Unit count is the Fixed count shifted up.
        assert_eq!(unit.to_bits(), (count as u64) << 31);
    }
}

#[test]
fn unit_to_fixed_rounds_to_nearest_ties_even() {
    use voxel_world::math::Fixed;

    // 31 bits fall off, so half a Fixed step is 2^30 Unit steps.
    let half_step: u64 = 1 << 30;

    // Exactly halfway between Fixed bits 0 and 1: ties to even keeps 0.
    let tie_low: Unit = Unit::from_bits(half_step).unwrap();
    assert_eq!(tie_low.to_fixed().to_bits(), 0, "a tie goes to even");

    // Exactly halfway between 1 and 2: ties to even goes up to 2.
    let tie_high: Unit = Unit::from_bits((1 << 31) + half_step).unwrap();
    assert_eq!(tie_high.to_fixed().to_bits(), 2, "a tie goes to even");

    // Just above halfway always goes up.
    let above: Unit = Unit::from_bits(half_step + 1).unwrap();
    assert_eq!(above.to_fixed().to_bits(), 1);

    // Just below always goes down.
    let below: Unit = Unit::from_bits(half_step - 1).unwrap();
    assert_eq!(below.to_fixed().to_bits(), 0);

    // The top of the range cannot round past one.
    assert_eq!(Unit::ONE.to_fixed(), Fixed::ONE);
    assert_eq!(Unit::ALMOST_ONE.to_fixed(), Fixed::ONE, "rounds up to one");
}

#[test]
fn out_of_range_fixed_is_refused_or_folded_as_documented() {
    use voxel_world::math::Fixed;

    // Refused by the checked form.
    assert_eq!(Unit::try_from_fixed(Fixed::from_integer(-3)), None);
    assert_eq!(Unit::try_from_fixed(Fixed::from_integer(2)), None);
    assert_eq!(Unit::try_from_fixed(Fixed::from_bits(-1)), None, "one step below zero");
    assert_eq!(
        Unit::try_from_fixed(Fixed::from_bits(Fixed::ONE.to_bits() + 1)),
        None,
        "one step above one"
    );

    // Folded by the clamping form, which is what its name promises.
    assert_eq!(Unit::from_fixed_clamped(Fixed::from_integer(-3)), Unit::ZERO);
    assert_eq!(Unit::from_fixed_clamped(Fixed::from_integer(2)), Unit::ONE);
    assert_eq!(Unit::from_fixed_clamped(Fixed::HALF), Unit::HALF);
}

#[test]
fn percentage_constructors_are_exact_where_they_can_be() {
    // Dyadic percentages land exactly.
    assert_eq!(Unit::percent(50), Some(Unit::HALF));
    assert_eq!(Unit::percent(25), Some(Unit::one_in(4)));
    assert_eq!(Unit::percent(0), Some(Unit::ZERO));
    assert_eq!(Unit::percent(100), Some(Unit::ONE));

    assert_eq!(Unit::permille(125), Some(Unit::out_of(1, 8)));
    assert_eq!(Unit::permille(500), Some(Unit::HALF));
    assert_eq!(Unit::basis_points(250), Some(Unit::out_of(1, 40)));
    assert_eq!(Unit::basis_points(10_000), Some(Unit::ONE));

    // Non-dyadic ones are correctly rounded rather than refused.
    assert_eq!(Unit::percent(33), Some(Unit::out_of(33, 100)));
    assert_eq!(Unit::permille(1), Some(Unit::out_of(1, 1000)));

    // Above the whole: refused, not capped.
    assert_eq!(Unit::percent(101), None);
    assert_eq!(Unit::permille(1_001), None);
    assert_eq!(Unit::basis_points(10_001), None);

    // The three agree where they describe the same number.
    assert_eq!(Unit::percent(2), Unit::permille(20));
    assert_eq!(Unit::permille(20), Unit::basis_points(200));
}
