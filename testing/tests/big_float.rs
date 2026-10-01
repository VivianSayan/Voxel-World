//! Growable floating point.
//!
//! The claim this type makes is **exactness** for addition, subtraction and
//! multiplication, so that is what the tests are built around. Three oracles:
//!
//! 1. **Exact rational arithmetic.** Expected decimal expansions and
//!    correctly-rounded quotients were computed with `fractions.Fraction`.
//! 2. **`f64`.** A finite `f64` is held exactly, so any sum or product of `f64`
//!    values must come back bit-identical after being rounded to 53 bits — and
//!    unlike the fixed-width float, *no* input range restriction is needed, because
//!    nothing is ever rounded on the way.
//! 3. **The laws themselves.** Associativity and distributivity are asserted as
//!    exact equalities, which no other float in this crate can satisfy.

use std::collections::BTreeMap;
use voxel_world::math::traits::{One, Semiring, Zero};
use voxel_world::math::{BigFloat, BigInt, BigUint, WideFloat};

/// Exact decimal expansions, computed with `fractions.Fraction`.
/// Each is `numerator / denominator` where the quotient terminates in binary.
const EXACT_DECIMALS: &[(i64, i64, &str)] = &[
    (1, 4, "0.25"),
    (1, 8, "0.125"),
    (3, 4, "0.75"),
    (-7, 16, "-0.4375"),
    (1, 1024, "0.0009765625"),
    (5, 2, "2.5"),
    (1, 1099511627776, "0.0000000000009094947017729282379150390625"),
    (123, 64, "1.921875"),
    (-1, 2, "-0.5"),
];

/// `1/3` rounded to a given number of bits, ties to even: significand and exponent.
const ROUNDED_THIRDS: &[(u64, &str, i64)] = &[
    (1, "1", -2),
    (2, "3", -3),
    (3, "5", -4),
    (8, "171", -9),
    (53, "6004799503160661", -54),
    (64, "12297829382473034411", -65),
    (100, "845100400152152934331135470251", -101),
    (256, "77194726158210796949047323339125271902179989777093709359638389338608753093291", -257),
];

fn float(value: i64) -> BigFloat {
    BigFloat::from_integer(value)
}

/// `f64` values spanning the whole exponent range, including subnormals — none of
/// which need restricting, since nothing here rounds.
fn probes() -> Vec<f64> {
    let mut values: Vec<f64> = vec![
        0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        3.0,
        0.1,
        -0.1,
        1e300,
        1e-300,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        std::f64::consts::PI,
        -123.456,
        1.0 / 3.0,
    ];

    let mut state: u64 = 0x853C49E6748FEA9B;
    for _ in 0..30 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        let candidate: f64 = f64::from_bits(state);
        if candidate.is_finite() {
            values.push(candidate);
        }
    }

    values
}

// ===========================================================================
// Exactness
// ===========================================================================

/// The headline property: the ring laws hold as written, not up to a rounding.
#[test]
fn the_ring_laws_hold_exactly() {
    let a: BigFloat = BigFloat::from_f64(0.1).unwrap();
    let b: BigFloat = BigFloat::from_f64(std::f64::consts::PI).unwrap();
    let c: BigFloat = float(1) .checked_div_exact(&float(1024)).unwrap();

    assert_eq!(
        a.clone() + b.clone() + c.clone(),
        a.clone() + (b.clone() + c.clone()),
        "addition associates exactly"
    );
    assert_eq!(
        a.clone() * b.clone() * c.clone(),
        a.clone() * (b.clone() * c.clone()),
        "multiplication associates exactly"
    );
    assert_eq!(
        a.clone() * (b.clone() + c.clone()),
        a.clone() * b.clone() + a.clone() * c.clone(),
        "multiplication distributes exactly"
    );
    assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());
    assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());
    assert_eq!(a.clone() - a.clone(), BigFloat::zero());

    // The contrast worth recording: f64 addition is not associative, and the
    // classic triple shows it. Note it took a specific triple to find — the laws
    // hold for f64 *most* of the time, which is exactly what makes relying on them
    // a trap.
    let (x, y, z) = (0.1f64, 0.2f64, 0.3f64);
    assert_ne!((x + y) + z, x + (y + z), "f64 is not associative here");
    assert_eq!((x + y) + z, 0.600_000_000_000_000_1);
    assert_eq!(x + (y + z), 0.6);

    // The same three values through this type associate exactly.
    let (bx, by, bz) = (
        BigFloat::from_f64(x).unwrap(),
        BigFloat::from_f64(y).unwrap(),
        BigFloat::from_f64(z).unwrap(),
    );
    assert_eq!(
        (bx.clone() + by.clone()) + bz.clone(),
        bx.clone() + (by.clone() + bz.clone()),
        "and here it does associate"
    );
}

/// A fixed-precision float rounds a sum of many terms; this one does not.
#[test]
fn a_long_sum_stays_exact() {
    // 0.1 added ten thousand times. Exactly 1000, with no drift, because each 0.1
    // here is the exact binary value of the f64 0.1 and the sum of ten thousand of
    // them is exactly ten thousand times it.
    let tenth: BigFloat = BigFloat::from_f64(0.1).unwrap();
    let mut total: BigFloat = BigFloat::zero();

    for _ in 0..10_000 {
        total += tenth.clone();
    }

    let expected: BigFloat = tenth.clone() * float(10_000);
    assert_eq!(total, expected, "the sum is exactly 10000 x 0.1");

    // And that is *not* 1000, because the f64 0.1 is not a tenth. The type is exact
    // about the value it was given, which is the honest thing to be.
    assert_ne!(total, float(1_000));

    // The f64 sum drifts from its own exact answer; this one cannot.
    let mut narrow: f64 = 0.0;
    for _ in 0..10_000 {
        narrow += 0.1;
    }
    assert_ne!(narrow, 0.1f64 * 10_000.0, "f64 drifts");
}

#[test]
fn multiplication_never_loses_a_bit() {
    // 3^1024 exactly, by repeated squaring. A fixed-precision float would have
    // rounded this away ten times over.
    let mut value: BigFloat = float(3);
    for _ in 0..10 {
        value = value.clone() * value.clone();
    }

    let expected: BigUint = BigUint::from(3u64).pow(1024);
    assert_eq!(value.significand(), &expected);
    assert_eq!(value.significand_bits(), expected.bit_length());
    assert!(!value.is_negative());

    // Exact division back down, since the significand divides evenly.
    let (root, remainder) = expected.div_rem(&BigUint::from(3u64).pow(512)).unwrap();
    assert!(remainder.is_zero());
    assert_eq!(root, BigUint::from(3u64).pow(512));
}

#[test]
fn addition_across_a_huge_exponent_gap_is_exact() {
    // 1 + 2^-2000, which needs 2001 bits and gets them.
    let tiny: BigFloat = float(1)
        .checked_div_exact(&BigFloat::from_big_int(&BigInt::from(2).pow(2000)))
        .expect("a power of two divides exactly");

    let sum: BigFloat = float(1) + tiny.clone();

    assert_eq!(sum.significand_bits(), 2001, "every bit of the gap is held");
    assert_ne!(sum, float(1), "the tiny addend is not lost");
    assert_eq!(sum - float(1), tiny, "and it comes back out exactly");
}

// ===========================================================================
// Against f64
// ===========================================================================

#[test]
fn round_trips_through_f64() {
    for value in probes() {
        let grown: BigFloat = BigFloat::from_f64(value).expect("finite");

        assert_eq!(grown.to_f64(), value, "round trip of {value}");
    }
}

#[test]
fn rejects_the_f64_values_it_has_no_room_for() {
    assert_eq!(BigFloat::from_f64(f64::NAN), None);
    assert_eq!(BigFloat::from_f64(f64::INFINITY), None);
    assert_eq!(BigFloat::from_f64(f64::NEG_INFINITY), None);
}

/// No exponent-gap restriction here, unlike the fixed-width float's tests: the sum
/// is exact for every pair, so rounding it to an `f64` must match `f64` addition.
#[test]
fn addition_matches_f64_for_every_pair() {
    let probes: Vec<f64> = probes();
    let mut compared: usize = 0;

    for left in &probes {
        for right in &probes {
            let expected: f64 = left + right;

            // Skip only where the f64 answer itself is not a clean target: an
            // overflow to infinity, or a subnormal that f64 rounds twice.
            if !expected.is_finite() || (!expected.is_normal() && expected != 0.0) {
                continue;
            }

            let sum: BigFloat = BigFloat::from_f64(*left).unwrap()
                + BigFloat::from_f64(*right).unwrap();

            assert_eq!(sum.to_f64(), expected, "{left} + {right}");
            compared += 1;
        }
    }

    assert!(compared > 1_000, "only {compared} pairs compared");
}

#[test]
fn multiplication_matches_f64_for_every_pair() {
    for left in probes() {
        for right in probes() {
            let expected: f64 = left * right;

            if !expected.is_finite() || (!expected.is_normal() && expected != 0.0) {
                continue;
            }

            let product: BigFloat =
                BigFloat::from_f64(left).unwrap() * BigFloat::from_f64(right).unwrap();

            assert_eq!(product.to_f64(), expected, "{left} * {right}");
        }
    }
}

#[test]
fn ordering_matches_f64() {
    let probes: Vec<f64> = probes();

    for left in &probes {
        for right in &probes {
            assert_eq!(
                BigFloat::from_f64(*left)
                    .unwrap()
                    .cmp(&BigFloat::from_f64(*right).unwrap()),
                left.partial_cmp(right).expect("no NaN here"),
                "comparing {left} with {right}"
            );
        }
    }
}

/// The significands here are minimal rather than a fixed width, so the exponent
/// alone does not decide the order — `3 × 2^0` beats `1 × 2^1`. Worth a test of its
/// own, since the fixed-width type *can* compare exponents first.
#[test]
fn ordering_does_not_just_compare_exponents() {
    let three: BigFloat = float(3); // significand 3, exponent 0
    let two: BigFloat = float(2); // significand 1, exponent 1

    assert_eq!(three.exponent().to_i128(), Some(0));
    assert_eq!(two.exponent().to_i128(), Some(1));
    assert!(three > two, "the larger exponent is not the larger value");

    // And with the signs flipped the order reverses.
    assert!(-three.clone() < -two.clone());

    // A longer significand with an equal leading position.
    let a: BigFloat = float(3); // 11b
    let b: BigFloat = float(7).checked_div_exact(&float(2)).unwrap(); // 3.5 = 111b x 2^-1
    assert!(b > a);
}

// ===========================================================================
// Division
// ===========================================================================

#[test]
fn exact_division_succeeds_only_when_it_is_exact() {
    // A power-of-two divisor always terminates.
    for (numerator, divisor, expected) in EXACT_DECIMALS {
        let quotient: BigFloat = float(*numerator)
            .checked_div_exact(&float(*divisor))
            .expect("a power-of-two divisor divides exactly");

        assert_eq!(quotient.to_string(), *expected, "{numerator} / {divisor}");
    }

    // And a divisor with an odd factor the numerator lacks does not.
    for (numerator, divisor) in [(1i64, 3i64), (1, 7), (2, 3), (5, 6), (-1, 3), (10, 7)] {
        assert_eq!(
            float(numerator).checked_div_exact(&float(divisor)),
            None,
            "{numerator} / {divisor} does not terminate"
        );
    }

    // But an odd divisor that *does* divide is fine — the test is divisibility, not
    // whether the divisor is a power of two.
    assert_eq!(
        float(9).checked_div_exact(&float(3)),
        Some(float(3)),
        "9 / 3 is exact"
    );
    assert_eq!(
        float(-20).checked_div_exact(&float(5)),
        Some(float(-4)),
        "signs multiply through"
    );

    // Dividing by zero refuses; zero divided by anything is zero.
    assert_eq!(float(1).checked_div_exact(&BigFloat::zero()), None);
    assert_eq!(
        BigFloat::zero().checked_div_exact(&float(5)),
        Some(BigFloat::zero())
    );
}

#[test]
fn rounded_division_is_correctly_rounded() {
    for (precision, significand, exponent) in ROUNDED_THIRDS {
        let third: BigFloat = float(1)
            .div_rounded(&float(3), *precision)
            .expect("not by zero");

        assert_eq!(
            third.significand().to_string(),
            *significand,
            "1/3 to {precision} bits: significand"
        );
        assert_eq!(
            third.exponent().to_i128(),
            Some(i128::from(*exponent)),
            "1/3 to {precision} bits: exponent"
        );
        assert!(
            third.significand_bits() <= *precision,
            "1/3 to {precision} bits used {} bits",
            third.significand_bits()
        );
    }
}

#[test]
fn rounded_division_agrees_with_the_fixed_width_float() {
    // At 256 bits the two types should produce the same value, reached by entirely
    // different code: Algorithm-D division on growable limbs against bit-at-a-time
    // division on a fixed array.
    type Fixed256 = WideFloat<4>;

    for (numerator, divisor) in [(1i64, 3i64), (1, 7), (22, 7), (355, 113), (-1, 3), (1000000007, 3)] {
        let grown: BigFloat = float(numerator)
            .div_rounded(&float(divisor), 256)
            .expect("not by zero");
        let fixed: Fixed256 =
            Fixed256::from_integer(numerator) / Fixed256::from_integer(divisor);

        // The fixed type normalises its significand's top bit to position 255; this
        // one makes the significand odd. Comparing the decimal text sidesteps both
        // conventions and compares the values.
        assert_eq!(
            grown.to_string(),
            fixed.to_string(),
            "{numerator} / {divisor} at 256 bits"
        );
    }
}

#[test]
fn rounded_division_refuses_a_zero_precision() {
    assert_eq!(float(1).div_rounded(&float(3), 0), None);
    assert_eq!(float(1).div_rounded(&BigFloat::zero(), 64), None);
    assert_eq!(
        BigFloat::zero().div_rounded(&float(3), 64),
        Some(BigFloat::zero())
    );
}

// ===========================================================================
// Growth and the pressure valve
// ===========================================================================

#[test]
fn rounding_trims_growth() {
    // Squaring grows the significand; rounding puts it back.
    let mut value: BigFloat = float(3);
    for _ in 0..8 {
        value = value.clone() * value.clone();
    }

    assert!(value.significand_bits() > 400, "growth is expected");

    let trimmed: BigFloat = value.rounded_to(64);
    assert!(trimmed.significand_bits() <= 64);

    // Still the same value to within the precision kept.
    let ratio: BigFloat = trimmed
        .div_rounded(&value, 128)
        .expect("neither is zero");
    assert!(ratio.to_f64() > 0.999_999_999, "trimmed too far: {ratio}");
    assert!(ratio.to_f64() < 1.000_000_001, "trimmed too far: {ratio}");

    // Rounding to more bits than a value has leaves it untouched.
    assert_eq!(float(3).rounded_to(1000), float(3));
    assert_eq!(float(3).rounded_to(0), float(3), "zero means no change");
    assert_eq!(BigFloat::zero().rounded_to(8), BigFloat::zero());
}

#[test]
fn rounding_is_ties_to_even() {
    // 0b1011 rounded to 3 bits: the dropped bit is 1 with nothing below, an exact
    // tie, and 0b101 is odd — so it rounds up to 0b110, which normalises to 3 x 2^1.
    let eleven: BigFloat = float(0b1011);
    let rounded: BigFloat = eleven.rounded_to(3);

    assert_eq!(rounded.significand().to_string(), "3");
    assert_eq!(rounded.exponent().to_i128(), Some(2));
    assert_eq!(rounded.to_f64(), 12.0);

    // 0b1001 to 3 bits: tie again, but 0b100 is even, so it stays — giving 8.
    assert_eq!(float(0b1001).rounded_to(3).to_f64(), 8.0);

    // 13 = 0b1101 to 3 bits is *also* a tie, not an above-halfway case: the
    // candidates are 12 and 14 and 13 is exactly their midpoint. With only one bit
    // dropped every value is either exact or an exact tie, so parity always decides.
    // 6 is even, so it stays: 12.
    assert_eq!(float(0b1101).rounded_to(3).to_f64(), 12.0);

    // Genuinely above and below halfway need two dropped bits. To 3 bits the
    // neighbours are 20 and 24, with 22 between them:
    assert_eq!(float(21).rounded_to(3).to_f64(), 20.0, "below halfway, rounds down");
    assert_eq!(float(23).rounded_to(3).to_f64(), 24.0, "above halfway, rounds up");
    assert_eq!(float(22).rounded_to(3).to_f64(), 24.0, "a tie, and 5 is odd");
    assert_eq!(float(18).rounded_to(3).to_f64(), 16.0, "a tie, and 4 is even");
}

#[test]
fn refuses_to_allocate_an_unreasonable_answer() {
    // A power of two has a one-bit significand however enormous it is, so a value
    // with a two-billion-bit exponent costs nothing to hold on its own.
    let high: BigFloat = BigFloat::from_big_int(&BigInt::from(2).pow(2_000_000_000));

    assert_eq!(high.significand_bits(), 1, "cheap on its own");
    assert_eq!(high.exponent().to_i128(), Some(2_000_000_000));

    // Adding one to it is where the cost appears: the exact answer genuinely has two
    // billion significant bits, so it refuses rather than allocating 250 MB.
    assert_eq!(
        high.checked_add(&float(1)),
        None,
        "the exact sum would need two billion bits"
    );
    assert_eq!(high.checked_sub(&float(1)), None);

    // Multiplying is still fine, because it does not need the gap.
    assert_eq!(
        high.checked_mul(&high).map(|value| value.significand_bits()),
        Some(1)
    );
    assert_eq!(
        high.checked_mul(&high).and_then(|v| v.exponent().to_i128()),
        Some(4_000_000_000)
    );
}

#[test]
#[should_panic(expected = "MAX_SIGNIFICAND_BITS")]
fn the_operator_panics_where_the_checked_method_refuses() {
    let high: BigFloat = BigFloat::from_big_int(&BigInt::from(2).pow(2_000_000_000));

    let _ = high + float(1);
}

// ===========================================================================
// Structure
// ===========================================================================

#[test]
fn the_significand_is_always_odd() {
    let cases: Vec<BigFloat> = vec![
        float(1),
        float(2),
        float(1024),
        float(-48),
        BigFloat::from_f64(0.5).unwrap(),
        BigFloat::from_f64(1e300).unwrap(),
        float(3) * float(1024),
        float(1).div_rounded(&float(3), 100).unwrap(),
        float(1) + float(1),
    ];

    for value in cases {
        assert!(
            value.significand().bit(0),
            "{value} has an even significand"
        );
    }

    // Two spellings of the same number normalise to one, which is what lets
    // `PartialEq` be derived.
    assert_eq!(float(2), float(1) + float(1));
    assert_eq!(float(1024).significand().to_string(), "1");
    assert_eq!(float(1024).exponent().to_i128(), Some(10));
}

#[test]
fn has_no_negative_zero() {
    assert_eq!(-BigFloat::zero(), BigFloat::zero());
    assert!(!(-BigFloat::zero()).is_negative());
    assert_eq!((float(5) - float(5)), BigFloat::zero());
    assert!(!(float(5) - float(5)).is_negative());
    assert_eq!(BigFloat::zero().to_string(), "0");
}

#[test]
fn can_key_an_ordered_map() {
    let mut map: BTreeMap<BigFloat, &str> = BTreeMap::new();

    map.insert(float(1).div_rounded(&float(3), 64).unwrap(), "a third");
    map.insert(float(-2), "minus two");
    map.insert(BigFloat::zero(), "zero");

    assert_eq!(
        map.values().copied().collect::<Vec<_>>(),
        vec!["minus two", "zero", "a third"]
    );

    for key in map.keys() {
        assert_eq!(key, key, "every value equals itself, unlike NaN");
    }
}

// ===========================================================================
// Text
// ===========================================================================

#[test]
fn prints_exact_decimals() {
    for (numerator, divisor, expected) in EXACT_DECIMALS {
        let value: BigFloat = float(*numerator)
            .checked_div_exact(&float(*divisor))
            .expect("exact");

        assert_eq!(value.to_string(), *expected);
    }

    assert_eq!(float(0).to_string(), "0");
    assert_eq!(float(-5).to_string(), "-5");
    assert_eq!(float(1024).to_string(), "1024");
}

#[test]
fn falls_back_to_hexadecimal_for_an_unprintable_exponent() {
    let huge: BigFloat = BigFloat::from_big_int(&BigInt::from(2).pow(1_000_000));
    let text: String = huge.to_string();

    assert!(text.starts_with("0x"), "expected the hex form, got {text}");
    assert!(text.contains('p'), "{text}");

    // Hex is exact at any size, and names both parts.
    assert_eq!(format!("{:x}", float(1)), "0x1p0");
    assert_eq!(format!("{:x}", float(-1024)), "-0x1p10");
    assert_eq!(format!("{:x}", BigFloat::zero()), "0x0p0");
}

// ===========================================================================
// Traits
// ===========================================================================

#[test]
fn zero_and_one_behave() {
    assert!(Zero::is_zero(&BigFloat::zero()));
    assert!(One::is_one(&BigFloat::one()));
    assert!(!One::is_one(&float(-1)));
    assert_eq!(<BigFloat as Zero>::zero(), BigFloat::zero());
    assert_eq!(<BigFloat as One>::one(), float(1));
    assert_eq!(float(7) * BigFloat::one(), float(7));
    assert_eq!(float(7) + BigFloat::zero(), float(7));
}

#[test]
fn power_comes_from_the_semiring_trait() {
    // Exact, at any size — which is the difference from a fixed-precision float.
    assert_eq!(Semiring::power(&float(3), 10), float(59049));
    assert_eq!(Semiring::power(&float(2), 100).significand_bits(), 1);
    assert_eq!(
        Semiring::power(&float(2), 100).exponent().to_i128(),
        Some(100)
    );
    assert_eq!(Semiring::power(&float(3), 0), BigFloat::one());
    assert!(Semiring::power(&float(-2), 101).is_negative());
    assert!(!Semiring::power(&float(-2), 100).is_negative());
}
