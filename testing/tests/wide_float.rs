//! Fixed-width extended-precision floating point.
//!
//! Three independent sources of truth are used, because a new numeric type cannot
//! be checked against itself:
//!
//! 1. **Exact rational arithmetic.** The expected significand and exponent for
//!    division and addition were computed with `fractions.Fraction` — the exact
//!    quotient, then correctly rounded to 256 bits, ties to even. Those are the
//!    tables below, and they are what proves the rounding is right rather than
//!    merely self-consistent.
//! 2. **`f64`.** For inputs whose exact result fits in 256 bits, `WideFloat`'s
//!    answer is exact, so rounding it to an `f64` must give bit-for-bit what `f64`
//!    arithmetic gives — both being round-to-nearest of the same exact value.
//! 3. **Known decimal expansions.** A binary fraction terminates in decimal, so
//!    the printed value can be checked against an expansion worked out
//!    independently.

use std::collections::BTreeMap;
use voxel_world::math::traits::{Field, One, Semiring, Zero};
use voxel_world::math::{WideFloat, WideUint};

type Float = WideFloat<4>; // 256 bits of significand
type Wide = WideFloat<4, 2>; // and a 128-bit exponent

/// `DIVISION_CASES`: correctly-rounded results computed with exact rational
/// arithmetic, independently of the implementation.
const DIVISION_CASES: &[(i64, i64, bool, [u64; 4], i64)] = &[
    (1, 3, false, [0xaaaaaaaaaaaaaaab, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa], -257),
    (1, 7, false, [0x9249249249249249, 0x4924924924924924, 0x2492492492492492, 0x9249249249249249], -258),
    (2, 3, false, [0xaaaaaaaaaaaaaaab, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa], -256),
    (10, 3, false, [0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0xd555555555555555], -254),
    (1, 10, false, [0xcccccccccccccccd, 0xcccccccccccccccc, 0xcccccccccccccccc, 0xcccccccccccccccc], -259),
    (-1, 3, true, [0xaaaaaaaaaaaaaaab, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa], -257),
    (1, -3, true, [0xaaaaaaaaaaaaaaab, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa], -257),
    (-7, -9, false, [0xc71c71c71c71c71c, 0x1c71c71c71c71c71, 0x71c71c71c71c71c7, 0xc71c71c71c71c71c], -256),
    (22, 7, false, [0x4924924924924925, 0x2492492492492492, 0x9249249249249249, 0xc924924924924924], -254),
    (355, 113, false, [0xc090fdbc090fdbc1, 0xdbc090fdbc090fdb, 0x0fdbc090fdbc090f, 0xc90fdbc090fdbc09], -254),
    (1, 1, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000], -255),
    (5, 5, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000], -255),
    (1000000007, 3, false, [0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x9ef21abd55555555], -227),
    (-999983, 999979, true, [0x265df49c3dfc2bf4, 0xf7b5e65666feddbc, 0x91a8ee9b74c90973, 0x8000218e1d6f9e5c], -255),
    (2, 1, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000], -254),
    (1, 2, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000], -256),
    (123456789, 987654321, false, [0x843c9c2a09b2ed99, 0x6dfcfb508523a06b, 0x6c43da251bbdf8a3, 0xffffffd8dcb34656], -259),
];
/// `ADDITION_CASES`: correctly-rounded results computed with exact rational
/// arithmetic, independently of the implementation.
const ADDITION_CASES: &[(i64, i64, bool, [u64; 4], i64)] = &[
    (1, 1, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000], -254),
    (1, -1, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000], 0),
    (1, 2, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0xc000000000000000], -254),
    (3, -1, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000], -254),
    (1000000, 1, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0xf424100000000000], -236),
    (-5, 3, true, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000], -254),
    (7, -7, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000], 0),
    (9007199254740992, 1, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000400], -202),
    (1, 9007199254740992, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000400], -202),
    (123456789, -123456788, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000], -255),
    (999999999999, 1, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0xe8d4a51000000000], -216),
    (-1, -1, true, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000], -254),
    (0, 5, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0xa000000000000000], -253),
    (5, 0, false, [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0xa000000000000000], -253),
];
/// Exact decimal expansions, verified with `fractions.Fraction`.
const DECIMALS: &[(i64, i64, &str)] = &[
    (1, 3, "0.33333333333333333333333333333333333333333333333333333333333333333333333333333477269475918240743756439197713339992851933339407271356417061724502809863386040450966191791724538137109342454899102494393283251649113865778227550062950967912911437451839447021484375"),
    (1, 7, "0.142857142857142857142857142857142857142857142857142857142857142857142857142856834422551603769834807630290614271443888714272698704236249153447493978864172770462215303303447418277622837596644780369157250175037613144760940964150819354472332634031772613525390625"),
    (1, 10, "0.1000000000000000000000000000000000000000000000000000000000000000000000000000002159042138773611156346587965700099892779000091109070346255925867542147950790606764492876875868072056640136823486537415899248774736707986673413250944264518693671561777591705322265625"),
    (2, 3, "0.6666666666666666666666666666666666666666666666666666666666666666666666666666695453895183648148751287839542667998570386667881454271283412344900561972677208090193238358344907627421868490979820498878656650329822773155645510012590193582582287490367889404296875"),
    (-1, 3, "-0.33333333333333333333333333333333333333333333333333333333333333333333333333333477269475918240743756439197713339992851933339407271356417061724502809863386040450966191791724538137109342454899102494393283251649113865778227550062950967912911437451839447021484375"),
    (1, 1024, "0.0009765625"),
    (355, 113, "3.14159292035398230088495575221238938053097345132743362831858407079646017699116542202687432310749182057491246794969857288558787178010145704318719507958778615666692403071398733179120236521781810030146027468489892974824739813044516267837025225162506103515625"),
];

// ===========================================================================
// Against exact rational arithmetic
// ===========================================================================

#[test]
fn division_is_correctly_rounded() {
    for (numerator, divisor, negative, significand, exponent) in DIVISION_CASES {
        let quotient: Float =
            Float::from_integer(*numerator) / Float::from_integer(*divisor);

        assert_eq!(
            quotient.is_negative(),
            *negative,
            "sign of {numerator} / {divisor}"
        );
        assert_eq!(
            quotient.significand(),
            WideUint::from_limbs(*significand),
            "significand of {numerator} / {divisor}"
        );
        assert_eq!(
            quotient.exponent().to_i128(),
            Some(i128::from(*exponent)),
            "exponent of {numerator} / {divisor}"
        );
    }
}

#[test]
fn addition_is_correctly_rounded() {
    for (left, right, negative, significand, exponent) in ADDITION_CASES {
        let sum: Float = Float::from_integer(*left) + Float::from_integer(*right);

        assert_eq!(sum.is_negative(), *negative, "sign of {left} + {right}");
        assert_eq!(
            sum.significand(),
            WideUint::from_limbs(*significand),
            "significand of {left} + {right}"
        );
        assert_eq!(
            sum.exponent().to_i128(),
            Some(i128::from(*exponent)),
            "exponent of {left} + {right}"
        );
    }
}

#[test]
fn prints_the_exact_decimal_expansion() {
    for (numerator, divisor, expected) in DECIMALS {
        let quotient: Float =
            Float::from_integer(*numerator) / Float::from_integer(*divisor);

        assert_eq!(
            quotient.to_string(),
            *expected,
            "decimal of {numerator} / {divisor}"
        );
    }
}

// ===========================================================================
// Against f64
// ===========================================================================

/// A spread of `f64` values with moderate exponents, so no product overflows and
/// no sum needs more than 256 bits to be exact.
fn probes() -> Vec<f64> {
    let mut values: Vec<f64> = vec![
        0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 3.0, 0.1, -0.1, 1e10, 1e-10, 123.456,
        -987.654, 1.0 / 3.0, std::f64::consts::PI, 65536.0, 1e-5, 7.0,
    ];

    let mut state: u64 = 0x2545F4914F6CDD1D;
    for _ in 0..40 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // A significand from the low bits and an exponent kept near zero, so the
        // exact result of any pairing stays inside 256 bits.
        let significand = (state >> 11) as f64 / (1u64 << 53) as f64;
        let exponent = (state % 41) as i32 - 20;
        values.push(significand * 2f64.powi(exponent));
    }

    values
}

#[test]
fn round_trips_through_f64() {
    for value in probes() {
        let wide: Float = Float::from_f64(value).expect("finite");

        assert_eq!(wide.to_f64(), value, "round trip of {value}");
    }
}

#[test]
fn rejects_the_f64_values_it_has_no_room_for() {
    assert_eq!(Float::from_f64(f64::NAN), None);
    assert_eq!(Float::from_f64(f64::INFINITY), None);
    assert_eq!(Float::from_f64(f64::NEG_INFINITY), None);

    // But every finite one is accepted, including the subnormals.
    assert!(Float::from_f64(f64::MIN_POSITIVE).is_some());
    assert!(Float::from_f64(f64::from_bits(1)).is_some());
    assert_eq!(Float::from_f64(f64::from_bits(1)).unwrap().to_f64(), f64::from_bits(1));
}

/// Multiplication of two `f64`s is *exact* at 256 bits — two 53-bit significands
/// make 106 — so rounding the result back must match `f64` multiplication bit for
/// bit.
#[test]
fn multiplication_matches_f64() {
    for left in probes() {
        for right in probes() {
            let expected: f64 = left * right;

            if !expected.is_normal() && expected != 0.0 {
                continue; // subnormal results round twice; tested separately
            }

            let product: Float =
                Float::from_f64(left).unwrap() * Float::from_f64(right).unwrap();

            assert_eq!(product.to_f64(), expected, "{left} * {right}");
        }
    }
}

/// Addition likewise, for inputs whose exponents are close enough that the exact
/// sum fits in 256 bits.
#[test]
fn addition_matches_f64() {
    let probes: Vec<f64> = probes();
    let mut compared: usize = 0;

    for left in &probes {
        for right in &probes {
            let expected: f64 = left + right;

            if !expected.is_normal() && expected != 0.0 {
                continue;
            }

            // The exact sum needs |exponent difference| + 53 bits. Well inside 256
            // for these, but check rather than assume.
            let gap: i32 = (left.abs().log2() - right.abs().log2()).abs() as i32;
            if *left != 0.0 && *right != 0.0 && gap > 150 {
                continue;
            }

            let sum: Float =
                Float::from_f64(*left).unwrap() + Float::from_f64(*right).unwrap();

            assert_eq!(sum.to_f64(), expected, "{left} + {right}");
            compared += 1;
        }
    }

    assert!(compared > 2_000, "only {compared} pairs compared");
}

#[test]
fn subtraction_matches_f64() {
    for left in probes() {
        for right in probes() {
            let expected: f64 = left - right;

            if !expected.is_normal() && expected != 0.0 {
                continue;
            }

            let gap: i32 = (left.abs().log2() - right.abs().log2()).abs() as i32;
            if left != 0.0 && right != 0.0 && gap > 150 {
                continue;
            }

            let difference: Float =
                Float::from_f64(left).unwrap() - Float::from_f64(right).unwrap();

            assert_eq!(difference.to_f64(), expected, "{left} - {right}");
        }
    }
}

#[test]
fn ordering_matches_f64() {
    let probes: Vec<f64> = probes();

    for left in &probes {
        for right in &probes {
            let wide = Float::from_f64(*left)
                .unwrap()
                .cmp(&Float::from_f64(*right).unwrap());

            assert_eq!(
                wide,
                left.partial_cmp(right).expect("no NaN here"),
                "comparing {left} with {right}"
            );
        }
    }
}

// ===========================================================================
// The point of the type: precision f64 cannot reach
// ===========================================================================

#[test]
fn carries_far_more_precision_than_f64() {
    // 1/3 to 256 bits has 77 correct decimal digits; an f64 manages about 16.
    let third: Float = Float::from_integer(1) / Float::from_integer(3);
    let text: String = third.to_string();

    let threes: usize = text
        .trim_start_matches("0.")
        .chars()
        .take_while(|digit| *digit == '3')
        .count();

    assert!(threes >= 76, "only {threes} correct digits: {text}");

    // And the f64 version diverges after about sixteen.
    let narrow: String = (1.0f64 / 3.0).to_string();
    assert!(narrow.len() < 25, "f64 gives {narrow}");
}

#[test]
fn a_thousand_bit_significand_is_more_precise_still() {
    type Huge = WideFloat<16>; // 1024 bits

    assert_eq!(Huge::PRECISION, 1024);

    let third: Huge = Huge::from_integer(1) / Huge::from_integer(3);
    let threes: usize = third
        .to_string()
        .trim_start_matches("0.")
        .chars()
        .take_while(|digit| *digit == '3')
        .count();

    // 1024 bits is about 308 decimal digits.
    assert!(threes >= 305, "only {threes} correct digits");
}

#[test]
fn accumulates_without_the_drift_an_f64_shows() {
    // A tenth added ten million times. In f64 the error accumulates visibly; with
    // 256 bits the rounding stays far below what any decimal readout shows.
    let tenth: Float = Float::from_integer(1) / Float::from_integer(10);
    let mut total: Float = Float::ZERO;
    let mut narrow: f64 = 0.0;

    for _ in 0..1_000_000 {
        total += tenth;
        narrow += 0.1;
    }

    // The exact answer is 100000.
    let exact: Float = Float::from_integer(100_000);
    let error: Float = (total - exact).abs();

    // The f64 sum is visibly off; ours is not, at any scale a person would read.
    assert_ne!(narrow, 100_000.0, "f64 is expected to drift");
    assert!(
        error < Float::from_integer(1) / Float::from_integer(1_000_000_000),
        "drifted to {total}"
    );
}

// ===========================================================================
// Structure and invariants
// ===========================================================================

#[test]
fn has_no_negative_zero() {
    let zero: Float = Float::ZERO;
    let negated: Float = -zero;

    assert_eq!(zero, negated);
    assert!(!negated.is_negative());

    // And a subtraction that cancels exactly gives the one canonical zero.
    let value: Float = Float::from_integer(7) / Float::from_integer(9);
    let cancelled: Float = value - value;

    assert_eq!(cancelled, Float::ZERO);
    assert!(!cancelled.is_negative());
    assert!(cancelled.is_zero());
}

/// No NaN means the order is total, so the type can key an ordered map — which an
/// `f64` cannot do without a wrapper.
#[test]
fn can_key_an_ordered_map() {
    let mut map: BTreeMap<Float, &str> = BTreeMap::new();

    map.insert(Float::from_integer(1) / Float::from_integer(3), "a third");
    map.insert(Float::from_integer(-2), "minus two");
    map.insert(Float::ZERO, "zero");

    let order: Vec<&str> = map.values().copied().collect();
    assert_eq!(order, vec!["minus two", "zero", "a third"]);

    // And every value equals itself, unlike NaN.
    for key in map.keys() {
        assert_eq!(key, key);
    }
}

#[test]
fn refuses_division_by_zero_rather_than_returning_an_infinity() {
    let one: Float = Float::one();

    assert_eq!(one.checked_div(&Float::ZERO), None);
    assert_eq!(Float::ZERO.checked_div(&Float::ZERO), None);
    assert_eq!(Field::inverse(&Float::ZERO), None);

    // Zero divided by something is plain zero.
    assert_eq!(Float::ZERO.checked_div(&one), Some(Float::ZERO));
}

#[test]
#[should_panic(expected = "division by zero")]
fn the_division_operator_panics_on_zero() {
    let _ = Float::one() / Float::ZERO;
}

#[test]
fn every_value_is_normalised() {
    // A non-zero significand always has its top bit set, which is what lets the
    // exponent be compared before the significand.
    for (numerator, divisor, ..) in DIVISION_CASES {
        let quotient: Float =
            Float::from_integer(*numerator) / Float::from_integer(*divisor);

        assert!(
            quotient.significand().bit(Float::PRECISION - 1),
            "{numerator} / {divisor} is not normalised"
        );
    }

    // Zero is the one exception, and is exactly zero.
    assert!(Float::ZERO.significand().is_zero());
    assert_eq!(Float::ZERO.exponent().to_i128(), Some(0));
}

// ===========================================================================
// Rounding
// ===========================================================================

#[test]
fn rounds_ties_to_even() {
    // Build a value whose last bit is exactly on the halfway mark: the significand
    // of all ones, plus a half of its last place.
    //
    // 2^PRECISION - 1, then adding 1/2 ulp. Ties to even rounds up here, because
    // the low bit is one, and the carry makes the significand a power of two.
    let all_ones: Float = Float::from_arb_uint(&WideUint::MAX);
    let half_ulp: Float = Float::from_arb_uint(&WideUint::one())
        / Float::from_integer(2);

    let sum: Float = all_ones + half_ulp;

    // Rounding up carried all the way, giving 2^PRECISION.
    assert_eq!(sum.significand(), WideUint::one().wrapping_shl(Float::PRECISION - 1));
    assert_eq!(sum.exponent().to_i128(), Some(1));

    // The other direction: an even last bit on an exact tie stays put.
    let even: Float = Float::from_arb_uint(&WideUint::MAX.wrapping_sub(&WideUint::one()));
    assert!(!even.significand().bit(0), "the last bit should be even");

    let stayed: Float = even + half_ulp;
    assert_eq!(stayed.significand(), even.significand(), "a tie to even holds");
}

#[test]
fn a_negligible_addend_leaves_the_value_alone() {
    let one: Float = Float::one();
    // Far below half a unit in the last place, in both directions.
    let tiny: Float = Float::from_f64(2f64.powi(-400)).unwrap();

    assert_eq!(one + tiny, one);
    assert_eq!(one - tiny, one);
    assert_eq!(-one + tiny, -one);
}

#[test]
fn cancellation_keeps_the_bits_it_should() {
    // Two values differing in only their last bit. Subtracting them must give
    // exactly one unit in the last place, not zero and not a value padded with
    // invented bits.
    let base: WideUint<4> = WideUint::MAX;
    let large: Float = Float::from_arb_uint(&base);
    let small: Float = Float::from_arb_uint(&base.wrapping_sub(&WideUint::one()));

    let difference: Float = large - small;

    assert_eq!(difference, Float::one(), "difference of neighbours is one");

    // And the halving case, where the gap is one and cancellation is severe: this
    // is the path that needs the full low half rather than two rounding bits.
    let half_step: Float = Float::from_arb_uint(&base) / Float::from_integer(2);
    let recovered: Float = Float::from_arb_uint(&base) - half_step;

    assert_eq!(recovered, half_step, "x - x/2 == x/2");
}

// ===========================================================================
// Exponent range
// ===========================================================================

#[test]
fn reaches_exponents_no_primitive_can() {
    // 2^100000, far past f64's 2^1024 ceiling.
    let two: Float = Float::from_integer(2);
    let huge: Float = two.power(100_000);

    // The significand is an *integer* in [2^(PRECISION-1), 2^PRECISION), not a
    // value in [1, 2) as IEEE 754 uses. So 2^100000 is held as
    // `2^(PRECISION-1) × 2^(100000 - (PRECISION-1))` and the stored exponent is
    // short of 100000 by exactly `PRECISION - 1`.
    assert_eq!(huge.significand(), WideUint::one().wrapping_shl(Float::PRECISION - 1));
    assert_eq!(
        huge.exponent().to_i128(),
        Some(100_000 - i128::from(Float::PRECISION - 1))
    );
    assert_eq!(huge.to_f64(), f64::INFINITY, "an f64 cannot hold it");

    // The value itself is what it should be, whatever the representation:
    assert_eq!(huge / two.power(99_999), two);

    // And the reciprocal is equally far below.
    let tiny: Float = Float::one() / huge;
    assert_eq!(
        tiny.exponent().to_i128(),
        Some(-100_000 - i128::from(Float::PRECISION - 1))
    );
    assert_eq!(tiny.to_f64(), 0.0, "an f64 rounds it to zero");

    // But the two still multiply back to one exactly, since both are powers of two.
    assert_eq!(huge * tiny, Float::one());
}

#[test]
fn a_wider_exponent_reaches_further_still() {
    // A single-limb exponent already reaches 2^(9.2e18); two limbs reach 2^(1.7e38).
    let two: Wide = Wide::from_integer(2);
    let huge: Wide = two.power(1_000_000);

    // Again offset by `PRECISION - 1`, as above.
    let offset: i128 = i128::from(Wide::PRECISION - 1);
    assert_eq!(huge.exponent().to_i128(), Some(1_000_000 - offset));

    // Squaring repeatedly, to an exponent past what one limb could hold.
    let mut value: Wide = huge;
    for _ in 0..45 {
        value = value * value;
    }

    // Squaring doubles the true exponent 45 times.
    let exponent: i128 = value.exponent().to_i128().expect("fits in i128");
    assert_eq!(exponent, (1_000_000_i128 << 45) - offset);
    assert!(
        exponent > i128::from(i64::MAX),
        "past a single limb's range, which is the point"
    );
}

#[test]
fn falls_back_to_hexadecimal_for_an_unprintable_exponent() {
    let two: Float = Float::from_integer(2);
    let huge: Float = two.power(1_000_000);

    // A million binary digits is 300,000 decimal ones; the decimal form is refused
    // and the exact hexadecimal one given instead.
    let text: String = huge.to_string();
    assert!(text.contains('p'), "expected the hex form, got {text}");
    assert!(text.starts_with("0x"), "{text}");

    // Small exponents still print in decimal.
    assert_eq!(Float::from_integer(5).to_string(), "5");
    assert_eq!((Float::one() / Float::from_integer(4)).to_string(), "0.25");
}

#[test]
fn hexadecimal_is_exact_and_names_both_parts() {
    // One is 2^(PRECISION-1) × 2^-(PRECISION-1).
    let one: Float = Float::one();
    let text: String = format!("{one:x}");

    assert_eq!(
        text,
        "0x8000000000000000000000000000000000000000000000000000000000000000p-255"
    );

    assert_eq!(format!("{:x}", Float::ZERO), "0x0p0");
    assert_eq!(format!("{:x}", -Float::one()), format!("-{text}"));
}

// ===========================================================================
// Algebra
// ===========================================================================

#[test]
fn the_ring_operations_behave() {
    let a: Float = Float::from_integer(3) / Float::from_integer(7);
    let b: Float = Float::from_integer(11) / Float::from_integer(13);

    assert_eq!(a + b, b + a, "addition commutes");
    assert_eq!(a * b, b * a, "multiplication commutes");
    assert_eq!(a - a, Float::ZERO);
    assert_eq!(a + Float::ZERO, a);
    assert_eq!(a * Float::one(), a);
    assert_eq!(a * Float::ZERO, Float::ZERO);
    assert_eq!(-(-a), a);

    // What does *not* hold: `(a / b) * b == a`. Each of the two operations rounds,
    // and the second rounding does not generally undo the first. This is double
    // rounding, it is true of `f64` too, and it is the price of fixed precision —
    // so the test states the real guarantee, which is a bound on the error.
    let round_trip: Float = (a / b) * b;
    let error: Float = (round_trip - a).abs();
    let bound: Float = a.abs() / Float::from_integer(2).power(Float::PRECISION - 4);

    assert_ne!(round_trip, a, "double rounding is expected here");
    assert!(error <= bound, "error {error} exceeded {bound}");

    // Powers of two are the case where it *is* exact, because nothing rounds.
    let eight: Float = Float::from_integer(8);
    assert_eq!((a / eight) * eight, a);
}

#[test]
fn inverse_comes_from_the_field_trait() {
    let value: Float = Float::from_integer(7);
    let inverse: Float = Field::inverse(&value).expect("not zero");

    assert_eq!(value * inverse, Float::one());

    // A power of two inverts exactly.
    let eight: Float = Float::from_integer(8);
    assert_eq!(
        Field::inverse(&eight),
        Some(Float::one() / eight)
    );
}

#[test]
fn power_comes_from_the_semiring_trait() {
    let two: Float = Float::from_integer(2);

    assert_eq!(two.power(0), Float::one());
    assert_eq!(two.power(10), Float::from_integer(1024));

    let half: Float = Float::one() / two;
    assert_eq!(half.power(10), Float::one() / Float::from_integer(1024));
}

#[test]
fn zero_and_one_behave() {
    assert!(Zero::is_zero(&Float::ZERO));
    assert!(One::is_one(&Float::one()));
    assert!(!Zero::is_zero(&Float::one()));
    assert!(!One::is_one(&Float::ZERO));
    assert_eq!(<Float as Zero>::zero(), Float::ZERO);
    assert_eq!(<Float as One>::one(), Float::one());
}

#[test]
fn works_at_one_limb_of_significand() {
    type Narrow = WideFloat<1>; // 64 bits, a little more than an f64

    assert_eq!(Narrow::PRECISION, 64);

    let third: Narrow = Narrow::from_integer(1) / Narrow::from_integer(3);
    let threes: usize = third
        .to_string()
        .trim_start_matches("0.")
        .chars()
        .take_while(|digit| *digit == '3')
        .count();

    // 64 bits is about 19 decimal digits — better than an f64's 16, as expected.
    assert!((18..=20).contains(&threes), "{threes} digits");
    assert_eq!(third * Narrow::from_integer(3), Narrow::one());
}
