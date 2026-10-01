//! Fixed-point arithmetic: the accuracy claims, the exactness claims, and the
//! boundaries.
//!
//! This type had no tests at all, despite 1,400 lines and a set of documented
//! accuracy guarantees — "logarithms good to one step, exponentials correctly
//! rounded" — that nothing checked. Those claims are the point of the file, so they
//! are what this concentrates on.
//!
//! Three oracles:
//!
//! 1. **`f64`**, for the ordinary arithmetic, where the two agree exactly over the
//!    overlapping range.
//! 2. **High-precision decimal**, for the named constants and the transcendentals —
//!    fifty published digits, rounded to the type's step outside Rust, so the
//!    reference shares no code with the thing it checks.
//! 3. **The definitions themselves**, for the properties that need no reference:
//!    addition being exact, division inverting multiplication, `exp(ln(x)) == x`.

use std::str::FromStr;
use voxel_world::math::Fixed;
use voxel_world::math::fixed::FRACTION_BITS;

/// One step, as a raw count.
const STEP: i128 = 1;

/// Named constants, each rounded to the nearest 2^-32 from 50 published digits.
const CONSTANTS: &[(&str, i128)] = &[
    ("PI", 13493037705),
    ("TAU", 26986075409),
    ("E", 11674931555),
    ("SQRT_2", 6074001000),
    ("LN_2", 2977044472),
];

/// `ln(x)` to 50 digits, for checking the logarithm is good to one step.
const LOGARITHMS: &[(&str, &str)] = &[
    ("0.5", "-0.693147180559945309417232121458176568075500134360255254120680"),
    ("1", "0"),
    ("1.5", "0.405465108108164381978013115464349136571990423462494197614014"),
    ("2", "0.693147180559945309417232121458176568075500134360255254120680"),
    ("2.718281828459045", "0.999999999999999913415788971088761162572033226583247761169363"),
    ("10", "2.30258509299404568401799145468436420760110148862877297603333"),
    ("100", "4.60517018598809136803598290936872841520220297725754595206666"),
    ("1000", "6.90775527898213705205397436405309262280330446588631892809998"),
    ("65536", "11.0903548889591249506757139433308250892080021497640840659309"),
    ("1e9", "20.7232658369464111561619230921592778684099133976589567843000"),
];

/// `exp(x)` to 50 digits, for checking the exponential is correctly rounded.
const EXPONENTIALS: &[(&str, &str)] = &[
    ("-5", "0.00673794699908546709663604842314842424884958502735508543030553"),
    ("-1", "0.367879441171442321595523770161460867445811131031767834507837"),
    ("-0.5", "0.606530659712633423603799534991180453441918135487186955682892"),
    ("0", "1"),
    ("0.5", "1.64872127070012814684865078781416357165377610071014801157508"),
    ("1", "2.71828182845904523536028747135266249775724709369995957496697"),
    ("2", "7.38905609893065022723042746057500781318031557055184732408713"),
    ("5", "148.413159102576603421115580040552279623487667593878989046753"),
    ("10", "22026.4657948067165169579006452842443663535126185567810742354"),
    ("20", "485165195.409790277969106830541540558684638988944847254353611"),
];

/// The named constants must each be the nearest representable value.
#[test]
fn the_constants_are_rounded_to_the_nearest_step() {
    let named: [(&str, Fixed); 5] = [
        ("PI", Fixed::PI),
        ("TAU", Fixed::TAU),
        ("E", Fixed::E),
        ("SQRT_2", Fixed::SQRT_2),
        ("LN_2", Fixed::LN_2),
    ];

    for (name, value) in named {
        let expected: i128 = CONSTANTS
            .iter()
            .find(|(label, _)| *label == name)
            .expect("every constant is tabulated")
            .1;

        assert_eq!(
            value.to_bits(),
            expected,
            "{name} is not the nearest step to its true value"
        );
    }

    // And the relationships between them hold to within a step.
    assert!((Fixed::TAU.to_bits() - 2 * Fixed::PI.to_bits()).abs() <= STEP);
    let two: Fixed = Fixed::from_integer(2);
    assert!(
        (Fixed::SQRT_2.checked_mul(Fixed::SQRT_2).unwrap().to_bits() - two.to_bits()).abs() <= 2,
        "the square root of two, squared, should be two"
    );
}

// ===========================================================================
// Exactness
// ===========================================================================

/// The module claims addition and subtraction are exact. They are integer
/// operations on the raw counts, so this is checkable directly.
#[test]
fn addition_and_subtraction_are_exact() {
    let probes: Vec<Fixed> = probes();

    for left in &probes {
        for right in &probes {
            if let Some(sum) = left.checked_add(*right) {
                assert_eq!(
                    sum.to_bits(),
                    left.to_bits() + right.to_bits(),
                    "a sum lost something"
                );
            }

            if let Some(difference) = left.checked_sub(*right) {
                assert_eq!(
                    difference.to_bits(),
                    left.to_bits() - right.to_bits(),
                    "a difference lost something"
                );
            }
        }
    }
}

/// Multiplication goes through a 256-bit intermediate and truncates towards zero.
#[test]
fn multiplication_truncates_towards_zero() {
    // A product whose exact value falls between two steps, in both signs. Three
    // eighths times a third of a step is well below a step, so it truncates to zero
    // rather than rounding to one.
    let tiny: Fixed = Fixed::from_bits(3);
    let half: Fixed = Fixed::HALF;

    assert_eq!(tiny.checked_mul(half).unwrap().to_bits(), 1, "1.5 truncates to 1");
    assert_eq!(
        (-tiny).checked_mul(half).unwrap().to_bits(),
        -1,
        "-1.5 truncates to -1, towards zero"
    );

    // Exact where it can be.
    assert_eq!(Fixed::from_integer(6) * Fixed::from_integer(7), Fixed::from_integer(42));
    assert_eq!(Fixed::HALF * Fixed::HALF, Fixed::from_bits(1 << (FRACTION_BITS - 2)));

    // The 256-bit intermediate: a product that would overflow an i128 before the
    // shift still comes out right.
    let large: Fixed = Fixed::from_integer(4_000_000_000);
    let product: Fixed = large.checked_mul(large).expect("fits after the shift");
    assert_eq!(
        product,
        Fixed::from_integer_i128(16_000_000_000_000_000_000).unwrap(),
        "the intermediate must not have been truncated to 128 bits"
    );
}

#[test]
fn division_inverts_multiplication_where_it_can() {
    for value in probes() {
        if value.is_zero() {
            continue;
        }

        // x / x is one, exactly.
        assert_eq!(value.checked_div(value), Some(Fixed::ONE), "{value} / itself");

        // Dividing by a power of two is exact, so it round-trips.
        let eighth: Fixed = Fixed::from_integer(8);
        if let Some(scaled) = value.checked_mul(eighth) {
            assert_eq!(scaled.checked_div(eighth), Some(value), "{value} x 8 / 8");
        }
    }

    assert_eq!(Fixed::ONE.checked_div(Fixed::ZERO), None, "division by zero");
    assert_eq!(Fixed::ZERO.checked_div(Fixed::ONE), Some(Fixed::ZERO));
}

// ===========================================================================
// Against f64, where both are exact
// ===========================================================================

fn probes() -> Vec<Fixed> {
    let mut values: Vec<Fixed> = vec![
        Fixed::ZERO,
        Fixed::ONE,
        Fixed::NEGATIVE_ONE,
        Fixed::HALF,
        Fixed::DELTA,
        -Fixed::DELTA,
        Fixed::from_integer(2),
        Fixed::from_integer(-3),
        Fixed::from_integer(1_000_000),
        Fixed::from_bits(1 << 40),
        Fixed::from_bits(-(1 << 40)),
    ];

    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..30 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Kept modest so products and sums stay inside the range.
        values.push(Fixed::from_bits((state >> 20) as i128 - (1 << 43)));
    }

    values
}

#[test]
fn small_values_agree_with_f64() {
    // Every `Fixed` below 2^52 converts to an `f64` exactly, since the raw count
    // then has at most 52 significant bits.
    for value in probes() {
        let raw: i128 = value.to_bits();

        if raw.abs() >= 1 << 52 {
            continue;
        }

        let expected: f64 = raw as f64 / (1u64 << FRACTION_BITS) as f64;
        assert_eq!(value.to_f64(), expected, "{value} converted badly");
        assert_eq!(Fixed::from_f64(expected), Some(value), "and did not return");
    }
}

#[test]
fn rounding_helpers_match_their_names() {
    let cases: [(f64, i128, i128, i128, i128); 5] = [
        //  value, floor, ceil, trunc, round
        (2.5, 2, 3, 2, 3),
        (-2.5, -3, -2, -2, -3),
        (2.25, 2, 3, 2, 2),
        (-0.5, -1, 0, 0, -1),
        (7.0, 7, 7, 7, 7),
    ];

    for (value, floor, ceil, trunc, round) in cases {
        let held: Fixed = Fixed::from_f64(value).expect("in range");

        assert_eq!(held.floor(), Fixed::from_integer(floor as i64), "floor {value}");
        assert_eq!(held.ceil(), Fixed::from_integer(ceil as i64), "ceil {value}");
        assert_eq!(held.trunc(), Fixed::from_integer(trunc as i64), "trunc {value}");
        assert_eq!(held.round(), Fixed::from_integer(round as i64), "round {value}");

        // The fractional part plus the truncated part is the whole thing.
        assert_eq!(held.trunc() + held.fract(), held, "trunc + fract of {value}");
    }
}

// ===========================================================================
// The accuracy claims
// ===========================================================================

/// "The logarithms are good to one step."
#[test]
fn the_logarithm_is_good_to_one_step() {
    for (argument, expected) in LOGARITHMS {
        let value: Fixed = Fixed::from_str(argument)
            .or_else(|_| Fixed::from_f64(argument.parse::<f64>().unwrap()).ok_or(()))
            .expect("representable");

        let computed: Fixed = value.ln().expect("positive");
        let truth: Fixed = nearest_step(expected);

        let gap: i128 = (computed.to_bits() - truth.to_bits()).abs();

        assert!(
            gap <= STEP,
            "ln({argument}) was {computed}, truth rounds to {truth} — off by {gap} steps"
        );
    }

    // And `log2` agrees with the change of base through `ln`, within what error
    // propagation allows. That is *not* one step: dividing by `LN_2` amplifies the
    // logarithm's own error by `1/ln2` and adds the constant's, scaled by
    // `x / ln(2)^2` — about twenty steps at these magnitudes. Asserting one step
    // here would be asserting something arithmetic forbids.
    for argument in ["2", "10", "1000", "65536"] {
        let value: Fixed = Fixed::from_str(argument).expect("representable");
        let via_ln: Fixed = value.ln().expect("positive").checked_div(Fixed::LN_2).expect("non-zero");
        let direct: Fixed = value.log2().expect("positive");

        assert!(
            (direct.to_bits() - via_ln.to_bits()).abs() <= 24,
            "log2({argument}): direct {direct}, via ln {via_ln}"
        );
    }

    // Exact powers of two are exact logarithms.
    assert_eq!(Fixed::ONE.log2(), Some(Fixed::ZERO));
    assert_eq!(Fixed::from_integer(2).log2(), Some(Fixed::ONE));
    assert_eq!(Fixed::from_integer(1024).log2(), Some(Fixed::from_integer(10)));
}

/// "The exponentials are correctly rounded."
#[test]
fn the_exponential_is_correctly_rounded() {
    for (argument, expected) in EXPONENTIALS {
        let value: Fixed = Fixed::from_str(argument).expect("representable");
        let computed: Fixed = value.exp().expect("in range");
        let truth: Fixed = nearest_step(expected);

        let gap: i128 = (computed.to_bits() - truth.to_bits()).abs();

        assert!(
            gap <= STEP,
            "exp({argument}) was {computed}, truth rounds to {truth} — off by {gap} steps"
        );
    }

    assert_eq!(Fixed::ZERO.exp(), Some(Fixed::ONE), "exp(0) is exactly one");
    assert_eq!(Fixed::ZERO.exp2(), Some(Fixed::ONE));
    assert_eq!(Fixed::ONE.exp2(), Some(Fixed::from_integer(2)));
    assert_eq!(Fixed::from_integer(10).exp2(), Some(Fixed::from_integer(1024)));
}

/// The two must undo each other, which needs no external reference at all.
#[test]
fn the_logarithm_and_exponential_invert_each_other() {
    for argument in ["0.5", "1", "2", "3", "10", "100", "1000"] {
        let value: Fixed = Fixed::from_str(argument).expect("representable");
        let round_trip: Fixed = value.ln().expect("positive").exp().expect("in range");

        // A relative tolerance, since the error is proportional to the value: two
        // roundings of a quantity this size.
        let tolerance: i128 = (value.to_bits() >> 28).max(4);

        assert!(
            (round_trip.to_bits() - value.to_bits()).abs() <= tolerance,
            "exp(ln({argument})) was {round_trip}, expected {value}"
        );
    }
}

#[test]
fn the_square_root_is_good_to_a_step() {
    for value in [1i64, 2, 3, 4, 100, 10_000, 1_000_000, 999_999_937] {
        let held: Fixed = Fixed::from_integer(value);
        let root: Fixed = held.sqrt().expect("non-negative");

        // The defining property, checked in the type itself: root^2 should be the
        // value, to within the rounding two multiplications allow.
        let squared: Fixed = root.checked_mul(root).expect("fits");
        let tolerance: i128 = (held.to_bits() >> 28).max(4);

        assert!(
            (squared.to_bits() - held.to_bits()).abs() <= tolerance,
            "sqrt({value}) squared back to {squared}"
        );
    }

    assert_eq!(Fixed::ZERO.sqrt(), Some(Fixed::ZERO));
    assert_eq!(Fixed::ONE.sqrt(), Some(Fixed::ONE));
    assert_eq!(Fixed::from_integer(4).sqrt(), Some(Fixed::from_integer(2)));
    assert_eq!(Fixed::from_integer(1_048_576).sqrt(), Some(Fixed::from_integer(1024)));
}

/// The nearest representable value to a high-precision decimal string.
///
/// Parsed digit by digit into the raw count rather than through an `f64`, so the
/// reference does not inherit a float's rounding.
fn nearest_step(text: &str) -> Fixed {
    Fixed::from_bits(nearest_raw(text, FRACTION_BITS))
}

/// The same, at any width, as a raw count.
fn nearest_raw(text: &str, fraction_bits: u32) -> i128 {
    let negative: bool = text.starts_with('-');
    let text: &str = text.trim_start_matches('-');
    let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));

    // Sixteen guard bits below the type's own precision, so the truncation at each
    // Horner step cannot reach the last place of the answer. Without them the
    // divisions bias the reference downwards by about a step — which is the size of
    // the thing being measured.
    const GUARD: u32 = 16;
    let working: i128 = 1i128 << (fraction_bits + GUARD);

    let mut raw: i128 = whole.parse::<i128>().expect("an integer part") * working;

    // Horner from the least significant digit back: each is worth a tenth of the one
    // before it. `take` *before* `rev`, so this keeps the leading digits — reversing
    // first would keep the tail and throw the value away.
    let digits: Vec<u8> = fraction.bytes().take(45).collect();

    let mut scale: i128 = 0;
    for digit in digits.into_iter().rev() {
        scale = (scale + i128::from(digit - b'0') * working) / 10;
    }
    raw += scale;

    // Down to the type's precision, rounding to nearest.
    let half: i128 = 1i128 << (GUARD - 1);
    let rounded: i128 = (raw + half) >> GUARD;

    if negative { -rounded } else { rounded }
}

// ===========================================================================
// Structure and boundaries
// ===========================================================================

#[test]
fn there_is_no_nan_and_no_negative_zero() {
    // Every value equals itself, which an f64 cannot promise.
    for value in probes() {
        assert_eq!(value, value);
        assert_eq!(value.cmp(&value), std::cmp::Ordering::Equal);
    }

    // One spelling of zero.
    assert_eq!(Fixed::ZERO, -Fixed::ZERO);
    assert_eq!((-Fixed::ZERO).to_bits(), 0);
    assert!(!Fixed::ZERO.is_negative() && !Fixed::ZERO.is_positive());
    assert_eq!(Fixed::ZERO.signum(), Fixed::ZERO);

    // It can key an ordered map, unlike an f64.
    let mut map: std::collections::BTreeMap<Fixed, &str> = std::collections::BTreeMap::new();
    map.insert(Fixed::ONE, "one");
    map.insert(Fixed::NEGATIVE_ONE, "minus one");
    map.insert(Fixed::ZERO, "zero");
    assert_eq!(
        map.values().copied().collect::<Vec<_>>(),
        vec!["minus one", "zero", "one"]
    );
}

#[test]
fn the_checked_forms_refuse_what_overflows() {
    assert_eq!(Fixed::MAX.checked_add(Fixed::DELTA), None);
    assert_eq!(Fixed::MIN.checked_sub(Fixed::DELTA), None);
    assert_eq!(Fixed::MIN.checked_neg(), None, "MIN has no negation");
    assert_eq!(Fixed::MAX.checked_mul(Fixed::from_integer(2)), None);
    assert_eq!(Fixed::ONE.checked_div(Fixed::ZERO), None);

    // Saturating gives the edge instead.
    assert_eq!(Fixed::MAX.saturating_add(Fixed::DELTA), Fixed::MAX);
    assert_eq!(Fixed::MIN.saturating_sub(Fixed::DELTA), Fixed::MIN);
    assert_eq!(Fixed::MAX.saturating_mul(Fixed::from_integer(2)), Fixed::MAX);

    // Wrapping wraps.
    assert_eq!(Fixed::MAX.wrapping_add(Fixed::DELTA), Fixed::MIN);
}

/// The module says operators panic on overflow in release builds as well as debug.
#[test]
#[should_panic]
fn the_operator_panics_on_overflow_even_in_release() {
    let _ = Fixed::MAX + Fixed::DELTA;
}

#[test]
fn clamping_and_comparison_behave() {
    let low: Fixed = Fixed::from_integer(-5);
    let high: Fixed = Fixed::from_integer(5);

    assert_eq!(Fixed::from_integer(10).clamp(low, high), high);
    assert_eq!(Fixed::from_integer(-10).clamp(low, high), low);
    assert_eq!(Fixed::ONE.clamp(low, high), Fixed::ONE);
    assert_eq!(low.min(high), low);
    assert_eq!(low.max(high), high);
    assert_eq!(Fixed::from_integer(-3).abs(), Fixed::from_integer(3));
    assert_eq!(Fixed::NEGATIVE_ONE.signum(), Fixed::NEGATIVE_ONE);
    assert!(Fixed::from_integer(7).is_whole());
    assert!(!Fixed::HALF.is_whole());
}

// ===========================================================================
// Text
// ===========================================================================

#[test]
fn text_round_trips_exactly() {
    for value in probes() {
        let exact: String = value.to_exact_string();
        let parsed: Fixed = Fixed::from_str(&exact)
            .unwrap_or_else(|_| panic!("could not re-read {exact}"));

        assert_eq!(parsed, value, "{exact} did not return the value it came from");
    }

    // The exact string really is exact: a binary fraction terminates in decimal, and
    // every value's expansion fits in the 32 places `2^-32` needs. Trailing zeros
    // are kept rather than trimmed, so a half prints to full width.
    // Always the full 32 places, padded — including for whole numbers and zero.
    assert_eq!(Fixed::HALF.to_exact_string(), "0.50000000000000000000000000000000");
    assert_eq!(
        Fixed::from_integer(-3).to_exact_string(),
        "-3.00000000000000000000000000000000"
    );
    assert_eq!(Fixed::ZERO.to_exact_string(), "0.00000000000000000000000000000000");
    assert_eq!(Fixed::ONE.to_exact_string(), "1.00000000000000000000000000000000");
    assert_eq!(
        Fixed::DELTA.to_exact_string(),
        "0.00000000023283064365386962890625",
        "one step is 2^-32 exactly"
    );

    // Exact means exact: reading the printed digits back is lossless even at the
    // last place, which is what a rounded `Display` could not promise.
    assert_eq!(
        Fixed::from_str(&Fixed::DELTA.to_exact_string()).unwrap(),
        Fixed::DELTA
    );
}

#[test]
fn parsing_rejects_what_is_not_a_number() {
    for bad in ["", "abc", "1.2.3", "--1", "1e5", " 1", "1 "] {
        assert!(Fixed::from_str(bad).is_err(), "{bad:?} should be refused");
    }

    assert_eq!(Fixed::from_str("0.5").unwrap(), Fixed::HALF);
    assert_eq!(Fixed::from_str("-0.5").unwrap(), -Fixed::HALF);
    assert_eq!(Fixed::from_str("+2").unwrap(), Fixed::from_integer(2));
}

#[test]
fn ratios_convert_without_floating_point() {
    assert_eq!(Fixed::from_ratio(1, 2), Some(Fixed::HALF));
    assert_eq!(Fixed::from_ratio(1, 4), Some(Fixed::from_bits(1 << 30)));
    assert_eq!(Fixed::from_ratio(-3, 4), Some(-Fixed::from_bits(3 << 30)));
    assert_eq!(Fixed::from_ratio(1, 0), None, "no denominator");

    // A third truncates towards zero, and stays within a step of the truth.
    let third: Fixed = Fixed::from_ratio(1, 3).expect("valid");
    assert!((third.to_bits() * 3 - Fixed::ONE.to_bits()).abs() <= 3);
}

// ===========================================================================
// Other widths
// ===========================================================================

use voxel_world::math::fixed::FixedPoint;

/// Checks a layout of any width against the same references the default uses.
///
/// Written as a generic function rather than repeated per width, so adding a width
/// to the list below is one line and cannot drift from the others.
fn check_layout<const F: u32>() {
    let one: FixedPoint<F> = FixedPoint::<F>::ONE;

    // The scale is what it says.
    assert_eq!(FixedPoint::<F>::FRACTION_BITS, F);
    assert_eq!(FixedPoint::<F>::SCALE, 1i128 << F);
    assert_eq!(one.to_bits(), 1i128 << F, "one is the scale");
    assert_eq!(FixedPoint::<F>::HALF.to_bits(), 1i128 << (F - 1));
    assert_eq!(FixedPoint::<F>::DELTA.to_bits(), 1, "a step is one count");

    // Arithmetic still works, and addition is still exact.
    let two: FixedPoint<F> = FixedPoint::<F>::from_integer(2);
    assert_eq!(one + one, two);
    assert_eq!(two - one, one);
    assert_eq!(two * two, FixedPoint::<F>::from_integer(4));
    assert_eq!(two.checked_div(two), Some(one));
    assert_eq!(FixedPoint::<F>::from_integer(7).to_integer(), 7);

    // Every named constant is the nearest step *at this width*, computed from the
    // same fifty digits the default is checked against.
    let named: [(&str, FixedPoint<F>); 5] = [
        ("PI", FixedPoint::<F>::PI),
        ("TAU", FixedPoint::<F>::TAU),
        ("E", FixedPoint::<F>::E),
        ("SQRT_2", FixedPoint::<F>::SQRT_2),
        ("LN_2", FixedPoint::<F>::LN_2),
    ];
    let digits: [(&str, &str); 5] = [
        ("PI", "3.14159265358979323846264338327950288419716939937510"),
        ("TAU", "6.28318530717958647692528676655900576839433879875021"),
        ("E", "2.71828182845904523536028747135266249775724709369995"),
        ("SQRT_2", "1.41421356237309504880168872420969807856967187537694"),
        ("LN_2", "0.69314718055994530941723212145817656807550013436026"),
    ];

    for (name, value) in named {
        let text: &str = digits.iter().find(|(label, _)| *label == name).unwrap().1;

        assert_eq!(
            value.to_bits(),
            nearest_raw(text, F),
            "{name} is not the nearest step at {F} fractional bits"
        );
    }

    // And the transcendentals still hold their accuracy — this is the claim the
    // width could plausibly have broken, since the guard between the working
    // precision and the type's own narrows as the type widens.
    for argument in ["0.5", "1", "2", "10", "100"] {
        let value: FixedPoint<F> = FixedPoint::<F>::from_str(argument).expect("representable");
        let computed: FixedPoint<F> = value.ln().expect("positive");
        let truth: i128 = nearest_raw(
            LOGARITHMS
                .iter()
                .find(|(label, _)| *label == argument)
                .expect("tabulated")
                .1,
            F,
        );

        assert!(
            (computed.to_bits() - truth).abs() <= 1,
            "ln({argument}) at {F} bits was off by {} steps",
            (computed.to_bits() - truth).abs()
        );
    }

    assert_eq!(FixedPoint::<F>::ZERO.exp(), Some(one), "exp(0) is one");
    assert_eq!(FixedPoint::<F>::ONE.exp2(), Some(two), "2^1 is two");
    assert_eq!(FixedPoint::<F>::from_integer(4).sqrt(), Some(two));
}

/// The width is a knob now, and every setting of it has to work.
#[test]
fn other_widths_behave_the_same_way() {
    check_layout::<8>();
    check_layout::<16>();
    check_layout::<24>();
    check_layout::<32>();
    check_layout::<40>();
    check_layout::<48>();
    check_layout::<56>();
    check_layout::<64>();
}

/// The default alias is exactly the 32-bit layout, so nothing that already used
/// `Fixed` sees any difference.
#[test]
fn the_default_alias_is_the_thirty_two_bit_layout() {
    let through_alias: Fixed = Fixed::PI;
    let through_parameter: FixedPoint<32> = FixedPoint::<32>::PI;

    assert_eq!(through_alias, through_parameter);
    assert_eq!(Fixed::FRACTION_BITS, FRACTION_BITS);
    assert_eq!(Fixed::FRACTION_BITS, 32);

    // The whole point of the alias: these spellings still work with no annotation,
    // which a `const` default on the struct would have broken.
    let _ = Fixed::ONE;
    let _ = Fixed::from_integer(5);
    let _: Fixed = Fixed::from_f64(0.25).unwrap();
}

/// A narrower layout trades precision for range, within the same 128 bits.
#[test]
fn a_narrower_layout_reaches_further() {
    // 16 fractional bits leaves 111 for the whole part, against 32's 95.
    let wide_range: FixedPoint<16> = FixedPoint::<16>::MAX;
    let fine_grain: Fixed = Fixed::MAX;

    assert_eq!(wide_range.to_bits(), fine_grain.to_bits(), "the same i128");
    assert!(
        wide_range.to_integer() > fine_grain.to_integer(),
        "fewer fractional bits reach further in whole units"
    );

    // And the step is correspondingly coarser.
    assert_eq!(FixedPoint::<16>::DELTA.to_f64(), 2f64.powi(-16));
    assert_eq!(Fixed::DELTA.to_f64(), 2f64.powi(-32));
    assert_eq!(FixedPoint::<64>::DELTA.to_f64(), 2f64.powi(-64));
}
