//! Decimal literals and decimal text, read exactly and never through a float.
//!
//! # What is actually being tested
//!
//! That `fixed!(0.1)` is the nearest `Fixed` to the rational `1/10` — not the nearest
//! `Fixed` to the nearest `f64` to `1/10`, which is a different number arrived at by
//! rounding twice.
//!
//! So the expected values here are computed from integer ratios inside the tests
//! themselves, with arithmetic that shares nothing with the parser under test. No
//! comparison goes through `f64` at any point; a test that did would be unable to
//! tell the two answers apart, which is the whole question.

use voxel_world::math::decimal::{self, Decimal, DecimalError};
use voxel_world::math::{Fixed, FixedPoint, Ratio, Unit};
use voxel_world::{fixed, fixed_point, unit};

/// `numerator / denominator` on a grid of `2^fraction_bits`, nearest, ties to even.
///
/// Written out here rather than called from the crate, so that it is an independent
/// check and not the implementation agreeing with itself.
fn expected_bits(numerator: u128, denominator: u128, fraction_bits: u32) -> u128 {
    let scaled: u128 = numerator << fraction_bits;
    let quotient: u128 = scaled / denominator;
    let remainder: u128 = scaled % denominator;
    let doubled: u128 = remainder * 2;

    if doubled > denominator || (doubled == denominator && quotient % 2 == 1) {
        quotient + 1
    } else {
        quotient
    }
}

// ---------------------------------------------------------------------------
// Values that land on the grid exactly
// ---------------------------------------------------------------------------

#[test]
fn dyadic_decimals_are_exact() {
    // Halves, quarters and eighths are representable, so no rounding is involved and
    // the answer can be stated as a fraction of ONE.
    assert_eq!(fixed!(0.5), Fixed::ONE / 2);
    assert_eq!(fixed!(0.25), Fixed::ONE / 4);
    assert_eq!(fixed!(1.5), Fixed::ONE + Fixed::ONE / 2);
    assert_eq!(fixed!(-17.625), Fixed::ZERO - (Fixed::from_integer(17) + Fixed::ONE * 5 / 8));
    assert_eq!(fixed!(0), Fixed::ZERO);
    assert_eq!(fixed!(1), Fixed::ONE);
    assert_eq!(fixed!(1000), Fixed::from_integer(1000));

    assert_eq!(unit!(0.5), Unit::HALF);
    assert_eq!(unit!(1.0), Unit::ONE);
    assert_eq!(unit!(1), Unit::ONE);
    assert_eq!(unit!(0.0), Unit::ZERO);
    assert_eq!(unit!(0), Unit::ZERO);
}

// ---------------------------------------------------------------------------
// Values that do not, which is where the double rounding would show
// ---------------------------------------------------------------------------

#[test]
fn non_dyadic_decimals_round_to_the_nearest_grid_point() {
    // A tenth is not representable in binary at any width, so each of these is a
    // genuine rounding, checked against the integer ratio it came from.
    let cases: [(Fixed, u128, u128); 6] = [
        (fixed!(0.1), 1, 10),
        (fixed!(0.2), 2, 10),
        (fixed!(0.3), 3, 10),
        (fixed!(0.7), 7, 10),
        (fixed!(0.333333), 333_333, 1_000_000),
        (fixed!(1.2), 12, 10),
    ];

    for (got, numerator, denominator) in cases {
        let expected: i128 = expected_bits(numerator, denominator, 32) as i128;

        assert_eq!(got.to_bits(), expected, "{numerator}/{denominator}");
    }
}

#[test]
fn unit_literals_round_to_the_nearest_grid_point() {
    let cases: [(Unit, u128, u128); 4] = [
        (unit!(0.1), 1, 10),
        (unit!(0.2), 2, 10),
        (unit!(0.025), 25, 1000),
        (unit!(0.333333), 333_333, 1_000_000),
    ];

    for (got, numerator, denominator) in cases {
        let expected: u64 = expected_bits(numerator, denominator, 63) as u64;

        assert_eq!(got.to_bits(), expected, "{numerator}/{denominator}");
    }
}

#[test]
fn a_literal_beats_the_route_through_a_float() {
    // The point of the whole exercise. `0.1` as an `f64` is slightly above a tenth,
    // and converting that gives a different raw integer from rounding the rational
    // once. If these two agreed, the macro would not be doing anything.
    let exact: Fixed = fixed!(0.1);
    let expected: i128 = expected_bits(1, 10, 32) as i128;

    assert_eq!(exact.to_bits(), expected, "the literal is the nearest to 1/10");

    // Documented rather than asserted as a difference, because at 32 fractional bits
    // the two happen to coincide for many values; what matters is which question was
    // answered, and `from_f64` truncates where the literal rounds.
    let through_float: Fixed = Fixed::from_f64(0.1).unwrap();

    assert!(
        (exact.to_bits() - through_float.to_bits()).abs() <= 1,
        "the two routes differ by at most a step, and by a different rule"
    );

    // At a width where a tenth needs more bits, the rules part company visibly:
    // truncation goes down, nearest goes up.
    let wide_exact: FixedPoint<60> = fixed_point!(60, 0.1);
    let wide_expected: i128 = expected_bits(1, 10, 60) as i128;

    assert_eq!(wide_exact.to_bits(), wide_expected);
    assert_ne!(
        wide_exact.to_bits(),
        FixedPoint::<60>::from_f64(0.1).unwrap().to_bits(),
        "truncating an f64 and rounding the rational are not the same operation"
    );
}

// ---------------------------------------------------------------------------
// Syntax
// ---------------------------------------------------------------------------

#[test]
fn scientific_notation_agrees_with_the_written_out_form() {
    assert_eq!(fixed!(1e3), fixed!(1000));
    assert_eq!(fixed!(3e6), fixed!(3_000_000));
    assert_eq!(fixed!(0.001), fixed!(1e-3));
    assert_eq!(fixed!(1.25e-4), fixed!(0.000125));
    assert_eq!(fixed!(1E6), fixed!(1e6));
    assert_eq!(fixed!(1e+6), fixed!(1e6));

    assert_eq!(unit!(0.01), unit!(1e-2));
    assert_eq!(unit!(2.5e-3), unit!(0.0025));
}

#[test]
fn underscores_are_ignored_wherever_they_fall() {
    assert_eq!(fixed!(1_000.25), fixed!(1000.25));
    assert_eq!(fixed!(1_000_000.000_001), fixed!(1000000.000001));
    assert_eq!(unit!(0.000_1), unit!(0.0001));
}

#[test]
fn signs_are_read_and_negative_zero_is_just_zero() {
    assert_eq!(fixed!(-1.5), Fixed::ZERO - fixed!(1.5));
    assert_eq!(fixed!(-0.1), Fixed::ZERO - fixed!(0.1));
    assert_eq!(fixed!(+1.2), fixed!(1.2));

    // There is no negative zero on this grid, so a signed zero is the ordinary one.
    assert_eq!(fixed!(-0.0), Fixed::ZERO);
    assert_eq!(fixed!(-0), Fixed::ZERO);
    assert_eq!(fixed!(-0.0).to_bits(), 0);
}

#[test]
fn malformed_text_is_rejected_with_a_reason() {
    for text in ["1..2", "1e", "e10", "--1", "1.2.3", "", "0x10", "1f32", "abc", "1e1e1"] {
        assert!(
            decimal::parse(text).is_err(),
            "{text:?} should not parse as a decimal"
        );
    }

    assert_eq!(decimal::parse(""), Err(DecimalError::Empty));
    assert_eq!(decimal::parse("1..2"), Err(DecimalError::Invalid));
    assert_eq!(decimal::parse("1e"), Err(DecimalError::Invalid));
    assert_eq!(decimal::parse("e10"), Err(DecimalError::Invalid));
    assert_eq!(decimal::parse("--1"), Err(DecimalError::Invalid));
}

#[test]
fn the_pieces_of_a_decimal_are_what_they_should_be() {
    let value: Decimal = decimal::parse("12.345e-2").unwrap();

    assert!(!value.is_negative());
    assert_eq!(value.significand(), 12_345);
    assert_eq!(value.exponent(), -5);

    // Leading zeros carry no information and do not count against the digit budget.
    let small: Decimal = decimal::parse("0.000001").unwrap();

    assert_eq!(small.significand(), 1);
    assert_eq!(small.exponent(), -6);
}

// ---------------------------------------------------------------------------
// Compile-time use
// ---------------------------------------------------------------------------

const GRAVITY: Fixed = fixed!(9.80665);
const A: Fixed = fixed!(1.25);
const B: Unit = unit!(0.125);
const NARROW: FixedPoint<16> = fixed_point!(16, 1.25);
const CHANCE: Unit = unit!(0.025);

#[test]
fn the_macros_work_in_const_contexts() {
    // The whole conversion runs while the program is being compiled, which is what
    // makes these usable for the constants they are meant for.
    assert_eq!(A, Fixed::ONE + Fixed::ONE / 4);
    assert_eq!(B, Unit::from_bits(1 << 60).unwrap());
    assert_eq!(NARROW.to_bits(), (1 << 16) + (1 << 14));
    assert_eq!(CHANCE.to_bits(), expected_bits(25, 1000, 63) as u64);
    assert_eq!(GRAVITY.to_bits(), expected_bits(980_665, 100_000, 32) as i128);
}

// ---------------------------------------------------------------------------
// Runtime text
// ---------------------------------------------------------------------------

#[test]
fn parsing_a_string_agrees_with_the_literal() {
    assert_eq!("1.25".parse::<Fixed>().unwrap(), fixed!(1.25));
    assert_eq!("0.1".parse::<Fixed>().unwrap(), fixed!(0.1));
    assert_eq!("-17.625".parse::<Fixed>().unwrap(), fixed!(-17.625));
    assert_eq!("1.25e-4".parse::<Fixed>().unwrap(), fixed!(1.25e-4));
    assert_eq!("1_000.25".parse::<Fixed>().unwrap(), fixed!(1_000.25));

    assert_eq!("0.125".parse::<Unit>().unwrap(), unit!(0.125));
    assert_eq!("0.2".parse::<Unit>().unwrap(), unit!(0.2));
    assert_eq!("1e-2".parse::<Unit>().unwrap(), unit!(0.01));
}

#[test]
fn runtime_parsing_reports_errors_rather_than_panicking() {
    assert!("".parse::<Fixed>().is_err());
    assert!("1..2".parse::<Fixed>().is_err());
    assert!("not a number".parse::<Fixed>().is_err());
    assert!("1e999999999".parse::<Fixed>().is_err());

    assert_eq!("1.1".parse::<Unit>(), Err(DecimalError::NotAUnit));
    assert_eq!("-0.1".parse::<Unit>(), Err(DecimalError::NotAUnit));
    assert!("1.0000001".parse::<Unit>().is_err());
}

#[test]
fn very_long_decimals_are_read_exactly() {
    // Past 38 significant digits the `u128` path gives up and the exact one takes
    // over. The two must agree about everything they both accept, and the long path
    // must not quietly drop the digits that made it long.
    let long: &str = "0.33333333333333333333333333333333333333333333333333";
    let parsed: Fixed = long.parse::<Fixed>().unwrap();

    // Independently: that string is 33...3 / 10^50, which at 32 fractional bits is
    // indistinguishable from a third.
    let third: i128 = expected_bits(1, 3, 32) as i128;

    assert_eq!(parsed.to_bits(), third);

    // Ties, written out exactly. A step at 32 fractional bits is 2^-32, so these are
    // half a step, one and a half, and two and a half — all exactly on the boundary,
    // and all resolved by rounding to the even neighbour.
    let half_step: &str = "0.000000000116415321826934814453125";
    let three_halves: &str = "0.000000000349245965480804443359375";
    let five_halves: &str = "0.000000000582076609134674072265625";

    assert_eq!(half_step.parse::<Fixed>().unwrap().to_bits(), 0, "0.5 ties to 0");
    assert_eq!(three_halves.parse::<Fixed>().unwrap().to_bits(), 2, "1.5 ties to 2");
    assert_eq!(five_halves.parse::<Fixed>().unwrap().to_bits(), 2, "2.5 ties to 2");

    // The same half-step tie with a tail hung off the end, far past the 38 digits the
    // fast path holds. The tail is the only thing distinguishing this from the tie
    // above, so if the long path dropped it the answer would come back 0.
    let above_half: &str = "0.0000000001164153218269348144531250000000000000001";

    assert_eq!(
        above_half.parse::<Fixed>().unwrap().to_bits(),
        1,
        "a tail beyond 38 digits still decides the rounding"
    );

    // The literal form refuses what it cannot hold rather than misreading it.
    assert_eq!(
        decimal::parse("1.0000000000000000000000000000000000000001"),
        Err(DecimalError::TooManyDigits)
    );
}

#[test]
fn the_boundaries_of_the_range_are_respected() {
    // The largest representable value is just under 2^95.
    assert!(decimal::parse("39614081257132168796771975168")
        .unwrap()
        .to_fixed_bits(32)
        .is_err(), "2^95 does not fit");

    assert!(decimal::parse("39614081257132168796771975167")
        .unwrap()
        .to_fixed_bits(32)
        .is_ok(), "one below does");

    // The negative side reaches exactly one further, as two's complement does.
    assert!(decimal::parse("-39614081257132168796771975168")
        .unwrap()
        .to_fixed_bits(32)
        .is_ok());
}

// ---------------------------------------------------------------------------
// The exact rational type, which needs no grid at all
// ---------------------------------------------------------------------------

#[test]
fn a_ratio_keeps_the_decimal_exactly() {
    assert_eq!(Ratio::from_decimal_str("0.125").unwrap(), Ratio::new(1, 8).unwrap());
    assert_eq!(Ratio::from_decimal_str("0.1").unwrap(), Ratio::new(1, 10).unwrap());
    assert_eq!(Ratio::from_decimal_str("2.5e-1").unwrap(), Ratio::new(1, 4).unwrap());
    assert_eq!(Ratio::from_decimal_str("3").unwrap(), Ratio::new(3, 1).unwrap());

    // Unlike the others, nothing was rounded: a third of a tenth is still a tenth.
    assert_eq!(Ratio::from_decimal_str("-0.1"), None, "this type is unsigned");
}
