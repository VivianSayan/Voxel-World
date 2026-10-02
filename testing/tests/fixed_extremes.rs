//! The edges of the fixed-point range, where the arithmetic used to go wrong.
//!
//! Every test here corresponds to a bug: a shift that wrapped before its guard could
//! fire, a negation with no counterpart, a magnitude taken of a value that has none.
//! They are extreme inputs precisely because that is where those faults lived —
//! ordinary values never reached them, so nothing noticed.

use voxel_world::math::{Fixed, FixedPoint};

// ---------------------------------------------------------------------------
// Exponentials at the ends of the range
// ---------------------------------------------------------------------------

#[test]
fn exponentials_report_extremes_instead_of_wrapping() {
    // `exp` rescaled its argument by a left shift before deciding whether the
    // answer could fit. A raw value near the top wrapped during that shift and the
    // function returned a confident, wrong, finite number.
    assert_eq!(Fixed::MAX.exp(), None, "exp(MAX) cannot fit");
    assert_eq!(Fixed::MIN.exp(), Some(Fixed::ZERO), "exp(MIN) rounds to zero");

    // `exp2` had no guard at all.
    assert_eq!(Fixed::MAX.exp2(), None, "exp2(MAX) cannot fit");
    assert_eq!(Fixed::MIN.exp2(), Some(Fixed::ZERO), "exp2(MIN) rounds to zero");
}

#[test]
fn the_exponential_limits_sit_where_the_layout_puts_them() {
    // Q95.32 holds just under 2^95, so 2^95 is the first exponent that cannot fit
    // and 2^94 is the last that can.
    assert!(Fixed::from_integer(94).exp2().is_some(), "2^94 fits");
    assert_eq!(Fixed::from_integer(95).exp2(), None, "2^95 does not");

    // The smallest positive value is 2^-32, so that exponent is the last nonzero
    // one. 2^-33 is exactly half a step, which ties to even and so rounds to zero.
    assert_eq!(Fixed::from_integer(-32).exp2(), Some(Fixed::from_bits(1)));
    assert_eq!(Fixed::from_integer(-33).exp2(), Some(Fixed::ZERO), "an exact tie");
    assert_eq!(Fixed::from_integer(-40).exp2(), Some(Fixed::ZERO));

    // And `exp` at the same places, carried across by ln 2.
    assert!(Fixed::from_integer(65).exp().is_some(), "e^65 fits");
    assert_eq!(Fixed::from_integer(66).exp(), None, "e^66 does not");
    assert_eq!(Fixed::from_integer(-30).exp(), Some(Fixed::ZERO));
}

#[test]
fn every_layout_handles_its_own_extremes() {
    // The limits are derived from FRACTION_BITS, so a narrow layout and a wide one
    // must each refuse at their own boundary rather than at the default's.
    fn check<const N: u32>() {
        assert_eq!(FixedPoint::<N>::MAX.exp(), None);
        assert_eq!(FixedPoint::<N>::MAX.exp2(), None);
        assert_eq!(FixedPoint::<N>::MIN.exp(), Some(FixedPoint::<N>::ZERO));
        assert_eq!(FixedPoint::<N>::MIN.exp2(), Some(FixedPoint::<N>::ZERO));

        // One whole unit is small enough to work at any width.
        assert!(FixedPoint::<N>::ONE.exp().is_some());
        assert!(FixedPoint::<N>::ONE.exp2().is_some());
        assert!(FixedPoint::<N>::ONE.ln().is_some());
        assert!(FixedPoint::<N>::ONE.sqrt().is_some());
    }

    check::<1>();
    check::<16>();
    check::<32>();
    check::<64>();
}

// ---------------------------------------------------------------------------
// MIN, which has no positive counterpart
// ---------------------------------------------------------------------------

#[test]
fn powers_of_the_minimum_do_not_panic() {
    // `powi` asked for `self.abs()` while deciding how to treat a negative exponent.
    // `abs` panics on MIN, so an Option-returning function aborted the program.
    assert_eq!(Fixed::MIN.powi(0), Some(Fixed::ONE), "anything to the zero is one");

    // These two only have to answer rather than panic; whether they fit is the
    // layout's business, not the caller's.
    let _ = Fixed::MIN.powi(1);
    let _ = Fixed::MIN.powi(-1);
    let _ = Fixed::MIN.powi(2);
    let _ = Fixed::MIN.powi(i32::MIN);
    let _ = Fixed::MAX.powi(i32::MAX);
}

#[test]
fn the_special_cases_of_powi_are_the_mathematical_ones() {
    assert_eq!(Fixed::ZERO.powi(0), Some(Fixed::ONE), "0^0 is one by convention");
    assert_eq!(Fixed::ZERO.powi(3), Some(Fixed::ZERO));
    assert_eq!(Fixed::ZERO.powi(-1), None, "0^-1 is a division by zero");

    assert_eq!(Fixed::NEGATIVE_ONE.powi(0), Some(Fixed::ONE));
    assert_eq!(Fixed::NEGATIVE_ONE.powi(1_000_000), Some(Fixed::ONE), "even");
    assert_eq!(
        Fixed::NEGATIVE_ONE.powi(1_000_001),
        Some(Fixed::NEGATIVE_ONE),
        "odd"
    );
    assert_eq!(Fixed::NEGATIVE_ONE.powi(-1), Some(Fixed::NEGATIVE_ONE));

    assert_eq!(Fixed::from_integer(2).powi(10), Some(Fixed::from_integer(1024)));
    assert_eq!(Fixed::from_integer(2).powi(-1), Some(Fixed::ONE / 2));
}

#[test]
fn hyperbolic_functions_of_the_minimum_do_not_overflow() {
    // These negated the raw value directly to get `exp(-x)`, which overflows on MIN.
    assert_eq!(Fixed::MIN.sinh(), None, "sinh(MIN) is far past the range");
    assert_eq!(Fixed::MIN.cosh(), None, "cosh(MIN) is far past the range");

    // tanh is bounded, so it answers for every input.
    assert_eq!(Fixed::MIN.tanh(), Fixed::NEGATIVE_ONE);
    assert_eq!(Fixed::MAX.tanh(), Fixed::ONE);
}

#[test]
fn the_hyperbolic_functions_keep_their_symmetry() {
    for step in 1i64..60 {
        let value = Fixed::from_bits(i128::from(step) * (1 << 32) / 8);
        let mirrored = Fixed::ZERO - value;

        // sinh is odd and cosh is even, and the implementation now gets there by
        // working on the magnitude, so these are exact rather than approximate.
        assert_eq!(mirrored.sinh(), value.sinh().map(|v| Fixed::ZERO - v), "sinh(-x)");
        assert_eq!(mirrored.cosh(), value.cosh(), "cosh(-x)");
        assert_eq!(mirrored.tanh(), Fixed::ZERO - value.tanh(), "tanh(-x)");
    }
}

// ---------------------------------------------------------------------------
// Inverse trigonometry at the domain edges
// ---------------------------------------------------------------------------

#[test]
fn arcsine_and_arccosine_accept_the_whole_closed_domain() {
    // The implementation forms `1 - x^2`, and the worry is a representable input
    // inside [-1, 1] failing because that intermediate rounded past zero.
    assert_eq!(Fixed::ONE.asin(), Some(Fixed::FRAC_PI_2));
    assert_eq!(Fixed::NEGATIVE_ONE.asin(), Some(Fixed::ZERO - Fixed::FRAC_PI_2));
    assert_eq!(Fixed::ONE.acos(), Some(Fixed::ZERO));
    assert_eq!(Fixed::NEGATIVE_ONE.acos(), Some(Fixed::PI));

    // And one step inside each end, which is where a truncation would bite.
    let delta = Fixed::from_bits(1);

    for value in [
        Fixed::ONE - delta,
        Fixed::NEGATIVE_ONE + delta,
        Fixed::ZERO,
        delta,
        Fixed::ZERO - delta,
    ] {
        assert!(value.asin().is_some(), "asin({value}) is inside the domain");
        assert!(value.acos().is_some(), "acos({value}) is inside the domain");
    }

    // Just outside is still an error, which is the point of checking the domain
    // rather than waiting for the square root to complain.
    assert_eq!((Fixed::ONE + delta).asin(), None);
    assert_eq!((Fixed::NEGATIVE_ONE - delta).acos(), None);
    assert_eq!(Fixed::MAX.asin(), None);
    assert_eq!(Fixed::MIN.acos(), None);
}

#[test]
fn inverse_trigonometry_works_at_every_width() {
    fn check<const N: u32>() {
        let one = FixedPoint::<N>::ONE;

        assert_eq!(one.asin(), Some(FixedPoint::<N>::FRAC_PI_2));
        assert_eq!(one.acos(), Some(FixedPoint::<N>::ZERO));
        assert_eq!(
            (FixedPoint::<N>::ZERO - one).acos(),
            Some(FixedPoint::<N>::PI)
        );
        assert_eq!(FixedPoint::<N>::MAX.asin(), None);
    }

    check::<1>();
    check::<16>();
    check::<32>();
    check::<64>();
}

// ---------------------------------------------------------------------------
// Shifts and rounding
// ---------------------------------------------------------------------------

#[test]
fn enormous_angles_reduce_without_overflowing() {
    // The reduction runs in 256 bits, so it is defined for every raw value. Accuracy
    // is gone long before here — the documentation says so — but the call must still
    // return, deterministically, rather than wrap.
    for angle in [Fixed::MAX, Fixed::MIN, Fixed::from_bits(i128::MAX / 2)] {
        let (sine, cosine) = angle.sin_cos();

        assert!(sine.abs() <= Fixed::ONE, "a sine left the unit interval");
        assert!(cosine.abs() <= Fixed::ONE, "a cosine left the unit interval");
    }
}

#[test]
fn rounding_is_symmetric_about_zero() {
    // The shared helper rounds to nearest with ties to even, so a value and its
    // negation must round to counterpart results rather than one being biased.
    for step in 1i64..500 {
        let value = Fixed::from_bits(i128::from(step) * 7919);

        assert_eq!(
            (Fixed::ZERO - value).sin(),
            Fixed::ZERO - value.sin(),
            "sin is odd at {value}"
        );
    }
}

#[test]
fn the_assignment_operators_exist_at_every_width() {
    // These were written for the `Fixed` alias alone, so a `FixedPoint<16>` had no
    // `+=` despite `+` working perfectly well.
    let mut narrow: FixedPoint<16> = FixedPoint::<16>::ONE;

    narrow += FixedPoint::<16>::ONE;
    narrow -= FixedPoint::<16>::ONE;
    narrow *= FixedPoint::<16>::from_integer(4);
    narrow /= FixedPoint::<16>::from_integer(2);
    narrow %= FixedPoint::<16>::from_integer(3);
    narrow <<= 1;
    narrow >>= 1;

    assert_eq!(narrow, FixedPoint::<16>::from_integer(2));

    let mut wide: FixedPoint<64> = FixedPoint::<64>::ONE;
    wide += FixedPoint::<64>::ONE;

    assert_eq!(wide, FixedPoint::<64>::from_integer(2));
}
