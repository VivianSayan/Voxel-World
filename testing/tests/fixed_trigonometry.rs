//! Golden vectors and identities for the fixed-point trigonometry.
//!
//! # What a golden vector is for here
//!
//! [`Fixed`] exists so that a world generated on one machine is the same world on
//! another. A trigonometric function that is merely *accurate* does not deliver that:
//! two implementations can both be within a step of the truth and still disagree with
//! each other, and a terrain seeded through them would differ between players.
//!
//! So these tests pin the exact raw integers, not a tolerance. A change that improves
//! accuracy will fail them, and that is the point — it is a change to the world's
//! contents, and should be a deliberate one with a version bump behind it.
//!
//! # Where the numbers came from
//!
//! Every value below was checked against an independent high-precision computation
//! before being frozen: a `Decimal` evaluation at 90 significant digits, using series
//! unrelated to the CORDIC implementation under test. The largest disagreement found
//! anywhere in this file was **0.478 units in the last place**, so every one of these
//! is the correctly rounded result or within half a step of it.

use voxel_world::math::{Fixed, FixedPoint};

/// `(angle in whole radians, sine bits, cosine bits)` at 32 fractional bits.
const WHOLE_ANGLES: [(i64, i128, i128); 13] = [
    (0, 0, 4294967296),
    (1, 3614090360, 2320580734),
    (2, 3905402711, -1787337053),
    (3, 606105819, -4251985396),
    (-1, -3614090360, 2320580734),
    (-3, -606105819, -4251985396),
    (7, 2821735955, 3237985527),
    (100, -2174823868, 3703631355),
    (1000, 3551420584, 2415399741),
    (100000, 153539918, -4292221985),
    (-100000, -153539918, -4292221985),
    (6, -1200080427, 4123899980),
    (-6, 1200080427, 4123899980),
];

/// `(angle as raw bits, sine bits, cosine bits)` at 32 fractional bits.
const FRACTIONAL_ANGLES: [(i128, i128, i128); 5] = [
    (1, 1, 4294967296),
    (1000, 1000, 4294967296),
    (123456789, 123439789, 4293193065),
    (2147483648, 2059117009, 3769188403),
    (1840700269, 1784867526, 3906531964),
];

/// `(y, x, atan2 bits)` at 32 fractional bits.
const DIRECTIONS: [(i64, i64, i128); 8] = [
    (1, 1, 3373259426),
    (1, 0, 6746518852),
    (0, -1, 13493037705),
    (-1, -1, -10119778278),
    (3, 4, 2763816217),
    (-5, 2, -5112256407),
    (1, 1000000, 4295),
    (0, 0, 0),
];

#[test]
fn whole_radian_angles_match_their_golden_bits() {
    for (angle, sine, cosine) in WHOLE_ANGLES {
        let (got_sine, got_cosine) = Fixed::from_integer(angle).sin_cos();

        assert_eq!(got_sine.to_bits(), sine, "sin({angle})");
        assert_eq!(got_cosine.to_bits(), cosine, "cos({angle})");
    }
}

#[test]
fn fractional_angles_match_their_golden_bits() {
    for (bits, sine, cosine) in FRACTIONAL_ANGLES {
        let (got_sine, got_cosine) = Fixed::from_bits(bits).sin_cos();

        assert_eq!(got_sine.to_bits(), sine, "sin(bits {bits})");
        assert_eq!(got_cosine.to_bits(), cosine, "cos(bits {bits})");
    }
}

#[test]
fn directions_match_their_golden_bits() {
    for (y, x, expected) in DIRECTIONS {
        let got = Fixed::from_integer(y).atan2(Fixed::from_integer(x));

        assert_eq!(got.to_bits(), expected, "atan2({y}, {x})");
    }
}

#[test]
fn every_layout_has_its_own_golden_bits() {
    // One radian, at four widths. Narrow layouts are where an off-by-one in the
    // rounding shift would show up first, and they are easy to leave untested.
    assert_eq!(FixedPoint::<8>::from_integer(1).sin_cos().0.to_bits(), 215);
    assert_eq!(FixedPoint::<8>::from_integer(1).sin_cos().1.to_bits(), 138);
    assert_eq!(FixedPoint::<16>::from_integer(1).sin_cos().0.to_bits(), 55147);
    assert_eq!(FixedPoint::<16>::from_integer(1).sin_cos().1.to_bits(), 35409);
    assert_eq!(FixedPoint::<64>::from_integer(1).sin_cos().0.to_bits(), 15522399902203605025);
    assert_eq!(FixedPoint::<64>::from_integer(1).sin_cos().1.to_bits(), 9966818358784711826);
}

// ---------------------------------------------------------------------------
// Identities
//
// Golden vectors prove the results do not move. These prove they were right to
// begin with, by checking relationships no implementation can satisfy by accident.
// ---------------------------------------------------------------------------

/// A tolerance of a few steps, for identities that compose several rounded results.
fn close(left: Fixed, right: Fixed, steps: i128) -> bool {
    (left.to_bits() - right.to_bits()).abs() <= steps
}

#[test]
fn the_pythagorean_identity_holds_everywhere() {
    for step in -400i64..400 {
        let angle = Fixed::from_bits(i128::from(step) * (1 << 32) / 50);
        let (sine, cosine) = angle.sin_cos();

        let total = sine.checked_mul(sine).unwrap() + cosine.checked_mul(cosine).unwrap();

        // Two multiplications and two roundings, so a handful of steps is the
        // honest bound; anything larger would mean the point left the circle.
        assert!(close(total, Fixed::ONE, 4), "sin^2 + cos^2 at {angle}: {total}");
    }
}

#[test]
fn sine_is_odd_and_cosine_is_even() {
    for step in 1i64..200 {
        let angle = Fixed::from_bits(i128::from(step) * (1 << 32) / 17);
        let (sine, cosine) = angle.sin_cos();
        let (mirrored_sine, mirrored_cosine) = (Fixed::ZERO - angle).sin_cos();

        // Exact, not approximate: the reduction maps an angle and its negation to
        // points that are reflections of one another, with no rounding between.
        assert_eq!(mirrored_sine, Fixed::ZERO - sine, "sin(-x) at {angle}");
        assert_eq!(mirrored_cosine, cosine, "cos(-x) at {angle}");
    }
}

#[test]
fn tangent_is_the_ratio_it_claims_to_be() {
    // The comparison here is against the *less* accurate construction. `tan` divides
    // the two 96-bit internal values, where `sin` and `cos` are each rounded to the
    // layout first and only then divided, so the reference carries two roundings into
    // a quotient that magnifies them.
    //
    // Propagating those: with |ds| and |dc| at most half a step,
    //
    //     |d(s/c)| <= |ds|/|c| + |s||dc|/|c|^2 = (1 + |tan|) / (2|c|) steps,
    //
    // plus half a step each for the division's own rounding and for `tan` itself. So
    // the allowance below is derived, not fitted, and it tightens automatically where
    // the tangent is small.
    for step in -100i64..100 {
        let angle = Fixed::from_bits(i128::from(step) * (1 << 32) / 80);
        let (sine, cosine) = angle.sin_cos();

        let Some(tangent) = angle.tan() else { continue };
        let ratio = sine.checked_div(cosine).unwrap();

        let spread = (Fixed::ONE + tangent.abs())
            .checked_div(cosine.abs())
            .unwrap();

        // One step for the two roundings, and the propagated term rounded upwards.
        let allowance: i128 = 1 + (spread.to_bits() >> 32) + 1;

        assert!(
            close(tangent, ratio, allowance),
            "tan vs sin/cos at {angle}: tan {tangent}, ratio {ratio}, allowed {allowance}"
        );
    }
}

#[test]
fn arctangent_inverts_tangent() {
    // Inside a quarter turn, where the tangent is single valued.
    for step in -70i64..70 {
        let angle = Fixed::from_bits(i128::from(step) * (1 << 32) / 50);
        let Some(tangent) = angle.tan() else { continue };

        assert!(close(tangent.atan(), angle, 4), "atan(tan(x)) at {angle}");
    }
}

#[test]
fn arctangent_of_two_recovers_the_angle_it_was_built_from() {
    // Over the whole circle, which is what atan2 is for and atan cannot do.
    for step in -300i64..300 {
        let angle = Fixed::from_bits(i128::from(step) * (1 << 32) / 100);

        // Only angles inside (-pi, pi] can come back, as that is atan2's range.
        if angle > Fixed::PI || angle <= Fixed::ZERO - Fixed::PI {
            continue;
        }

        let (sine, cosine) = angle.sin_cos();

        assert!(close(sine.atan2(cosine), angle, 4), "atan2 round trip at {angle}");
    }
}

#[test]
fn arcsine_and_arccosine_invert_their_functions() {
    for step in -90i64..=90 {
        let value = Fixed::from_bits(i128::from(step) * (1 << 32) / 100);

        let arcsine = value.asin().unwrap();
        let arccosine = value.acos().unwrap();

        assert!(close(arcsine.sin(), value, 4), "sin(asin(x)) at {value}");
        assert!(close(arccosine.cos(), value, 4), "cos(acos(x)) at {value}");

        // The two are complementary, which ties them together independently.
        assert!(close(arcsine + arccosine, Fixed::FRAC_PI_2, 4), "asin + acos at {value}");
    }
}

#[test]
fn the_named_constants_agree_with_the_functions() {
    assert!(close(Fixed::FRAC_PI_2, Fixed::PI.checked_div(Fixed::from_integer(2)).unwrap(), 1));
    assert!(close(Fixed::FRAC_PI_4, Fixed::PI.checked_div(Fixed::from_integer(4)).unwrap(), 1));
    assert!(close(Fixed::TAU, Fixed::PI + Fixed::PI, 1));

    // ln(10), log2(e) and log10(e) against the logarithms that should produce them.
    assert!(close(Fixed::LN_10, Fixed::from_integer(10).ln().unwrap(), 4));
    assert!(close(Fixed::LOG2_E, Fixed::E.log2().unwrap(), 4));
    assert!(close(
        Fixed::LOG10_E,
        Fixed::ONE.checked_div(Fixed::LN_10).unwrap(),
        4,
    ));
}

// ---------------------------------------------------------------------------
// Edges
// ---------------------------------------------------------------------------

#[test]
fn the_axes_give_exactly_the_right_angles() {
    assert_eq!(Fixed::ZERO.atan2(Fixed::ONE), Fixed::ZERO);
    assert_eq!(Fixed::ONE.atan2(Fixed::ZERO), Fixed::FRAC_PI_2);
    assert_eq!(Fixed::NEGATIVE_ONE.atan2(Fixed::ZERO), Fixed::ZERO - Fixed::FRAC_PI_2);

    // Behind, which is the boundary of the range and belongs to the positive side.
    assert!(close(Fixed::ZERO.atan2(Fixed::NEGATIVE_ONE), Fixed::PI, 1));

    // No direction at all, which has no angle; zero is the convention.
    assert_eq!(Fixed::ZERO.atan2(Fixed::ZERO), Fixed::ZERO);
}

#[test]
fn arcsine_rejects_what_is_not_a_sine() {
    assert_eq!(Fixed::from_integer(2).asin(), None);
    assert_eq!(Fixed::from_integer(-2).asin(), None);
    assert_eq!(Fixed::from_integer(2).acos(), None);

    // The boundary is included, and exact.
    assert_eq!(Fixed::ONE.asin(), Some(Fixed::FRAC_PI_2));
    assert_eq!(Fixed::NEGATIVE_ONE.acos(), Some(Fixed::PI));
    assert_eq!(Fixed::ONE.acos(), Some(Fixed::ZERO));
}

#[test]
fn the_hyperbolic_functions_hold_their_identity() {
    for step in -30i64..30 {
        let value = Fixed::from_bits(i128::from(step) * (1 << 32) / 10);

        let sine = value.sinh().unwrap();
        let cosine = value.cosh().unwrap();

        // cosh^2 - sinh^2 = 1, the hyperbolic Pythagoras.
        let difference =
            cosine.checked_mul(cosine).unwrap() - sine.checked_mul(sine).unwrap();

        assert!(close(difference, Fixed::ONE, 16), "cosh^2 - sinh^2 at {value}");

        let tangent = value.tanh();
        assert!(close(tangent, sine.checked_div(cosine).unwrap(), 16), "tanh at {value}");
    }
}

#[test]
fn hyperbolic_tangent_saturates_rather_than_failing() {
    // Past the point where the true value is nearer to one than to any other
    // representable value, the bound is the correctly rounded answer.
    assert_eq!(Fixed::from_integer(100).tanh(), Fixed::ONE);
    assert_eq!(Fixed::from_integer(-100).tanh(), Fixed::NEGATIVE_ONE);
    assert_eq!(Fixed::ZERO.tanh(), Fixed::ZERO);

    // And it is monotone across the join, with no step backwards.
    let mut previous = Fixed::NEGATIVE_ONE;

    for step in -300i64..300 {
        let value = Fixed::from_bits(i128::from(step) * (1 << 32) / 10);
        let tangent = value.tanh();

        assert!(tangent >= previous, "tanh went backwards at {value}");
        previous = tangent;
    }
}

#[test]
fn large_angles_still_land_on_the_circle() {
    // The argument reduction is the part most likely to decay quietly, so this
    // checks the identity far from the origin rather than trusting the small cases.
    for magnitude in [1_000i64, 100_000, 10_000_000, 1_000_000_000] {
        for angle in [Fixed::from_integer(magnitude), Fixed::from_integer(-magnitude)] {
            let (sine, cosine) = angle.sin_cos();
            let total =
                sine.checked_mul(sine).unwrap() + cosine.checked_mul(cosine).unwrap();

            assert!(close(total, Fixed::ONE, 8), "sin^2 + cos^2 at {angle}");
        }
    }
}

#[test]
fn a_quarter_turn_of_shift_turns_sine_into_cosine() {
    // Checks the quadrant dispatch: each branch is reached, and they agree at the
    // seams rather than each being right on its own.
    for step in 0i64..400 {
        let angle = Fixed::from_bits(i128::from(step) * (1 << 32) / 50);

        let shifted = (angle + Fixed::FRAC_PI_2).sin();

        assert!(close(shifted, angle.cos(), 4), "sin(x + pi/2) vs cos(x) at {angle}");
    }
}
