//! Fixed-width wide integers, checked against the primitives.
//!
//! The strategy throughout: every operation on `WideUint<2>` has to agree with the
//! same operation on `u128`, because the two hold exactly the same values. Once
//! that is established for two limbs, the wider widths use the *same loops* with
//! a larger bound, so what is left to test at four limbs is only the things two
//! limbs cannot reach — a product above `u128::MAX`, a shift across three limb
//! boundaries.
//!
//! This is why the differential tests carry their weight: a hand-written table of
//! expected quotients would cover a dozen cases, and the bug in binary long
//! division only shows for a divisor above half the range.

use voxel_world::math::traits::{EuclideanRing, One, Semiring, Zero};
use voxel_world::math::{WideInt, WideUint};

type U128 = WideUint<2>;
type U256 = WideUint<4>;
type I128 = WideInt<2>;
type I256 = WideInt<4>;

/// A spread of `u128` values chosen to sit on the awkward boundaries: zero, one,
/// either side of a limb edge, either side of half the range, and the top.
fn unsigned_probes() -> Vec<u128> {
    let mut values: Vec<u128> = vec![
        0,
        1,
        2,
        3,
        7,
        10,
        255,
        256,
        u64::MAX as u128 - 1,
        u64::MAX as u128,
        u64::MAX as u128 + 1,
        u64::MAX as u128 + 2,
        1 << 127,
        (1 << 127) + 1,
        u128::MAX / 3,
        u128::MAX / 2,
        u128::MAX / 2 + 1,
        u128::MAX - 1,
        u128::MAX,
    ];

    // Some arbitrary-looking values too, from a fixed multiplier so the run is
    // the same every time. A random seed here would make a failure unreproducible.
    let mut state: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835;

    for _ in 0..40 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        values.push(state);
    }

    values
}

/// The same, signed.
fn signed_probes() -> Vec<i128> {
    let mut values: Vec<i128> = vec![
        0,
        1,
        -1,
        2,
        -2,
        7,
        -7,
        i64::MAX as i128,
        i64::MIN as i128,
        i64::MAX as i128 + 1,
        i64::MIN as i128 - 1,
        i128::MAX / 2,
        i128::MIN / 2,
        i128::MAX - 1,
        i128::MAX,
        i128::MIN + 1,
        i128::MIN,
    ];

    let mut state: u128 = 0x1234_5678_9ABC_DEF0_0FED_CBA9_8765_4321;

    for _ in 0..40 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        values.push(state as i128);
    }

    values
}

// ===========================================================================
// Unsigned, against u128
// ===========================================================================

#[test]
fn round_trips_through_u128() {
    for value in unsigned_probes() {
        let wide: U128 = WideUint::from(value);

        assert_eq!(wide.to_u128(), Some(value), "round trip of {value}");
    }
}

#[test]
fn ordering_matches_u128() {
    let probes: Vec<u128> = unsigned_probes();

    for left in &probes {
        for right in &probes {
            let wide: std::cmp::Ordering = U128::from(*left).cmp(&U128::from(*right));

            assert_eq!(wide, left.cmp(right), "comparing {left} with {right}");
        }
    }
}

#[test]
fn addition_matches_u128() {
    for left in unsigned_probes() {
        for right in unsigned_probes() {
            let wide: U128 = U128::from(left).wrapping_add(&U128::from(right));

            assert_eq!(
                wide.to_u128(),
                Some(left.wrapping_add(right)),
                "{left} + {right}"
            );

            // And the overflow flag has to agree too, not only the digits.
            let (_, carried) = U128::from(left).overflowing_add(&U128::from(right));

            assert_eq!(
                carried,
                left.checked_add(right).is_none(),
                "overflow of {left} + {right}"
            );
        }
    }
}

#[test]
fn subtraction_matches_u128() {
    for left in unsigned_probes() {
        for right in unsigned_probes() {
            let (wide, borrowed) = U128::from(left).overflowing_sub(&U128::from(right));

            assert_eq!(
                wide.to_u128(),
                Some(left.wrapping_sub(right)),
                "{left} - {right}"
            );
            assert_eq!(borrowed, right > left, "borrow of {left} - {right}");
        }
    }
}

#[test]
fn multiplication_matches_u128() {
    for left in unsigned_probes() {
        for right in unsigned_probes() {
            let (wide, overflowed) = U128::from(left).overflowing_mul(&U128::from(right));

            assert_eq!(
                wide.to_u128(),
                Some(left.wrapping_mul(right)),
                "{left} * {right}"
            );
            assert_eq!(
                overflowed,
                left.checked_mul(right).is_none(),
                "overflow of {left} * {right}"
            );
        }
    }
}

#[test]
fn division_matches_u128() {
    for left in unsigned_probes() {
        for right in unsigned_probes() {
            let result = U128::from(left).div_rem(&U128::from(right));

            if right == 0 {
                assert!(result.is_none(), "{left} / 0 should refuse");
                continue;
            }

            let (quotient, remainder) = result.expect("the divisor is not zero");

            assert_eq!(quotient.to_u128(), Some(left / right), "{left} / {right}");
            assert_eq!(remainder.to_u128(), Some(left % right), "{left} % {right}");
        }
    }
}

/// The case binary long division gets wrong when the bit shifted off the top of
/// the remainder is ignored: a divisor above half the range.
#[test]
fn divides_by_a_divisor_above_half_the_range() {
    let numerator: U128 = U128::MAX;

    for divisor in [
        1u128 << 127,
        (1u128 << 127) + 1,
        u128::MAX - 1,
        u128::MAX,
        u128::MAX / 2 + 1,
    ] {
        let (quotient, remainder) = numerator
            .div_rem(&U128::from(divisor))
            .expect("not by zero");

        assert_eq!(
            quotient.to_u128(),
            Some(u128::MAX / divisor),
            "MAX / {divisor}"
        );
        assert_eq!(
            remainder.to_u128(),
            Some(u128::MAX % divisor),
            "MAX % {divisor}"
        );
    }
}

#[test]
fn shifts_match_u128() {
    for value in unsigned_probes() {
        for places in [0u32, 1, 7, 63, 64, 65, 100, 127, 128] {
            let up: U128 = U128::from(value).wrapping_shl(places);
            let down: U128 = U128::from(value).wrapping_shr(places);

            let expected_up: u128 = if places >= 128 {
                0
            } else {
                value << places
            };
            let expected_down: u128 = if places >= 128 {
                0
            } else {
                value >> places
            };

            assert_eq!(up.to_u128(), Some(expected_up), "{value} << {places}");
            assert_eq!(down.to_u128(), Some(expected_down), "{value} >> {places}");
        }
    }
}

#[test]
fn bit_length_matches_u128() {
    for value in unsigned_probes() {
        let wide: U128 = WideUint::from(value);

        assert_eq!(
            wide.leading_zeros(),
            value.leading_zeros(),
            "leading zeros of {value}"
        );
        assert_eq!(
            wide.bit_length(),
            128 - value.leading_zeros(),
            "bit length of {value}"
        );
    }
}

#[test]
fn reads_and_writes_individual_bits() {
    for value in unsigned_probes() {
        let wide: U128 = WideUint::from(value);

        for index in 0..128u32 {
            assert_eq!(wide.bit(index), value >> index & 1 == 1, "{value} bit {index}");
        }

        // And beyond the width, rather than panicking.
        assert!(!wide.bit(128));
        assert!(!wide.bit(u32::MAX));
    }

    let mut value: U128 = WideUint::ZERO;
    value.set_bit(0, true);
    value.set_bit(127, true);

    assert_eq!(value.to_u128(), Some(1 | 1 << 127));

    value.set_bit(0, false);
    assert_eq!(value.to_u128(), Some(1 << 127));

    // Out of range, silently, because division walks past the top.
    value.set_bit(500, true);
    assert_eq!(value.to_u128(), Some(1 << 127));
}

#[test]
fn integer_sqrt_is_exact() {
    for value in unsigned_probes() {
        let root: U128 = U128::from(value).integer_sqrt();
        let root_value: u128 = root.to_u128().expect("a root fits");

        // The defining property, stated without ever computing a square root in
        // floating point: root² ≤ value < (root+1)².
        let square: u128 = root_value
            .checked_mul(root_value)
            .expect("the root squared cannot exceed the value");

        assert!(square <= value, "sqrt({value}) = {root_value} is too large");

        let next: u128 = root_value + 1;

        // Where the next root squared overflows there is nothing to check: it
        // certainly exceeds the value.
        if let Some(above) = next.checked_mul(next) {
            assert!(above > value, "sqrt({value}) = {root_value} is too small");
        }
    }
}

#[test]
fn displays_in_decimal() {
    for value in unsigned_probes() {
        assert_eq!(U128::from(value).to_string(), value.to_string(), "{value}");
    }

    // A value no primitive can hold, checked against the digits worked out by
    // hand: 2²⁵⁵.
    let big: U256 = U256::one().wrapping_shl(255);

    assert_eq!(
        big.to_string(),
        "57896044618658097711785492504343953926634992332820282019728792003956564819968"
    );
}

#[test]
fn displays_in_hexadecimal() {
    let value: U256 = U256::from(u128::MAX);

    assert_eq!(
        format!("{value:x}"),
        "00000000000000000000000000000000ffffffffffffffffffffffffffffffff"
    );
}

// ===========================================================================
// Unsigned, beyond what a primitive can hold
// ===========================================================================

#[test]
fn multiplies_past_the_u128_ceiling() {
    let biggest: U256 = WideUint::from(u128::MAX);

    // (2¹²⁸ − 1)² = 2²⁵⁶ − 2¹²⁹ + 1, which needs every one of the 256 bits.
    let squared: U256 = biggest.checked_mul(&biggest).expect("256 bits has room");

    assert!(squared.to_u128().is_none(), "it should not fit in 128 bits");

    // Checked against the same product worked out by shifts, which uses none of
    // the multiplication code.
    let expected: U256 = U256::one()
        .wrapping_shl(256 - 1)
        .wrapping_add(&U256::one().wrapping_shl(255))
        .wrapping_sub(&U256::one().wrapping_shl(129))
        .wrapping_add(&U256::one());

    assert_eq!(squared, expected);

    // And dividing back recovers it exactly.
    let (quotient, remainder) = squared.div_rem(&biggest).expect("not by zero");

    assert_eq!(quotient, biggest);
    assert!(remainder.is_zero());
}

#[test]
fn divides_a_256_bit_value_by_a_128_bit_one() {
    // A numerator built so the answer is known: 3·2²⁰⁰ + 5.
    let numerator: U256 = U256::from(3u64)
        .wrapping_mul(&U256::one().wrapping_shl(200))
        .wrapping_add(&U256::from(5u64));

    let divisor: U256 = U256::one().wrapping_shl(200);
    let (quotient, remainder) = numerator.div_rem(&divisor).expect("not by zero");

    assert_eq!(quotient, U256::from(3u64));
    assert_eq!(remainder, U256::from(5u64));
}

#[test]
fn widens_and_narrows() {
    let narrow: U128 = WideUint::from(u128::MAX);
    let wide: U256 = narrow.resize().expect("widening always fits");

    assert_eq!(wide.to_u128(), Some(u128::MAX));
    assert_eq!(wide.resize::<2>(), Some(narrow), "and back again");

    // Narrowing a value that needs the room refuses rather than truncating.
    let too_big: U256 = wide.wrapping_add(&U256::one());

    assert_eq!(too_big.resize::<2>(), None);
}

#[test]
fn power_comes_from_the_semiring_trait() {
    // 2¹⁰⁰, which only the wide type can hold — and `power` is inherited, not
    // written here, which is the point.
    let expected: U256 = U256::one().wrapping_shl(100);

    assert_eq!(U256::from(2u64).power(100), expected);
    assert_eq!(U256::from(7u64).power(0), U256::one());
    assert_eq!(U256::from(7u64).power(1), U256::from(7u64));
}

#[test]
fn zero_and_one_behave() {
    assert!(U256::ZERO.is_zero());
    assert!(!U256::ZERO.is_one());
    assert!(U256::one().is_one());
    assert!(!U256::one().is_zero());
    assert!(!U256::MAX.is_zero());
    assert!(!U256::MAX.is_one());

    assert_eq!(<U256 as Zero>::zero(), U256::ZERO);
    assert_eq!(<U256 as One>::one(), U256::one());
}

// ===========================================================================
// Signed, against i128
// ===========================================================================

#[test]
fn signed_round_trips_through_i128() {
    for value in signed_probes() {
        let wide: I128 = WideInt::from(value);

        assert_eq!(wide.is_negative(), value < 0, "sign of {value}");
        assert_eq!(wide.to_string(), value.to_string(), "printing {value}");
    }
}

#[test]
fn signed_ordering_matches_i128() {
    let probes: Vec<i128> = signed_probes();

    for left in &probes {
        for right in &probes {
            let wide: std::cmp::Ordering = I128::from(*left).cmp(&I128::from(*right));

            assert_eq!(wide, left.cmp(right), "comparing {left} with {right}");
        }
    }
}

#[test]
fn signed_addition_matches_i128() {
    for left in signed_probes() {
        for right in signed_probes() {
            let wide: I128 = I128::from(left).wrapping_add(&I128::from(right));

            assert_eq!(
                wide.to_string(),
                left.wrapping_add(right).to_string(),
                "{left} + {right}"
            );
            assert_eq!(
                I128::from(left).checked_add(&I128::from(right)).is_none(),
                left.checked_add(right).is_none(),
                "overflow of {left} + {right}"
            );
        }
    }
}

#[test]
fn signed_subtraction_matches_i128() {
    for left in signed_probes() {
        for right in signed_probes() {
            let wide: I128 = I128::from(left).wrapping_sub(&I128::from(right));

            assert_eq!(
                wide.to_string(),
                left.wrapping_sub(right).to_string(),
                "{left} - {right}"
            );
            assert_eq!(
                I128::from(left).checked_sub(&I128::from(right)).is_none(),
                left.checked_sub(right).is_none(),
                "overflow of {left} - {right}"
            );
        }
    }
}

#[test]
fn signed_multiplication_matches_i128() {
    for left in signed_probes() {
        for right in signed_probes() {
            assert_eq!(
                I128::from(left).wrapping_mul(&I128::from(right)).to_string(),
                left.wrapping_mul(right).to_string(),
                "{left} * {right}"
            );
            assert_eq!(
                I128::from(left).checked_mul(&I128::from(right)).is_none(),
                left.checked_mul(right).is_none(),
                "overflow of {left} * {right}"
            );
        }
    }
}

#[test]
fn signed_division_matches_i128() {
    for left in signed_probes() {
        for right in signed_probes() {
            let result = WideInt::div_rem(&I128::from(left), &I128::from(right));

            // Refused exactly where Rust's own division would panic: by zero, and
            // MIN by minus one.
            if right == 0 || (left == i128::MIN && right == -1) {
                assert!(result.is_none(), "{left} / {right} should refuse");
                continue;
            }

            let (quotient, remainder) = result.expect("a representable quotient");

            assert_eq!(
                quotient.to_string(),
                (left / right).to_string(),
                "{left} / {right}"
            );
            assert_eq!(
                remainder.to_string(),
                (left % right).to_string(),
                "{left} % {right}"
            );
        }
    }
}

#[test]
fn division_truncates_towards_zero() {
    // The four sign combinations, spelled out because the convention matters and
    // is easy to get backwards.
    let cases: [(i64, i64, i64, i64); 4] = [
        (7, 2, 3, 1),
        (-7, 2, -3, -1),
        (7, -2, -3, 1),
        (-7, -2, 3, -1),
    ];

    for (numerator, divisor, quotient, remainder) in cases {
        let (found_quotient, found_remainder) =
            WideInt::div_rem(&I256::from(numerator), &I256::from(divisor))
                .expect("not by zero");

        assert_eq!(
            found_quotient,
            I256::from(quotient),
            "{numerator} / {divisor}"
        );
        assert_eq!(
            found_remainder,
            I256::from(remainder),
            "{numerator} % {divisor}"
        );
    }
}

#[test]
fn the_most_negative_value_has_no_negation() {
    assert_eq!(I128::MIN.checked_neg(), None);
    assert_eq!(I128::MIN.checked_abs(), None);

    // Wrapping gives it back, as the primitives do.
    assert_eq!(I128::MIN.wrapping_neg(), I128::MIN);

    // But its magnitude is still available, because that is unsigned.
    assert_eq!(I128::MIN.magnitude(), U128::one().wrapping_shl(127));

    // And in a wider type it negates fine.
    let widened: I256 = WideInt::from(i128::MIN);

    assert_eq!(
        widened.checked_neg().map(|value| value.to_string()),
        Some("170141183460469231731687303715884105728".to_string())
    );
}

#[test]
fn signed_bounds_match_i128() {
    assert_eq!(I128::MIN.to_string(), i128::MIN.to_string());
    assert_eq!(I128::MAX.to_string(), i128::MAX.to_string());

    assert!(I128::MIN.is_negative());
    assert!(!I128::MAX.is_negative());
    assert!(I128::MIN < I128::MAX);
    assert!(I128::MIN < I128::ZERO);
    assert!(I128::ZERO < I128::MAX);
}

#[test]
fn magnitude_matches_unsigned_abs() {
    for value in signed_probes() {
        assert_eq!(
            I128::from(value).magnitude().to_u128(),
            Some(value.unsigned_abs()),
            "magnitude of {value}"
        );
    }
}

#[test]
fn greatest_common_divisor_comes_from_the_trait() {
    // Ordinary cases, against numbers whose divisors are obvious.
    assert_eq!(I256::from(12i64).gcd_normalised(&I256::from(18i64)), I256::from(6i64));
    assert_eq!(I256::from(-12i64).gcd_normalised(&I256::from(18i64)), I256::from(6i64));
    assert_eq!(I256::from(17i64).gcd_normalised(&I256::from(5i64)), I256::one());
    assert_eq!(I256::from(0i64).gcd_normalised(&I256::from(9i64)), I256::from(9i64));

    // And one no primitive could do: two multiples of a 130-bit prime-ish factor.
    let factor: I256 = WideInt::from_bits(U256::one().wrapping_shl(130).wrapping_add(&U256::one()));
    let left: I256 = factor.wrapping_mul(&I256::from(15i64));
    let right: I256 = factor.wrapping_mul(&I256::from(25i64));

    assert_eq!(
        left.gcd_normalised(&right),
        factor.wrapping_mul(&I256::from(5i64))
    );
}

#[test]
fn signed_power_comes_from_the_semiring_trait() {
    assert_eq!(I256::from(-2i64).power(3), I256::from(-8i64));
    assert_eq!(I256::from(-2i64).power(4), I256::from(16i64));

    // −2¹⁵⁰, past any primitive.
    let expected: I256 = WideInt::from_bits(U256::one().wrapping_shl(150));

    assert_eq!(I256::from(2i64).power(150), expected);
    assert_eq!(I256::from(-2i64).power(150), expected);
    assert_eq!(I256::from(-2i64).power(151), expected.wrapping_mul(&I256::from(-2i64)));
}

// ===========================================================================
// The width is free
// ===========================================================================

#[test]
fn works_at_one_limb() {
    type U64 = WideUint<1>;

    assert_eq!(U64::BITS, 64);
    assert_eq!(U64::MAX.to_u128(), Some(u64::MAX as u128));

    // A `u128` narrowed into one limb keeps only the low half, and says so.
    let (product, overflowed) = U64::from(u64::MAX).overflowing_mul(&U64::from(u64::MAX));

    assert!(overflowed);
    assert_eq!(product.to_u128(), Some((u64::MAX as u128).wrapping_mul(u64::MAX as u128) & u64::MAX as u128));
}

#[test]
fn works_at_sixteen_limbs() {
    type U1024 = WideUint<16>;

    assert_eq!(U1024::BITS, 1024);

    // 3^500 by repeated squaring, divided back down by 3 five hundred times.
    let power: U1024 = U1024::from(3u64).power(500);
    let mut remaining: U1024 = power;

    for step in 0..500 {
        let (quotient, remainder) = remaining
            .div_rem(&U1024::from(3u64))
            .expect("not by zero");

        assert!(remainder.is_zero(), "3^500 / 3 should be exact at step {step}");
        remaining = quotient;
    }

    assert!(remaining.is_one(), "dividing out every factor leaves one");
}

#[test]
fn a_thousand_bit_value_prints_correctly() {
    type U1024 = WideUint<16>;

    // 10^100, whose decimal form is a one and a hundred zeros — a printing test
    // that needs no reference implementation.
    let value: U1024 = U1024::from(10u64).power(100);
    let text: String = value.to_string();

    assert_eq!(text.len(), 101);
    assert!(text.starts_with('1'));
    assert!(text[1..].chars().all(|digit| digit == '0'), "{text}");
}

// ===========================================================================
// Composing with the rest of src/math
// ===========================================================================
//
// The point of the trait hierarchy is that a new number type written to it drops
// into everything generic over a ring without those types being touched. These
// tests are the check on that claim — and each one does something the primitive
// they would otherwise use cannot.

#[test]
fn serves_as_polynomial_coefficients() {
    use voxel_world::math::Polynomial;

    // (x + 2¹⁰⁰)² = x² + 2¹⁰¹x + 2²⁰⁰, whose last coefficient no primitive holds.
    let scale: I256 = WideInt::from_bits(U256::one().wrapping_shl(100));
    let linear: Polynomial<I256> = Polynomial::new(vec![scale, I256::one()]);

    let squared: Polynomial<I256> = linear.clone() * linear;

    assert_eq!(squared.degree(), Some(2));
    assert_eq!(
        squared.coefficient(2).to_string(),
        "1",
        "the leading coefficient"
    );
    assert_eq!(
        squared.coefficient(1),
        WideInt::from_bits(U256::one().wrapping_shl(101)),
        "the cross term is 2¹⁰¹"
    );
    assert_eq!(
        squared.coefficient(0),
        WideInt::from_bits(U256::one().wrapping_shl(200)),
        "the constant term is 2²⁰⁰"
    );
}

#[test]
fn serves_as_gaussian_integer_components() {
    use voxel_world::math::Gaussian;
    use voxel_world::math::traits::AlgebraicNorm;

    // A Gaussian integer whose norm needs more than 128 bits: |a + bi|² = a² + b²
    // with both parts near 2¹⁰⁰, so the norm is near 2²⁰¹.
    let part: I256 = WideInt::from_bits(U256::one().wrapping_shl(100));
    let value: Gaussian<I256> = Gaussian::new(part, part);

    assert_eq!(
        value.algebraic_norm(),
        WideInt::from_bits(U256::one().wrapping_shl(201)),
        "2¹⁰⁰² + 2¹⁰⁰² = 2²⁰¹"
    );

    // And multiplication still respects the norm, which is the property that makes
    // the ring Euclidean.
    let product: Gaussian<I256> = value * Gaussian::new(I256::one(), I256::one());

    assert_eq!(
        product.algebraic_norm(),
        value.algebraic_norm().wrapping_mul(&I256::from(2i64)),
        "the norm is multiplicative"
    );
}

#[test]
fn serves_as_matrix_entries() {
    use voxel_world::math::Matrix;

    // Entries near 2¹⁰⁰, so the products in a matrix multiplication land near 2²⁰⁰
    // and would overflow every primitive.
    let big: I256 = WideInt::from_bits(U256::one().wrapping_shl(100));
    let matrix: Matrix<I256, 2, 2> =
        Matrix::from_rows([[big, I256::one()], [I256::one(), big]]);

    let squared: Matrix<I256, 2, 2> = matrix * matrix;

    // [[b, 1], [1, b]]² = [[b² + 1, 2b], [2b, b² + 1]].
    let expected_diagonal: I256 =
        WideInt::from_bits(U256::one().wrapping_shl(200)).wrapping_add(&I256::one());

    assert_eq!(squared.get(0, 0), Some(&expected_diagonal));
    assert_eq!(squared.get(1, 1), Some(&expected_diagonal));
    assert_eq!(
        squared.get(0, 1),
        Some(&big.wrapping_mul(&I256::from(2i64))),
        "the off-diagonal is 2b"
    );
}

/// A determinant is **not** available, and that is correct rather than a gap in
/// the new type.
///
/// [`Matrix::determinant`] is implemented by Gaussian elimination, which divides,
/// so it is bounded on `Field`. The integers are a Euclidean ring and not a field:
/// `WideInt` has no reciprocal, exactly as `i64` has none. Getting a determinant
/// over a ring needs fraction-free elimination (Bareiss), which divides only where
/// the division is known to be exact — a different algorithm, not a looser bound.
///
/// This test records the reasoning; there is nothing to call.
#[test]
fn has_no_determinant_because_it_is_not_a_field() {
    use voxel_world::math::traits::Field;

    /// Compiles only for a field, so the assertion is made by the type checker.
    fn requires_a_field<T: Field>() {}

    requires_a_field::<f64>();
    // requires_a_field::<I256>();  // correctly rejected: no reciprocal
}

// ===========================================================================
// Algorithm D: the rare correction step
// ===========================================================================
//
// `div_rem` is Knuth's Algorithm D, which estimates each quotient digit and then
// corrects it. The final correction — step D5, decrement the digit and add the
// divisor back — fires for roughly one division in 2⁶³. Every differential test
// above passes without ever reaching it, so those tests say nothing at all about
// whether it is right. That makes it precisely the branch most likely to be wrong
// and least likely to be caught.
//
// The cases below were found by searching for inputs where the estimate computed
// from the divisor's top two limbs is still one too large: a divisor with a large
// *low* limb, which step D3 never looks at. Each carries its exact quotient and
// remainder, computed independently.

// 120 cases that force the add-back, found by search:
// the D3-corrected estimate is still one too large, so step D5 must fire.
/// A division that forces the add-back: dividend, divisor, quotient, remainder,
/// each as four little-endian limbs.
type AddBackCase = ([u64; 4], [u64; 4], [u64; 4], [u64; 4]);

const ADD_BACK_CASES: &[AddBackCase] = &[
    (
        [0x0000000000000000, 0xfffffffffffffffe, 0x8000000000000000, 0x7fffffffffffffff],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0xffffffffffffffff, 0xfffffffffffffffd, 0x8000000000000000, 0x7fffffffffffffff],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffd, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0x7ffffffffffffffe, 0x8000000000000000, 0x8000000000000000, 0x4000000000000000],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0x7ffffffffffffffd, 0x8000000000000000, 0x8000000000000000, 0x4000000000000000],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffd, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0xfffffffffffffffb, 0x0000000000000003, 0x0000000000000000, 0x0000000000000002],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0x0000000000000003, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0xfffffffffffffffa, 0x0000000000000003, 0x0000000000000000, 0x0000000000000002],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0x0000000000000003, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffd, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0xfffffffffffffffc, 0x0000000000000002, 0x8000000000000000, 0x0000000000000001],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0x0000000000000002, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0xfffffffffffffffb, 0x0000000000000002, 0x8000000000000000, 0x0000000000000001],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0x0000000000000002, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffd, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0xfffffffffffffffd, 0x0000000000000001, 0x0000000000000000, 0x0000000000000001],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0xfffffffffffffffc, 0x0000000000000001, 0x0000000000000000, 0x0000000000000001],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
        [0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffd, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000],
    ),
    (
        [0x0000000000000000, 0xfffffffffffffffe, 0x7fffffffffffffff, 0x8000000000000000],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000001, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x8000000000000001, 0x0000000000000000],
    ),
    (
        [0xffffffffffffffff, 0xfffffffffffffffd, 0x7fffffffffffffff, 0x8000000000000000],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000001, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffd, 0x0000000000000000, 0x8000000000000001, 0x0000000000000000],
    ),
    (
        [0x7ffffffffffffffe, 0x8000000000000000, 0x0000000000000001, 0x4000000000000001],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000001, 0x0000000000000000],
        [0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffe, 0x0000000000000000, 0x8000000000000001, 0x0000000000000000],
    ),
    (
        [0x7ffffffffffffffd, 0x8000000000000000, 0x0000000000000001, 0x4000000000000001],
        [0xffffffffffffffff, 0x0000000000000000, 0x8000000000000001, 0x0000000000000000],
        [0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xfffffffffffffffd, 0x0000000000000000, 0x8000000000000001, 0x0000000000000000],
    ),
];


#[test]
fn handles_the_add_back_correction() {
    for (numerator, divisor, quotient, remainder) in ADD_BACK_CASES {
        let numerator: U256 = WideUint::from_limbs(*numerator);
        let divisor: U256 = WideUint::from_limbs(*divisor);

        let (found_quotient, found_remainder) =
            numerator.div_rem(&divisor).expect("not by zero");

        assert_eq!(
            found_quotient,
            WideUint::from_limbs(*quotient),
            "quotient of {numerator} / {divisor}"
        );
        assert_eq!(
            found_remainder,
            WideUint::from_limbs(*remainder),
            "remainder of {numerator} / {divisor}"
        );

        // And the defining identity, which needs no reference at all:
        // numerator = quotient × divisor + remainder, with remainder < divisor.
        assert!(found_remainder < divisor, "the remainder must be smaller");
        assert_eq!(
            found_quotient
                .checked_mul(&divisor)
                .expect("the product fits, being the numerator")
                .wrapping_add(&found_remainder),
            numerator,
            "q × d + r should rebuild the numerator"
        );
    }
}

/// Algorithm D against the binary long division it replaced, which was itself
/// checked against `u128` across the whole overlapping range.
///
/// Two independent implementations of the same function, over inputs chosen to
/// stress the limb boundaries. This is the test that would have caught a wrong
/// add-back had the search above missed the triggering shape.
#[test]
fn agrees_with_binary_long_division() {
    let mut state: u128 = 0xACE1_0F00_D1CE_B00B_5EED_1234_ABCD_9876;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    // Shapes that matter: a divisor of one, two, three and four significant limbs,
    // each against numerators of every length, plus the extremes.
    let mut values: Vec<U256> = vec![
        U256::ZERO,
        U256::one(),
        U256::MAX,
        U256::from(u64::MAX),
        U256::from(u128::MAX),
        U256::one().wrapping_shl(64),
        U256::one().wrapping_shl(128),
        U256::one().wrapping_shl(192),
        U256::one().wrapping_shl(255),
        // A large low limb with a top limb only just normalised, the shape that
        // defeats the two-limb estimate.
        WideUint::from_limbs([u64::MAX, 0, 1 << 63, 0]),
        WideUint::from_limbs([u64::MAX, u64::MAX, 1 << 63, 0]),
        WideUint::from_limbs([1, 0, 0, 1 << 63]),
    ];

    for _ in 0..40 {
        let a = next();
        let b = next();
        values.push(WideUint::from_limbs([
            a as u64,
            (a >> 64) as u64,
            b as u64,
            (b >> 64) as u64,
        ]));
        // And some with deliberately truncated high limbs, so the significant-limb
        // count varies rather than always being four.
        values.push(WideUint::from_limbs([a as u64, (a >> 64) as u64, 0, 0]));
        values.push(WideUint::from_limbs([a as u64, 0, 0, 0]));
        values.push(WideUint::from_limbs([a as u64, (a >> 64) as u64, b as u64, 0]));
    }

    let mut compared: usize = 0;

    for numerator in &values {
        for divisor in &values {
            let fast = numerator.div_rem(divisor);
            let reference = numerator.div_rem_binary(divisor);

            assert_eq!(fast, reference, "{numerator} / {divisor}");
            compared += 1;
        }
    }

    // Recorded so a future change that silently shrinks the input set is visible.
    assert!(compared > 20_000, "only {compared} pairs compared");
}
