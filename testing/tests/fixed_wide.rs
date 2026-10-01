//! The hand-rolled 256-bit arithmetic inside [`Fixed`], checked against
//! [`WideUint<4>`].
//!
//! `fixed::wide` is specialised and fast: it divides a 256-bit value by walking
//! 128 bits with native `u128` compares, where the general type walks 256 bits
//! with a four-limb compare at each. Measured, that is 124× on a multiply-then-
//! divide, which is why `wide` exists at all.
//!
//! The price of a second implementation is a second place for a bug to live, and
//! `wide`'s own tests can only check it against what its author expected. These
//! tests remove that: both implementations run over the same inputs and have to
//! produce the same answer. The general one is slow, which does not matter in a
//! test, and was itself checked against `u128` over its whole overlapping range —
//! so it is a genuinely independent reference rather than a restatement.

use voxel_world::math::WideUint;
use voxel_world::math::fixed::wide;

type U256 = WideUint<4>;

/// `wide`'s two-halves value as the general type.
fn widen(value: wide::U256) -> U256 {
    let mut limbs = [0u64; 4];

    limbs[0] = value.low as u64;
    limbs[1] = (value.low >> 64) as u64;
    limbs[2] = value.high as u64;
    limbs[3] = (value.high >> 64) as u64;

    WideUint::from_limbs(limbs)
}

/// Inputs chosen to sit on the boundaries where a hand-carried algorithm goes
/// wrong: limb edges, half the range, and the extremes — plus a fixed pseudorandom
/// spread, fixed so that a failure can be reproduced.
fn probes() -> Vec<u128> {
    let mut values: Vec<u128> = vec![
        0,
        1,
        2,
        3,
        u32::MAX as u128,
        1 << 63,
        (1 << 63) - 1,
        (1 << 63) + 1,
        u64::MAX as u128,
        u64::MAX as u128 + 1,
        1 << 96,
        1 << 127,
        (1 << 127) + 1,
        u128::MAX / 2,
        u128::MAX / 2 + 1,
        u128::MAX - 1,
        u128::MAX,
    ];

    let mut state: u128 = 0xDEAD_BEEF_CAFE_F00D_0123_4567_89AB_CDEF;

    for _ in 0..60 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        values.push(state);
    }

    values
}

#[test]
fn multiply_agrees() {
    for left in probes() {
        for right in probes() {
            let fast: U256 = widen(wide::multiply(left, right));
            // Exact, because 128 × 128 bits always fits in 256.
            let reference: U256 = U256::from(left)
                .checked_mul(&U256::from(right))
                .expect("a 256-bit product of two 128-bit values always fits");

            assert_eq!(fast, reference, "{left} * {right}");
        }
    }
}

#[test]
fn shift_left_agrees() {
    for value in probes() {
        for places in 0..128u32 {
            let fast: U256 = widen(wide::shift_left(value, places));
            let reference: U256 = U256::from(value).wrapping_shl(places);

            assert_eq!(fast, reference, "{value} << {places}");
        }
    }
}

#[test]
fn shift_right_agrees() {
    for high in probes() {
        for low in probes() {
            let value = wide::U256 { high, low };
            let wide_value: U256 = widen(value);

            for places in [0u32, 1, 7, 63, 64, 65, 100, 127] {
                let shifted: U256 = wide_value.wrapping_shr(places);

                // The wrapping form keeps the low 128 bits of the shifted value.
                assert_eq!(
                    wide::shift_right_wrapping(value, places),
                    shifted.as_u128_wrapping(),
                    "wrapping {high}:{low} >> {places}"
                );

                // And the checked form agrees about whether it still fits.
                assert_eq!(
                    wide::shift_right(value, places),
                    shifted.to_u128(),
                    "checked {high}:{low} >> {places}"
                );
            }
        }
    }
}

/// # Why the oracle here is `div_rem_binary` and not `div_rem`
///
/// `wide::divide` keeps a native fast path for a numerator that fits in 128 bits
/// and hands anything wider to `WideUint::div_rem`. Checking it against `div_rem`
/// would therefore be checking that function against itself for every input that
/// takes the wide path — the test would pass whatever either one did.
///
/// `div_rem_binary` is the bit-at-a-time implementation, which shares no code with
/// Algorithm D. That keeps this test meaningful for both of `wide::divide`'s paths.
#[test]
fn divide_agrees() {
    for high in probes() {
        for low in probes() {
            let numerator = wide::U256 { high, low };
            let wide_numerator: U256 = widen(numerator);

            for divisor in probes() {
                let fast: Option<u128> = wide::divide(numerator, divisor);

                let reference: Option<u128> = wide_numerator
                    .div_rem_binary(&U256::from(divisor))
                    // `wide::divide` refuses a quotient too wide for 128 bits,
                    // where the general one simply returns it.
                    .and_then(|(quotient, _)| quotient.to_u128());

                assert_eq!(fast, reference, "{high}:{low} / {divisor}");
            }
        }
    }
}

#[test]
fn square_root_agrees() {
    for high in probes() {
        for low in probes() {
            let value = wide::U256 { high, low };

            let fast: u128 = wide::square_root(value);
            let reference: U256 = widen(value).integer_sqrt();

            assert_eq!(
                Some(fast),
                reference.to_u128(),
                "sqrt of {high}:{low}"
            );
        }
    }
}

/// `apply_sign` has no counterpart to compare against, so it is checked against
/// the property it exists to enforce.
#[test]
fn apply_sign_accepts_exactly_the_representable_magnitudes() {
    for magnitude in probes() {
        let positive: Option<i128> = wide::apply_sign(magnitude, false);
        let negative: Option<i128> = wide::apply_sign(magnitude, true);

        assert_eq!(
            positive,
            i128::try_from(magnitude).ok(),
            "positive {magnitude}"
        );

        match negative {
            // Accepted: negating it has to give back the magnitude.
            Some(value) => {
                assert!(value <= 0, "{magnitude} negated should not be positive");
                assert_eq!(value.unsigned_abs(), magnitude, "negated {magnitude}");
            }
            // Refused: it has to be genuinely out of range, which is one past
            // `i128::MAX` because the negative range holds one more value.
            None => assert!(
                magnitude > i128::MAX as u128 + 1,
                "{magnitude} should have been representable as a negative"
            ),
        }
    }
}
