//! The quadratic integer rings, and the trait foundation under them.

use voxel_world::math::traits::{
    AlgebraicNorm, CommutativeRing, Conjugate, EuclideanRing, Field, One, Ring, RoundedDiv,
    Semiring, Zero,
};
use voxel_world::math::{Eisenstein, Fixed, Gaussian, Ratio};

type G = Gaussian<i64>;
type E = Eisenstein<i64>;

#[test]
fn the_ring_laws_hold_for_both_lattices() {
    let values: Vec<G> = (-3..=3)
        .flat_map(|re| (-3..=3).map(move |im| Gaussian::new(re, im)))
        .collect();

    for a in &values {
        // Identities.
        assert_eq!(*a + G::zero(), *a);
        assert_eq!(*a * G::one(), *a);
        assert_eq!(*a + -*a, G::zero(), "every element has an additive inverse");

        for b in &values {
            // Commutative, and the norm multiplies.
            assert_eq!(*a + *b, *b + *a);
            assert_eq!(*a * *b, *b * *a);
            assert_eq!(
                (*a * *b).algebraic_norm(),
                a.algebraic_norm() * b.algebraic_norm()
            );

            for c in values.iter().take(5) {
                assert_eq!((*a + *b) + *c, *a + (*b + *c), "associative");
                assert_eq!(*a * (*b + *c), *a * *b + *a * *c, "distributive");
            }
        }
    }

    // The same for the triangular lattice, including its shear term.
    let values: Vec<E> = (-3..=3)
        .flat_map(|a| (-3..=3).map(move |b| Eisenstein::new(a, b)))
        .collect();

    for a in &values {
        assert_eq!(*a + E::zero(), *a);
        assert_eq!(*a * E::one(), *a);
        assert_eq!(*a + -*a, E::zero());

        for b in &values {
            assert_eq!(*a * *b, *b * *a);
            assert_eq!(
                (*a * *b).algebraic_norm(),
                a.algebraic_norm() * b.algebraic_norm(),
                "{a} * {b}"
            );
        }
    }
}

#[test]
fn the_norm_is_positive_and_only_zero_at_the_origin() {
    for re in -5..=5i64 {
        for im in -5..=5i64 {
            let gaussian: G = Gaussian::new(re, im);
            let eisenstein: E = Eisenstein::new(re, im);

            assert_eq!(gaussian.algebraic_norm() == 0, gaussian.is_zero());
            assert_eq!(eisenstein.algebraic_norm() == 0, eisenstein.is_zero());
            assert!(gaussian.algebraic_norm() >= 0);
            assert!(
                eisenstein.algebraic_norm() >= 0,
                "a² - ab + b² at {re},{im}"
            );
        }
    }

    assert_eq!(Gaussian::new(3, 4).algebraic_norm(), 25, "3² + 4²");
    assert_eq!(Eisenstein::new(3, 4).algebraic_norm(), 13, "9 - 12 + 16");
}

#[test]
fn the_units_are_exactly_the_elements_of_norm_one() {
    // Four rotations of a square, six of a hexagon.
    let gaussian = G::units();
    assert_eq!(gaussian.len(), 4);
    assert!(
        gaussian
            .iter()
            .all(|unit| unit.algebraic_norm() == 1 && unit.is_unit())
    );

    let eisenstein = E::units();
    assert_eq!(eisenstein.len(), 6);
    assert!(
        eisenstein
            .iter()
            .all(|unit| unit.algebraic_norm() == 1 && unit.is_unit())
    );

    // And nothing else in range is a unit.
    for re in -4..=4i64 {
        for im in -4..=4i64 {
            let value: G = Gaussian::new(re, im);
            assert_eq!(value.is_unit(), gaussian.contains(&value), "{value}");

            let value: E = Eisenstein::new(re, im);
            assert_eq!(value.is_unit(), eisenstein.contains(&value), "{value}");
        }
    }
}

#[test]
fn turning_by_a_unit_cycles_back_round() {
    // i is a quarter turn, so four of them return.
    let mut value: G = Gaussian::new(2, 1);
    let start = value;
    for _ in 0..4 {
        value = value.turned();
    }
    assert_eq!(value, start);
    assert_eq!(Gaussian::new(2, 1).turned(), Gaussian::new(-1, 2));

    // ω is a third of a turn, so ω³ is one.
    assert!(E::omega().power(3).is_one());
    assert_eq!(E::omega().power(2), Eisenstein::new(-1, -1), "ω² = -1 - ω");

    // And the sixth root returns after six.
    let mut value: E = Eisenstein::new(2, 1);
    let start = value;
    for _ in 0..6 {
        value = value.turned();
    }
    assert_eq!(value, start);
}

#[test]
fn division_leaves_a_remainder_smaller_than_the_divisor() {
    // The property Euclid's algorithm needs, over the whole neighbourhood.
    for ar in -6..=6i64 {
        for ai in -6..=6i64 {
            for br in -4..=4i64 {
                for bi in -4..=4i64 {
                    let a: G = Gaussian::new(ar, ai);
                    let b: G = Gaussian::new(br, bi);

                    match a.div_rem(&b) {
                        Some((quotient, remainder)) => {
                            assert_eq!(a, quotient * b + remainder, "{a} / {b}");
                            assert!(
                                remainder.algebraic_norm() < b.algebraic_norm(),
                                "{a} / {b} left {remainder}"
                            );
                        }
                        None => assert!(b.is_zero(), "only zero has no quotient"),
                    }
                }
            }
        }
    }
}

#[test]
fn division_leaves_a_smaller_remainder_on_the_triangular_lattice_too() {
    // The sheared cell is why `div_rem` tries the neighbouring candidates: plain
    // coordinate rounding is not always the nearest point here.
    for aa in -6..=6i64 {
        for ab in -6..=6i64 {
            for ba in -4..=4i64 {
                for bb in -4..=4i64 {
                    let a: E = Eisenstein::new(aa, ab);
                    let b: E = Eisenstein::new(ba, bb);

                    match a.div_rem(&b) {
                        Some((quotient, remainder)) => {
                            assert_eq!(a, quotient * b + remainder, "{a} / {b}");
                            assert!(
                                remainder.algebraic_norm() < b.algebraic_norm(),
                                "{a} / {b} left {remainder}"
                            );
                        }
                        None => assert!(b.is_zero()),
                    }
                }
            }
        }
    }
}

#[test]
fn the_greatest_common_divisor_divides_both_and_is_largest() {
    // `gcd` is provided by the trait, so this exercises one implementation
    // serving integers and both lattices.
    assert_eq!(12i64.gcd_normalised(&18), 6);
    assert_eq!(17i64.gcd_normalised(&5), 1);
    assert_eq!(0i64.gcd_normalised(&7), 7);

    for ar in -5..=5i64 {
        for ai in -5..=5i64 {
            let a: G = Gaussian::new(ar, ai);
            let b: G = Gaussian::new(3, 1);
            let divisor = a.gcd(&b);

            if divisor.is_zero() {
                assert!(a.is_zero() && b.is_zero());
                continue;
            }

            assert!(divisor.divides(&a), "{divisor} should divide {a}");
            assert!(divisor.divides(&b), "{divisor} should divide {b}");
        }
    }

    // 2 factors in Z[i] as -i(1 + i)², so 1 + i divides it: gcd(2, 1 + i) is
    // 1 + i up to a unit, of norm 2.
    let two: G = Gaussian::new(2, 0);
    let one_plus_i: G = Gaussian::new(1, 1);
    assert_eq!(two.gcd(&one_plus_i).algebraic_norm(), 2);
    assert!(one_plus_i.divides(&two), "1 + i divides 2");

    // Whereas 2 and 2i are both associates of (1 + i)², so 2 divides both and
    // their gcd has norm 4 rather than 2.
    let two_i: G = Gaussian::new(0, 2);
    assert_eq!(two.gcd(&two_i).algebraic_norm(), 4);
    assert!(two.divides(&two_i), "2 divides 2i, with quotient i");

    // 3 stays prime in Z[i], so it shares nothing with 1 + i.
    let three: G = Gaussian::new(3, 0);
    assert!(three.gcd(&one_plus_i).is_unit(), "coprime");
}

#[test]
fn conjugation_is_an_involution_that_gives_the_norm() {
    for re in -4..=4i64 {
        for im in -4..=4i64 {
            let value: G = Gaussian::new(re, im);
            assert_eq!(value.conjugate().conjugate(), value);
            assert_eq!(
                value * value.conjugate(),
                Gaussian::new(value.algebraic_norm(), 0)
            );

            let value: E = Eisenstein::new(re, im);
            assert_eq!(value.conjugate().conjugate(), value, "{value}");
            assert_eq!(
                value * value.conjugate(),
                Eisenstein::new(value.algebraic_norm(), 0),
                "{value}"
            );
        }
    }
}

#[test]
fn rounded_division_picks_the_nearest_not_the_truncated() {
    assert_eq!(7i64.div_rounded(&2), Some(4), "3.5 rounds away from zero");
    assert_eq!(5i64.div_rounded(&2), Some(3), "2.5 rounds away from zero");
    assert_eq!(4i64.div_rounded(&3), Some(1), "1.33 rounds down");
    assert_eq!(5i64.div_rounded(&3), Some(2), "1.67 rounds up");
    assert_eq!((-7i64).div_rounded(&2), Some(-4));
    assert_eq!((-5i64).div_rounded(&3), Some(-2));
    assert_eq!(7i64.div_rounded(&-2), Some(-4));
    assert_eq!(1i64.div_rounded(&0), None);

    // The comparison avoids overflow at the limits rather than doubling.
    assert_eq!(i64::MAX.div_rounded(&i64::MAX), Some(1));
    assert_eq!(i64::MIN.div_rounded(&-1), None, "no representable answer");
}

#[test]
fn the_traits_reach_the_number_types_the_crate_already_had() {
    // Powers and multiples come from the trait, by squaring and doubling.
    assert_eq!(3i64.power(5), 243);
    assert_eq!(2i64.power(0), 1);
    assert_eq!(7i64.multiple(6), 42);
    assert_eq!(Gaussian::new(0i64, 1).power(4), G::one(), "i⁴ = 1");

    // Fixed point is a field: it divides.
    assert_eq!(
        Fixed::from(4).inverse(),
        Some(Fixed::from(1) / Fixed::from(4))
    );
    assert_eq!(Fixed::ZERO.inverse(), None);
    assert!(Fixed::ONE.is_one() && Fixed::ZERO.is_zero());

    // Ratio is a semiring and deliberately no more: it cannot be negative, so it
    // has no additive inverse and therefore is not a ring.
    assert!(Ratio::ONE.is_one());
    assert_eq!(Ratio::from(3u64).power(2), Ratio::from(9u64));
    fn accumulate<T: Semiring>(values: &[T]) -> T {
        values
            .iter()
            .fold(T::zero(), |total, value| total + value.clone())
    }
    assert_eq!(accumulate(&[Ratio::ONE, Ratio::ONE]), Ratio::from(2u64));
    assert_eq!(accumulate(&[2u32, 3, 4]), 9, "unsigned integers too");
}

#[test]
fn a_ring_algorithm_can_be_written_once_for_every_ring() {
    // The point of the traits: this is written once and serves integers, both
    // lattices, and fixed point.
    fn evaluate<T: Ring>(coefficients: &[T], at: &T) -> T {
        coefficients
            .iter()
            .rev()
            .fold(T::zero(), |total, coefficient| {
                total * at.clone() + coefficient.clone()
            })
    }

    // 1 + 2x + 3x² at x = 2 is 1 + 4 + 12 = 17.
    assert_eq!(evaluate(&[1i64, 2, 3], &2), 17);
    assert_eq!(
        evaluate(&[Fixed::ONE, Fixed::ONE], &Fixed::from(3)),
        Fixed::from(4)
    );

    // And over Z[i]: 1 + x at x = i is 1 + i.
    assert_eq!(
        evaluate(&[G::one(), G::one()], &Gaussian::new(0, 1)),
        Gaussian::new(1, 1)
    );

    fn is_commutative<T: CommutativeRing>(a: &T, b: &T) -> bool {
        a.clone() * b.clone() == b.clone() * a.clone()
    }
    assert!(is_commutative(
        &Gaussian::new(2i64, 3),
        &Gaussian::new(1, -1)
    ));
    assert!(is_commutative(
        &Eisenstein::new(2i64, 3),
        &Eisenstein::new(1, -1)
    ));
}

#[test]
fn both_lattices_print_readably() {
    assert_eq!(Gaussian::new(3i64, 4).to_string(), "3 + 4i");
    assert_eq!(Gaussian::new(3i64, -4).to_string(), "3 - 4i");
    assert_eq!(Gaussian::new(3i64, 0).to_string(), "3");
    assert_eq!(Gaussian::new(0i64, 1).to_string(), "0 + 1i");

    assert_eq!(Eisenstein::new(3i64, 4).to_string(), "3 + 4\u{3c9}");
    assert_eq!(Eisenstein::new(3i64, -4).to_string(), "3 - 4\u{3c9}");
    assert_eq!(Eisenstein::new(-2i64, 0).to_string(), "-2");
}
