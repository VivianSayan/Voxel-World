//! The split algebras: j² = +1 in two dimensions and in four.

use voxel_world::math::traits::{AlgebraicNorm, CommutativeRing, Conjugate, One, Ring, Zero};
use voxel_world::math::{Complex, Dual, SplitComplex, SplitQuaternion};

type S = SplitComplex<i64>;
type Q = SplitQuaternion<i64>;

#[test]
fn the_unit_squares_to_one_rather_than_minus_one() {
    let j: S = SplitComplex::unit();

    assert_eq!(j * j, S::one(), "j² = +1");
    assert_eq!(j.algebraic_norm(), -1, "and its norm is negative");

    // Which is the one difference from the complex numbers, where i² = -1.
    let i: Complex<f64> = Complex::new(0.0, 1.0);
    assert_eq!(i * i, Complex::new(-1.0, 0.0));

    // And from the dual numbers, where the square vanishes.
    let epsilon: Dual<f64> = Dual::new(0.0, 1.0);
    assert_eq!(epsilon * epsilon, Dual::new(0.0, 0.0));
}

#[test]
fn the_ring_laws_hold_and_multiplication_commutes() {
    let values: Vec<S> = (-3..=3)
        .flat_map(|re| (-3..=3).map(move |sp| SplitComplex::new(re, sp)))
        .collect();

    for a in &values {
        assert_eq!(*a + S::zero(), *a);
        assert_eq!(*a * S::one(), *a);
        assert_eq!(*a + -*a, S::zero());

        for b in &values {
            assert_eq!(*a + *b, *b + *a);
            assert_eq!(*a * *b, *b * *a, "two dimensions always commute");

            // The norm multiplies, even though it is indefinite.
            assert_eq!(
                (*a * *b).algebraic_norm(),
                a.algebraic_norm() * b.algebraic_norm(),
                "{a} * {b}"
            );

            for c in values.iter().take(5) {
                assert_eq!(*a * (*b + *c), *a * *b + *a * *c);
            }
        }
    }

    fn commutes<T: CommutativeRing>(a: &T, b: &T) -> bool {
        a.clone() * b.clone() == b.clone() * a.clone()
    }
    assert!(commutes(
        &SplitComplex::new(2i64, 3),
        &SplitComplex::new(1, -1)
    ));
}

#[test]
fn the_null_lines_are_non_zero_values_that_multiply_to_nothing() {
    let up: S = SplitComplex::new(1, 1);
    let down: S = SplitComplex::new(1, -1);

    assert!(!up.is_zero() && !down.is_zero());
    assert_eq!(up.algebraic_norm(), 0);
    assert_eq!(down.algebraic_norm(), 0);
    assert!(up.is_null() && down.is_null());

    // Two non-zero values whose product is zero: a zero divisor, which is what
    // keeps this from being a field.
    assert_eq!(up * down, S::zero());

    // Every value on a null line behaves so, and only those.
    for re in -5..=5i64 {
        for sp in -5..=5i64 {
            let value: S = SplitComplex::new(re, sp);
            assert_eq!(value.is_null(), re.abs() == sp.abs(), "{value}");
        }
    }

    assert_eq!(S::null_basis(), [up, down]);
}

#[test]
fn division_works_everywhere_off_the_null_lines() {
    for re in -4..=4i64 {
        for sp in -4..=4i64 {
            let value: SplitComplex<f64> = SplitComplex::new(re as f64, sp as f64);

            match value.inverse() {
                Some(inverse) => {
                    let product = value * inverse;
                    assert!((product.re - 1.0).abs() < 1e-9, "{value}");
                    assert!(product.sp.abs() < 1e-9, "{value}");
                }
                None => assert_eq!(value.algebraic_norm(), 0.0, "only the null lines fail"),
            }
        }
    }

    // Including the origin, which is null like the rest of the cone.
    assert_eq!(SplitComplex::<f64>::zero().inverse(), None);
}

#[test]
fn a_unit_norm_value_is_a_hyperbolic_rotation() {
    // Multiplying by something of norm one preserves the norm, which is the
    // hyperbolic analogue of a turn preserving length.
    let boost: S = SplitComplex::new(3, 2);
    assert_eq!(
        boost.algebraic_norm(),
        5,
        "not one, so it scales as well as boosts"
    );

    let unit: S = SplitComplex::new(1, 0);
    assert_eq!(unit.algebraic_norm(), 1);

    for re in -4..=4i64 {
        for sp in -4..=4i64 {
            let value: S = SplitComplex::new(re, sp);
            assert_eq!(
                (value * boost).algebraic_norm(),
                value.algebraic_norm() * 5,
                "the norm scales by the multiplier's"
            );
        }
    }
}

#[test]
fn the_four_dimensional_relations_are_what_they_should_be() {
    let one: Q = Q::one();
    let i: Q = SplitQuaternion::imaginary();
    let j: Q = SplitQuaternion::split();
    let k: Q = SplitQuaternion::product();

    // One unit squares to minus one, the other two to plus one.
    assert_eq!(i * i, -one, "i² = -1");
    assert_eq!(j * j, one, "j² = +1");
    assert_eq!(k * k, one, "k² = +1");

    // k is the product of the other two.
    assert_eq!(i * j, k, "ij = k");
    assert_eq!(j * i, -k, "ji = -k");
    assert_eq!(j * k, -i, "jk = -i");
    assert_eq!(k * j, i, "kj = i");
    assert_eq!(k * i, j, "ki = j");
    assert_eq!(i * k, -j, "ik = -j");
}

#[test]
fn multiplication_does_not_commute_in_four_dimensions() {
    let i: Q = SplitQuaternion::imaginary();
    let j: Q = SplitQuaternion::split();

    assert_ne!(i * j, j * i, "which is why it is no commutative ring");
    assert_eq!(i * j, -(j * i), "the units anticommute");

    // It is still associative, which a ring requires.
    let k: Q = SplitQuaternion::product();
    assert_eq!((i * j) * k, i * (j * k));

    let a: Q = SplitQuaternion::new(1, 2, 3, 4);
    let b: Q = SplitQuaternion::new(-1, 1, 0, 2);
    let c: Q = SplitQuaternion::new(2, 0, -1, 1);
    assert_eq!((a * b) * c, a * (b * c), "associative");
    assert_eq!(a * (b + c), a * b + a * c, "distributive");

    // A generic function needing only a ring accepts it; one needing
    // commutativity would not compile for this type.
    fn cube<T: Ring>(value: &T) -> T {
        value.power(3)
    }
    assert_eq!(cube(&i), -i, "i³ = -i");
    assert_eq!(cube(&j), j, "j³ = j");
}

#[test]
fn the_norm_is_indefinite_and_multiplies() {
    let a: Q = SplitQuaternion::new(1, 2, 3, 4);

    // w² + x² - y² - z², so a non-zero value can have a negative norm or none.
    assert_eq!(a.algebraic_norm(), 1 + 4 - 9 - 16);

    let null: Q = SplitQuaternion::new(1, 0, 1, 0);
    assert_eq!(null.algebraic_norm(), 0);
    assert!(null.is_null() && !null.is_zero());

    // Multiplicative, which is what makes it the determinant of the matrix this
    // value stands for.
    let values: Vec<Q> = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (1, 1, 0, 0),
        (2, -1, 3, 1),
        (1, 0, 1, 0),
    ]
    .into_iter()
    .map(|(w, x, y, z)| SplitQuaternion::new(w, x, y, z))
    .collect();

    for a in &values {
        for b in &values {
            assert_eq!(
                (*a * *b).algebraic_norm(),
                a.algebraic_norm() * b.algebraic_norm(),
                "{a} times {b}"
            );
        }
    }
}

#[test]
fn conjugation_gives_the_norm_and_undoes_itself() {
    for value in [
        SplitComplex::new(3i64, 1),
        SplitComplex::new(0, 0),
        SplitComplex::new(-2, 5),
    ] {
        assert_eq!(value.conjugate().conjugate(), value);
        assert_eq!(
            value * value.conjugate(),
            SplitComplex::new(value.algebraic_norm(), 0)
        );
    }

    for value in [
        SplitQuaternion::new(1i64, 2, 3, 4),
        SplitQuaternion::new(0, 1, 0, 0),
        SplitQuaternion::new(2, -1, 3, 1),
    ] {
        assert_eq!(value.conjugate().conjugate(), value);
        assert_eq!(
            value * value.conjugate(),
            SplitQuaternion::new(value.algebraic_norm(), 0, 0, 0),
            "{value}"
        );
    }
}

#[test]
fn division_in_four_dimensions_fails_only_on_the_cone() {
    for (w, x, y, z) in [
        (1.0, 0.0, 0.0, 0.0),
        (2.0, 1.0, 0.0, 0.0),
        (1.0, 2.0, 3.0, 4.0),
        (1.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ] {
        let value: SplitQuaternion<f64> = SplitQuaternion::new(w, x, y, z);

        match value.inverse() {
            Some(inverse) => {
                // Both orders, since multiplication does not commute.
                for product in [value * inverse, inverse * value] {
                    assert!((product.w - 1.0).abs() < 1e-9, "{value}");
                    assert!(product.x.abs() < 1e-9);
                    assert!(product.y.abs() < 1e-9);
                    assert!(product.z.abs() < 1e-9);
                }
            }
            None => assert_eq!(value.algebraic_norm(), 0.0, "{value} is on the cone"),
        }
    }
}

#[test]
fn both_print_readably() {
    assert_eq!(SplitComplex::new(3i64, 4).to_string(), "3 + 4j");
    assert_eq!(SplitComplex::new(3i64, -4).to_string(), "3 - 4j");
    assert_eq!(SplitComplex::new(3i64, 0).to_string(), "3");

    assert_eq!(
        SplitQuaternion::new(1i64, 2, -3, 4).to_string(),
        "1 + 2i - 3j + 4k"
    );
    assert_eq!(Q::one().to_string(), "1 + 0i + 0j + 0k");
}

#[test]
fn scaling_is_component_wise_unlike_multiplying() {
    // The shared macro's operation, which every one of these algebras has.
    assert_eq!(SplitComplex::new(2i64, 3).scale(3), SplitComplex::new(6, 9));
    assert_eq!(
        SplitQuaternion::new(1i64, 2, 3, 4).scale(2),
        SplitQuaternion::new(2, 4, 6, 8)
    );

    // Which is not the same as multiplying by the real value, in general — but
    // is, for these, since a real number is central.
    let three: S = SplitComplex::new(3, 0);
    assert_eq!(SplitComplex::new(2i64, 3) * three, SplitComplex::new(6, 9));
}

#[test]
fn the_older_algebras_are_in_the_hierarchy_too() {
    use voxel_world::math::Quaternion;
    use voxel_world::math::traits::AlgebraicNorm;

    // Complex, Dual and Quaternion now answer to the same traits, so generic
    // code written once serves all five algebras.
    fn identities_behave<T: Ring>(value: &T) -> bool {
        value.clone() + T::zero() == *value
            && value.clone() * T::one() == *value
            && (value.clone() + -value.clone()).is_zero()
    }

    assert!(identities_behave(&Complex::new(3.0, 4.0)));
    assert!(identities_behave(&Dual::new(2.0, 1.0)));
    assert!(identities_behave(&Quaternion::new(1.0, 2.0, 3.0, 4.0)));
    assert!(identities_behave(&SplitComplex::new(3i64, 1)));
    assert!(identities_behave(&SplitQuaternion::new(1i64, 2, 3, 4)));

    // The algebraic norm is the squared magnitude, and multiplies.
    assert_eq!(Complex::new(3.0, 4.0).algebraic_norm(), 25.0);
    assert_eq!(Complex::new(3.0, 4.0).norm(), 5.0, "the magnitude differs");

    let a: Complex<f64> = Complex::new(1.0, 2.0);
    let b: Complex<f64> = Complex::new(-3.0, 1.0);
    assert!(((a * b).algebraic_norm() - a.algebraic_norm() * b.algebraic_norm()).abs() < 1e-9);

    // Dual's norm ignores the dual part, which is why it has zero divisors.
    assert_eq!(Dual::new(2.0, 9.0).algebraic_norm(), 4.0);
    let epsilon: Dual<f64> = Dual::new(0.0, 1.0);
    assert_eq!(epsilon.algebraic_norm(), 0.0);
    assert!(!epsilon.is_zero(), "non-zero with a vanishing norm");

    // A quaternion's is positive for everything but zero, unlike a split one's.
    assert_eq!(Quaternion::new(1.0, 2.0, 3.0, 4.0).algebraic_norm(), 30.0);
    assert!(SplitQuaternion::new(1i64, 2, 3, 4).algebraic_norm() < 0);

    // Conjugation reaches all of them.
    assert_eq!(Complex::new(3.0, 4.0).conjugate(), Complex::new(3.0, -4.0));
    assert_eq!(Dual::new(3.0, 4.0).conjugate(), Dual::new(3.0, -4.0));
    assert_eq!(
        Quaternion::new(1.0, 2.0, 3.0, 4.0).conjugate(),
        Quaternion::new(1.0, -2.0, -3.0, -4.0)
    );

    // Commutativity is claimed only where it holds: two dimensions yes, four no.
    fn commutes<T: CommutativeRing>(a: &T, b: &T) -> bool {
        a.clone() * b.clone() == b.clone() * a.clone()
    }
    assert!(commutes(&Complex::new(1.0, 2.0), &Complex::new(3.0, -1.0)));
    assert!(commutes(&Dual::new(1.0, 2.0), &Dual::new(3.0, -1.0)));
    // `commutes(&Quaternion::..)` would not compile, which is the point.
    let i: Quaternion<f64> = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let j: Quaternion<f64> = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    assert_ne!(i * j, j * i);
}
