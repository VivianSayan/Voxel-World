//! Polynomials over a ring, and GF(pⁿ) as a quotient of them.

use voxel_world::math::polynomial::{Extension, Modulus, is_irreducible};
use voxel_world::math::traits::{EuclideanRing, Field, One, Ring, Semiring, Zero};
use voxel_world::math::{Polynomial, PrimeField, is_prime};

type Z = Polynomial<i64>;
type F2 = PrimeField<2>;
type F5 = PrimeField<5>;

/// GF(4) = GF(2) adjoined a root of x² + x + 1.
struct Quadratic;

impl Modulus<2> for Quadratic {
    const DEGREE: usize = 2;

    fn polynomial() -> Polynomial<F2> {
        Polynomial::new(vec![F2::new(1), F2::new(1), F2::new(1)])
    }
}

/// GF(8) = GF(2) adjoined a root of x³ + x + 1.
struct Cubic;

impl Modulus<2> for Cubic {
    const DEGREE: usize = 3;

    fn polynomial() -> Polynomial<F2> {
        Polynomial::new(vec![F2::new(1), F2::new(1), F2::new(0), F2::new(1)])
    }
}

/// GF(25) = GF(5) adjoined a root of x² + 2.
struct FiveSquared;

impl Modulus<5> for FiveSquared {
    const DEGREE: usize = 2;

    fn polynomial() -> Polynomial<F5> {
        Polynomial::new(vec![F5::new(2), F5::new(0), F5::new(1)])
    }
}

type GF4 = Extension<2, Quadratic>;
type GF8 = Extension<2, Cubic>;
type GF25 = Extension<5, FiveSquared>;

#[test]
fn a_polynomial_knows_its_degree_and_keeps_no_trailing_zeros() {
    let p: Z = Polynomial::new(vec![1, 2, 3]);
    assert_eq!(p.degree(), Some(2));
    assert_eq!(p.coefficients(), &[1, 2, 3]);
    assert_eq!(p.leading_coefficient(), Some(3));

    // Trailing zeros are dropped, so the degree means something and equality is
    // structural.
    let padded: Z = Polynomial::new(vec![1, 2, 3, 0, 0]);
    assert_eq!(padded, p);
    assert_eq!(padded.degree(), Some(2));

    // Zero has no degree at all, rather than a conventional one.
    let zero: Z = Polynomial::zero();
    assert_eq!(zero.degree(), None);
    assert!(zero.is_zero());
    assert_eq!(Polynomial::new(vec![0, 0, 0]), zero);

    assert_eq!(Z::one().degree(), Some(0));
    assert!(Z::one().is_one() && Z::one().is_monic());
    assert_eq!(Z::variable().degree(), Some(1));
    assert_eq!(Polynomial::term(7i64, 3).coefficients(), &[0, 0, 0, 7]);
    assert_eq!(p.coefficient(9), 0, "beyond the degree is zero");
}

#[test]
fn the_ring_laws_hold_over_the_integers() {
    let values: Vec<Z> = vec![
        Polynomial::zero(),
        Polynomial::one(),
        Polynomial::new(vec![1, 2]),
        Polynomial::new(vec![-3, 0, 1]),
        Polynomial::new(vec![2, -1, 0, 4]),
    ];

    for a in &values {
        assert_eq!(a.clone() + Z::zero(), *a);
        assert_eq!(a.clone() * Z::one(), *a);
        assert_eq!(a.clone() + -a.clone(), Z::zero());
        assert_eq!(a.clone() * Z::zero(), Z::zero());

        for b in &values {
            assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());
            assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());
            assert_eq!(a.clone() - b.clone() + b.clone(), *a);

            // Degrees add, unless something is zero.
            if let (Some(left), Some(right)) = (a.degree(), b.degree()) {
                assert_eq!((a.clone() * b.clone()).degree(), Some(left + right));
            }

            for c in &values {
                assert_eq!(
                    a.clone() * (b.clone() + c.clone()),
                    a.clone() * b.clone() + a.clone() * c.clone(),
                    "distributive"
                );
            }
        }
    }
}

#[test]
fn evaluation_agrees_with_the_arithmetic() {
    // (1 + 2x + 3x²) at 2 is 17.
    let p: Z = Polynomial::new(vec![1, 2, 3]);
    assert_eq!(p.evaluate(&0), 1);
    assert_eq!(p.evaluate(&1), 6);
    assert_eq!(p.evaluate(&2), 17);
    assert_eq!(p.evaluate(&-1), 2);

    // Evaluating a product is multiplying the evaluations, at every point.
    let q: Z = Polynomial::new(vec![-1, 1]);
    for at in -4..=4i64 {
        assert_eq!(
            (p.clone() * q.clone()).evaluate(&at),
            p.evaluate(&at) * q.evaluate(&at)
        );
        assert_eq!(
            (p.clone() + q.clone()).evaluate(&at),
            p.evaluate(&at) + q.evaluate(&at)
        );
    }

    // x - 1 has a root at 1.
    assert_eq!(q.evaluate(&1), 0);
}

#[test]
fn the_derivative_lowers_the_degree_and_obeys_the_product_rule() {
    // d/dx (1 + 2x + 3x²) = 2 + 6x.
    let p: Z = Polynomial::new(vec![1, 2, 3]);
    assert_eq!(p.derivative(), Polynomial::new(vec![2, 6]));

    assert_eq!(Z::one().derivative(), Z::zero(), "a constant has none");
    assert_eq!(Z::zero().derivative(), Z::zero());
    assert_eq!(Z::variable().derivative(), Z::one());

    let q: Z = Polynomial::new(vec![0, -1, 4]);
    assert_eq!(
        (p.clone() * q.clone()).derivative(),
        p.derivative() * q.clone() + p.clone() * q.derivative(),
        "the product rule"
    );

    // Over GF(2) the squaring term vanishes, since its degree is the
    // characteristic: d/dx x² = 2x = 0.
    let squared: Polynomial<F2> = Polynomial::term(F2::one(), 2);
    assert!(squared.derivative().is_zero());
}

#[test]
fn division_over_a_field_leaves_a_remainder_of_lower_degree() {
    // x² - 1 = (x - 1)(x + 1) exactly, over GF(5).
    let difference: Polynomial<F5> =
        Polynomial::new(vec![F5::from_signed(-1), F5::new(0), F5::new(1)]);
    let factor: Polynomial<F5> = Polynomial::new(vec![F5::from_signed(-1), F5::new(1)]);

    let (quotient, remainder) = difference.div_rem(&factor).expect("not dividing by zero");
    assert!(remainder.is_zero(), "it divides exactly");
    assert_eq!(
        quotient,
        Polynomial::new(vec![F5::new(1), F5::new(1)]),
        "x + 1"
    );
    assert!(factor.divides(&difference));

    // And where it does not divide, the remainder is smaller than the divisor.
    let awkward: Polynomial<F5> =
        Polynomial::new(vec![F5::new(1), F5::new(2), F5::new(3), F5::new(4)]);

    for constant in 0..5u64 {
        for linear in 1..5u64 {
            let divisor: Polynomial<F5> = Polynomial::new(vec![F5::new(constant), F5::new(linear)]);
            let (quotient, remainder) = awkward.div_rem(&divisor).expect("non-zero divisor");

            assert_eq!(quotient * divisor.clone() + remainder.clone(), awkward);
            assert!(
                remainder.euclidean_size() < divisor.euclidean_size(),
                "{remainder} against {divisor}"
            );
        }
    }

    assert_eq!(awkward.div_rem(&Polynomial::zero()), None);
}

#[test]
fn the_greatest_common_divisor_is_monic_and_divides_both() {
    // (x - 1)(x - 2) and (x - 1)(x - 3) share x - 1.
    let root = |value: i64| -> Polynomial<F5> {
        Polynomial::new(vec![F5::from_signed(-value), F5::new(1)])
    };

    let left = root(1) * root(2);
    let right = root(1) * root(3);
    let divisor = left.gcd_normalised(&right);

    assert_eq!(divisor, root(1), "x - 1, monic already");
    assert!(divisor.is_monic(), "the canonical associate over a field");
    assert!(divisor.divides(&left) && divisor.divides(&right));

    // Scaling either side changes nothing, which is what normalising is for.
    let scaled = left.clone() * Polynomial::constant(F5::new(3));
    assert_eq!(scaled.gcd_normalised(&right), divisor);

    // Coprime polynomials share only a constant.
    assert_eq!(root(2).gcd_normalised(&root(3)), Polynomial::one());

    // And the extended form produces the coefficients that build it.
    let (found, first, second) = left.extended_gcd(&right);
    assert_eq!(first * left + second * right, found);
}

#[test]
fn a_polynomial_over_a_ring_needs_no_field_to_add_or_multiply() {
    // The point of splitting the bounds: Z[x] is a ring and has no division.
    fn square<T: Ring>(value: &Polynomial<T>) -> Polynomial<T> {
        value.clone() * value.clone()
    }

    let p: Z = Polynomial::new(vec![1, 1]);
    assert_eq!(square(&p), Polynomial::new(vec![1, 2, 1]), "(1 + x)²");

    // Over the natural numbers it is only a semiring, and that is enough to
    // evaluate.
    let counts: Polynomial<u32> = Polynomial::new(vec![1, 2, 3]);
    assert_eq!(counts.evaluate(&2), 17);
    assert_eq!(counts.degree(), Some(2));
}

#[test]
fn irreducibility_is_checked_rather_than_assumed() {
    // x² + x + 1 has no root in GF(2), so it is irreducible.
    assert!(is_irreducible(&Quadratic::polynomial()));
    assert!(is_irreducible(&Cubic::polynomial()));
    assert!(is_irreducible(&FiveSquared::polynomial()));

    // x² + 1 factors over GF(2) as (x + 1)², so it defines no field.
    let reducible: Polynomial<F2> = Polynomial::new(vec![F2::new(1), F2::new(0), F2::new(1)]);
    assert!(!is_irreducible(&reducible));

    let linear: Polynomial<F2> = Polynomial::new(vec![F2::new(1), F2::new(1)]);
    assert!(is_irreducible(&linear), "degree one always is");
    assert!(
        !is_irreducible(&Polynomial::<F2>::one()),
        "a constant is not"
    );
    assert!(!is_irreducible(&Polynomial::<F2>::zero()));

    // x² + 2 has no square root of -2 in GF(5), so it is irreducible there.
    let squares: Vec<u64> = (0..5u64).map(|value| (value * value) % 5).collect();
    assert!(!squares.contains(&3), "-2 = 3 is not a square mod 5");
}

#[test]
fn every_element_of_a_finite_field_but_zero_divides() {
    // GF(4): four elements, three of them invertible.
    let elements: Vec<GF4> = (0..4u64)
        .map(|bits| {
            Extension::new(Polynomial::new(vec![
                F2::new(bits & 1),
                F2::new((bits >> 1) & 1),
            ]))
        })
        .collect();

    assert_eq!(GF4::order(), 4);
    assert_eq!(elements.len(), 4);

    for element in &elements {
        if element.is_zero() {
            assert_eq!(element.inverse(), None);
            continue;
        }

        let inverse = element.inverse().expect("a field inverts everything else");
        assert!((element.clone() * inverse).is_one(), "{element}");
    }

    // And GF(8) and GF(25) likewise, which exercises a cubic and a larger base.
    assert_eq!(GF8::order(), 8);
    assert_eq!(GF25::order(), 25);

    let x: GF8 = Extension::variable();
    assert!((x.clone() * x.inverse().unwrap()).is_one());

    let y: GF25 = Extension::variable();
    assert!((y.clone() * y.inverse().unwrap()).is_one());
}

#[test]
fn the_multiplicative_group_of_a_finite_field_is_cyclic() {
    // x generates every non-zero element of GF(4), so x³ = 1.
    let x: GF4 = Extension::variable();
    assert!(x.power(3).is_one());
    assert!(
        !x.power(1).is_one() && !x.power(2).is_one(),
        "order exactly 3"
    );

    // In GF(8) the non-zero elements number seven, so x⁷ = 1.
    let x: GF8 = Extension::variable();
    assert!(x.power(7).is_one());
    assert!(!x.power(1).is_one());

    // And every non-zero element satisfies a^(q-1) = 1, which is Fermat again.
    let elements: Vec<GF4> = (1..4u64)
        .map(|bits| {
            Extension::new(Polynomial::new(vec![
                F2::new(bits & 1),
                F2::new((bits >> 1) & 1),
            ]))
        })
        .collect();

    for element in &elements {
        assert!(element.power(3).is_one(), "{element}");
    }
}

#[test]
fn gf4_is_not_the_integers_modulo_four() {
    // The reason the extension exists: Z/4Z has 2 * 2 = 0, so 2 cannot be
    // inverted and it is no field. GF(4) has no such element.
    use voxel_world::math::Modulo;

    let two: Modulo<4> = Modulo::new(2);
    assert!((two * two).is_zero(), "a zero divisor");
    assert_eq!(two.inverse(), None);
    const { assert!(!Modulo::<4>::IS_FIELD) };

    // Whereas in GF(4) nothing but zero multiplies to zero.
    for left in 1..4u64 {
        for right in 1..4u64 {
            let a: GF4 = Extension::new(Polynomial::new(vec![
                F2::new(left & 1),
                F2::new((left >> 1) & 1),
            ]));
            let b: GF4 = Extension::new(Polynomial::new(vec![
                F2::new(right & 1),
                F2::new((right >> 1) & 1),
            ]));

            assert!(!(a * b).is_zero(), "no zero divisors in a field");
        }
    }
}

#[test]
fn the_field_laws_hold_across_a_whole_finite_field() {
    let elements: Vec<GF8> = (0..8u64)
        .map(|bits| {
            Extension::new(Polynomial::new(vec![
                F2::new(bits & 1),
                F2::new((bits >> 1) & 1),
                F2::new((bits >> 2) & 1),
            ]))
        })
        .collect();

    for a in &elements {
        assert_eq!(a.clone() + GF8::zero(), *a);
        assert_eq!(a.clone() * GF8::one(), *a);
        assert_eq!(a.clone() + -a.clone(), GF8::zero());

        for b in &elements {
            assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());
            assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());
            assert_eq!(a.clone() - b.clone() + b.clone(), *a);

            for c in elements.iter().take(4) {
                assert_eq!(
                    (a.clone() + b.clone()) + c.clone(),
                    a.clone() + (b.clone() + c.clone())
                );
                assert_eq!(
                    a.clone() * (b.clone() + c.clone()),
                    a.clone() * b.clone() + a.clone() * c.clone()
                );
            }
        }
    }
}

#[test]
fn the_primality_test_agrees_with_trial_division() {
    // Miller-Rabin replaced trial division, so the two are compared directly
    // over a range where the slow one is still affordable.
    fn by_trial(candidate: u64) -> bool {
        if candidate < 2 {
            return false;
        }

        let mut divisor: u64 = 2;

        while divisor * divisor <= candidate {
            if candidate.is_multiple_of(divisor) {
                return false;
            }

            divisor += 1;
        }

        true
    }

    for candidate in 0..20_000u64 {
        assert_eq!(is_prime(candidate), by_trial(candidate), "at {candidate}");
    }

    // The cases that catch a witness set that is too small: strong pseudoprimes
    // to the first few bases.
    for composite in [
        2047u64,
        1_373_653,
        25_326_001,
        3_215_031_751,
        2_152_302_898_747,
    ] {
        assert!(!is_prime(composite), "{composite} is composite");
    }

    // And large primes, where trial division would take billions of steps.
    for prime in [
        2_147_483_647u64,
        1_000_000_007,
        67_280_421_310_721,
        18_446_744_073_709_551_557,
    ] {
        assert!(is_prime(prime), "{prime} is prime");
    }

    assert!(
        !is_prime(u64::MAX),
        "3 * 5 * 17 * 257 * 641 * 65537 * 6700417"
    );
}

#[test]
fn polynomials_print_highest_degree_first() {
    assert_eq!(
        Polynomial::new(vec![1i64, 2, 3]).to_string(),
        "3x^2 + 2x + 1"
    );
    assert_eq!(Polynomial::new(vec![0i64, 1]).to_string(), "x");
    assert_eq!(Polynomial::new(vec![5i64]).to_string(), "5");
    assert_eq!(Z::zero().to_string(), "0");
    assert_eq!(
        Polynomial::new(vec![1i64, 0, 1]).to_string(),
        "x^2 + 1",
        "a zero term is skipped"
    );

    // From an iterator of coefficients, lowest first.
    let collected: Z = [1i64, 2, 3].into_iter().collect();
    assert_eq!(collected, Polynomial::new(vec![1, 2, 3]));
}
