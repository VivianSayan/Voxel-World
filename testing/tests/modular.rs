//! Modular arithmetic: the ring for any modulus, the field for a prime one.

use voxel_world::math::traits::{CommutativeRing, EuclideanRing, Field, One, Ring, Semiring, Zero};
use voxel_world::math::{Modulo, PrimeField, is_prime};

type Clock = Modulo<12>;
type F7 = PrimeField<7>;

#[test]
fn values_stay_reduced_however_they_are_made() {
    assert_eq!(Clock::new(0).value(), 0);
    assert_eq!(Clock::new(11).value(), 11);
    assert_eq!(Clock::new(12).value(), 0);
    assert_eq!(Clock::new(25).value(), 1);
    assert_eq!(Clock::new(u64::MAX).value(), u64::MAX % 12);

    // Negatives take the non-negative residue, not the truncated remainder.
    assert_eq!(Clock::from_signed(-1).value(), 11, "not -1");
    assert_eq!(Clock::from_signed(-12).value(), 0);
    assert_eq!(Clock::from_signed(-25).value(), 11);
    assert_eq!(Modulo::<7>::from_signed(-1).value(), 6);

    assert_eq!(Clock::modulus(), 12);
    assert_eq!(Modulo::<7>::modulus(), 7);
}

#[test]
fn the_ring_laws_hold_for_a_composite_modulus() {
    let values: Vec<Clock> = (0..12).map(Modulo::new).collect();

    for a in &values {
        assert_eq!(*a + Clock::zero(), *a);
        assert_eq!(*a * Clock::one(), *a);

        // Every value has an additive inverse, which is what unsigned storage
        // alone cannot give and what makes this a ring rather than a semiring.
        assert_eq!(*a + -*a, Clock::zero(), "{a}");

        for b in &values {
            assert_eq!(*a + *b, *b + *a);
            assert_eq!(*a * *b, *b * *a);
            assert_eq!(*a - *b + *b, *a, "subtraction undoes addition");

            for c in &values {
                assert_eq!((*a + *b) + *c, *a + (*b + *c));
                assert_eq!(*a * (*b + *c), *a * *b + *a * *c);
            }
        }
    }
}

#[test]
fn arithmetic_wraps_the_way_a_clock_does() {
    assert_eq!((Clock::new(10) + Clock::new(5)).value(), 3);
    assert_eq!((Clock::new(3) - Clock::new(5)).value(), 10);
    assert_eq!((Clock::new(5) * Clock::new(5)).value(), 1);
    assert_eq!((-Clock::new(5)).value(), 7);
    assert_eq!((-Clock::new(0)).value(), 0, "zero negates to itself");

    let mut running: Clock = Modulo::new(9);
    running += Modulo::new(6);
    assert_eq!(running.value(), 3);
    running -= Modulo::new(4);
    assert_eq!(running.value(), 11);
    running *= Modulo::new(2);
    assert_eq!(running.value(), 10);
}

#[test]
fn only_values_coprime_with_the_modulus_can_be_divided_by() {
    // In Z/12Z: 1, 5, 7, 11 are invertible; the rest share a factor with 12.
    let invertible: Vec<u64> = (0..12)
        .filter(|value| Clock::new(*value).is_invertible())
        .collect();
    assert_eq!(invertible, vec![1, 5, 7, 11]);

    for value in invertible {
        let element: Clock = Modulo::new(value);
        let inverse = element.inverse().expect("coprime with 12");
        assert_eq!((element * inverse).value(), 1, "{element}");
    }

    assert_eq!(Clock::new(4).inverse(), None, "4 shares a factor with 12");
    assert_eq!(Clock::new(0).inverse(), None);
    assert_eq!(Clock::new(6).divide(Modulo::new(4)), None);
    assert_eq!(Clock::new(10).divide(Modulo::new(5)), Some(Modulo::new(2)));
}

#[test]
fn whether_a_modulus_makes_a_field_is_known_while_compiling() {
    // Worked out by a const fn, so these are constants rather than calls.
    const SEVEN_IS: bool = Modulo::<7>::IS_FIELD;
    const TWELVE_IS: bool = Modulo::<12>::IS_FIELD;

    // Clippy notes these are constant, which is the point being made: the
    // answer was settled by the compiler, not worked out at run time.
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(SEVEN_IS);
        assert!(!TWELVE_IS);
    }

    assert!(is_prime(2) && is_prime(3) && is_prime(5) && is_prime(7));
    assert!(is_prime(97) && is_prime(7919));
    assert!(!is_prime(0) && !is_prime(1) && !is_prime(4) && !is_prime(9));
    assert!(!is_prime(7917), "3 * 7 * 13 * 29");
    assert!(is_prime(2_147_483_647), "a Mersenne prime");
}

#[test]
fn every_value_but_zero_divides_in_a_prime_field() {
    for value in 1..7u64 {
        let element: F7 = PrimeField::new(value);
        let inverse = element.inverse().expect("a prime field inverts everything");

        assert_eq!((element * inverse).value(), 1, "{element}");
        assert_eq!(element.divide(&element), Some(F7::one()));
    }

    assert_eq!(F7::new(0).inverse(), None, "zero never divides");
    assert_eq!(F7::new(3).divide(&F7::new(0)), None);

    // A larger field, to exercise the widened multiplication.
    type Big = PrimeField<2_147_483_647>;
    let value: Big = PrimeField::new(2_147_483_646);
    let inverse = value.inverse().expect("prime modulus");
    assert_eq!((value * inverse).value(), 1);
}

#[test]
fn fermats_little_theorem_holds_which_is_what_primality_buys() {
    // a^(p-1) == 1 for every non-zero a, in every prime field.
    fn check<const P: u64>() {
        for value in 1..P {
            let element: PrimeField<P> = PrimeField::new(value);
            assert_eq!(
                element.power((P - 1) as u32).value(),
                1,
                "{element} to the {}",
                P - 1
            );
        }
    }

    check::<2>();
    check::<3>();
    check::<7>();
    check::<13>();
    check::<97>();

    // And it fails for a composite modulus, which is why the type is separate.
    assert_ne!(Modulo::<12>::new(5).power(11).value(), 1);
}

#[test]
fn a_prime_field_is_euclidean_in_the_way_every_field_is() {
    // Written so that code expecting div_rem accepts a field without a special
    // case: the quotient is exact and the remainder always zero.
    let a: F7 = PrimeField::new(5);
    let b: F7 = PrimeField::new(3);

    let (quotient, remainder) = a.div_rem(&b).expect("b is not zero");
    assert_eq!(quotient * b + remainder, a);
    assert!(remainder.is_zero(), "a field leaves nothing over");
    assert!(b.divides(&a), "everything divides everything");

    assert_eq!(a.div_rem(&F7::zero()), None);
    assert_eq!(F7::zero().euclidean_size(), 0);
    assert_eq!(a.euclidean_size(), 1);

    // So the trait's gcd terminates here too, trivially.
    assert!(!a.gcd(&b).is_zero());
}

#[test]
fn the_traits_carry_over_so_generic_code_accepts_both() {
    // The same function over a ring and over a field.
    fn sum_of_powers<T: Semiring>(base: &T, up_to: u32) -> T {
        (0..=up_to).fold(T::zero(), |total, power| total + base.power(power))
    }

    // 1 + 3 + 9 = 13, which is 1 mod 12.
    assert_eq!(sum_of_powers(&Clock::new(3), 2).value(), 1);
    assert_eq!(sum_of_powers(&F7::new(2), 3).value(), 15 % 7);

    fn negate_twice<T: Ring>(value: &T) -> T {
        -(-value.clone())
    }
    assert_eq!(negate_twice(&Clock::new(5)), Clock::new(5));
    assert_eq!(negate_twice(&F7::new(5)), F7::new(5));

    fn halve<T: Field>(value: &T) -> Option<T> {
        value.divide(&(T::one() + T::one()))
    }
    // In GF(7), 2 inverts to 4, so half of 3 is 5 — and doubling it returns 3.
    let half = halve(&F7::new(3)).expect("2 is invertible");
    assert_eq!((half + half).value(), 3);

    fn is_commutative<T: CommutativeRing>(a: &T, b: &T) -> bool {
        a.clone() * b.clone() == b.clone() * a.clone()
    }
    assert!(is_commutative(&Clock::new(7), &Clock::new(5)));
    assert!(is_commutative(&F7::new(3), &F7::new(4)));
}

#[test]
fn the_degenerate_moduli_behave() {
    // Z/1Z is the zero ring: one element, where one and zero coincide.
    type Trivial = Modulo<1>;
    assert_eq!(Trivial::new(5).value(), 0);
    assert_eq!(Trivial::one().value(), 0);
    assert!(Trivial::one().is_zero() && Trivial::one().is_one());
    assert_eq!((Trivial::one() + Trivial::one()).value(), 0);

    // Z/2Z is the smallest field: addition is exclusive or.
    type Bit = PrimeField<2>;
    assert_eq!((Bit::one() + Bit::one()).value(), 0);
    assert_eq!(Bit::one().inverse(), Some(Bit::one()));
    assert_eq!((-Bit::one()).value(), 1, "its own additive inverse");
}

#[test]
fn both_print_with_the_modulus_since_the_value_alone_is_ambiguous() {
    assert_eq!(Clock::new(10).to_string(), "10 (mod 12)");
    assert_eq!(Clock::from_signed(-1).to_string(), "11 (mod 12)");
    assert_eq!(F7::new(3).to_string(), "3", "the field is in the type");

    // And convert from either sign of integer.
    assert_eq!(Clock::from(25u64), Clock::new(1));
    assert_eq!(Clock::from(-1i64), Clock::new(11));
    assert_eq!(F7::from(9u64).value(), 2);
    assert_eq!(F7::from(-1i64).value(), 6);

    // A field element can drop to the ring beneath it.
    assert_eq!(F7::new(4).residue(), Modulo::<7>::new(4));
    assert_eq!(F7::order(), 7);
}
