//! Interval arithmetic, and the sparse form that keeps the gaps.

use voxel_world::math::traits::{One, Zero};
use voxel_world::math::{Interval, IntervalSet};

type I = Interval<i64>;
type S = IntervalSet<i64>;

fn span(low: i64, high: i64) -> I {
    Interval::new(low, high).expect("ordered")
}

#[test]
fn an_interval_holds_its_ends_and_refuses_to_be_backwards() {
    let a: I = span(1, 3);

    assert_eq!(*a.low(), 1);
    assert_eq!(*a.high(), 3);
    assert_eq!(a.width(), 2);
    assert!(a.contains(&1) && a.contains(&2) && a.contains(&3));
    assert!(!a.contains(&0) && !a.contains(&4));

    assert_eq!(Interval::new(3, 1), None, "backwards is no interval");
    assert_eq!(Interval::between(3, 1), a, "unless the order is asked for");

    let point: I = Interval::point(5);
    assert!(point.is_point());
    assert_eq!(point.width(), 0);
    assert!(!a.is_point());
}

#[test]
fn addition_adds_the_ends_and_subtraction_crosses_them() {
    assert_eq!(span(1, 3) + span(-2, 1), span(-1, 4));
    assert_eq!(span(0, 0) + span(2, 5), span(2, 5));

    // The largest difference is the largest minus the smallest.
    assert_eq!(span(1, 3) - span(-2, 1), span(0, 5));
    assert_eq!(-span(1, 3), span(-3, -1), "negation swaps the ends");

    // Every sum of members is in the result, and the ends are attained.
    let a: I = span(1, 3);
    let b: I = span(-2, 1);
    let sum = a + b;

    for left in 1..=3i64 {
        for right in -2..=1i64 {
            assert!(sum.contains(&(left + right)), "{left} + {right}");
        }
    }
    assert_eq!(*sum.low(), 1 + -2);
    assert_eq!(*sum.high(), 3 + 1);
}

#[test]
fn subtracting_an_interval_from_itself_does_not_cancel() {
    // The reason this is not a ring: there is no additive inverse.
    let a: I = span(1, 2);

    assert_eq!(a - a, span(-1, 1));
    assert_ne!(a - a, I::zero(), "a range does not cancel with itself");

    // Only a single point does.
    let point: I = Interval::point(7);
    assert_eq!(point - point, I::zero());
    assert!(I::zero().is_zero() && I::one().is_one());
}

#[test]
fn multiplication_must_try_all_four_corners() {
    // A negative end times a negative end is the largest product, so which
    // corner wins depends on the signs.
    assert_eq!(span(1, 3) * span(-2, 1), span(-6, 3));
    assert_eq!(span(-3, -1) * span(-2, -1), span(1, 6), "both negative");
    assert_eq!(
        span(-2, 3) * span(-4, 5),
        span(-12, 15),
        "both straddle zero"
    );
    assert_eq!(span(2, 3) * span(4, 5), span(8, 15), "both positive");

    // Exhaustively: the product holds every product of members, and its ends are
    // attained by some pair.
    for al in -4..=4i64 {
        for ah in al..=4i64 {
            for bl in -4..=4i64 {
                for bh in bl..=4i64 {
                    let product = span(al, ah) * span(bl, bh);
                    let mut lowest = i64::MAX;
                    let mut highest = i64::MIN;

                    for left in al..=ah {
                        for right in bl..=bh {
                            assert!(product.contains(&(left * right)));
                            lowest = lowest.min(left * right);
                            highest = highest.max(left * right);
                        }
                    }

                    assert_eq!(*product.low(), lowest, "tight lower end");
                    assert_eq!(*product.high(), highest, "tight upper end");
                }
            }
        }
    }
}

#[test]
fn distribution_only_holds_one_way_round() {
    // The second reason this is not a ring: X(Y + Z) is contained in XY + XZ
    // rather than equal to it.
    let x: I = span(-1, 1);
    let y: I = Interval::point(1);
    let z: I = Interval::point(-1);

    let together = x * (y + z);
    let apart = x * y + x * z;

    assert_eq!(together, I::zero());
    assert_eq!(apart, span(-2, 2));
    assert_ne!(together, apart, "distributivity fails");
    assert!(apart.encloses(&together), "but containment holds");

    // Which is true generally, and is the property the arithmetic does promise.
    for xl in -3..=3i64 {
        for xh in xl..=3i64 {
            for yl in -3..=3i64 {
                for yh in yl..=3i64 {
                    let x = span(xl, xh);
                    let y = span(yl, yh);
                    let z = span(-2, 1);

                    assert!((x * y + x * z).encloses(&(x * (y + z))));
                }
            }
        }
    }
}

#[test]
fn intersection_and_hull_do_what_they_say() {
    assert_eq!(span(1, 5).intersect(&span(3, 8)), Some(span(3, 5)));
    assert_eq!(span(1, 2).intersect(&span(5, 6)), None, "disjoint");
    assert_eq!(span(1, 5).intersect(&span(5, 9)), Some(Interval::point(5)));

    assert_eq!(span(1, 2).hull(&span(5, 6)), span(1, 6), "the gap is lost");
    assert!(span(1, 9).encloses(&span(3, 4)));
    assert!(!span(3, 4).encloses(&span(1, 9)));
    assert!(span(1, 5).overlaps(&span(5, 9)));
    assert!(!span(1, 4).overlaps(&span(5, 9)));
}

#[test]
fn division_needs_a_divisor_that_avoids_zero() {
    let a: Interval<f64> = Interval::new(1.0, 2.0).unwrap();
    let b: Interval<f64> = Interval::new(2.0, 4.0).unwrap();

    let quotient = a.divide(&b).expect("b avoids zero");
    assert!((quotient.low() - 0.25).abs() < 1e-9);
    assert!((quotient.high() - 1.0).abs() < 1e-9);

    // A divisor straddling zero has no single-interval answer.
    let straddling: Interval<f64> = Interval::new(-1.0, 1.0).unwrap();
    assert!(straddling.contains_zero());
    assert_eq!(a.divide(&straddling), None);
    assert_eq!(straddling.reciprocal(), None);

    // Nor does one that only touches it.
    let touching: Interval<f64> = Interval::new(0.0, 1.0).unwrap();
    assert!(touching.contains_zero());
    assert_eq!(touching.reciprocal(), None);
}

#[test]
fn a_sparse_set_keeps_its_gaps_where_one_interval_cannot() {
    let split: S = IntervalSet::new(vec![span(0, 1), span(5, 6)]);
    let shift: S = IntervalSet::from(span(0, 1));

    // The headline property: the gap survives addition.
    let moved = split.clone() + shift;
    assert_eq!(moved, IntervalSet::new(vec![span(0, 2), span(5, 7)]));
    assert_eq!(moved.len(), 2, "still two pieces");

    // Where a single interval has to swallow it.
    assert_eq!(split.hull(), Some(span(0, 6)));
    assert_eq!(
        split.hull().unwrap() + span(0, 1),
        span(0, 7),
        "the hull claims everything between"
    );

    // And the gap is genuinely absent from the set.
    assert!(moved.contains(&0) && moved.contains(&2));
    assert!(!moved.contains(&3) && !moved.contains(&4));
    assert!(moved.contains(&5) && moved.contains(&7));
}

#[test]
fn pieces_are_sorted_merged_and_unique() {
    // Given out of order, overlapping and touching, one set comes back.
    let messy: S = IntervalSet::new(vec![
        span(5, 6),
        span(0, 2),
        span(1, 3),
        span(3, 4),
        span(10, 10),
    ]);

    // [0,2], [1,3] and [3,4] overlap or touch, so they become [0,4]. [5,6] does
    // not touch it: these are ranges of a continuum, and 4.5 lies in neither —
    // so they stay apart even though no *integer* separates them.
    assert_eq!(messy.parts(), &[span(0, 4), span(5, 6), span(10, 10)]);
    assert_eq!(messy.len(), 3);

    // Touching pieces merge, since [0,1] and [1,2] share the point 1.
    let touching: S = IntervalSet::new(vec![span(0, 1), span(1, 2)]);
    assert_eq!(touching.parts(), &[span(0, 2)]);

    // So the representation is unique and equality means what it should.
    assert_eq!(
        IntervalSet::new(vec![span(0, 2), span(1, 3)]),
        IntervalSet::from(span(0, 3))
    );

    assert!(S::empty().is_empty());
    assert_eq!(S::empty().hull(), None);
    assert_eq!(S::empty().len(), 0);
}

#[test]
fn set_operations_behave() {
    let left: S = IntervalSet::new(vec![span(0, 3), span(8, 10)]);
    let right: S = IntervalSet::new(vec![span(2, 9)]);

    assert_eq!(left.union(&right), IntervalSet::from(span(0, 10)));
    assert_eq!(
        left.intersect(&right),
        IntervalSet::new(vec![span(2, 3), span(8, 9)])
    );

    // Intersecting with something disjoint gives nothing.
    assert!(left.intersect(&IntervalSet::from(span(20, 30))).is_empty());

    // Membership agrees with the pieces, everywhere.
    for value in -2..=12i64 {
        assert_eq!(
            left.intersect(&right).contains(&value),
            left.contains(&value) && right.contains(&value),
            "at {value}"
        );
        assert_eq!(
            left.union(&right).contains(&value),
            left.contains(&value) || right.contains(&value),
            "at {value}"
        );
    }
}

#[test]
fn multiplying_sets_pairs_every_piece() {
    let a: S = IntervalSet::new(vec![span(1, 2), span(10, 11)]);
    let b: S = IntervalSet::from(span(2, 3));

    let product = a.clone() * b.clone();

    // Two pieces times one gives two, which stay apart.
    assert_eq!(product.len(), 2);
    assert_eq!(product, IntervalSet::new(vec![span(2, 6), span(20, 33)]));

    // Every product of members is in the result.
    for left in [1, 2, 10, 11i64] {
        for right in 2..=3i64 {
            assert!(product.contains(&(left * right)), "{left} * {right}");
        }
    }

    // Nothing in the gap is.
    assert!(!product.contains(&10));
    assert!(!product.contains(&19));

    assert_eq!(
        -a.clone(),
        IntervalSet::new(vec![span(-11, -10), span(-2, -1)])
    );
    assert!(S::zero().is_zero() && S::one().is_one());
}

#[test]
fn a_sparse_reciprocal_splits_a_piece_that_straddles_zero() {
    // What the sparse form buys over a single interval: a divisor holding zero
    // still gives something, by splitting at zero.
    let straddling: IntervalSet<f64> = IntervalSet::from(Interval::new(-1.0, 1.0).unwrap());

    assert_eq!(
        Interval::new(-1.0, 1.0).unwrap().reciprocal(),
        None,
        "one interval cannot"
    );

    let inverted = straddling.reciprocal();
    assert!(!inverted.is_empty(), "the sparse form can");

    // A piece away from zero inverts exactly.
    let clean: IntervalSet<f64> = IntervalSet::from(Interval::new(2.0, 4.0).unwrap());
    let parts = clean.reciprocal();
    assert_eq!(parts.len(), 1);
    assert!((parts.parts()[0].low() - 0.25).abs() < 1e-9);
    assert!((parts.parts()[0].high() - 0.5).abs() < 1e-9);

    // And the pieces either side of zero are handled separately.
    let two_sided: IntervalSet<f64> = IntervalSet::new(vec![
        Interval::new(-4.0, -2.0).unwrap(),
        Interval::new(2.0, 4.0).unwrap(),
    ]);
    let inverted = two_sided.reciprocal();
    assert_eq!(inverted.len(), 2, "no zero to straddle, so both invert");
    assert!(inverted.contains(&0.25) && inverted.contains(&-0.25));
}

#[test]
fn both_print_readably() {
    assert_eq!(span(1, 3).to_string(), "[1, 3]");
    assert_eq!(Interval::point(5i64).to_string(), "[5, 5]");

    assert_eq!(
        IntervalSet::new(vec![span(0, 1), span(5, 6)]).to_string(),
        "[0, 1] \u{222a} [5, 6]"
    );
    assert_eq!(S::empty().to_string(), "\u{2205}");

    // And a set can be collected from pieces.
    let collected: S = [span(5, 6), span(0, 1)].into_iter().collect();
    assert_eq!(collected, IntervalSet::new(vec![span(0, 1), span(5, 6)]));
}
