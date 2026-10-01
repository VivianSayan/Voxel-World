//! Matrices whose shape is in their type, and how they act on vectors.

use voxel_world::math::traits::{LinearMap, One, Ring, Zero};
use voxel_world::math::{Matrix, PrimeField, Vector2, Vector3, Vector4};

type M2 = Matrix<i64, 2, 2>;
type M3 = Matrix<i64, 3, 3>;

#[test]
fn the_shape_is_part_of_the_type() {
    assert_eq!(Matrix::<i64, 2, 3>::shape(), (2, 3));
    assert!(M3::is_square());
    assert!(!Matrix::<i64, 2, 3>::is_square());

    let a: Matrix<i64, 2, 3> = Matrix::from_rows([[1, 2, 3], [4, 5, 6]]);

    assert_eq!(a.row(0), &[1, 2, 3]);
    assert_eq!(a.row(1), &[4, 5, 6]);
    assert_eq!(a.column(1), [2, 5]);
    assert_eq!(a.get(1, 2), Some(&6));
    assert_eq!(a.get(2, 0), None, "outside the matrix");

    // Transposing swaps the dimensions in the type as well as the values.
    let flipped: Matrix<i64, 3, 2> = a.transpose();
    assert_eq!(flipped.row(0), &[1, 4]);
    assert_eq!(flipped.transpose(), a, "twice returns");

    // Built from columns instead.
    let by_column: Matrix<i64, 2, 3> = Matrix::from_columns([[1, 4], [2, 5], [3, 6]]);
    assert_eq!(by_column, a);
}

#[test]
fn multiplication_cancels_the_inner_dimension() {
    let a: Matrix<i64, 2, 3> = Matrix::from_rows([[1, 2, 3], [4, 5, 6]]);
    let b: Matrix<i64, 3, 2> = Matrix::from_rows([[7, 8], [9, 10], [11, 12]]);

    // 2x3 times 3x2 is 2x2, and the type says so.
    let product: Matrix<i64, 2, 2> = a * b;
    assert_eq!(product.row(0), &[58, 64]);
    assert_eq!(product.row(1), &[139, 154]);

    // The other order gives 3x3 instead, which is a different type entirely.
    let other: Matrix<i64, 3, 3> = b * a;
    assert_eq!(other.row(0), &[39, 54, 69]);

    // The identity does nothing, from either side.
    let m: M3 = Matrix::from_rows([[1, 2, 3], [4, 5, 6], [7, 8, 10]]);
    assert_eq!(m * M3::identity(), m);
    assert_eq!(M3::identity() * m, m);
    assert!(M3::identity().is_one());
    assert!(M3::zeroed().is_zero());
}

#[test]
fn square_matrices_are_a_ring_but_not_a_commutative_one() {
    let a: M2 = Matrix::from_rows([[1, 2], [3, 4]]);
    let b: M2 = Matrix::from_rows([[0, 1], [1, 0]]);

    // Ring laws.
    assert_eq!(a + M2::zero(), a);
    assert_eq!(a * M2::one(), a);
    assert_eq!(a + -a, M2::zero());
    assert_eq!(a - b + b, a);
    assert_eq!(a * (b + a), a * b + a * a, "distributive");

    // But order matters, which is why CommutativeRing is not implemented.
    assert_ne!(a * b, b * a);
    assert_eq!((a * b).row(0), &[2, 1]);
    assert_eq!((b * a).row(0), &[3, 4]);

    // A generic ring function accepts them; asking for commutativity would not
    // compile.
    fn cube<T: Ring>(value: &T) -> T {
        value.power(3)
    }
    assert_eq!(cube(&b), b, "a swap undone twice is a swap");
    assert_eq!(cube(&M2::one()), M2::one());

    assert_eq!(a.trace(), 5);
    assert_eq!(M3::identity().trace(), 3);
    assert_eq!(Matrix::<i64, 3, 3>::diagonal([2, 3, 4]).trace(), 9);
}

#[test]
fn the_determinant_is_zero_exactly_when_the_map_collapses() {
    // 2x2 by hand: ad - bc.
    let a: Matrix<f64, 2, 2> = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    assert!((a.determinant() - -2.0).abs() < 1e-9);

    // A repeated row collapses the space.
    let flat: Matrix<f64, 2, 2> = Matrix::from_rows([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(flat.determinant(), 0.0);
    assert!(!flat.is_invertible());
    assert_eq!(flat.inverse(), None);

    // Identity is one; a swap is minus one.
    assert!((Matrix::<f64, 3, 3>::identity().determinant() - 1.0).abs() < 1e-9);
    let swap: Matrix<f64, 2, 2> = Matrix::from_rows([[0.0, 1.0], [1.0, 0.0]]);
    assert!((swap.determinant() - -1.0).abs() < 1e-9);

    // Scaling a row scales the determinant.
    let scaled: Matrix<f64, 2, 2> = Matrix::from_rows([[3.0, 6.0], [3.0, 4.0]]);
    assert!((scaled.determinant() - 3.0 * a.determinant()).abs() < 1e-9);

    // And the determinant multiplies, which is the property that matters.
    let b: Matrix<f64, 2, 2> = Matrix::from_rows([[2.0, 0.0], [1.0, 3.0]]);
    assert!(((a * b).determinant() - a.determinant() * b.determinant()).abs() < 1e-9);
}

#[test]
fn an_inverse_undoes_the_matrix_from_both_sides() {
    let a: Matrix<f64, 3, 3> =
        Matrix::from_rows([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]);

    let inverse = a.inverse().expect("not singular");

    for product in [a * inverse, inverse * a] {
        for row in 0..3 {
            for column in 0..3 {
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!(
                    (product.get(row, column).unwrap() - expected).abs() < 1e-9,
                    "at {row},{column}"
                );
            }
        }
    }

    // Over an exact field it is exact, with no tolerance needed.
    type F = PrimeField<7>;
    let exact: Matrix<F, 2, 2> =
        Matrix::from_rows([[F::new(1), F::new(2)], [F::new(3), F::new(4)]]);
    let inverse = exact.inverse().expect("determinant is not zero mod 7");
    assert_eq!(exact * inverse, Matrix::<F, 2, 2>::identity());
    assert_eq!(inverse * exact, Matrix::<F, 2, 2>::identity());
}

#[test]
fn rank_counts_the_dimensions_that_survive() {
    // Full rank.
    let full: Matrix<f64, 3, 3> = Matrix::identity();
    assert_eq!(full.rank(), 3);

    // A repeated row loses one.
    let flat: Matrix<f64, 3, 3> =
        Matrix::from_rows([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]);
    assert_eq!(flat.rank(), 2);

    // Everything zero survives nothing.
    assert_eq!(Matrix::<f64, 3, 3>::zeroed().rank(), 0);

    // A rectangular matrix cannot exceed its smaller dimension, however it is
    // filled: three rows of four cannot raise dimension past three.
    let wide: Matrix<f64, 3, 4> = Matrix::from_rows([
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 6.0],
        [0.0, 0.0, 1.0, 7.0],
    ]);
    assert_eq!(wide.rank(), 3);

    let tall: Matrix<f64, 4, 3> = wide.transpose();
    assert_eq!(tall.rank(), 3, "transposing does not change the rank");
}

#[test]
fn a_square_matrix_transforms_a_vector_of_its_own_size() {
    // A rotation by a quarter turn, in exact integers.
    let turn: M2 = Matrix::from_rows([[0, -1], [1, 0]]);
    assert_eq!(turn * Vector2::new(1, 0), Vector2::new(0, 1));
    assert_eq!(turn * Vector2::new(0, 1), Vector2::new(-1, 0));

    let scale: M3 = Matrix::diagonal([2, 3, 4]);
    assert_eq!(scale * Vector3::new(1, 1, 1), Vector3::new(2, 3, 4));

    let identity: Matrix<i64, 4, 4> = Matrix::identity();
    assert_eq!(
        identity * Vector4::new(1, 2, 3, 4),
        Vector4::new(1, 2, 3, 4)
    );

    // Composing matrices composes the transformations.
    let twice = turn * turn;
    assert_eq!(twice * Vector2::new(1, 0), Vector2::new(-1, 0));
    assert_eq!(
        twice * Vector2::new(1, 0),
        turn * (turn * Vector2::new(1, 0))
    );
}

#[test]
fn a_rectangular_matrix_moves_between_dimensions() {
    // Up: three components become four, with a one in the last place.
    let raise: Matrix<i64, 4, 3> = Matrix::point_embedding();
    assert_eq!(
        raise * Vector3::new(2, 3, 4),
        Vector4::new(2, 3, 4, 0),
        "the last row is zero, so this is a direction"
    );

    // With a one in the corner instead, it becomes a point.
    let as_point: Matrix<i64, 4, 3> =
        Matrix::from_rows([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]);
    assert_eq!(as_point * Vector3::new(2, 3, 4), Vector4::new(2, 3, 4, 0));

    // Down: four components become three, dropping the last.
    let lower: Matrix<i64, 3, 4> = Matrix::drop_last();
    assert_eq!(lower * Vector4::new(2, 3, 4, 9), Vector3::new(2, 3, 4));

    // Round trip loses whatever the fourth component held, as it must.
    let there_and_back = lower * raise;
    assert_eq!(there_and_back.row(0), &[1, 0, 0]);
    assert_eq!(there_and_back, Matrix::<i64, 3, 3>::identity());

    // Two and three dimensions likewise.
    let up: Matrix<i64, 3, 2> = Matrix::from_rows([[1, 0], [0, 1], [0, 0]]);
    assert_eq!(up * Vector2::new(5, 6), Vector3::new(5, 6, 0));

    let down: Matrix<i64, 2, 3> = Matrix::from_rows([[1, 0, 0], [0, 1, 0]]);
    assert_eq!(down * Vector3::new(5, 6, 7), Vector2::new(5, 6));
}

#[test]
fn a_translation_needs_the_extra_dimension_to_be_linear() {
    // The reason homogeneous coordinates exist: a shift is not linear, but it
    // becomes a matrix once a point carries a one.
    let shift: Matrix<i64, 4, 4> =
        Matrix::from_rows([[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]);

    // A point, with one in the last place, moves.
    assert_eq!(
        shift * Vector4::new(1, 2, 3, 1),
        Vector4::new(11, 22, 33, 1)
    );

    // A direction, with zero there, does not — which is the whole point of the
    // distinction.
    assert_eq!(shift * Vector4::new(1, 2, 3, 0), Vector4::new(1, 2, 3, 0));
}

#[test]
fn the_linear_map_trait_hides_which_representation_is_held() {
    // A matrix applied through the trait, over arrays so that every size works.
    let scale: M3 = Matrix::diagonal([2, 3, 4]);
    assert_eq!(scale.apply([1, 1, 1]), [2, 3, 4]);

    let raise: Matrix<i64, 4, 3> = Matrix::point_embedding();
    assert_eq!(raise.apply([1, 2, 3]), [1, 2, 3, 0]);

    // Written once, taking whatever maps this space to that one.
    fn twice<M>(map: &M, input: M::Input) -> M::Output
    where
        M: LinearMap,
        M::Output: Clone + Into<M::Input>,
    {
        let once: M::Output = map.apply(input);

        map.apply(once.into())
    }

    assert_eq!(twice(&scale, [1, 1, 1]), [4, 9, 16]);
}

#[test]
fn matrices_over_any_ring_work_the_same_way() {
    // Integers, a prime field, and anything else that adds and multiplies.
    type F = PrimeField<5>;

    let a: Matrix<F, 2, 2> = Matrix::from_rows([[F::new(1), F::new(2)], [F::new(3), F::new(4)]]);

    // Arithmetic wraps with the field.
    let squared = a * a;
    assert_eq!(squared.get(0, 0), Some(&F::new(7 % 5)));

    assert!(Matrix::<F, 2, 2>::identity().is_one());
    assert_eq!(a.trace(), F::new(0), "1 + 4 = 5 = 0 mod 5");

    // And over an unsigned semiring, where subtraction is unavailable but
    // multiplication is not.
    let counts: Matrix<u32, 2, 2> = Matrix::from_rows([[1, 2], [3, 4]]);
    assert_eq!((counts * counts).get(0, 0), Some(&7));
    assert_eq!(counts.scale(2).row(0), &[2, 4]);
}

#[test]
fn matrices_print_a_row_per_line() {
    let a: Matrix<i64, 2, 3> = Matrix::from_rows([[1, 2, 3], [4, 5, 6]]);

    assert_eq!(a.to_string(), "[1, 2, 3]\n[4, 5, 6]");
    assert_eq!(M2::identity().to_string(), "[1, 0]\n[0, 1]");

    // Mapping every entry changes the component type as well as the values.
    let doubled: Matrix<i64, 2, 3> = a.map(|entry| entry * 2);
    assert_eq!(doubled.row(0), &[2, 4, 6]);

    let described: Matrix<String, 2, 3> = a.map(|entry| entry.to_string());
    assert_eq!(described.get(0, 0).map(String::as_str), Some("1"));
}
