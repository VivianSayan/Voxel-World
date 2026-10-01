//! Matrices whose shape is part of their type.
//!
//! [`Matrix<T, ROWS, COLS>`](Matrix) carries its dimensions as const generics, so
//! a product that does not line up is a compile error rather than a run-time
//! check:
//!
//! ```compile_fail
//! use voxel_world::math::Matrix;
//!
//! let wide: Matrix<i64, 2, 3> = Matrix::filled(1);
//! let also: Matrix<i64, 2, 3> = Matrix::filled(1);
//!
//! // 2x3 times 2x3 does not exist, and will not compile.
//! let _ = wide * also;
//! ```
//!
//! # Rectangular matrices change dimension
//!
//! A square matrix maps a space to itself; a rectangular one moves between
//! spaces, which is what makes raising and lowering dimension a multiplication
//! rather than a special case. A `Matrix<T, 4, 3>` takes a three-vector to a
//! four-vector, and a `Matrix<T, 3, 4>` brings it back — with whatever the fourth
//! component carried either kept or dropped, according to what is in the matrix.
//!
//! # Row-major, and why the vectors stay as they are
//!
//! Rows are stored together, so `rows[i][j]` is row `i`, column `j` — the order
//! the type is written in and read in. Column-major storage matters when handing
//! a matrix to a graphics API, which is not what this is for.
//!
//! [`Vector2`], [`Vector3`] and [`Vector4`] are untouched. They already convert
//! to and from arrays, and arrays are what the general multiplication takes, so
//! nothing had to be rebuilt to make them work together.

use crate::math::traits::{Field, LinearMap, One, Ring, Semiring, Zero};
use crate::math::{Vector2, Vector3, Vector4};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// A matrix of `ROWS` by `COLS` values.
///
/// # Example
///
/// ```
/// use voxel_world::math::Matrix;
///
/// let a: Matrix<i64, 2, 3> = Matrix::from_rows([[1, 2, 3], [4, 5, 6]]);
/// let b: Matrix<i64, 3, 2> = Matrix::from_rows([[7, 8], [9, 10], [11, 12]]);
///
/// // The inner dimensions cancel, leaving 2 by 2.
/// let product: Matrix<i64, 2, 2> = a * b;
///
/// assert_eq!(product.row(0), &[58, 64]);
/// assert_eq!(product.row(1), &[139, 154]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Matrix<T, const ROWS: usize, const COLS: usize> {
    rows: [[T; COLS]; ROWS],
}

impl<T, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// From its rows, each one left to right.
    pub const fn from_rows(rows: [[T; COLS]; ROWS]) -> Self {
        Self { rows }
    }

    /// One row.
    pub fn row(&self, row: usize) -> &[T; COLS] {
        &self.rows[row]
    }

    /// Every row, top to bottom.
    pub fn rows(&self) -> &[[T; COLS]; ROWS] {
        &self.rows
    }

    /// One value, or `None` outside the matrix.
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.rows.get(row)?.get(column)
    }

    /// One value, to change.
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        self.rows.get_mut(row)?.get_mut(column)
    }

    /// How many rows and columns, which the type already knows.
    pub const fn shape() -> (usize, usize) {
        (ROWS, COLS)
    }

    /// Whether the matrix is square, and so maps a space to itself.
    pub const fn is_square() -> bool {
        ROWS == COLS
    }
}

impl<T: Clone, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Every entry the same.
    pub fn filled(value: T) -> Self {
        Self {
            rows: std::array::from_fn(|_| std::array::from_fn(|_| value.clone())),
        }
    }

    /// From a function of the position, which is how the named matrices below are
    /// all built.
    pub fn from_fn(mut entry: impl FnMut(usize, usize) -> T) -> Self {
        Self {
            rows: std::array::from_fn(|row| std::array::from_fn(|column| entry(row, column))),
        }
    }

    /// From its columns rather than its rows.
    pub fn from_columns(columns: [[T; ROWS]; COLS]) -> Self {
        Self::from_fn(|row, column| columns[column][row].clone())
    }

    /// One column, gathered — unlike a row, which is stored together.
    pub fn column(&self, column: usize) -> [T; ROWS] {
        std::array::from_fn(|row| self.rows[row][column].clone())
    }

    /// The matrix with rows and columns exchanged, which turns a map between two
    /// spaces into one going the other way.
    pub fn transpose(&self) -> Matrix<T, COLS, ROWS> {
        Matrix::from_fn(|row, column| self.rows[column][row].clone())
    }

    /// Every entry put through a function.
    pub fn map<U: Clone>(&self, mut change: impl FnMut(&T) -> U) -> Matrix<U, ROWS, COLS> {
        Matrix::from_fn(|row, column| change(&self.rows[row][column]))
    }
}

impl<T: Semiring, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Every entry zero.
    pub fn zeroed() -> Self {
        Self::filled(T::zero())
    }

    /// Every entry multiplied by one value.
    ///
    /// Scaling, which is not the same as multiplying by a matrix — though for a
    /// square matrix it is the same as multiplying by that value's diagonal.
    pub fn scale(&self, factor: T) -> Self {
        self.map(|entry| entry.clone() * factor.clone())
    }
}

impl<T: Semiring, const N: usize> Matrix<T, N, N> {
    /// The identity: ones down the diagonal, zeros elsewhere.
    pub fn identity() -> Self {
        Self::from_fn(|row, column| if row == column { T::one() } else { T::zero() })
    }

    /// A diagonal matrix from the values along it.
    pub fn diagonal(values: [T; N]) -> Self {
        Self::from_fn(|row, column| {
            if row == column {
                values[row].clone()
            } else {
                T::zero()
            }
        })
    }

    /// The sum down the diagonal.
    ///
    /// Unchanged by a change of basis, which makes it one of the few numbers that
    /// describes the map rather than the matrix written for it.
    pub fn trace(&self) -> T {
        (0..N).fold(T::zero(), |total, index| {
            total + self.rows[index][index].clone()
        })
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

impl<T: Semiring, const ROWS: usize, const COLS: usize> Add for Matrix<T, ROWS, COLS> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::from_fn(|row, column| {
            self.rows[row][column].clone() + other.rows[row][column].clone()
        })
    }
}

impl<T: Ring, const ROWS: usize, const COLS: usize> Sub for Matrix<T, ROWS, COLS> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::from_fn(|row, column| {
            self.rows[row][column].clone() - other.rows[row][column].clone()
        })
    }
}

impl<T: Ring, const ROWS: usize, const COLS: usize> Neg for Matrix<T, ROWS, COLS> {
    type Output = Self;

    fn neg(self) -> Self {
        self.map(|entry| -entry.clone())
    }
}

impl<T: Semiring, const ROWS: usize, const COLS: usize> AddAssign for Matrix<T, ROWS, COLS> {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<T: Ring, const ROWS: usize, const COLS: usize> SubAssign for Matrix<T, ROWS, COLS> {
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

/// Row by column, with the inner dimensions cancelling.
///
/// The shapes are checked while compiling: `ROWS × COLS` times `COLS × OTHER`
/// gives `ROWS × OTHER`, and nothing else type-checks. That is the whole reason
/// the dimensions are in the type.
impl<T: Semiring, const ROWS: usize, const COLS: usize, const OTHER: usize>
    Mul<Matrix<T, COLS, OTHER>> for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T, ROWS, OTHER>;

    fn mul(self, other: Matrix<T, COLS, OTHER>) -> Matrix<T, ROWS, OTHER> {
        Matrix::from_fn(|row, column| {
            (0..COLS).fold(T::zero(), |total, index| {
                total + self.rows[row][index].clone() * other.rows[index][column].clone()
            })
        })
    }
}

/// A matrix applied to a column of values, which is what a matrix is for.
///
/// Arrays rather than a vector type, so that every size works from one
/// implementation; the vector types convert in and out through their own
/// `to_array`, and have their own implementations below for the sizes that have
/// names.
impl<T: Semiring, const ROWS: usize, const COLS: usize> Mul<[T; COLS]> for Matrix<T, ROWS, COLS> {
    type Output = [T; ROWS];

    fn mul(self, column: [T; COLS]) -> [T; ROWS] {
        std::array::from_fn(|row| {
            (0..COLS).fold(T::zero(), |total, index| {
                total + self.rows[row][index].clone() * column[index].clone()
            })
        })
    }
}

impl<T: Semiring, const N: usize> Zero for Matrix<T, N, N> {
    fn zero() -> Self {
        Self::zeroed()
    }

    fn is_zero(&self) -> bool {
        self.rows
            .iter()
            .all(|row| row.iter().all(|entry| entry.is_zero()))
    }
}

impl<T: Semiring, const N: usize> One for Matrix<T, N, N> {
    /// The identity, which is what does nothing under multiplication.
    fn one() -> Self {
        Self::identity()
    }

    fn is_one(&self) -> bool {
        (0..N).all(|row| {
            (0..N).all(|column| {
                if row == column {
                    self.rows[row][column].is_one()
                } else {
                    self.rows[row][column].is_zero()
                }
            })
        })
    }
}

impl<T: Semiring, const N: usize> Semiring for Matrix<T, N, N> {}

/// Square matrices are a ring — and a famously non-commutative one, so
/// [`CommutativeRing`](crate::math::traits::CommutativeRing) is deliberately not
/// implemented.
///
/// Only the square ones: a rectangular matrix cannot be multiplied by itself, so
/// there is no ring to be had.
impl<T: Ring, const N: usize> Ring for Matrix<T, N, N> {}

// ---------------------------------------------------------------------------
// Elimination: determinant, inverse, rank
// ---------------------------------------------------------------------------

impl<T: Field, const N: usize> Matrix<T, N, N> {
    /// The determinant, by elimination.
    ///
    /// Zero exactly when the matrix collapses its space — when some direction is
    /// mapped to nothing — which is the same as having no inverse.
    ///
    /// Cubic in the size, where expanding by cofactors would be factorial. That
    /// gap is the whole reason to eliminate rather than expand: at size ten,
    /// cofactors cost three million times as much.
    pub fn determinant(&self) -> T {
        let (upper, swaps, singular) = self.eliminated();

        if singular {
            return T::zero();
        }

        let product: T = (0..N).fold(T::one(), |total, index| {
            total * upper.rows[index][index].clone()
        });

        if swaps % 2 == 0 { product } else { -product }
    }

    /// The inverse, or `None` when the determinant is zero.
    ///
    /// By Gauss–Jordan elimination on this matrix beside the identity: whatever
    /// row operations reduce one to the identity turn the other into the inverse.
    pub fn inverse(&self) -> Option<Self> {
        let mut working: Self = self.clone();
        let mut result: Self = Self::identity();

        for pivot in 0..N {
            // Any non-zero entry will serve as a pivot. For a floating-point
            // matrix the *largest* would be more stable, which needs an ordering
            // this signature does not ask for; an exact field does not care.
            let found: usize = (pivot..N).find(|row| !working.rows[*row][pivot].is_zero())?;

            working.rows.swap(pivot, found);
            result.rows.swap(pivot, found);

            let scale: T = working.rows[pivot][pivot].clone().inverse()?;

            for column in 0..N {
                working.rows[pivot][column] = working.rows[pivot][column].clone() * scale.clone();
                result.rows[pivot][column] = result.rows[pivot][column].clone() * scale.clone();
            }

            for row in 0..N {
                if row == pivot {
                    continue;
                }

                let factor: T = working.rows[row][pivot].clone();

                if factor.is_zero() {
                    continue;
                }

                for column in 0..N {
                    working.rows[row][column] = working.rows[row][column].clone()
                        - factor.clone() * working.rows[pivot][column].clone();
                    result.rows[row][column] = result.rows[row][column].clone()
                        - factor.clone() * result.rows[pivot][column].clone();
                }
            }
        }

        Some(result)
    }

    /// Whether the matrix has an inverse.
    pub fn is_invertible(&self) -> bool {
        !self.determinant().is_zero()
    }

    /// Row-reduces to upper triangular, reporting how many rows were swapped and
    /// whether a pivot was missing.
    ///
    /// A swap flips the determinant's sign, which is why they are counted rather
    /// than only performed.
    fn eliminated(&self) -> (Self, usize, bool) {
        let mut working: Self = self.clone();
        let mut swaps: usize = 0;

        for pivot in 0..N {
            let Some(found) = (pivot..N).find(|row| !working.rows[*row][pivot].is_zero()) else {
                // A column with nothing to pivot on: the matrix is singular, and
                // no further reduction changes that.
                return (working, swaps, true);
            };

            if found != pivot {
                working.rows.swap(pivot, found);
                swaps += 1;
            }

            let Some(scale) = working.rows[pivot][pivot].clone().inverse() else {
                return (working, swaps, true);
            };

            for row in pivot + 1..N {
                let factor: T = working.rows[row][pivot].clone() * scale.clone();

                if factor.is_zero() {
                    continue;
                }

                for column in pivot..N {
                    working.rows[row][column] = working.rows[row][column].clone()
                        - factor.clone() * working.rows[pivot][column].clone();
                }
            }
        }

        (working, swaps, false)
    }
}

impl<T: Field, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// How many dimensions survive the map.
    ///
    /// The number of independent rows, found by elimination. A square matrix is
    /// invertible exactly when this is its size; a rectangular one cannot raise
    /// dimension beyond it, however many rows it has.
    pub fn rank(&self) -> usize {
        let mut working: Self = self.clone();
        let mut rank: usize = 0;

        for column in 0..COLS {
            if rank == ROWS {
                break;
            }

            let Some(found) = (rank..ROWS).find(|row| !working.rows[*row][column].is_zero()) else {
                continue;
            };

            working.rows.swap(rank, found);

            let Some(scale) = working.rows[rank][column].clone().inverse() else {
                continue;
            };

            for index in 0..COLS {
                working.rows[rank][index] = working.rows[rank][index].clone() * scale.clone();
            }

            for row in 0..ROWS {
                if row == rank {
                    continue;
                }

                let factor: T = working.rows[row][column].clone();

                if factor.is_zero() {
                    continue;
                }

                for index in 0..COLS {
                    working.rows[row][index] = working.rows[row][index].clone()
                        - factor.clone() * working.rows[rank][index].clone();
                }
            }

            rank += 1;
        }

        rank
    }
}

// ---------------------------------------------------------------------------
// Acting on vectors, including between dimensions
// ---------------------------------------------------------------------------

/// A matrix acting on a column of values.
impl<T: Semiring, const ROWS: usize, const COLS: usize> LinearMap for Matrix<T, ROWS, COLS> {
    type Input = [T; COLS];
    type Output = [T; ROWS];

    fn apply(&self, input: [T; COLS]) -> [T; ROWS] {
        self.clone() * input
    }
}

impl<T: Semiring> Mul<Vector2<T>> for Matrix<T, 2, 2> {
    type Output = Vector2<T>;

    fn mul(self, vector: Vector2<T>) -> Vector2<T> {
        let [x, y] = self * vector.to_array();

        Vector2::new(x, y)
    }
}

impl<T: Semiring> Mul<Vector3<T>> for Matrix<T, 3, 3> {
    type Output = Vector3<T>;

    fn mul(self, vector: Vector3<T>) -> Vector3<T> {
        let [x, y, z] = self * vector.to_array();

        Vector3::new(x, y, z)
    }
}

impl<T: Semiring> Mul<Vector4<T>> for Matrix<T, 4, 4> {
    type Output = Vector4<T>;

    fn mul(self, vector: Vector4<T>) -> Vector4<T> {
        let [x, y, z, w] = self * vector.to_array();

        Vector4::new(x, y, z, w)
    }
}

/// Raises a two-vector into three dimensions.
impl<T: Semiring> Mul<Vector2<T>> for Matrix<T, 3, 2> {
    type Output = Vector3<T>;

    fn mul(self, vector: Vector2<T>) -> Vector3<T> {
        let [x, y, z] = self * vector.to_array();

        Vector3::new(x, y, z)
    }
}

/// Lowers a three-vector into two dimensions.
impl<T: Semiring> Mul<Vector3<T>> for Matrix<T, 2, 3> {
    type Output = Vector2<T>;

    fn mul(self, vector: Vector3<T>) -> Vector2<T> {
        let [x, y] = self * vector.to_array();

        Vector2::new(x, y)
    }
}

/// Raises a three-vector into four dimensions, which is how a point becomes
/// homogeneous.
impl<T: Semiring> Mul<Vector3<T>> for Matrix<T, 4, 3> {
    type Output = Vector4<T>;

    fn mul(self, vector: Vector3<T>) -> Vector4<T> {
        let [x, y, z, w] = self * vector.to_array();

        Vector4::new(x, y, z, w)
    }
}

/// Lowers a four-vector into three dimensions, which is how one comes back.
impl<T: Semiring> Mul<Vector4<T>> for Matrix<T, 3, 4> {
    type Output = Vector3<T>;

    fn mul(self, vector: Vector4<T>) -> Vector3<T> {
        let [x, y, z] = self * vector.to_array();

        Vector3::new(x, y, z)
    }
}

impl<T: Semiring> Matrix<T, 4, 3> {
    /// The matrix that raises a three-vector to a four-vector with a one in the
    /// last place.
    ///
    /// The homogeneous embedding: a point rather than a direction, so that a
    /// translation in a four-by-four matrix reaches it.
    pub fn point_embedding() -> Self {
        Self::from_rows([
            [T::one(), T::zero(), T::zero()],
            [T::zero(), T::one(), T::zero()],
            [T::zero(), T::zero(), T::one()],
            [T::zero(), T::zero(), T::zero()],
        ])
    }
}

impl<T: Semiring> Matrix<T, 3, 4> {
    /// The matrix that drops a four-vector's last component.
    ///
    /// The projection back, which discards whatever the fourth component held
    /// rather than dividing by it — a perspective divide is not linear and cannot
    /// be a matrix.
    pub fn drop_last() -> Self {
        Self::from_rows([
            [T::one(), T::zero(), T::zero(), T::zero()],
            [T::zero(), T::one(), T::zero(), T::zero()],
            [T::zero(), T::zero(), T::one(), T::zero()],
        ])
    }
}

impl<T: fmt::Display, const ROWS: usize, const COLS: usize> fmt::Display for Matrix<T, ROWS, COLS> {
    /// One row per line, bracketed.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, row) in self.rows.iter().enumerate() {
            if index > 0 {
                formatter.write_str("\n")?;
            }

            formatter.write_str("[")?;

            for (position, entry) in row.iter().enumerate() {
                if position > 0 {
                    formatter.write_str(", ")?;
                }

                write!(formatter, "{entry}")?;
            }

            formatter.write_str("]")?;
        }

        Ok(())
    }
}
