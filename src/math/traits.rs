//! What a number type can do, named so that algorithms can ask for it.
//!
//! The shapes here are the algebraic ones: a [`Ring`] has addition and
//! multiplication, a [`Field`] can divide, a [`EuclideanRing`] has division with
//! a remainder and therefore a greatest common divisor. Naming them means a
//! polynomial or a matrix can be written once over any of them, rather than once
//! per element type.
//!
//! # Why functions rather than associated constants
//!
//! `T::ZERO` as a constant reads better and works in a `const` item, and that is
//! what these were at first. It does not survive contact with a ring that
//! allocates: [`Polynomial`](crate::math::Polynomial) holds its coefficients in a
//! `Vec`, and `Vec` cannot be built with an element in a constant, so its one
//! would have had to be the empty vector — which is its zero. A silently wrong
//! identity is worse than a function call.
//!
//! The concrete types keep their constants — [`Fixed::ZERO`](crate::math::Fixed),
//! [`Ratio::ZERO`](crate::math::Ratio) — so a `const` context still works
//! wherever the type is known rather than generic.
//!
//! # What is deliberately not a ring
//!
//! Unsigned integers. A ring needs an additive inverse for every element, and
//! `u32` has none: `-1u32` is not a thing. `Z/nZ` over unsigned storage *is* a
//! ring, and that is the type's own doing — negation there is `n - x` — not
//! something the storage provides.
//!
//! Floating point is admitted with a caveat that belongs in the open: `f64`
//! breaks the ring laws. Addition is not associative, and `NaN != NaN` breaks
//! reflexivity of equality. It is implemented because refusing would make the
//! traits unusable for geometry, not because the laws hold.

use crate::math::{Fixed, Ratio};
use std::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// Identities
// ---------------------------------------------------------------------------

/// A type with an additive identity.
pub trait Zero: Sized {
    /// The value that changes nothing when added.
    fn zero() -> Self;

    /// Whether this *is* that value.
    ///
    /// Separate from comparing against [`Zero::zero`] because a type may have
    /// several representations of it — a rational with any denominator, a float
    /// with either sign.
    fn is_zero(&self) -> bool;
}

/// A type with a multiplicative identity.
pub trait One: Sized {
    /// The value that changes nothing when multiplied.
    fn one() -> Self;

    /// Whether this *is* that value.
    fn is_one(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Rings and fields
// ---------------------------------------------------------------------------

/// Addition and multiplication with both identities, but no promise that
/// anything can be subtracted.
///
/// The honest home for the types that add and multiply but cannot negate:
/// unsigned integers, and [`Ratio`], which is a non-negative fraction. An
/// algorithm that only ever accumulates — summing, evaluating with positive
/// coefficients, weighting — wants this rather than [`Ring`], and gains those
/// types by asking for less.
///
/// # Laws
///
/// Addition is associative and commutative with identity [`Zero::zero`];
/// multiplication is associative with identity [`One::one`] and distributes over
/// addition. Multiplication is **not** assumed commutative.
pub trait Semiring:
    Zero + One + Clone + PartialEq + Add<Output = Self> + Mul<Output = Self>
{
    /// This value multiplied by itself `power` times, by repeated squaring.
    ///
    /// A provided method rather than a requirement: every ring can do it, and
    /// the halving means a power of a thousand costs ten multiplications rather
    /// than a thousand.
    fn power(&self, power: u32) -> Self {
        if power == 0 {
            return Self::one();
        }

        let mut result: Self = Self::one();
        let mut base: Self = self.clone();
        let mut remaining: u32 = power;

        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result * base.clone();
            }

            remaining >>= 1;

            // Squaring after the last bit has been consumed is wasted work, and
            // worse than wasted for a type whose multiplication refuses to
            // overflow: `(-2).power(150)` fits in 256 bits, but the squaring
            // beyond the top bit of the exponent reaches `2²⁵⁶` and panics on a
            // value that is never used.
            if remaining == 0 {
                break;
            }

            base = base.clone() * base;
        }

        result
    }

    /// This value added to itself `count` times.
    ///
    /// Doubling rather than repeated addition, for the same reason
    /// [`Semiring::power`] squares.
    fn multiple(&self, count: u32) -> Self {
        if count == 0 {
            return Self::zero();
        }

        let mut result: Self = Self::zero();
        let mut base: Self = self.clone();
        let mut remaining: u32 = count;

        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result + base.clone();
            }

            remaining >>= 1;

            // As in [`Semiring::power`]: the last doubling is never read, and can
            // overflow a value the answer itself does not.
            if remaining == 0 {
                break;
            }

            base = base.clone() + base;
        }

        result
    }
}

/// A semiring where everything can also be subtracted.
///
/// The bound to ask for when an algorithm needs arithmetic but not division:
/// evaluating a polynomial, multiplying a matrix, taking a difference.
///
/// # Laws
///
/// Those of [`Semiring`], and additionally an additive inverse for every
/// element. That is exactly what unsigned integers and [`Ratio`] lack, and why
/// they stop at [`Semiring`].
pub trait Ring: Semiring + Sub<Output = Self> + Neg<Output = Self> {}

/// A ring whose multiplication does not depend on order.
///
/// A marker: there is nothing to implement, only a promise that `a * b == b * a`.
/// Integers, rationals and the quadratic integer rings qualify; quaternions do
/// not.
pub trait CommutativeRing: Ring {}

/// A commutative ring where division leaves a remainder smaller than the
/// divisor, which is what makes Euclid's algorithm terminate.
///
/// The integers are the first example; so are
/// [`Gaussian`](crate::math::Gaussian) and
/// [`Eisenstein`](crate::math::Eisenstein) integers, which is the point of
/// naming the shape — the same greatest-common-divisor code serves all three.
pub trait EuclideanRing: CommutativeRing {
    /// How large a value counts as, for comparing a remainder with a divisor.
    ///
    /// The integers use the absolute value; the quadratic rings use their
    /// multiplicative norm. Only the ordering matters, never the arithmetic.
    type Size: Ord;

    /// This value's size.
    fn euclidean_size(&self) -> Self::Size;

    /// The quotient and remainder, or `None` when dividing by zero.
    ///
    /// The remainder must be strictly smaller than the divisor by
    /// [`EuclideanRing::euclidean_size`], which is what
    /// [`EuclideanRing::gcd`] relies on to finish.
    fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)>;

    /// The greatest common divisor, by Euclid's algorithm.
    ///
    /// Provided, because the algorithm is the same in every Euclidean ring once
    /// [`EuclideanRing::div_rem`] exists — which is the whole reason this trait
    /// is worth having. The result is unique only up to multiplication by a
    /// unit, so `gcd(2, 4)` may be `2` or `-2`; use
    /// [`EuclideanRing::gcd_normalised`] where a canonical answer matters.
    fn gcd(&self, other: &Self) -> Self {
        let mut left: Self = self.clone();
        let mut right: Self = other.clone();

        while !right.is_zero() {
            let Some((_, remainder)) = left.div_rem(&right) else {
                break;
            };

            left = right;
            right = remainder;
        }

        left
    }

    /// The greatest common divisor in whichever associate the type calls
    /// canonical.
    ///
    /// Defaults to [`EuclideanRing::gcd`]; a type with a preferred associate —
    /// a non-negative integer, a monic polynomial — overrides it.
    fn gcd_normalised(&self, other: &Self) -> Self {
        self.gcd(other)
    }

    /// Whether this value divides `multiple` exactly.
    fn divides(&self, multiple: &Self) -> bool {
        matches!(multiple.div_rem(self), Some((_, remainder)) if remainder.is_zero())
    }
}

/// A commutative ring where every value but zero has a multiplicative inverse.
pub trait Field: CommutativeRing {
    /// The value this multiplies with to give [`One::one`], or `None` for zero.
    fn inverse(&self) -> Option<Self>;

    /// This divided by `divisor`, or `None` when the divisor has no inverse.
    fn divide(&self, divisor: &Self) -> Option<Self> {
        Some(self.clone() * divisor.inverse()?)
    }
}

// ---------------------------------------------------------------------------
// Conjugation and norms
// ---------------------------------------------------------------------------

/// A type with a conjugate: the involution that flips its non-real parts.
///
/// `conjugate(conjugate(x)) == x`, and for the complex-like types
/// `x * conjugate(x)` is the [`AlgebraicNorm`] lifted back into the type.
pub trait Conjugate {
    /// This value with its non-real parts negated.
    fn conjugate(&self) -> Self;
}

/// A type with a multiplicative norm: a size that multiplies.
///
/// `algebraic_norm(a * b) == algebraic_norm(a) * algebraic_norm(b)`. That is what
/// makes it useful rather than decorative: it turns a question about the ring
/// into a question about its base, so a Gaussian integer is prime only if its
/// norm is, and a unit is exactly an element of norm one.
///
/// # Why not simply `norm`
///
/// Because this crate already uses `norm` for the *magnitude* — a vector's
/// length, a quaternion's `sqrt(w² + x² + y² + z²)` — and the two are different
/// quantities. The algebraic norm is the magnitude *squared*, which is the one
/// that stays inside the ring: `Complex<i64>` has an integer algebraic norm and
/// an irrational magnitude. Giving them one name would have made
/// `value.norm()` mean one thing through an inherent method and another through
/// a trait, decided by which was in scope.
pub trait AlgebraicNorm {
    /// What the norm is measured in, usually the component type.
    type Output;

    /// This value's multiplicative norm.
    fn algebraic_norm(&self) -> Self::Output;
}

/// Something that transforms a value linearly: a matrix, a rotation, a
/// projection.
///
/// Named so that code can take "whatever maps this space to that one" rather
/// than a particular representation. A three-by-three
/// [`Matrix`](crate::math::Matrix) and a
/// [`UnitQuaternion`](crate::math::UnitQuaternion) both rotate a three-vector,
/// and neither should have to know which the caller holds.
///
/// # Laws
///
/// `apply(a + b) == apply(a) + apply(b)`, and `apply(scaled)` is the scaled
/// `apply`. A translation is therefore **not** one of these, which is exactly why
/// translations are done by raising a point into one more dimension first: the
/// extra component is what makes a shift linear.
pub trait LinearMap {
    /// What it takes.
    type Input;

    /// What it gives back, which need not be the same space.
    type Output;

    /// The value transformed.
    fn apply(&self, input: Self::Input) -> Self::Output;
}

// ---------------------------------------------------------------------------
// Rounded division, which the quadratic rings need
// ---------------------------------------------------------------------------

/// Division that rounds to the nearest value rather than towards zero.
///
/// Dividing in [`Gaussian`](crate::math::Gaussian) or
/// [`Eisenstein`](crate::math::Eisenstein) means taking the nearest lattice
/// point to an exact rational answer, and truncating instead would leave a
/// remainder too large for Euclid's algorithm to make progress on.
pub trait RoundedDiv: Sized {
    /// This divided by `divisor`, rounded to nearest, or `None` when the divisor
    /// is zero or the rounding would overflow.
    ///
    /// A value exactly halfway may round either way: the quadratic rings need
    /// only that the remainder shrink, not which way a tie goes.
    fn div_rounded(&self, divisor: &Self) -> Option<Self>;
}

// ---------------------------------------------------------------------------
// The types this crate already has
// ---------------------------------------------------------------------------

/// Signed integers: rings, Euclidean, and roundable.
macro_rules! implement_integer {
    ($($type:ty),* $(,)?) => {
        $(
            impl Zero for $type {
                fn zero() -> Self {
                    0
                }

                fn is_zero(&self) -> bool {
                    *self == 0
                }
            }

            impl One for $type {
                fn one() -> Self {
                    1
                }

                fn is_one(&self) -> bool {
                    *self == 1
                }
            }

            impl Semiring for $type {}
            impl Ring for $type {}
            impl CommutativeRing for $type {}

            impl EuclideanRing for $type {
                /// The absolute value, widened so that the most negative value
                /// has one.
                type Size = u128;

                fn euclidean_size(&self) -> u128 {
                    self.unsigned_abs() as u128
                }

                fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
                    Some((
                        self.checked_div(*divisor)?,
                        self.checked_rem(*divisor)?,
                    ))
                }

                /// Non-negative, which is the usual convention for integers.
                fn gcd_normalised(&self, other: &Self) -> Self {
                    self.gcd(other).abs()
                }
            }

            impl RoundedDiv for $type {
                fn div_rounded(&self, divisor: &Self) -> Option<Self> {
                    let quotient: Self = self.checked_div(*divisor)?;
                    let remainder: Self = self.checked_rem(*divisor)?;

                    if remainder == 0 {
                        return Some(quotient);
                    }

                    // Compared as `|r| >= (|d| + 1) / 2` rather than
                    // `2 * |r| >= |d|`, which would overflow near the limits.
                    let size: u128 = remainder.unsigned_abs() as u128;
                    let half: u128 = (divisor.unsigned_abs() as u128).div_ceil(2);

                    if size < half {
                        return Some(quotient);
                    }

                    // Away from zero, in whichever direction the exact answer
                    // lies.
                    let step: Self = if (*self < 0) == (*divisor < 0) { 1 } else { -1 };

                    quotient.checked_add(step)
                }
            }
        )*
    };
}

implement_integer!(i8, i16, i32, i64, i128, isize);

/// Unsigned integers: semirings, since `-1u32` does not exist.
macro_rules! implement_unsigned {
    ($($type:ty),* $(,)?) => {
        $(
            impl Zero for $type {
                fn zero() -> Self {
                    0
                }

                fn is_zero(&self) -> bool {
                    *self == 0
                }
            }

            impl One for $type {
                fn one() -> Self {
                    1
                }

                fn is_one(&self) -> bool {
                    *self == 1
                }
            }

            impl Semiring for $type {}
        )*
    };
}

implement_unsigned!(u8, u16, u32, u64, u128, usize);

impl Zero for Fixed {
    fn zero() -> Self {
        Fixed::ZERO
    }

    fn is_zero(&self) -> bool {
        *self == Fixed::ZERO
    }
}

impl One for Fixed {
    fn one() -> Self {
        Fixed::ONE
    }

    fn is_one(&self) -> bool {
        *self == Fixed::ONE
    }
}

impl Semiring for Fixed {}
impl Ring for Fixed {}
impl CommutativeRing for Fixed {}

/// Fixed point divides exactly in the sense a field needs, up to the rounding
/// its last fractional bit forces.
impl Field for Fixed {
    fn inverse(&self) -> Option<Self> {
        Fixed::ONE.checked_div(*self)
    }
}

impl Zero for Ratio {
    fn zero() -> Self {
        Ratio::ZERO
    }

    fn is_zero(&self) -> bool {
        self.numerator() == 0
    }
}

impl One for Ratio {
    fn one() -> Self {
        Ratio::ONE
    }

    fn is_one(&self) -> bool {
        *self == Ratio::ONE
    }
}

/// A non-negative fraction adds and multiplies, but `-1/2` is not a [`Ratio`],
/// so it stops here: no additive inverse means no [`Ring`], and therefore no
/// [`Field`] either, however divisible it is.
///
/// [`Ratio::reciprocal`] is still there for the multiplicative inverse; it is
/// only the trait that cannot be claimed.
impl Semiring for Ratio {}

/// Floats, with the caveat in the module documentation: the laws do not hold,
/// but geometry needs them.
macro_rules! implement_float {
    ($($type:ty),* $(,)?) => {
        $(
            impl Zero for $type {
                fn zero() -> Self {
                    0.0
                }

                fn is_zero(&self) -> bool {
                    *self == 0.0
                }
            }

            impl One for $type {
                fn one() -> Self {
                    1.0
                }

                fn is_one(&self) -> bool {
                    *self == 1.0
                }
            }

            impl Semiring for $type {}
            impl Ring for $type {}
            impl CommutativeRing for $type {}

            impl Field for $type {
                fn inverse(&self) -> Option<Self> {
                    if *self == 0.0 {
                        return None;
                    }

                    Some(1.0 / *self)
                }
            }

            impl RoundedDiv for $type {
                fn div_rounded(&self, divisor: &Self) -> Option<Self> {
                    if *divisor == 0.0 {
                        return None;
                    }

                    Some((*self / *divisor).round())
                }
            }
        )*
    };
}

implement_float!(f32, f64);
