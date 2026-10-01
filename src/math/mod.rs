//! General-purpose arithmetic: vectors, matrices, and the number types they hold.
//!
//! # Shapes
//!
//! [`linear`] holds the vectors, generic over their component type so the same
//! shapes carry voxel coordinates, directions and gradients. [`matrix`] holds
//! both square matrices and rectangular ones, the latter for moving between
//! dimensions. [`rotation`] wraps a quaternion in a type that is always
//! normalized.
//!
//! # Numbers instead of `f64`
//!
//! [`fixed`] is a fixed-point number, for arithmetic that has to come out
//! bit-identical on every platform and every build. [`rational`] is an exact
//! fraction of two integers. Those are the two alternatives where determinism or
//! exactness matters: a fixed-point value rounds anything to its step, a rational
//! holds a third exactly but can run out of room.
//!
//! There are two families of oversized numbers here, and the difference between
//! them is when the size is chosen.
//!
//! **Fixed at compile time.** [`wide_integer`] holds an integer of `N × 64` bits,
//! signed or unsigned, in an array — no allocation, `Copy`, a predictable cost, and
//! it overflows like a primitive does. [`wide_float`] builds a float on it with a
//! chosen precision. These are what to use when the size is known: a 256-bit
//! coordinate, a hash, anything in a hot loop.
//!
//! **Grown at run time.** [`big_integer`] holds an integer in a `Vec` that
//! lengthens as the value does, so [`BigInt`] arithmetic never overflows at all.
//! [`big_float`] builds a float on that whose addition, subtraction and
//! multiplication are **exact** — which makes it the one float here whose ring laws
//! hold exactly rather than up to a rounding. These are what to use when the size is
//! not known: a factorial, an accumulating product, a parsed number.
//!
//! The rule of thumb: if you are picking a width large enough that it surely cannot
//! overflow, you wanted the growable one.
//!
//! [`interval`] is a pair of bounds enclosing a value, which is how to carry an
//! error term through a calculation rather than hoping it stays small. It is
//! deliberately *not* a ring: see its own documentation for why.
//!
//! # Algebras
//!
//! [`hypercomplex`] holds the extensions of the reals: complex numbers for the
//! plane, dual numbers for exact derivatives, quaternions for rotation, and the
//! split forms where a squared unit is `+1` rather than `−1`. [`quadratic`] holds
//! the Gaussian and Eisenstein integers, two lattices in the plane that divide
//! with a remainder. [`modular`] holds arithmetic modulo a number and the prime
//! fields, [`polynomial`] holds the polynomial ring and the finite field
//! extensions built from it.
//!
//! # The traits
//!
//! [`traits`] names the shapes those types have in common — semiring, ring,
//! commutative ring, Euclidean ring, field — so that an algorithm asks for what
//! it needs and no more. It is worth reading first: it is the reason a new number
//! type gets Euclid's algorithm, exponentiation by squaring, matrices and
//! polynomials without writing any of them.
//!
//! Nothing here is seeded or random, and nothing here is specific to voxels.

pub mod wide_integer;
pub mod wide_float;
pub(crate) mod cordic;
pub mod big_float;
pub mod big_integer;
pub mod fixed;
pub mod hypercomplex;
pub mod interval;
pub mod linear;
pub mod matrix;
pub mod modular;
pub mod polynomial;
pub mod quadratic;
pub mod rational;
pub mod unit_interval;
pub mod rotation;
pub mod traits;

pub use wide_integer::{WideInt, WideUint};
pub use wide_float::WideFloat;
pub use big_float::BigFloat;
pub use big_integer::{BigInt, BigUint, ParseBigError};
pub use fixed::{Fixed, FixedPoint};
pub use hypercomplex::{Complex, Dual, Quaternion, SplitComplex, SplitQuaternion};
pub use interval::{Interval, IntervalSet};
pub use linear::{Vector2, Vector3, Vector4};
pub use matrix::Matrix;
pub use modular::{Modulo, PrimeField, is_prime};
pub use polynomial::{Extension, Polynomial};
pub use quadratic::{Eisenstein, Gaussian};
pub use rational::Ratio;
pub use unit_interval::{RatioOutOfRange, Unit};
pub use rotation::UnitQuaternion;
