//! Named types for the quantities the world is made of.
//!
//! Almost everything here wraps one number, exactly as [`Seed`] wraps a `u128`,
//! and for the same reason: a `u128` is a `u128`, so nothing stops a seed being
//! passed where a hash belongs. Giving each quantity its own type makes the
//! compiler refuse the mix-up instead of leaving it to be found in a generated
//! world months later.
//!
//! Three things they buy beyond the name.
//!
//! **The conversion lives with the type.** Turning an octree depth into a bit
//! shift is `tree_depth - depth`, written out at every call site and wrong in
//! several ways: it underflows above the root, overflows past the width of an
//! `i128`, and reads as a depth rather than a count of bits. It is now
//! [`TreeDepth::shift_for`](crate::spatial::TreeDepth::shift_for), written once.
//!
//! **Ranges hold by construction.** A [`Probability`] is in `[0, 1]` and a
//! [`NoiseValue`] in `[-1, 1]` because there is no way to build one that is
//! not. Nothing downstream has to check, and nothing has to document what it
//! assumes. [`Weights`] extends the same idea to a whole list: non-empty,
//! finite, non-negative and summing to a positive finite total, checked once at
//! construction rather than at every draw.
//!
//! **Positions know which space they are in.** A voxel coordinate, a node
//! address at some depth, and a fraction inside a cell are all "the position"
//! in conversation and all completely different numbers. [`VoxelPosition3`](crate::spatial::VoxelPosition3),
//! [`NodePosition3`](crate::spatial::NodePosition3) and [`LocalPosition3`](crate::spatial::LocalPosition3) cannot be swapped, and a
//! [`NodePosition3`](crate::spatial::NodePosition3) carries the depth it is an address in, so two addresses
//! from different levels can never be compared as though they were the same
//! kind of thing.
//!
//! Scalar wrappers carry no extra payload. Compound types deliberately keep
//! useful metadata: node positions retain a depth and weights cache a total.
//!
//! [`Seed`]: crate::random::seed::Seed

pub mod digest;
pub mod identifier;
pub mod rate;
pub mod scalar;
pub mod time;
pub mod weights;

// `Ratio` is a general rational number and lives in `math`, but it is re-exported
// here because the quantities that use it are units.
//
// Spatial types are **not** re-exported. They used to be, and it made `units`
// depend on `spatial` while `spatial` depended back on `units` — a cycle that
// bought nothing but a shorter import. Reach for `crate::spatial` directly.
pub use crate::math::rational::Ratio;
pub use digest::{Checksum, ContentHash};
pub use identifier::{AnyId, Id, IdKind, tag_for_name};
pub use time::{Seconds, Tick, TickDuration, TickRate};
pub use rate::Rate;
pub use scalar::{NoiseValue, Probability, UniformNoise, UniformProbability, UnitValue};
pub use crate::math::{RatioOutOfRange, Unit};
pub use crate::random::unit::UniformUnit;
pub use weights::Weights;
