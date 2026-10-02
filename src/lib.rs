//! Foundations for deterministic, voxel-native worlds.
//!
//! Domain quantities cannot be silently interchanged:
//! ```compile_fail
//! use voxel_world::{random::seed::Seed, units::Probability};
//! Seed::from_raw(1).chance(0.5); // Requires Probability.
//! ```
//! ```compile_fail
//! use voxel_world::time::{Tick, TickDuration};
//! Tick::new(10).advanced_by(Tick::new(2)); // A point is not a duration.
//! ```
//! ```compile_fail
//! use voxel_world::math::{Quaternion, Vector3};
//! Quaternion::new(2.0, 0.0, 0.0, 0.0).rotate(Vector3::X); // Requires UnitQuaternion.
//! ```
//! ```compile_fail
//! use voxel_world::{structures::collections::FuzzySet, structures::traits::MeasuredMut};
//! FuzzySet::new().set_measure("rock", 2.0); // Requires Probability, including through traits.
//! ```

pub mod math;
pub mod random;
pub mod spatial;
pub mod structures;
pub mod units;

/// Time quantities, frame timing, and clocks.
pub mod time;
pub mod world;
