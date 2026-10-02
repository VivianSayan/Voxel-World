//! Spatial addresses, distances, and stateless noise fields.
//!
//! [`depth`] holds the two kinds of octree level, [`position`] the three kinds
//! of address (a voxel, a node at some level, and a fraction inside a cell),
//! and [`measure`] the distance between them. [`noise`] holds the fields that
//! turn a position and a seed into a value, with no state and no storage: the
//! same position always gives the same sample, so a chunk can be generated,
//! thrown away and generated again.

pub mod depth;
pub mod measure;
pub mod morton;
pub mod noise;
pub mod position;
pub mod kinematics;
pub mod precise;

pub use depth::{Depth, TreeDepth};
pub use kinematics::{
    Acceleration2, Acceleration3, Acceleration4, Displacement2, Displacement3, Displacement4,
    Force2, Force3, Force4, Mass, Velocity2, Velocity3, Velocity4,
};
pub use measure::WorldUnits;
pub use morton::MortonKey;
pub use position::{
    LocalPosition2, LocalPosition3, LocalPosition4, NodePosition2, NodePosition3, NodePosition4,
    VoxelPosition2, VoxelPosition3, VoxelPosition4,
};
