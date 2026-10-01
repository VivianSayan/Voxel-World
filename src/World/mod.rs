//! Voxel type registration, storage, and deterministic generation.
//!
//! [`voxel_type`] and [`voxel_type_index`] are the registry: a type is declared
//! by name and the index hands it the [`VoxelTypeId`] it answers to.
//! [`octree`] is the storage, where a node is either one type, a weighted
//! choice between types, eight children, or a promise that a generator will
//! supply it. [`generator`] is that generator, and [`chunk`] is the dense block
//! it fills.
//!
//! Nothing here holds a random stream. Content is a pure function of the world
//! seed and the voxel's own coordinate, so a chunk can be generated, thrown
//! away and generated again, in any order and on any thread, and come back the
//! same.

pub mod ids;
pub mod octree;
pub mod voxel_components;
pub mod voxel_type;
pub mod voxel_type_index;

pub use chunk::Chunk;
pub use generator::ChunkGenerator;
pub use ids::VoxelTypeId;
pub use octree::Octree;
pub use voxel_type::VoxelType;
pub use voxel_type_index::VoxelTypeIndex;
pub mod chunk;
pub mod generator;
