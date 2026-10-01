//! Stateless generation keyed by global voxel positions and stable type names.
//!
//! A generator holds no stream and no cache. Every voxel is a pure function of
//! the world seed and its own global coordinate, so chunks may be generated in
//! any order, on any thread, thrown away and generated again, and come back
//! identical.

mod chunks;
pub use chunks::ChunkGenerator;
