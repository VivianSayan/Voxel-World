//! A small dense chunk connecting generation to voxel queries.
//!
//! The unit a generator fills and a query reads from: every voxel stored, no
//! compression, no sharing. The octree is where sparse storage belongs; this is
//! the flat form to hand to whatever consumes a region.

use super::VoxelTypeId;
use crate::spatial::VoxelPosition3;
use crate::structures::storage::Grid3;

/// A 16-by-16-by-16 block of voxel types, placed at a global voxel coordinate.
///
/// The voxels are stored x fastest, then y, then z, which is the order
/// [`ChunkGenerator::generate`](crate::world::ChunkGenerator::generate) writes
/// them in.
///
/// The origin is the chunk's lowest corner, and is not required to be a
/// multiple of the edge: a generator may fill a block anywhere.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub(crate) origin: VoxelPosition3,
    pub(crate) voxels: Grid3<VoxelTypeId>,
}

impl Chunk {
    /// How many voxels the chunk spans along each axis.
    pub const EDGE: usize = 16;
    /// How many voxels the chunk holds in total.
    pub const VOLUME: usize = Self::EDGE * Self::EDGE * Self::EDGE;

    /// The chunk's lowest corner, in global voxel coordinates.
    pub fn origin(&self) -> VoxelPosition3 {
        self.origin
    }

    /// Every voxel, x fastest then y then z, as one flat slice of
    /// [`Self::VOLUME`] entries.
    pub fn voxels(&self) -> &[VoxelTypeId] {
        self.voxels.as_slice()
    }

    /// The type at a global coordinate, or `None` for a coordinate outside this
    /// chunk.
    ///
    /// The position is taken relative to the origin, which is why a coordinate
    /// below it, or `EDGE` or more past it on any axis, reads as absent rather
    /// than wrapping onto another row.
    pub fn voxel(&self, position: VoxelPosition3) -> Option<VoxelTypeId> {
        let point = position.to_array();
        let origin = self.origin.to_array();
        let mut local = [0usize; 3];
        for axis in 0..3 {
            local[axis] = usize::try_from(point[axis].checked_sub(origin[axis])?).ok()?;
            if local[axis] >= Self::EDGE {
                return None;
            }
        }
        self.voxels.get(local).copied()
    }
}
