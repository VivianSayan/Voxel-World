//! The registry of voxel types, and where their ids come from.

use crate::world::ids::VoxelTypeId;
use crate::world::voxel_type::VoxelType;

/// Every voxel type the world knows, in the order they were added.
///
/// A type's id is its position here, so lookup is an index rather than a hash,
/// and ids are only meaningful within one index. Names are what survive between
/// runs: two registries that were given the same names in a different order
/// hand out different ids, which is why a generator's palette is keyed on the
/// name and sorted by it.
pub struct VoxelTypeIndex {
    voxel_types: Vec<VoxelType>,
}

impl VoxelTypeIndex {
    /// An empty registry.
    pub fn new() -> Self {
        VoxelTypeIndex {
            voxel_types: Vec::new(),
        }
    }

    /// Adds a type and gives it the id it will answer to, which is its
    /// position in the index.
    ///
    /// `None` for an empty name, and for one already registered, since the name
    /// is the identity everything outside this registry uses. The check walks
    /// the existing types, which is fine for the handful of types a world
    /// declares at startup.
    pub fn add_voxel_type(&mut self, mut voxel_type: VoxelType) -> Option<VoxelTypeId> {
        if voxel_type.name().is_empty()
            || self
                .voxel_types
                .iter()
                .any(|held| held.name() == voxel_type.name())
        {
            return None;
        }
        let id: VoxelTypeId = VoxelTypeId::new(self.voxel_types.len() as u64);

        voxel_type.id = id;
        self.voxel_types.push(voxel_type);

        Some(id)
    }

    /// The type an id names, or `None` when nothing has that id.
    ///
    /// [`VoxelTypeId::NONE`] reads as absent rather than as an index, since its
    /// value is past the end of any index that will ever exist.
    pub fn get_voxel_type(&self, id: VoxelTypeId) -> Option<&VoxelType> {
        self.voxel_types.get(usize::try_from(id.value()).ok()?)
    }

    /// How many types are registered.
    pub fn len(&self) -> usize {
        self.voxel_types.len()
    }

    /// Whether no type is registered yet.
    pub fn is_empty(&self) -> bool {
        self.voxel_types.is_empty()
    }
}

/// An empty registry; see [`VoxelTypeIndex::new`].
impl Default for VoxelTypeIndex {
    fn default() -> Self {
        Self::new()
    }
}
