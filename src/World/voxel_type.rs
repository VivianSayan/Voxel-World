//! What one kind of voxel is: a name, the id the registry gave it, and the
//! components that describe how it behaves.

use crate::world::ids::VoxelTypeId;
use crate::world::voxel_components::VoxelComponent;

/// A kind of voxel, before or after it has been registered.
///
/// The name is the stable identity, the one a save file and a generator's
/// palette refer to. The id is a local slot in whichever
/// [`VoxelTypeIndex`](crate::world::VoxelTypeIndex) the type was added to, and
/// is [`VoxelTypeId::NONE`] until then.
pub struct VoxelType {
    /// Assigned by the index this type is added to, and [`VoxelTypeId::NONE`]
    /// until then.
    pub(crate) id: VoxelTypeId,
    name: String,
    properties: Vec<VoxelComponent>,
}

impl VoxelType {
    /// An unregistered type with a name and no components yet.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: VoxelTypeId::NONE,
            name: name.into(),
            properties: Vec::new(),
        }
    }

    /// The id the registry gave this type, or [`VoxelTypeId::NONE`] while it is
    /// unregistered.
    pub fn id(&self) -> VoxelTypeId {
        self.id
    }

    /// The type's name, which is its identity across saves and across runs.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The components describing how this type behaves.
    pub fn properties(&self) -> &[VoxelComponent] {
        &self.properties
    }
}
