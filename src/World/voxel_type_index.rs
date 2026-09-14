use crate::World::voxel_type::VoxelType;

struct VoxelTypeIndex {
    voxel_types: Vec<VoxelType>,
}

impl VoxelTypeIndex {
    pub fn new() -> Self {
        VoxelTypeIndex {
            voxel_types: Vec::new(),
        }
    }

    pub fn add_voxel_type(&mut self, voxel_type: VoxelType) {
        voxel_type.id = self.voxel_types.len() as usize;
        self.voxel_types.push(voxel_type);
    }

    pub fn get_voxel_type(&self, id: usize) -> Option<&VoxelType> {
        self.voxel_types.get(id)
    }
}