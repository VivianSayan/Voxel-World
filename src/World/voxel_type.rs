use crate::World::voxel_components::VoxelComponent;

pub struct VoxelType {
    pub id: usize,
    name: String,
    properties: Vec<VoxelComponent>,
}