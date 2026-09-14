use crate::World::voxel_components::VoxelComponent;

struct VoxelType {
    id: u32,
    name: String,
    properties: Vec<VoxelComponent>,
}