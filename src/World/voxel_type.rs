use crate::World::voxel_components::VoxelComponent;

struct VoxelType {
    id: usize,
    name: String,
    properties: Vec<VoxelComponent>,
}