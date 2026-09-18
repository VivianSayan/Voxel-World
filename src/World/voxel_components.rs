pub struct VoxelComponent {}

trait Component {
    
}


// Voxel Archetype
// ├── Visual
// │   ├── Surface
// │   ├── Opacity
// │   ├── Emissive
// │   ├── Roughness
// │   ├── Reflectivity
// │   └── RefractiveIndex
// │
// ├── Physical
// │   ├── Phase
// │   ├── Mobility
// │   ├── Density
// │   ├── Hardness
// │   ├── Friction
// │   ├── Cohesion
// │   └── Collision
// │
// ├── Simulation
// │   ├── Tick
// │   ├── RandomTick
// │   ├── Temperature
// │   ├── Flammable
// │   ├── Conductive
// │   └── custom simulation components
// │
// ├── Interaction / Events
// │   ├── OnPlaced
// │   ├── OnRemoved
// │   ├── OnNeighborChanged
// │   ├── OnEntered
// │   └── OnInteract
// │
// └── Metadata
//     ├── Name / ID
//     ├── Tags
//     └── custom creator data