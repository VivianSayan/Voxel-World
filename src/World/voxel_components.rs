//! What a voxel type is made of, beyond its name.
//!
//! A placeholder so far: the tree below sketches the components a type will
//! carry, grouped by what reads them.

/// One property attached to a voxel type. Not yet implemented.
pub struct VoxelComponent {}

/// What every voxel component will implement. Not yet implemented.
pub trait Component {}

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
