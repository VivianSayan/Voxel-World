# Voxel World

**Voxel World** is an open-source, voxel-native game engine focused on large procedural worlds, modular voxel behavior, efficient simulation, and Vulkan rendering.

The engine is written primarily in **Rust** and is designed from the ground up around voxels rather than adapting a conventional mesh-oriented game engine to support them.

> **Current status:** Very early development.  
> The project is currently being established and major architectural decisions are still subject to change.

## Goals

Voxel World aims to provide a flexible foundation for games and simulations where voxels are a fundamental part of the world rather than an additional terrain feature.

Major goals include:

- Large voxel worlds
- Procedural world generation
- Efficient world streaming
- Hierarchical voxel storage
- Multiple voxel storage and compression strategies
- Vulkan-based rendering designed around voxel data
- Modular voxel properties and behaviors
- Event-driven simulation where possible
- Entity Component System architecture
- Support for creator-defined components and systems
- Deterministic systems where useful
- Strong debugging and visualization tools

## Voxel-Native Design

Voxel World treats voxel data as first-class world state.

The engine is not based around converting voxel worlds into a conventional scene or object hierarchy. Instead, storage, rendering, simulation, spatial queries, and world streaming are intended to operate directly around voxel-oriented structures.

Different regions of the world may use different internal representations depending on their contents and requirements.

Large empty areas, homogeneous materials, complex terrain, and actively simulated regions should not necessarily need to be represented in the same way.

## Component-Based Voxels

Voxel types are intended to be modular rather than defined by one fixed material structure.

A voxel type may contain components describing properties such as:

- Physical phase
- Mobility
- Density
- Collision behavior
- Rendering properties
- Interaction properties
- Simulation behavior

Creators should also be able to define additional components and systems for their own games.

For example, simulation may be opt-in through components such as:

- `Tick`
- `RandomTick`
- Neighbor-change reactions
- Placement or removal reactions

This allows static voxels to remain inexpensive while more complex voxel types can participate in simulation when required.

## Entity Component System

Voxel World is being designed around Entity Component System principles.

Dynamic objects such as characters, creatures, machines, projectiles, and other entities can be represented through composable components rather than rigid inheritance hierarchies.

Voxel data and ordinary entities may use similar component concepts while retaining storage models appropriate to their very different scale and performance requirements.

## Rendering

Rendering is a core part of the project and will be developed relatively early.

Voxel World intends to use **Vulkan** directly, allowing the renderer to be designed around voxel-specific requirements instead of adapting voxel data to an existing general-purpose rendering architecture.

Rendering will also serve as an important development and debugging tool.

Planned debugging views may include visualization of:

- Voxel hierarchy
- Loaded world regions
- Spatial boundaries
- Level of detail
- Ray traversal
- Collision data
- Active simulation regions
- Entity positions

## Current Scope

Voxel World is **not** currently intended to become a complete replacement for general-purpose engines such as Godot, Unity, or Unreal Engine.

The project will prioritize features required by voxel-based games and simulations rather than implementing features simply because conventional game engines usually contain them.

In particular, early development will focus on:

1. Core spatial and voxel structures
2. Basic Vulkan rendering
3. World generation and streaming
4. Entity and component architecture
5. Voxel interaction and simulation
6. Development and debugging tools

Additional systems will be introduced as actual use cases require them.

## Development Status

Voxel World is currently in its earliest stage.

There is not yet a stable public API.

Expect:

- Breaking changes
- Experimental implementations
- Reworked architecture
- Incomplete documentation
- Missing systems
- Rapid iteration

Backward compatibility is not currently guaranteed.

## Contributing

Contributions, ideas, testing, and discussion are welcome.

For small fixes and documentation improvements, feel free to open a pull request.

For major features, architectural changes, new dependencies, or large refactors, please open an issue or discussion first so the direction can be agreed upon before significant work is done.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more information.

## Project Principles

Voxel World follows several general design principles:

1. **Voxels are first-class world data.**
2. **Rendering should adapt to voxel representation rather than dictate it.**
3. **Large worlds should not require all terrain to remain loaded.**
4. **Different kinds of voxel data may use different storage representations.**
5. **Simulation should avoid unnecessary constant polling.**
6. **Extensibility should come through modular components and systems.**
7. **The engine should remain independent of any particular game.**
8. **Features should solve real use cases rather than imitate other engines by default.**

These principles may evolve as the project develops, but they describe the current direction of the engine.

## License

Voxel World is licensed under the **MIT License**.

You are free to use, modify, distribute, and build commercial or non-commercial projects with Voxel World under the terms of that license.

See [`LICENSE`](LICENSE) for the full license text.