# Voxel World

**Voxel World** is an open-source, voxel-native game engine focused on large procedural worlds, modular voxel behavior, efficient simulation, and Vulkan rendering.

It is written primarily in **Rust** and is being designed from the ground up around voxels rather than adapting a conventional mesh-oriented engine to support them.

This is also very much a passion project.

I started Voxel World because I wanted an engine where voxels are not treated as a special terrain system bolted onto something else, but as the fundamental structure of the world itself. I am building the engine while learning more about Rust, Vulkan, rendering, data structures, procedural generation, and engine architecture along the way.

That means the project is both an engine and an ongoing learning process. There will be mistakes, experiments, rewrites, and ideas that turn out not to work. That is part of the project.

My hope is that Voxel World eventually becomes useful not only for the games and experiments I want to make, but for anyone else interested in building large, strange, procedural, destructible, or simulation-heavy voxel worlds.

> **Current status:** Very early development.  
> Major architectural decisions are still being explored and almost everything may change.

## Goals

Voxel World aims to provide a flexible foundation for games and simulations where voxels are a fundamental part of the world rather than an additional terrain feature.

Some of the major goals are:

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

Not all of these systems exist yet. Some are currently little more than designs, prototypes, or ideas waiting to be tested.

## Why Voxel-Native?

Voxel World treats voxel data as first-class world state.

The engine is not intended to generate a voxel world and then convert it into a conventional scene or object hierarchy. Storage, rendering, simulation, spatial queries, and world streaming are instead being designed around voxel-oriented structures directly.

One important idea behind the project is that not every part of a voxel world needs to be represented in the same way.

A huge empty region, a solid mass of material, complicated terrain, and a heavily simulated machine all have very different requirements. The engine should be able to take advantage of that.

Different regions may therefore use different storage representations depending on what they contain and what needs to be done with them.

## Component-Based Voxels

Voxel types are intended to be modular rather than defined by one enormous fixed material structure.

A voxel type might contain components describing things such as:

- Physical phase
- Mobility
- Density
- Collision behavior
- Rendering properties
- Interaction properties
- Simulation behavior

Creators should also be able to define their own components and systems.

Simulation can then be opt-in rather than something every voxel constantly participates in. A voxel might, for example, use components such as:

- `Tick`
- `RandomTick`
- Neighbor-change reactions
- Placement reactions
- Removal reactions

Most voxels in a world are probably doing absolutely nothing most of the time.

The engine should let them be cheap accordingly.

## Entity Component System

Voxel World is also being designed around Entity Component System principles.

Dynamic objects such as characters, creatures, machines, projectiles, and other entities can be represented through composable components instead of rigid inheritance hierarchies.

Voxel data and ordinary entities may share similar ideas about components and systems while still using very different storage models internally.

A world may contain billions of voxels but only thousands of conventional entities. Treating both exactly the same would make little sense.

## Rendering

Rendering is a core part of the project and something I want to develop relatively early.

Voxel World intends to use **Vulkan** directly so that the renderer can be designed around voxel-specific requirements instead of forcing voxel data through an existing general-purpose rendering architecture.

Rendering is also extremely useful while developing the engine itself.

Being able to actually see what spatial structures, generation systems, traversal algorithms, and storage systems are doing makes debugging considerably easier than staring at numbers in a terminal.

Planned debugging views may eventually include visualization of:

- Voxel hierarchy
- Loaded world regions
- Spatial boundaries
- Level of detail
- Ray traversal
- Collision data
- Active simulation regions
- Entity positions

## Scope

Voxel World is **not** currently intended to become a complete replacement for general-purpose engines such as Godot, Unity, or Unreal Engine.

I do not want to implement systems simply because "game engines are supposed to have them."

The project will instead concentrate on features that are useful for voxel-based games and simulations.

Early development is currently focused roughly around:

1. Core spatial and voxel structures
2. Basic Vulkan rendering
3. World generation and streaming
4. Entity and component architecture
5. Voxel interaction and simulation
6. Development and debugging tools

Other systems can be added when there is an actual reason for them to exist.

## Development Status

Voxel World is currently in its earliest stages.

There is no stable public API, and there probably will not be one for quite some time.

Expect:

- Breaking changes
- Experiments
- Bad ideas that get replaced by better ones
- Architecture being rewritten
- Incomplete documentation
- Missing systems
- Rapid iteration

Backward compatibility is not currently a priority.

At this stage, discovering the right architecture matters much more than preserving the wrong one.

I am also learning parts of the technology stack while building the project, particularly Rust and lower-level engine development. If you find something questionable, inefficient, unidiomatic, or simply wrong, there is a very real possibility that you are right.

Please tell me.

## Contributing

Contributions are extremely welcome.

That includes code, but it is absolutely not limited to code.

If you want to:

- Fix something
- Improve documentation
- Suggest a better architecture
- Point out terrible Rust
- Test something
- Benchmark something
- Explain why an idea will explode
- Propose a feature
- Build an experiment
- Ask questions
- Discuss voxel-engine design

...please do.

This project is meant to be useful to people, and I would much rather build it alongside people who find the subject interesting than pretend I already know every correct answer.

For small fixes and documentation improvements, feel free to open a pull request.

For major features, architectural changes, new dependencies, or large refactors, please open an issue or discussion first. Not because ideas need to pass some grand committee, but because the architecture is changing quickly and it is better to make sure someone is not spending hours implementing something that conflicts with another part of the engine.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more information.

## Project Principles

There are a few principles currently guiding development:

1. **Voxels are first-class world data.**
2. **Rendering should adapt to voxel representation rather than dictate it.**
3. **Large worlds should not require all terrain to remain loaded.**
4. **Different kinds of voxel data may use different storage representations.**
5. **Simulation should avoid unnecessary constant polling.**
6. **Extensibility should come through modular components and systems.**
7. **The engine should remain independent of any particular game.**
8. **Features should solve real use cases rather than imitate other engines by default.**
9. **Experimentation is welcome.**
10. **Being wrong and replacing something is better than being afraid to try it.**

These principles are not commandments. They describe how I currently think the engine should be built, and they may evolve as the project does.

## License

Voxel World is licensed under the **MIT License**.

You are free to use, modify, distribute, and build commercial or non-commercial projects with Voxel World under the terms of that license.

If this project eventually helps someone build something interesting, that is exactly what I want it to do.

See [`LICENSE`](LICENSE) for the full license text.