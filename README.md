# Voxel World

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Status: Early Development](https://img.shields.io/badge/status-early%20development-orange.svg)
![Rust](https://img.shields.io/badge/language-Rust-orange.svg)

**Voxel World** is an open-source, voxel-native game engine focused on large procedural worlds, modular voxel behavior, efficient simulation, and Vulkan rendering.

It is written in **Rust** and designed from the ground up around voxels — not as a terrain system bolted onto a conventional mesh-oriented engine, but as the fundamental structure of the world itself.

> **Status: very early development.** Major architectural decisions are still being explored and almost everything may change. There is no stable public API, no build instructions yet, and no screenshots — those are coming as the renderer and world systems take shape.

This is also a passion project and a learning process. I'm building the engine while learning Rust, Vulkan, rendering, data structures, procedural generation, and engine architecture along the way. There will be mistakes, experiments, and rewrites — that's part of the project.

---

## Why Voxel-Native?

Most engines generate voxel data and then convert it into a conventional scene or object hierarchy. Voxel World instead designs storage, rendering, simulation, spatial queries, and world streaming directly around voxel data as first-class world state.

Not every region of a world needs the same representation — an empty void, a solid mass, complex terrain, and a simulated machine all have different requirements, so different regions can use different storage strategies.

Voxel types themselves are modular: instead of one fixed material definition, a voxel is composed of components (physical phase, density, collision, rendering, simulation behavior, etc.), and creators can define their own. Simulation is opt-in per voxel — most voxels are doing nothing most of the time, and the engine should let them be cheap accordingly.

The engine also follows **Entity Component System** principles for dynamic objects (characters, machines, projectiles), sharing conceptual ground with the voxel component model while using different storage internally — a world may hold billions of voxels but only thousands of entities, and treating both the same would waste effort.

*(Full architectural writeup planned for a `docs/ARCHITECTURE.md` as the design solidifies.)*

## Rendering

Voxel World renders directly with **Vulkan**, designed around voxel-specific requirements rather than routed through a general-purpose rendering pipeline. Rendering is being prioritized early, partly because it makes debugging spatial structures, generation, and traversal far easier than reading numbers in a terminal.

Planned debug visualizations include voxel hierarchy, loaded regions, spatial boundaries, LOD, ray traversal, collision data, active simulation regions, and entity positions.

## Scope

Voxel World is **not** aiming to replace general-purpose engines like Godot, Unity, or Unreal. It won't implement systems just because "engines are supposed to have them" — only features that serve voxel-based games and simulations.

Current focus areas, roughly in order:

1. Core spatial and voxel structures
2. Basic Vulkan rendering
3. World generation and streaming
4. Entity and component architecture
5. Voxel interaction and simulation
6. Development and debugging tools

## Goals

- Large voxel worlds with efficient streaming
- Hierarchical voxel storage with multiple storage/compression strategies
- Vulkan-based rendering designed around voxel data
- Modular, creator-extensible voxel components and behaviors
- Event-driven simulation, deterministic where useful
- Entity Component System architecture
- Strong debugging and visualization tools

Not all of these exist yet — some are designs or prototypes waiting to be tested.

## Building from Source

No build instructions yet — the project is too early for a reliable setup path. This section will be filled in once there's a runnable milestone. In the meantime, cloning the repo and reading through the source is the best way to see where things stand.

## Contributing

Contributions are extremely welcome — code, documentation, architecture critique, benchmarks, bug reports, questions, or just discussion of voxel-engine design.

- **Small fixes / docs improvements:** open a pull request directly.
- **Major features, architectural changes, new dependencies, or large refactors:** open an issue or discussion first. The architecture is moving quickly, and it's better to check for conflicts before investing hours into something.

I'm learning parts of this stack (especially Rust and lower-level engine work) as I go. If something looks questionable, inefficient, unidiomatic, or wrong — there's a real chance it is. Please tell me.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## Project Principles

1. Voxels are first-class world data.
2. Rendering should adapt to voxel representation, not dictate it.
3. Large worlds should not require all terrain to stay loaded.
4. Different voxel data may use different storage representations.
5. Simulation should avoid unnecessary constant polling.
6. Extensibility comes through modular components and systems.
7. The engine stays independent of any particular game.
8. Features should solve real use cases, not imitate other engines by default.
9. Experimentation is welcome.
10. Being wrong and replacing something beats being afraid to try it.

These aren't commandments — they describe how I currently think the engine should be built, and may evolve with the project.

## License

Voxel World is licensed under the **MIT License**. You're free to use, modify, distribute, and build commercial or non-commercial projects with it. See [`LICENSE`](LICENSE) for the full text.
