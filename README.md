# Voxel World

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Status: Early Development](https://img.shields.io/badge/status-early%20development-orange.svg)
![Rust](https://img.shields.io/badge/language-Rust-orange.svg)
**Voxel World** is an open-source voxel game engine written in Rust.
The main idea is to build the engine around voxels from the beginning, rather than treating them as terrain that eventually gets converted into a more conventional scene or mesh system.
The long-term goal is to support large procedural worlds, modular voxel behaviour, efficient simulation, and Vulkan-based rendering.
> **Status: very early development.** The project is still experimental, major parts of the architecture are being figured out, and almost everything may change.
This is also a learning project. I'm building Voxel World while learning Rust, Vulkan, rendering, engine architecture, procedural generation, and a lot of lower-level systems work along the way.
There will be mistakes, rewrites, dead ends, questionable experiments, and ideas that turn out not to work. That's expected.
For now, the goal is simple:
**get voxels on the screen, then build outward from there.**
---
## Why Voxel-Native?
A lot of engines can render voxel worlds, but they are still fundamentally designed around conventional objects, meshes, and scenes.
Voxel World is an attempt to see what happens when storage, rendering, simulation, spatial queries, streaming, and world generation are designed around voxel data directly.
Different parts of a world may also want different representations.
An empty region, a completely solid region, complicated terrain, and a heavily simulated machine do not necessarily need to be stored in the same way.
The current direction is therefore to allow multiple storage and compression strategies rather than forcing the entire world into one representation.
Voxel types are also intended to be modular. Instead of every voxel being one fixed material definition, different behaviours and properties can eventually be composed from components.
Most voxels should also be cheap when they are doing nothing. Constantly updating billions of blocks just because they exist would be rather silly.
Dynamic objects such as characters, machines, and projectiles will likely use an ECS-style architecture. Voxels and entities may share some ideas, but they have very different storage requirements, so I do not intend to treat them as the same thing internally.
There will eventually be a more complete architecture document once enough of this has survived contact with actual code.
## Rendering
Rendering uses **Vulkan** directly.
Part of the reason rendering is being worked on early is simply that seeing the world is much more useful than debugging spatial structures and generation entirely through terminal output.
The renderer is still extremely basic.
Eventually I would like it to support useful development visualizations for things such as:
- voxel hierarchies
- loaded regions
- spatial boundaries
- LOD
- ray traversal
- collision data
- active simulation regions
- entity positions
## Scope
Voxel World is not intended to compete with general-purpose engines like Godot, Unity, or Unreal.
The point is not to implement every feature usually associated with a game engine.
If something does not help voxel-based games or simulations, there is a good chance it does not belong here.
The current rough priorities are:
1. Basic Vulkan rendering
2. Core spatial and voxel structures
3. World generation and streaming
4. Entity and component systems
5. Voxel interaction and simulation
6. Development and debugging tools
The order will probably change as the project develops.
## Goals
Some of the things I would like Voxel World to eventually support include:
- large voxel worlds with efficient streaming
- hierarchical voxel storage
- multiple storage and compression strategies
- Vulkan rendering designed around voxel data
- modular voxel properties and behaviours
- creator-defined systems and components
- event-driven and scheduled simulation
- deterministic world generation where useful
- ECS-style dynamic entities
- strong debugging and visualization tools
A lot of this does **not** exist yet.
Some parts are prototypes, some are design ideas, and some are currently little more than notes waiting to be tested.
## Building from Source
There are no proper build instructions yet.
The project is still changing too quickly for me to pretend there is a polished setup process.
Once there is a stable runnable milestone, this section will be expanded.
For now, the source itself is the best indication of what currently works.
## Contributing
Contributions are welcome, including code, documentation, benchmarks, bug reports, architecture criticism, questions, or general discussion about voxel-engine design.
For small fixes and documentation changes, opening a pull request directly is fine.
For large features, new dependencies, major refactors, or architecture changes, opening an issue or discussion first is probably a good idea. The design is still moving around enough that two people could otherwise spend a lot of time solving the same problem in completely different directions.
I'm learning parts of this stack as I go.
If something looks strange, inefficient, unidiomatic, overcomplicated, or simply wrong, there is a decent chance it is.
Please point it out.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more details.
## Project Principles
These are less rules and more the current direction of the project:
1. Voxels are first-class world data.
2. Rendering should adapt to voxel representation rather than dictate it.
3. Large worlds should not require everything to remain loaded.
4. Different kinds of voxel data may use different storage strategies.
5. Simulation should avoid doing work when nothing needs updating.
6. Extensibility should come through modular systems and components.
7. The engine should remain independent of any particular game.
8. Features should exist because they solve a real problem, not because other engines have them.
9. Experimentation is encouraged.
10. Replacing a bad idea is better than being afraid to try one.
These will probably evolve too.
## License
Voxel World is licensed under the **MIT License**.
You are free to use, modify, distribute, and build commercial or non-commercial projects with it.
See [`LICENSE`](LICENSE) for the full text.