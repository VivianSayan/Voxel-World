# Voxel World — Voxel Component Architecture

> **Status:** Design draft  
> **Purpose:** Define the component-based voxel model for Voxel World, including visual, physical, simulation, interaction, and metadata components, plus the intended ECS-like usage model.

---

# 1. Core Design Principle

Voxel World treats voxel types as **compositions of components**, not as one large fixed material structure.

A voxel type should not be forced to contain every property the engine can possibly understand. Instead, it contains only the components relevant to that voxel.

The central rule is:

> **Properties describe what a voxel is. Behaviors describe what a voxel does. Systems only query the components they care about.**

This keeps voxel definitions modular and allows creators to add new voxel concepts without requiring changes to the engine's core voxel schema.

Examples:

```text
Stone
├── Phase(Solid)
├── Mobility(Static)
├── Density(...)
├── Collision(Solid)
└── Surface(...)
```

```text
Sand
├── Phase(Solid)
├── Mobility(Granular)
├── Density(...)
├── Friction(...)
├── Surface(...)
└── OnNeighborChanged(check_support)
```

```text
Water
├── Phase(Liquid)
├── Mobility(Flowing)
├── Density(...)
├── Surface(...)
└── Tick(update_flow)
```

```text
Grass
├── Phase(Solid)
├── Mobility(Static)
├── Surface(...)
└── RandomTick(...)
```

A creator can also add entirely new components:

```text
Magnetic(...)
Radioactive(...)
ManaConductive(...)
Photosynthetic(...)
SignalEmitter(...)
CustomAnything(...)
```

---

# 2. Voxel ECS Philosophy

Voxel World uses an ECS-like component philosophy for voxels.

However, voxels should **not necessarily use the exact same physical storage backend as ordinary entities**.

The conceptual split is:

```text
Entity ECS
    Dynamic objects
    Characters
    Creatures
    Machines
    Projectiles
    Items
    Effects

Voxel Component System
    Spatially fixed voxel cells
    Material/state descriptions
    Extremely large counts
    Specialized octree storage
```

Both systems can share common concepts:

```text
Entity / Voxel
    ↓
Component composition
    ↓
Systems query relevant components
```

But voxel storage must remain optimized for potentially enormous world volumes.

---

# 3. Voxel Type / Archetype Concept

Individual voxels should generally avoid storing a full component collection directly.

Instead, voxels should reference a shared voxel type/archetype definition.

Conceptually:

```rust
struct Voxel {
    archetype: VoxelArchetypeId,
}
```

An archetype contains the component set shared by all voxels of that type.

This allows millions of Stone voxels to reference the same Stone definition rather than duplicate all physical and visual data.

Possible structure:

```rust
struct VoxelArchetype {
    id: VoxelArchetypeId,
    components: ComponentSet,
}
```

The implementation may later use:

- Type-indexed component storage
- Archetype tables
- Component registries
- Packed IDs
- Static component layouts
- Compile-time component registration
- Runtime creator/plugin registration

The public API should remain independent from whichever storage strategy proves fastest.

---

# 4. Component Categories

The current conceptual categories are:

```text
Voxel Components
├── Visual
├── Physical
├── Simulation / Behavior
├── Interaction / Events
└── Metadata / Identity
```

These categories are organizational rather than necessarily distinct engine storage types.

---

# 5. Visual Components

Visual components describe how a voxel should be rendered.

The renderer should query visual information without needing to understand unrelated properties such as density, ticking, or physical phase.

Possible visual components include:

## 5.1 Surface

Describes the primary rendered appearance.

Possible information:

```text
Surface
├── Texture / texture set
├── Material reference
├── UV behavior
├── Surface shader data
└── Face-dependent variants
```

Example:

```rust
Surface {
    texture: STONE_TEXTURE,
}
```

---

## 5.2 Opacity

Controls whether and how strongly the voxel blocks light/visibility.

```rust
Opacity(1.0)
```

Possible uses:

- Fully opaque stone
- Semi-transparent glass
- Transparent liquids
- Fog-like materials

---

## 5.3 Emissive

Indicates that the voxel emits visible light.

```rust
Emissive {
    intensity: 3.0,
}
```

Potential later additions:

- Emission spectrum / color
- Flicker behavior
- Distance falloff metadata

---

## 5.4 Roughness

Surface roughness used by the renderer.

```rust
Roughness(0.8)
```

---

## 5.5 Reflectivity

Controls reflective behavior.

```rust
Reflectivity(0.4)
```

---

## 5.6 Refractive Index

Useful for glass, liquids, crystals, and other refractive materials.

```rust
RefractiveIndex(1.33)
```

---

## 5.7 Possible Future Visual Components

Potential future additions may include:

- Normal-map data
- Subsurface scattering
- Anisotropy
- Metallic behavior
- Animated surface
- Procedural surface
- Face-specific appearance
- Biome tint
- Wetness tint
- Damage overlays

These should be added only when actual use cases require them.

---

# 6. Physical Components

Physical components describe basic material behavior that may be queried by physics, collision, simulation, or gameplay systems.

---

## 6.1 Phase

The current intended phases are:

```rust
enum Phase {
    Solid,
    Liquid,
    Gas,
}
```

Plasma is intentionally omitted because it does not currently add useful engine behavior distinct enough to justify a core phase.

Examples:

```text
Stone → Solid
Sand  → Solid
Water → Liquid
Air   → Gas
```

Phase does **not** define movement behavior by itself.

---

## 6.2 Mobility

Mobility describes how the material tends to move or redistribute.

Current concept:

```rust
enum Mobility {
    Static,
    Granular,
    Flowing,
    Diffusive,
}
```

Examples:

```text
Stone
Phase: Solid
Mobility: Static

Sand
Phase: Solid
Mobility: Granular

Water
Phase: Liquid
Mobility: Flowing

Air
Phase: Gas
Mobility: Diffusive
```

Separating `Phase` and `Mobility` allows unusual combinations.

A creator could define:

```text
Phase: Solid
Mobility: Flowing
```

for a fictional material if desired.

---

## 6.3 Density

```rust
Density(f64)
```

Potential uses:

- Buoyancy
- Fluid layering
- Falling materials
- Physics
- Mass calculations
- Pressure systems

---

## 6.4 Hardness

```rust
Hardness(f64)
```

Potential uses:

- Mining speed
- Impact resistance
- Tool requirements
- Breakage systems

---

## 6.5 Friction

```rust
Friction(f64)
```

Potential uses:

- Entity movement
- Sliding
- Granular movement
- Physics response

---

## 6.6 Cohesion

```rust
Cohesion(f64)
```

Potential uses:

- Granular materials
- Clumping
- Soil
- Snow
- Mud
- Fictional materials

---

## 6.7 Collision

Describes whether and how the voxel participates in collision.

Possible conceptual forms:

```rust
enum Collision {
    None,
    Solid,
    Custom(...),
}
```

Collision should remain independent from visual opacity and physical phase.

Examples:

- Glass: solid collision, transparent visual
- Air: no collision
- Water: non-solid or fluid collision behavior
- Force-field voxel: invisible but collidable

---

## 6.8 Permeability

Potential property for systems involving fluid/gas flow through porous matter.

```rust
Permeability(f64)
```

This may remain optional until needed.

---

# 7. Simulation / Behavior Components

Simulation should be **opt-in**.

Static voxels should cost essentially nothing merely because simulated voxel types exist elsewhere in the world.

The engine should not scan all voxels every frame asking whether they need updates.

Instead, systems should maintain active/indexed sets of voxel archetypes or resolved voxels carrying relevant components.

---

# 8. Tick Component

A voxel can opt into scheduled ticking.

Conceptually:

```rust
Tick {
    interval: TickInterval,
    method: VoxelMethod,
}
```

Example:

```text
Tick(
    interval = 20,
    method = update_flow
)
```

Possible uses:

- Fluid movement
- Burning
- Decay
- Machine logic
- Growth
- Chemical reactions

The engine should schedule only voxels that actually contain a `Tick` component.

---

# 9. RandomTick Component

A voxel can opt into stochastic ticking.

The earlier simple idea was:

```rust
RandomTick {
    probability: f64,
    method: VoxelMethod,
}
```

However, repeatedly rolling a probability every engine tick is potentially wasteful.

A better design is to schedule the **next random tick directly**.

## 9.1 Poisson-Style Random Tick Scheduling

For a memoryless random event with an average rate, the waiting time between events follows an exponential distribution in continuous time.

For discrete engine ticks, the equivalent waiting-time model is geometric.

The engine can expose this as a Poisson-process-style random tick API.

Conceptually:

```rust
RandomTick {
    rate: f64,
    method: VoxelMethod,
}
```

When the voxel is scheduled:

```text
Generate next waiting interval
        ↓
Schedule voxel once
        ↓
Wait
        ↓
Call method
        ↓
Generate next waiting interval
        ↓
Reschedule
```

This avoids:

```text
every engine tick
    ↓
visit every RandomTick voxel
    ↓
roll random chance
    ↓
usually do nothing
```

Instead:

```text
priority queue / timing wheel / scheduler
    ↓
only visit voxel when its next event is due
```

This is much more scalable.

Possible public forms:

```rust
RandomTick::mean_interval(500, grow)
```

or:

```rust
RandomTick::rate(0.002, grow)
```

The exact API should be decided later.

Important design goal:

> Random ticking should be random in timing, but should not require constant random checks.

---

# 10. Event / Interaction Components

These components react to specific events.

They should be event-driven rather than polled.

---

## 10.1 OnPlaced

Called when a voxel is placed or becomes concretely instantiated due to gameplay.

```rust
OnPlaced(method)
```

Possible uses:

- Initialize state
- Notify neighbors
- Start scheduled behavior
- Spawn effects

---

## 10.2 OnRemoved

```rust
OnRemoved(method)
```

Possible uses:

- Drop items
- Notify neighbors
- Stop simulation
- Trigger collapse
- Release fluid/gas

---

## 10.3 OnNeighborChanged

```rust
OnNeighborChanged(method)
```

Important for granular materials.

Example:

```text
Sand
    ↓
supporting voxel removed
    ↓
OnNeighborChanged
    ↓
check whether sand should fall
```

This avoids constant ticking of every sand voxel.

---

## 10.4 OnEntered

Potential event for entities or other simulation objects entering a voxel.

```rust
OnEntered(method)
```

Possible uses:

- Damage
- Trigger zones
- Gas exposure
- Teleportation
- Environmental effects

---

## 10.5 OnInteract

```rust
OnInteract(method)
```

Possible uses:

- Switches
- Doors
- Machines
- Containers
- Harvestable resources

---

## 10.6 Possible Future Events

Potential later events:

```text
OnExited
OnBroken
OnDamaged
OnTemperatureChanged
OnPressureChanged
OnSignalReceived
OnLightChanged
OnFluidContact
OnEntityContact
```

These should be added only when actual engine use cases require them.

---

# 11. Metadata / Identity Components

Metadata should remain separate from simulation and rendering.

Possible components:

## 11.1 Name / ID

Voxel archetypes need stable identifiers.

```text
stone
sand
water
air
iron_ore
```

The engine should distinguish human-readable names from stable IDs if necessary.

---

## 11.2 Tags

Tags provide lightweight categorization.

Examples:

```text
rock
soil
flammable
organic
ore
transparent
natural
constructed
```

Tags should not replace real components where structured data is required.

---

## 11.3 Custom Creator Data

Creators may need arbitrary structured metadata.

The component framework should allow custom component types without requiring upstream engine changes.

---

# 12. Custom Components

Custom components are a major design requirement.

Creators should be able to define components such as:

```rust
struct Magnetic {
    strength: f64,
}

struct Radioactive {
    emission_rate: f64,
}

struct ManaConductive {
    conductivity: f64,
}
```

Custom systems can then query them.

Conceptually:

```rust
fn magnetic_system(query: Query<&Magnetic>) {
    ...
}
```

For voxels, the actual API may differ because data may live in octree/archetype structures rather than ordinary entity storage.

The important public design principle is:

> Custom creator components should behave like first-class engine components.

---

# 13. System Query Philosophy

Subsystems should only query the components they need.

Examples:

```text
Renderer
→ Surface
→ Opacity
→ Emissive
→ Roughness
→ Reflectivity
→ RefractiveIndex
```

```text
Collision System
→ Collision
→ Friction
```

```text
Granular System
→ Mobility(Granular)
→ Density
→ Cohesion
→ Friction
```

```text
Random Tick Scheduler
→ RandomTick
```

```text
Gas System
→ Phase(Gas)
→ Mobility(Diffusive)
```

The renderer should not need to know whether something ticks.

The random-tick scheduler should not need to know its texture.

This keeps systems decoupled.

---

# 14. Example Voxel Archetypes

## Stone

```text
Stone
├── Phase(Solid)
├── Mobility(Static)
├── Density(...)
├── Hardness(...)
├── Friction(...)
├── Collision(Solid)
├── Surface(stone)
├── Opacity(1.0)
└── Roughness(...)
```

## Sand

```text
Sand
├── Phase(Solid)
├── Mobility(Granular)
├── Density(...)
├── Hardness(...)
├── Friction(...)
├── Cohesion(...)
├── Collision(Solid)
├── Surface(sand)
└── OnNeighborChanged(check_support)
```

## Water

```text
Water
├── Phase(Liquid)
├── Mobility(Flowing)
├── Density(...)
├── Collision(None / Fluid)
├── Surface(water)
├── Opacity(...)
├── Reflectivity(...)
├── RefractiveIndex(...)
└── Tick(update_flow)
```

## Air

```text
Air
├── Phase(Gas)
├── Mobility(Diffusive)
├── Density(...)
├── Collision(None)
└── Opacity(0.0)
```

## Grass

```text
Grass
├── Phase(Solid)
├── Mobility(Static)
├── Collision(Solid)
├── Surface(grass)
└── RandomTick(growth_rate, grow_or_spread)
```

---

# 15. Conceptual Component Grammar

The API should aim to feel compositional.

A voxel definition could conceptually read like:

```rust
voxel("sand")
    .with(Phase::Solid)
    .with(Mobility::Granular)
    .with(Density(1.6))
    .with(Friction(0.7))
    .with(Collision::Solid)
    .with(Surface::new("sand"))
    .with(OnNeighborChanged::new(check_support));
```

A custom voxel:

```rust
voxel("radioactive_crystal")
    .with(Phase::Solid)
    .with(Mobility::Static)
    .with(Surface::new("radioactive_crystal"))
    .with(Emissive::new(2.0))
    .with(Radioactive {
        emission_rate: 0.03,
    })
    .with(RandomTick::rate(
        0.001,
        emit_radiation,
    ));
```

This exact syntax is not fixed, but the intended feel is:

```text
voxel
    + components
    + optional event/simulation hooks
    + creator-defined components
```

---

# 16. Performance Principles

Voxel components must be designed around scale.

Important rules:

1. **Do not store full component data per voxel when it can be shared by archetype.**
2. **Do not scan all voxels for behavior.**
3. **Ticking is opt-in.**
4. **Random ticking should be scheduled rather than probability-polled every frame/tick.**
5. **Event reactions should be event-driven.**
6. **Static voxels should remain extremely cheap.**
7. **Custom components should not force dynamic dispatch on every voxel access if avoidable.**
8. **Systems should query only relevant archetypes/components.**
9. **Resolved voxel detail should exist only where world generation requires it.**
10. **The component API may be friendly while the underlying storage remains aggressively optimized.**

---

# 17. Implementation Checklist

## Core Component System

- [ ] Define `VoxelArchetypeId`
- [ ] Define voxel archetype registry
- [ ] Define component type registration system
- [ ] Define component storage strategy
- [ ] Support built-in components
- [ ] Support custom creator components
- [ ] Implement component lookup/query API
- [ ] Implement archetype sharing/deduplication
- [ ] Add serialization for voxel archetypes
- [ ] Add stable IDs for voxel types/components

## Visual Components

- [ ] `Surface`
- [ ] `Opacity`
- [ ] `Emissive`
- [ ] `Roughness`
- [ ] `Reflectivity`
- [ ] `RefractiveIndex`
- [ ] Renderer queries visual components only
- [ ] Define defaults / absence behavior
- [ ] Add visual debug inspection

## Physical Components

- [ ] `Phase`
- [ ] `Mobility`
- [ ] `Density`
- [ ] `Hardness`
- [ ] `Friction`
- [ ] `Cohesion`
- [ ] `Collision`
- [ ] Consider `Permeability`
- [ ] Define physical query API

## Simulation Components

- [ ] `Tick`
- [ ] Tick scheduler
- [ ] `RandomTick`
- [ ] Random tick waiting-time model
- [ ] Random tick scheduler
- [ ] Ensure static voxels have no scheduler cost
- [ ] Define custom simulation component support

## Event Components

- [ ] `OnPlaced`
- [ ] `OnRemoved`
- [ ] `OnNeighborChanged`
- [ ] `OnEntered`
- [ ] `OnInteract`
- [ ] Event dispatcher
- [ ] Efficient neighbor-change propagation
- [ ] Decide whether hooks are functions, IDs, systems, or callbacks

## Custom Components

- [ ] Creator API for declaring components
- [ ] Creator API for registering systems
- [ ] Custom component serialization
- [ ] Versioning strategy for custom components
- [ ] Plugin/module ownership of component types
- [ ] Error handling for missing custom components/plugins

## Performance

- [ ] Benchmark component lookup
- [ ] Benchmark archetype lookup
- [ ] Benchmark event dispatch
- [ ] Benchmark tick scheduling
- [ ] Benchmark random-tick scheduling
- [ ] Verify no full-world tick scans
- [ ] Verify no unnecessary per-voxel component duplication

---

# 18. Open Questions

- [ ] Should voxel components and entity components share one registry?
- [ ] Should some components be compile-time/static only?
- [ ] How should custom methods/hooks be represented?
- [ ] How should plugin-defined component serialization work?
- [ ] Should `Phase` and `Mobility` be enums, tags, or ordinary components?
- [ ] Should `Collision` be a single component or multiple specialized collision components?
- [ ] Should tick scheduling use a heap, timing wheel, buckets, or another structure?
- [ ] Should RandomTick expose mean interval, rate, or both?
- [ ] How should voxel state that differs from archetype defaults be represented?
- [ ] When should mutable per-voxel state force octree refinement?
