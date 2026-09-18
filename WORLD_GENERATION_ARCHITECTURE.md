# Voxel World — Procedural Generation and Octree Architecture

> **Status:** Design draft  
> **Purpose:** Describe Voxel World's lazy octree generation model, procedural generator API, noise system, random generation, and implementation roadmap.

---

# 1. Core Generation Philosophy

Voxel World should avoid generating detailed voxel data merely because that data *could* exist.

The world should remain as abstract and unresolved as possible until some system actually requires more detail.

The core principle is:

> **Do not generate detail until visibility, interaction, collision, simulation, or another concrete requirement forces that detail to exist.**

This means a deep underground region may exist only as:

- An implicit generator rule
- A coarse octree node
- A uniform material
- A probability distribution
- Deferred procedural rules

The engine should not generate every voxel hundreds or thousands of units beneath a player merely because the surrounding chunk is loaded.

---

# 2. World Truth vs Stored Realization

Voxel World distinguishes between:

```text
World truth
    What deterministically exists according to generation rules

Stored realization
    What the engine has actually materialized into explicit octree/voxel data
```

An untouched coordinate can have a deterministic answer without that answer being stored.

Conceptually:

```text
World = Generator + Persistent Modifications
```

rather than:

```text
World = Every voxel ever generated
```

---

# 3. Octree as the Primary Spatial Representation

Chunks/regions should themselves be octree-based rather than being dense chunks containing an optional octree.

The octree depth defines available spatial precision.

Deepest-level coordinates use `i128`.

Example:

```rust
struct VoxelCoord {
    x: i128,
    y: i128,
    z: i128,
}
```

Shallower octree coordinates can be derived through bit shifting:

```rust
let ancestor_x = voxel_x >> levels;
let ancestor_y = voxel_y >> levels;
let ancestor_z = voxel_z >> levels;
```

Signed right shift provides the desired floor-like behavior for negative two's-complement coordinates.

The low bits can identify the child octant.

This gives the coordinate system a natural hierarchical relationship.

---

# 4. Why `i128`

`i128` provides an enormous coordinate range and allows:

- Exact global voxel coordinates
- Exact ancestor derivation
- Cheap bit shifting
- Stable deterministic world generation
- Extremely large worlds

Global world addressing should remain integer-based.

Rendering should convert to local/camera-relative coordinates rather than converting huge `i128` coordinates directly to `f64`, since `f64` cannot exactly represent all `i128` values.

---

# 5. Octree Node Representation

A node may use different representations depending on what is known and what is needed.

Current basic concepts:

```text
Octree Node
├── Implicit
├── Uniform
├── Probability
├── Children
└── Explicit / deepest voxel data
```

The exact enum/storage structure is not yet fixed.

---

# 6. Implicit Nodes

An implicit region has no locally materialized detail.

The world generator is sufficient to determine what exists there when queried.

Conceptually:

```text
Implicit
- no local voxel data
- consult generator if resolution becomes necessary
```

This should be the cheapest untouched-world state.

---

# 7. Uniform Nodes

A uniform node means the entire represented region contains the same voxel archetype/material.

Example:

```text
Uniform(Stone)
```

Large homogeneous volumes should collapse into a single node.

---

# 8. Probability Nodes

A probability node stores a statistical composition without fixing every voxel position.

Example:

```text
Probability
- Stone: 0.70
- Dirt:  0.20
- Iron:  0.10
```

The exact voxel arrangement is deferred until the region must be resolved.

Generation must be deterministic:

```text
same world seed
+ same node coordinate
+ same depth
+ same rules
= same realized result
```

This allows probabilistic regions to remain cheap until needed.

---

# 9. Probability Uniformity Formula

If a node contains `N` independent samples and voxel type `i` has probability `p_i`, the probability that the entire node is uniform is:

```text
P(uniform) = Σ p_i^N
```

For an immediate octree subdivision with eight children:

```text
P(uniform children) = Σ p_i^8
```

If the node is known to be uniform, the conditional probability that the uniform type is `i` is:

```text
P(i | uniform) = p_i^N / Σ p_j^N
```

This can allow probabilistic nodes to collapse without explicitly generating every descendant.

---

# 10. Children

A node subdivides into eight children only when more spatial detail is required.

The **generation API should not expose `Subdivide` as a creator action**.

Subdivision is an engine concern.

The generator describes what should exist.

The octree decides how deeply it must refine to represent that answer.

---

# 11. When the Engine Should Refine

Potential refinement triggers include:

- Region can become visible to the player
- Region borders exposed air/open space
- Player digs/mines into the region
- Ray query requires precise information
- Collision requires precise information
- Nearby simulation requires precise information
- Renderer needs finer LOD
- A generator cannot prove a coarse node result
- A mutation requires explicit local state

The guiding rule is:

> **Generation follows demand, not mere proximity.**

A player walking over a mountain should not generate kilometers of detailed rock beneath them.

---

# 12. Exposure-Driven Generation

One major intended optimization is to resolve voxel data primarily around visible/exposed boundaries.

Conceptually:

```text
abstract node
    ↓
does this region border potentially visible/open space?
    ↓
no  → remain abstract
yes → resolve only as deeply as necessary
```

When a player digs:

```text
player removes voxel
    ↓
new boundary becomes exposed
    ↓
neighboring unresolved region is queried
    ↓
generator resolves enough detail
    ↓
new visible surface appears
```

The interior remains unresolved.

---

# 13. Persistent Modifications

Untouched terrain can be regenerated from:

```text
world seed
+ generator rules
+ coordinates
```

Player/simulation modifications must be persisted.

Conceptually:

```text
Generated world truth
+
Persistent edits/deltas
=
Current world
```

Modified nodes may need to remain explicit or store sparse deltas.

---

# 14. Procedural Generator API Philosophy

The generator API should be designed from the **creator-facing feel first**.

The intended public experience is similar to a behavior tree:

- Readable
- Compositional
- Easy to assemble
- Easy to inspect visually
- Easy to extend

Under the hood, the engine may compile the tree into a DAG and then into a flat execution plan.

Conceptually:

```text
Tree-like creator API
        ↓
DAG
        ↓
Optimized/flat execution plan
```

---

# 15. Why DAG Under the Hood

A DAG allows multiple generator rules to reuse the same inputs.

Example:

```text
Noise("continental")
        ↓
used by terrain rule
used by biome rule
used by moisture rule
```

Without DAG sharing, the same expensive source might be recomputed multiple times.

A DAG allows:

- Shared noise calculations
- Shared height calculations
- Shared random values
- Common-subexpression reuse
- Dead-rule elimination
- Constant folding
- Cached intermediate values
- Potential batching/SIMD later

The user can still see and author a tree-like structure.

---

# 16. Generator Node Categories

Current conceptual categories:

```text
Generator Node
├── Composite
├── Decorator
├── Condition
├── Source
└── Leaf / Output
```

---

# 17. Composite Nodes

Composites combine multiple child nodes.

## 17.1 Sequence

Evaluate children in order.

Conceptually:

```text
Sequence
├── Rule A
├── Rule B
└── Rule C
```

Possible semantics:

```text
apply A
then apply B
then apply C
```

Exact result propagation rules are still open.

---

## 17.2 Selector / FirstMatch

Evaluate children in order and use the first matching/successful result.

Example:

```text
FirstMatch
├── if AboveSurface → Air
├── if Cave         → Air
├── if Ore          → Ore
└── otherwise       → Stone
```

---

## 17.3 All

All child conditions must match.

---

## 17.4 Any

At least one child condition must match.

---

# 18. Decorators

Decorators wrap or modify one child.

Potential decorators:

```text
Not
When
Chance
TransformResult
```

Repeat was mentioned conceptually from behavior-tree terminology, but may not make sense for generation and should only be added if needed.

---

# 19. Conditions

Conditions evaluate predicates.

Initial useful conditions:

```text
GreaterThan
LessThan
Between
Equal
HasTag
```

Conditions should support regional reasoning where possible.

Instead of only:

```text
true / false
```

coarse generation may use:

```rust
enum ConditionState {
    AlwaysTrue,
    AlwaysFalse,
    Mixed,
    Unknown,
}
```

This lets the octree determine whether subdivision is required.

---

# 20. Sources

Sources produce reusable values.

Potential standard sources:

```text
Constant
Position
Height
Random
Noise
Distance
ParentValue
Custom
```

These should be DAG-shareable.

---

# 21. Leaves / Outputs

Leaves produce generation results.

Current basic results include:

```text
Uniform
Probability
NoChange / Continue
Mixed / unresolved
```

`Subdivide` is intentionally excluded from the creator-facing grammar.

The engine subdivides when the current node cannot be represented sufficiently at its current depth.

---

# 22. Conceptual Generator Grammar

A simple generator may look like:

```rust
let terrain = sequence([
    first_match([
        when(above_surface(), uniform(AIR)),
        when(cave_noise(), uniform(AIR)),
        otherwise(uniform(STONE)),
    ]),

    when(
        ore_noise(),
        probability([
            (IRON_ORE, 0.8),
            (STONE, 0.2),
        ]),
    ),
]);
```

The exact Rust syntax is not fixed.

The desired feel is:

```text
Readable decision process
+
Composable standard nodes
+
Custom extension points
```

---

# 23. Example Tree Form

```text
Sequence
├── FirstMatch
│   ├── When(AboveSurface)
│   │   └── Uniform(Air)
│   ├── When(CaveNoise)
│   │   └── Uniform(Air)
│   └── Otherwise
│       └── Uniform(Stone)
│
└── When(OreNoise)
    └── Probability
        ├── IronOre: 0.8
        └── Stone:   0.2
```

---

# 24. Custom Generator Components

Advanced creators should be able to write custom generator nodes.

Conceptually:

```rust
trait GeneratorComponent {
    fn evaluate(
        &self,
        context: &GenerationContext,
    ) -> GenerationValue;
}
```

Optional regional analysis:

```rust
trait GeneratorComponent {
    fn evaluate(
        &self,
        context: &GenerationContext,
    ) -> GenerationValue;

    fn evaluate_region(
        &self,
        region: &GenerationRegion,
    ) -> RegionEvaluation {
        RegionEvaluation::Unknown
    }
}
```

Custom nodes should plug into the same tree/DAG as built-ins.

Example:

```text
Noise
    ↓
CustomGeologicalWarp
    ↓
Between(...)
    ↓
Probability(...)
```

The goal is:

> Beginners compose built-ins.  
> Intermediate users build complex trees.  
> Advanced users write new generator nodes.

---

# 25. Regional Evaluation

For lazy octree generation to be efficient, generators should ideally answer questions about **regions**, not only points.

A point sampler:

```rust
fn sample(position: Position) -> Value;
```

is not enough to cheaply prove that a large octree node is uniform.

Where possible, sources should provide conservative regional information:

```rust
fn evaluate_region(region: Region) -> RegionEvaluation;
```

Possible responses:

```rust
enum RegionEvaluation<T> {
    Uniform(T),
    Bounds(ValueRange),
    Mixed,
    Unknown,
}
```

The bounds do not need to be exact.

They only need to safely contain all possible values.

---

# 26. Range Propagation

Some generator components can propagate ranges cheaply.

Example:

```text
Constant(x)
→ [x, x]
```

```text
Add([a,b], [c,d])
→ [a+c, b+d]
```

```text
Multiply
→ min/max of the four endpoint products
```

```text
Clamp
→ clamped interval
```

```text
Height
→ node minimum Y to maximum Y
```

A condition can then determine:

```text
AlwaysTrue
AlwaysFalse
Mixed
```

without evaluating every voxel.

---

# 27. Noise Is the Difficult Case

Ordinary noise functions generally cannot provide exact min/max values over a 3D region without extensive sampling.

Sampling every point would defeat the purpose of lazy generation.

Therefore Voxel World may benefit from a custom noise system designed specifically around the octree.

---

# 28. Octree-Aligned Noise

The proposed custom noise system is Perlin-like but spatially aligned with octree levels.

Conceptually:

```text
Octree level -3 ↔ Noise level -3
Octree level -2 ↔ Noise level -2
Octree level -1 ↔ Noise level -1
Octree level  0 ↔ Noise level  0
Octree level  1 ↔ Noise level  1
Octree level  2 ↔ Noise level  2
...
```

Each level represents a power-of-two spatial scale.

This means noise frequency naturally matches the octree hierarchy.

---

# 29. Integer Noise Levels

Noise frequency is intentionally restricted to power-of-two level changes.

Example:

```text
Level -3 → 8× larger spatial scale
Level -2 → 4× larger
Level -1 → 2× larger
Level  0 → root scale
Level  1 → half scale
Level  2 → quarter scale
Level  3 → eighth scale
```

Negative levels allow structures larger than the root octree lattice.

This sacrifices arbitrary frequencies such as `1.37×`, but provides:

- Exact octree alignment
- Cheap coordinate derivation
- Easier bounds
- Deterministic hierarchy
- Simpler caching
- Natural lazy generation

This is currently considered a worthwhile tradeoff.

---

# 30. Hierarchical Noise Contributions

Noise can be defined as a sum of bounded contributions:

```text
Noise(position) =
    level_0(position)
  + level_1(position)
  + level_2(position)
  + ...
```

Each level has a known maximum amplitude:

```text
-A[level] <= level_n(position) <= +A[level]
```

If amplitudes decrease geometrically:

```text
A_n = A_0 * persistence^n
```

then the maximum remaining contribution below some depth can be computed analytically.

For persistence `p`:

```text
Remaining after depth d
= A_0 * p^(d+1) / (1 - p)
```

for `0 < p < 1`.

This gives a conservative range for unresolved deeper levels.

---

# 31. Noise Threshold Example

Suppose the already-known contribution at a node is:

```text
0.72
```

and all unresolved deeper levels together can contribute at most:

```text
±0.125
```

Then the final value must lie inside:

```text
[0.595, 0.845]
```

If the cave threshold is:

```text
0.40
```

the threshold cannot be crossed.

The node can remain coarse.

If instead:

```text
known contribution = 0.49
remaining          = ±0.03125
```

then:

```text
[0.45875, 0.52125]
```

crosses a threshold of `0.50`.

The engine must descend further.

---

# 32. Gradient Generation

The proposed noise may use Perlin-like gradient vectors.

Instead of a small fixed set of gradients, Voxel World may generate deterministic directions over the unit sphere.

The gradient for a lattice position should be derived from:

```text
world seed
+ noise level
+ x
+ y
+ z
```

using a deterministic stateless permutation/hash.

No persistent RNG object is required for each lattice point.

---

# 33. Stateless Coordinate-Derived Randomness

A coordinate-derived pseudo-random function is preferred for procedural lattice values.

Conceptually:

```text
(seed, level, x, y, z, attempt)
        ↓
bit permutation / hash
        ↓
pseudo-random bits
```

Requirements:

- Deterministic
- Repeatable
- Cheap
- Good avalanche/decorrelation
- Nearby coordinates should produce unrelated-looking outputs

Caching is optional.

For cheap permutation logic, recomputation may be faster than hash-map caching.

---

# 34. Random Unit Sphere Gradient

A trig-free unit-sphere gradient may be generated with the Marsaglia method.

Conceptually:

```text
u ∈ [-1, 1]
v ∈ [-1, 1]

s = u² + v²

reject if s >= 1

x = 2u * sqrt(1 - s)
y = 2v * sqrt(1 - s)
z = 1 - 2s
```

This generates directions uniformly over the sphere without requiring `sin` or `cos`.

The rejection loop can use deterministic coordinate-derived attempts.

---

# 35. Why Not Scale `u` and `v` Smaller?

Restricting:

```text
u, v ∈ [-k, k]
```

with `k < 1` breaks the uniform sphere distribution.

It biases directions toward one part of the sphere.

Therefore the full method with rejection should be used if uniform directional distribution matters.

---

# 36. Noise / Octree Integration Goal

The defining advantage of the custom noise system is:

> The noise hierarchy speaks the same spatial language as the octree hierarchy.

This allows generation systems to reason about:

```text
known coarse contribution
+
maximum unresolved fine contribution
```

and potentially prove that a region cannot cross a threshold.

That lets the engine stop refinement early.

---

# 37. Generator Execution Model

The public generator may look like a tree.

Internally:

```text
Creator Tree
    ↓
Normalize
    ↓
DAG
    ↓
Deduplicate shared sources
    ↓
Optimize
    ↓
Flat execution plan
```

Potential optimizations:

- Shared source reuse
- Constant folding
- Dead branch removal
- Cached regional evaluations
- Cached noise samples
- Short-circuit conditions
- Batch evaluation
- SIMD later
- GPU evaluation later if useful

The friendly API must not imply slow execution.

---

# 38. Generator Result Philosophy

The generator should answer:

> **What should exist here?**

The octree should answer:

> **How deeply must I resolve this region?**

The renderer should answer:

> **How much detail do I need to display?**

Simulation should answer:

> **How much detail do I need to update behavior?**

These responsibilities should remain separate.

---

# 39. Example Lazy Cave Generation

```text
Player approaches region
        ↓
Renderer needs surface detail
        ↓
Generator evaluates coarse node
        ↓
Cave noise bounds prove node solid
        ↓
Keep node collapsed
```

Another node:

```text
Generator evaluates coarse node
        ↓
Cave threshold may be crossed
        ↓
Engine subdivides
        ↓
Evaluate 8 children
        ↓
Some children prove solid
Some prove air
Some remain uncertain
        ↓
Only uncertain children descend further
```

---

# 40. Example Mining Flow

```text
Player mines exposed voxel
        ↓
Voxel is removed
        ↓
Previously hidden neighboring region is now exposed
        ↓
Generation system queries that region
        ↓
Octree resolves only enough detail to establish new surface
        ↓
Player sees newly exposed material
        ↓
Deeper interior remains unresolved
```

---

# 41. Generator API Checklist

## Public Tree API

- [ ] Define generator node trait/interface
- [ ] Define tree construction API
- [ ] Define `Sequence`
- [ ] Define `Selector` / `FirstMatch`
- [ ] Define `All`
- [ ] Define `Any`
- [ ] Define `When`
- [ ] Define `Not`
- [ ] Define `Otherwise`
- [ ] Define condition API
- [ ] Define source API
- [ ] Define output/leaf API
- [ ] Make API pleasant to read in Rust
- [ ] Make nested generator trees easy to inspect/debug

## Built-In Sources

- [ ] `Constant`
- [ ] `Position`
- [ ] `Height`
- [ ] `Random`
- [ ] `Noise`
- [ ] Consider `Distance`
- [ ] Consider `ParentValue`

## Built-In Conditions

- [ ] `GreaterThan`
- [ ] `LessThan`
- [ ] `Between`
- [ ] `Equal`
- [ ] `All`
- [ ] `Any`
- [ ] `Not`

## Built-In Outputs

- [ ] `Uniform`
- [ ] `Probability`
- [ ] `NoChange`
- [ ] Define `Mixed` / unresolved semantics

## Custom Generator Support

- [ ] Custom source registration
- [ ] Custom condition registration
- [ ] Custom decorator registration
- [ ] Custom output registration
- [ ] Custom composite registration if needed
- [ ] Optional regional-analysis API
- [ ] Versioning/serialization of custom generator nodes

---

# 42. DAG / Compilation Checklist

- [ ] Convert creator tree to DAG
- [ ] Deduplicate identical shared sources
- [ ] Assign stable internal node IDs
- [ ] Topologically order DAG
- [ ] Compile DAG into flat execution plan
- [ ] Implement short-circuiting
- [ ] Cache shared source results during evaluation
- [ ] Add constant folding
- [ ] Add dead-node elimination
- [ ] Add debug representation of compiled plan
- [ ] Benchmark tree authoring vs compiled execution overhead

---

# 43. Octree Checklist

- [ ] Define global `i128` voxel coordinates
- [ ] Define octree depth/level representation
- [ ] Define parent coordinate bit shifting
- [ ] Define child-octant extraction
- [ ] Verify negative-coordinate behavior
- [ ] Define `Implicit` node representation
- [ ] Define `Uniform` node representation
- [ ] Define `Probability` node representation
- [ ] Define `Children` node representation
- [ ] Define deepest explicit voxel representation
- [ ] Define refinement triggers
- [ ] Define collapse/recompression rules
- [ ] Define persistent-modification representation
- [ ] Define exposure-driven refinement
- [ ] Define ray-query-driven refinement
- [ ] Define collision-driven refinement
- [ ] Define simulation-driven refinement

---

# 44. Probability Checklist

- [ ] Define probability distribution representation
- [ ] Normalize/validate weights
- [ ] Deterministic realization from coordinate + seed
- [ ] Implement `Σ p_i^N` uniform probability
- [ ] Implement conditional uniform-type selection
- [ ] Decide when probability nodes may remain unresolved
- [ ] Decide when probability nodes must become explicit

---

# 45. Noise Checklist

- [ ] Define octree-aligned noise levels
- [ ] Support negative noise levels
- [ ] Define amplitude per level
- [ ] Define persistence
- [ ] Define deterministic gradient derivation
- [ ] Implement coordinate permutation/hash
- [ ] Verify avalanche/decorrelation
- [ ] Implement unit-sphere gradient generation
- [ ] Implement Perlin-like interpolation
- [ ] Define conservative remaining-amplitude bound
- [ ] Expose `sample(point)`
- [ ] Expose `sample_level(point, level)`
- [ ] Expose regional bound/uncertainty API
- [ ] Benchmark against existing noise crates
- [ ] Visualize noise for directional artifacts
- [ ] Test negative coordinates
- [ ] Test huge `i128` coordinates
- [ ] Verify deterministic results across runs/platforms where required

---

# 46. Rendering / Generation Integration Checklist

- [ ] Renderer can request octree detail by spatial need
- [ ] Renderer does not force full voxel realization
- [ ] Distant nodes render at coarse octree depth
- [ ] Exposed surfaces trigger local refinement
- [ ] Debug view for octree depth
- [ ] Debug view for unresolved vs resolved regions
- [ ] Debug view for uniform/probability nodes
- [ ] Debug view for generator decisions
- [ ] Debug view for noise bounds/thresholds
- [ ] Camera-relative rendering for huge coordinates

---

# 47. Persistence Checklist

- [ ] Store world seed
- [ ] Store generator version/configuration
- [ ] Store persistent modifications only where possible
- [ ] Store custom generator identity/version
- [ ] Decide how generator changes affect old worlds
- [ ] Version noise algorithm
- [ ] Version RNG/permutation behavior
- [ ] Preserve deterministic regeneration for existing worlds
- [ ] Define migration strategy for future generator revisions

---

# 48. Open Questions

- [ ] Exact internal enum/state layout for octree nodes
- [ ] Whether `Probability` should represent voxels, child nodes, or both
- [ ] How to represent mutable state inside otherwise abstract nodes
- [ ] How to collapse modified regions safely
- [ ] How much generation should happen for collision before visual exposure
- [ ] Whether structures need a separate deferred representation
- [ ] Whether biome generation uses the same DAG system
- [ ] Whether entity spawning uses the same DAG system
- [ ] How to expose generator graphs to tools/editors
- [ ] Whether the tree UI should eventually have a visual editor
- [ ] Exact semantics of `Sequence`
- [ ] Exact semantics of `NoChange`
- [ ] How generator outputs combine when multiple rules modify a node
- [ ] Whether regional bounds are stored/cached
- [ ] How long generated noise/intermediate caches live
- [ ] Whether custom generators can opt into SIMD/GPU execution later

---

# 49. Core Principles Summary

1. **Voxels should not be generated merely because they are nearby.**
2. **Unseen interiors should remain unresolved.**
3. **The octree is both storage hierarchy and spatial generation hierarchy.**
4. **The generator describes world truth; the octree decides resolution.**
5. **The renderer should consume only as much detail as needed.**
6. **Player interaction gradually materializes the world.**
7. **Untouched terrain should be reproducible from seed and rules.**
8. **Modified terrain must be persisted.**
9. **The public generator should feel tree-like and compositional.**
10. **The internal representation should be DAG/compiled for efficiency.**
11. **Custom generation nodes must be possible without replacing the whole system.**
12. **Noise should align with octree levels so coarse uniformity can sometimes be proven early.**
13. **Power-of-two spatial levels are an intentional constraint in exchange for efficiency and predictability.**
14. **Global coordinates remain exact integers; rendering becomes local/camera-relative.**
15. **Friendly creator APIs must not come at the cost of slow runtime execution.**
