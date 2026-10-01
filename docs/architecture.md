# Foundations and API migration

This is still one crate. Its modules follow domains rather than accumulating
unrelated code under `Misc`:

| Module | Responsibility |
| --- | --- |
| `math` | Vectors, matrices, arbitrary-width integers, rings and fields, validated rotations |
| `random` | Streams, seed derivation, validated distributions |
| `spatial` | Coordinates, octree levels, spatial noise |
| `time` | Time points, durations, frequencies, tick scheduling |
| `units` | Probabilities, ratios, weights, identifiers and scalar constraints |
| `structures` | Generic collections, indices, mappings and sampling |
| `world` | Voxel registry, storage, chunk generation |

`misc` retains compatibility re-exports, and `units` still re-exports spatial
quantities and time measures. Prefer the domain modules in new code. The physical
module directories are lowercase. The existing `engine` files remain placeholders;
this refactor does not implement rendering, streaming or a full simulation engine.

## Type boundaries

Raw numbers are accepted when constructing a value or validating a distribution.
Once validated, pass the domain value through the rest of the API.

| Boundary | New contract |
| --- | --- |
| `Seed::chance` | `Probability`, not an arbitrary `f64` |
| `Seed::at_voxel` / `at_depth` | `VoxelPosition3` / `Depth` |
| `Weights::index_for` | `UnitValue` in `[0, 1)` |
| Fuzzy-set memberships | `Probability`; aggregate membership remains `f64` |
| `Measured` trait | Separate `Measure` and `Total` associated types |
| Rotation, interpolation | `UnitQuaternion`; interpolation uses `Probability` |
| Time points vs. elapsed time | `Tick` vs. `TickDuration`; seconds use `Seconds` |
| Scheduled world events | `TickScheduler` separates deadlines from delays |
| Distribution parameters | Reusable constructors return `Option` on invalid input |

`Random` convenience distribution methods validate in release builds too and panic
on invalid parameters. Use constructors such as `Gamma::new` for fallible input and
reuse the result for repeated sampling. A finite parameter does not guarantee a
finite sample from an unbounded floating-point distribution.

Construction and conversion changes that callers must handle:

- `TreeDepth::new` now returns `Option`, rejecting floors above 127.
- `TreeDepth::shift_for` and `node_width` reject unsupported levels. Use `clamp`
  explicitly when saturation to the supported range is intended.
- `VoxelPosition::node_at` / `local_at`, and node `parent` / `ancestor` / `origin`
  return `Option` for unsupported levels or unrepresentable coordinates.
- `LocalPosition::new` validates; `LocalPosition::clamped` explicitly saturates.
- `TickRate::new` rejects zero. Pause a clock by not advancing it.
- `Zipf::new` and voxel registration return `Option` on invalid inputs.
- Build rotations with `UnitQuaternion::new` or `from_axis_angle`; raw
  `Quaternion` remains available for general algebra.

## Randomness and reproducibility

Geometric samples count **failures before the first success**. Zero probability
returns `u64::MAX`. Ratios above one are rejected. When
`floor(denominator / numerator) <= 256`, sampling uses exact integer trials.
Rarer events use the existing deterministic Q64 inverse approximation, not an
exact rational geometric law. Repeated fixed-point truncation introduces bias;
it becomes important near probabilities of `1 / u64::MAX`. The tests cover means
and survival frequencies through the expected `1 / 10^10` use case, but are not a
formal error bound or a general PRNG certification.

Use `GeometricRatio::one_in(n)` or `GeometricRatio::new(ratio)` to cache the Q64
power table when sampling the same probability repeatedly. The cached sampler
matches the one-shot sampler's outputs and stream consumption. The switch at 256
is a cost heuristic, not a mathematical threshold.

Uniform floats now stay in their promised half-open interval. Weighted collections
honor weights for both single and multiple choices. Distinct sampling shares one
partial Fisher–Yates implementation, using a sparse map for small samples and a
contiguous pool for dense ones. Indexable collections pick a single index in O(1).

`Seed` is an immutable 128-bit identity and derivation key. A one-shot
`Seed::sample` whitens the seed and is repeatable; it never masquerades as an
advancing source. Use `SeedCursor` when a small owner needs successive draws,
or `Random` for a full xoshiro256** stream. `RandomSource` is the sampler-facing
word source, while `DrawSource` is the owner-facing trait for state that advances
between calls. `PortableDistribution` lets persistent generation require a
sampler whose implementation is bit-for-bit portable.

All 128 seed bits now reach the xoshiro state without first being folded to 64
bits. `SeedCursor` walks a full-cycle 128-bit Weyl sequence and whitens every
position before folding it to a word. `Seed::try_entropy` uses the operating
system; `Seed::entropy` retains a documented best-effort fallback, and neither
makes the simulation PRNG cryptographically secure. Low-level mixing remains a
crate-private implementation detail rather than part of the public random API.

`RandomState` is the fixed-layout, versioned snapshot for exact stream restore;
it includes the cached spare normal as well as all four xoshiro words. A
`SeedCursor` can likewise be encoded from its origin and current position.
Persist `SEED_ALGORITHM_VERSION` with generated-world metadata, and use the
version already embedded in `RandomState` for saved streams. Fixed-vector tests
guard both algorithms against accidental changes.

An unordered collection's iteration order is **not** a portable identity, even
with a fixed hasher. Seed world content by stable names and global coordinates;
do not seed it from hash-table traversal or registration slots. The example
`ChunkGenerator` sorts its palette by registered name, then derives each voxel
from its coordinate and a named seed domain. Tests compare materials by name
across registry orders and chunk visitation orders. `VoxelTypeId` is a local
registry handle, not a serialized material identity.

The corrected float conversions, selection policies, unsupported-depth handling,
noise arithmetic, and revised seed expansion can change results created by older
builds for a given seed. The random formats are versioned now, but the complete
world generator, palette, and save schema still need their own compatibility
version before worlds can be promised stable across releases. Integer-only paths
avoid platform transcendental differences; samplers using `ln`, `exp`, `powf` or
trigonometry do not promise cross-platform bit-for-bit equality. This PRNG is for
simulation, not cryptography.

## Connected example and checks

`cargo run` registers named voxel types, generates a 16³ dense chunk and queries
its origin. This small workflow exercises the boundaries without committing to
the eventual streaming architecture. Octree children now share one allocation
per group of eight instead of eight separate child allocations.

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo test --release
cargo doc --no-deps
cargo bench --bench sampling
```

The tests include deterministic edge cases, fixed-seed distribution sanity checks,
and compile-fail examples for invalid type combinations. The dependency-free
benchmark reports medians over seven trials for cached geometric sampling,
indexed selection and sparse distinct sampling. It includes simple reference
implementations for the replaced collection strategies; timings are local
microbenchmarks, not end-to-end engine speedups.
