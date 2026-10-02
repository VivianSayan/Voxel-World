# Structure reference

This is the project-level guide to the reusable data structures under
[`src/structures`](../src/structures). It is written for both people and coding
assistants: start with the decision tables, then open the relevant entry rather
than scanning every implementation.

The source and its Rustdoc remain authoritative when exact signatures matter.
Keep this document synchronized when a public structure, alias, important
invariant, or capability trait changes.

## Imports and terminology

Common structures and capability traits are available through:

```rust
use voxel_world::structures::prelude::*;
```

For application code, explicit short paths are often clearer:

```rust
use voxel_world::structures::{OrderedSet, PropertyQuery, SlotMap};
use voxel_world::structures::traits::{Collection, Pending};
```

Other public types remain available from their family module. Scheduling has a
dedicated home under `structures::scheduling`; the former
`collections::sequences` and `collections::rota` paths continue to work.
The broad `structures::prelude` is convenient for experiments and examples.

In this document:

- **dense** means every index from `0..len()` exists;
- **sparse** means integer indices may have gaps;
- **unique** means a logical element cannot occur twice;
- **canonical order** means equal logical contents iterate identically,
  regardless of insertion history;
- **deterministic order** is the weaker promise that the same operations replay
  within one executable/data-format version;
- ranges and intervals are half-open (`start..end`) unless stated otherwise.

Hash-backed structures use the project's fixed, fast hasher. It is suitable for
trusted game data, but it is not collision-attack resistant and hash iteration
is not a portable save format. For seeded world generation, prefer canonical
structures or sort/canonicalize values before consuming randomness.

## Fast selection guide

### Collection or sequence

| Need | Prefer | Why |
|---|---|---|
| Unique values, no meaningful order | `Set<T>` | General-purpose O(1)-expected membership |
| Dense non-negative integer membership | `BitSet` | One bit per possible member |
| Unique values with explicit order | `OrderedSet<T>` | Vector order plus O(1)-expected lookup |
| Unique values with unique labels and order | `LabeledOrderedSet<T, L>` | Ordered set plus bidirectional labels |
| Last `n` distinct values | `BoundedOrderedSet<T>` | FIFO eviction with deduplication |
| Last `n` values, duplicates allowed | `RingBuffer<T>` | Fixed-capacity deque |
| Work ordered by arbitrary priority | `PriorityQueue<T, P>` | Stable min-priority queue |
| Work ordered by a small integer range | `BucketQueue<T>` | Cheap bucketed priorities |
| Values due at future steps, duplicates allowed | `Scheduler<T>` | Sparse absolute-time buckets |
| Work spread evenly over a fixed number of turns | `Rota<T, GROUPS>` | Balanced groups, constant-time membership |
| The same, with repeats | `MultiRota<T, GROUPS>` | Each copy takes its own turn |
| The same, order preserved within and across turns | `OrderedRota<T, GROUPS>` | Groups are subsequences of the whole |
| The same, order preserved with repeats | `OrderedMultiRota<T, GROUPS>` | Ordered queue split into even shares |
| One pending due time per value | `UniqueScheduler<T>` | Scheduling again moves the value |
| Recurring work on a drawn interval | `StochasticScheduler<T, R>` | Re-arms itself from a mean time to happen |
| The same, one pending occurrence per value | `UniqueStochasticScheduler<T, R>` | Scheduling again re-rolls the value |
| Dense values with long equal runs | `RunLengthSequence<T>` | Stores maximal adjacent runs |
| Dense values with few distinct values | `Palette<T>` | Stores packed palette indices |
| Dense values whose pattern changes | `CompactSequence<T>` | Chooses uniform/palette/runs/dense |
| Sparse signed indices, one value per index | `SparseSequence<T>` | Ordered gaps and reverse value index |
| Sparse signed indices, many values per index | `SparseSetSequence<T>` | A set at each occupied index |
| Counts per distinct value | `MultiSet<T>` | Integer multiplicity |
| Arbitrary non-negative importance | `WeightedSet<T>` | Floating-point measure and weighted choice |
| Graded membership in `[0, 1]` | `FuzzySet<T>` | Probability-like fuzzy-set operations |
| Connected components that only merge | `DisjointSets<T>` | Union-find with path compression |
| Unique values queried by derived labels | `LabelIndexedSet<T, L, F>` | Maintains secondary label groups |
| Subscribers notified through one callback | `SubscriptionSet<S, F>` | Unique subscriber registry |

### Mapping or index

| Relationship | Prefer | Reverse lookup |
|---|---|---|
| One key to one value | `FastHashMap<K, V>` | No |
| One-to-one | `BiMap<L, R>` | Yes, unique both ways |
| One key to many values; each value has one owner | `UniqueMultiMap<K, V>` | Yes, one key |
| Many-to-many | `MultiMap<K, V>` | Yes, many keys |
| Each element has exactly one shared label | `GroupedSingleMap<E, L>` | Label group plus group sizes |
| Each element has any number of shared labels | `GroupedMultiMap<E, L>` | Label groups plus group sizes |
| Symmetric exclusive pairs in one domain | `PairMap<T>` | `partner(value)` |
| Non-overlapping spans | `IntervalMap<K, V>` | Point and overlap queries |
| Bounded recency cache | `LruCache<K, V>` | No; tracks least-recently-used key |
| A set is the key | `SetKeyMap<K, V>` | No |
| Text used as an identity, compared constantly | `Name` | Global registry, for reading a name back |
| Several maps read as one, in precedence order | `LayeredMap<K, V>` | `find_layer(key)` gives the layer a value came from |
| Hierarchical typed tags | `TagIndex` | Tree and cross-indices |
| Schema-aware property expressions | `PropertyQuery<E, P, V>` | Cached inverted indices |
| The same, released when a condition comes true | `ConditionIndex<E, P, V>` | Property to the conditions that read it |
| The same, asked when work comes up rather than watched | `Gate<E, P, V>` | Element to the condition it is held to |

### Storage or coordinates

| Need | Prefer |
|---|---|
| Stable typed handles with stale-handle detection | `SlotMap<K, T>` |
| Checked dense rectangular coordinates | `Grid<T, N>` / `Grid2` / `Grid3` / `Grid4` |
| Checked boolean rectangular coordinates | `BitGrid<N>` / `BitGrid2` / `BitGrid3` / `BitGrid4` |
| Few distinct values at arbitrary positions | `Palette<T>` |
| Long adjacent equal values | `RunLengthSequence<T>` |
| Automatically selected dense compression | `CompactSequence<T>` |

## Sets and membership structures

### `Set<T>`

**Purpose.** The default unordered unique collection. It wraps
`FastHashSet<T>` and supports borrowed lookup, replacement, set algebra,
retention, draining, capacity control, random choice, and Cartesian products.

**Choose it when:** identity/membership matters and ordering does not. Typical
uses include dirty chunks, active entity IDs, visited nodes, and unique tags.

**Avoid it when:** values form a dense integer domain (`BitSet`), require a
stable content-defined order, or must retain insertion order (`OrderedSet`).

**Capabilities.** `Collection`, mutable collection traits, `UniqueCollection`,
`Choose`, `SetAlgebra`, `Capacity`, `DeterministicOrder`, and content hashing
when the element implements `StableHash`.

```rust
use voxel_world::structures::collections::Set;

let mut dirty = Set::new();
assert!(dirty.insert([2, 4, 8]));
assert!(!dirty.insert([2, 4, 8]));
assert!(dirty.contains(&[2, 4, 8]));
dirty.remove(&[2, 4, 8]);
```

### `NestedSet<T>`

**Purpose.** A hash-array-mapped-trie set with the ordinary unique-set
operations and an observable `depth()`. It is useful when the trie-shaped
representation itself is relevant to experiments or structural sharing work.

**Choose it when:** HAMT depth/layout is part of the problem. For normal unique
membership, prefer `Set<T>`, whose intent and performance are more obvious.

**Capabilities.** Collection insertion/removal, uniqueness, choice, set
algebra, deterministic traversal, and content hashing.

`nested_set::Iter<'_, T>` is the concrete borrowing iterator returned by
`iter()`. Callers normally use it through `impl Iterator` behavior rather than
constructing or naming it.

```rust
use voxel_world::structures::collections::NestedSet;

let mut ids = NestedSet::new();
ids.insert(10);
ids.insert(20);
assert!(ids.contains(&10));
let trie_depth = ids.depth();
```

### `BitSet`

**Purpose.** A set of non-negative `usize` members backed by `u64` words.
Iteration is increasing, and union/intersection/difference operate word-wise.
Storage grows to the highest inserted member, not to the number of members.

**Choose it when:** the possible IDs are reasonably dense and bounded—for
example occupancy masks, visited voxel indices, feature masks, or component
presence. Avoid it for a few enormous IDs because all intervening words exist.

**Capabilities.** `ValueCollection`, `ValueSetAlgebra`, canonical order,
deterministic order, random member choice, raw word access, bit-capacity
management, and content hashing.

```rust
use voxel_world::structures::collections::BitSet;

let mut visible = BitSet::new();
visible.insert(65);
assert_eq!(visible.word(1), 0b10);
assert_eq!(visible.iter().collect::<Vec<_>>(), vec![65]);
```

### `DisjointSets<T>` and `DisjointGroup<'_, T>`

**Purpose.** A union-find structure for a partition that only coalesces. It
uses union by size and path compression, making repeated joins and connectivity
checks effectively constant time. `groups()` materializes the complete current
partition; `DisjointGroup` is the borrowed group view exposed by `Grouping`.

**Choose it when:** discovering connected cave regions, mesh islands, networks,
or graph components. It cannot split a component after a join; rebuild if
deletions must break connectivity.

**Capabilities.** Collection insertion, `Grouping`, capacity control,
representative/group-size queries, deterministic traversal, and
partition-aware content hashing.

`disjoint_sets::DisjointGroup<'_, T>` is the read-only iterator returned by the
`Grouping` trait for one component. It is a view, not a separately owned group.

```rust
use voxel_world::structures::collections::DisjointSets;

let mut regions = DisjointSets::new();
regions.join("north", "west"); // Inserts missing elements too.
assert!(regions.joined(&"north", &"west"));
assert_eq!(regions.group_size(&"north"), 2);
```

### `LabelIndexedSet<T, L, F>`

**Purpose.** A unique set plus secondary groups derived by a labeler callback.
Each item may produce zero, one, or many labels. It supports any/all-label
queries and keeps label group-size indices for largest/smallest group queries.

**Choose it when:** an item owns its classification and queries repeatedly ask
for items with categories, components, states, or tags. If labels are primary
stored data rather than derived data, use a grouped map instead.

**Invariant.** Anything used by the labeler must not mutate while the item is
stored. Use `replace` after changing an item, or `reindex_all` after an external
change, otherwise the secondary index becomes stale.

**Capabilities.** Unique collection operations, choice, grouping, group sizes,
set algebra, and deterministic traversal.

```rust
use voxel_world::structures::collections::LabelIndexedSet;

let mut words = LabelIndexedSet::new(|word: &&str, emit: &mut dyn FnMut(usize)| {
    emit(word.len());
    if word.starts_with('r') {
        emit(100); // An additional derived category.
    }
});
words.insert("rock");
words.insert("sand");
let four_letters = words.with_any_label(&[4]);
assert_eq!(four_letters.len(), 2);
```

### `SubscriptionSet<S, F>`

**Purpose.** A unique subscriber registry paired with one callback. `notify`
uses an immutable callback; `notify_mut` permits the callback to mutate state it
captures. Subscribers themselves are passed by shared reference in both cases.

**Choose it when:** an owning object publishes one kind of event to registered
listeners without needing a full event bus. Avoid callbacks that structurally
modify the same set during notification; queue those changes for afterward.

**Capabilities.** Unique collection insertion/removal, choice, retention,
subscriber iteration, and deterministic traversal.

```rust
use voxel_world::structures::collections::SubscriptionSet;

let mut deliveries = 0;
{
    let mut watchers = SubscriptionSet::new(|subscriber: &i32, amount: &i32| {
        deliveries += subscriber * amount;
    });
    watchers.insert(1);
    watchers.insert(10);
    watchers.notify_mut(&5);
}
assert_eq!(deliveries, 55);
```

## Ordered, priority, and time sequences

### `OrderedSet<T>` and `WorkQueue<T>`

**Purpose.** A unique sequence implemented as a `Vec` plus a reverse hash
index. Membership is O(1) expected; indexing is O(1); stable insertion/removal
at arbitrary positions is O(n). `WorkQueue<T>` is an intent-revealing alias.

**Choose it when:** both uniqueness and explicit order matter—tool selections,
ordered registries, frontier work without duplicates, or deterministic
processing order. Use `swap_remove_at` when order does not need preserving.

**Capabilities.** Collection and sequence mutation, uniqueness, insertion at a
position, reorder/sort/shuffle, set algebra, capacity control, canonical order,
choice, and content hashing.

```rust
use voxel_world::structures::collections::OrderedSet;

let mut work = OrderedSet::new();
work.push("mesh");
work.push("upload");
work.push("mesh"); // Duplicate: no change.
assert_eq!(work.pop_first(), Some("mesh"));
```

### `LabeledOrderedSet<T, L>` and `KeyedOrderedSet<T, K>`

**Purpose.** An `OrderedSet<T>` with a globally unique label for every item.
It supports lookup/removal by item, label, or position while retaining explicit
order. `KeyedOrderedSet<T, K>` is the same type with a key-oriented name.

**Choose it when:** an ordered registry also needs stable names or keys—for
example render passes, layers, tools, or ordered systems. Both items and labels
must be unique.

**Capabilities.** Collection/sequence operations, reordering, bidirectional
map traits, unique reverse values, and canonical order.

```rust
use voxel_world::structures::collections::KeyedOrderedSet;

let mut passes = KeyedOrderedSet::new();
assert!(passes.push("shadow pass", "shadow"));
assert!(passes.push("world pass", "world"));
assert_eq!(passes.get_by_label(&"world"), Some(&"world pass"));
```

### `BoundedOrderedSet<T>`

**Purpose.** A fixed-limit, insertion-ordered unique set. Inserting a new value
at the limit evicts the oldest value; reinserting an existing value is a no-op.
A zero-limit set immediately returns every inserted value as the eviction.

**Choose it when:** retaining the last `n` distinct IDs, suppressing recent
duplicates, or implementing a small FIFO uniqueness window. Use `LruCache` if
successful reads should also refresh recency.

**Capabilities.** `Bounded`, `EvictingInsert`, `FixedCapacity`, collection
removal, sequence access, and uniqueness.

```rust
use voxel_world::structures::collections::BoundedOrderedSet;

let mut recent = BoundedOrderedSet::new(2);
assert_eq!(recent.insert(10), None);
assert_eq!(recent.insert(20), None);
assert_eq!(recent.insert(30), Some(10));
```

### `RingBuffer<T>`

**Purpose.** A fixed-capacity deque. Pushing at the back evicts the front;
pushing at the front evicts the back. Duplicates are allowed. Construction
requires a non-zero capacity.

**Choose it when:** retaining rolling samples, command history, recent log
entries, or a fixed simulation window. Use `BoundedOrderedSet` if duplicates
must be suppressed.

**Capabilities.** Dense sequence access/mutation, reorder operations, bounded
evicting insertion, random choice/removal, canonical order, and content hash.

```rust
use voxel_world::structures::collections::RingBuffer;

let mut history = RingBuffer::new(3);
history.push_back(1);
history.push_back(2);
history.push_back(3);
assert_eq!(history.push_back(4), Some(1));
assert_eq!(history.get_from_back(0), Some(&4));
```

### `PriorityQueue<T, P = i64>`

**Purpose.** A stable min-priority queue of unique items. Lower priorities are
served first and equal priorities retain insertion order. Changing a priority
does not duplicate an item. Push, remove, and priority changes are O(log n).

**Choose it when:** priorities are arbitrary or ordered values, as in path
search, streaming decisions, and job scheduling. For a small dense `usize`
priority domain, `BucketQueue` is normally cheaper.

**Capabilities.** Unique collection operations, sequence-style ordered access,
item-to-priority map access, priority range queries, `PriorityQueueLike`,
choice, reorder helpers, and canonical order.

```rust
use voxel_world::structures::collections::PriorityQueue;

let mut jobs = PriorityQueue::new();
jobs.push("far chunk", 20);
jobs.push("near chunk", 2);
assert_eq!(jobs.pop_first(), Some(("near chunk", 2)));
```

### `BucketQueue<T>`

**Purpose.** A stable min-priority queue whose priorities are `usize` bucket
indices in `0..levels`. Each bucket is FIFO. `push` returns `false` for an
out-of-range priority. Duplicates are allowed.

**Choose it when:** the priority range is small and known, such as light levels,
distance bands, or bounded flood-fill costs. Avoid thousands of mostly empty
buckets or priorities that require arbitrary ordered values.

**Capabilities.** Collection-like iteration, `PriorityQueueLike`, per-bucket
inspection/reservation, canonical order, and content hash.

```rust
use voxel_world::structures::collections::BucketQueue;

let mut light = BucketQueue::new(16);
assert!(light.push(12, [1, 2, 3]));
assert!(light.push(3, [4, 5, 6]));
assert_eq!(light.pop(), Some((3, [4, 5, 6])));
```

### `Scheduler<T>`

**Purpose.** Values grouped at sparse absolute `u64` steps. Multiple equal
values may be scheduled, including at one step. It can advance, jump to the
next occupied step, drain due values, cancel matching values, or move an entire
step's work.

**Choose it when:** simulation events, retries, delayed reactions, or turn
events are sparse in time and duplicates are meaningful. The current step is
monotonic during normal advancement.

**Capabilities.** Absolute and delayed scheduling, step/range inspection,
bulk scheduling, retention/cancellation, `RangeQuery`, and canonical order.

```rust
use voxel_world::structures::collections::Scheduler;

let mut ticks = Scheduler::starting_at(100);
ticks.schedule(2, "grow");       // Due at 102.
ticks.schedule_at(105, "decay");
assert!(ticks.advance_by(1).is_empty());
assert_eq!(ticks.advance(), vec!["grow"]);
```

### `UniqueScheduler<T>`

**Purpose.** A scheduler in which each value has at most one pending due step.
Scheduling an existing value moves it and returns its former step. Due steps
remain ordered; values at one step retain their bucket order.

**Choose it when:** repeated requests should debounce or replace prior work,
such as “remesh this chunk once,” cooldown expiry, or one pending retry per ID.

**Capabilities.** Delayed/absolute scheduling, cancellation, ordered iteration,
advancement, reverse `step_of` lookup, and due-step range queries.

```rust
use voxel_world::structures::collections::UniqueScheduler;

let mut remesh = UniqueScheduler::new();
assert_eq!(remesh.schedule(5, 42), None);
assert_eq!(remesh.schedule(2, 42), Some(5));
assert_eq!(remesh.step_of(&42), Some(2));
```

### `RunLengthSequence<T>`

**Purpose.** A dense sequence compressed into maximal adjacent equal runs.
Point and range writes split or merge runs as needed. Point lookup is O(log r),
where `r` is the number of runs; iteration can expose runs or logical values.

**Choose it when:** values form long horizontal/vertical bands, homogeneous
columns, or interval-like states. Avoid it for noisy data or many scattered
point writes, where run count approaches sequence length.

**Capabilities.** Dense `Sequence`, overlapping `RangeQuery`, run-capacity
management, canonical order, uniformity checks, and content hash.

```rust
use voxel_world::structures::collections::RunLengthSequence;

let mut column = RunLengthSequence::filled(256, 0u16);
column.fill(60, 72, 3);
assert_eq!(column.get(65), Some(&3));
assert_eq!(column.run_count(), 3);
```

## Rotas: evenly divided turns

A rota holds a population and keeps it split into `GROUPS` shares whose sizes
never differ by more than one. Take one share per turn and every member is
covered exactly once per round, with no turn carrying noticeably more than
another. `GROUPS` is a const generic, so the number of turns is fixed at compile
time and a rota over eight ticks is a `Rota<T, 8>`.

Nothing in the family mentions ticks: a group is whatever the caller decides a
turn is. Spreading periodic work is the motivating case, but the structure only
promises the even division.

|  | Unique members | Repeats allowed |
|---|---|---|
| Order does not matter | `Rota<T, GROUPS>` | `MultiRota<T, GROUPS>` |
| Order kept, within a group and overall | `OrderedRota<T, GROUPS>` | `OrderedMultiRota<T, GROUPS>` |

**The shared invariant.** Insertion goes to a currently smallest group.
Removal that would open a gap of two moves one member from the fullest group to
close it, so **a member does not always stay in the group it first landed in**.
Read `group_of` again rather than caching it. All four implement
`BalancedPartition`, so `is_balanced()` is true whenever it is asked.

### `Rota<T, const GROUPS: usize>`

**Purpose.** Unique members divided into `GROUPS` near-equal groups, with no
order kept inside a group. Insertion, removal, membership, and `group_of` are
constant time apart from an `O(GROUPS)` scan over group counts; removal swaps
with the group's last member, which is what scrambles the order.

**Choose it when:** a population needs periodic work spread across turns and
the order within a turn is irrelevant — entity checks, chunk revalidation,
listener polling.

**Capabilities.** Unique collection operations, set algebra over the members
(the result is a fresh even spread), random choice, capacity, `Partitioned`,
`BalancedPartition`, deterministic order, and content hash. The content hash
covers the division as well as the members, since the division is what the
structure maintains.

```rust
use voxel_world::structures::collections::Rota;

let mut due: Rota<u32, 8> = (0..100).collect();
due.remove(&17);
assert!(due.group_sizes().iter().all(|size| (12..=13).contains(size)));

// One turn's worth; every member is visited once per eight turns.
for entity in due.group(3) {
    let _ = entity;
}
```

### `MultiRota<T, const GROUPS: usize>`

**Purpose.** As `Rota`, but a member may be held several times and each copy is
placed independently, so copies of one value usually land in different groups.
Removing one copy scans the group it lives in: `O(len / GROUPS)`.

**Choose it when:** the population is counted rather than enumerated — several
pending jobs of one kind, repeated work for one owner.

**Capabilities.** Collection operations, `Measured`/`MeasuredMut` by copy count,
`WeightedChoose`, capacity, `Partitioned`, `BalancedPartition`, deterministic
order, and content hash.

```rust
use voxel_world::structures::collections::MultiRota;

let mut work: MultiRota<&str, 3> = MultiRota::new();
work.insert_times("regrow", 5);
assert_eq!(work.count_of(&"regrow"), 5);
assert!(work.group_sizes().iter().all(|size| *size == 2 || *size == 1));
```

### `OrderedRota<T, const GROUPS: usize>`

**Purpose.** Unique members divided into near-equal groups, where every member
keeps the position it was added at. Reading the rota gives that order; reading
one group gives the same order restricted to its members, so a group is always
a subsequence of the whole. Balancing moves the *latest* member of the fullest
group, so those that have waited longest keep their group.

**Choose it when:** the work is a queue whose order matters and must still be
spread — pending changes applied a slice per turn, where earlier changes must
be applied before later ones.

**Invariant.** Positions are an internal `u64` counter that advances on every
insertion and restarts on `clear`. Operations cost a tree lookup rather than
`Rota`'s constant time; `Capacity` is deliberately not implemented, because the
ordered storage is tree-backed and has nothing to reserve.

**Capabilities.** Unique collection operations, random choice, `Partitioned`,
`BalancedPartition`, deterministic order, and content hash over order and
division alike.

```rust
use voxel_world::structures::collections::OrderedRota;

let mut pending: OrderedRota<u32, 3> = (0..10).collect();
pending.remove(&5);
assert_eq!(pending.first(), Some(&0));

// Each group reads back in the overall order.
let turn: Vec<u32> = pending.group(0).copied().collect();
assert!(turn.windows(2).all(|pair| pair[0] < pair[1]));
```

### `OrderedMultiRota<T, const GROUPS: usize>`

**Purpose.** As `OrderedRota`, with repeats. Each copy takes its own position
and its own group; `remove_one` takes the earliest copy, so repeated work
behaves as a queue.

**Choose it when:** an ordered backlog contains the same item more than once and
still has to be spread over turns.

**Capabilities.** Collection operations, `Measured`/`MeasuredMut` by copy count,
`WeightedChoose`, random choice, `Partitioned`, `BalancedPartition`,
deterministic order, and content hash.

```rust
use voxel_world::structures::collections::OrderedMultiRota;

let mut backlog: OrderedMultiRota<&str, 2> = ["a", "b", "a"].into_iter().collect();
assert_eq!(backlog.count_of(&"a"), 2);
backlog.remove_one(&"a"); // The earliest copy.
assert_eq!(backlog.iter().copied().collect::<Vec<&str>>(), ["b", "a"]);
```

## Drawn recurrences

`Scheduler` fires once at a step the caller names, and a `Rota` recurs on an
exact period. Between them sits work that should happen *about* every so often:
a plot regrows, a den restocks, a rumour spreads. A `StochasticScheduler` holds
that work, draws each next occurrence itself, and places the draws so that no
one step collects the crowd.

|  | Repeats allowed | One per value |
|---|---|---|
| Drawn recurrence | `StochasticScheduler<T, R>` | `UniqueStochasticScheduler<T, R>` |

**Cadence ids are locked to their scheduler.** `register` returns a `CadenceId`
carrying the scheduler that issued it as well as the position it sits at, so
handing one scheduler another's id is refused rather than silently binding work
to whatever recurrence happens to occupy that position. The tag is folded from
the scheduler's domain seed, so it replays and survives a clone. `SlotMap` solves
the same problem for values behind handles, with a generation per slot and a
phantom type per map.

**Cadence.** A `Cadence` describes a recurrence by its **mean time to happen**:
the chance of firing is the same on every step, so the wait has no memory and
"on average every six hundred steps" needs no further explanation. Register a
cadence once and share its `CadenceId`; the cadence owns the sampler its mean
needs, including the table a rare chance builds, so a cadence per entry would
build that table per entry.

**Spread.** `Cadence::spread` does not change the distribution. The mean decides
roughly where an occurrence lands; the spread decides how much room the
scheduler has to even out the load once it is there. Above zero, two candidate
steps are drawn within that many steps of the draw and the emptier is taken —
sampling two and keeping the better is what turns a distribution with occasional
pile-ups into one without, cutting the tallest step from roughly
`log n / log log n` entries to about `log log n` for a draw and two lookups.

**Backlog.** Advancing fifty steps at once leaves fifty steps of owed work, and
no amount of spreading helps after the fact. `OnBacklog` says what "late" means
for the work in question: `Coalesce` (the default) delivers one firing that
reports how many occurrences it stands for, `FireAll` delivers one firing each,
and `Cap(n)` delivers at most `n` now and keeps the rest owed so the backlog
drains over the steps that follow.

**Randomness.** `R` is where each entry draws from, and any `StochasticStream` will
do: a `SeedCursor`, which retains its 128-bit origin and current position and is
replayable from the world seed; a `Random`, which is a xoshiro stream of its
own; or `EventRandom` to mix the two within one scheduler. An immutable `Seed`
is deliberately not a `StochasticStream`: repeatedly sampling one is a repeatable
one-shot decision, not an advancing schedule. Entries that say nothing derive
a seed from the scheduler's domain, the cadence, the value and a serial number,
then turn it into the selected source, so the whole schedule is a function of
the world seed rather than of when anything was inserted.

### `StochasticScheduler<T, R = SeedCursor>`

**Purpose.** Recurring work whose next occurrence is drawn from a cadence rather
than named by the caller. An entry re-arms itself every time it fires; nothing
outside re-schedules it. A value may be held several times, and each copy keeps
its own randomness and its own place.

**Choose it when:** many entities each need something to happen on their own
irregular schedule — growth, decay, spawning, wandering — and the per-step cost
must not scale with the population.

**Invariant.** Every entry has exactly one pending occurrence, always at a step
after the current one. Firing removes it and places the next.

**Capabilities.** Registration and sharing of cadences, insertion with derived
or supplied randomness, step and range inspection, cancellation by value or at a
known step, advancement one step or many, `Collection`, `RangeQuery`, and
`DeterministicOrder`.

Inserting returns the step the value landed on, and every `Firing` carries the
step its entry was re-armed to, so a caller that keeps its own index by due step
never has to search for where an entry went. `cancel_at` is the matching
removal: it costs the length of one step, where `cancel` walks the schedule.

```rust
use voxel_world::random::Seed;
use voxel_world::structures::collections::{Cadence, StochasticScheduler};

let mut growth: StochasticScheduler<u32> =
    StochasticScheduler::new(Seed::from_raw(1).child("growth"));
let slow = growth.register(Cadence::mtth(600).spread(4));

growth.insert(42, slow);

for firing in growth.advance() {
    // `occurrences` is one unless a backlog was coalesced.
    let _ = (firing.value, firing.occurrences);
}
```

### `UniqueStochasticScheduler<T, R = SeedCursor>`

**Purpose.** As `StochasticScheduler`, with one pending occurrence per value.
Scheduling a value that is already pending moves and re-rolls it rather than
adding a second occurrence, which is also how a value's cadence is changed.

**Choose it when:** the value stands for a thing rather than a job. A plot of
land has one next growth, not four, however many times it was registered.

**Invariant.** A value appears at most once. `due_at`, membership and
cancellation are direct lookups rather than searches, which is what the extra
index buys. Keeping that index costs one map write per firing and no search,
because the inner scheduler reports where each entry went: a step costs what its
firings cost, not what the population costs.

**Capabilities.** Those of `StochasticScheduler`, plus `due_at`, membership by
value, and `UniqueCollection`.

```rust
use voxel_world::random::Seed;
use voxel_world::structures::collections::{Cadence, UniqueStochasticScheduler};

let mut regrow: UniqueStochasticScheduler<u32> =
    UniqueStochasticScheduler::new(Seed::from_raw(1).child("regrowth"));
let cadence = regrow.register(Cadence::mtth(200));

assert!(regrow.insert(7, cadence));
assert!(regrow.insert(7, cadence)); // Moves it; still one pending.
assert_eq!(regrow.len(), 1);
```

## Sparse sequences

### `SparseSequence<T>`

**Purpose.** One value at each occupied signed `i64` index, with gaps allowed
and duplicate values permitted at different indices. Indices stay ordered and
a reverse index supports value-based removal and membership.

`insert_at` replaces at the exact index. `insert_shifting` makes room by moving
following entries. `compact` closes all gaps while preserving order.

**Choose it when:** coordinates or timeline positions matter but most positions
are empty—for example sparse animation keys, lanes, or editor tracks.

**Capabilities.** `SparseIndexed`, collection insertion/removal, ordered range
queries, reordering, choice, and canonical order.

```rust
use voxel_world::structures::collections::SparseSequence;

let mut keys = SparseSequence::new();
keys.insert_at(-20, "start");
keys.insert_at(100, "end");
assert_eq!(keys.get(100), Some(&"end"));
keys.compact();
assert_eq!(keys.iter().map(|(i, _)| i).collect::<Vec<_>>(), vec![0, 1]);
```

### `SparseSetSequence<T>`

**Purpose.** A sparse signed-index sequence with a `Set<T>` at each occupied
index. An item may occur in several slots, but only once in any one slot.
`len()` counts all stored item occurrences; `slot_count()` counts occupied
indices.

**Choose it when:** several unique events/entities may share a sparse time,
height, lane, or coordinate. Use `SparseSequence` if each index has exactly one
value.

**Capabilities.** `SparseIndexed`, slot and item insertion/removal, range
queries, reverse membership, slot reorder/sort/shuffle, and canonical order.

```rust
use voxel_world::structures::collections::SparseSetSequence;

let mut events = SparseSetSequence::new();
events.insert_at(20, "rain");
events.insert_at(20, "thunder");
assert_eq!(events.slot_count(), 1);
assert_eq!(events.len(), 2);
```

## Measured collections

### `MultiSet<T>`

**Purpose.** A bag/multiset storing an integer count for every distinct value.
`len()` and `distinct_len()` count distinct values; `total_count()` counts all
occurrences. Set union keeps maximum counts, intersection keeps minimum counts,
difference subtracts, and `sum` adds counts.

**Choose it when:** inventory stacks, histogram bins, vote/frequency counts, or
weighted random selection by integer multiplicity are needed.

**Capabilities.** Collection operations, measured mutation, weighted and
uniform choice, set algebra, deterministic traversal, and conversion through
iterators of distinct values or occurrences.

```rust
use voxel_world::structures::collections::MultiSet;

let mut inventory = MultiSet::new();
inventory.insert_times("stone", 64);
inventory.remove_times(&"stone", 3);
assert_eq!(inventory.count_of(&"stone"), 61);
```

### `WeightedSet<T>`

**Purpose.** A unique collection with a finite non-negative `f64` weight for
each item. Zero weight means absence. Weighted choice is proportional to stored
weights; the weights do not need to sum to one.

**Choose it when:** loot rarity, heuristic importance, influence, or continuous
sampling weight is intended. Use `FuzzySet` when the number means “degree of
membership” and must remain a probability.

**Capabilities.** Collection operations, `Measured`/`MeasuredMut`, weighted
choice, set algebra, deterministic traversal, and conversion to `MultiSet`.
Invalid, negative, or infinite weights panic rather than entering storage.

```rust
use voxel_world::structures::collections::WeightedSet;

let mut ores = WeightedSet::new();
ores.set_weight("iron", 8.0);
ores.set_weight("diamond", 0.2);
assert_eq!(ores.total_weight(), 8.2);
```

### `FuzzySet<T>`

**Purpose.** A fuzzy set whose stored membership is in `(0, 1]`; zero is
absence. Union uses `a + b - ab`, intersection uses `ab`, difference uses
`min(a, 1 - b)`, and complement uses `1 - a`. `realize` independently keeps
each member according to its membership probability.

**Choose it when:** classifications or states are uncertain or gradual—biome
affinity, semantic matching, confidence, or soft rules. Do not use it merely as
an arbitrary weight table.

**Capabilities.** Collection operations, measured mutation using the typed
`Probability` value, weighted choice, normalization, fuzzy algebra, approximate
equality, and deterministic traversal. `MEMBERSHIP_TOLERANCE` is the suggested
comparison tolerance.

```rust
use voxel_world::{
    structures::collections::FuzzySet,
    units::Probability,
};

let mut affinity = FuzzySet::new();
affinity.set_membership("forest", Probability::clamped(0.8));
affinity.set_membership("desert", Probability::clamped(0.1));
assert!(affinity.membership_of(&"forest") > affinity.membership_of(&"desert"));
```

## Maps and relationships

### `FastHashMap<K, V>`, `FastHashSet<T>`, and hashing aliases

**Purpose.** Standard `HashMap`/`HashSet` aliases using `FastBuildHasher`, whose
hasher is `FastHasher`. `hash_one` and `unordered_hash` are helpers for direct
hashing. They live in `voxel_world::structures::hashing`.

**Choose them when:** a project-specific wrapper adds no useful invariant and
trusted game data needs quick lookup. Never expose this hasher to adversarial
keys, and never treat iteration order or standard `Hash` output as persistent
cross-version data.

`FastHashMap` implements the project's `Map`/`MapMut` capability traits.

### `BiMap<L, R>` and `Overwritten<L, R>`

**Purpose.** A one-to-one bidirectional map. Every left and right value appears
in at most one pair. `insert` resolves conflicts and reports displaced pairs in
`Overwritten`; `insert_no_overwrite` rejects any conflict.

**Choose it when:** both directions are primary and unique—for example runtime
IDs ↔ external IDs, names ↔ handles, or ports ↔ connections.

**Capabilities.** `Map`/`MapMut`, `ValueIndexed`/`ValueIndexedMut`,
`UniqueValueMap`, capacity management, deterministic map order, and invariant
checking.

```rust
use voxel_world::structures::mappings::BiMap;

let mut names = BiMap::new();
names.insert(7, "player");
assert_eq!(names.get_by_left(&7), Some(&"player"));
assert_eq!(names.get_by_right(&"player"), Some(&7));
```

### `UniqueMultiMap<K, V>` and `OneToManyMap<K, V>`

**Purpose.** A key owns any number of values, but each value belongs to at most
one key. `insert` refuses a value already owned elsewhere; `insert_or_move`
transfers it and returns the former key. `OneToManyMap` is an alias.

**Choose it when:** children have exactly one parent, chunks own entities, or a
resource belongs to one group while groups hold many resources.

**Capabilities.** Forward and reverse map traits, `UniqueValueMap`, pair and
owner lookup, capacity management, retention, deterministic map order, and
invariant checking.

```rust
use voxel_world::structures::mappings::OneToManyMap;

let mut ownership = OneToManyMap::new();
ownership.insert("chunk-a", 10);
assert_eq!(ownership.key_of(&10), Some(&"chunk-a"));
assert_eq!(ownership.insert_or_move("chunk-b", 10), Some("chunk-a"));
```

### `MultiMap<K, V>` and `ManyToManyMap<K, V>`

**Purpose.** A many-to-many relation with full forward and reverse indices.
Both sides are unique sets, and `pair_count()` distinguishes edge count from
the number of keys or values. `ManyToManyMap` is an alias.

**Choose it when:** entities may have many tags and tags many entities,
resources have many users, or a graph-like bipartite relation needs symmetric
queries. Use `GroupedMultiMap` when largest/smallest group queries matter.

**Capabilities.** `Map`/`MapMut`, shared reverse-value map traits, removal from
either side, retention, capacity management, deterministic map order, and
invariant checking.

```rust
use voxel_world::structures::mappings::ManyToManyMap;

let mut tags = ManyToManyMap::new();
tags.insert(7, "solid");
tags.insert(7, "opaque");
assert!(tags.contains_pair(&7, &"opaque"));
assert_eq!(tags.keys_of(&"solid").unwrap().len(), 1);
```

### `GroupedSingleMap<E, L>` and `PartitionMap<E, L>`

**Purpose.** Every element maps to exactly one label, while each label groups
many elements. It additionally indexes group sizes, enabling cheap largest,
smallest, and size-based group queries. `PartitionMap` is the same type.

**Choose it when:** elements partition into teams, biomes, states, regions, or
owners and queries care about the groups themselves. `insert` relabels an
existing element and returns its old label.

**Capabilities.** Map mutation, shared reverse lookup, `Grouping`,
`GroupSizes`, group insertion/removal, and deterministic map order.

```rust
use voxel_world::structures::{mappings::PartitionMap, traits::Grouping};

let mut teams = PartitionMap::new();
teams.insert("alice", "red");
teams.insert("bob", "red");
assert_eq!(teams.group(&"red").unwrap().len(), 2);
```

### `GroupedMultiMap<E, L>` and `LabelMap<E, L>`

**Purpose.** Every element may have any number of labels, every label groups
many elements, and group sizes are indexed. `replace_labels` atomically changes
an element's label set. `LabelMap` is an alias.

**Choose it when:** tags are explicitly stored rather than derived and frequent
queries need label groups or largest/smallest labels. If labels can always be
computed from each value, `LabelIndexedSet` avoids storing them separately.

**Capabilities.** Forward/reverse shared map traits, `Grouping`, `GroupSizes`,
group replacement/removal, retention, deterministic map order, and invariant
maintenance.

```rust
use voxel_world::structures::{mappings::LabelMap, traits::Grouping};

let mut labels = LabelMap::new();
labels.insert(1, "hot");
labels.insert(1, "fluid");
assert_eq!(labels.get(&1).unwrap().len(), 2);
assert_eq!(labels.group(&"hot").unwrap().len(), 1);
```

### `PairMap<T>`

**Purpose.** A symmetric collection of exclusive unordered pairs in one value
domain. A value has at most one partner. Self-pairs are supported. Pair order is
irrelevant, and `partner(value)` performs direct reverse lookup.

**Choose it when:** matching endpoints, portal partners, paired devices, or
temporary one-to-one relationships where “left” and “right” have no meaning.
Use `BiMap` if the two sides have different types or roles.

**Capabilities.** Pair insertion/removal, partner lookup, capacity management,
deterministic map order, and invariant checking. It intentionally is not a
`Map` because there is no directional key/value relation.

```rust
use voxel_world::structures::mappings::PairMap;

let mut portals = PairMap::new();
assert!(portals.insert("blue", "orange"));
assert_eq!(portals.partner(&"orange"), Some(&"blue"));
```

### `LayeredMap<K, V>`

**Purpose.** An ordered stack of maps read as one, where the earliest layer
holding a key decides its value. Nothing is merged or copied: the layers stay as
they are, so any one of them can be reloaded, swapped or saved by itself, and a
value hidden by a higher layer comes back when that layer's copy is removed.

**Choose it when:** the sources of a value must stay separable. Layered
configuration — runtime override over world configuration over engine defaults —
or inherited properties, where an instance falls back on its archetype and then
on base defaults. Also when what is shadowed has to remain recoverable, which a
merged map cannot offer.

**Avoid it when:** the layers are never taken apart. A `FastHashMap` built by
merging the sources is smaller, and every operation on it is one hash look-up
rather than one per layer. `len` in particular is not constant time here.

**Invariant.** The value for a key is the one in the earliest layer holding it,
and lower occurrences stay intact and become visible again when the ones above
them are removed. There is always at least one layer, so insertion cannot fail;
`remove_layer` refuses to remove the last one.

**Mutation differs from Python's `ChainMap`.** `ChainMap` writes everything to
the first map, piling a new entry on top of the one it shadows. Here a write to
an existing key goes to *the layer that already holds it*, so editing a
configuration value changes the configuration rather than leaving a runtime
override behind it. A key no layer holds is new, so it goes into layer 0.
`insert_into` writes to a chosen layer where the `ChainMap` behaviour is wanted.

**Shadowing.** `occurrences(key)` lists every layer holding a key, highest
precedence first, so `skip(1)` is what removal would uncover in turn.
`is_shadowed` answers whether removing the visible value reveals another rather
than removing the key, and `remove_everywhere` is for when it should really be
gone. `promote` moves a visible value up into a higher-precedence layer without
disturbing what is under it.

**Complexity.** A look-up is at most one hash look-up per layer and stops at the
first hit, so everything scales with the number of layers, which is expected to
be small. Nothing is cached, because `layer_mut` hands out the backing maps and
an index that is sometimes wrong is worse than none. The consequence is that
`len` counts distinct visible keys by walking every pair in every layer, with a
shortcut for a single layer; `is_empty` and `total_len` are cheap. Visible
iteration is lazy and allocates nothing — a pair is visible when no earlier layer
holds its key — at a look-up per earlier layer.

**Capabilities.** `Map` over the visible view, `DeterministicMapOrder`,
`ContentHashable` over the visible pairs, plus direct access to each layer.
`MapMut` is deliberately not implemented: its `remove` promises to remove a key
and everything it maps to, which a structure whose removal can uncover a lower
value cannot honour. `Capacity` is not implemented either, because no single
number satisfies both halves of its contract — the sum of the layers' capacities
overstates what can be added before a reallocation, since new keys all go to
layer 0, while layer 0's own capacity can be below the visible length. `reserve`
and `shrink_to_fit` exist as inherent methods with those meanings spelled out.

**Not canonical.** Layer order fixes precedence and each layer is hashed with a
fixed hasher, so iteration replays; but within a layer the order still follows
that map's insertion history, so two layered maps holding the same pairs can
iterate differently. `CanonicalMapOrder` is therefore not claimed.

```rust
use voxel_world::structures::hashing::FastHashMap;
use voxel_world::structures::mappings::LayeredMap;

let defaults: FastHashMap<&str, u32> = [("volume", 50), ("distance", 8)].into_iter().collect();

let mut settings: LayeredMap<&str, u32> = LayeredMap::new();
settings.push_layer(defaults);

settings.insert("volume", 80);
assert_eq!(settings.get(&"volume"), Some(&80));
assert_eq!(settings.find_layer(&"volume"), Some(1), "changed where it lived");

// Put an override above the default instead, then take it away again.
assert_eq!(settings.insert_into(0, "distance", 32), Ok(None));
assert_eq!(settings.get(&"distance"), Some(&32));
assert_eq!(settings.remove(&"distance"), Some(32));
assert_eq!(settings.get(&"distance"), Some(&8), "the default reappears");
```

### `IntervalMap<K, V>`

**Purpose.** A canonical map of non-overlapping half-open intervals to values.
Later insertion overrides overlap by trimming or splitting existing spans;
adjacent equal spans merge. Unmapped gaps remain gaps.

**Choose it when:** large coordinate/time ranges share one value—permissions,
terrain strata, address ranges, or timeline states. Use `RunLengthSequence` for
a fully dense zero-based sequence instead.

**Capabilities.** Point lookup, containing-span lookup, overlapping range
queries, span iteration, capacity management, canonical order, and content
hashing.

```rust
use voxel_world::structures::mappings::IntervalMap;

let mut strata = IntervalMap::new();
strata.insert(0, 40, "stone");
strata.insert(10, 15, "ore");
assert_eq!(strata.get(&12), Some(&"ore"));
assert_eq!(strata.get(&15), Some(&"stone"));
```

### `LruCache<K, V>` and `Evicted<K, V>`

**Purpose.** A bounded least-recently-used map. `get` refreshes recency;
`peek` reads without refreshing. Insertion returns `Evicted`, which separately
reports a replaced value and a capacity-dropped least-recent entry.

**Choose it when:** cached chunks, decoded assets, query results, or temporary
resources must remain within a hard entry count. The limit is count-based, not
byte-based.

**Capabilities.** `Bounded`, `EvictingInsert`, key lookup/removal, recency-order
iteration, next-eviction inspection, deterministic map order, and content hash.

```rust
use voxel_world::structures::mappings::LruCache;

let mut chunks = LruCache::new(2);
chunks.insert(1, "one");
chunks.insert(2, "two");
chunks.get(&1); // 1 is now most recent.
let result = chunks.insert(3, "three");
assert_eq!(result.dropped, Some((2, "two")));
```

### `SetKeyMap<K, V>`

**Purpose.** An intent alias for `FastHashMap<Set<K>, V>`. `Set<K>` hashes by
membership, so insertion order does not affect key equality or hash.

**Choose it when:** a combination of unordered features, ingredients, or flags
is itself the lookup key. There is no wrapper or extra runtime behavior.

```rust
use voxel_world::structures::{collections::Set, mappings::SetKeyMap};

let mut recipes = SetKeyMap::default();
let ingredients = Set::from(["sand", "heat"]);
recipes.insert(ingredients, "glass");
```

## Names: text compared as an integer

`Name` holds text as the hash of that text, so comparing two names is comparing
two integers. It is this crate's answer to Godot's `StringName`, for text used as
an *identity* — voxel kinds, property names, tag segments, anything read from
data and then compared a great many times. Text that is *content* — a sign's
inscription, a player's message — wants a `String`.

**Measured.** On the hashing benchmark, 200,000 inserts and look-ups cost 14.8 ms
keyed by short strings against 2.7 ms keyed by an integer. Direct comparison of
2,000 names against a needle, in release, runs 3.4× faster than the same
comparison on `String`.

**The id is the hash, not a registration number.** A name's id is a pure function
of its text, so nothing has to be registered first, two processes agree without
speaking, and the id is the same every run. That is the difference from an
interner handing out 0, 1, 2… in order of first use: those ids move when a mod
loads earlier, cannot be written to a save, and make any map keyed by them
iterate differently between runs.

**Compile time.** `Name::new` is a `const fn`, so `const STONE: Name =
Name::new("stone")` is worked out while compiling and a comparison against it is
one integer compare against an immediate. The `name!` macro forces the same thing
inside an expression, by evaluating in a constant block rather than trusting the
optimiser.

**Reading one back.** An id cannot be reversed — that is what hashing means. A
global registry maps registered ids back to their text for printing, and
`Name::text` consults it; an unregistered name prints as its id. `Name::intern`
copies and registers, `Name::register` takes `&'static str` without copying, and
`Name::register_all` does a start-up table. Comparison never touches the
registry, so the lock is never on a hot path.

**Collisions** need about five billion distinct names to become likely; a world
with a hundred thousand sits near one chance in four billion. The registry checks
anyway, because the failure would be silent and total, and `intern` panics rather
than let it pass — `try_intern` reports it instead.

**Ordering.** `Ord` compares ids, so it is stable across runs and machines but
**not alphabetical**. It is the right order for a `BTreeMap<Name, _>` that should
iterate canonically, and the wrong one for anything shown to a person, which
should sort on `text`.

```rust
use voxel_world::name;
use voxel_world::structures::Name;

const STONE: Name = Name::new("stone");

assert_eq!(name!("stone"), STONE);
assert_eq!(Name::intern("stone"), STONE, "the same id either way");
assert_eq!(STONE.text(), Some("stone"), "and now it can be read back");
```

## Query indices

### `TagIndex`

**Purpose.** A hierarchical index of slash-separated tag paths. Each segment
may be a bare tag or `type:tag`. Adding a deep path creates its prefixes;
removing a path removes its subtree and prunes unused ancestors. The index can
query exact paths, prefixes, tags/types at one or any depth, sub-indices, and
tag↔type associations.

**Choose it when:** data has a taxonomy such as
`world/biome:forest/tree:oak`, and both hierarchy and typed tag queries matter.
It indexes paths themselves; use `PropertyQuery` when querying many entities by
property expressions.

```rust
use voxel_world::structures::indices::TagIndex;

let mut index = TagIndex::new();
index.add("world/biome:forest/tree:oak");
assert!(index.has("world/biome:forest"));
assert!(index.has_exact("world/biome:forest/tree:oak"));
assert!(index.has_type_any_depth("tree"));
```

## Value kinds: the property schema

A property store is generic over one value type `V`. While `V` is a single
concrete type the compiler already guarantees that every value in a property is
the same kind of thing. The moment `V` becomes an enum so that different
properties can hold different things, that guarantee is gone: a weather and a
health have the same Rust type, and nothing stops one being written to the other.

The `Kinded` trait closes that gap. A value type names its own kind type and
reports each value's kind; a property is registered with the kind it accepts;
and every write and every comparison is checked against it.

```rust
use voxel_world::structures::traits::Kinded;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Value { Health(u32), Weather(&'static str) }

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind { Health, Weather }

impl Kinded for Value {
    type Kind = Kind;

    fn kind(&self) -> Kind {
        match self {
            Self::Health(_) => Kind::Health,
            Self::Weather(_) => Kind::Weather,
        }
    }
}
```

**Operators.** `Expr` compares with `has` and `is`, lists alternatives with
`one_of`, and orders with `less`, `at_most`, `greater`, `at_least` and `between`,
all combined by `and`, `or` and `negate`. Ordering is carried by a single
`Compare(P, Comparison, V)` node rather than four variants, so adding an operator
later costs one arm rather than one arm in every walk over an expression.

**Type safety, as far as Rust reaches.** The ordering constructors live in an
impl block bounded on `V: PartialOrd`, so a value type with no order at all
cannot have `Expr::less` written against it — that is a compile error, not a
schema error. *Which properties* may be ordered is still a run-time question,
because the property is named by a run-time value, so it is checked in
`validate` alongside the value kind and reported as `SchemaError::NotOrdered`.

The check works by asking whether a value compares with itself: a kind is
ordered exactly when it does. A tagged value type says which of its kinds have
an order by writing `PartialOrd` **by hand** — returning `None` across kinds and
for categorical ones:

```rust
impl PartialOrd for PropertyValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Depth(a), Self::Depth(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}
```

Deriving it would be wrong, and quietly so: `derive(PartialOrd)` orders by
variant position, making `Health(5) < Weather(Rain)` answer `true`. The same
trap as deriving `Ord` on a tagged value — see also `check_kinds`.

**Comparisons cannot use the inverted index.** `is` and `one_of` are answered by
`members_with`, a direct look-up per value. There is no index for "above five",
so a comparison in a *set query* walks the property's members: linear in how many
hold the property. Single-element evaluation is unaffected — `ConditionIndex` and
`Gate` look up the one element's values and compare — so gating and conditions
pay nothing for using comparisons.

**The invariant.** Every registered property has a fixed storage contract and a
fixed value-kind contract, and every stored value and every value used in a query
or a condition against it satisfies both. `SchemaError` says which of the five
ways a use broke it: the property was never registered, the value was the wrong
kind, the property is a flag and holds no value, it was re-registered for a
different value kind, or it was re-registered with a different storage kind.

**Registration is idempotent only on both halves.** Registering `Single + UInt`
again as `Single + UInt` is a repeat and does nothing. As `Single + Text` it is a
`ValueKindConflict`, and as `Multi + UInt` a `Conflict`; neither is applied,
because values already stored would otherwise answer to a contract nothing
checked them against. Changing a property's type on purpose is
`unregister_property` followed by a fresh registration, which drops the values
with it rather than silently reinterpreting them.

**Kinds are orthogonal to cardinality.** `Flag`, `Single`, `UniqueSingle`,
`Multi` and `UniqueMulti` still decide how many values an element may hold and
whether values may be shared. The value kind decides what those values may be.
Flags have no value kind, which is why they are registered with `register_flag`
rather than being given one that is never consulted.

**Equality is guaranteed, not assumed.** A store holds values in hashed sets and
compares them to answer `Is` and to find owners, so equality has to be total. A
value type is `Element`, which is `Eq + Hash + Clone`, and deriving `Eq` on an
enum asserts `Eq` for every variant's payload in turn — so a payload without
total equality, such as an `f64`, keeps the whole value type out of a store at
compile time. `a == b` is therefore always safe to write and always means what it
says about the payloads.

Two further rules bind equality to the schema, and matter only for a hand-written
`Eq` or `kind`: a kind is stable across calls, and equal values have equal kinds.
The second is what stops a value being stored under one kind and found under
another, since storage looks values up by equality while the schema checks them
by kind. A derived `Eq` satisfies both for free. `check_kinds` verifies them over
a sample for an implementation written by hand.

**The single-kind case costs nothing.** A type with one logical kind implements
`Kinded` with `Kind = ()`, and the check compares two zero-sized values. It is
implemented here for the standard value types, so a store over `String` or `u64`
needs no work.

**Why a trait rather than an enum in this crate.** The kinds belong to the
caller's world. Naming the kind type as an associated type keeps the check fully
general while staying type safe: a property's declared kind and a value's
reported kind are the same Rust type, and kinds from two different value types
cannot be crossed. Rust cannot require that `V` *be* an enum; requiring that it
report a kind is the enforceable form of the same idea.

**Writes and expressions are checked; simple predicates are total.** Everything
that stores a value returns `Result`, and so does everything that takes an
expression: `query`, `query_uncached`, `query_all`, `query_any`,
`ConditionIndex::satisfies` and `ConditionIndex::watch`. An expression that
compares a property against the wrong kind, compares a value against a flag, or
names an unregistered property is malformed rather than merely unmatched, and
says so instead of returning an empty set.

Validation happens once at the boundary; evaluation and caching then recurse
through an internal unchecked path, so a tree of `n` nodes is walked for
validation once rather than once per level. Single-value predicates like
`has_value` stay total, because a value of the wrong kind cannot be present and
answering `false` is correct rather than lax.

**There is no implicit registration.** A write to an unregistered property is an
error rather than defining the property from whatever arrived first, which is how
a typo used to become a column.

### `PropertyQuery<E, P, V>`

**Purpose.** A schema-aware property database and inverted query index. A
property is registered as one of:

- `Flag`: presence only;
- `Single`: one value per element, values may be shared;
- `UniqueSingle`: one value per element and one element per value;
- `Multi`: many values per element, values may be shared;
- `UniqueMulti`: many values per element, but each value belongs to one element.

Queries are expression trees made with `Expr::has`, `Expr::is`, `Expr::and`,
`Expr::or`, and `Expr::negate`. Results are cached and mutations invalidate only
caches that depend on the changed property. `FuseMode` controls whether merging
another query rejects or overwrites conflicting schemas; `FuseError` reports
the property involved. `Cardinality` and `Uniqueness` describe a
`PropertyKind` programmatically. `property_query::RefIter<'_, T>` is the small
zero/one/many borrowing iterator returned by value and member lookups; callers
normally consume it simply as an iterator.

**Choose it when:** many entities need repeated AND/OR/NOT queries across a
dynamic typed schema, as in ECS-style component/property selection. It uses one
value type `V` for all properties; use typed component stores when each property
needs a fundamentally different Rust type.

```rust
use voxel_world::structures::indices::{Expr, PropertyKind, PropertyQuery};

let mut query = PropertyQuery::<u32, &str, &str>::new();
query.register_property("biome", PropertyKind::Single);
query.register_property("active", PropertyKind::Flag);
query.set(1, "biome", "forest");
query.set_flag(1, "active");

let expression = Expr::and([
    Expr::is("biome", "forest"),
    Expr::has("active"),
]);
assert!(query.query(&expression).contains(&1));
```

### `ConditionIndex<E, P, V>`

**Schema.** Properties and facts are registered with the value kind they accept,
and conditions are validated against it when they are registered. See
[Value kinds](#value-kinds-the-property-schema).

**Purpose.** Element state together with conditions registered over it. Where
`PropertyQuery` answers *which elements match this expression now*, this answers
*which registered conditions have just become true*. It owns the state the
conditions read, so every write works out what it released, and no caller has to
remember which conditions might care about the write it is making.

**Choose it when:** something should happen the moment state allows it — a seed
germinates when the ground thaws, a job unblocks when its materials arrive — and
you do not want to re-scan a population to find out. If a few ticks of latency
is acceptable, `PropertyQuery` behind a `Rota` scan is cheaper and needs nothing
new.

**Invariant.** One condition per element. A condition fires on the crossing from
false to true, never on the level, and one that already holds when it is
registered fires at once. `OnSatisfied` decides what happens next: `Once` drops
it, `Rearm` keeps it for the next crossing.

**Results.** Satisfied elements go onto a ready list that the caller drains with
`take_ready`. Nothing is called back from inside a setter, so writes stay cheap
and the results have one order.

**Cost.** Each condition records the properties it reads, and the index keeps the
reverse map. An element's property write re-checks that element alone, and only
if its condition reads that property; a world fact re-checks the conditions that
read it. Negation is the exception: `Not` and the empty `And` read the universe,
so they are re-checked whenever an element joins or leaves.

**Facts.** A property declared with `register_fact` belongs to the world rather
than to any element — the season, the weather — so it is written once instead of
being copied onto every element that reads it.

**Deliberately excluded.** A condition reads its own element's properties and the
world's facts, never another element's. That is a join, and joins are what turn
a watch list into a rule network.

**Capabilities.** `Collection` and `UniqueCollection` over the watched elements,
`CollectionRemove` (which drops conditions, not state), `Map` from element to
condition, `Pending`, and `DeterministicOrder`.

**Emptying it.** `clear` drops every condition and leaves the state and facts
standing, because that is what removing every element of the collection means
here. `reset` puts the whole index back to new.

```rust
use voxel_world::structures::indices::{ConditionIndex, Expr, PropertyKind};

let mut gates: ConditionIndex<u32, &str, &str> = ConditionIndex::new();
gates.register_property("ground", PropertyKind::Single);
gates.register_fact("season");

gates.watch(
    7,
    Expr::and([Expr::is("ground", "thawed"), Expr::is("season", "spring")]),
);

gates.set(7, "ground", "thawed");
assert!(gates.take_ready().is_empty());

gates.set_fact("season", "spring");
assert_eq!(gates.take_ready(), vec![7]);
```

### `Gate<E, P, V>` and `PropertyStore<E, P, V>`

**Purpose.** A condition per element, answered at the moment work comes up.
Where `ConditionIndex` reports a condition *becoming* true, a gate answers *is
this true now* — level, not edge. It has no memory, no ready list and no notion
of a crossing: ask it twice with nothing changed and it answers the same both
times.

**Choose it when:** work already has its own schedule and the condition decides
whether each occurrence counts. A plot is due to grow, but only if the ground is
thawed; a creature's turn comes round, but only if it is awake. Choose
`ConditionIndex` instead when the condition itself is what should set work
going.

**An element with no condition is allowed.** A gate holds only the conditions
that actually restrict something, and everything else passes, so gating a few
members of a large population costs only those few.

**Gated operations.** All three scheduling families take a gate, and what a
refused condition means is the difference between them:

| Structure | Method | A refused element |
|---|---|---|
| `Scheduler`, `UniqueScheduler` | `advance_with`, `advance_by_with` | Dropped. The occurrence was one-shot and is spent |
| `StochasticScheduler`, `UniqueStochasticScheduler` | `advance_with`, `advance_by_with` | Re-armed. It already drew its next wait, so it comes round again on its own cadence |
| `Rota`, `MultiRota`, `OrderedRota`, `OrderedMultiRota` | `group_with` | Left out of this turn. A turn is a read, so it comes round again next round |

One gate can serve all of them at once, which is why this is a separate
structure rather than a field on each: the same conditions gate a one-shot
queue, a recurring schedule and a rota without being written three times.

**`PropertyStore`** is the state underneath: element properties, world facts,
the value-kind schema, and single-element expression evaluation. Both
`ConditionIndex` and `Gate` hold one, so the schema rules live in one place. A
gate can hand out `store_mut` because it keeps nothing derived from the state; a
condition index cannot, because its edge tracking would go stale.

**Capabilities.** `Collection`, `UniqueCollection` and `CollectionRemove` over
the restricted elements, `Map` from element to condition, and
`DeterministicOrder`. Removing through `CollectionRemove` lifts a restriction
rather than dropping the element's state.

```rust
use voxel_world::structures::collections::Scheduler;
use voxel_world::structures::indices::{Expr, Gate, PropertyKind};

let mut gate: Gate<u32, &str, &str> = Gate::new();
gate.register_property("ground", PropertyKind::Single, ())?;
gate.require(7, Expr::is("ground", "thawed"))?;

let mut work: Scheduler<u32> = Scheduler::new();
work.schedule(1, 7);
work.schedule(1, 8);

// Seven is frozen when its turn comes, so it is dropped; eight has no
// condition, so it passes.
gate.set(7, "ground", "frozen")?;
assert_eq!(work.advance_with(&gate), vec![8]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Dense and handle storage

### `Palette<T>`

**Purpose.** A fixed-length dense sequence that stores each distinct value
once and packs per-position palette indices at 0, 1, 2, 4, 8, 16, or 32 bits.
Reads are O(1). Adding enough distinct values to cross an index-width boundary
re-packs the index array. Unused palette entries remain until `compact()`.

**Choose it when:** a large array has few distinct values regardless of their
spatial arrangement, especially voxel material IDs. For long contiguous bands,
`RunLengthSequence` may be smaller. For changing data shape, use
`CompactSequence`.

**Capabilities.** Dense `Sequence`, collection inspection, random choice,
uniformity/count queries, explicit palette compaction, canonical order, and
content hashing.

```rust
use voxel_world::structures::storage::Palette;

let mut voxels = Palette::filled(32 * 32 * 32, 0u16);
voxels.set(123, 7);
assert_eq!(voxels.distinct_len(), 2);
assert_eq!(voxels.get(123), Some(&7));
```

### `CompactSequence<T>` and `CompactKind`

**Purpose.** One fixed-length dense-sequence API backed by whichever of four
representations currently estimates smallest:

- `Uniform`: one shared value;
- `Palette`: few distinct values anywhere;
- `Runs`: long adjacent equal ranges;
- `Dense`: one direct value per position.

Construction from values and `compact()` inspect the complete logical
sequence. Point/range mutation stays in the current representation (except the
first differing uniform write becomes a palette), so a loop of edits does not
silently become O(n²). Call `compact()` once after a substantial batch.
`kind()` is diagnostic; logical equality and content hashes ignore the current
physical representation.

**Choose it when:** dense data shifts between homogeneous, palette-friendly,
run-friendly, and noisy phases and the caller should not own that policy. Use a
specific representation when the data shape is known and conversion overhead
is unwanted.

**Capabilities.** Dense `Sequence`, collection inspection, random choice,
storage diagnostics (`kind`, `index_bits`, `packed_bytes`), canonical order,
and content hashing.

```rust
use voxel_world::structures::storage::{CompactKind, CompactSequence};

let mut cells = CompactSequence::filled(4096, 0u16);
assert_eq!(cells.kind(), CompactKind::Uniform);
cells.set(17, 3);
assert_eq!(cells.kind(), CompactKind::Palette);
// Reconsider all representations once, after the edit batch.
cells.compact();
```

### `Grid<T, const N: usize>` and `Grid2` / `Grid3` / `Grid4`

**Purpose.** A checked rectangular dense grid backed by one flat `Vec<T>`.
Axis zero changes fastest. For `[x, y, z]` in `[width, height, depth]`:

```text
x + width * (y + height * z)
```

Construction returns `None` if dimension multiplication overflows;
`from_vec` also rejects the wrong value count. Coordinate access returns
`None` out of bounds. A zero-sized axis makes a valid empty grid.

**Choose it when:** dimensions are fixed for the grid's lifetime and every cell
stores a value—images, chunk arrays, simulation fields, time slices, or lookup
tables. Flat slices are suitable for serialization and buffer upload. `N` can
be any const dimension; aliases exist for the common 2D, 3D, and 4D cases.

```rust
use voxel_world::structures::storage::{Grid2, Grid4};

let mut image = Grid2::filled([3, 2], 0u8).unwrap();
*image.get_mut([2, 1]).unwrap() = 9;
assert_eq!(image.index_of([2, 1]), Some(5));

let samples = Grid4::from_vec([2, 1, 1, 2], vec![10, 11, 12, 13]).unwrap();
assert_eq!(samples.get([1, 0, 0, 1]), Some(&13));
```

### `BitGrid<const N: usize>` and `BitGrid2` / `BitGrid3` / `BitGrid4`

**Purpose.** The same checked coordinate system as `Grid`, with boolean cells
stored through `BitSet`. A new grid has no per-cell bit allocation; storage
grows only as high as its greatest set flat index. Valid never-set cells read
as `Some(false)`.

**Choose it when:** coordinates represent occupancy, visibility, visited state,
selection, or another boolean mask. It is especially compact when most cells
are clear or the full volume is moderate.

```rust
use voxel_world::structures::storage::BitGrid3;

let mut occupied = BitGrid3::new([16, 16, 16]).unwrap();
assert_eq!(occupied.set([3, 4, 5], true), Some(false));
assert_eq!(occupied.get([3, 4, 5]), Some(true));
assert_eq!(occupied.count_ones(), 1);
```

### `SlotMap<K, T>`

**Purpose.** Owns values in reusable slots and returns typed generational
`Id<K>` handles. Removing a value increments its slot generation, so an old ID
does not accidentally resolve to the value that later reuses that slot. The
low 40 ID bits encode the slot and the remaining bits encode roughly 16 million
generations per slot.

**Choose it when:** entities, assets, nodes, or resources need cheap stable
handles, dense ownership, slot reuse, and stale-handle rejection. A typed ID
also prevents mixing unrelated handle kinds at compile time.

**Caveat.** An ID proves its slot and generation, not the identity of the
`SlotMap` instance. Do not resolve an ID against a different map of the same ID
kind.

**Capabilities.** `HandleStore`, capacity management, iteration by slot,
deterministic/canonical order, retention, and content hashing.

```rust
use voxel_world::{define_id_kinds, structures::storage::SlotMap};

define_id_kinds! { EntityKind => "entity" }

let mut entities = SlotMap::<EntityKind, &str>::new();
let old = entities.insert("slime");
assert_eq!(entities.remove(old), Some("slime"));
let new = entities.insert("bat");
assert_eq!(entities.get(old), None); // Stale handle rejected.
assert_eq!(entities.get(new), Some(&"bat"));
```

## Capability traits

The traits in [`src/structures/traits`](../src/structures/traits) let generic
algorithms request behavior rather than concrete storage. Many useful methods
are trait methods, so import the prelude or the specific trait.

| Trait family | Contract |
|---|---|
| `Collection`, `CollectionInsert`, `CollectionRemove`, `CollectionMut` | Borrowed elements, size, membership, and mutation |
| `ValueCollection` | Collection whose members are computed and returned by value |
| `UniqueCollection` | No repeated logical elements |
| `SetAlgebra`, `ValueSetAlgebra` | Union, intersection, difference, subset/disjoint tests |
| `Choose`, `ChooseMut` | Random selection or random removal |
| `Sequence`, `SequenceMut`, `InsertAt` | Dense positional access and mutation |
| `SparseIndexed` | Signed integer positions with possible gaps |
| `Reorder` | Sort, reverse, shuffle, and related order changes |
| `FixedCapacity` | Sequence with a hard element capacity |
| `Measured`, `MeasuredMut` | Per-element count, weight, or membership |
| `WeightedChoose`, `ChooseByMeasure` | Sampling in proportion to a measure |
| `Map`, `MapMut` | Forward key-to-value lookup and mutation |
| `ValueIndexed`, `ValueIndexedMut` | Reverse lookup from values |
| `UniqueValueMap`, `SharedValueMap` | Whether a value has one or many keys |
| `Grouping`, `GroupSizes` | Label groups and queries by group size |
| `Partitioned` | Numbered groups that together cover every member |
| `BalancedPartition` | Partition whose group sizes never differ by more than one |
| `Capacity` | Reserve/shrink allocated space |
| `Bounded`, `EvictingInsert` | Hard limit and insertion that reports displacement |
| `RangeQuery` | Ordered query between bounds |
| `Pending` | Work held until something releases it, then collected in a batch |
| `Kinded` | A value type that reports which of its logical kinds a value is |
| `check_kinds`, `KindContract` | Verifies a hand-written value type against the `Kinded` contract, and how it failed |
| `PriorityQueueLike` | Common min-priority queue behavior |
| `HandleStore` | Values owned behind opaque typed handles |
| `StochasticStream` (in `random`) | Owned randomness a structure can draw from and that advances itself |
| `DeterministicOrder`, `DeterministicMapOrder` | Same operations replay within one format/executable version |
| `CanonicalOrder`, `CanonicalMapOrder` | Iteration determined by logical contents alone |
| `StableHash` | Architecture-independent stable identity for one value |
| `ContentHashable` | Content-derived hash for a whole structure |
| `Key`, `Element` | Common `Eq + Hash` and `Clone + Eq + Hash` bounds |

The crate also implements appropriate collection traits for `Vec` and
`VecDeque`, and map traits for standard `HashMap`, so generic algorithms do not
need custom wrappers merely to participate in the capability system.

## Representation and performance rules of thumb

1. Start with semantic correctness: choose by relationship and invariants, not
   by a hoped-for micro-optimization.
2. Use `BitSet`/`BitGrid` only when the highest addressable index or volume is
   controlled; sparse enormous numeric IDs are a poor fit.
3. Use `Palette` for low distinct-count data, run length for adjacency, and
   `CompactSequence` when the winning pattern can change over time.
4. Batch mutations before `Palette::compact` or `CompactSequence::compact`.
   Compaction intentionally scans/rebuilds storage.
5. Prefer `BucketQueue` over `PriorityQueue` only for a genuinely small bounded
   priority domain.
6. Reverse-indexed structures spend memory to make reverse queries cheap. Use a
   plain map if those queries are rare.
7. Group-size indices are useful only when smallest/largest/size-filtered group
   queries are real requirements; otherwise `MultiMap`/`UniqueMultiMap` are
   simpler.
8. `DisjointSets` is excellent for additions and joins, but not dynamic graph
   deletion.
9. Do not derive persistent generation behavior from hash-table traversal.
   Canonicalize first or use a structure that promises canonical order.
10. Treat `check_invariants()` methods as debug/test diagnostics, not hot-loop
    operations.

## Public inventory and aliases

This checklist is useful when maintaining the document.

- Sets: `Set`, `NestedSet`, `nested_set::Iter`, `BitSet`, `DisjointSets`,
  `DisjointGroup`, `LabelIndexedSet`, `SubscriptionSet`.
- Ordered/time sequences: `OrderedSet`, `WorkQueue`, `LabeledOrderedSet`,
  `KeyedOrderedSet`, `BoundedOrderedSet`, `RingBuffer`, `PriorityQueue`,
  `BucketQueue`, `Scheduler`, `UniqueScheduler`, `RunLengthSequence`.
- Rotas: `Rota`, `MultiRota`, `OrderedRota`, `OrderedMultiRota`.
- Drawn recurrences: `Cadence`, `OnBacklog`, `CadenceId`, `Firing`,
  `StochasticScheduler`, `UniqueStochasticScheduler`.
- Sparse/measured collections: `SparseSequence`, `SparseSetSequence`,
  `MultiSet`, `WeightedSet`, `FuzzySet`, `MEMBERSHIP_TOLERANCE`.
- Maps: `BiMap`, `Overwritten`, `UniqueMultiMap`, `OneToManyMap`, `MultiMap`,
  `ManyToManyMap`, `GroupedSingleMap`, `PartitionMap`, `GroupedMultiMap`,
  `LabelMap`, `PairMap`, `IntervalMap`, `LayeredMap`, `LruCache`, `Evicted`,
  `SetKeyMap`.
- Indices: `TagIndex`, `PropertyQuery`, `ConditionIndex`, `OnSatisfied`, `Expr`,
  `PropertyKind`, `Cardinality`, `Uniqueness`, `FuseMode`, `FuseError`,
  `SchemaError`, `Comparison`, `Gate`, `PropertyStore`,
  `property_query::RefIter`.
- Storage: `Palette`, `CompactSequence`, `CompactKind`, `Grid`, `Grid2`,
  `Grid3`, `Grid4`, `BitGrid`, `BitGrid2`, `BitGrid3`, `BitGrid4`, `SlotMap`.
- Hashing utilities: `FastHasher`, `FastBuildHasher`, `FastHashMap`,
  `FastHashSet`, `hash_one`, `unordered_hash`, `stable_hash_ordered`,
  `stable_hash_unordered`.
- Names: `Name`, `NameCollision`, `name!`.
- Schema: `Kinded`, `KindContract`, `check_kinds`.

Internal helpers such as `Tally`, `SparseCore`, bucket size indices, rota group
balancing, and compact run internals are deliberately omitted: they are implementation details rather
than structures callers should select directly.
