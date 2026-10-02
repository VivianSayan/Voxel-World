#![warn(missing_docs)]

//! General-purpose data structures, ported from the Godot `Utils/Structure`
//! scripts.
//!
//! - `traits`: the capability traits the structures share. Generic code can
//!   require exactly what it needs, such as `SetAlgebra` or `GroupSizes`.
//! - `collections`: element containers (sets, sequences, measured, sparse).
//! - `mappings`: key -> value relationships (single, multi, grouped).
//! - `indices`: `TagIndex`, `PropertyQuery` and `ConditionIndex`.
//! - `storage`: handle stores, compact sequences and checked dense grids.
//!
//! `use crate::structures::prelude::*;` brings every structure and
//! trait into scope. Much of the shared behavior (set algebra, sorting,
//! random choice, grouping) lives only on the traits.
//!
//! Single-index keys need `Eq + Hash` (`Key`); structures that duplicate keys
//! into secondary indices additionally need `Clone` (`Element`). Hashing uses
//! a fixed fast hasher. Random picks from unordered containers depend on their
//! iteration order; canonicalize inputs before using them in world generation.

pub mod collections;
pub mod hashing;
pub mod name;
pub use name::{Name, NameCollision};
pub mod indices;
pub mod mappings;
pub mod sampling;
pub mod storage;
pub mod traits;

pub(crate) mod buckets;

/// Convenient re-exports of every public structure and capability trait.
///
/// Import this module with `use crate::structures::prelude::*;` when
/// name collisions are not a concern.
pub mod prelude {
    pub use super::collections::{
        BoundedOrderedSet, Cadence, CadenceId, Firing, FuzzySet, KeyedOrderedSet, LabelIndexedSet,
        LabeledOrderedSet, MEMBERSHIP_TOLERANCE, MultiRota, MultiSet, NestedSet, OnBacklog,
        OrderedMultiRota, OrderedRota, OrderedSet, PriorityQueue, RingBuffer, Rota, Scheduler, Set,
        SparseSequence, SparseSetSequence, StochasticScheduler, SubscriptionSet, UniqueScheduler,
        UniqueStochasticScheduler, WeightedSet, WorkQueue,
    };
    pub use super::indices::{
        Cardinality, Comparability, Comparison, ConditionIndex, Expr, FuseError, FuseMode,
        OnSatisfied, PropertyKind, PropertyQuery, SchemaError, TagIndex, Uniqueness,
    };
    pub use super::mappings::{
        BiMap, GroupedMultiMap, GroupedSingleMap, LabelMap, ManyToManyMap, MultiMap, OneToManyMap,
        Overwritten, PairMap, PartitionMap, SetKeyMap, UniqueMultiMap,
    };
    pub use super::storage::{
        BitGrid, BitGrid2, BitGrid3, BitGrid4, CompactKind, CompactSequence, Grid, Grid2, Grid3,
        Grid4, Palette, SlotMap,
    };
    pub use super::traits::{
        BalancedPartition, Bounded, CanonicalMapOrder, CanonicalOrder, Capacity, Choose,
        ChooseByMeasure, ChooseMut, Collection, CollectionInsert, CollectionMut, CollectionRemove,
        ContentHashable, DeterministicMapOrder, DeterministicOrder, Element, EvictingInsert,
        FixedCapacity, GroupSizes, Grouping, HandleStore, InsertAt, Key, Map, MapMut, Measured,
        MeasuredMut, Partitioned, PriorityQueueLike, RangeQuery, Reorder, Sequence, SequenceMut, Shuffle,
        SetAlgebra, SharedValueMap, SparseIndexed, StableHash, UniqueCollection, UniqueValueMap,
        ValueCollection, ValueIndexed, ValueIndexedMut, ValueSetAlgebra, WeightedChoose,
    };
}
