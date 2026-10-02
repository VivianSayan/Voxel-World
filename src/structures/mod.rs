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
//! Import common structures directly from this module and capability traits
//! from [`traits`]. The optional [`prelude`] gathers the most frequently used
//! names. Much of the shared behavior (set algebra, sorting,
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
pub mod properties;
pub mod sampling;
pub mod scheduling;
pub mod storage;
pub mod traits;

pub(crate) mod buckets;

pub use collections::{
    BitSet, BoundedOrderedSet, BucketQueue, Cadence, CadenceId, DisjointSets, Firing, FuzzySet,
    KeyedOrderedSet, LabelIndexedSet, LabeledOrderedSet, MEMBERSHIP_TOLERANCE, MultiRota, MultiSet,
    NestedSet, OnBacklog, OrderedMultiRota, OrderedRota, OrderedSet, PriorityQueue, RingBuffer,
    Rota, RunLengthSequence, Scheduler, Set, SparseSequence, SparseSetSequence,
    StochasticScheduler, SubscriptionSet, TickRota, TickRotaUpdate, TickScheduler, UniqueScheduler,
    UniqueStochasticScheduler, WeightedSet, WorkQueue,
};
pub use hashing::{FastHashMap, FastHashSet};
pub use indices::{
    Cardinality, Comparability, Comparison, ConditionIndex, Expr, FuseError, FuseMode, Gate,
    OnSatisfied, PropertyKind, PropertyQuery, PropertyStore, SchemaError, TagIndex, Uniqueness,
};
pub use mappings::{
    BiMap, Evicted, GroupedMultiMap, GroupedSingleMap, IntervalMap, LabelMap, LayeredMap, LruCache,
    ManyToManyMap, MultiMap, OneToManyMap, Overwritten, PairMap, PartitionMap, SetKeyMap,
    UniqueMultiMap,
};
pub use storage::{
    BitGrid, BitGrid2, BitGrid3, BitGrid4, CompactKind, CompactSequence, Grid, Grid2, Grid3, Grid4,
    Palette, SlotMap,
};

/// Common structures and capability traits for exploratory code.
///
/// Import this module with `use crate::structures::prelude::*;` when
/// name collisions are not a concern. Use explicit imports from this module or
/// its families for stable, readable production code.
pub mod prelude {
    pub use super::collections::{
        BoundedOrderedSet, Cadence, CadenceId, Firing, FuzzySet, KeyedOrderedSet, LabelIndexedSet,
        LabeledOrderedSet, MEMBERSHIP_TOLERANCE, MultiRota, MultiSet, NestedSet, OnBacklog,
        OrderedMultiRota, OrderedRota, OrderedSet, PriorityQueue, RingBuffer, Rota, Scheduler, Set,
        SparseSequence, SparseSetSequence, StochasticScheduler, SubscriptionSet, TickRota,
        TickRotaUpdate, TickScheduler, UniqueScheduler, UniqueStochasticScheduler, WeightedSet,
        WorkQueue,
    };
    pub use super::indices::{
        Cardinality, Comparability, Comparison, ConditionIndex, Expr, FuseError, FuseMode, Gate,
        OnSatisfied, PropertyKind, PropertyQuery, PropertyStore, SchemaError, TagIndex, Uniqueness,
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
        FixedCapacity, GroupSizes, Grouping, HandleStore, InsertAt, Key, Kinded, Map, MapMut,
        Measured, MeasuredMut, Partitioned, Pending, PriorityQueueLike, RangeQuery, Reorder,
        Sequence, SequenceMut, SetAlgebra, SharedValueMap, Shuffle, SparseIndexed, StableHash,
        UniqueCollection, UniqueValueMap, ValueCollection, ValueIndexed, ValueIndexedMut,
        ValueSetAlgebra, WeightedChoose,
    };
}
