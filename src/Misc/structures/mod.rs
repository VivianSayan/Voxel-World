#![warn(missing_docs)]

//! General-purpose data structures, ported from the Godot `Utils/Structure`
//! scripts.
//!
//! - `traits`: the capability traits the structures share. Generic code can
//!   require exactly what it needs, such as `SetAlgebra` or `GroupSizes`.
//! - `collections`: element containers (sets, sequences, measured, sparse).
//! - `mappings`: key -> value relationships (single, multi, grouped).
//! - `indices`: `TagIndex` and `PropertyQuery`.
//!
//! `use crate::misc::structures::prelude::*;` brings every structure and
//! trait into scope. Much of the shared behavior (set algebra, sorting,
//! random choice, grouping) lives only on the traits.
//!
//! Single-index keys need `Eq + Hash` (`Key`); structures that duplicate keys
//! into secondary indices additionally need `Clone` (`Element`). Hashing uses
//! a fast, deterministic hasher, so seeded random picks replay exactly.

pub mod collections;
pub mod hashing;
pub mod indices;
pub mod mappings;
pub mod sampling;
pub mod traits;

pub(crate) mod buckets;

/// Convenient re-exports of every public structure and capability trait.
///
/// Import this module with `use crate::misc::structures::prelude::*;` when
/// name collisions are not a concern.
pub mod prelude {
    pub use super::collections::{
        FuzzySet, KeyedOrderedSet, LabelIndexedSet, LabeledOrderedSet, MEMBERSHIP_TOLERANCE,
        MultiSet, NestedSet, OrderedSet, PriorityQueue, RingBuffer, Set, SparseSequence,
        SparseSetSequence, SubscriptionSet, WeightedSet,
    };
    pub use super::indices::{
        Cardinality, Expr, FuseError, FuseMode, PropertyKind, PropertyQuery, TagIndex, Uniqueness,
    };
    pub use super::mappings::{
        BiMap, GroupedMultiMap, GroupedSingleMap, LabelMap, ManyToManyMap, MultiMap, OneToManyMap,
        Overwritten, PairMap, PartitionMap, SetKeyMap, SingleMap, UniqueMultiMap,
    };
    pub use super::traits::{
        Choose, ChooseMut, Collection, CollectionInsert, CollectionMut, CollectionRemove, Element,
        FixedCapacity, GroupSizes, Grouping, InsertAt, Key, Map, MapMut, Measured, MeasuredMut,
        Reorder, Sequence, SequenceMut, SetAlgebra, SharedValueMap, SparseIndexed,
        UniqueCollection, UniqueValueMap, ValueIndexed, ValueIndexedMut,
    };
}
