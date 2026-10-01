//! The capability traits every structure is built around.
//!
//! Collections:
//! - `Collection` / `CollectionMut`: size, membership, iteration, add/remove.
//! - `ValueCollection`: the same, for containers whose elements are computed
//!   rather than stored and so cannot be borrowed.
//! - `UniqueCollection`: no repeats.
//! - `SetAlgebra`: union, intersection, difference, subset tests.
//! - `Choose` / `ChooseMut`: random selection and removal.
//! - `Sequence` / `SequenceMut` / `InsertAt`: dense positions.
//! - `Reorder`: sorting, reversing, shuffling.
//! - `FixedCapacity`: bounded size.
//! - `SparseIndexed`: integer indices with gaps.
//! - `Measured` / `MeasuredMut`: a count, weight or membership per element.
//! - `WeightedChoose`: picking in proportion to those quantities.
//!
//! Properties any of them may claim:
//! - `Capacity`: room to grow, reserved and released on request.
//! - `Bounded` / `EvictingInsert`: a hard limit and what insertion displaces.
//! - `DeterministicOrder` / `DeterministicMapOrder`: iteration that replays in
//!   one executable/data version; `CanonicalOrder` for content-defined order.
//! - `ContentHashable`: one hash standing for everything held.
//! - `StableHash`: an architecture-independent identity for a single value.
//! - `RangeQuery`: everything between two bounds, in order.
//! - `PriorityQueueLike`: work served by priority across different backends.
//! - `HandleStore`: owned values reached through opaque handles.
//!
//! Mappings: `Map`, `MapMut`, `ValueIndexed`, `ValueIndexedMut`,
//! `UniqueValueMap`, `SharedValueMap`.
//!
//! Groups: `Grouping`, `GroupSizes`.
//!
//! The std traits (`FromIterator`, `Extend`, `IntoIterator`, `Index`,
//! `PartialEq`, `Hash`, operators) are implemented wherever they fit.

pub mod capacity;
pub mod collection;
pub mod content;
pub mod determinism;
pub mod grouping;
pub mod map;
pub mod measured;
pub(crate) mod operators;
pub mod partition;
pub mod priority;
pub mod query;
pub mod readiness;
pub mod schema;
pub mod sequence;
pub mod stable;
pub mod storage;

pub use capacity::{Bounded, Capacity, EvictingInsert};
pub use collection::{
    Choose, ChooseMut, Collection, CollectionInsert, CollectionMut, CollectionRemove, Element, Key,
    SetAlgebra, UniqueCollection, ValueCollection, ValueSetAlgebra,
};
pub use content::ContentHashable;
pub use determinism::{
    CanonicalMapOrder, CanonicalOrder, DeterministicMapOrder, DeterministicOrder,
};
pub use grouping::{GroupSizes, Grouping};
pub use map::{Map, MapMut, SharedValueMap, UniqueValueMap, ValueIndexed, ValueIndexedMut};
pub use measured::{ChooseByMeasure, Measured, MeasuredMut, WeightedChoose};
pub use partition::{BalancedPartition, Partitioned};
pub use priority::PriorityQueueLike;
pub use query::RangeQuery;
pub use readiness::Pending;
pub use schema::{KindContract, Kinded, check_kinds};
pub use sequence::{FixedCapacity, InsertAt, Reorder, Sequence, SequenceMut, SparseIndexed};
pub use stable::{StableHash, stable_hash_ordered, stable_hash_unordered};
pub use storage::HandleStore;
