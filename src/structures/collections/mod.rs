//! Element-focused collections, grouped by family:
//!
//! | Family      | Structures                                                   |
//! |-------------|--------------------------------------------------------------|
//! | `sets`      | `Set`, `NestedSet`, `BitSet`, `DisjointSets`, `LabelIndexedSet`, `SubscriptionSet` |
//! | `sequences` | `OrderedSet`, `LabeledOrderedSet`, `PriorityQueue`, `BucketQueue`, `RingBuffer`, `RunLengthSequence`, `Scheduler` |
//! | `standard`  | the traits above, on `Vec` and `VecDeque`                    |
//! | `measured`  | `MultiSet`, `WeightedSet`, `FuzzySet`                        |
//! | `sparse`    | `SparseSequence`, `SparseSetSequence`                        |
//!
//! `properties` is where the structures claim the traits that are promises
//! rather than operations: reserving room, iterating in an order that replays,
//! and hashing by contents.

pub mod measured;
pub mod properties;
pub mod rota;
pub mod sequences;
pub mod sets;
pub mod sparse;
mod tick_scheduler;
pub use tick_scheduler::TickScheduler;
pub mod standard;

pub use measured::{FuzzySet, MEMBERSHIP_TOLERANCE, MultiSet, WeightedSet};
pub use rota::{MultiRota, OrderedMultiRota, OrderedRota, Rota, TickRota, TickRotaUpdate};
pub use sequences::{
    BoundedOrderedSet, BucketQueue, Cadence, CadenceId, Firing, KeyedOrderedSet, LabeledOrderedSet,
    OnBacklog, OrderedSet, PriorityQueue, RingBuffer, RunLengthSequence, Scheduler,
    StochasticScheduler, UniqueScheduler, UniqueStochasticScheduler, WorkQueue,
};
pub use sets::{BitSet, DisjointSets, LabelIndexedSet, NestedSet, Set, SubscriptionSet};
pub use sparse::{SparseSequence, SparseSetSequence};
