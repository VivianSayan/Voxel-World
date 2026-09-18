//! Element-focused collections, grouped by family:
//!
//! | Family      | Structures                                                   |
//! |-------------|--------------------------------------------------------------|
//! | `sets`      | `Set`, `NestedSet`, `LabelIndexedSet`, `SubscriptionSet`     |
//! | `sequences` | `OrderedSet`, `LabeledOrderedSet`, `PriorityQueue`, `RingBuffer` |
//! | `measured`  | `MultiSet`, `WeightedSet`, `FuzzySet`                        |
//! | `sparse`    | `SparseSequence`, `SparseSetSequence`                        |

pub mod measured;
pub mod sequences;
pub mod sets;
pub mod sparse;

pub use measured::{FuzzySet, MEMBERSHIP_TOLERANCE, MultiSet, WeightedSet};
pub use sequences::{KeyedOrderedSet, LabeledOrderedSet, OrderedSet, PriorityQueue, RingBuffer};
pub use sets::{LabelIndexedSet, NestedSet, Set, SubscriptionSet};
pub use sparse::{SparseSequence, SparseSetSequence};
