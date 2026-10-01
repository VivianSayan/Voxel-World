//! Collections whose elements sit at dense positions.

pub mod bounded_ordered_set;
pub mod bucket_queue;
pub mod cadence;
pub mod labeled_ordered_set;
pub mod ordered_set;
pub mod priority_queue;
pub mod ring_buffer;
pub mod run_length;
pub mod scheduler;
pub mod stochastic_scheduler;
pub mod unique_scheduler;
pub mod unique_stochastic_scheduler;

pub use bounded_ordered_set::BoundedOrderedSet;
pub use bucket_queue::BucketQueue;
pub use cadence::{Cadence, OnBacklog};
pub use labeled_ordered_set::LabeledOrderedSet;
pub use ordered_set::OrderedSet;
pub use priority_queue::PriorityQueue;
pub use ring_buffer::RingBuffer;
pub use run_length::RunLengthSequence;
pub use scheduler::Scheduler;
pub use stochastic_scheduler::{CadenceId, Firing, StochasticScheduler};
pub use unique_scheduler::UniqueScheduler;
pub use unique_stochastic_scheduler::UniqueStochasticScheduler;

/// An ordered set with one globally unique key per element.
pub type KeyedOrderedSet<T, K> = LabeledOrderedSet<T, K>;

/// Unique FIFO work awaiting processing.
pub type WorkQueue<T> = OrderedSet<T>;
