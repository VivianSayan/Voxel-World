//! Collections whose elements sit at dense positions.

pub mod labeled_ordered_set;
pub mod ordered_set;
pub mod priority_queue;
pub mod ring_buffer;

pub use labeled_ordered_set::LabeledOrderedSet;
pub use ordered_set::OrderedSet;
pub use priority_queue::PriorityQueue;
pub use ring_buffer::RingBuffer;

/// An ordered set with one globally unique key per element.
pub type KeyedOrderedSet<T, K> = LabeledOrderedSet<T, K>;
