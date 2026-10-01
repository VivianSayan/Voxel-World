//! Unordered collections of distinct elements.

pub mod bit_set;
pub mod disjoint_sets;
pub mod label_indexed_set;
pub mod nested_set;
pub mod set;
pub mod subscription_set;

pub use bit_set::BitSet;
pub use disjoint_sets::DisjointSets;
pub use label_indexed_set::LabelIndexedSet;
pub use nested_set::NestedSet;
pub use set::Set;
pub use subscription_set::SubscriptionSet;
