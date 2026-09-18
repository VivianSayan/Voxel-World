//! Unordered collections of distinct elements.

pub mod label_indexed_set;
pub mod nested_set;
pub mod set;
pub mod subscription_set;

pub use label_indexed_set::LabelIndexedSet;
pub use nested_set::NestedSet;
pub use set::Set;
pub use subscription_set::SubscriptionSet;
