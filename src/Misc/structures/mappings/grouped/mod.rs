//! Maps from elements to labels that also track each label's group.

pub mod grouped_multi_map;
pub mod grouped_single_map;

pub use grouped_multi_map::GroupedMultiMap;
pub use grouped_single_map::GroupedSingleMap;

/// Elements partitioned into exactly one labeled group each.
pub type PartitionMap<E, L> = GroupedSingleMap<E, L>;

/// Elements carrying any number of labels, indexed in both directions.
pub type LabelMap<E, L> = GroupedMultiMap<E, L>;
