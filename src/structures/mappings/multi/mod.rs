//! Maps with a set of values per key.

pub mod multi_map;
pub mod unique_multi_map;

pub use multi_map::MultiMap;
pub use unique_multi_map::UniqueMultiMap;

/// A one-to-many map in which every value has exactly one owner.
pub type OneToManyMap<K, V> = UniqueMultiMap<K, V>;

/// A many-to-many relation indexed in both directions.
pub type ManyToManyMap<K, V> = MultiMap<K, V>;
