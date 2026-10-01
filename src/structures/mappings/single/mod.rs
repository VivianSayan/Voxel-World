//! Maps with one value per key.

pub mod bi_map;
pub mod interval_map;
pub mod layered_map;
pub mod lru_cache;
pub mod pair_map;

use crate::structures::collections::sets::set::Set;
use crate::structures::hashing::FastHashMap;
pub use bi_map::{BiMap, Overwritten};
pub use interval_map::IntervalMap;
pub use layered_map::LayeredMap;
pub use lru_cache::{Evicted, LruCache};
pub use pair_map::PairMap;

/// A map keyed by set membership rather than by an identity.
///
/// The Godot original needed string keys to compare structures by content;
/// Rust hashes a set by its contents already, so this is only a name for the
/// shape. There is no wrapper behind it: `SingleMap` used to be one, and was
/// removed once it turned out to add nothing to `FastHashMap` but a second set
/// of names for the same methods.
pub type SetKeyMap<K, V> = FastHashMap<Set<K>, V>;
