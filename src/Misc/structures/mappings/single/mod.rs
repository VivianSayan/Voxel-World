//! Maps with one value per key.

pub mod bi_map;
pub mod pair_map;
pub mod single_map;

pub use bi_map::{BiMap, Overwritten};
pub use pair_map::PairMap;
pub use single_map::{SetKeyMap, SingleMap};
