//! Key -> value relationships.
//!
//! | Map                 | key -> value | value -> key | Family    |
//! |---------------------|--------------|--------------|-----------|
//! | `FastHashMap`       | one          | not indexed  | (standard)|
//! | `BiMap`             | one          | one          | `Single`  |
//! | `UniqueMultiMap`    | many         | one          | `Multi`   |
//! | `MultiMap`          | many         | many         | `Multi`   |
//! | `GroupedSingleMap`  | one          | many + sizes | `Grouped` |
//! | `GroupedMultiMap`   | many         | many + sizes | `Grouped` |

pub mod grouped;
pub mod multi;
pub mod single;
mod standard;

pub use grouped::{GroupedMultiMap, GroupedSingleMap, LabelMap, PartitionMap};
pub use multi::{ManyToManyMap, MultiMap, OneToManyMap, UniqueMultiMap};
pub use single::{
    BiMap, Evicted, IntervalMap, LayeredMap, LruCache, Overwritten, PairMap, SetKeyMap,
};
