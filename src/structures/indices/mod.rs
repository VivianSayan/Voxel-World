//! Query structures built from the collections and mappings.

pub mod condition_index;
pub mod gate;
pub mod property_query;
pub mod property_store;
pub mod tag_index;

pub use condition_index::{ConditionIndex, OnSatisfied};
pub use gate::Gate;
pub use property_query::{
    Cardinality, Comparability, Comparison, Expr, FuseError, FuseMode, PropertyKind, PropertyQuery,
    SchemaError, Uniqueness,
};
pub use property_store::PropertyStore;
pub use tag_index::TagIndex;
