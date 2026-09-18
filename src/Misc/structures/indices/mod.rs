//! Query structures built from the collections and mappings.

pub mod property_query;
pub mod tag_index;

pub use property_query::{
    Cardinality, Expr, FuseError, FuseMode, PropertyKind, PropertyQuery, Uniqueness,
};
pub use tag_index::TagIndex;
