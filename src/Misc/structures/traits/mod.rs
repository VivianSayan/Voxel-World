//! The capability traits every structure is built around.
//!
//! Collections:
//! - `Collection` / `CollectionMut`: size, membership, iteration, add/remove.
//! - `UniqueCollection`: no repeats.
//! - `SetAlgebra`: union, intersection, difference, subset tests.
//! - `Choose` / `ChooseMut`: random selection and removal.
//! - `Sequence` / `SequenceMut` / `InsertAt`: dense positions.
//! - `Reorder`: sorting, reversing, shuffling.
//! - `FixedCapacity`: bounded size.
//! - `SparseIndexed`: integer indices with gaps.
//! - `Measured` / `MeasuredMut`: a count, weight or membership per element.
//!
//! Mappings: `Map`, `MapMut`, `ValueIndexed`, `ValueIndexedMut`,
//! `UniqueValueMap`, `SharedValueMap`.
//!
//! Groups: `Grouping`, `GroupSizes`.
//!
//! The std traits (`FromIterator`, `Extend`, `IntoIterator`, `Index`,
//! `PartialEq`, `Hash`, operators) are implemented wherever they fit.

pub mod collection;
pub mod grouping;
pub mod map;
pub mod measured;
pub(crate) mod operators;
pub mod sequence;

pub use collection::{
    Choose, ChooseMut, Collection, CollectionInsert, CollectionMut, CollectionRemove, Element, Key,
    SetAlgebra, UniqueCollection,
};
pub use grouping::{GroupSizes, Grouping};
pub use map::{Map, MapMut, SharedValueMap, UniqueValueMap, ValueIndexed, ValueIndexedMut};
pub use measured::{Measured, MeasuredMut};
pub use sequence::{FixedCapacity, InsertAt, Reorder, Sequence, SequenceMut, SparseIndexed};
