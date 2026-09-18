//! Sequences stored at arbitrary integer indices, with gaps.

pub mod sparse_sequence;
pub mod sparse_set_sequence;

pub(crate) mod sparse_core;

pub use sparse_sequence::SparseSequence;
pub use sparse_set_sequence::SparseSetSequence;
