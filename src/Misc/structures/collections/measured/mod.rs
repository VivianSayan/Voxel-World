//! Collections that attach a count, weight or membership to each element.

pub mod fuzzy_set;
pub mod multi_set;
pub mod weighted_set;

pub(crate) mod tally;

pub use fuzzy_set::{FuzzySet, MEMBERSHIP_TOLERANCE};
pub use multi_set::MultiSet;
pub use weighted_set::WeightedSet;
