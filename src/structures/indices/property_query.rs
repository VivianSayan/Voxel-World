//! Assigns properties to elements and answers queries over them.
//!
//! Each property has a kind that decides its storage:
//!
//! | Kind            | Per element     | Per value      | Stored in          |
//! |-----------------|-----------------|----------------|--------------------|
//! | `FLAG`          | present or not  | -              | `Set`              |
//! | `SINGLE`        | one value       | many elements  | `GroupedSingleMap` |
//! | `UNIQUE_SINGLE` | one value       | one element    | `BiMap`            |
//! | `MULTI`         | many values     | many elements  | `GroupedMultiMap`  |
//! | `UNIQUE_MULTI`  | many values     | one element    | `UniqueMultiMap`   |
//!
//! Queries are `Expr` trees (`has`, `is`, `and`, `or`, `negate`). Every
//! evaluated node is cached, and a write to a property drops only the
//! cached results that depend on it. `and`/`or` children are a set, so
//! the same query written in a different order shares a cache entry.
//!
//! Flag properties carry no value: set them with `set_flag` and query them
//! with `Expr::has`.

use crate::structures::buckets::{add_to_bucket, remove_from_bucket};
use crate::structures::collections::sets::set::Set;
use crate::structures::hashing::{FastHashMap, FastHashSet};
use crate::structures::mappings::grouped::grouped_multi_map::GroupedMultiMap;
use crate::structures::mappings::grouped::grouped_single_map::GroupedSingleMap;
use crate::structures::mappings::multi::unique_multi_map::UniqueMultiMap;
use crate::structures::mappings::single::bi_map::BiMap;
use crate::structures::traits::{Collection, Element, Grouping, Kinded, SetAlgebra};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Number of values an element may hold for one property.
pub enum Cardinality {
    /// Presence alone is meaningful; the property carries no value.
    Flag,
    /// At most one value per element.
    Single,
    /// Any number of distinct values per element.
    Multi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Whether the same property value may be held by multiple elements.
pub enum Uniqueness {
    /// A value may be shared by any number of elements.
    Shared,
    /// A value may belong to at most one element.
    Unique,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Storage and constraint policy for a registered property.
pub enum PropertyKind {
    /// Boolean presence with no associated value.
    Flag,
    /// One value per element; values may be shared.
    Single,
    /// One value per element and at most one element per value.
    UniqueSingle,
    /// Many values per element; values may be shared.
    Multi,
    /// Many values per element, but each value has at most one owner.
    UniqueMulti,
}

impl PropertyKind {
    /// Uppercase compatibility alias for [`PropertyKind::Flag`].
    pub const FLAG: Self = Self::Flag;
    /// Uppercase compatibility alias for [`PropertyKind::Single`].
    pub const SINGLE: Self = Self::Single;
    /// Uppercase compatibility alias for [`PropertyKind::UniqueSingle`].
    pub const UNIQUE_SINGLE: Self = Self::UniqueSingle;
    /// Uppercase compatibility alias for [`PropertyKind::Multi`].
    pub const MULTI: Self = Self::Multi;
    /// Uppercase compatibility alias for [`PropertyKind::UniqueMulti`].
    pub const UNIQUE_MULTI: Self = Self::UniqueMulti;

    /// Returns whether this kind stores no value, one value, or many values.
    pub const fn cardinality(self) -> Cardinality {
        match self {
            Self::Flag => Cardinality::Flag,
            Self::Single | Self::UniqueSingle => Cardinality::Single,
            Self::Multi | Self::UniqueMulti => Cardinality::Multi,
        }
    }

    /// Returns whether values may be shared between elements.
    pub const fn uniqueness(self) -> Uniqueness {
        match self {
            Self::UniqueSingle | Self::UniqueMulti => Uniqueness::Unique,
            Self::Flag | Self::Single | Self::Multi => Uniqueness::Shared,
        }
    }
}

/// Whether a property's values have a meaningful order.
///
/// Declared when the property is registered, rather than worked out from the
/// value type. A [`PartialOrd`] implementation cannot be asked: `derive` gives
/// every enum an order by declaration position, so a categorical kind would
/// answer "yes, ordered" and then compare labels by the accident of how they
/// were written down. Saying which is intended is the only reliable way to know.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comparability {
    /// The values run from smaller to larger, so `<` and `>` mean something:
    /// a depth, a temperature, a count.
    Ordered,
    /// The values are labels with no order between them: a weather, a material,
    /// a phase. A comparison against one is refused.
    Categorical,
}

/// How `fuse` treats properties both sides have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuseMode {
    /// Single values already set here are kept; multi values are added to.
    Merge,
    /// The other side's values replace these.
    Overwrite,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Schema conflict returned by [`PropertyQuery::fuse`].
///
/// `P` is the caller's property-identifier type.
pub struct FuseError<P> {
    /// Property whose registered kinds disagree.
    pub property: P,
    /// Kind already registered in the destination query.
    pub current: PropertyKind,
    /// Incompatible kind registered in the incoming query.
    pub incoming: PropertyKind,
}

/// A write or a query that the schema does not allow.
///
/// `P` is the caller's property-identifier type and `K` its value-kind type,
/// which is [`Kinded::Kind`] for the store's value type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchemaError<P, K> {
    /// A property was used before it was registered.
    ///
    /// A schema-aware store does not invent properties from the first write:
    /// the kind and the value kind have to be declared, and guessing them from
    /// whatever arrived first is how a typo becomes a column.
    Unregistered {
        /// The property that was used.
        property: P,
    },
    /// A value of the wrong kind was written to, or compared against, a
    /// property.
    WrongValueKind {
        /// The property being written or queried.
        property: P,
        /// The kind the property was registered with.
        expected: K,
        /// The kind of the value supplied.
        found: K,
    },
    /// A value was given for a property that holds none.
    FlagTakesNoValue {
        /// The flag property.
        property: P,
    },
    /// A comparison was made against a kind that has no order.
    ///
    /// `Health < 5` is a question; `Weather < Rain` is not, and neither is a
    /// comparison between two kinds. Caught when the expression is built, so
    /// that it is a refusal rather than a condition that never holds.
    NotOrdered {
        /// The property being compared.
        property: P,
        /// The kind of the value it was compared against.
        kind: K,
    },
    /// A property was registered again as ordered where it was categorical, or
    /// the other way about.
    ComparabilityConflict {
        /// The property being re-registered.
        property: P,
        /// What it was registered as, and keeps.
        current: Comparability,
        /// What was offered now.
        incoming: Comparability,
    },
    /// A property was registered again for a different value kind.
    ///
    /// Separate from [`SchemaError::WrongValueKind`], which is a *value* that
    /// does not fit a property. This is an attempt to change what the property
    /// accepts, which would leave everything already stored under it answering
    /// to a contract it was never checked against.
    ValueKindConflict {
        /// The property being re-registered.
        property: P,
        /// The value kind it was registered with, and keeps.
        expected: K,
        /// The value kind offered now.
        incoming: K,
    },
    /// A property was registered twice with different terms.
    Conflict {
        /// The property registered twice.
        property: P,
        /// The kind it already has.
        current: PropertyKind,
        /// The kind offered now.
        incoming: PropertyKind,
    },
}

impl<P: fmt::Debug, K: fmt::Debug> fmt::Display for SchemaError<P, K> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unregistered { property } => {
                write!(formatter, "property {property:?} is not registered")
            }
            Self::WrongValueKind {
                property,
                expected,
                found,
            } => write!(
                formatter,
                "property {property:?} holds {expected:?} values, not {found:?}"
            ),
            Self::FlagTakesNoValue { property } => {
                write!(
                    formatter,
                    "property {property:?} is a flag and holds no value"
                )
            }
            Self::NotOrdered { property, kind } => write!(
                formatter,
                "property {property:?} holds {kind:?} values, which have no order to compare"
            ),
            Self::ComparabilityConflict {
                property,
                current,
                incoming,
            } => write!(
                formatter,
                "property {property:?} is registered as {current:?} and cannot become {incoming:?}"
            ),
            Self::ValueKindConflict {
                property,
                expected,
                incoming,
            } => write!(
                formatter,
                "property {property:?} already accepts {expected:?} values and cannot be \
                 re-registered for {incoming:?}"
            ),
            Self::Conflict {
                property,
                current,
                incoming,
            } => write!(
                formatter,
                "property {property:?} is registered as {current:?}, not {incoming:?}"
            ),
        }
    }
}

impl<P: fmt::Debug, K: fmt::Debug> std::error::Error for SchemaError<P, K> {}

/// How a property's value is compared against one in an expression.
///
/// Kept as one operator alongside a value rather than four separate expression
/// nodes, so that adding another costs one arm here instead of one arm in every
/// walk over an expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comparison {
    /// Strictly below the given value.
    Less,
    /// Below or equal to it.
    AtMost,
    /// Strictly above it.
    Greater,
    /// Above or equal to it.
    AtLeast,
}

impl Comparison {
    /// Whether `held` stands in this relation to `bound`.
    ///
    /// `false` when the two do not compare at all, which for a tagged value
    /// type means they are of different kinds or of a kind with no order. A
    /// well-formed expression cannot reach that, because
    /// [`PropertyQuery::validate`] refuses a comparison against an unordered
    /// kind; it is the honest answer for one built by hand regardless.
    pub fn holds<V: PartialOrd>(self, held: &V, bound: &V) -> bool {
        let Some(order) = held.partial_cmp(bound) else {
            return false;
        };

        match self {
            Self::Less => order.is_lt(),
            Self::AtMost => order.is_le(),
            Self::Greater => order.is_gt(),
            Self::AtLeast => order.is_ge(),
        }
    }
}

include!("property_query/expressions.rs");
include!("property_query/storage.rs");
include!("property_query/query.rs");
