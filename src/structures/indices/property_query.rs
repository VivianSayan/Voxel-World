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

// ---------------------------------------------------------------------------
// Query expressions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
/// Composable property expression over property identifiers `P` and values `V`.
///
/// Expressions are normally constructed with [`Expr::has`], [`Expr::is`],
/// [`Expr::and`], [`Expr::or`], and [`Expr::negate`], which return shared
/// [`Arc`] nodes suitable for query caching.
pub enum Expr<P, V> {
    /// Elements that have the property at all.
    Has(P),
    /// Elements whose property holds the value.
    Is(P, V),
    /// Elements whose property holds a value standing in this relation to the
    /// given one.
    ///
    /// For a property holding many values, any one of them satisfying it is
    /// enough, as with [`Expr::Is`].
    Compare(P, Comparison, V),
    /// Elements whose property holds any of these values.
    ///
    /// The same answer as an `Or` of [`Expr::Is`] nodes, in one node and one
    /// pass.
    OneOf(P, Set<V>),
    /// Elements matching every child. With no children: every element.
    And(Set<Arc<Expr<P, V>>>),
    /// Elements matching any child. With no children: nothing.
    Or(Set<Arc<Expr<P, V>>>),
    /// Elements of the universe not matching the child.
    Not(Arc<Expr<P, V>>),
}

impl<P: Element, V: Element> Expr<P, V> {
    /// Creates an expression matching elements that hold `property`.
    pub fn has(property: P) -> Arc<Self> {
        Arc::new(Self::Has(property))
    }

    /// Creates an expression matching elements whose `property` contains `value`.
    pub fn is(property: P, value: V) -> Arc<Self> {
        Arc::new(Self::Is(property, value))
    }

    /// Creates an expression matching elements whose `property` holds any of
    /// `values`.
    pub fn one_of(property: P, values: impl IntoIterator<Item = V>) -> Arc<Self> {
        Arc::new(Self::OneOf(property, values.into_iter().collect()))
    }

    /// Creates a conjunction of `children`; no children means every element.
    pub fn and(children: impl IntoIterator<Item = Arc<Self>>) -> Arc<Self> {
        Arc::new(Self::And(children.into_iter().collect()))
    }

    /// Creates a disjunction of `children`; no children means no elements.
    pub fn or(children: impl IntoIterator<Item = Arc<Self>>) -> Arc<Self> {
        Arc::new(Self::Or(children.into_iter().collect()))
    }

    /// Creates the universe-relative negation of `child`.
    pub fn negate(child: Arc<Self>) -> Arc<Self> {
        Arc::new(Self::Not(child))
    }

    /// Calls `visit` with each property this expression reads, and with
    /// `None` if it reads the universe or the full element list.
    ///
    /// What a cache or a watch list keys itself on: anything that has to know
    /// when an expression's answer might have changed needs exactly this set.
    pub(crate) fn for_each_dependency(&self, visit: &mut impl FnMut(Option<&P>)) {
        match self {
            Self::Has(property)
            | Self::Is(property, _)
            | Self::Compare(property, _, _)
            | Self::OneOf(property, _) => visit(Some(property)),
            Self::And(children) if children.is_empty() => visit(None),
            Self::And(children) | Self::Or(children) => {
                for child in children {
                    child.for_each_dependency(visit);
                }
            }
            Self::Not(child) => {
                visit(None);
                child.for_each_dependency(visit);
            }
        }
    }
}

/// The comparing constructors, which exist only for a value type that has an
/// order at all.
///
/// A value type with no [`PartialOrd`] cannot have `Expr::less` written against
/// it, so the mistake is a compile error rather than a schema error. Within a
/// type that has one, *which properties* may be compared is still a schema
/// question, since the property is named by a run-time value: see
/// [`PropertyQuery::validate`].
///
/// ```compile_fail
/// use voxel_world::structures::indices::Expr;
///
/// // A value type with equality but no order at all.
/// #[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// struct Tag(u32);
///
/// // `Expr::is` is fine; `Expr::less` does not exist for this type.
/// let _ = Expr::less("depth", Tag(1));
/// ```
impl<P: Element, V: Element + PartialOrd> Expr<P, V> {
    /// Elements whose `property` holds a value below `value`.
    pub fn less(property: P, value: V) -> Arc<Self> {
        Self::compare(property, Comparison::Less, value)
    }

    /// Elements whose `property` holds a value at or below `value`.
    pub fn at_most(property: P, value: V) -> Arc<Self> {
        Self::compare(property, Comparison::AtMost, value)
    }

    /// Elements whose `property` holds a value above `value`.
    pub fn greater(property: P, value: V) -> Arc<Self> {
        Self::compare(property, Comparison::Greater, value)
    }

    /// Elements whose `property` holds a value at or above `value`.
    pub fn at_least(property: P, value: V) -> Arc<Self> {
        Self::compare(property, Comparison::AtLeast, value)
    }

    /// Elements whose `property` holds a value standing in `comparison` to
    /// `value`.
    pub fn compare(property: P, comparison: Comparison, value: V) -> Arc<Self> {
        Arc::new(Self::Compare(property, comparison, value))
    }

    /// Elements whose `property` holds a value from `low` to `high` inclusive.
    pub fn between(property: P, low: V, high: V) -> Arc<Self>
    where
        P: Clone,
    {
        Self::and([
            Self::at_least(property.clone(), low),
            Self::at_most(property, high),
        ])
    }
}

impl<P: Element, V: Element> PartialEq for Expr<P, V> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Has(a), Self::Has(b)) => a == b,
            (Self::Is(a, x), Self::Is(b, y)) => a == b && x == y,
            (Self::Compare(a, i, x), Self::Compare(b, j, y)) => a == b && i == j && x == y,
            (Self::OneOf(a, x), Self::OneOf(b, y)) => a == b && x == y,
            (Self::And(a), Self::And(b)) | (Self::Or(a), Self::Or(b)) => a == b,
            (Self::Not(a), Self::Not(b)) => a == b,
            _ => false,
        }
    }
}

impl<P: Element, V: Element> Eq for Expr<P, V> {}

impl<P: Element, V: Element> Hash for Expr<P, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Has(property) => property.hash(state),
            Self::Is(property, value) => {
                property.hash(state);
                value.hash(state);
            }
            Self::Compare(property, comparison, value) => {
                property.hash(state);
                comparison.hash(state);
                value.hash(state);
            }
            Self::OneOf(property, values) => {
                property.hash(state);
                values.hash(state);
            }
            Self::And(children) | Self::Or(children) => children.hash(state),
            Self::Not(child) => child.hash(state),
        }
    }
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Borrowed values or elements: none, one, or a whole set.
pub enum RefIter<'a, T> {
    /// Iterator with no values.
    Empty,
    /// Iterator over zero or one borrowed value.
    One(Option<&'a T>),
    /// Iterator over a borrowed set of values.
    Set(std::collections::hash_set::Iter<'a, T>),
}

impl<'a, T> RefIter<'a, T> {
    fn from_set(set: Option<&'a Set<T>>) -> Self {
        match set {
            Some(set) => Self::Set(set.into_iter()),
            None => Self::Empty,
        }
    }
}

impl<'a, T> Iterator for RefIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match self {
            Self::Empty => None,
            Self::One(item) => item.take(),
            Self::Set(items) => items.next(),
        }
    }
}

#[derive(Clone, Debug)]
enum Storage<E, V> {
    Flag(Set<E>),
    SingleShared(GroupedSingleMap<E, V>),
    SingleUnique(BiMap<E, V>),
    MultiShared(GroupedMultiMap<E, V>),
    MultiUnique(UniqueMultiMap<E, V>),
}

impl<E: Element, V: Element> Storage<E, V> {
    fn new(kind: PropertyKind) -> Self {
        match kind {
            PropertyKind::Flag => Self::Flag(Set::new()),
            PropertyKind::Single => Self::SingleShared(GroupedSingleMap::new()),
            PropertyKind::UniqueSingle => Self::SingleUnique(BiMap::new()),
            PropertyKind::Multi => Self::MultiShared(GroupedMultiMap::new()),
            PropertyKind::UniqueMulti => Self::MultiUnique(UniqueMultiMap::new()),
        }
    }

    fn kind(&self) -> PropertyKind {
        match self {
            Self::Flag(_) => PropertyKind::FLAG,
            Self::SingleShared(_) => PropertyKind::SINGLE,
            Self::SingleUnique(_) => PropertyKind::UNIQUE_SINGLE,
            Self::MultiShared(_) => PropertyKind::MULTI,
            Self::MultiUnique(_) => PropertyKind::UNIQUE_MULTI,
        }
    }

    fn member_count(&self) -> usize {
        match self {
            Self::Flag(set) => set.len(),
            Self::SingleShared(map) => map.len(),
            Self::SingleUnique(map) => map.len(),
            Self::MultiShared(map) => map.len(),
            Self::MultiUnique(map) => map.len(),
        }
    }

    fn contains_member(&self, element: &E) -> bool {
        match self {
            Self::Flag(set) => set.contains(element),
            Self::SingleShared(map) => map.contains(element),
            Self::SingleUnique(map) => map.contains_left(element),
            Self::MultiShared(map) => map.contains(element),
            Self::MultiUnique(map) => map.contains_key(element),
        }
    }

    fn has_value(&self, value: &V) -> bool {
        match self {
            Self::Flag(_) => false,
            Self::SingleShared(map) => map.contains_label(value),
            Self::SingleUnique(map) => map.contains_right(value),
            Self::MultiShared(map) => map.contains_label(value),
            Self::MultiUnique(map) => map.contains_value(value),
        }
    }

    fn has_pair(&self, element: &E, value: &V) -> bool {
        match self {
            Self::Flag(_) => false,
            Self::SingleShared(map) => map.contains_pair(element, value),
            Self::SingleUnique(map) => map.contains_pair(element, value),
            Self::MultiShared(map) => map.contains_pair(element, value),
            Self::MultiUnique(map) => map.contains_pair(element, value),
        }
    }

    fn get(&self, element: &E) -> Option<&V> {
        match self {
            Self::SingleShared(map) => map.get(element),
            Self::SingleUnique(map) => map.get_by_left(element),
            _ => None,
        }
    }

    fn values(&self, element: &E) -> RefIter<'_, V> {
        match self {
            Self::Flag(_) => RefIter::Empty,
            Self::SingleShared(map) => RefIter::One(map.get(element)),
            Self::SingleUnique(map) => RefIter::One(map.get_by_left(element)),
            Self::MultiShared(map) => RefIter::from_set(map.get(element)),
            Self::MultiUnique(map) => RefIter::from_set(map.get(element)),
        }
    }

    fn members_with(&self, value: &V) -> RefIter<'_, E> {
        match self {
            Self::Flag(_) => RefIter::Empty,
            Self::SingleShared(map) => RefIter::from_set(map.group(value)),
            Self::SingleUnique(map) => RefIter::One(map.get_by_right(value)),
            Self::MultiShared(map) => RefIter::from_set(map.group(value)),
            Self::MultiUnique(map) => RefIter::One(map.key_of(value)),
        }
    }

    fn value_member_count(&self, value: &V) -> usize {
        match self {
            Self::SingleShared(map) => map.group_len(value),
            Self::MultiShared(map) => map.group_len(value),
            _ => self.has_value(value) as usize,
        }
    }

    fn members(&self) -> Box<dyn Iterator<Item = &E> + '_> {
        match self {
            Self::Flag(set) => Box::new(set.iter()),
            Self::SingleShared(map) => Box::new(map.elements()),
            Self::SingleUnique(map) => Box::new(map.left_values()),
            Self::MultiShared(map) => Box::new(map.elements()),
            Self::MultiUnique(map) => Box::new(map.keys()),
        }
    }

    /// The element currently holding `value`, for unique properties.
    fn owner_of(&self, value: &V) -> Option<&E> {
        match self {
            Self::SingleUnique(map) => map.get_by_right(value),
            Self::MultiUnique(map) => map.key_of(value),
            _ => None,
        }
    }

    fn remove_member(&mut self, element: &E) -> bool {
        match self {
            Self::Flag(set) => set.remove(element),
            Self::SingleShared(map) => map.remove(element).is_some(),
            Self::SingleUnique(map) => map.remove_by_left(element).is_some(),
            Self::MultiShared(map) => map.remove(element).is_some(),
            Self::MultiUnique(map) => map.remove(element).is_some(),
        }
    }

    fn remove_value(&mut self, element: &E, value: &V) -> bool {
        match self {
            Self::Flag(_) => false,
            Self::SingleShared(map) => map.remove_pair(element, value),
            Self::SingleUnique(map) => map.remove_pair(element, value),
            Self::MultiShared(map) => map.remove_pair(element, value),
            Self::MultiUnique(map) => map.remove_pair(element, value),
        }
    }

    fn set_flag(&mut self, element: E) {
        if let Self::Flag(set) = self {
            set.insert(element);
        }
    }

    /// Makes `value` the element's only value, taking it from any other
    /// owner of a unique property.
    fn put_single(&mut self, element: E, value: V) {
        match self {
            Self::Flag(set) => {
                set.insert(element);
            }
            Self::SingleShared(map) => {
                map.insert(element, value);
            }
            Self::SingleUnique(map) => {
                map.insert(element, value);
            }
            Self::MultiShared(map) => map.replace_labels(&element, &Set::from([value])),
            Self::MultiUnique(map) => {
                map.remove(&element);
                map.insert_or_move(element, value);
            }
        }
    }

    /// Adds `value` to a multi property; replaces the value of a single one.
    fn put_value(&mut self, element: E, value: V) {
        match self {
            Self::MultiShared(map) => {
                map.insert(element, value);
            }
            Self::MultiUnique(map) => {
                map.insert_or_move(element, value);
            }
            _ => self.put_single(element, value),
        }
    }
}

impl<E: Element, V: Element> PartialEq for Storage<E, V> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Flag(a), Self::Flag(b)) => a == b,
            (Self::SingleShared(a), Self::SingleShared(b)) => a == b,
            (Self::SingleUnique(a), Self::SingleUnique(b)) => a == b,
            (Self::MultiShared(a), Self::MultiShared(b)) => a == b,
            (Self::MultiUnique(a), Self::MultiUnique(b)) => a == b,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// PropertyQuery
// ---------------------------------------------------------------------------

type ExprRef<P, V> = Arc<Expr<P, V>>;

#[derive(Clone, Debug)]
/// Property store and cached expression engine.
///
/// `E` identifies elements, `P` identifies properties, and `V` is the common
/// value type used by valued properties. All three types are cloned into
/// secondary indices and therefore implement [`Element`].
pub struct PropertyQuery<E, P, V: Kinded> {
    storages: FastHashMap<P, Storage<E, V>>,
    element_properties: FastHashMap<E, Set<P>>,
    universe: Set<E>,

    /// The value kind each valued property accepts. Flags have none.
    value_kinds: FastHashMap<P, V::Kind>,

    /// Whether each valued property's values are ordered or merely labels.
    comparabilities: FastHashMap<P, Comparability>,

    cache: FastHashMap<ExprRef<P, V>, Arc<Set<E>>>,
    property_dependents: FastHashMap<P, FastHashSet<ExprRef<P, V>>>,
    universe_dependents: FastHashSet<ExprRef<P, V>>,
}

impl<E, P, V: Kinded> Default for PropertyQuery<E, P, V> {
    fn default() -> Self {
        Self {
            storages: FastHashMap::default(),
            element_properties: FastHashMap::default(),
            universe: Set::default(),
            value_kinds: FastHashMap::default(),
            comparabilities: FastHashMap::default(),
            cache: FastHashMap::default(),
            property_dependents: FastHashMap::default(),
            universe_dependents: FastHashSet::default(),
        }
    }
}

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> PropertyQuery<E, P, V> {
    /// Creates an empty query with no registered properties or elements.
    pub fn new() -> Self {
        Self::default()
    }

    /// A query with these properties registered up front, each with the value
    /// kind it accepts.
    ///
    /// Use [`PropertyKind::Flag`] with any value kind for a flag: it holds no
    /// values, so what is declared for it is never consulted.
    pub fn with_properties(
        properties: impl IntoIterator<Item = (P, PropertyKind, V::Kind, Comparability)>,
    ) -> Result<Self, SchemaError<P, V::Kind>> {
        let mut query = Self::new();

        for (property, kind, value_kind, comparability) in properties {
            match kind.cardinality() {
                Cardinality::Flag => query.register_flag(property)?,
                _ => query.register_property(property, kind, value_kind, comparability)?,
            }
        }

        Ok(query)
    }

    // --- Schema ------------------------------------------------------------

    /// Returns false if the property already exists, whatever its kind.
    /// Registering the same property again is accepted only when **both** the
    /// storage kind and the value kind match what it already has. Changing
    /// either would leave values already stored under the property answering to
    /// a contract nothing checked them against, so it is refused rather than
    /// applied: use [`PropertyQuery::unregister_property`] and register again to
    /// migrate deliberately.
    pub fn register_property(
        &mut self,
        property: P,
        kind: PropertyKind,
        value_kind: V::Kind,
        comparability: Comparability,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        if kind.cardinality() == Cardinality::Flag {
            return Err(SchemaError::FlagTakesNoValue { property });
        }

        self.declare(property.clone(), kind)?;

        // All three halves of the contract have to agree for a re-registration
        // to be a repeat rather than a redefinition.
        if let Some(expected) = self.value_kinds.get(&property)
            && *expected != value_kind
        {
            return Err(SchemaError::ValueKindConflict {
                expected: expected.clone(),
                property,
                incoming: value_kind,
            });
        }

        if let Some(current) = self.comparabilities.get(&property).copied() {
            if current != comparability {
                return Err(SchemaError::ComparabilityConflict {
                    current,
                    property,
                    incoming: comparability,
                });
            }

            return Ok(());
        }

        self.value_kinds.insert(property.clone(), value_kind);
        self.comparabilities.insert(property, comparability);

        Ok(())
    }

    /// Declares a property whose values run from smaller to larger, so that
    /// comparisons against it are allowed.
    ///
    /// [`PropertyQuery::register_property`] with
    /// [`Comparability::Ordered`], for the common case where naming the enum adds
    /// nothing.
    pub fn register_ordered_property(
        &mut self,
        property: P,
        kind: PropertyKind,
        value_kind: V::Kind,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.register_property(property, kind, value_kind, Comparability::Ordered)
    }

    /// Declares a property whose values are labels with no order, so that a
    /// comparison against it is refused.
    ///
    /// [`PropertyQuery::register_property`] with
    /// [`Comparability::Categorical`].
    pub fn register_categorical_property(
        &mut self,
        property: P,
        kind: PropertyKind,
        value_kind: V::Kind,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.register_property(property, kind, value_kind, Comparability::Categorical)
    }

    /// Whether a property's values are ordered, or `None` for a flag or an
    /// unregistered property.
    pub fn comparability(&self, property: &P) -> Option<Comparability> {
        self.comparabilities.get(property).copied()
    }

    /// Registers a property that carries no value.
    ///
    /// Separate from [`PropertyQuery::register_property`] because a flag has no
    /// value kind to declare, rather than having one that is ignored.
    pub fn register_flag(&mut self, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        self.declare(property, PropertyKind::Flag)
    }

    /// Adds the storage for a property, or reports why it cannot.
    ///
    /// Registering the same property with the same kind again is accepted and
    /// does nothing, so that setting a schema up twice is not an error; only a
    /// disagreement is. The caller checks the value kind separately, since both
    /// halves of the contract have to match for a re-registration to be a
    /// repeat rather than a redefinition.
    fn declare(&mut self, property: P, kind: PropertyKind) -> Result<(), SchemaError<P, V::Kind>> {
        if let Some(storage) = self.storages.get(&property) {
            if storage.kind() == kind {
                return Ok(());
            }

            return Err(SchemaError::Conflict {
                property,
                current: storage.kind(),
                incoming: kind,
            });
        }

        self.invalidate(&property);
        self.storages.insert(property, Storage::new(kind));

        Ok(())
    }

    /// The value kind a property accepts, or `None` for a flag or an
    /// unregistered property.
    pub fn value_kind(&self, property: &P) -> Option<&V::Kind> {
        self.value_kinds.get(property)
    }

    /// Checks a value against what a property accepts.
    ///
    /// The whole schema rule in one place: the property must be registered, it
    /// must hold values at all, and the value must be of the kind it was
    /// registered with.
    pub fn check(&self, property: &P, value: &V) -> Result<(), SchemaError<P, V::Kind>> {
        let Some(storage) = self.storages.get(property) else {
            return Err(SchemaError::Unregistered {
                property: property.clone(),
            });
        };

        if storage.kind().cardinality() == Cardinality::Flag {
            return Err(SchemaError::FlagTakesNoValue {
                property: property.clone(),
            });
        }

        let found: V::Kind = value.kind();

        match self.value_kinds.get(property) {
            Some(expected) if *expected != found => Err(SchemaError::WrongValueKind {
                property: property.clone(),
                expected: expected.clone(),
                found,
            }),
            _ => Ok(()),
        }
    }

    /// The elements whose `property` holds a value standing in `comparison` to
    /// `value`.
    ///
    /// The one query that cannot use the inverted index: it answers "which
    /// elements hold exactly this value", and there is no index for "above
    /// five". So this walks the property's members, which is linear in how many
    /// hold the property rather than constant. Single-element evaluation —
    /// [`ConditionIndex`](super::ConditionIndex) and [`Gate`](super::Gate) — is
    /// unaffected, since it looks up the one element's values and compares.
    fn compared(&self, property: &P, comparison: Comparison, value: &V) -> Set<E> {
        let members: Vec<E> = self.members_of(property).cloned().collect();

        members
            .into_iter()
            .filter(|element| {
                self.values(element, property)
                    .any(|held| comparison.holds(held, value))
            })
            .collect()
    }

    /// Checks a value the way [`PropertyQuery::check`] does, and that its kind
    /// has an order to compare along.
    ///
    /// A kind is ordered exactly when a value of it compares with itself, which
    /// is what a hand-written [`PartialOrd`] on a tagged value type reports by
    /// returning `None` for the kinds that have no order.
    pub fn check_ordered(&self, property: &P, value: &V) -> Result<(), SchemaError<P, V::Kind>> {
        self.check(property, value)?;

        if value.partial_cmp(value).is_none() {
            return Err(SchemaError::NotOrdered {
                property: property.clone(),
                kind: value.kind(),
            });
        }

        Ok(())
    }

    /// Checks every value an expression compares against the schema.
    ///
    /// Where a mistyped condition is caught: an expression that asks for a
    /// health of "banana" can never match anything, and saying so when it is
    /// built beats waiting for it never to fire.
    pub fn validate(&self, expression: &Expr<P, V>) -> Result<(), SchemaError<P, V::Kind>> {
        match expression {
            Expr::Has(property) => {
                if self.storages.contains_key(property) {
                    Ok(())
                } else {
                    Err(SchemaError::Unregistered {
                        property: property.clone(),
                    })
                }
            }
            Expr::Is(property, value) => self.check(property, value),
            Expr::Compare(property, _, value) => self.check_ordered(property, value),
            Expr::OneOf(property, values) => values
                .iter()
                .try_for_each(|value| self.check(property, value)),
            Expr::And(children) | Expr::Or(children) => {
                children.iter().try_for_each(|child| self.validate(child))
            }
            Expr::Not(child) => self.validate(child),
        }
    }

    /// Removes a property from every element.
    pub fn unregister_property(&mut self, property: &P) -> bool {
        let Some(storage) = self.storages.remove(property) else {
            return false;
        };
        let members: Vec<E> = storage.members().cloned().collect();
        for element in &members {
            self.unlink(element, property);
        }
        self.value_kinds.remove(property);
        self.comparabilities.remove(property);
        self.invalidate(property);
        true
    }

    /// Returns whether `property` is registered in the schema.
    pub fn has_property(&self, property: &P) -> bool {
        self.storages.contains_key(property)
    }

    /// Returns the registered storage kind of `property`.
    pub fn property_kind(&self, property: &P) -> Option<PropertyKind> {
        self.storages.get(property).map(Storage::kind)
    }

    /// Iterates over registered property identifiers.
    pub fn properties(&self) -> impl Iterator<Item = &P> {
        self.storages.keys()
    }

    // --- Reading -------------------------------------------------------------

    /// Number of elements holding at least one property.
    pub fn len(&self) -> usize {
        self.element_properties.len()
    }

    /// Returns whether no element currently holds a property.
    ///
    /// Elements present only in [`Self::universe`] do not affect this result.
    pub fn is_empty(&self) -> bool {
        self.element_properties.is_empty()
    }

    /// True when the element holds at least one property.
    pub fn contains(&self, element: &E) -> bool {
        self.element_properties.contains_key(element)
    }

    /// Every element ever added and not removed, with or without properties.
    pub fn universe(&self) -> &Set<E> {
        &self.universe
    }

    /// Elements holding at least one property.
    pub fn elements(&self) -> impl Iterator<Item = &E> {
        self.element_properties.keys()
    }

    /// Returns the properties currently held by `element`.
    pub fn properties_of(&self, element: &E) -> Option<&Set<P>> {
        self.element_properties.get(element)
    }

    /// The value of a single-valued property.
    pub fn get(&self, element: &E, property: &P) -> Option<&V> {
        self.storages.get(property)?.get(element)
    }

    /// Every value of a property: none, one, or many.
    pub fn values(&self, element: &E, property: &P) -> RefIter<'_, V> {
        match self.storages.get(property) {
            Some(storage) => storage.values(element),
            None => RefIter::Empty,
        }
    }

    /// True when the element has the property, flag or valued.
    pub fn has(&self, element: &E, property: &P) -> bool {
        self.storages
            .get(property)
            .is_some_and(|storage| storage.contains_member(element))
    }

    /// Returns whether `element` holds the exact `(property, value)` pair.
    pub fn has_value(&self, element: &E, property: &P, value: &V) -> bool {
        self.storages
            .get(property)
            .is_some_and(|storage| storage.has_pair(element, value))
    }

    /// True when any element holds `value` for `property`.
    pub fn is_value_used(&self, property: &P, value: &V) -> bool {
        self.storages
            .get(property)
            .is_some_and(|storage| storage.has_value(value))
    }

    /// Iterates over all elements that hold `property`.
    pub fn members_of(&self, property: &P) -> Box<dyn Iterator<Item = &E> + '_> {
        match self.storages.get(property) {
            Some(storage) => storage.members(),
            None => Box::new(std::iter::empty()),
        }
    }

    /// Iterates over elements whose `property` contains `value`.
    pub fn members_with(&self, property: &P, value: &V) -> RefIter<'_, E> {
        match self.storages.get(property) {
            Some(storage) => storage.members_with(value),
            None => RefIter::Empty,
        }
    }

    /// Number of elements holding the property.
    pub fn property_len(&self, property: &P) -> usize {
        self.storages.get(property).map_or(0, Storage::member_count)
    }

    /// Number of elements holding `value` for the property.
    pub fn value_len(&self, property: &P, value: &V) -> usize {
        self.storages
            .get(property)
            .map_or(0, |storage| storage.value_member_count(value))
    }

    /// True when the element holds every one of the `(property, value)` pairs.
    pub fn matches<'a>(&self, element: &E, pairs: impl IntoIterator<Item = (&'a P, &'a V)>) -> bool
    where
        P: 'a,
        V: 'a,
    {
        pairs
            .into_iter()
            .all(|(property, value)| self.has_value(element, property, value))
    }

    // --- Writing -------------------------------------------------------------

    /// Adds an element to the universe without giving it properties.
    pub fn insert_element(&mut self, element: E) -> bool {
        if self.universe.contains(&element) {
            return false;
        }
        self.universe.insert(element);
        self.invalidate_universe();
        true
    }

    /// Adds an element with a list of property values.
    ///
    /// Multi properties collect every value given; single properties keep the
    /// last one. Stops at the first value the schema rejects, leaving the ones
    /// before it written.
    pub fn add(
        &mut self,
        element: E,
        pairs: impl IntoIterator<Item = (P, V)>,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.insert_element(element.clone());

        for (property, value) in pairs {
            self.add_value(element.clone(), property, value)?;
        }

        Ok(())
    }

    /// Sets a property to exactly `value`.
    ///
    /// On a unique property, `value` is taken from its old owner. Rejected
    /// unless the property is registered and accepts values of this kind.
    pub fn set(
        &mut self,
        element: E,
        property: P,
        value: V,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.write(
            element,
            property,
            |storage, element, value| storage.put_single(element, value),
            value,
        )
    }

    /// Adds `value` to a multi property, or sets a single one.
    ///
    /// Rejected unless the property is registered and accepts values of this
    /// kind.
    pub fn add_value(
        &mut self,
        element: E,
        property: P,
        value: V,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.write(
            element,
            property,
            |storage, element, value| storage.put_value(element, value),
            value,
        )
    }

    /// Replaces all of a property's values. For single properties the last
    /// value wins. Unregistered properties become `MULTI`.
    pub fn set_values(
        &mut self,
        element: E,
        property: P,
        values: impl IntoIterator<Item = V>,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        let values: Vec<V> = values.into_iter().collect();

        for value in &values {
            self.check(&property, value)?;
        }

        self.insert_element(element.clone());
        self.clear_property(&element, &property);

        for value in values {
            self.add_value(element.clone(), property.clone(), value)?;
        }

        Ok(())
    }

    /// Sets a flag, which the property must be registered as.
    pub fn set_flag(&mut self, element: E, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        match self.storages.get(&property).map(Storage::kind) {
            Some(kind) if kind.cardinality() == Cardinality::Flag => {}
            Some(current) => {
                return Err(SchemaError::Conflict {
                    property,
                    current,
                    incoming: PropertyKind::Flag,
                });
            }
            None => return Err(SchemaError::Unregistered { property }),
        }

        self.insert_element(element.clone());
        self.storages
            .get_mut(&property)
            .unwrap()
            .set_flag(element.clone());
        self.sync(&element, &property);

        Ok(())
    }

    /// The shared path behind every write: makes sure the property exists and
    /// the element is known, applies `apply` to the storage, and brings the
    /// indices back in step.
    ///
    /// A property declared as a flag ignores the value and is simply set, since
    /// there is nowhere to put one. For a unique property, writing a value that
    /// another element already holds takes it away from that element, so its
    /// old owner is looked up first and re-synced afterwards along with the new
    /// one.
    fn write(
        &mut self,
        element: E,
        property: P,
        apply: impl FnOnce(&mut Storage<E, V>, E, V),
        value: V,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.check(&property, &value)?;
        self.insert_element(element.clone());

        let storage = &self.storages[&property];

        let previous_owner = storage
            .owner_of(&value)
            .filter(|owner| **owner != element)
            .cloned();

        apply(
            self.storages.get_mut(&property).unwrap(),
            element.clone(),
            value,
        );

        self.sync(&element, &property);
        if let Some(owner) = previous_owner {
            self.sync(&owner, &property);
        }

        Ok(())
    }

    /// Removes one exact value from an element's property.
    ///
    /// Returns `false` for flags, missing properties, or missing pairs.
    pub fn remove_value(&mut self, element: &E, property: &P, value: &V) -> bool {
        let Some(storage) = self.storages.get(property) else {
            return false;
        };
        if !storage.has_pair(element, value) {
            return false;
        }
        self.storages
            .get_mut(property)
            .unwrap()
            .remove_value(element, value);
        self.sync(element, property);
        true
    }

    /// Removes a property from one element.
    pub fn clear_property(&mut self, element: &E, property: &P) -> bool {
        if !self.has(element, property) {
            return false;
        }
        self.storages
            .get_mut(property)
            .unwrap()
            .remove_member(element);
        self.sync(element, property);
        true
    }

    /// Removes an element and all its properties.
    pub fn remove_element(&mut self, element: &E) -> bool {
        let properties = self.element_properties.remove(element);
        for property in properties.iter().flatten() {
            self.storages
                .get_mut(property)
                .unwrap()
                .remove_member(element);
            self.invalidate(property);
        }
        let in_universe = self.universe.remove(element);
        self.invalidate_universe();
        properties.is_some() || in_universe
    }

    /// Removes every element yielded by `elements`, including all properties.
    pub fn remove_elements<'a>(&mut self, elements: impl IntoIterator<Item = &'a E>)
    where
        E: 'a,
    {
        for element in elements {
            self.remove_element(element);
        }
    }

    /// Removes every element and property.
    pub fn clear(&mut self) {
        self.storages.clear();
        self.element_properties.clear();
        self.universe.clear();
        self.clear_cache();
    }

    /// Copies every element, property and value of `other` into this query.
    /// Returns a schema error before mutating either query when a property is
    /// registered with incompatible kinds.
    pub fn fuse(&mut self, other: &Self, mode: FuseMode) -> Result<(), FuseError<P>> {
        for (property, incoming) in &other.storages {
            if let Some(current) = self.storages.get(property)
                && current.kind() != incoming.kind()
            {
                return Err(FuseError {
                    property: property.clone(),
                    current: current.kind(),
                    incoming: incoming.kind(),
                });
            }
        }
        for (property, storage) in &other.storages {
            if self.has_property(property) {
                continue;
            }

            self.storages
                .insert(property.clone(), Storage::new(storage.kind()));
            self.invalidate(property);

            if let Some(comparability) = other.comparabilities.get(property) {
                self.comparabilities
                    .insert(property.clone(), *comparability);
            }

            if let Some(value_kind) = other.value_kinds.get(property) {
                self.value_kinds
                    .insert(property.clone(), value_kind.clone());
            }
        }
        for element in other.universe.iter() {
            self.insert_element(element.clone());
        }
        for (element, properties) in &other.element_properties {
            for property in properties {
                match other.storages[property].kind().cardinality() {
                    Cardinality::Flag => {
                        let _ = self.set_flag(element.clone(), property.clone());
                    }
                    Cardinality::Single => {
                        if mode == FuseMode::Merge && self.has(element, property) {
                            continue;
                        }
                        if let Some(value) = other.get(element, property) {
                            let _ = self.set(element.clone(), property.clone(), value.clone());
                        }
                    }
                    Cardinality::Multi => {
                        let values = other.values(element, property).cloned();
                        match mode {
                            FuseMode::Overwrite => {
                                let _ = self.set_values(element.clone(), property.clone(), values);
                            }
                            FuseMode::Merge => {
                                for value in values {
                                    let _ =
                                        self.add_value(element.clone(), property.clone(), value);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Brings the element -> properties index in line with the storage.
    fn sync(&mut self, element: &E, property: &P) {
        let holds = self
            .storages
            .get(property)
            .is_some_and(|storage| storage.contains_member(element));
        let was_element = self.element_properties.contains_key(element);
        if holds {
            add_to_bucket(&mut self.element_properties, element, property.clone());
        } else {
            remove_from_bucket(&mut self.element_properties, element, property);
        }
        self.invalidate(property);
        if was_element != self.element_properties.contains_key(element) {
            self.invalidate_universe();
        }
    }

    fn unlink(&mut self, element: &E, property: &P) {
        let was_element = self.element_properties.contains_key(element);
        remove_from_bucket(&mut self.element_properties, element, property);
        if was_element != self.element_properties.contains_key(element) {
            self.invalidate_universe();
        }
    }

    // --- Queries ------------------------------------------------------------

    /// Elements matching `expr`. Results are cached until a property they
    /// depend on changes.
    pub fn query(&mut self, expr: &ExprRef<P, V>) -> Result<Arc<Set<E>>, SchemaError<P, V::Kind>> {
        self.validate(expr)?;

        Ok(self.query_unchecked(expr))
    }

    /// Evaluates a validated expression, caching it and every subtree.
    ///
    /// Private because it trusts the expression. The public entry points check
    /// once at the boundary and then recurse through here, so a tree of `n`
    /// nodes is walked for validation once rather than once per level.
    fn query_unchecked(&mut self, expr: &ExprRef<P, V>) -> Arc<Set<E>> {
        if let Some(hit) = self.cache.get(expr) {
            return hit.clone();
        }

        let result = match &**expr {
            Expr::Has(property) => self.members_of(property).cloned().collect(),
            Expr::Is(property, value) => self.members_with(property, value).cloned().collect(),
            Expr::Compare(property, comparison, value) => {
                self.compared(property, *comparison, value)
            }
            Expr::OneOf(property, values) => values
                .iter()
                .flat_map(|value| self.members_with(property, value).cloned())
                .collect(),
            Expr::And(children) if children.is_empty() => self.universe.clone(),
            Expr::And(children) => {
                let sets: Vec<Arc<Set<E>>> = children
                    .iter()
                    .map(|child| self.query_unchecked(child))
                    .collect();
                let rest: Vec<&Set<E>> = sets[1..].iter().map(|set| &**set).collect();
                sets[0].intersection_all(&rest)
            }
            Expr::Or(children) => {
                let sets: Vec<Arc<Set<E>>> = children
                    .iter()
                    .map(|child| self.query_unchecked(child))
                    .collect();
                match sets.split_first() {
                    Some((first, rest)) => {
                        let rest: Vec<&Set<E>> = rest.iter().map(|set| &**set).collect();
                        first.union_all(&rest)
                    }
                    None => Set::new(),
                }
            }
            Expr::Not(child) => {
                let excluded = self.query_unchecked(child);
                self.universe.difference(&excluded)
            }
        };

        let result = Arc::new(result);
        let (properties, uses_universe) = Self::dependencies_of(expr);
        for property in properties {
            self.property_dependents
                .entry(property)
                .or_default()
                .insert(expr.clone());
        }
        if uses_universe {
            self.universe_dependents.insert(expr.clone());
        }
        self.cache.insert(expr.clone(), result.clone());
        result
    }

    /// Evaluates without populating or consulting the cache.
    pub fn query_uncached(&self, expr: &ExprRef<P, V>) -> Result<Set<E>, SchemaError<P, V::Kind>> {
        self.validate(expr)?;

        Ok(self.evaluate(expr))
    }

    /// Evaluates a validated expression without touching the cache.
    fn evaluate(&self, expr: &ExprRef<P, V>) -> Set<E> {
        match &**expr {
            Expr::Has(property) => self.members_of(property).cloned().collect(),
            Expr::Is(property, value) => self.members_with(property, value).cloned().collect(),
            Expr::Compare(property, comparison, value) => {
                self.compared(property, *comparison, value)
            }
            Expr::OneOf(property, values) => values
                .iter()
                .flat_map(|value| self.members_with(property, value).cloned())
                .collect(),
            Expr::And(children) if children.is_empty() => self.universe.clone(),
            Expr::And(children) => {
                let sets: Vec<Set<E>> = children.iter().map(|child| self.evaluate(child)).collect();
                match sets.split_first() {
                    Some((first, rest)) => {
                        let rest: Vec<&Set<E>> = rest.iter().collect();
                        first.intersection_all(&rest)
                    }
                    None => unreachable!(),
                }
            }
            Expr::Or(children) => {
                let sets: Vec<Set<E>> = children.iter().map(|child| self.evaluate(child)).collect();
                match sets.split_first() {
                    Some((first, rest)) => {
                        let rest: Vec<&Set<E>> = rest.iter().collect();
                        first.union_all(&rest)
                    }
                    None => Set::new(),
                }
            }
            Expr::Not(child) => self.universe.difference(&self.evaluate(child)),
        }
    }

    /// Elements holding every `(property, value)` pair. With no pairs,
    /// every element.
    pub fn query_all(
        &mut self,
        pairs: impl IntoIterator<Item = (P, V)>,
    ) -> Result<Arc<Set<E>>, SchemaError<P, V::Kind>> {
        let expr = Expr::and(
            pairs
                .into_iter()
                .map(|(property, value)| Expr::is(property, value)),
        );
        self.query(&expr)
    }

    /// Elements holding any of the `(property, value)` pairs.
    pub fn query_any(
        &mut self,
        pairs: impl IntoIterator<Item = (P, V)>,
    ) -> Result<Arc<Set<E>>, SchemaError<P, V::Kind>> {
        let expr = Expr::or(
            pairs
                .into_iter()
                .map(|(property, value)| Expr::is(property, value)),
        );
        self.query(&expr)
    }

    /// Discards every cached expression result and dependency record.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.property_dependents.clear();
        self.universe_dependents.clear();
    }

    /// Number of expression results currently cached.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    fn invalidate(&mut self, property: &P) {
        if self.cache.is_empty() {
            return;
        }
        if let Some(dependents) = self.property_dependents.remove(property) {
            for expr in dependents {
                self.evict_cached(&expr);
            }
        }
    }

    fn invalidate_universe(&mut self) {
        if self.cache.is_empty() {
            return;
        }
        for expr in std::mem::take(&mut self.universe_dependents) {
            self.evict_cached(&expr);
        }
    }

    fn dependencies_of(expr: &ExprRef<P, V>) -> (FastHashSet<P>, bool) {
        let mut properties = FastHashSet::default();
        let mut uses_universe = false;
        expr.for_each_dependency(&mut |dependency| match dependency {
            Some(property) => {
                properties.insert(property.clone());
            }
            None => uses_universe = true,
        });
        (properties, uses_universe)
    }

    fn evict_cached(&mut self, expr: &ExprRef<P, V>) {
        self.cache.remove(expr);
        let (properties, uses_universe) = Self::dependencies_of(expr);
        for property in properties {
            let remove_bucket =
                self.property_dependents
                    .get_mut(&property)
                    .is_some_and(|dependents| {
                        dependents.remove(expr);
                        dependents.is_empty()
                    });
            if remove_bucket {
                self.property_dependents.remove(&property);
            }
        }
        if uses_universe {
            self.universe_dependents.remove(expr);
        }
    }

    // --- Comparison ----------------------------------------------------------

    /// True when every property value here also holds in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.storages.iter().all(|(property, storage)| {
            let Some(theirs) = other.storages.get(property) else {
                return false;
            };
            storage.kind() == theirs.kind()
                && storage
                    .members()
                    .all(|element| match storage.kind().cardinality() {
                        Cardinality::Flag => theirs.contains_member(element),
                        _ => storage
                            .values(element)
                            .all(|value| theirs.has_pair(element, value)),
                    })
        })
    }
}

/// The elements holding at least one property.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Collection
    for PropertyQuery<E, P, V>
{
    type Item = E;

    fn len(&self) -> usize {
        self.element_properties.len()
    }

    fn contains(&self, element: &E) -> bool {
        self.element_properties.contains_key(element)
    }

    fn elements(&self) -> impl Iterator<Item = &E> {
        self.element_properties.keys()
    }
}

/// Equal when both hold the same elements with the same property values.
/// Caches are ignored.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> PartialEq
    for PropertyQuery<E, P, V>
{
    fn eq(&self, other: &Self) -> bool {
        self.universe == other.universe && self.storages == other.storages
    }
}

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Eq for PropertyQuery<E, P, V> {}
