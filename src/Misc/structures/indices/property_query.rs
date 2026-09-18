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

use crate::misc::structures::buckets::{add_to_bucket, remove_from_bucket};
use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::hashing::{FastHashMap, FastHashSet};
use crate::misc::structures::mappings::grouped::grouped_multi_map::GroupedMultiMap;
use crate::misc::structures::mappings::grouped::grouped_single_map::GroupedSingleMap;
use crate::misc::structures::mappings::multi::unique_multi_map::UniqueMultiMap;
use crate::misc::structures::mappings::single::bi_map::BiMap;
use crate::misc::structures::traits::{Collection, Element, Grouping, SetAlgebra};
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
    fn for_each_dependency(&self, visit: &mut impl FnMut(Option<&P>)) {
        match self {
            Self::Has(property) | Self::Is(property, _) => visit(Some(property)),
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

impl<P: Element, V: Element> PartialEq for Expr<P, V> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Has(a), Self::Has(b)) => a == b,
            (Self::Is(a, x), Self::Is(b, y)) => a == b && x == y,
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
pub struct PropertyQuery<E, P, V> {
    storages: FastHashMap<P, Storage<E, V>>,
    element_properties: FastHashMap<E, Set<P>>,
    universe: Set<E>,

    cache: FastHashMap<ExprRef<P, V>, Arc<Set<E>>>,
    property_dependents: FastHashMap<P, FastHashSet<ExprRef<P, V>>>,
    universe_dependents: FastHashSet<ExprRef<P, V>>,
}

impl<E, P, V> Default for PropertyQuery<E, P, V> {
    fn default() -> Self {
        Self {
            storages: FastHashMap::default(),
            element_properties: FastHashMap::default(),
            universe: Set::default(),
            cache: FastHashMap::default(),
            property_dependents: FastHashMap::default(),
            universe_dependents: FastHashSet::default(),
        }
    }
}

impl<E: Element, P: Element, V: Element> PropertyQuery<E, P, V> {
    /// Creates an empty query with no registered properties or elements.
    pub fn new() -> Self {
        Self::default()
    }

    /// A query with these properties registered up front.
    pub fn with_properties(properties: impl IntoIterator<Item = (P, PropertyKind)>) -> Self {
        let mut query = Self::new();
        for (property, kind) in properties {
            query.register_property(property, kind);
        }
        query
    }

    // --- Schema ------------------------------------------------------------

    /// Returns false if the property already exists, whatever its kind.
    pub fn register_property(&mut self, property: P, kind: PropertyKind) -> bool {
        if self.storages.contains_key(&property) {
            return false;
        }
        self.invalidate(&property);
        self.storages.insert(property, Storage::new(kind));
        true
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

    /// Adds an element with a list of property values. Unregistered
    /// properties become `SINGLE`. Multi properties collect every value
    /// given; single properties keep the last one.
    pub fn add(&mut self, element: E, pairs: impl IntoIterator<Item = (P, V)>) {
        self.insert_element(element.clone());
        for (property, value) in pairs {
            self.ensure_property(&property, PropertyKind::SINGLE);
            self.add_value(element.clone(), property, value);
        }
    }

    /// Sets a property to exactly `value`. Unregistered properties become
    /// `SINGLE`. On a unique property, `value` is taken from its old owner.
    pub fn set(&mut self, element: E, property: P, value: V) {
        self.write(
            element,
            property,
            PropertyKind::SINGLE,
            |storage, element, value| storage.put_single(element, value),
            value,
        );
    }

    /// Adds `value` to a multi property, or sets a single one. Unregistered
    /// properties become `MULTI`.
    pub fn add_value(&mut self, element: E, property: P, value: V) {
        self.write(
            element,
            property,
            PropertyKind::MULTI,
            |storage, element, value| storage.put_value(element, value),
            value,
        );
    }

    /// Replaces all of a property's values. For single properties the last
    /// value wins. Unregistered properties become `MULTI`.
    pub fn set_values(&mut self, element: E, property: P, values: impl IntoIterator<Item = V>) {
        self.ensure_property(&property, PropertyKind::MULTI);
        self.insert_element(element.clone());
        self.clear_property(&element, &property);
        for value in values {
            self.add_value(element.clone(), property.clone(), value);
        }
    }

    /// Sets a flag. Unregistered properties become `FLAG`.
    pub fn set_flag(&mut self, element: E, property: P) {
        self.ensure_property(&property, PropertyKind::FLAG);
        self.insert_element(element.clone());
        self.storages
            .get_mut(&property)
            .unwrap()
            .set_flag(element.clone());
        self.sync(&element, &property);
    }

    fn write(
        &mut self,
        element: E,
        property: P,
        default_kind: PropertyKind,
        apply: impl FnOnce(&mut Storage<E, V>, E, V),
        value: V,
    ) {
        self.ensure_property(&property, default_kind);
        self.insert_element(element.clone());

        let storage = &self.storages[&property];
        if storage.kind().cardinality() == Cardinality::Flag {
            self.storages
                .get_mut(&property)
                .unwrap()
                .set_flag(element.clone());
            self.sync(&element, &property);
            return;
        }

        // A unique value moving between elements changes its old owner too.
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
            if !self.has_property(property) {
                self.register_property(property.clone(), storage.kind());
            }
        }
        for element in other.universe.iter() {
            self.insert_element(element.clone());
        }
        for (element, properties) in &other.element_properties {
            for property in properties {
                match other.storages[property].kind().cardinality() {
                    Cardinality::Flag => self.set_flag(element.clone(), property.clone()),
                    Cardinality::Single => {
                        if mode == FuseMode::Merge && self.has(element, property) {
                            continue;
                        }
                        if let Some(value) = other.get(element, property) {
                            self.set(element.clone(), property.clone(), value.clone());
                        }
                    }
                    Cardinality::Multi => {
                        let values = other.values(element, property).cloned();
                        match mode {
                            FuseMode::Overwrite => {
                                self.set_values(element.clone(), property.clone(), values)
                            }
                            FuseMode::Merge => {
                                for value in values {
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

    fn ensure_property(&mut self, property: &P, kind: PropertyKind) {
        if !self.storages.contains_key(property) {
            self.register_property(property.clone(), kind);
        }
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
    pub fn query(&mut self, expr: &ExprRef<P, V>) -> Arc<Set<E>> {
        if let Some(hit) = self.cache.get(expr) {
            return hit.clone();
        }

        let result = match &**expr {
            Expr::Has(property) => self.members_of(property).cloned().collect(),
            Expr::Is(property, value) => self.members_with(property, value).cloned().collect(),
            Expr::And(children) if children.is_empty() => self.universe.clone(),
            Expr::And(children) => {
                let sets: Vec<Arc<Set<E>>> =
                    children.iter().map(|child| self.query(child)).collect();
                let rest: Vec<&Set<E>> = sets[1..].iter().map(|set| &**set).collect();
                sets[0].intersection_all(&rest)
            }
            Expr::Or(children) => {
                let sets: Vec<Arc<Set<E>>> =
                    children.iter().map(|child| self.query(child)).collect();
                match sets.split_first() {
                    Some((first, rest)) => {
                        let rest: Vec<&Set<E>> = rest.iter().map(|set| &**set).collect();
                        first.union_all(&rest)
                    }
                    None => Set::new(),
                }
            }
            Expr::Not(child) => {
                let excluded = self.query(child);
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
    pub fn query_uncached(&self, expr: &ExprRef<P, V>) -> Set<E> {
        match &**expr {
            Expr::Has(property) => self.members_of(property).cloned().collect(),
            Expr::Is(property, value) => self.members_with(property, value).cloned().collect(),
            Expr::And(children) if children.is_empty() => self.universe.clone(),
            Expr::And(children) => {
                let sets: Vec<Set<E>> = children
                    .iter()
                    .map(|child| self.query_uncached(child))
                    .collect();
                match sets.split_first() {
                    Some((first, rest)) => {
                        let rest: Vec<&Set<E>> = rest.iter().collect();
                        first.intersection_all(&rest)
                    }
                    None => unreachable!(),
                }
            }
            Expr::Or(children) => {
                let sets: Vec<Set<E>> = children
                    .iter()
                    .map(|child| self.query_uncached(child))
                    .collect();
                match sets.split_first() {
                    Some((first, rest)) => {
                        let rest: Vec<&Set<E>> = rest.iter().collect();
                        first.union_all(&rest)
                    }
                    None => Set::new(),
                }
            }
            Expr::Not(child) => self.universe.difference(&self.query_uncached(child)),
        }
    }

    /// Elements holding every `(property, value)` pair. With no pairs,
    /// every element.
    pub fn query_all(&mut self, pairs: impl IntoIterator<Item = (P, V)>) -> Arc<Set<E>> {
        let expr = Expr::and(
            pairs
                .into_iter()
                .map(|(property, value)| Expr::is(property, value)),
        );
        self.query(&expr)
    }

    /// Elements holding any of the `(property, value)` pairs.
    pub fn query_any(&mut self, pairs: impl IntoIterator<Item = (P, V)>) -> Arc<Set<E>> {
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
impl<E: Element, P: Element, V: Element> Collection for PropertyQuery<E, P, V> {
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
impl<E: Element, P: Element, V: Element> PartialEq for PropertyQuery<E, P, V> {
    fn eq(&self, other: &Self) -> bool {
        self.universe == other.universe && self.storages == other.storages
    }
}

impl<E: Element, P: Element, V: Element> Eq for PropertyQuery<E, P, V> {}
