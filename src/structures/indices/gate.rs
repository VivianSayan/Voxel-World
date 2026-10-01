//! Conditions asked at the moment work comes up, rather than watched.

use super::property_query::{Comparability, Expr, PropertyKind, PropertyQuery, SchemaError};
use super::property_store::PropertyStore;
use crate::structures::hashing::FastHashMap;
use crate::structures::traits::{
    Collection, CollectionRemove, DeterministicOrder, Element, Kinded, Map, UniqueCollection,
};
use std::fmt;
use std::sync::Arc;

/// A condition per element, answered at the moment it is asked.
///
/// # What it is for
///
/// Deciding, as a scheduler or a rota hands work out, whether each piece of it
/// should happen at all. A plot is due to grow, but only if the ground is
/// thawed; a creature's turn comes round, but only if it is awake.
///
/// ```text
/// tick arrives
///   -> the structure works out what is due
///     -> the gate is asked about each one
///       -> only what the gate allows is handed back
/// ```
///
/// # Level, not edge
///
/// A gate answers *is this true now*. It has no memory, no ready list and no
/// notion of a condition having just become true — ask it twice with nothing
/// changed and it answers the same both times. That is the whole difference
/// from [`ConditionIndex`](super::ConditionIndex), which reports the moment a
/// condition *becomes* true and then says nothing until it has gone false and
/// come back.
///
/// Use a gate when work already has its own schedule and the condition decides
/// whether each occurrence counts. Use a condition index when the condition
/// itself is what should set work going.
///
/// # An element with no condition is allowed
///
/// [`Gate::allows`] is true for anything the gate has never been told about, so
/// a gate holds only the conditions that actually restrict something and the
/// rest pass freely. Register a condition for an element to restrict it, and
/// [`Gate::clear_condition`] to let it through again.
///
/// # What it costs
///
/// One hash look-up per question, plus evaluating the expression: a look-up per
/// leaf against the element's properties or the world's facts. Nothing is
/// cached and nothing is recomputed on a write, which is what makes the state
/// safe to change freely between ticks — including through
/// [`Gate::store_mut`], which a condition index cannot offer because its edge
/// tracking would go stale.
///
/// # Example
///
/// ```
/// use voxel_world::structures::indices::{Expr, Gate, PropertyKind};
/// use voxel_world::structures::collections::Scheduler;
///
/// let mut gate: Gate<u32, &str, &str> = Gate::new();
/// gate.register_categorical_property("ground", PropertyKind::Single, ())?;
/// gate.require(7, Expr::is("ground", "thawed"))?;
///
/// let mut work: Scheduler<u32> = Scheduler::new();
/// work.schedule(1, 7);
/// work.schedule(1, 8);
///
/// // Seven is frozen when its turn comes, so it is dropped; eight has no
/// // condition, so it passes.
/// gate.set(7, "ground", "frozen")?;
/// assert_eq!(work.advance_with(&gate), vec![8]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct Gate<E, P, V: Kinded> {
    store: PropertyStore<E, P, V>,
    conditions: FastHashMap<E, Arc<Expr<P, V>>>,
}

impl<E, P, V: Kinded> Default for Gate<E, P, V> {
    fn default() -> Self {
        Self {
            store: PropertyStore::default(),
            conditions: FastHashMap::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Asking it
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Gate<E, P, V> {
    /// An empty gate: no properties, no facts, and nothing restricted.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether an element may act as things stand.
    ///
    /// True when no condition is registered for it, and otherwise whether its
    /// condition holds right now. The condition was checked against the schema
    /// when it was registered, so this only evaluates.
    pub fn allows(&self, element: &E) -> bool {
        match self.conditions.get(element) {
            Some(condition) => self.store.satisfies_unchecked(element, condition),
            None => true,
        }
    }

    /// Whether an element is held back as things stand, which is the opposite
    /// of [`Gate::allows`].
    pub fn refuses(&self, element: &E) -> bool {
        !self.allows(element)
    }

    /// Restricts an element to a condition, replacing whatever it had.
    ///
    /// The expression is checked against the schema, so one that could never
    /// hold is refused here rather than silently barring the element for ever.
    /// Returns the condition it replaced.
    // `Result<Option<_>>` is the ordinary shape: the replaced condition, or why
    // the new one was refused. A type alias would hide that rather than help.
    #[allow(clippy::type_complexity)]
    pub fn require(
        &mut self,
        element: E,
        condition: Arc<Expr<P, V>>,
    ) -> Result<Option<Arc<Expr<P, V>>>, SchemaError<P, V::Kind>> {
        self.store.validate(&condition)?;

        Ok(self.conditions.insert(element, condition))
    }

    /// Lets an element through again, and returns the condition it had.
    pub fn clear_condition(&mut self, element: &E) -> Option<Arc<Expr<P, V>>> {
        self.conditions.remove(element)
    }

    /// The condition an element is held to, if any.
    pub fn condition(&self, element: &E) -> Option<&Arc<Expr<P, V>>> {
        self.conditions.get(element)
    }

    /// Whether an element has a condition at all, whether or not it holds.
    pub fn is_restricted(&self, element: &E) -> bool {
        self.conditions.contains_key(element)
    }

    /// How many elements are restricted.
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// Whether nothing is restricted, in which case the gate allows everything.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Every restricted element.
    pub fn restricted(&self) -> impl Iterator<Item = &E> {
        self.conditions.keys()
    }

    /// Whether an element satisfies an expression the gate has not been given,
    /// checked against the schema first.
    pub fn satisfies(
        &self,
        element: &E,
        expression: &Expr<P, V>,
    ) -> Result<bool, SchemaError<P, V::Kind>> {
        self.store.satisfies(element, expression)
    }

    /// Drops every condition, leaving the state and facts standing.
    pub fn clear(&mut self) {
        self.conditions.clear();
    }

    /// Drops every condition, and the state and facts with them.
    pub fn reset(&mut self) {
        self.conditions.clear();
        self.store.clear();
    }
}

// ---------------------------------------------------------------------------
// The state conditions are judged against
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Gate<E, P, V> {
    /// The state, for reading.
    pub fn store(&self) -> &PropertyStore<E, P, V> {
        &self.store
    }

    /// The state, to change directly.
    ///
    /// Safe to hand out here, unlike on a condition index, because a gate keeps
    /// nothing derived from the state: it looks afresh every time it is asked.
    pub fn store_mut(&mut self) -> &mut PropertyStore<E, P, V> {
        &mut self.store
    }

    /// The element properties, for reading.
    pub fn state(&self) -> &PropertyQuery<E, P, V> {
        self.store.properties()
    }

    /// Declares a property that elements carry individually, and the kind of
    /// value it accepts.
    pub fn register_property(
        &mut self,
        property: P,
        kind: PropertyKind,
        value_kind: V::Kind,
        comparability: Comparability,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.store
            .register_property(property, kind, value_kind, comparability)
    }

    /// Declares a per-element property whose values run from smaller to larger,
    /// so that comparisons against it are allowed.
    pub fn register_ordered_property(
        &mut self,
        property: P,
        kind: PropertyKind,
        value_kind: V::Kind,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.register_property(property, kind, value_kind, Comparability::Ordered)
    }

    /// Declares a per-element property whose values are labels with no order, so
    /// that a comparison against it is refused.
    pub fn register_categorical_property(
        &mut self,
        property: P,
        kind: PropertyKind,
        value_kind: V::Kind,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.register_property(property, kind, value_kind, Comparability::Categorical)
    }

    /// Declares a world-level property whose values run from smaller to larger.
    pub fn register_ordered_fact(
        &mut self,
        property: P,
        value_kind: V::Kind,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.register_fact(property, value_kind, Comparability::Ordered)
    }

    /// Declares a world-level property whose values are labels with no order.
    pub fn register_categorical_fact(
        &mut self,
        property: P,
        value_kind: V::Kind,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.register_fact(property, value_kind, Comparability::Categorical)
    }

    /// Whether a property or fact was declared as having an order.
    pub fn comparability(&self, property: &P) -> Option<Comparability> {
        self.store.comparability(property)
    }

    /// Declares a per-element property that carries no value.
    pub fn register_flag(&mut self, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.register_flag(property)
    }

    /// Declares a property belonging to the world rather than to any element.
    pub fn register_fact(
        &mut self,
        property: P,
        value_kind: V::Kind,
        comparability: Comparability,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.store
            .register_fact(property, value_kind, comparability)
    }

    /// Declares a world-level property that carries no value.
    pub fn register_fact_flag(&mut self, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.register_fact_flag(property)
    }

    /// Checks an expression against the schema.
    pub fn validate(&self, expression: &Expr<P, V>) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.validate(expression)
    }

    /// Sets an element's property to one value, replacing any it held.
    pub fn set(
        &mut self,
        element: E,
        property: P,
        value: V,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.set(element, property, value)
    }

    /// Adds a value to an element's property, keeping the ones it has.
    pub fn add_value(
        &mut self,
        element: E,
        property: P,
        value: V,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.add_value(element, property, value)
    }

    /// Marks a valueless property on an element.
    pub fn set_flag(&mut self, element: E, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.set_flag(element, property)
    }

    /// Removes an element's property entirely.
    pub fn clear_property(&mut self, element: &E, property: &P) -> bool {
        self.store.clear_property(element, property)
    }

    /// Removes an element's state. Its condition stays, and bars it until the
    /// element is back, since an element outside the universe satisfies
    /// nothing.
    pub fn remove_element(&mut self, element: &E) -> bool {
        self.store.remove_element(element)
    }

    /// Adds an element with no properties.
    pub fn insert_element(&mut self, element: E) -> bool {
        self.store.insert_element(element)
    }

    /// Sets a fact to one value, replacing any it held.
    pub fn set_fact(&mut self, property: P, value: V) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.set_fact(property, value)
    }

    /// Raises a fact that carries no value.
    pub fn set_fact_flag(&mut self, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.set_fact_flag(property)
    }

    /// Drops a fact, and returns whether it was set.
    pub fn clear_fact(&mut self, property: &P) -> bool {
        self.store.clear_fact(property)
    }

    /// Whether a fact is set at all.
    pub fn fact_holds(&self, property: &P) -> bool {
        self.store.fact_holds(property)
    }

    /// Whether a fact carries a value.
    pub fn fact_is(&self, property: &P, value: &V) -> bool {
        self.store.fact_is(property, value)
    }
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// The restricted elements, not the ones a gate happens to allow.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Collection for Gate<E, P, V> {
    type Item = E;

    fn len(&self) -> usize {
        self.conditions.len()
    }

    fn contains(&self, item: &E) -> bool {
        self.is_restricted(item)
    }

    fn elements(&self) -> impl Iterator<Item = &E> {
        self.restricted()
    }
}

/// One condition per element; requiring again replaces it.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> UniqueCollection for Gate<E, P, V> {}

/// Removing an element means lifting its restriction, not dropping its state.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> CollectionRemove for Gate<E, P, V> {
    fn remove(&mut self, item: &E) -> bool {
        self.clear_condition(item).is_some()
    }

    fn clear(&mut self) {
        self.conditions.clear();
    }

    fn retain<F: FnMut(&E) -> bool>(&mut self, mut keep: F) {
        self.conditions.retain(|element, _| keep(element));
    }
}

/// Each restricted element looks up the condition it is held to.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Map for Gate<E, P, V> {
    type Key = E;
    type Value = Arc<Expr<P, V>>;
    type Mapped = Arc<Expr<P, V>>;

    fn len(&self) -> usize {
        self.conditions.len()
    }

    fn contains_key(&self, key: &E) -> bool {
        self.is_restricted(key)
    }

    fn get(&self, key: &E) -> Option<&Arc<Expr<P, V>>> {
        self.condition(key)
    }

    fn contains_pair(&self, key: &E, value: &Arc<Expr<P, V>>) -> bool {
        self.condition(key).is_some_and(|held| held == value)
    }

    fn keys(&self) -> impl Iterator<Item = &E> {
        self.restricted()
    }

    fn pairs(&self) -> impl Iterator<Item = (&E, &Arc<Expr<P, V>>)> {
        self.conditions.iter()
    }
}

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> DeterministicOrder
    for Gate<E, P, V>
{
}

impl<E: Element + fmt::Debug, P: Element + fmt::Debug, V: Element + Kinded + PartialOrd> fmt::Debug
    for Gate<E, P, V>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gate")
            .field("restricted", &self.conditions.len())
            .field("store", &self.store)
            .finish()
    }
}

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> fmt::Display for Gate<E, P, V> {
    /// How much it is holding conditions for.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} restricted", self.conditions.len())
    }
}
