//! Conditions registered against element state, reported when they come true.

use super::property_query::{Comparability, Expr, PropertyKind, PropertyQuery, SchemaError};
use super::property_store::PropertyStore;
use crate::structures::collections::{OrderedSet, Set};
use crate::structures::hashing::FastHashMap;
use crate::structures::traits::{
    Collection, CollectionRemove, DeterministicOrder, Element, Kinded, Map, Pending,
    UniqueCollection,
};
use std::fmt;
use std::sync::Arc;

/// What becomes of a condition once it has been met.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OnSatisfied {
    /// The condition is dropped, having fired once.
    ///
    /// The default, and what a gate wants: a seed germinates when the ground
    /// thaws and is then no longer waiting on anything.
    #[default]
    Once,
    /// The condition stays, and fires again the next time it becomes true
    /// having been false in between.
    ///
    /// It does **not** fire repeatedly while the condition holds. Something
    /// that is already satisfied has nothing to report until it stops being
    /// satisfied and starts again.
    Rearm,
}

/// One registered condition.
struct Watch<P, V> {
    expression: Arc<Expr<P, V>>,
    mode: OnSatisfied,
    /// The properties whose change could flip this condition, which is what the
    /// reverse index is built from.
    dependencies: Set<P>,
    /// Whether the condition reads the universe, and so must be re-checked
    /// whenever the set of elements changes.
    universal: bool,
    /// The last answer, so that only a false-to-true crossing fires.
    satisfied: bool,
}

impl<P: Clone, V> Clone for Watch<P, V> {
    fn clone(&self) -> Self {
        Self {
            expression: Arc::clone(&self.expression),
            mode: self.mode,
            dependencies: self.dependencies.clone(),
            universal: self.universal,
            satisfied: self.satisfied,
        }
    }
}

/// Element state, plus conditions over it that report themselves when met.
///
/// # What it is
///
/// [`PropertyQuery`] answers *which elements match this expression now*: the
/// caller asks, and the index looks. This is the other direction. Conditions are
/// registered ahead of time, the index owns the state they read, and every
/// change works out which of them have just become true. Nothing has to be
/// polled, and no caller has to remember which conditions might care about the
/// write it is making.
///
/// The condition language is [`Expr`], the same one [`PropertyQuery`] takes, so
/// an expression can be evaluated either way round without being written twice.
///
/// # Type parameters
///
/// - `E` identifies the elements conditions are registered for and state is
///   held against.
/// - `P` identifies properties, both an element's own and the world's facts.
/// - `V` is the value type shared by every valued property.
///
/// All three are [`Element`], because all three are cloned into the indices.
///
/// # When a condition fires
///
/// On the crossing from false to true, never on the level. A condition that is
/// already true when it is registered fires at once, so that registration order
/// cannot quietly change behaviour. What happens next is the condition's
/// [`OnSatisfied`] mode: it is dropped, or it re-arms and waits for the next
/// crossing.
///
/// Firing does not call anything. Satisfied elements are put on a ready list
/// that the caller drains with [`ConditionIndex::take_ready`] whenever it
/// suits. Running triggers from inside a setter would interleave them with
/// whatever the caller was part-way through and make their order depend on call
/// sites; a drained list keeps writes cheap, batches the work, and gives the
/// results one order.
///
/// # What it costs
///
/// Each condition records the properties it reads, taken from the expression
/// when it is registered, and the index keeps the reverse map. A write to an
/// element's property re-checks that element's condition, and only if the
/// condition reads that property. A change to a world fact re-checks the
/// conditions that read it, and no others.
///
/// Negation is the expensive case. `Not`, and the empty `And`, read the
/// universe rather than any one property, so they have to be re-checked
/// whenever an element joins or leaves. Conditions built only from `Has`, `Is`,
/// `And` and `Or` cost nothing on membership changes.
///
/// # What it deliberately does not do
///
/// A condition reads its own element's properties and the world's facts. It
/// cannot read *another* element's properties: that is a join, and joins are
/// what turn a watch list into a rule network, with join memories and their own
/// failure modes. Where one element's state should gate another's, register a
/// fact or mirror the value onto the elements that care.
///
/// # Example
///
/// ```
/// use voxel_world::structures::indices::{ConditionIndex, Expr, PropertyKind};
/// use voxel_world::structures::traits::Kinded;
///
/// // One value type covering several logical kinds, as a schema-aware store
/// // expects.
/// #[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// enum Value {
///     Ground(&'static str),
///     Season(&'static str),
///     Depth(u32),
/// }
///
/// #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
/// enum Kind {
///     Ground,
///     Season,
///     Depth,
/// }
///
/// impl PartialOrd for Value {
///     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
///         match (self, other) {
///             (Self::Depth(a), Self::Depth(b)) => a.partial_cmp(b),
///             _ => None,
///         }
///     }
/// }
///
/// impl Kinded for Value {
///     type Kind = Kind;
///
///     fn kind(&self) -> Kind {
///         match self {
///             Self::Ground(_) => Kind::Ground,
///             Self::Season(_) => Kind::Season,
///             Self::Depth(_) => Kind::Depth,
///         }
///     }
/// }
///
/// let mut gates: ConditionIndex<u32, &str, Value> = ConditionIndex::new();
/// gates.register_categorical_property("ground", PropertyKind::Single, Kind::Ground)?;
/// gates.register_categorical_fact("season", Kind::Season)?;
///
/// // This seed germinates on thawed ground, in spring.
/// gates.watch(
///     7,
///     Expr::and([
///         Expr::is("ground", Value::Ground("thawed")),
///         Expr::is("season", Value::Season("spring")),
///     ]),
/// )?;
///
/// gates.set(7, "ground", Value::Ground("thawed"))?;
/// assert!(gates.take_ready().is_empty(), "not spring yet");
///
/// // One fact, and everything waiting on spring comes through at once.
/// gates.set_fact("season", Value::Season("spring"))?;
/// assert_eq!(gates.take_ready(), vec![7]);
///
/// // A depth is not a ground, and the schema says so rather than storing it.
/// assert!(gates.set(7, "ground", Value::Depth(3)).is_err());
/// // So is a condition that could never match.
/// assert!(gates.watch(8, Expr::is("ground", Value::Depth(3))).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ConditionIndex<E, P, V: Kinded> {
    store: PropertyStore<E, P, V>,
    watches: FastHashMap<E, Watch<P, V>>,
    /// Property to the elements whose condition reads it.
    watchers: FastHashMap<P, Set<E>>,
    /// The elements whose condition reads the universe.
    universal: Set<E>,
    ready: OrderedSet<E>,
}

impl<E, P, V: Kinded> Default for ConditionIndex<E, P, V> {
    fn default() -> Self {
        Self {
            store: PropertyStore::default(),
            watches: FastHashMap::default(),
            watchers: FastHashMap::default(),
            universal: Set::default(),
            ready: OrderedSet::default(),
        }
    }
}

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Clone for ConditionIndex<E, P, V> {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            watches: self.watches.clone(),
            watchers: self.watchers.clone(),
            universal: self.universal.clone(),
            ready: self.ready.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// The state conditions are judged against
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> ConditionIndex<E, P, V> {
    /// An empty index, with no properties, facts or conditions.
    pub fn new() -> Self {
        Self::default()
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

    /// Declares a property that belongs to the world rather than to any
    /// element, and that every condition therefore reads alike.
    ///
    /// The season, the weather, whether a war is on: state that would otherwise
    /// have to be copied onto every element that cares about it, and written
    /// again on each of them whenever it changed.
    pub fn register_fact(
        &mut self,
        property: P,
        value_kind: V::Kind,
        comparability: Comparability,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.store
            .register_fact(property, value_kind, comparability)
    }

    /// Declares a world-level property that carries no value, and is therefore
    /// only ever raised or dropped.
    pub fn register_fact_flag(&mut self, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.register_fact_flag(property)
    }

    /// Whether a property is the world's rather than an element's.
    pub fn is_fact(&self, property: &P) -> bool {
        self.store.is_fact(property)
    }

    /// The value kind a property or fact accepts, or `None` for a flag or an
    /// unregistered name.
    pub fn value_kind(&self, property: &P) -> Option<&V::Kind> {
        self.store.value_kind(property)
    }

    /// Checks a value against what a property or fact accepts.
    pub fn check(&self, property: &P, value: &V) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.check(property, value)
    }

    /// Checks every value an expression compares against the schema.
    ///
    /// Where a mistyped condition is caught: an expression that asks for a
    /// health of "banana" can never match anything, and saying so when it is
    /// built beats waiting for it never to fire.
    pub fn validate(&self, expression: &Expr<P, V>) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.validate(expression)
    }

    /// The state conditions are evaluated against, for reading.
    ///
    /// Mutating methods are not forwarded from here on purpose: a write that
    /// went straight to the store would not be noticed, and conditions would
    /// silently stop firing.
    pub fn store(&self) -> &PropertyStore<E, P, V> {
        &self.store
    }

    /// The element properties, for reading.
    pub fn state(&self) -> &PropertyQuery<E, P, V> {
        self.store.properties()
    }
}

// ---------------------------------------------------------------------------
// Element state
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> ConditionIndex<E, P, V> {
    /// Adds an element with no properties, and returns whether it was new.
    pub fn insert_element(&mut self, element: E) -> bool {
        let added: bool = self.store.insert_element(element);

        if added {
            self.recheck_universal();
        }

        added
    }

    /// Sets an element's property to one value, replacing any it held.
    pub fn set(
        &mut self,
        element: E,
        property: P,
        value: V,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        let fresh: bool = !self.store.in_universe(&element);

        self.store.set(element.clone(), property.clone(), value)?;
        self.settle(&element, &property, fresh);

        Ok(())
    }

    /// Adds a value to an element's property, keeping the ones it has.
    pub fn add_value(
        &mut self,
        element: E,
        property: P,
        value: V,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        let fresh: bool = !self.store.in_universe(&element);

        self.store
            .add_value(element.clone(), property.clone(), value)?;
        self.settle(&element, &property, fresh);

        Ok(())
    }

    /// Marks a valueless property on an element.
    pub fn set_flag(&mut self, element: E, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        let fresh: bool = !self.store.in_universe(&element);

        self.store.set_flag(element.clone(), property.clone())?;
        self.settle(&element, &property, fresh);

        Ok(())
    }

    /// Removes one value from an element's property, and returns whether it
    /// was there.
    pub fn remove_value(&mut self, element: &E, property: &P, value: &V) -> bool {
        let removed: bool = self.store.remove_value(element, property, value);

        if removed {
            self.settle(element, property, false);
        }

        removed
    }

    /// Removes an element's property entirely, and returns whether it had one.
    pub fn clear_property(&mut self, element: &E, property: &P) -> bool {
        let cleared: bool = self.store.clear_property(element, property);

        if cleared {
            self.settle(element, property, false);
        }

        cleared
    }

    /// Removes an element and everything it held, and returns whether it was
    /// there.
    ///
    /// Its condition stays registered. An element that has left the universe
    /// satisfies nothing — negation included, since `Not` matches only within
    /// the universe — so the condition simply waits, and works again if the
    /// element comes back.
    pub fn remove_element(&mut self, element: &E) -> bool {
        let removed: bool = self.store.remove_element(element);

        if removed {
            self.recheck(element);
            self.recheck_universal();
        }

        removed
    }
}

// ---------------------------------------------------------------------------
// World facts
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> ConditionIndex<E, P, V> {
    /// Sets a fact to one value, replacing any it held.
    pub fn set_fact(&mut self, property: P, value: V) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.set_fact(property.clone(), value)?;
        self.recheck_watchers(&property);

        Ok(())
    }

    /// Adds a value to a fact, keeping the ones it has.
    pub fn add_fact_value(&mut self, property: P, value: V) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.add_fact_value(property.clone(), value)?;
        self.recheck_watchers(&property);

        Ok(())
    }

    /// Raises a fact that carries no value.
    pub fn set_fact_flag(&mut self, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        self.store.set_fact_flag(property.clone())?;
        self.recheck_watchers(&property);

        Ok(())
    }

    /// Drops a fact, and returns whether it was set.
    pub fn clear_fact(&mut self, property: &P) -> bool {
        if !self.store.clear_fact(property) {
            return false;
        }

        self.recheck_watchers(property);

        true
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
// Conditions
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> ConditionIndex<E, P, V> {
    /// Registers a condition for an element, to fire once when it comes true.
    ///
    /// Replaces whatever that element was waiting on. Returns whether it was
    /// satisfied straight away, in which case the element is already on the
    /// ready list.
    ///
    /// The expression is checked against the schema first, so a condition that
    /// compares a property against a value of the wrong kind is refused here
    /// rather than silently never matching.
    pub fn watch(
        &mut self,
        element: E,
        expression: Arc<Expr<P, V>>,
    ) -> Result<bool, SchemaError<P, V::Kind>> {
        self.watch_with(element, expression, OnSatisfied::Once)
    }

    /// Registers a condition for an element with an explicit
    /// [`OnSatisfied`] mode.
    ///
    /// Replaces whatever that element was waiting on. Returns whether it was
    /// satisfied straight away, and refuses an expression the schema rejects,
    /// in which case what the element was waiting on is left alone.
    pub fn watch_with(
        &mut self,
        element: E,
        expression: Arc<Expr<P, V>>,
        mode: OnSatisfied,
    ) -> Result<bool, SchemaError<P, V::Kind>> {
        self.validate(&expression)?;
        self.unwatch(&element);

        let (dependencies, universal): (Set<P>, bool) = Self::dependencies(&expression);

        for property in dependencies.iter() {
            self.watchers
                .entry(property.clone())
                .or_default()
                .insert(element.clone());
        }

        if universal {
            self.universal.insert(element.clone());
        }

        let satisfied: bool = self.store.satisfies_unchecked(&element, &expression);

        self.watches.insert(
            element.clone(),
            Watch {
                expression,
                mode,
                dependencies,
                universal,
                satisfied,
            },
        );

        if satisfied {
            self.fire(&element);
        }

        Ok(satisfied)
    }

    /// Drops an element's condition, and returns whether it had one.
    ///
    /// Leaves its state and its place on the ready list alone.
    pub fn unwatch(&mut self, element: &E) -> bool {
        let Some(watch) = self.watches.remove(element) else {
            return false;
        };

        self.forget(element, &watch);

        true
    }

    /// The condition an element is waiting on, if any.
    pub fn condition(&self, element: &E) -> Option<&Arc<Expr<P, V>>> {
        self.watches.get(element).map(|watch| &watch.expression)
    }

    /// What happens to an element's condition when it is met.
    pub fn mode(&self, element: &E) -> Option<OnSatisfied> {
        self.watches.get(element).map(|watch| watch.mode)
    }

    /// Whether an element has a condition registered.
    pub fn is_watching(&self, element: &E) -> bool {
        self.watches.contains_key(element)
    }

    /// Whether an element's condition currently holds.
    ///
    /// This is the stored answer, which is what firing is decided from, not a
    /// fresh evaluation.
    pub fn is_satisfied(&self, element: &E) -> bool {
        self.watches
            .get(element)
            .is_some_and(|watch| watch.satisfied)
    }

    /// How many conditions are registered.
    pub fn len(&self) -> usize {
        self.watches.len()
    }

    /// Whether no condition is registered.
    pub fn is_empty(&self) -> bool {
        self.watches.is_empty()
    }

    /// Every element with a condition registered.
    pub fn watched(&self) -> impl Iterator<Item = &E> {
        self.watches.keys()
    }

    /// Whether an element satisfies an expression as the index stands.
    ///
    /// Checked against the schema first, so an expression that could never
    /// match reports why instead of answering `false`.
    pub fn satisfies(
        &self,
        element: &E,
        expression: &Expr<P, V>,
    ) -> Result<bool, SchemaError<P, V::Kind>> {
        self.store.satisfies(element, expression)
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> ConditionIndex<E, P, V> {
    /// Takes everything whose condition has been met since the last call, in
    /// the order the conditions came true.
    pub fn take_ready(&mut self) -> Vec<E> {
        std::mem::take(&mut self.ready).into_vec()
    }

    /// What is waiting to be taken, without taking it.
    pub fn ready(&self) -> impl Iterator<Item = &E> {
        self.ready.iter()
    }

    /// How many elements are waiting to be taken.
    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    /// Whether anything is waiting to be taken.
    pub fn has_ready(&self) -> bool {
        !self.ready.is_empty()
    }

    /// Drops what is waiting without reporting it.
    pub fn clear_ready(&mut self) {
        self.ready.clear();
    }

    /// Drops every condition, leaving the state, the facts and the ready list
    /// alone.
    pub fn unwatch_all(&mut self) {
        self.watches.clear();
        self.watchers.clear();
        self.universal.clear();
    }

    /// Removes every condition, every fact and all state.
    pub fn reset(&mut self) {
        self.store.clear();
        self.watches.clear();
        self.watchers.clear();
        self.universal.clear();
        self.ready.clear();
    }
}

// ---------------------------------------------------------------------------
// Keeping the conditions up to date
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> ConditionIndex<E, P, V> {
    /// Whether an element is in the universe at all, which is what negation
    /// and the empty conjunction read.
    ///
    /// Not the same as holding a property: an element can be in the universe
    /// with nothing on it, and [`PropertyQuery::contains`] answers the narrower
    /// question.
    /// The properties an expression reads, and whether it reads the universe.
    fn dependencies(expression: &Expr<P, V>) -> (Set<P>, bool) {
        let mut dependencies: Set<P> = Set::new();
        let mut universal: bool = false;

        expression.for_each_dependency(&mut |property| match property {
            Some(property) => {
                dependencies.insert(property.clone());
            }
            None => universal = true,
        });

        (dependencies, universal)
    }

    /// Re-checks whatever a write to one element's property could have changed.
    fn settle(&mut self, element: &E, property: &P, fresh: bool) {
        if fresh {
            self.recheck_universal();
        }

        if self
            .watches
            .get(element)
            .is_some_and(|watch| watch.dependencies.contains(property))
        {
            self.recheck(element);
        }
    }

    /// Re-checks every condition that reads a property, for a fact that has
    /// changed under all of them at once.
    fn recheck_watchers(&mut self, property: &P) {
        let Some(watchers) = self.watchers.get(property) else {
            return;
        };

        for element in watchers.iter().cloned().collect::<Vec<E>>() {
            self.recheck(&element);
        }
    }

    /// Re-checks every condition that reads the universe, for a membership
    /// change that could have flipped any of them.
    fn recheck_universal(&mut self) {
        if self.universal.is_empty() {
            return;
        }

        for element in self.universal.iter().cloned().collect::<Vec<E>>() {
            self.recheck(&element);
        }
    }

    /// Re-evaluates one condition and acts on a crossing.
    fn recheck(&mut self, element: &E) {
        let Some(watch) = self.watches.get(element) else {
            return;
        };

        let expression: Arc<Expr<P, V>> = Arc::clone(&watch.expression);
        let was: bool = watch.satisfied;
        let now: bool = self.store.satisfies_unchecked(element, &expression);

        if now == was {
            return;
        }

        if let Some(watch) = self.watches.get_mut(element) {
            watch.satisfied = now;
        }

        if now {
            self.fire(element);
        }
    }

    /// Reports a satisfied condition and retires or re-arms it.
    fn fire(&mut self, element: &E) {
        if self.mode(element) == Some(OnSatisfied::Once) {
            self.unwatch(element);
        }

        self.ready.push(element.clone());
    }

    /// Takes an element out of the reverse indices.
    fn forget(&mut self, element: &E, watch: &Watch<P, V>) {
        for property in watch.dependencies.iter() {
            let Some(watchers) = self.watchers.get_mut(property) else {
                continue;
            };

            watchers.remove(element);

            if watchers.is_empty() {
                self.watchers.remove(property);
            }
        }

        if watch.universal {
            self.universal.remove(element);
        }
    }
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// The elements with a condition registered, not the ones that are ready.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Collection
    for ConditionIndex<E, P, V>
{
    type Item = E;

    fn len(&self) -> usize {
        self.watches.len()
    }

    fn contains(&self, item: &E) -> bool {
        self.is_watching(item)
    }

    fn elements(&self) -> impl Iterator<Item = &E> {
        self.watched()
    }
}

/// One condition per element; registering again replaces it.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> UniqueCollection
    for ConditionIndex<E, P, V>
{
}

/// Removing an element means dropping its condition, not its state: state is
/// what conditions are judged against, and outlives any one of them.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> CollectionRemove
    for ConditionIndex<E, P, V>
{
    fn remove(&mut self, item: &E) -> bool {
        self.unwatch(item)
    }

    fn clear(&mut self) {
        self.unwatch_all();
    }

    fn retain<F: FnMut(&E) -> bool>(&mut self, mut keep: F) {
        for element in self
            .watches
            .keys()
            .filter(|element| !keep(element))
            .cloned()
            .collect::<Vec<E>>()
        {
            self.unwatch(&element);
        }
    }
}

/// Each watched element looks up the condition it is waiting on.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Map for ConditionIndex<E, P, V> {
    type Key = E;
    type Value = Arc<Expr<P, V>>;
    type Mapped = Arc<Expr<P, V>>;

    fn len(&self) -> usize {
        self.watches.len()
    }

    fn contains_key(&self, key: &E) -> bool {
        self.is_watching(key)
    }

    fn get(&self, key: &E) -> Option<&Arc<Expr<P, V>>> {
        self.condition(key)
    }

    fn contains_pair(&self, key: &E, value: &Arc<Expr<P, V>>) -> bool {
        self.condition(key).is_some_and(|held| held == value)
    }

    fn keys(&self) -> impl Iterator<Item = &E> {
        self.watched()
    }

    fn pairs(&self) -> impl Iterator<Item = (&E, &Arc<Expr<P, V>>)> {
        self.watches
            .iter()
            .map(|(element, watch)| (element, &watch.expression))
    }
}

/// Held means waiting on a condition; ready means the condition has been met
/// and the element has not been collected yet. An element retired by
/// [`OnSatisfied::Once`] is ready without still being held.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> Pending for ConditionIndex<E, P, V> {
    type Ready = E;

    fn pending_len(&self) -> usize {
        self.watches.len()
    }

    fn ready_len(&self) -> usize {
        self.ready.len()
    }

    fn take_ready(&mut self) -> Vec<E> {
        ConditionIndex::take_ready(self)
    }
}

/// The ready list keeps the order conditions came true in, and every index
/// behind it iterates the same way for the same sequence of writes.
impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> DeterministicOrder
    for ConditionIndex<E, P, V>
{
}

impl<
    E: Element + fmt::Debug,
    P: Element + fmt::Debug,
    V: Element + Kinded + PartialOrd + fmt::Debug,
> fmt::Debug for ConditionIndex<E, P, V>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConditionIndex")
            .field("watching", &self.watches.len())
            .field("store", &self.store)
            .field("ready", &self.ready.len())
            .finish()
    }
}

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> fmt::Display
    for ConditionIndex<E, P, V>
{
    /// As how much is waiting and how much of it is ready.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} watching, {} ready",
            self.watches.len(),
            self.ready.len()
        )
    }
}
