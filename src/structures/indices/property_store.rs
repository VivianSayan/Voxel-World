//! Element properties and world facts, with expressions evaluated against them.

use super::property_query::{Comparability, Expr, PropertyKind, PropertyQuery, SchemaError};
use crate::structures::collections::Set;
use crate::structures::hashing::FastHashMap;
use crate::structures::traits::{Element, Kinded};
use std::fmt;

/// State that expressions are asked about: what each element holds, and what is
/// true of the world.
///
/// # What it adds to [`PropertyQuery`]
///
/// Two things. **Facts**, which belong to the world rather than to any element
/// and are therefore read the same way by every expression — the season, the
/// weather, whether a war is on. Without them, state that every element cares
/// about has to be copied onto every element and written again on each of them
/// whenever it changes.
///
/// And **single-element evaluation**: whether one element satisfies an
/// expression, answered without building the set of everything that matches,
/// which for a negation would be the whole universe.
///
/// # Who uses it
///
/// [`ConditionIndex`](super::ConditionIndex) holds one and watches it, reporting
/// conditions as they *become* true. [`Gate`](super::Gate) holds one and asks it,
/// answering whether a condition is true *now*. The schema rules are here so
/// that there is one copy of them rather than one per user.
///
/// # Schema
///
/// Every property is registered with the kind of value it accepts, and a
/// property is the world's or an element's, never both. See
/// [`Kinded`] and [`SchemaError`].
#[derive(Clone)]
pub struct PropertyStore<E, P, V: Kinded> {
    state: PropertyQuery<E, P, V>,
    /// World-level properties, which every expression reads alike.
    facts: FastHashMap<P, Set<V>>,
    /// Which properties are the world's rather than an element's.
    fact_properties: Set<P>,
    /// The value kind each valued fact accepts. Flag facts have none.
    fact_kinds: FastHashMap<P, V::Kind>,
    /// Whether each valued fact's values are ordered or merely labels.
    fact_comparabilities: FastHashMap<P, Comparability>,
}

impl<E, P, V: Kinded> Default for PropertyStore<E, P, V> {
    fn default() -> Self {
        Self {
            state: PropertyQuery::default(),
            facts: FastHashMap::default(),
            fact_properties: Set::default(),
            fact_kinds: FastHashMap::default(),
            fact_comparabilities: FastHashMap::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> PropertyStore<E, P, V> {
    /// An empty store, with no properties, facts or elements.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a property that elements carry individually: the kind of value
    /// it accepts, and whether those values have an order.
    pub fn register_property(
        &mut self,
        property: P,
        kind: PropertyKind,
        value_kind: V::Kind,
        comparability: Comparability,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.reserve_name(&property)?;

        self.state
            .register_property(property, kind, value_kind, comparability)
    }

    /// Declares a per-element property whose values run from smaller to larger.
    pub fn register_ordered_property(
        &mut self,
        property: P,
        kind: PropertyKind,
        value_kind: V::Kind,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.register_property(property, kind, value_kind, Comparability::Ordered)
    }

    /// Declares a per-element property whose values are labels with no order.
    pub fn register_categorical_property(
        &mut self,
        property: P,
        kind: PropertyKind,
        value_kind: V::Kind,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.register_property(property, kind, value_kind, Comparability::Categorical)
    }

    /// Declares a per-element property that carries no value.
    pub fn register_flag(&mut self, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        self.reserve_name(&property)?;

        self.state.register_flag(property)
    }

    /// Declares a property that belongs to the world rather than to any
    /// element, and that every expression therefore reads alike.
    ///
    /// Registering a fact again is accepted only when it names the same value
    /// kind: changing what it accepts would leave expressions already built
    /// against it comparing values nothing re-checked.
    pub fn register_fact(
        &mut self,
        property: P,
        value_kind: V::Kind,
        comparability: Comparability,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        if self.state.has_property(&property) {
            return Err(SchemaError::Conflict {
                property,
                current: PropertyKind::Single,
                incoming: PropertyKind::Single,
            });
        }

        if self.fact_properties.contains(&property) {
            return match self.fact_kinds.get(&property) {
                Some(expected) if *expected != value_kind => Err(SchemaError::ValueKindConflict {
                    expected: expected.clone(),
                    property,
                    incoming: value_kind,
                }),
                Some(_) => Ok(()),
                // Already a flag fact, which holds no value at all.
                None => Err(SchemaError::Conflict {
                    property,
                    current: PropertyKind::Flag,
                    incoming: PropertyKind::Single,
                }),
            };
        }

        self.fact_properties.insert(property.clone());
        self.fact_kinds.insert(property.clone(), value_kind);
        self.fact_comparabilities.insert(property, comparability);

        Ok(())
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

    /// Declares a world-level property that carries no value, and is therefore
    /// only ever raised or dropped.
    pub fn register_fact_flag(&mut self, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        if self.state.has_property(&property) {
            return Err(SchemaError::Conflict {
                property,
                current: PropertyKind::Flag,
                incoming: PropertyKind::Flag,
            });
        }

        if self.fact_kinds.contains_key(&property) {
            return Err(SchemaError::Conflict {
                property,
                current: PropertyKind::Single,
                incoming: PropertyKind::Flag,
            });
        }

        self.fact_properties.insert(property);

        Ok(())
    }

    /// Whether a property is the world's rather than an element's.
    pub fn is_fact(&self, property: &P) -> bool {
        self.fact_properties.contains(property)
    }

    /// The value kind a property or fact accepts, or `None` for a flag or an
    /// unregistered name.
    pub fn value_kind(&self, property: &P) -> Option<&V::Kind> {
        if self.fact_properties.contains(property) {
            return self.fact_kinds.get(property);
        }

        self.state.value_kind(property)
    }

    /// The element properties, for reading.
    pub fn properties(&self) -> &PropertyQuery<E, P, V> {
        &self.state
    }

    /// Rejects a name the other half of the schema has already taken.
    ///
    /// A property is the world's or an element's, never both: one name with two
    /// meanings is exactly the ambiguity a schema exists to prevent.
    fn reserve_name(&self, property: &P) -> Result<(), SchemaError<P, V::Kind>> {
        if self.fact_properties.contains(property) {
            return Err(SchemaError::Conflict {
                property: property.clone(),
                current: PropertyKind::Single,
                incoming: PropertyKind::Single,
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Element state
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> PropertyStore<E, P, V> {
    /// Adds an element with no properties, and returns whether it was new.
    pub fn insert_element(&mut self, element: E) -> bool {
        self.state.insert_element(element)
    }

    /// Sets an element's property to one value, replacing any it held.
    pub fn set(
        &mut self,
        element: E,
        property: P,
        value: V,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.state.set(element, property, value)
    }

    /// Adds a value to an element's property, keeping the ones it has.
    pub fn add_value(
        &mut self,
        element: E,
        property: P,
        value: V,
    ) -> Result<(), SchemaError<P, V::Kind>> {
        self.state.add_value(element, property, value)
    }

    /// Marks a valueless property on an element.
    pub fn set_flag(&mut self, element: E, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        self.state.set_flag(element, property)
    }

    /// Removes one value from an element's property.
    pub fn remove_value(&mut self, element: &E, property: &P, value: &V) -> bool {
        self.state.remove_value(element, property, value)
    }

    /// Removes an element's property entirely.
    pub fn clear_property(&mut self, element: &E, property: &P) -> bool {
        self.state.clear_property(element, property)
    }

    /// Removes an element and everything it held.
    pub fn remove_element(&mut self, element: &E) -> bool {
        self.state.remove_element(element)
    }

    /// Whether an element is in the universe at all, which is what negation and
    /// the empty conjunction read.
    ///
    /// Not the same as holding a property: an element can be in the universe
    /// with nothing on it.
    pub fn in_universe(&self, element: &E) -> bool {
        self.state.universe().contains(element)
    }

    /// Removes every property, fact and element.
    pub fn clear(&mut self) {
        self.state.clear();
        self.facts.clear();
        self.fact_properties.clear();
        self.fact_kinds.clear();
        self.fact_comparabilities.clear();
    }
}

// ---------------------------------------------------------------------------
// World facts
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> PropertyStore<E, P, V> {
    /// Sets a fact to one value, replacing any it held.
    pub fn set_fact(&mut self, property: P, value: V) -> Result<(), SchemaError<P, V::Kind>> {
        self.check(&property, &value)?;

        let mut values: Set<V> = Set::new();
        values.insert(value);
        self.facts.insert(property, values);

        Ok(())
    }

    /// Adds a value to a fact, keeping the ones it has.
    pub fn add_fact_value(&mut self, property: P, value: V) -> Result<(), SchemaError<P, V::Kind>> {
        self.check(&property, &value)?;

        self.facts.entry(property).or_default().insert(value);

        Ok(())
    }

    /// Raises a fact that carries no value.
    pub fn set_fact_flag(&mut self, property: P) -> Result<(), SchemaError<P, V::Kind>> {
        if !self.fact_properties.contains(&property) {
            return Err(SchemaError::Unregistered { property });
        }

        if self.fact_kinds.contains_key(&property) {
            return Err(SchemaError::Conflict {
                property,
                current: PropertyKind::Single,
                incoming: PropertyKind::Flag,
            });
        }

        self.facts.entry(property).or_default();

        Ok(())
    }

    /// Drops a fact, and returns whether it was set.
    pub fn clear_fact(&mut self, property: &P) -> bool {
        self.facts.remove(property).is_some()
    }

    /// Whether a fact is set at all.
    pub fn fact_holds(&self, property: &P) -> bool {
        self.facts.contains_key(property)
    }

    /// Whether a fact carries a value.
    pub fn fact_is(&self, property: &P, value: &V) -> bool {
        self.facts
            .get(property)
            .is_some_and(|values| values.contains(value))
    }
}

// ---------------------------------------------------------------------------
// Asking about it
// ---------------------------------------------------------------------------

impl<E: Element, P: Element, V: Element + Kinded + PartialOrd> PropertyStore<E, P, V> {
    /// Checks a value against what a property or fact accepts.
    ///
    /// The same rule either side of the element/world divide: the name must be
    /// registered, it must hold values, and the value must be of the declared
    /// kind.
    pub fn check(&self, property: &P, value: &V) -> Result<(), SchemaError<P, V::Kind>> {
        if !self.fact_properties.contains(property) {
            return self.state.check(property, value);
        }

        let found: V::Kind = value.kind();

        match self.fact_kinds.get(property) {
            Some(expected) if *expected != found => Err(SchemaError::WrongValueKind {
                property: property.clone(),
                expected: expected.clone(),
                found,
            }),
            Some(_) => Ok(()),
            None => Err(SchemaError::FlagTakesNoValue {
                property: property.clone(),
            }),
        }
    }

    /// Checks a value the way [`PropertyStore::check`] does, and that its kind
    /// has an order to compare along.
    pub fn check_ordered(&self, property: &P, value: &V) -> Result<(), SchemaError<P, V::Kind>> {
        self.check(property, value)?;

        if self.comparability(property) != Some(Comparability::Ordered) {
            return Err(SchemaError::NotOrdered {
                property: property.clone(),
                kind: value.kind(),
            });
        }

        Ok(())
    }

    /// Whether a property or fact was declared as having an order.
    ///
    /// The declaration is what `check_ordered` consults: asking [`PartialOrd`]
    /// instead would accept a categorical property whose implementation was
    /// derived, then compare its labels by the order they were written in.
    pub fn comparability(&self, property: &P) -> Option<Comparability> {
        if self.fact_properties.contains(property) {
            return self.fact_comparabilities.get(property).copied();
        }

        self.state.comparability(property)
    }

    /// Checks every value an expression compares, across both element
    /// properties and world facts.
    pub fn validate(&self, expression: &Expr<P, V>) -> Result<(), SchemaError<P, V::Kind>> {
        match expression {
            Expr::Has(property) => {
                if self.fact_properties.contains(property) || self.state.has_property(property) {
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

    /// Whether an element satisfies an expression as the store stands.
    ///
    /// The expression is checked against the schema first, so one that could
    /// never match reports why instead of answering `false`.
    pub fn satisfies(
        &self,
        element: &E,
        expression: &Expr<P, V>,
    ) -> Result<bool, SchemaError<P, V::Kind>> {
        self.validate(expression)?;

        Ok(self.satisfies_unchecked(element, expression))
    }

    /// Answers for an expression already known to fit the schema.
    ///
    /// Callers that validated an expression when it was registered re-check it
    /// through here, so the tree is walked for validation once rather than once
    /// per evaluation.
    pub(crate) fn satisfies_unchecked(&self, element: &E, expression: &Expr<P, V>) -> bool {
        match expression {
            Expr::Has(property) => {
                if self.fact_properties.contains(property) {
                    self.facts.contains_key(property)
                } else {
                    self.state.has(element, property)
                }
            }
            Expr::Is(property, value) => {
                if self.fact_properties.contains(property) {
                    self.fact_is(property, value)
                } else {
                    self.state.has_value(element, property, value)
                }
            }
            Expr::Compare(property, comparison, value) => {
                if self.fact_properties.contains(property) {
                    self.facts.get(property).is_some_and(|values| {
                        values.iter().any(|held| comparison.holds(held, value))
                    })
                } else {
                    self.state
                        .values(element, property)
                        .any(|held| comparison.holds(held, value))
                }
            }
            Expr::OneOf(property, values) => values.iter().any(|value| {
                if self.fact_properties.contains(property) {
                    self.fact_is(property, value)
                } else {
                    self.state.has_value(element, property, value)
                }
            }),
            Expr::And(children) if children.is_empty() => self.in_universe(element),
            Expr::And(children) => children
                .iter()
                .all(|child| self.satisfies_unchecked(element, child)),
            Expr::Or(children) => children
                .iter()
                .any(|child| self.satisfies_unchecked(element, child)),
            Expr::Not(child) => {
                self.in_universe(element) && !self.satisfies_unchecked(element, child)
            }
        }
    }
}

impl<E: Element, P: Element + fmt::Debug, V: Element + Kinded + PartialOrd> fmt::Debug
    for PropertyStore<E, P, V>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PropertyStore")
            .field("elements", &self.state.len())
            .field("facts", &self.facts.len())
            .finish()
    }
}
