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
