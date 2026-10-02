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
