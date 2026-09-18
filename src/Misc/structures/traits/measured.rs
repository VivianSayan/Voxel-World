//! Traits for collections that attach a quantity to each element: a count,
//! a weight or a degree of membership.

use crate::misc::structures::traits::collection::{Collection, CollectionInsert, CollectionRemove};

/// Read access to collections that associate a quantity with each element.
pub trait Measured: Collection {
    /// Quantity type, such as `usize` counts or `f64` weights.
    type Measure: Copy + PartialOrd;

    /// The element's quantity; the zero value when absent.
    fn measure_of(&self, item: &Self::Item) -> Self::Measure;

    /// Sum of all quantities.
    fn total_measure(&self) -> Self::Measure;

    /// Number of different elements.
    fn distinct_len(&self) -> usize;

    /// Each distinct element with its quantity.
    fn measures(&self) -> impl Iterator<Item = (&Self::Item, Self::Measure)>;
}

/// Mutation operations for measured collections.
pub trait MeasuredMut: Measured + CollectionInsert + CollectionRemove {
    /// Sets the quantity outright. A zero quantity removes the element.
    fn set_measure(&mut self, item: Self::Item, measure: Self::Measure);

    /// Combines `amount` into the quantity the way the collection
    /// accumulates: adding counts or weights, or reinforcing membership.
    fn add_measure(&mut self, item: Self::Item, amount: Self::Measure);

    /// Takes `amount` away, removing the element if nothing is left.
    fn subtract_measure(&mut self, item: &Self::Item, amount: Self::Measure);
}
