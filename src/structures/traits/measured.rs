//! Traits for collections that attach a quantity to each element: a count,
//! a weight or a degree of membership.

use crate::random::Random;
use crate::structures::traits::collection::{
    Choose, Collection, CollectionInsert, CollectionRemove,
};

/// Read access to collections that associate a quantity with each element.
pub trait Measured: Collection {
    /// Quantity type, such as `usize` counts or `f64` weights.
    type Measure: Copy + PartialOrd;

    /// Aggregate quantity; summing memberships does not yield a probability.
    type Total: Copy + PartialOrd;

    /// The element's quantity; the zero value when absent.
    fn measure_of(&self, item: &Self::Item) -> Self::Measure;

    /// Sum of all quantities.
    fn total_measure(&self) -> Self::Total;

    /// Number of different elements.
    fn distinct_len(&self) -> usize {
        self.len()
    }

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

/// Picking from a measured collection in proportion to the quantities it holds.
///
/// # Why this exists beside [`Choose`]
///
/// [`Choose`] is each collection's *own* rule for picking, and that rule is not
/// the same everywhere: a `Vec`, a `Set` or an `OrderedSet` picks evenly, while
/// a `MultiSet`, a `WeightedSet` and a `FuzzySet` already pick in proportion to
/// their quantities. Code generic over `C: Choose` therefore cannot tell which
/// it is getting. This trait says so in the type: a bound of `WeightedChoose`
/// is a promise that the quantities decide, and `weighted_chance_of` exposes
/// the resulting probability, which nothing else did.
///
/// The conversion from a quantity to a weight belongs to the implementation,
/// since a count and a membership are not the same kind of number: a multiset
/// weighs by its counts, a weighted set by its weights, a fuzzy set by how
/// strongly each element belongs.
///
/// Every draw is a pure function of the generator's state, so a world seeded
/// through one of these replays exactly, provided the collection also promises
/// [`DeterministicOrder`](super::DeterministicOrder) — which the three in this
/// crate do.
pub trait WeightedChoose: Measured {
    /// One element, with a chance proportional to its quantity.
    ///
    /// `None` when the collection is empty or nothing in it has a positive
    /// finite quantity. Walks the elements once and allocates nothing.
    fn choose_weighted(&self, random: &mut Random) -> Option<&Self::Item>;

    /// Up to `amount` distinct elements, each drawn in proportion to its
    /// quantity among those not yet taken.
    ///
    /// Fewer than asked for when the collection is smaller, or when too few
    /// elements carry a positive quantity. Never repeats an element, which is
    /// what makes this different from calling
    /// [`choose_weighted`](WeightedChoose::choose_weighted) repeatedly.
    fn choose_multiple_weighted(&self, random: &mut Random, amount: usize) -> Vec<&Self::Item>;

    /// The chance of [`choose_weighted`](WeightedChoose::choose_weighted)
    /// returning this element, as its share of the total.
    ///
    /// Zero for an element the collection does not hold.
    fn weighted_chance_of(&self, item: &Self::Item) -> crate::units::Probability;
}

/// Marker for collections whose ordinary [`Choose`] policy is exactly their
/// [`WeightedChoose`] policy.
///
/// Implemented explicitly rather than through a blanket: implementing both
/// source traits does not prove that their methods use the same policy.
pub trait ChooseByMeasure: WeightedChoose + Choose {}
