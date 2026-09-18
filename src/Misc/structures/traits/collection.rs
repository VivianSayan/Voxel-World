//! Traits for element containers, from the most general to the most
//! specific. Generic code can ask for exactly the capabilities it needs,
//! for example `C: CollectionMut + UniqueCollection` for "any set I can
//! add to".

use crate::misc::random::Random;
use crate::misc::structures::sampling;
use std::hash::Hash;

/// A hash-table key. Unlike `Element`, this does not require cloning and is
/// enough for single-index structures such as `Set` and `SingleMap`.
pub trait Key: Eq + Hash {}

impl<T: Eq + Hash> Key for T {}

/// A key that can also be cloned into secondary indices. Large elements are
/// best stored behind an `Rc`/`Arc` or represented by ids.
pub trait Element: Key + Clone {}

impl<T: Key + Clone> Element for T {}

/// Read access shared by every element container.
pub trait Collection {
    /// Type of element stored by the collection.
    type Item;

    /// Number of entries yielded by `elements()`.
    ///
    /// Quantified collections expose their aggregate amount through
    /// `Measured::total_measure`; `len` remains the number of distinct stored
    /// entries so generic collection code has one consistent meaning.
    fn len(&self) -> usize;

    /// Returns `true` when the collection contains no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns whether `item` is present.
    fn contains(&self, item: &Self::Item) -> bool;

    /// The stored elements. Sequences yield repeats in order; measured
    /// collections yield each distinct element once.
    fn elements(&self) -> impl Iterator<Item = &Self::Item>;
}

/// Adding elements.
pub trait CollectionInsert: Collection {
    /// Adds one element. Returns whether the collection changed.
    fn insert(&mut self, item: Self::Item) -> bool;
}

/// Removing elements without requiring the collection to support a generic
/// insertion operation. This matters for collections whose insertion needs
/// additional information, such as a label or priority.
pub trait CollectionRemove: Collection {
    /// Removes one occurrence. Returns whether the collection changed.
    fn remove(&mut self, item: &Self::Item) -> bool;

    /// Removes every element, retaining allocated storage where practical.
    fn clear(&mut self);

    /// Keeps only the elements `keep` accepts; every occurrence of the
    /// others is removed.
    fn retain<F: FnMut(&Self::Item) -> bool>(&mut self, keep: F);
}

/// Collections supporting both generic insertion and removal.
pub trait CollectionMut: CollectionInsert + CollectionRemove {}

impl<C: CollectionInsert + CollectionRemove> CollectionMut for C {}

/// Marker: never holds the same element twice.
pub trait UniqueCollection: Collection {}

/// Set algebra, named after `HashSet`'s methods. Results are new
/// collections. Types implementing this also get the `|`, `&`, `-` and `^`
/// operators on references.
pub trait SetAlgebra: Collection + Sized {
    /// Returns every element present in either collection.
    fn union(&self, other: &Self) -> Self;

    /// Returns every element present in both collections.
    fn intersection(&self, other: &Self) -> Self;

    /// Returns elements present in `self` but absent from `other`.
    fn difference(&self, other: &Self) -> Self;

    /// Returns elements present in exactly one collection.
    fn symmetric_difference(&self, other: &Self) -> Self {
        self.difference(other).union(&other.difference(self))
    }

    /// Returns whether every element of `self` is present in `other`.
    fn is_subset(&self, other: &Self) -> bool;

    /// Returns whether every element of `other` is present in `self`.
    fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Returns whether the collections have no element in common.
    fn is_disjoint(&self, other: &Self) -> bool {
        !self.elements().any(|item| other.contains(item))
    }

    /// Unites `self` with every collection in `others`.
    fn union_all(&self, others: &[&Self]) -> Self
    where
        Self: Clone,
    {
        others
            .iter()
            .fold(self.clone(), |union, other| union.union(other))
    }

    /// Intersects `self` with every collection in `others`.
    fn intersection_all(&self, others: &[&Self]) -> Self
    where
        Self: Clone,
    {
        others.iter().fold(self.clone(), |intersection, other| {
            intersection.intersection(other)
        })
    }
}

/// Random selection, named after `rand`'s `choose` methods. The default
/// picks uniformly among `elements()`; weighted collections pick by weight.
pub trait Choose: Collection {
    /// Uniformly chooses one element using `random`, or returns `None` when
    /// the collection is empty.
    fn choose(&self, random: &mut Random) -> Option<&Self::Item> {
        let mut chosen = None;
        for (seen, item) in self.elements().enumerate() {
            if random.uniform_index(seen + 1) == 0 {
                chosen = Some(item);
            }
        }
        chosen
    }

    /// Up to `amount` distinct entries, in random order.
    fn choose_multiple(&self, random: &mut Random, amount: usize) -> Vec<&Self::Item> {
        let mut chosen = Vec::with_capacity(amount.min(self.len()));
        for (seen, item) in self.elements().enumerate() {
            if seen < amount {
                chosen.push(item);
            } else if amount > 0 {
                let replacement = random.uniform_index(seen + 1);
                if replacement < amount {
                    chosen[replacement] = item;
                }
            }
        }
        sampling::shuffle(&mut chosen, random);
        chosen
    }
}

/// Random removal, available to every mutable `Choose` collection.
pub trait ChooseMut: Choose + CollectionRemove
where
    Self::Item: Clone,
{
    /// Uniformly removes and returns one element using `random`.
    fn take_random(&mut self, random: &mut Random) -> Option<Self::Item> {
        let item = self.choose(random)?.clone();
        self.remove(&item);
        Some(item)
    }

    /// Uniformly removes and returns up to `amount` distinct entries.
    fn take_multiple_random(&mut self, random: &mut Random, amount: usize) -> Vec<Self::Item> {
        let items: Vec<Self::Item> = self
            .choose_multiple(random, amount)
            .into_iter()
            .cloned()
            .collect();
        for item in &items {
            self.remove(item);
        }
        items
    }
}

impl<C: Choose + CollectionRemove> ChooseMut for C where C::Item: Clone {}
