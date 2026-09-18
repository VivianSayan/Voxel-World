//! Globally exclusive, unordered pairs.
//!
//! Unlike `BiMap<T, T>`, the two sides are one domain: an element can belong
//! to at most one pair regardless of which position it was inserted in.

use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::traits::Element;

#[derive(Clone, Debug)]
/// Globally exclusive unordered pairs of values of type `T`.
///
/// Each value has at most one partner. `(a, b)` and `(b, a)` identify the
/// same relationship, and `(a, a)` is a supported self-pair.
pub struct PairMap<T> {
    pairs: Set<Set<T>>,
    partners: FastHashMap<T, T>,
}

impl<T> Default for PairMap<T> {
    fn default() -> Self {
        Self {
            pairs: Set::new(),
            partners: FastHashMap::default(),
        }
    }
}

impl<T: Element> PairMap<T> {
    /// Creates an empty pair map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty map sized for approximately `capacity` pairs.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            pairs: Set::with_capacity(capacity),
            partners: FastHashMap::with_capacity_and_hasher(
                capacity.saturating_mul(2),
                Default::default(),
            ),
        }
    }

    /// Reserves storage for at least `additional_pairs` more pairs.
    pub fn reserve(&mut self, additional_pairs: usize) {
        self.pairs.reserve(additional_pairs);
        self.partners.reserve(additional_pairs.saturating_mul(2));
    }

    /// Releases unused allocation from the pair and partner indices.
    pub fn shrink_to_fit(&mut self) {
        self.pairs.shrink_to_fit();
        self.partners.shrink_to_fit();
    }

    /// Checks that pair storage and partner lookups describe the same
    /// globally exclusive relationships.
    pub fn check_invariants(&self) -> bool {
        self.pairs.iter().all(|pair| {
            if pair.len() == 1 {
                let value = pair.iter().next().unwrap();
                self.partners.get(value) == Some(value)
            } else if pair.len() == 2 {
                let mut values = pair.iter();
                let first = values.next().unwrap();
                let second = values.next().unwrap();
                self.partners.get(first) == Some(second) && self.partners.get(second) == Some(first)
            } else {
                false
            }
        }) && self.partners.keys().all(|value| {
            self.pairs
                .contains(&Set::from([value.clone(), self.partners[value].clone()]))
        })
    }

    /// Returns the number of relationships, counting a self-pair once.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Returns whether no relationships are stored.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Returns whether `value` currently has a partner.
    pub fn contains(&self, value: &T) -> bool {
        self.partners.contains_key(value)
    }

    /// Returns whether `first` and `second` are paired, in either order.
    pub fn contains_pair(&self, first: &T, second: &T) -> bool {
        self.partners.get(first) == Some(second)
    }

    /// Returns the partner of `value`; a self-pair returns `value` itself.
    pub fn partner(&self, value: &T) -> Option<&T> {
        self.partners.get(value)
    }

    /// Every pair as a one- or two-element unordered set. A self-pair has one
    /// member.
    pub fn pairs(&self) -> impl Iterator<Item = &Set<T>> {
        self.pairs.iter()
    }

    /// Pairs two elements, first removing any pairs they currently belong to.
    /// Returns whether the relationship changed.
    pub fn insert(&mut self, first: T, second: T) -> bool {
        if self.contains_pair(&first, &second) {
            return false;
        }
        self.remove(&first);
        self.remove(&second);

        self.pairs
            .insert(Set::from([first.clone(), second.clone()]));
        self.partners.insert(first.clone(), second.clone());
        self.partners.insert(second, first);
        true
    }

    /// Breaks the pair containing `value`, returning its partner.
    pub fn remove(&mut self, value: &T) -> Option<T> {
        let partner = self.partners.remove(value)?;
        if partner != *value {
            self.partners.remove(&partner);
        }
        self.pairs
            .remove(&Set::from([value.clone(), partner.clone()]));
        Some(partner)
    }

    /// Removes every relationship while retaining allocated storage.
    pub fn clear(&mut self) {
        self.pairs.clear();
        self.partners.clear();
    }
}

impl<T: Element> PartialEq for PairMap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.pairs == other.pairs
    }
}

impl<T: Element> Eq for PairMap<T> {}
