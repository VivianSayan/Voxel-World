//! Relative weights for choosing between things, checked once instead of at
//! every use.
//!
//! Weighted selection turns up wherever the world picks one of several
//! outcomes, and each place that took a bare `&[f64]` had to decide again what
//! to do about an empty list, a negative weight, a NaN, or a total of zero.
//! Some of them checked in debug only, one of them normalised and then threw
//! the total away. [`Weights`] answers those once, at construction, so that
//! everything downstream can assume a usable distribution.
//!
//! Weights are *relative*: `[3.0, 1.0]` means three to one. They do not have to
//! sum to anything, and the total is kept so nothing has to add them up again.

use crate::units::scalar::{Probability, UnitValue};
use std::fmt;

/// A non-empty list of finite, non-negative weights with a positive finite
/// total.
///
/// The invariants hold for the whole life of the value: there is no way to
/// build one that is empty, holds a negative or non-finite weight, or sums to
/// zero or infinity. Individual weights may be zero, which is how an outcome is
/// switched off without changing the indices of the others.
///
/// The total is summed once at construction. Every consumer wants it, and
/// summing per draw was the cost this type exists to remove.
#[derive(Clone, PartialEq, Debug)]
pub struct Weights {
    weights: Vec<f64>,
    total: f64,
}

impl Weights {
    /// The weights as given, or `None` if they are not a usable distribution:
    /// an empty list, a negative or non-finite weight, or a total that is zero,
    /// infinite or NaN.
    ///
    /// The constructor for weights that came from configuration or a save,
    /// where a bad one should be reported rather than quietly ignored. The
    /// total is computed here, so the check for overflow is the same work as
    /// the caching.
    pub fn new(weights: impl IntoIterator<Item = f64>) -> Option<Self> {
        let weights: Vec<f64> = weights.into_iter().collect();

        if weights.is_empty() {
            return None;
        }

        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return None;
        }

        let total: f64 = weights.iter().sum();

        if !total.is_finite() || total <= 0.0 {
            return None;
        }

        Some(Self { weights, total })
    }

    /// The weights with the unusable ones zeroed: a negative, NaN or infinite
    /// weight becomes zero rather than rejecting the whole list.
    ///
    /// The constructor for weights that came out of arithmetic, such as the
    /// draws behind [`Dirichlet`](crate::random::Dirichlet), where one outcome
    /// underflowing should drop that outcome rather than the distribution.
    /// Still `None` for an empty list, or when nothing positive is left.
    pub fn sanitised(weights: impl IntoIterator<Item = f64>) -> Option<Self> {
        Self::new(weights.into_iter().map(|weight| {
            if weight.is_finite() && weight > 0.0 {
                weight
            } else {
                0.0
            }
        }))
    }

    /// `count` outcomes, all equally likely: `count` weights of one. `None`
    /// for a count of zero.
    pub fn uniform(count: usize) -> Option<Self> {
        Self::new(std::iter::repeat_n(1.0, count))
    }

    /// The weights themselves, in the order they were given.
    pub fn as_slice(&self) -> &[f64] {
        &self.weights
    }

    /// The weights added together. Positive and finite, and computed once at
    /// construction rather than here.
    pub fn total(&self) -> f64 {
        self.total
    }

    /// How many outcomes there are, counting those weighted zero.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Always false, since an empty list cannot be built. Present because
    /// generic code, and clippy, expect it beside [`Weights::len`].
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// One weight, or `None` past the end.
    pub fn get(&self, index: usize) -> Option<f64> {
        self.weights.get(index).copied()
    }

    /// The weights in order, by value.
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.weights.iter().copied()
    }

    /// The share of the total one outcome takes, which is its probability.
    /// [`Probability::NEVER`] past the end of the list.
    pub fn probability_of(&self, index: usize) -> Probability {
        let Some(weight) = self.get(index) else {
            return Probability::NEVER;
        };

        Probability::clamped(weight / self.total)
    }

    /// Every weight as its share of the total, in order. The shares sum to one
    /// up to rounding, which is why [`Weights::cumulative`] exists for anything
    /// that has to walk them.
    pub fn to_probabilities(&self) -> Vec<Probability> {
        self.weights
            .iter()
            .map(|weight| Probability::clamped(weight / self.total))
            .collect()
    }

    /// Running totals, where entry `i` is the chance of landing anywhere in
    /// `0..=i`: the thresholds a draw in `[0, 1)` is compared against in order.
    ///
    /// Every entry from the last positive weight onwards is forced to
    /// [`Probability::ALWAYS`], so a draw always finds an entry to stop at
    /// however the divisions rounded, and a rounding remainder can never fall
    /// to an outcome weighted zero.
    pub fn cumulative(&self) -> Vec<Probability> {
        let mut cumulative: Vec<Probability> = Vec::with_capacity(self.weights.len());
        let mut running: f64 = 0.0;

        for weight in &self.weights {
            running += weight / self.total;
            cumulative.push(Probability::clamped(running));
        }

        let last_positive = self
            .weights
            .iter()
            .rposition(|weight| *weight > 0.0)
            .unwrap();
        for threshold in &mut cumulative[last_positive..] {
            *threshold = Probability::ALWAYS;
        }

        cumulative
    }

    /// The outcome a draw in `[0, 1)` lands on.
    ///
    /// Scales the draw by the total and walks the weights, subtracting each in
    /// turn, which is inverse-transform sampling without building the
    /// cumulative list. O(n) per draw and allocates nothing. Outcomes weighted
    /// zero are skipped and can never be returned. Should rounding let the
    /// target survive the whole walk, the last outcome that could have been
    /// chosen is returned, which is the only honest answer left.
    ///
    /// For repeated draws from one distribution, build a
    /// [`Categorical`](crate::random::Categorical), which trades a table for
    /// O(1) draws, or an
    /// [`IntegerCategorical`](crate::random::IntegerCategorical) where the
    /// shares must be exact.
    pub fn index_for(&self, unit: UnitValue) -> usize {
        let mut target: f64 = unit.value() * self.total;
        let mut last_positive: usize = 0;

        for (index, &weight) in self.weights.iter().enumerate() {
            if weight > 0.0 {
                if target < weight {
                    return index;
                }

                target -= weight;
                last_positive = index;
            }
        }

        last_positive
    }
}

impl fmt::Display for Weights {
    /// As the weights and their total, such as `[3.0, 1.0] of 4`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?} of {}", self.weights, self.total)
    }
}

impl AsRef<[f64]> for Weights {
    fn as_ref(&self) -> &[f64] {
        &self.weights
    }
}

/// One weight by position. Panics past the end, as slice indexing does; use
/// [`Weights::get`] where that is not wanted.
impl std::ops::Index<usize> for Weights {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.weights[index]
    }
}
