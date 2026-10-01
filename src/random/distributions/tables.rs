//! Picking among a fixed set of outcomes by weight.

use super::continuous::Gamma;
use super::exact::chance_fraction;
use super::{Distribution, PortableDistribution};
use crate::random::source::RandomSource;
use crate::units::{UnitValue, Weights};
use crate::units::Unit;

/// Walks the cumulative weights: O(n) per draw, no setup and no allocation.
/// For many draws from the same weights, build a [`Categorical`] instead.
///
/// Exact.
impl Distribution for Weights {
    type Output = usize;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> usize {
        self.index_for(UnitValue::new(source.unit_f64()).unwrap())
    }
}

impl PortableDistribution for Weights {}

/// Categorical distribution over indices `0..n`. Probabilities are normalized,
/// so any non-negative weights work. O(n) setup, O(1) sampling (alias method).
///
/// Exact in the sense that matters for reproducibility: only arithmetic, so the
/// same seed picks the same index everywhere. The alias table itself is built
/// in floating point, so each index's probability can be off from its weight's
/// share by a rounding error. [`IntegerCategorical`] has none.
#[derive(Clone, Debug)]
pub struct Categorical {
    /// The coin for each column, on the `2^-63` grid. Built in floating point by
    /// Vose's method below, then put on the grid once so that sampling is an
    /// integer comparison rather than a float one.
    probability: Vec<Unit>,
    alias: Vec<usize>,
}

impl Categorical {
    /// Builds the alias table, by Vose's method.
    ///
    /// Each outcome is scaled to its share of an even split, and the ones with
    /// less than a full column are paired with ones that have more, until every
    /// column holds either one outcome or two. A draw then picks a column and
    /// one coin decides between its pair, which is what makes sampling O(1).
    ///
    /// The weights are relative, not probabilities; [`Weights`] has already
    /// checked that they make a usable distribution and holds their total, so
    /// nothing is validated or summed again here. Whatever is left in either
    /// list at the end is a full column up to rounding, which is what the table
    /// is initialised to.
    pub fn new(weights: &Weights) -> Self {
        let count: usize = weights.len();
        let total: f64 = weights.total();

        let mut scaled: Vec<f64> = weights
            .iter()
            .map(|weight| (weight / total) * count as f64)
            .collect();
        let mut probability: Vec<f64> = vec![1.0; count];
        let mut alias: Vec<usize> = (0..count).collect();

        let (mut small, mut large): (Vec<usize>, Vec<usize>) =
            (0..count).partition(|&index| scaled[index] < 1.0);

        while let (Some(&less), Some(&more)) = (small.last(), large.last()) {
            small.pop();
            probability[less] = scaled[less];
            alias[less] = more;

            scaled[more] -= 1.0 - scaled[less];
            if scaled[more] < 1.0 {
                large.pop();
                small.push(more);
            }
        }
        // Vose's method leaves each column full "up to rounding", so a value can
        // land a hair outside `[0, 1]`; `clamped` folds those back in.
        Self {
            probability: probability.into_iter().map(Unit::clamped).collect(),
            alias,
        }
    }

    /// How many outcomes the table holds.
    pub fn len(&self) -> usize {
        self.probability.len()
    }

    /// Whether the table holds no outcomes, which a [`Weights`] cannot
    /// produce.
    pub fn is_empty(&self) -> bool {
        self.probability.is_empty()
    }
}

impl Distribution for Categorical {
    type Output = usize;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> usize {
        let column: usize = source.bounded_u64(self.probability.len() as u64) as usize;
        if self.probability[column].decide_from(source) {
            column
        } else {
            self.alias[column]
        }
    }
}

impl PortableDistribution for Categorical {}

/// Categorical distribution over integer weights, with every probability
/// exactly its weight over the total.
///
/// The alias table is built in integers: each column's threshold is a whole
/// number out of the total, so nothing is rounded, the leftover columns come
/// out at exactly one, and the draw compares against the threshold exactly.
/// The table for block palettes and loot: "stone 70, dirt 25, ore 5" means
/// precisely 70%, 25% and 5%.
///
/// Exact. O(n) setup, O(1) sampling, usually one word per column choice and
/// one per threshold.
#[derive(Clone, Debug)]
pub struct IntegerCategorical {
    /// Out of `total`: the chance of keeping the column rather than its alias.
    threshold: Vec<u128>,
    alias: Vec<usize>,
    total: u128,
}

impl IntegerCategorical {
    /// Builds the alias table in integers, by Vose's method as in
    /// [`Categorical::new`], with `total` standing in for a full column.
    ///
    /// Each weight is scaled by the number of outcomes, so a column holding
    /// exactly its fair share sits at `total`; those products reach at most
    /// `u64::MAX * count`, which is why the table is `u128`. A large column
    /// pays for the gap it fills, which can never underflow, since it held at
    /// least `total` and the gap is at most `total`. The scaled weights always
    /// sum to `count * total`, so whatever is left at the end holds exactly
    /// `total`: a full column, which is what the table is initialised to.
    ///
    /// `None` when there are no weights or they are all zero.
    pub fn new(weights: &[u64]) -> Option<Self> {
        let count: usize = weights.len();
        let total: u128 = weights.iter().map(|&weight| weight as u128).sum();

        if total == 0 {
            return None;
        }

        let mut scaled: Vec<u128> = weights
            .iter()
            .map(|&weight| weight as u128 * count as u128)
            .collect();
        let mut threshold: Vec<u128> = vec![total; count];
        let mut alias: Vec<usize> = (0..count).collect();

        let (mut small, mut large): (Vec<usize>, Vec<usize>) =
            (0..count).partition(|&index| scaled[index] < total);

        while let (Some(&less), Some(&more)) = (small.last(), large.last()) {
            small.pop();
            threshold[less] = scaled[less];
            alias[less] = more;

            scaled[more] -= total - scaled[less];
            if scaled[more] < total {
                large.pop();
                small.push(more);
            }
        }
        Some(Self {
            threshold,
            alias,
            total,
        })
    }

    /// How many outcomes the table holds, counting those weighted zero.
    pub fn len(&self) -> usize {
        self.threshold.len()
    }

    /// Whether the table holds no outcomes, which [`IntegerCategorical::new`]
    /// cannot produce.
    pub fn is_empty(&self) -> bool {
        self.threshold.is_empty()
    }
}

impl Distribution for IntegerCategorical {
    type Output = usize;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> usize {
        let column: usize = source.bounded_u64(self.threshold.len() as u64) as usize;
        if chance_fraction(source, self.threshold[column], self.total) {
            column
        } else {
            self.alias[column]
        }
    }
}

impl PortableDistribution for IntegerCategorical {}

/// Picks one of a set of values in proportion to its weight.
///
/// Not a [`Distribution`], since what it hands back borrows from it; its
/// `sample` takes a [`Random`](crate::random::Random) all the same, and
/// [`Seed::sample_weighted`](crate::random::seed::Seed::sample_weighted) picks
/// from it by seed.
#[derive(Clone, Debug)]
pub struct WeightedDiscrete<T> {
    values: Vec<T>,
    table: Categorical,
}

impl<T> WeightedDiscrete<T> {
    /// `None` when the entries do not make a usable distribution: none at all,
    /// or no positive weight between them.
    pub fn new(entries: impl IntoIterator<Item = (T, f64)>) -> Option<Self> {
        let (values, weights): (Vec<T>, Vec<f64>) = entries.into_iter().unzip();
        let weights: Weights = Weights::new(weights)?;

        Some(Self {
            values,
            table: Categorical::new(&weights),
        })
    }

    /// The values, in the order they were given.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// One draw, as a position in [`WeightedDiscrete::values`], for a caller
    /// that wants the index rather than the value.
    pub fn sample_index<S: RandomSource + ?Sized>(&self, source: &mut S) -> usize {
        self.table.sample(source)
    }

    /// One value, drawn in proportion to its weight.
    pub fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> &T {
        &self.values[self.table.sample(source)]
    }
}

/// A random set of weights, drawn from a Dirichlet distribution.
///
/// Gives a whole distribution rather than one value: useful for deciding, per
/// region, how a handful of outcomes should be balanced there. Every
/// concentration at 1 spreads them evenly over all the possible splits; higher
/// values pull towards equal shares, lower towards one outcome taking almost
/// everything. A concentration that is not positive and finite gives its
/// outcome no share.
///
/// Samples `None` only if every share underflows to zero, which takes very
/// small concentrations.
///
/// Platform-dependent: built on [`Gamma`].
#[derive(Clone, Debug, PartialEq)]
pub struct Dirichlet {
    concentrations: Vec<f64>,
}

impl Dirichlet {
    /// `None` when no concentration is positive and finite, which leaves no
    /// distribution to draw.
    pub fn new(concentrations: &[f64]) -> Option<Self> {
        concentrations
            .iter()
            .any(|concentration| concentration.is_finite() && *concentration > 0.0)
            .then(|| Self {
                concentrations: concentrations.to_vec(),
            })
    }
}

impl Distribution for Dirichlet {
    type Output = Option<Weights>;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> Option<Weights> {
        let drawn: Vec<f64> = self
            .concentrations
            .iter()
            .map(|&concentration| match Gamma::new(concentration, 1.0) {
                Some(gamma) => gamma.sample(source),
                None => 0.0,
            })
            .collect();

        Weights::sanitised(drawn)
    }
}
