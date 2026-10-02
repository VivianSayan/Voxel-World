//! Bulk weighted picking: loot tables, spawn groups, resource bundles.
//!
//! # Three questions, not one
//!
//! Filling a chest is usually written as one loop, and that loop quietly answers three
//! separate questions at once. Keeping them apart is what makes this reusable:
//!
//! ```text
//! Count       how many weighted selections happen at all?
//! Selection   which entries receive those selections?
//! Quantity    how much does each selected entry contribute?
//! ```
//!
//! Each is a different statistical object and each can be chosen independently. A
//! table that produces "usually about three things, and the coins among them come in
//! heaps of about eight" is a Poisson count with a Poisson quantity, and neither has
//! anything to say about the other.
//!
//! # Nothing happened, against something happened and it was nothing
//!
//! These are different, and the design keeps them different:
//!
//! ```text
//! count = 0            no selection happened. The chest was empty.
//! a Dud entry chosen   a selection happened, and what it produced was nothing.
//! ```
//!
//! The first needs no special entry in the table — it falls out of the count model, so
//! there is no need for a fake `"Empty"` weighted row to make an empty result
//! possible. The second is an ordinary weighted outcome whose value happens to mean
//! nothing, which is the caller's business:
//!
//! ```ignore
//! enum Drop {
//!     Item(Item),
//!     Dud,
//! }
//! ```
//!
//! # The pipeline
//!
//! ```text
//! CountDistribution
//!         ↓  N selections
//! Multinomial over the entry weights
//!         ↓  selections per entry
//! QuantityDistribution
//!         ↓  total quantity per selected entry
//! Vec<BulkPickResult<T>>, in entry order
//! ```
//!
//! # Portability
//!
//! Every count and quantity model here is built on the crate's **exact** integer
//! distributions — [`UniformU64`], [`BinomialRatio`], [`PoissonRatio`],
//! [`GeometricRatio`], [`NegativeBinomial::by_gaps`] — and the selection stage uses
//! [`Multinomial::portable`]. No floating point takes part, so the same seed and the
//! same table give the same result on every target.
//!
//! The faster `f64` samplers ([`Binomial`](super::Binomial),
//! [`Poisson`](super::Poisson)) remain available directly through
//! [`distributions`](super::distributions) for code that wants speed over replay.
//!
//! # What consumes randomness
//!
//! Deliberately sparse, so a table costs what it looks like it costs:
//!
//! | | draws |
//! |---|---|
//! | [`Count::Fixed`] | none |
//! | count of zero | none after the count; no selection, no quantity |
//! | an entry selected zero times | none for that entry |
//! | [`Quantity::Fixed`] | none, it is a multiplication |
//! | [`Quantity::Binomial`] over `n` selections | one draw, not `n` |
//! | [`Quantity::Poisson`] over `n` selections | one draw, not `n` |
//! | [`Quantity::Geometric`] over `n` selections | one draw, not `n` |

use super::distributions::{
    BinomialRatio, Distribution, GeometricRatio, Multinomial, NegativeBinomial, PoissonRatio,
    UniformU64,
};
use crate::math::{Fixed, Ratio};
use crate::random::source::StochasticSource;
use crate::units::Unit;
use std::fmt;

/// How many separate uniform draws a repeated quantity model will make before giving
/// up and asking the caller for a model that aggregates.
///
/// [`Quantity::Uniform`] has no closed form for the sum of `n` draws, so it is the one
/// model that must actually repeat. The cap keeps a large selection count from turning
/// into an unbounded loop; a table that reaches it wants
/// [`Quantity::Binomial`] or [`Quantity::Poisson`], which aggregate exactly.
pub const REPEATED_DRAW_LIMIT: u64 = 1 << 16;

// ---------------------------------------------------------------------------
// What can go wrong
// ---------------------------------------------------------------------------

/// Why a bulk pick could not be answered.
///
/// An empty result is **not** one of these: a count of zero is an ordinary success.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BulkError {
    /// The table has no entries, and the count model asked for at least one selection.
    EmptyTable,
    /// Every entry has zero weight, and the count model asked for at least one
    /// selection. There is nothing that *can* be chosen, which is a mistake in the
    /// table rather than an unlucky draw.
    AllWeightsZero,
    /// The weights sum past what can be held, so their ratios cannot be formed.
    WeightOverflow,
    /// A count model's parameters do not describe a distribution — a uniform range
    /// whose low bound is above its high one, or a negative mean.
    InvalidCount,
    /// A quantity model's parameters do not describe a distribution.
    InvalidQuantity,
    /// The quantity for one entry, or the parameters needed to draw it, ran past what
    /// a `u64` can hold.
    QuantityOverflow,
    /// A repeated quantity model was asked for more draws than
    /// [`REPEATED_DRAW_LIMIT`].
    TooManyRepeats,
}

impl fmt::Display for BulkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyTable => "the table has no entries to select from",
            Self::AllWeightsZero => "every entry has zero weight, so nothing can be selected",
            Self::WeightOverflow => "the entry weights sum past what can be held",
            Self::InvalidCount => "the count model's parameters do not describe a distribution",
            Self::InvalidQuantity => {
                "the quantity model's parameters do not describe a distribution"
            }
            Self::QuantityOverflow => "an entry's quantity ran past what a u64 can hold",
            Self::TooManyRepeats => {
                "a repeated quantity model would need more draws than the limit allows"
            }
        })
    }
}

impl std::error::Error for BulkError {}

// ---------------------------------------------------------------------------
// Stage one: how many selections
// ---------------------------------------------------------------------------

/// How many weighted selections a bulk pick makes.
///
/// # Question
///
/// "How many things come out of this chest at all?"
///
/// This is the stage that lets a result be empty without the table needing a fake
/// entry for it. Implement it for a model of your own; the picker asks nothing else.
pub trait CountDistribution {
    /// The number of weighted selections to make.
    fn sample_count<S: StochasticSource + ?Sized>(&self, source: &mut S)
    -> Result<u64, BulkError>;
}

/// The count models.
///
/// # Example
///
/// ```
/// use voxel_world::fixed;
/// use voxel_world::random::bulk_pick::Count;
/// use voxel_world::unit;
///
/// // Always five.
/// let exact = Count::Fixed(5);
///
/// // Nought to eight, flat.
/// let spread = Count::Uniform { low: 0, high: 8 };
///
/// // Ten chances, each independently taken four times in ten.
/// let chances = Count::Binomial { trials: 10, chance: unit!(0.4) };
///
/// // Usually about three, sometimes none, rarely many.
/// let heap = Count::Poisson { mean: fixed!(3.0) };
///
/// // Keep going until it stops.
/// let run = Count::Geometric { chance: unit!(0.3) };
/// # let _ = (exact, spread, chances, heap, run);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Count {
    /// Exactly this many. Costs no draw at all.
    Fixed(u64),
    /// Uniformly between `low` and `high`, both included.
    ///
    /// `low = 0` is what makes an empty result possible here.
    Uniform {
        /// The fewest selections, included.
        low: u64,
        /// The most selections, included.
        high: u64,
    },
    /// `trials` independent opportunities, each taken with chance `chance`.
    ///
    /// The model for "there can be at most ten things, and each one separately either
    /// exists or does not".
    Binomial {
        /// How many opportunities there are.
        trials: u64,
        /// The chance each one is taken.
        chance: Unit,
    },
    /// A count scattered about `mean`, with no upper bound.
    ///
    /// Usually what is wanted for loot, spawns and ore: sometimes nothing, usually
    /// near the mean, occasionally a windfall.
    Poisson {
        /// The average number of selections.
        mean: Fixed,
    },
    /// How many failures before the first success at `chance`.
    ///
    /// "Usually a few, sometimes more, rarely many." Counts failures only, which is
    /// the crate's convention throughout — see [`Geometric`](super::Geometric).
    Geometric {
        /// The stopping chance. A larger chance means a shorter run.
        chance: Unit,
    },
}

impl CountDistribution for Count {
    fn sample_count<S: StochasticSource + ?Sized>(
        &self,
        source: &mut S,
    ) -> Result<u64, BulkError> {
        match *self {
            // No draw: the answer was known before the source was consulted.
            Self::Fixed(count) => Ok(count),

            Self::Uniform { low, high } => Ok(UniformU64::new(low, high)
                .ok_or(BulkError::InvalidCount)?
                .sample(source)),

            Self::Binomial { trials, chance } => Ok(BinomialRatio::new(trials, chance.to_ratio())
                .ok_or(BulkError::InvalidCount)?
                .sample(source)),

            Self::Poisson { mean } => {
                Ok(PoissonRatio::new(rate_of(mean).ok_or(BulkError::InvalidCount)?).sample(source))
            }

            Self::Geometric { chance } => Ok(GeometricRatio::new(chance.to_ratio())
                .ok_or(BulkError::InvalidCount)?
                .sample(source)),
        }
    }
}

/// A non-negative [`Fixed`] as the exact [`Ratio`] a rational sampler takes.
///
/// `Fixed` counts in units of `2^-FRACTION_BITS`, so the raw integer over that scale
/// *is* the value — no rounding enters, which is why the Poisson mean stays exact.
fn rate_of(mean: Fixed) -> Option<Ratio> {
    let raw: i128 = mean.to_bits();

    if raw < 0 {
        return None;
    }

    Ratio::new(u64::try_from(raw).ok()?, 1u64 << Fixed::FRACTION_BITS)
}

// ---------------------------------------------------------------------------
// Stage three: how much each selection yields
// ---------------------------------------------------------------------------

/// How much one entry produces, given how many times it was selected.
///
/// # Question
///
/// "This entry came up four times. How many arrows is that?"
///
/// # Why the model is handed the count rather than asked repeatedly
///
/// Because most of them can answer in one draw. Four independent `Poisson(λ)` draws
/// sum to one `Poisson(4λ)`; four `Binomial(t, p)` draws to one `Binomial(4t, p)`;
/// four `Fixed(k)` to a multiplication. Only the model knows that, so only the model
/// should decide — the picker says how many selections there were and stays out of it.
///
/// This mirrors [`Shuffle`](crate::structures::traits::Shuffle): the caller provides
/// randomness, the implementor owns the mathematics.
pub trait QuantityDistribution {
    /// The total produced by `selections` selections of this entry.
    ///
    /// Never called with `selections == 0`; an unselected entry costs no draw.
    fn sample_total<S: StochasticSource + ?Sized>(
        &self,
        selections: u64,
        source: &mut S,
    ) -> Result<u64, BulkError>;
}

/// The quantity models.
///
/// # Example
///
/// ```
/// use voxel_world::fixed;
/// use voxel_world::random::bulk_pick::Quantity;
/// use voxel_world::unit;
///
/// let one_each = Quantity::Fixed(1);
/// let arrows = Quantity::Uniform { low: 4, high: 12 };
/// let scattered = Quantity::Binomial { trials: 8, chance: unit!(0.5) };
/// let coins = Quantity::Poisson { mean: fixed!(8.0) };
/// let streak = Quantity::Geometric { chance: unit!(0.4) };
/// # let _ = (one_each, arrows, scattered, coins, streak);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quantity {
    /// Exactly this much per selection. Aggregates by multiplication, with no draw.
    Fixed(u64),
    /// Uniformly between `low` and `high` per selection, both included.
    ///
    /// The one model with no closed form for a sum, so it really does draw once per
    /// selection, up to [`REPEATED_DRAW_LIMIT`].
    Uniform {
        /// The least one selection can produce, included.
        low: u64,
        /// The most one selection can produce, included.
        high: u64,
    },
    /// `trials` chances at one unit each per selection, at chance `chance`.
    ///
    /// Aggregates exactly: `n` draws of `Binomial(t, p)` are one `Binomial(n·t, p)`.
    Binomial {
        /// Chances per selection.
        trials: u64,
        /// The chance each one yields a unit.
        chance: Unit,
    },
    /// An amount scattered about `mean` per selection.
    ///
    /// Aggregates exactly: `n` draws of `Poisson(λ)` are one `Poisson(n·λ)`.
    Poisson {
        /// The average amount one selection produces.
        mean: Fixed,
    },
    /// How many units before stopping, per selection, at chance `chance`.
    ///
    /// Aggregates exactly through [`NegativeBinomial`], which the crate documents as
    /// this model with `R = 1` — so `n` geometric draws are one negative binomial with
    /// `R = n`, and the distribution is unchanged rather than approximated.
    Geometric {
        /// The stopping chance.
        chance: Unit,
    },
}

impl QuantityDistribution for Quantity {
    fn sample_total<S: StochasticSource + ?Sized>(
        &self,
        selections: u64,
        source: &mut S,
    ) -> Result<u64, BulkError> {
        match *self {
            // A multiplication, not a draw.
            Self::Fixed(each) => selections
                .checked_mul(each)
                .ok_or(BulkError::QuantityOverflow),

            Self::Uniform { low, high } => {
                if selections > REPEATED_DRAW_LIMIT {
                    return Err(BulkError::TooManyRepeats);
                }

                let spread: UniformU64 =
                    UniformU64::new(low, high).ok_or(BulkError::InvalidQuantity)?;
                let mut total: u64 = 0;

                for _ in 0..selections {
                    total = total
                        .checked_add(spread.sample(source))
                        .ok_or(BulkError::QuantityOverflow)?;
                }

                Ok(total)
            }

            // One draw: the sum of n binomials with the same chance is a binomial.
            Self::Binomial { trials, chance } => {
                let combined: u64 = trials
                    .checked_mul(selections)
                    .ok_or(BulkError::QuantityOverflow)?;

                Ok(BinomialRatio::new(combined, chance.to_ratio())
                    .ok_or(BulkError::InvalidQuantity)?
                    .sample(source))
            }

            // One draw: the sum of n Poissons is a Poisson of the summed mean.
            Self::Poisson { mean } => {
                let rate: Ratio = rate_of(mean).ok_or(BulkError::InvalidQuantity)?;

                // `n` lots of `a/b` is `n*a/b`, so only the numerator grows — and it
                // is checked, because a large count times a large mean leaves `u64`.
                let combined: Ratio = Ratio::new(
                    rate.numerator()
                        .checked_mul(selections)
                        .ok_or(BulkError::QuantityOverflow)?,
                    rate.denominator(),
                )
                .ok_or(BulkError::QuantityOverflow)?;

                Ok(PoissonRatio::new(combined).sample(source))
            }

            // One draw: n geometrics are a negative binomial with R = n.
            Self::Geometric { chance } => {
                Ok(NegativeBinomial::by_gaps(selections, chance).sample(source))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

/// One weighted row of a bulk table: what it is, how likely, and how much.
///
/// Nothing here is loot-specific. `T` is whatever the table produces — an item, a
/// creature, an ore, a decoration, a crafting output, a member of a population.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BulkPickEntry<T, Q> {
    /// What this row produces.
    pub value: T,
    /// How likely this row is, relative to the others.
    ///
    /// Only the ratios matter: `1, 2, 3` and `10, 20, 30` are the same table. Zero
    /// means the row can never be selected, which is a useful way to disable one
    /// without removing it.
    pub weight: u64,
    /// How much this row produces per selection.
    pub quantity: Q,
}

impl<T, Q> BulkPickEntry<T, Q> {
    /// A row.
    pub const fn new(value: T, weight: u64, quantity: Q) -> Self {
        Self {
            value,
            weight,
            quantity,
        }
    }
}

/// A weighted table with its own count model.
///
/// # Why the count belongs to the table
///
/// Because it is part of what the table *is*. A chest that yields about three things
/// and a chest that yields exactly ten are different tables, not the same table called
/// with different arguments — and a caller who has to remember to pass the count can
/// pass the wrong one. So the call reads
///
/// ```text
/// seed.bulk_pick(&loot)
/// ```
///
/// rather than `seed.bulk_pick(&entries, 10)`.
///
/// # Example
///
/// ```
/// use voxel_world::fixed;
/// use voxel_world::random::bulk_pick::{BulkPickEntry, BulkPickTable, Count, Quantity};
/// use voxel_world::random::seed::Seed;
/// use voxel_world::unit;
///
/// #[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// enum Item {
///     Iron,
///     Arrows,
///     Coins,
/// }
///
/// let loot = BulkPickTable {
///     count: Count::Poisson { mean: fixed!(3.0) },
///     entries: vec![
///         BulkPickEntry::new(Item::Iron, 40, Quantity::Binomial { trials: 8, chance: unit!(0.5) }),
///         BulkPickEntry::new(Item::Arrows, 25, Quantity::Uniform { low: 4, high: 12 }),
///         BulkPickEntry::new(Item::Coins, 30, Quantity::Poisson { mean: fixed!(8.0) }),
///     ],
/// };
///
/// let chest = Seed::from_integer(1u64).child("loot").index(42);
/// let contents = chest.bulk_pick(&loot).expect("a valid table");
///
/// // The same chest always holds the same things.
/// assert_eq!(contents, chest.bulk_pick(&loot).unwrap());
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BulkPickTable<T, C, Q> {
    /// How many weighted selections one pick makes.
    pub count: C,
    /// The rows, in the order results will be reported.
    pub entries: Vec<BulkPickEntry<T, Q>>,
}

impl<T, Q> BulkPickTable<T, Count, Q> {
    /// A table that always makes exactly `count` selections.
    ///
    /// The common simple case, spelled without naming the count model.
    pub fn fixed_count(count: u64, entries: Vec<BulkPickEntry<T, Q>>) -> Self {
        Self {
            count: Count::Fixed(count),
            entries,
        }
    }
}

impl<T, C, Q> BulkPickTable<T, C, Q> {
    /// A table with any count model.
    pub const fn new(count: C, entries: Vec<BulkPickEntry<T, Q>>) -> Self {
        Self { count, entries }
    }

    /// The sum of every weight, or [`BulkError::WeightOverflow`] if it will not fit.
    ///
    /// Summed in a `u128` so the check itself cannot overflow.
    pub fn total_weight(&self) -> Result<u64, BulkError> {
        let total: u128 = self
            .entries
            .iter()
            .map(|entry| u128::from(entry.weight))
            .sum();

        u64::try_from(total).map_err(|_| BulkError::WeightOverflow)
    }
}

/// What one entry produced.
///
/// Only entries actually selected appear, in the table's own entry order — never in
/// the order a hash map happened to walk.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BulkPickResult<T> {
    /// The entry's value.
    pub value: T,
    /// How many times it was selected. Always at least one.
    pub selections: u64,
    /// How much it produced across all of those selections.
    ///
    /// May be zero even though it was selected: a `Binomial` or `Poisson` quantity can
    /// legitimately come out empty, which is "something happened and it yielded
    /// nothing" rather than "nothing happened".
    pub quantity: u64,
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// Draws a whole bulk pick from any stochastic source.
///
/// # Question
///
/// "What does this table produce, this time?"
///
/// The one implementation. [`Random::bulk_pick`](crate::random::Random::bulk_pick) and
/// [`Seed::bulk_pick`](crate::random::seed::Seed::bulk_pick) both come here, so a
/// stream and a seed cannot drift apart on what a table means.
///
/// # The three stages
///
/// 1. The count model says how many selections happen. Zero returns an empty result
///    immediately, having touched nothing else.
/// 2. [`Multinomial::portable`] splits those selections across the weights, giving a
///    count per entry directly rather than a sequence to be tallied afterwards.
/// 3. Each selected entry's quantity model turns its selection count into a total.
///
/// # Errors
///
/// Only for a table that cannot be drawn from — see [`BulkError`]. An empty result is
/// a success.
pub fn bulk_pick<T, C, Q, S>(
    table: &BulkPickTable<T, C, Q>,
    source: &mut S,
) -> Result<Vec<BulkPickResult<T>>, BulkError>
where
    T: Clone,
    C: CountDistribution,
    Q: QuantityDistribution,
    S: StochasticSource + ?Sized,
{
    let count: u64 = table.count.sample_count(source)?;

    // Stage one answered zero. That is an ordinary empty chest, and nothing further
    // is drawn — not the selection, not any quantity.
    if count == 0 {
        return Ok(Vec::new());
    }

    if table.entries.is_empty() {
        return Err(BulkError::EmptyTable);
    }

    // Checked for its own sake: the weights must be summable for the table to mean
    // anything, and saying so here is clearer than letting the split fail later.
    if table.total_weight()? == 0 {
        return Err(BulkError::AllWeightsZero);
    }

    // The caller's weights, unchanged. `Multinomial::portable_weights` works in the
    // integers themselves, so the conditional chance at each step is the exact
    // fraction `weight / unclaimed` — no grid, and no rounding, between the number
    // written in the table and the draw it governs.
    let weights: Vec<u64> = table.entries.iter().map(|entry| entry.weight).collect();

    let selections: Vec<u64> = Multinomial::portable_weights(count, &weights)
        .ok_or(BulkError::AllWeightsZero)?
        .sample(source);

    let mut results: Vec<BulkPickResult<T>> = Vec::new();

    for (entry, selected) in table.entries.iter().zip(selections) {
        if selected == 0 {
            // No draw for an entry that did not come up.
            continue;
        }

        results.push(BulkPickResult {
            value: entry.value.clone(),
            selections: selected,
            quantity: entry.quantity.sample_total(selected, source)?,
        });
    }

    Ok(results)
}
