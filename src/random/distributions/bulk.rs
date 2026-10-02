//! Asking a whole population at once instead of one thing at a time.
//!
//! Every sampler here answers a question about *many* trials with far less work than
//! the obvious loop, which asks each trial in turn. The saving is not small: giving
//! ten thousand voxels a one-in-a-thousand chance asks ten thousand questions the
//! obvious way and about ten here.
//!
//! # How cost is described
//!
//! Throughout this module, cost is counted in **distribution samples**, not in words
//! taken from the source. The two differ: several of these samplers use rejection, so
//! a single sample consumes a variable number of words with no fixed upper bound —
//! only an expected one. Where a count of words *is* guaranteed, it says so.
//!
//! | question | sampler |
//! |---|---|
//! | How many of N succeed? | [`Binomial`](super::Binomial) |
//! | *Which* of N succeed? | [`SparseSuccesses`] |
//! | How many fall into each category? | [`Multinomial`] |
//! | How many successes when drawing without replacement? | [`Hypergeometric`] |
//! | How many failures before the R-th success? | [`NegativeBinomial`] |
//! | How many succeed when each has its own chance? | [`PoissonBinomial`] |

use super::{
    Binomial, BinomialRatio, Distribution, Gamma, GeometricRatio, Poisson, PortableDistribution,
};
use crate::math::Ratio;
use crate::units::Rate;
use crate::random::source::StochasticSource;
use crate::units::Unit;

// ---------------------------------------------------------------------------
// Which ones succeeded
// ---------------------------------------------------------------------------

/// The indices that succeed, found by jumping between them.
///
/// # Question
///
/// "Which of these N positions succeed at probability P?"
///
/// # Example
///
/// ```
/// use voxel_world::random::Random;
/// use voxel_world::random::distributions::SparseSuccesses;
/// use voxel_world::random::seed::Seed;
/// use voxel_world::units::Unit;
///
/// let mut random = Random::new(Seed::from_integer(7u64));
///
/// // Which of 10,000 voxels get a random tick, at one in a thousand?
/// let ticked: Vec<u64> = random
///     .sample(&SparseSuccesses::new(10_000, Unit::one_in(1_000)))
///     .collect();
///
/// assert!(ticked.iter().all(|index| *index < 10_000));
/// assert!(ticked.windows(2).all(|pair| pair[0] < pair[1]), "ascending");
/// ```
///
/// # Why this is not a loop
///
/// Testing each position costs one draw per position. Instead this asks "how far to
/// the next success?" — which is exactly a [`Geometric`](super::Geometric) gap — and jumps there. The
/// cost is one geometric *sample* per success, not one question per trial, so at one
/// in a thousand over ten thousand positions it is about ten samples rather than ten
/// thousand Bernoulli questions.
///
/// The indices come out in ascending order, which is also the order they are found
/// in, so nothing is sorted afterwards.
///
/// A chance of one degenerates to yielding every index, which is correct but is the
/// one case where the loop would have been just as good.
///
/// # Cost and portability
///
/// One geometric *sample* per success rather than one per position. Each sample is
/// integer-only and consumes a variable number of words — [`GeometricRatio`] counts
/// trials for a common chance and inverts a table for a rare one — so the work is
/// proportional to the number of successes, not a fixed count of draws.
///
/// Bit-portable: nothing here touches `ln`, `exp` or any other function this project
/// does not assume agrees across targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseSuccesses {
    count: u64,
    chance: Unit,
}

impl SparseSuccesses {
    /// The successes among `count` positions, each at `chance`.
    pub const fn new(count: u64, chance: Unit) -> Self {
        Self { count, chance }
    }
}

/// Integer-only throughout, by way of [`GeometricRatio`].
impl PortableDistribution for SparseSuccesses {}

impl Distribution for SparseSuccesses {
    /// Collected rather than borrowed from the source, because a [`Distribution`]
    /// hands back a value rather than something still holding the generator. At one
    /// success in a thousand that is a short vector; at a high chance, prefer
    /// [`Binomial`] for the count or test positions directly.
    type Output = std::vec::IntoIter<u64>;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Self::Output {
        let mut found: Vec<u64> = Vec::new();

        if self.chance.is_zero() || self.count == 0 {
            return found.into_iter();
        }

        // A certain chance has no gaps to jump, and the geometric would return
        // zero for ever.
        if self.chance.is_one() {
            return (0..self.count).collect::<Vec<u64>>().into_iter();
        }

        // Through the *rational* geometric, not the floating one. A `Unit` is `k/2^63`
        // exactly, so `to_ratio` is lossless — where `to_probability` would have
        // narrowed 63 fractional bits to an `f64`'s 53 and dragged in `ln`, which is
        // not portable. The table is built once here rather than per gap.
        let gaps = GeometricRatio::new(self.chance.to_ratio())
            .expect("a Unit is never above one, so its ratio is proper");
        let mut index: u64 = 0;

        loop {
            // The gap counts *failures*, so the success sits one past them.
            let skip: u64 = gaps.sample(source);

            match index.checked_add(skip) {
                Some(next) if next < self.count => {
                    found.push(next);
                    // Past the success, ready for the next gap.
                    match next.checked_add(1) {
                        Some(after) => index = after,
                        None => break,
                    }
                }
                _ => break,
            }
        }

        found.into_iter()
    }
}

// ---------------------------------------------------------------------------
// Splitting a count between categories
// ---------------------------------------------------------------------------

/// How many of `count` independent items land in each category.
///
/// # Question
///
/// "Distribute N items among these categories according to their probabilities. How
/// many land in each?"
///
/// # Example
///
/// ```
/// use voxel_world::random::Random;
/// use voxel_world::random::distributions::Multinomial;
/// use voxel_world::random::seed::Seed;
/// use voxel_world::units::Unit;
///
/// let mut random = Random::new(Seed::from_integer(3u64));
///
/// // Ten thousand tiles: 40% forest, 30% grass, 20% desert, 10% tundra.
/// let shares = [
///     Unit::percent(40).unwrap(),
///     Unit::percent(30).unwrap(),
///     Unit::percent(20).unwrap(),
///     Unit::percent(10).unwrap(),
/// ];
/// let counts = random.sample(&Multinomial::new(10_000, &shares).unwrap());
///
/// assert_eq!(counts.iter().sum::<u64>(), 10_000, "every tile is placed");
/// ```
///
/// # Why not roll each item
///
/// Rolling the items one at a time costs N categorical samples. This costs one
/// binomial sample per *category*, whatever N is — so ten thousand tiles across four
/// categories is `O(categories)` samples rather than `O(items)`.
///
/// Four binomial samples is not four words from the source: [`Binomial`] uses
/// transformed rejection above ten expected successes, so each sample takes a
/// variable number of words. The win is in the number of samples, which is what
/// dominates.
///
/// # How the shares are treated
///
/// The counts **always sum to exactly `count`**, which is the property callers
/// actually depend on: no item is lost or invented.
///
/// Shares are taken as weights and normalised, so they need not sum to one — and
/// they generally cannot, since a chance like a third is not representable. The
/// last category absorbs whatever the earlier ones left, which is what makes the
/// total exact. That also means the last category carries the rounding, so put the
/// one you care least about last if it matters.
///
/// `None` when there are no categories, or when every share is zero and there is
/// nothing to distribute among.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Multinomial {
    count: u64,
    /// The weights, as plain integers.
    ///
    /// This has always been the algorithm's real currency: a [`Unit`] share was only
    /// ever read as `share.to_bits()` and normalised against the others, so nothing is
    /// lost by holding the integers themselves — and [`Multinomial::portable_weights`]
    /// can then take a caller's exact weights without rounding them onto any grid.
    weights: Vec<u128>,
    exact_conditional: bool,
}

impl Multinomial {
    /// `count` items split by these shares, or `None` if the shares are unusable.
    pub fn new(count: u64, shares: &[Unit]) -> Option<Self> {
        if shares.is_empty() {
            return None;
        }

        // Summed in the integer representation, so the check needs no float and
        // cannot overflow: each share is at most `2^63` and there are at most
        // `usize::MAX` of them, held in a `u128`.
        let weights: Vec<u128> = shares
            .iter()
            .map(|share| u128::from(share.to_bits()))
            .collect();

        if weights.iter().sum::<u128>() == 0 {
            return None;
        }

        Some(Self {
            count,
            weights,
            exact_conditional: false,
        })
    }

    /// The same split, drawn so the result is identical on every target.
    ///
    /// # Question
    ///
    /// "I need this split to come out the same for every player. How?"
    ///
    /// # What changes
    ///
    /// The conditional binomial at each step is drawn with
    /// [`BinomialRatio`](super::BinomialRatio) — integer-only, bit-for-bit portable —
    /// instead of [`Binomial`], whose chance is an `f64` and whose rejection sampler
    /// therefore depends on the platform's floating point. Everything else is the
    /// same decomposition.
    ///
    /// # What it costs
    ///
    /// `BinomialRatio` counts set bits in drawn words, so it spends roughly one word
    /// per 64 trials where `Binomial` uses a constant handful. That is cheap for the
    /// counts a loot table or spawn group produces and expensive for a count in the
    /// millions, which is why it is a choice rather than the default.
    ///
    /// # Accuracy
    ///
    /// The conditional chance is `weight / unclaimed`, which can need more than 64
    /// bits in each half. Where it does, both halves are shifted down together until
    /// they fit a [`Ratio`](crate::math::Ratio) — the same value to within the last
    /// bit, and shifted deterministically, so portability is not traded for it.
    pub fn portable(count: u64, shares: &[Unit]) -> Option<Self> {
        let mut split = Self::new(count, shares)?;
        split.exact_conditional = true;

        Some(split)
    }

    /// A portable split over **exact integer weights**, with no grid in between.
    ///
    /// # Question
    ///
    /// "My table has weights `40, 25, 30, 5`. Can the split use those numbers
    /// themselves?"
    ///
    /// # Why this exists beside the [`Unit`] constructors
    ///
    /// A weight is a whole number and the ratios between whole numbers are exact.
    /// Passing them as [`Unit`] shares means first dividing by the total and rounding
    /// the quotient onto the `2^-63` grid — so `1, 2` arrives as a pair of values whose
    /// ratio is *within one part in `2^63`* of a half, rather than a half.
    ///
    /// Nothing in a game would ever see that. It is still a rounding that need not
    /// happen, and removing it makes the probabilities exact by construction instead of
    /// exact to a bound. A positive weight stays strictly positive here because its
    /// conditional chance is the fraction `weight / unclaimed`, which has a positive
    /// numerator by definition.
    ///
    /// # Range
    ///
    /// `None` when there are no weights or all of them are zero. The weights are
    /// summed in a `u128`, so any `u64` table sums without overflowing; where the
    /// running total does exceed a `u64`, the conditional fraction is reduced by a
    /// shared shift before becoming a [`Ratio`], which keeps it deterministic. A table
    /// whose weights fit a `u64` — which [`BulkPickTable`](crate::random::BulkPickTable)
    /// enforces — never reaches that path and is exact throughout.
    pub fn portable_weights(count: u64, weights: &[u64]) -> Option<Self> {
        if weights.is_empty() {
            return None;
        }

        let weights: Vec<u128> = weights.iter().copied().map(u128::from).collect();

        if weights.iter().sum::<u128>() == 0 {
            return None;
        }

        Some(Self {
            count,
            weights,
            exact_conditional: true,
        })
    }

    /// Whether this split draws identically on every target.
    pub const fn is_portable(&self) -> bool {
        self.exact_conditional
    }

    /// How many categories the split has.
    pub fn categories(&self) -> usize {
        self.weights.len()
    }
}

impl Distribution for Multinomial {
    type Output = Vec<u64>;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Vec<u64> {
        let mut counts: Vec<u64> = vec![0; self.weights.len()];
        let mut remaining: u64 = self.count;
        // Tracked as integers so the conditional chance below is exact.
        let mut unclaimed: u128 = self
            .weights
            .iter()
            .copied()
            .sum();

        // Conditional binomials: each category takes its share of what is left, at
        // its share of the weight that is left. The last one is not drawn at all —
        // it gets the remainder, which is what makes the total exact.
        for (index, weight) in self.weights.iter().copied().enumerate() {
            if remaining == 0 {
                break;
            }

            if index + 1 == self.weights.len() {
                counts[index] = remaining;
                break;
            }

            if weight == 0 || unclaimed == 0 {
                continue;
            }

            let taken: u64 = if self.exact_conditional {
                // `weight / unclaimed` as a Ratio, which needs both halves in a u64.
                // Shifting them down together keeps the value to within its last bit
                // and is itself deterministic.
                let shift: u32 = unclaimed
                    .checked_ilog2()
                    .map_or(0, |highest| highest.saturating_sub(63));

                let chance: Ratio = Ratio::new((weight >> shift) as u64, (unclaimed >> shift) as u64)
                    .unwrap_or(Ratio::ONE);

                BinomialRatio::new(remaining, chance)
                    .expect("a conditional share cannot exceed one")
                    .sample(source)
            } else {
                // This category's chance *among what is still unclaimed*.
                let conditional: Unit = Unit::from_bits_clamped(
                    ((weight << 63) / unclaimed).min(u128::from(Unit::STEPS)) as u64,
                );

                Binomial::new(remaining, conditional.to_probability()).sample(source)
            };

            counts[index] = taken;
            remaining -= taken;
            unclaimed -= weight;
        }

        counts
    }
}

// ---------------------------------------------------------------------------
// Drawing without replacement
// ---------------------------------------------------------------------------

/// How many successes are drawn from a finite population without replacement.
///
/// # Question
///
/// "If I draw K items from a population holding S successes and the rest failures,
/// how many successes do I get?"
///
/// # Example
///
/// ```
/// use voxel_world::random::Random;
/// use voxel_world::random::distributions::Hypergeometric;
/// use voxel_world::random::seed::Seed;
///
/// let mut random = Random::new(Seed::from_integer(11u64));
///
/// // A deck of 30 rare and 70 common. Draw ten cards — how many are rare?
/// let rares = random.sample(&Hypergeometric::new(100, 30, 10).unwrap());
///
/// assert!(rares <= 10, "no more than were drawn");
/// assert!(rares <= 30, "no more than exist");
/// ```
///
/// # Why this is not a binomial
///
/// A binomial assumes the odds never change. Here each draw removes a card, so
/// taking a rare one makes the next rare one less likely. Over a large population
/// the difference is slight; over a small one it is the whole story. Drawing ten
/// cards from a deck of twenty is nothing like ten independent coin flips.
///
/// `None` when there are more successes than population.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hypergeometric {
    population: u64,
    successes: u64,
    draws: u64,
    /// Forces the portable walk whatever the size.
    walk_only: bool,
}

impl Hypergeometric {
    /// `draws` taken from a `population` containing `successes` of interest.
    ///
    /// Drawing more than the population takes all of it.
    ///
    /// Switches algorithm by size, so see [`Hypergeometric::walking`] if the draw
    /// has to be reproducible across platforms.
    pub fn new(population: u64, successes: u64, draws: u64) -> Option<Self> {
        if successes > population {
            return None;
        }

        Some(Self {
            population,
            successes,
            draws: draws.min(population),
            walk_only: false,
        })
    }

    /// The same draw, always taken one item at a time.
    ///
    /// # Why this exists
    ///
    /// [`Hypergeometric::new`] is faster but **not bit-portable** for a large draw:
    /// past 32 reduced draws it uses HRUA, which needs `ln` and `ln_gamma`, and this
    /// project does not assume those agree across targets. This form uses nothing but
    /// integer comparisons, so the same seed draws the same hand everywhere — at a
    /// cost of one draw per item taken.
    ///
    /// Use it for anything that has to replay: loot from a finite pool, a world's
    /// starting inventory. Use `new` for anything transient.
    pub fn walking(population: u64, successes: u64, draws: u64) -> Option<Self> {
        let mut deck = Self::new(population, successes, draws)?;
        deck.walk_only = true;

        Some(deck)
    }

    /// The equivalent problem with the shortest loop, and how to undo the swap.
    ///
    /// # The three symmetries
    ///
    /// The hypergeometric has three, and each turns a large loop into a smaller one:
    ///
    /// 1. **Successes and draws are interchangeable.** Drawing `n` from a population
    ///    holding `K` successes gives the same distribution as drawing `K` from one
    ///    holding `n` — both count the size of the overlap between a set of `n` and a
    ///    set of `K`, and that is symmetric. No undo is needed: the *distribution* is
    ///    the same, so the answer needs no translating back.
    /// 2. **Count failures instead.** `X ~ Hyper(N, K, n)` implies
    ///    `n − X ~ Hyper(N, N−K, n)`, so the answer mirrors as `draws − X`.
    /// 3. **Look at what is left behind.** `X ~ Hyper(N, K, n)` implies
    ///    `K − X ~ Hyper(N, K, N−n)`, so the answer mirrors as `successes − X`.
    ///
    /// Applying all three reduces the work to
    ///
    /// ```text
    /// O(min(draws, successes, population − draws, population − successes))
    /// ```
    ///
    /// so taking 500,000 from a million that holds only 30 successes is a 30-step
    /// problem. Only a genuinely balanced one — half drawn, half of it successes —
    /// stays large, and that is what the constant-time path is for.
    ///
    /// Returns `(successes_after_swap, reduced_successes, reduced_draws,
    /// mirror_successes, mirror_draws)`.
    fn reduced(&self) -> (u64, u64, u64, bool, bool) {
        let population: u64 = self.population;
        let mut successes: u64 = self.successes;
        let mut draws: u64 = self.draws;

        // (1) Loop over the smaller of the two interchangeable counts.
        if draws > successes {
            std::mem::swap(&mut successes, &mut draws);
        }

        // Counted after the swap, since the swap changes what "successes" means.
        let failures: u64 = population - successes;

        // (3) then (2): if either side is still the larger half, take the complement
        // and mirror the answer back afterwards.
        let mirror_draws: bool = draws > population - draws;
        let reduced_draws: u64 = if mirror_draws {
            population - draws
        } else {
            draws
        };

        let mirror_successes: bool = successes > failures;
        let reduced_successes: u64 = if mirror_successes { failures } else { successes };

        (
            successes,
            reduced_successes,
            reduced_draws,
            mirror_successes,
            mirror_draws,
        )
    }

    /// Whether this draw is bit-portable across targets.
    ///
    /// True when the walk is forced, and also when the reduced problem is small
    /// enough that the walk would be chosen anyway.
    pub fn is_portable(&self) -> bool {
        self.walk_only || self.reduced().2 <= WALK_LIMIT
    }

    /// How many are drawn.
    pub const fn draws(&self) -> u64 {
        self.draws
    }

    /// How many successes the population holds.
    pub const fn successes(&self) -> u64 {
        self.successes
    }
}

impl Distribution for Hypergeometric {
    type Output = u64;

    /// # Cost
    ///
    /// The obvious loop takes one draw per item *drawn*, which is fine for a hand of
    /// cards and hopeless for taking half a million from a million. The three
    /// symmetries below fix that: the work is
    ///
    /// ```text
    /// O(min(draws, successes, population − draws, population − successes))
    /// ```
    ///
    /// so taking 500,000 from a million that holds only 30 successes costs **30**
    /// draws, not half a million. Only a genuinely balanced problem — half the
    /// population drawn, half of it successes — is expensive, and that is the case
    /// [`Random::hypergeometric_with`](crate::random::Random::hypergeometric_with)
    /// exists for.
    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> u64 {
        let (successes, effective_successes, effective_draws, mirror_successes, mirror_draws) =
            self.reduced();

        // The reduced problem, solved the way the other discrete samplers here do it:
        // walk it while the walk is short, and switch to a constant-time algorithm
        // once it is not. Both are **exact** — this is a change of algorithm, not an
        // approximation, in the same way [`Binomial`] moves from inversion to
        // transformed rejection past ten expected successes.
        //
        // The walk is bit-portable; HRUA is not, because it needs `ln`. That is why
        // `walk_only` exists and why this type is not marked
        // [`PortableDistribution`](super::PortableDistribution).
        let mut found: u64 = if self.walk_only || effective_draws <= WALK_LIMIT {
            walk_without_replacement(
                source,
                self.population,
                effective_successes,
                effective_draws,
            )
        } else {
            hypergeometric_hrua(source, self.population, effective_successes, effective_draws)
        };

        // Undo the mirrors, innermost first.
        if mirror_successes {
            found = effective_draws - found;
        }

        if mirror_draws {
            found = successes - found;
        }

        found
    }
}

// NOTE: `Hypergeometric` deliberately does **not** implement
// `PortableDistribution`. It used to, with the comment "exact integer comparisons
// only" — true of the walk, and false the moment the constant-time path was added:
// HRUA uses `ln` and `ln_gamma`, which this project does not assume are
// bit-identical across targets. `Hypergeometric::walking` is the portable form.

// ---------------------------------------------------------------------------
// Waiting for several successes
// ---------------------------------------------------------------------------

/// How many **failures** occur before the R-th success.
///
/// # Question
///
/// "How many attempts fail before I get R successes?"
///
/// # Example
///
/// ```
/// use voxel_world::random::Random;
/// use voxel_world::random::distributions::NegativeBinomial;
/// use voxel_world::random::seed::Seed;
/// use voxel_world::units::Unit;
///
/// let mut random = Random::new(Seed::from_integer(5u64));
///
/// // Inspecting candidate spots at a one-in-eight success rate: how many are
/// // rejected before three are placed?
/// let rejected = random.sample(&NegativeBinomial::new(3, Unit::one_in(8)));
///
/// // Total positions inspected is the failures plus the three that worked.
/// let inspected = rejected + 3;
/// # let _ = inspected;
/// ```
///
/// # The convention, stated plainly
///
/// This counts **failures only**. The R successes are not included, so the total
/// number of trials is `result + R`. That matches [`Geometric`](super::Geometric), which is this with
/// `R = 1`, and the two agree exactly.
///
/// A chance of zero never succeeds, so it gives `u64::MAX` rather than looping for
/// ever — again matching [`Geometric`](super::Geometric).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegativeBinomial {
    successes: u64,
    chance: Unit,
    /// Forces the portable gap sum whatever `successes` is.
    gaps_only: bool,
}

impl NegativeBinomial {
    /// The failures before `successes` successes, each at `chance`.
    ///
    /// Switches algorithm by size, so see [`NegativeBinomial::by_gaps`] if the draw
    /// has to be reproducible across platforms.
    pub const fn new(successes: u64, chance: Unit) -> Self {
        Self {
            successes,
            chance,
            gaps_only: false,
        }
    }

    /// The same draw, always as a sum of geometric gaps.
    ///
    /// # Why this exists
    ///
    /// [`NegativeBinomial::new`] switches to a Gamma–Poisson mixture past twenty
    /// successes, which is exact *in distribution* but reaches `ln` and `exp` and so
    /// is **not bit-portable**. This form stays with the gap sum, which is
    /// integer-only — at a cost of one geometric sample per success.
    pub const fn by_gaps(successes: u64, chance: Unit) -> Self {
        Self {
            successes,
            chance,
            gaps_only: true,
        }
    }

    /// Whether this draw is bit-portable across targets.
    pub const fn is_portable(&self) -> bool {
        self.gaps_only || self.successes <= GAP_SUM_LIMIT
    }
}

impl Distribution for NegativeBinomial {
    type Output = u64;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> u64 {
        if self.successes == 0 {
            return 0;
        }

        if self.chance.is_zero() {
            return u64::MAX;
        }

        if self.chance.is_one() {
            return 0;
        }

        // Above a handful of successes, summing geometric gaps one at a time is the
        // wrong shape: it costs one draw per success. The Gamma-Poisson mixture is
        // an *identity*, not an approximation —
        //
        //     lambda ~ Gamma(r, (1-p)/p),  X ~ Poisson(lambda)  =>  X ~ NegBin(r, p)
        //
        // — and costs two samples whatever `r` is. The crossover is around
        // twenty, where a Gamma draw plus a Poisson draw stops being more work than
        // the gaps they replace.
        //
        // Both paths already use `ln`, since `Geometric` does, so neither is
        // bit-portable and the switch costs no guarantee that was being kept.
        if !self.gaps_only && self.successes > GAP_SUM_LIMIT {
            let probability: f64 = self.chance.to_f64();
            let scale: f64 = (1.0 - probability) / probability;

            if let Some(shape) = Gamma::new(self.successes as f64, scale) {
                let mean: f64 = shape.sample(source);

                if let Some(rate) = Rate::new(mean) {
                    return Poisson::new(rate).sample(source);
                }

                // A mean of zero means no failures at all, which is the answer.
                return 0;
            }
        }

        // A sum of independent geometric gaps: the wait for the first success, then
        // the wait for the second, and so on. Through the rational sampler, so the
        // chance keeps all 63 of its bits and this path is bit-portable.
        let gap = GeometricRatio::new(self.chance.to_ratio())
            .expect("a Unit is never above one, so its ratio is proper");
        let mut failures: u64 = 0;

        for _ in 0..self.successes {
            failures = failures.saturating_add(gap.sample(source));
        }

        failures
    }
}

// ---------------------------------------------------------------------------
// Everyone with their own chance
// ---------------------------------------------------------------------------

/// How many succeed when every trial has its own probability.
///
/// # Question
///
/// "Each candidate has a different chance. How many succeed altogether?"
///
/// # Example
///
/// ```
/// use voxel_world::random::Random;
/// use voxel_world::random::distributions::PoissonBinomial;
/// use voxel_world::random::seed::Seed;
/// use voxel_world::units::Unit;
///
/// let mut random = Random::new(Seed::from_integer(9u64));
///
/// // Three plots, each with its own soil quality.
/// let chances = [
///     Unit::percent(80).unwrap(),
///     Unit::percent(50).unwrap(),
///     Unit::percent(5).unwrap(),
/// ];
/// let germinated = random.sample(&PoissonBinomial::new(&chances));
///
/// assert!(germinated <= 3);
/// ```
///
/// # Cost, and why there is no clever version
///
/// This is one draw per trial — the same as looping. Unlike a [`Binomial`], there is
/// no shortcut: the trials genuinely differ, so each has to be asked.
///
/// What it does buy is saying the question once, and correctness: the obvious
/// mistake is to average the chances and pass that to a binomial, which gets the
/// mean right and the spread wrong.
///
/// For very large inputs,
/// [`Random::poisson_binomial_with`](crate::random::Random::poisson_binomial_with)
/// can trade that per-trial cost for a skew-corrected normal, under a proven bound.
/// This distribution itself is always exact.
#[derive(Clone, Debug, PartialEq)]
pub struct PoissonBinomial {
    chances: Vec<Unit>,
}

impl PoissonBinomial {
    /// One trial per chance given.
    pub fn new(chances: &[Unit]) -> Self {
        Self {
            chances: chances.to_vec(),
        }
    }

    /// How many trials there are.
    pub fn trials(&self) -> usize {
        self.chances.len()
    }
}

impl Distribution for PoissonBinomial {
    type Output = u64;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> u64 {
        self.chances
            .iter()
            .filter(|chance| chance.decide_from(source))
            .count() as u64
    }
}

/// Each trial is an exact integer comparison, so the whole count is portable.
impl PortableDistribution for PoissonBinomial {}

/// Above how many successes the Gamma–Poisson mixture is worth its setup.
///
/// Below this the gap sum is both cheaper and portable; above it the mixture is a
/// handful of samples whatever `r` is, at the cost of `ln` and `exp`.
const GAP_SUM_LIMIT: u64 = 20;

/// How long a walk is still cheaper than setting up the constant-time algorithm.
///
/// Below this, taking the items one at a time costs a draw each and nothing else.
/// Above it, the log-gamma setup of [`hypergeometric_hrua`] pays for itself, since
/// that needs only a couple of draws however large the problem gets.
const WALK_LIMIT: u64 = 32;

/// The definition, one item at a time: exact, and obvious.
fn walk_without_replacement<S: StochasticSource + ?Sized>(
    source: &mut S,
    population: u64,
    successes: u64,
    draws: u64,
) -> u64 {
    let mut left_in_population: u64 = population;
    let mut left_successes: u64 = successes;
    let mut found: u64 = 0;

    for _ in 0..draws {
        if left_in_population == 0 {
            break;
        }

        // The chance the next one taken is a success, given what remains.
        if source.bounded_u64(left_in_population) < left_successes {
            found += 1;
            left_successes -= 1;
        }

        left_in_population -= 1;
    }

    found
}

/// Stadlober's ratio-of-uniforms sampler (HRUA), exact and constant-time.
///
/// # What it does
///
/// The same idea as the transformed rejection the binomial and Poisson samplers
/// use: cover the distribution with a hat that is cheap to draw from, draw under
/// the hat, and accept when the point falls below the true probability. The
/// acceptance rate does not depend on the size of the problem, so drawing half a
/// million from a million costs the same *expected* couple of iterations as drawing
/// ten from twenty. Expected-constant, not constant: the rejection loop has a high
/// acceptance rate but no hard bound.
///
/// The two squeeze tests before the logarithm are what keep it cheap: most points
/// are accepted or rejected by arithmetic alone, and the expensive `ln` is reached
/// only for the few near the boundary.
///
/// Caller must have reduced the parameters already — this expects `draws` and
/// `successes` each to be no more than half the population, which the symmetry
/// reduction in [`Hypergeometric`] guarantees.
fn hypergeometric_hrua<S: StochasticSource + ?Sized>(
    source: &mut S,
    population: u64,
    successes: u64,
    draws: u64,
) -> u64 {
    /// Stadlober's hat constants.
    const WIDTH: f64 = 1.715_527_769_921_413_5;
    const OFFSET: f64 = 0.898_916_162_058_898_8;

    let total: f64 = population as f64;
    let good: f64 = successes as f64;
    let taken: f64 = draws as f64;

    let share: f64 = good / total;
    let rest: f64 = 1.0 - share;

    // The centre and spread of the hat, from the distribution's own mean and
    // variance. The variance carries the finite-population correction, which is
    // exactly what makes this not a binomial.
    let centre: f64 = taken * share + 0.5;
    let variance: f64 = (total - taken) * taken * share * rest / (total - 1.0);
    let spread: f64 = (variance + 0.5).sqrt();
    let hat: f64 = WIDTH * spread + OFFSET;

    // The mode, where the hat is anchored.
    let mode: f64 = (((taken + 1.0) * (good + 1.0)) / (total + 2.0)).floor();
    let at_mode: f64 = log_ways(mode, good, taken, total);

    // Nothing above this is reachable, and the hat is truncated there.
    let ceiling: f64 = (draws.min(successes) as f64 + 1.0).min(centre + 16.0 * spread);

    loop {
        let height: f64 = source.unit_f64();
        let across: f64 = source.unit_f64();

        if height == 0.0 {
            continue;
        }

        let candidate: f64 = centre + hat * (across - 0.5) / height;

        if candidate < 0.0 || candidate >= ceiling {
            continue;
        }

        let value: f64 = candidate.floor();
        let ratio: f64 = at_mode - log_ways(value, good, taken, total);

        // Squeeze from below: accept without touching a logarithm.
        if height * (4.0 - height) - 3.0 <= ratio {
            return value as u64;
        }

        // Squeeze from above: reject without touching a logarithm.
        if height * (height - ratio) >= 1.0 {
            continue;
        }

        // The exact test, for the few that neither squeeze settled.
        if 2.0 * height.ln() <= ratio {
            return value as u64;
        }
    }
}

/// The log of how many hands give `hits` successes, up to a constant.
///
/// `ln C(K, k) + ln C(N-K, n-k)` with the binomial coefficients written as gamma
/// functions, so it is continuous and usable inside the rejection test.
fn log_ways(hits: f64, successes: f64, draws: f64, population: f64) -> f64 {
    super::discrete::ln_gamma(hits + 1.0)
        + super::discrete::ln_gamma(successes - hits + 1.0)
        + super::discrete::ln_gamma(draws - hits + 1.0)
        + super::discrete::ln_gamma(population - successes - draws + hits + 1.0)
}
