//! When a sampler is allowed to swap an exact algorithm for an approximate one.
//!
//! Some distributions are slow to sample exactly and very close to a simpler one
//! in the right regime. A binomial over a million trials at a chance of one in a
//! thousand is, for practical purposes, a Poisson — and a Poisson is far cheaper.
//! This module decides when that swap is allowed, and says so out loud.
//!
//! # Why this is not a pile of thresholds
//!
//! The usual advice is folklore: "use the normal approximation when `np ≥ 10` and
//! `n(1−p) ≥ 10`". Those numbers are not wrong, but they are not *bounds* either —
//! they do not say how far off the answer will be, so there is nothing to check and
//! nothing to tighten.
//!
//! The two swaps here have real theorems behind them, so the error can be computed
//! rather than assumed:
//!
//! | swap | bound | what it measures |
//! |---|---|---|
//! | binomial → Poisson | **Le Cam**: `n·p²` | total variation |
//! | binomial → normal | **Berry–Esseen**: `0.4748·(p²+q²)/√(npq)` | Kolmogorov distance |
//! | Poisson → normal | Berry–Esseen: `0.4748/√λ` | Kolmogorov distance |
//! | Poisson-binomial → normal | **Lyapunov–Berry–Esseen**: `0.5600·ρ/σ³` | Kolmogorov distance |
//! | hypergeometric → binomial | `draws / (population − draws)` | **not a distance** — see below |
//!
//! # The tolerance does not mean one thing
//!
//! This matters, so it is said plainly rather than buried: **[`Approximation`] does
//! not carry a single universal error measure.** The number a policy tolerates is
//! compared against whichever quantity the particular swap has a theorem for, and
//! those quantities are not interchangeable:
//!
//! - **Total variation** (the Poisson swap) is the strong one: *no* event at all has
//!   its probability shifted by more than the bound.
//! - **Kolmogorov distance** (the normal swaps) is weaker: it constrains the
//!   *cumulative* distributions only. A rare tail can be proportionally far out while
//!   the bound is comfortably satisfied, because the bound is absolute and the tail is
//!   small.
//! - **The hypergeometric → binomial quantity is not a distance between
//!   distributions at all** — see [`binomial_odds_drift_for_hypergeometric`]. It is a bound
//!   on how far the per-draw odds drift, which is the mechanism the approximation
//!   ignores. It is in the same policy because it is the right dial for that swap, not
//!   because it is commensurable with the others.
//!
//! So `AllowError(one in ten thousand)` means "a tenth of a basis point by whichever
//! measure this swap can prove", not "a tenth of a basis point of the same thing
//! everywhere". For anything where a rare outcome carries real weight — a jackpot, a
//! one-in-a-million drop — [`Approximation::Exact`] is the only honest setting, and it
//! is the default.
//!
//! Each bound is a public function — [`poisson_error_for_binomial`],
//! [`normal_error_for_binomial`], [`normal_error_for_poisson`],
//! [`normal_error_for_poisson_binomial`] and
//! [`binomial_odds_drift_for_hypergeometric`] — so a test can check the number that
//! decided rather than only the decision.
//!
//! # Why rounding a normal keeps the bound
//!
//! A normal is continuous and a count is not, so an approximate draw is rounded.
//! That is not a fudge the bound has to survive in spite of — it is what makes the
//! bound apply, and the reasoning is worth writing down.
//!
//! Rounding to nearest sends `x` to `k` exactly when `x ∈ [k − ½, k + ½)`, so the
//! rounded normal's own distribution function at an integer `k` is `Φ(k + ½)`. That
//! is the **continuity-corrected** normal approximation, arrived at for free. And
//! because a count's distribution function is a step that does not move between
//! integers, `F(k) = F(k + ½)`, so
//!
//! ```text
//! |P(rounded ≤ k) − P(count ≤ k)| = |Φ(k + ½) − F(k + ½)| ≤ the Berry–Esseen bound
//! ```
//!
//! — the bound covers the thing actually sampled, not an idealisation of it.
//!
//! Clamping into the valid range only helps: it moves mass from outside the support
//! onto the endpoints, where the true distribution already has all of its own, so
//! every cumulative probability either stays put or moves towards the truth.
//!
//! # What is *not* bounded
//!
//! Only the uncorrected normal. A skew-corrected draw — a Cornish–Fisher or
//! Edgeworth refinement — is a *different* distribution, and Berry–Esseen says
//! nothing about it. This module therefore does not apply one, and the reason it
//! costs nothing to omit is itself provable: for a sum of Bernoulli trials,
//!
//! ```text
//! ρ ≥ σ²/2   (because p² + q² ≥ ½)      |μ₃| ≤ σ²   (because |q − p| ≤ 1)
//! ```
//!
//! so `|skewness| ≤ 1/σ` while the bound is `≥ 0.28/σ`, giving
//!
//! ```text
//! |skewness| ≤ 3.58 × (the bound)
//! ```
//!
//! Any policy tight enough to permit the swap has therefore already forced the
//! skewness below a few times its own tolerance, and the correction it would buy is
//! smaller than the error already accepted. Trading a proven bound for that would be
//! a poor bargain.
//!
//! # Determinism
//!
//! Which algorithm gets chosen depends only on the parameters, never on the draw,
//! so the same parameters always take the same path. The approximations themselves
//! use `ln`, `exp` and `sqrt`, which are platform-dependent — so an approximate
//! sampler is **not** bit-portable across targets, while the exact ones are. That
//! is stated on each sampler that can take an approximate path.

use crate::units::Unit;

/// How much error a sampler may trade for speed.
///
/// # Question
///
/// "Is this sampler allowed to use a faster approximate algorithm, and how wrong is
/// it allowed to be?"
///
/// # Example
///
/// ```
/// use voxel_world::random::{Approximation, Random};
/// use voxel_world::random::seed::Seed;
/// use voxel_world::units::Unit;
///
/// let mut random = Random::new(Seed::from_integer(1u64));
///
/// // A loot drop where the rare outcome matters: never approximate.
/// let rare = random.binomial_with(1_000_000, Unit::one_in(1_000_000), Approximation::Exact);
///
/// // Ambient scenery, where nobody can tell: let it choose.
/// let grass = random.binomial_with(1_000_000, Unit::percent(30).unwrap(), Approximation::Auto);
///
/// // Or state the tolerance outright.
/// let budget = Approximation::AllowError(Unit::one_in(10_000));
/// let count = random.binomial_with(100_000, Unit::percent(2).unwrap(), budget);
/// # let _ = (rare, grass, count);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Approximation {
    /// Never approximate. The sampler uses its exact algorithm whatever the cost.
    ///
    /// The default, and the right choice when rare outcomes matter or when the
    /// result has to be identical on every platform.
    #[default]
    Exact,

    /// Approximate when the proven error is below one in ten thousand.
    ///
    /// A conservative budget: far below anything a player could notice in aggregate
    /// counts, and tight enough that the distribution's shape is preserved.
    Auto,

    /// Approximate when the proven error is below one in a hundred.
    ///
    /// For counts that only set a mood — how many leaves, how many pebbles — where
    /// a percent of drift in the distribution is invisible.
    Fast,

    /// Approximate only when the proven error is below this.
    ///
    /// The bound is computed from the theorems named in the module documentation,
    /// not guessed. A tolerance of zero is the same as [`Approximation::Exact`],
    /// since no approximation has zero proven error.
    AllowError(Unit),
}

impl Approximation {
    /// The error this policy will tolerate, or `None` for [`Approximation::Exact`].
    pub fn tolerance(self) -> Option<Unit> {
        match self {
            Self::Exact => None,
            Self::Auto => Some(Unit::one_in(10_000)),
            Self::Fast => Some(Unit::one_in(100)),
            Self::AllowError(budget) if budget.is_zero() => None,
            Self::AllowError(budget) => Some(budget),
        }
    }

    /// Whether a swap whose proven error is `error` is allowed.
    ///
    /// A non-finite or negative `error` is never allowed, since it is not a proof of
    /// anything.
    pub fn permits(self, error: f64) -> bool {
        let Some(tolerance) = self.tolerance() else {
            return false;
        };

        error.is_finite() && error >= 0.0 && error <= tolerance.to_f64()
    }
}

/// Which algorithm a binomial draw used.
///
/// Exposed for tests and debugging: the ordinary
/// [`Random::binomial`](crate::random::Random::binomial) does not report it, and
/// [`Random::binomial_with_info`](crate::random::Random::binomial_with_info) does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinomialAlgorithm {
    /// The exact sampler. Always correct, and bit-portable.
    Exact,
    /// A Poisson with mean `n·p`, allowed by Le Cam's bound.
    Poisson,
    /// A normal with mean `n·p` and variance `n·p·(1−p)`, rounded and clamped to
    /// `[0, n]`, allowed by the Berry–Esseen bound.
    Normal,
}

/// Which algorithm a Poisson draw used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoissonAlgorithm {
    /// The exact sampler. Always correct.
    Exact,
    /// A normal with mean and variance `λ`, rounded and clamped at zero.
    Normal,
}

/// Which algorithm a Poisson-binomial draw used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoissonBinomialAlgorithm {
    /// The exact sampler: one coin per trial.
    Exact,
    /// A normal with the trials' pooled mean and variance, rounded and clamped to
    /// `[0, trials]`. No skew correction — see the module documentation for why one
    /// would break the bound and buy nothing in the regime that permits the swap.
    Normal,
}

/// Which algorithm a hypergeometric draw used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypergeometricAlgorithm {
    /// The exact sampler, which tracks what is left in the population.
    Exact,
    /// A binomial at the population's success rate, allowed when the draw takes a
    /// small enough fraction that removal barely changes the odds.
    Binomial,
}

// ---------------------------------------------------------------------------
// The bounds themselves
// ---------------------------------------------------------------------------

/// Le Cam's bound on approximating a binomial by a Poisson.
///
/// ```text
/// d_TV(Binomial(n, p), Poisson(np))  ≤  Σ pᵢ²  =  n·p²
/// ```
///
/// # Which total variation
///
/// This matters, because the two conventions differ by a factor of two. Le Cam's
/// theorem is usually stated in the ℓ¹ form,
///
/// ```text
/// Σₖ |P(S = k) − Poisson(λ){k}|  ≤  2 Σ pᵢ²
/// ```
///
/// and total variation here is the **supremum** convention,
/// `d_TV(X, Y) = sup_A |P(X ∈ A) − P(Y ∈ A)|`, which is half the ℓ¹ distance. So the
/// bound returned is `Σ pᵢ²` and not `2 Σ pᵢ²`. Under the ℓ¹ convention every number
/// here would double, which is why the convention is named rather than assumed.
///
/// In total variation, which is the strong statement: no event whatsoever has its
/// probability shifted by more than this. The bound is good exactly where the
/// approximation is famous for working — many trials, tiny chance — and it grows
/// with `p`, which is why a coin flip is never approximated this way however many
/// times it is tossed.
pub fn poisson_error_for_binomial(trials: u64, chance: f64) -> f64 {
    trials as f64 * chance * chance
}

/// The Berry–Esseen bound on approximating a binomial by a normal.
///
/// `sup |F(x) − Φ(x)| ≤ 0.4748 · (p² + q²) / √(n·p·q)`
///
/// The constant is Shevtsova's (2011), the best known for the general case. This is
/// **Kolmogorov distance**: the largest gap between the two cumulative
/// distributions. It is a weaker statement than total variation — a rare tail can
/// still be proportionally far out while the bound is satisfied — which is why the
/// module documentation says what it does about jackpots.
///
/// Infinite when `p` is zero or one, where the binomial is a point mass and has no
/// normal to be approximated by.
pub fn normal_error_for_binomial(trials: u64, chance: f64) -> f64 {
    /// Shevtsova's constant for the classical Berry–Esseen bound.
    const BERRY_ESSEEN: f64 = 0.4748;

    let complement: f64 = 1.0 - chance;
    let variance: f64 = trials as f64 * chance * complement;

    if variance <= 0.0 {
        return f64::INFINITY;
    }

    BERRY_ESSEEN * (chance * chance + complement * complement) / variance.sqrt()
}

/// The Berry–Esseen bound on approximating a Poisson by a normal.
///
/// A Poisson of mean `λ` is a sum of `λ` independent Poissons of mean one, so the
/// same theorem applies and reduces to `0.4748 / √λ`. Kolmogorov distance again,
/// with the same caveat about tails.
pub fn normal_error_for_poisson(mean: f64) -> f64 {
    const BERRY_ESSEEN: f64 = 0.4748;

    if mean <= 0.0 {
        return f64::INFINITY;
    }

    BERRY_ESSEEN / mean.sqrt()
}

/// How far the per-draw odds can drift over a hypergeometric draw.
///
/// ```text
/// draws / (population − draws)
/// ```
///
/// # This is not a distributional distance, and the name says so
///
/// Drawing without replacement changes the odds as the population shrinks, and that
/// shrinking is exactly what the binomial approximation ignores. After `k` of `N`
/// have been taken, at most `k / (N − k)` of what remains has gone, so this bounds
/// the *relative* movement in the success rate across the whole draw.
///
/// It is **not** a total variation or Kolmogorov distance, and it is deliberately
/// named `odds_drift` rather than `error` so it cannot be mistaken for one. It
/// bounds the mechanism rather than the outcome: a small drift means the odds barely
/// moved, which is a good reason to expect the two distributions to be close, but it
/// is not a theorem that says how close.
///
/// It is nonetheless a real quantity rather than a rule of thumb — the familiar
/// "five percent of the population" advice is this number at about `0.053` — and it
/// is the right dial for this swap. See the module documentation on why a single
/// tolerance spanning three different measures is still worth having.
pub fn binomial_odds_drift_for_hypergeometric(population: u64, draws: u64) -> f64 {
    if draws >= population {
        return f64::INFINITY;
    }

    draws as f64 / (population - draws) as f64
}

/// The Berry–Esseen bound for a sum of trials that do **not** share a probability.
///
/// This is the Lyapunov form, which drops the assumption that the trials are
/// identically distributed:
///
/// ```text
/// sup |F(x) − Φ(x)|  ≤  0.5600 · ρ / σ³
///
/// σ² = Σ pᵢ(1 − pᵢ)
/// ρ  = Σ pᵢ(1 − pᵢ)·(pᵢ² + (1 − pᵢ)²)
/// ```
///
/// `ρ` is the summed third absolute central moment: for a single Bernoulli trial,
/// `E|X − p|³` is `p(1−p)³ + (1−p)p³`, which factors to the form above. The constant
/// is Shevtsova's (2010) for the non-identical case, and is necessarily larger than
/// the `0.4748` the identical case allows.
///
/// Kolmogorov distance again, with the same caveat as the other normal bounds: it
/// constrains the cumulative distributions, not the ratio of individual
/// probabilities, so a rare tail can still be proportionally far out.
///
/// Infinite when the variance is zero — every trial certain or impossible — since
/// there is no spread for a normal to approximate.
pub fn normal_error_for_poisson_binomial(chances: &[f64]) -> f64 {
    /// Shevtsova's constant for the Lyapunov (non-identical) case.
    const LYAPUNOV: f64 = 0.5600;

    let mut variance: f64 = 0.0;
    let mut third_moment: f64 = 0.0;

    for chance in chances {
        let complement: f64 = 1.0 - chance;
        let spread: f64 = chance * complement;

        variance += spread;
        // E|X − p|³ for one Bernoulli trial.
        third_moment += spread * (chance * chance + complement * complement);
    }

    if variance <= 0.0 {
        return f64::INFINITY;
    }

    LYAPUNOV * third_moment / variance.powf(1.5)
}

/// The mean, standard deviation and skewness of a sum of differing Bernoulli trials.
///
/// Returned together because the sampler needs all three and they share a pass. The
/// skewness is the standardised third central moment, `Σ p(1−p)(1−2p) / σ³`, which
/// is what the Cornish–Fisher correction in
/// [`Random::poisson_binomial_with`](crate::random::Random::poisson_binomial_with)
/// uses to put the asymmetry back that a plain normal would lose.
///
/// `None` when the variance is zero, where there is nothing to describe.
pub fn poisson_binomial_moments(chances: &[f64]) -> Option<(f64, f64, f64)> {
    let mut mean: f64 = 0.0;
    let mut variance: f64 = 0.0;
    let mut third: f64 = 0.0;

    for chance in chances {
        let complement: f64 = 1.0 - chance;

        mean += chance;
        variance += chance * complement;
        third += chance * complement * (complement - chance);
    }

    if variance <= 0.0 {
        return None;
    }

    let deviation: f64 = variance.sqrt();

    Some((mean, deviation, third / variance.powf(1.5)))
}
