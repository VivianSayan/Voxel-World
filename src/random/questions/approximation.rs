// ---------------------------------------------------------------------------
// Trading exactness for speed, deliberately
// ---------------------------------------------------------------------------

/// The same questions, with control over whether an approximation may be used.
///
/// The plain [`Random::binomial`], [`Random::poisson`] and
/// [`Random::hypergeometric`] are always exact. These take an
/// [`Approximation`] and may take a cheaper path when a *proven* error bound
/// allows it — see [`approximation`] for the theorems.
///
/// The `_with_info` forms also report which algorithm ran, which is how the tests
/// check that the selection rules behave at their boundaries.
impl Random {
    /// How many of `trials` succeed, with an approximation policy.
    ///
    /// # Question
    ///
    /// "Out of N independent attempts at probability P, how many succeed — and may
    /// you approximate to answer faster?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::{Approximation, Random};
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Unit;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Ambient scenery: a percent of drift is invisible.
    /// let pebbles = random.binomial_with(1_000_000, Unit::one_in(500), Approximation::Fast);
    /// assert!(pebbles <= 1_000_000);
    /// ```
    pub fn binomial_with(&mut self, trials: u64, chance: Unit, policy: Approximation) -> u64 {
        self.binomial_with_info(trials, chance, policy).0
    }

    /// As [`Random::binomial_with`], also reporting which algorithm ran.
    ///
    /// # Question
    ///
    /// "...and which algorithm did you actually use?"
    ///
    /// # How the choice is made
    ///
    /// Both candidate errors are computed from the parameters, then offered to the
    /// policy. A Poisson swap is preferred when both are allowed, because its bound
    /// is in total variation — a stronger statement than the normal swap's
    /// Kolmogorov distance, which can be satisfied while a rare tail is still
    /// proportionally far out.
    ///
    /// [`Approximation::Exact`] never takes either, whatever the parameters.
    pub fn binomial_with_info(
        &mut self,
        trials: u64,
        chance: Unit,
        policy: Approximation,
    ) -> (u64, BinomialAlgorithm) {
        let probability: f64 = chance.to_f64();

        // A point mass has no approximation worth making, and the bounds are
        // infinite there anyway.
        if trials == 0 || chance.is_zero() {
            return (0, BinomialAlgorithm::Exact);
        }

        if chance.is_one() {
            return (trials, BinomialAlgorithm::Exact);
        }

        let le_cam: f64 = approximation::poisson_error_for_binomial(trials, probability);

        if policy.permits(le_cam) {
            let mean: f64 = trials as f64 * probability;
            let count: u64 = match Rate::new(mean) {
                Some(rate) => self.poisson(rate).min(trials),
                // A mean of zero: no successes to draw.
                None => 0,
            };

            return (count, BinomialAlgorithm::Poisson);
        }

        let berry_esseen: f64 = approximation::normal_error_for_binomial(trials, probability);

        if policy.permits(berry_esseen) {
            let mean: f64 = trials as f64 * probability;
            let deviation: f64 = (mean * (1.0 - probability)).sqrt();
            // Discrete and bounded, so the continuous draw is rounded and folded
            // back into range rather than trusted at the edges.
            let drawn: f64 = self.normal(mean, deviation).round();
            let count: u64 = drawn.clamp(0.0, trials as f64) as u64;

            return (count, BinomialAlgorithm::Normal);
        }

        (
            self.sample(&Binomial::new(trials, chance.to_probability())),
            BinomialAlgorithm::Exact,
        )
    }

    /// How many events occur at this average rate, with an approximation policy.
    ///
    /// # Question
    ///
    /// "How many events happen when the average is lambda — and may you
    /// approximate?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::{Approximation, Random};
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Rate;
    /// # let mut random = Random::new(Seed::from_integer(2u64));
    /// // How many rocks in this chunk, if the average is 4.2?
    /// let rocks = random.poisson_with(Rate::new(4.2).unwrap(), Approximation::Auto);
    /// # let _ = rocks;
    /// ```
    pub fn poisson_with(&mut self, mean: Rate, policy: Approximation) -> u64 {
        self.poisson_with_info(mean, policy).0
    }

    /// As [`Random::poisson_with`], also reporting which algorithm ran.
    ///
    /// A normal swap needs a large lambda — the bound is `0.4748 / √λ`, so a mean of
    /// four is nowhere near good enough however many times it is drawn. The result
    /// is rounded and floored at zero, since a Poisson count is discrete and
    /// non-negative and a normal is neither.
    ///
    /// # Question
    ///
    /// "...and which algorithm did you use?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::random::Approximation;
    /// # use voxel_world::units::Rate;
    /// let (count, how) = random.poisson_with_info(Rate::new(4.2).unwrap(), Approximation::Auto);
    /// # let _ = (count, how);
    /// ```
    pub fn poisson_with_info(
        &mut self,
        mean: Rate,
        policy: Approximation,
    ) -> (u64, PoissonAlgorithm) {
        let lambda: f64 = mean.value();
        let berry_esseen: f64 = approximation::normal_error_for_poisson(lambda);

        if policy.permits(berry_esseen) {
            let drawn: f64 = self.normal(lambda, lambda.sqrt()).round();

            return (drawn.max(0.0) as u64, PoissonAlgorithm::Normal);
        }

        (self.poisson(mean), PoissonAlgorithm::Exact)
    }

    /// Successes drawn without replacement, with an approximation policy.
    ///
    /// # Question
    ///
    /// "Drawing K from a population of S successes, how many do I get — and may you
    /// ignore that the population shrinks?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::{Approximation, Random};
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(3u64));
    /// // Ten from a million: removal barely moves the odds.
    /// let found = random
    ///     .hypergeometric_with(1_000_000, 300_000, 10, Approximation::Auto)
    ///     .unwrap();
    /// assert!(found <= 10);
    /// ```
    pub fn hypergeometric_with(
        &mut self,
        population: u64,
        successes: u64,
        draws: u64,
        policy: Approximation,
    ) -> Option<u64> {
        Some(
            self.hypergeometric_with_info(population, successes, draws, policy)?
                .0,
        )
    }

    /// As [`Random::hypergeometric_with`], also reporting which algorithm ran.
    ///
    /// The binomial swap ignores that each draw removes an item, so it is allowed
    /// only when the draw is a small enough fraction of the population that the odds
    /// barely move. The familiar "five percent" advice is this bound at about
    /// `0.053`; stating it as a bound means a tighter policy gets a tighter rule
    /// rather than the same folklore.
    ///
    /// # Question
    ///
    /// "...and which algorithm did you use?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::random::Approximation;
    /// let (found, how) = random
    ///     .hypergeometric_with_info(1_000, 300, 10, Approximation::Auto)
    ///     .expect("valid");
    /// # let _ = (found, how);
    /// ```
    pub fn hypergeometric_with_info(
        &mut self,
        population: u64,
        successes: u64,
        draws: u64,
        policy: Approximation,
    ) -> Option<(u64, HypergeometricAlgorithm)> {
        let deck = Hypergeometric::new(population, successes, draws)?;
        let drift: f64 =
            approximation::binomial_odds_drift_for_hypergeometric(population, deck.draws());

        if policy.permits(drift) && population > 0 {
            // The population's success rate, held exactly rather than through a float.
            let rate: Unit = Unit::out_of(successes, population);
            let count: u64 = self
                .sample(&Binomial::new(deck.draws(), rate.to_probability()))
                .min(deck.draws());

            return Some((count, HypergeometricAlgorithm::Binomial));
        }

        Some((self.sample(&deck), HypergeometricAlgorithm::Exact))
    }

    /// How many succeed when every trial has its own chance, with a policy.
    ///
    /// # Question
    ///
    /// "Each candidate has a different chance — how many succeed, and may you
    /// approximate to avoid asking every one?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::{Approximation, Random};
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Unit;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // A hundred thousand plots, each with its own soil quality.
    /// let soils: Vec<Unit> = (0..100_000)
    ///     .map(|n| Unit::out_of(n % 90 + 5, 100))
    ///     .collect();
    ///
    /// let germinated = random.poisson_binomial_with(&soils, Approximation::Auto);
    /// assert!(germinated <= 100_000);
    /// ```
    pub fn poisson_binomial_with(&mut self, chances: &[Unit], policy: Approximation) -> u64 {
        self.poisson_binomial_with_info(chances, policy).0
    }

    /// As [`Random::poisson_binomial_with`], also reporting which algorithm ran.
    ///
    /// # Question
    ///
    /// "...and which algorithm did you use?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::{Approximation, PoissonBinomialAlgorithm, Random};
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Unit;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let few = [Unit::HALF, Unit::one_in(4)];
    /// let (_, how) = random.poisson_binomial_with_info(&few, Approximation::Auto);
    ///
    /// assert_eq!(how, PoissonBinomialAlgorithm::Exact, "two trials prove nothing");
    /// ```
    ///
    /// # The approximation, and what bounds it
    ///
    /// ```text
    /// count = round(μ + σ·z),  clamped to [0, trials]
    /// μ = Σ pᵢ        σ² = Σ pᵢ(1 − pᵢ)
    /// ```
    ///
    /// Allowed under the **Lyapunov** form of Berry–Esseen — see
    /// [`normal_error_for_poisson_binomial`](crate::random::approximation::normal_error_for_poisson_binomial)
    /// — which drops the assumption that the trials share a probability. Its constant
    /// is larger than the identical case's, so this needs *more* evidence to be
    /// allowed, not less.
    ///
    /// Rounding is what makes that bound describe the draw rather than an
    /// idealisation of it, and clamping only moves cumulative probabilities towards
    /// the truth; both arguments are in [`approximation`].
    ///
    /// # No skew correction, deliberately
    ///
    /// A sum of Bernoulli trials with differing chances is generally skewed, and a
    /// Cornish–Fisher refinement would put that asymmetry back. It is not applied,
    /// for two reasons: it is a different distribution from the one Berry–Esseen
    /// bounds, so the guarantee would be lost; and in the regime where the gate
    /// permits the swap at all, `|skewness| ≤ 3.58 ×` the bound, so the correction is
    /// smaller than the error already being accepted.
    pub fn poisson_binomial_with_info(
        &mut self,
        chances: &[Unit],
        policy: Approximation,
    ) -> (u64, PoissonBinomialAlgorithm) {
        let values: Vec<f64> = chances.iter().map(|chance| chance.to_f64()).collect();
        let error: f64 = approximation::normal_error_for_poisson_binomial(&values);

        if let Some((mean, deviation, _skewness)) =
            approximation::poisson_binomial_moments(&values).filter(|_| policy.permits(error))
        {
            // The plain normal, rounded and clamped — no skew correction. Rounding is
            // what gives the continuity correction and keeps the Berry-Esseen bound
            // applicable to the thing actually drawn; a Cornish-Fisher refinement
            // would be a different distribution the theorem says nothing about, and
            // the gate has already forced the skewness below a few times its own
            // tolerance. See `approximation` for both arguments.
            let drawn: f64 = self.normal(mean, deviation).round();
            let count: u64 = drawn.clamp(0.0, chances.len() as f64) as u64;

            return (count, PoissonBinomialAlgorithm::Normal);
        }

        (
            self.sample(&PoissonBinomial::new(chances)),
            PoissonBinomialAlgorithm::Exact,
        )
    }
}
