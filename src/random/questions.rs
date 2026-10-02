//! What [`Random`] can be asked.
//!
//! Every method here is a shorthand: it builds the distribution of the same name and
//! samples it once, panicking on parameters its `new` would refuse. The algorithm,
//! and whether it is exact or platform-dependent, is documented on the distribution.
//! For many draws from one distribution, build it once and use
//! [`Random::sample`].
//!
//! Split from the generator itself because the two change for different reasons: the
//! generator is a fixed algorithm that should almost never move, while this grows
//! every time there is a new question worth asking.

use crate::math::Fixed;
use crate::random::bulk_pick::{
    BulkError, BulkPickResult, BulkPickTable, CountDistribution, QuantityDistribution, bulk_pick,
};
use crate::structures::traits::Shuffle;
use crate::random::fixed_point::UniformFixedRange;
use crate::math::linear::{Vector2, Vector3, Vector4};
use crate::random::approximation::{
    Approximation, BinomialAlgorithm, HypergeometricAlgorithm, PoissonAlgorithm,
    PoissonBinomialAlgorithm,
};
use crate::random::distributions::*;
use crate::random::source::StochasticSource;
use crate::random::{Random, approximation, distributions};
use crate::units::{Probability, Rate, Ratio, Unit, Weights};

impl Random {
    /// A uniformly random [`Fixed`] in `[low, high)`.
    ///
    /// # Question
    ///
    /// "What is a random value between these two bounds, deterministically?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::math::Fixed;
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let height = random.fixed_range(Fixed::from_integer(60), Fixed::from_integer(80));
    ///
    /// assert!(height >= Fixed::from_integer(60) && height < Fixed::from_integer(80));
    /// ```
    ///
    /// # Panics
    ///
    /// If the range is empty or inverted, as the other shorthands here do for
    /// parameters their distribution would refuse. Use
    /// [`UniformFixedRange::new`](crate::random::UniformFixedRange::new) where the
    /// bounds come from somewhere that might produce an empty one.
    ///
    /// # Why not `low + self.fixed() * (high - low)`
    ///
    /// Because that is not uniform. Scaling a `[0, 1)` draw maps several fractions
    /// onto the same step and misses others, so some values become likelier than their
    /// neighbours. This draws over the steps between the bounds instead, so every
    /// representable value in the range is equally likely.
    pub fn fixed_range(&mut self, low: Fixed, high: Fixed) -> Fixed {
        let range = UniformFixedRange::new(low, high)
            .expect("a fixed-point range must hold at least one value");

        self.sample(&range)
    }

    /// One draw from any distribution.
    ///
    /// # Question
    ///
    /// "How do I draw from a distribution I have already built?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::random::distributions::Bernoulli;
    /// # use voxel_world::units::Probability;
    /// let coin = Bernoulli::new(Probability::new(0.5).unwrap());
    /// let heads = random.sample(&coin);
    /// // Worth building once and reusing when the parameters do not change.
    /// # let _ = heads;
    /// ```
    pub fn sample<D: Distribution>(&mut self, distribution: &D) -> D::Output {
        distribution.sample(self)
    }

    /// Uniform in `[low, high)`. See [`Uniform`].
    ///
    /// # Question
    ///
    /// "What is a random number between these two bounds?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // A tree somewhere between 4 and 9 metres tall.
    /// let height = random.uniform_f64(4.0, 9.0);
    /// assert!((4.0..9.0).contains(&height));
    /// ```
    pub fn uniform_f64(&mut self, low: f64, high: f64) -> f64 {
        self.sample(&Uniform::new(low, high).expect("invalid uniform_f64 parameters"))
    }

    /// Uniform in `[low, high)`. See [`Uniform`].
    ///
    /// # Question
    ///
    /// "The same, in single precision?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let height = random.uniform_f32(4.0, 9.0);
    /// assert!((4.0..9.0).contains(&height));
    /// ```
    pub fn uniform_f32(&mut self, low: f32, high: f32) -> f32 {
        let value = self.uniform_f64(low as f64, high as f64) as f32;
        if low < high {
            value.min(high.next_down()).max(low)
        } else {
            low
        }
    }

    /// Mean 0, standard deviation 1. See [`Normal`].
    ///
    /// # Question
    ///
    /// "What is a value from the standard bell curve?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let z = random.standard_normal();
    /// // Scale and shift it yourself, or use `normal` directly.
    /// let damage = 50.0 + z * 8.0;
    /// # let _ = damage;
    /// ```
    pub fn standard_normal(&mut self) -> f64 {
        distributions::standard_normal(self)
    }

    /// A bell curve around `mean`, spread by `std_dev`.
    /// # Question
    ///
    /// "What is a value that clusters around an average, with occasional outliers?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Enemy health: usually 100, sometimes a bit more or less.
    /// let health = random.normal(100.0, 15.0);
    /// assert!(health.is_finite());
    /// ```
    ///
    /// Unbounded in both directions — for a value that must stay in a range, use
    /// [`Random::truncated_normal`]. See [`Normal`].
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        self.sample(&Normal::new(mean, std_dev).expect("invalid normal parameters"))
    }

    /// A bell curve, resampled until it lands inside `[low, high]`.
    /// # Question
    ///
    /// "What is a value clustered around an average that must never leave these bounds?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Tree height: averages 8m, never below 2m or above 20m.
    /// let height = random.truncated_normal(8.0, 3.0, 2.0, 20.0);
    /// assert!((2.0..=20.0).contains(&height));
    /// ```
    ///
    /// See [`TruncatedNormal`].
    pub fn truncated_normal(&mut self, mean: f64, std_dev: f64, low: f64, high: f64) -> f64 {
        self.sample(
            &TruncatedNormal::new(mean, std_dev, low, high)
                .expect("invalid truncated_normal parameters"),
        )
    }

    /// A value whose *logarithm* is normally distributed.
    /// # Question
    ///
    /// "What is a positive value where a few are very large and most are small?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Ore vein sizes: mostly small, occasionally huge.
    /// let vein = random.log_normal(2.0, 0.8);
    /// assert!(vein > 0.0);
    /// ```
    ///
    /// Always positive, with a long right tail — the shape of settlement sizes, wealth
    /// and vein richness. See [`LogNormal`].
    pub fn log_normal(&mut self, mu: f64, sigma: f64) -> f64 {
        self.sample(&LogNormal::new(mu, sigma).expect("invalid log_normal parameters"))
    }

    /// How long until the next event, at a constant rate.
    /// # Question
    ///
    /// "How much time passes before the next event?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Rate;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Lightning strikes twice a minute on average: how long until the next?
    /// let wait = random.exponential(Rate::new(2.0).unwrap());
    /// assert!(wait >= 0.0);
    /// ```
    ///
    /// Memoryless: waiting longer does not make the event more due. The continuous twin
    /// of [`Random::geometric`]. See [`Exponential`].
    pub fn exponential(&mut self, rate: Rate) -> f64 {
        self.sample(&Exponential::new(rate))
    }

    /// Between `low` and `high`, peaking at `mode`. See [`Triangular`].
    ///
    /// # Question
    ///
    /// "What is a value between two bounds that peaks at a most-likely point?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Wait times: between 1 and 10 seconds, usually about 3.
    /// let wait = random.triangular(1.0, 3.0, 10.0);
    /// assert!((1.0..=10.0).contains(&wait));
    /// ```
    pub fn triangular(&mut self, low: f64, mode: f64, high: f64) -> f64 {
        self.sample(&Triangular::new(low, mode, high).expect("invalid triangular parameters"))
    }

    /// The wait for `shape` events, each at scale `scale`.
    /// # Question
    ///
    /// "How long until several events have all happened?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Time for three growth stages to complete.
    /// let grown = random.gamma(3.0, 2.0);
    /// assert!(grown >= 0.0);
    /// ```
    ///
    /// A sum of `shape` exponential waits. See [`Gamma`].
    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        self.sample(&Gamma::new(shape, scale).expect("invalid gamma parameters"))
    }

    /// A fraction in `[0, 1]` whose shape is set by two counts.
    /// # Question
    ///
    /// "What is a proportion, when I have some prior belief about where it sits?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // A blend factor that favours the middle.
    /// let blend = random.beta(5.0, 5.0);
    /// assert!((0.0..=1.0).contains(&blend));
    /// ```
    ///
    /// `alpha` pulls towards one and `beta` towards zero; equal values are symmetric.
    /// See [`Beta`].
    pub fn beta(&mut self, alpha: f64, beta: f64) -> f64 {
        self.sample(&Beta::new(alpha, beta).expect("invalid beta parameters"))
    }

    /// A value in `[0, 1]` biased towards one end by `alpha`.
    /// # Question
    ///
    /// "What is a fraction that leans towards zero or towards one?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Loot quality that mostly comes out poor.
    /// let quality = random.power(0.3);
    /// assert!((0.0..=1.0).contains(&quality));
    /// ```
    ///
    /// An `alpha` above one leans high, below one leans low, exactly one is uniform.
    /// See [`Power`].
    pub fn power(&mut self, alpha: f64) -> f64 {
        self.sample(&Power::new(alpha).expect("invalid power parameters"))
    }

    /// A heavy-tailed value around `location`.
    /// # Question
    ///
    /// "What is a value that is usually near the centre but occasionally absurdly far?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let jitter = random.cauchy(0.0, 1.0);
    /// assert!(jitter.is_finite());
    /// ```
    ///
    /// **Has no mean and no variance** — averaging samples does not converge. Use it
    /// deliberately for wild outliers, not as a heavier normal. See [`Cauchy`].
    pub fn cauchy(&mut self, location: f64, scale: f64) -> f64 {
        self.sample(&Cauchy::new(location, scale).expect("invalid cauchy parameters"))
    }

    /// A value above `scale` with a power-law tail.
    /// # Question
    ///
    /// "What is a value where a few outcomes dominate everything else?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Settlement populations: a handful of cities, many hamlets.
    /// let population = random.pareto(100.0, 1.5);
    /// assert!(population >= 100.0);
    /// ```
    ///
    /// The eighty-twenty shape. Never below `scale`. See [`Pareto`].
    pub fn pareto(&mut self, scale: f64, shape: f64) -> f64 {
        self.sample(&Pareto::new(scale, shape).expect("invalid pareto parameters"))
    }

    /// Uniform integer in `[low, high]`, both inclusive. See [`UniformU64`].
    ///
    /// # Question
    ///
    /// "What is a random whole number between these bounds, both included?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // A stack of 1 to 64 items.
    /// let stack = random.uniform_u64(1, 64);
    /// assert!((1..=64).contains(&stack));
    /// ```
    pub fn uniform_u64(&mut self, low: u64, high: u64) -> u64 {
        self.sample(&UniformU64::new(low, high).expect("lower bound exceeds upper bound"))
    }

    /// Uniform integer in `[low, high]`, both inclusive. See [`UniformI64`].
    ///
    /// # Question
    ///
    /// "The same, allowing negatives?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let offset = random.uniform_i64(-32, 32);
    /// assert!((-32..=32).contains(&offset));
    /// ```
    pub fn uniform_i64(&mut self, low: i64, high: i64) -> i64 {
        self.sample(&UniformI64::new(low, high).expect("lower bound exceeds upper bound"))
    }

    /// Uniform index in `[0, length)`. Exact.
    ///
    /// # Question
    ///
    /// "Which item of this collection should I pick?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let loot = ["sword", "shield", "potion"];
    /// let picked = loot[random.uniform_index(loot.len())];
    /// # let _ = picked;
    /// ```
    /// One element of a slice, or [`None`] when it is empty.
    ///
    /// # Question
    ///
    /// "Which one of these?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let loot = ["sword", "shield", "potion"];
    ///
    /// assert!(loot.contains(random.pick(&loot).unwrap()));
    /// assert_eq!(random.pick::<u8>(&[]), None);
    /// ```
    ///
    /// The counterpart of [`Seed::pick`](crate::random::seed::Seed::pick), which
    /// answers the same question for a fixed seed. Prefer this to indexing with
    /// [`Random::uniform_index`], which panics on an empty slice where this reports it.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        (!items.is_empty()).then(|| &items[self.index_below(items.len())])
    }

    /// One element of a slice, mutably, or [`None`] when it is empty.
    ///
    /// # Question
    ///
    /// "Which one of these should I change?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(2u64));
    /// let mut health = [10u32; 4];
    ///
    /// if let Some(one) = random.pick_mut(&mut health) {
    ///     *one -= 1;
    /// }
    ///
    /// assert_eq!(health.iter().sum::<u32>(), 39);
    /// ```
    pub fn pick_mut<'a, T>(&mut self, items: &'a mut [T]) -> Option<&'a mut T> {
        if items.is_empty() {
            return None;
        }

        let index: usize = self.index_below(items.len());

        Some(&mut items[index])
    }

    pub fn uniform_index(&mut self, length: usize) -> usize {
        assert!(length > 0, "cannot sample an empty index range");
        self.bounded_u64(length as u64) as usize
    }

    /// Whether one event happens.
    /// # Question
    ///
    /// "Does this one thing happen?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Probability;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Does this chest contain a rare item?
    /// let rare = random.bernoulli(Probability::new(0.05).unwrap());
    /// # let _ = rare;
    /// ```
    ///
    /// For an exact fraction such as one in three, use [`Random::chance_ratio`]; for a
    /// chance held to 63 bits, [`Random::bernoulli_unit`]. See [`Bernoulli`].
    pub fn bernoulli(&mut self, chance: Probability) -> bool {
        self.sample(&Bernoulli::new(chance))
    }

    /// Whether one event happens, at a chance held to 63 bits.
    ///
    /// # Question
    ///
    /// "Does this one thing happen, when the chance has to be exact?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Unit;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // A one-in-a-million drop, realised to the last bit.
    /// let jackpot = random.bernoulli_unit(Unit::one_in(1_000_000));
    /// # let _ = jackpot;
    /// ```
    ///
    /// The realised frequency matches the stated chance to within `2^-64`, where
    /// [`Random::bernoulli`] rounds it onto an `f64` first. Integer-only, so the same
    /// seed decides the same way on every target.
    pub fn bernoulli_unit(&mut self, chance: Unit) -> bool {
        chance.decide_from(self)
    }

    /// True with exactly a [`Ratio`]'s chance. See [`BernoulliRatio`].
    /// Panics for a ratio above one.
    ///
    /// # Question
    ///
    /// "Does this happen, at an exact fraction such as one in three?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::units::Ratio;
    /// // Exactly one in three, which no f64 holds.
    /// let happened = random.chance_ratio(Ratio::one_in(3).unwrap());
    /// # let _ = happened;
    /// ```
    pub fn chance_ratio(&mut self, chance: Ratio) -> bool {
        self.sample(&BernoulliRatio::new(chance).expect("a probability must be at most one"))
    }

    /// True one time in `count`; never for a `count` of zero. See
    /// [`BernoulliRatio`].
    ///
    /// # Question
    ///
    /// "Does this happen, one time in N?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // A rare drop: one chest in four hundred.
    /// if random.one_in(400) { /* rare item */ }
    /// ```
    pub fn one_in(&mut self, count: u64) -> bool {
        self.sample(&BernoulliRatio::one_in(count))
    }

    /// 64 coins at once, one per bit. See [`BernoulliMask`].
    /// Panics for a ratio above one.
    ///
    /// # Question
    ///
    /// "Which of these 64 slots succeed, packed into one word?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::units::Ratio;
    /// let mask = random.chance_mask(Ratio::one_in(4).unwrap());
    /// let how_many = mask.count_ones();
    /// # let _ = how_many;
    /// ```
    pub fn chance_mask(&mut self, chance: Ratio) -> u64 {
        self.sample(&BernoulliMask::new(chance).expect("a probability must be at most one"))
    }

    /// How many of `trials` independent attempts succeed.
    /// # Question
    ///
    /// "Out of N independent attempts at probability P, how many succeed?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Probability;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // How many of 10,000 blocks get a random tick at 1 in 1,000?
    /// let ticked = random.binomial(10_000, Probability::new(0.001).unwrap());
    /// assert!(ticked <= 10_000);
    /// ```
    ///
    /// One call instead of N coin flips. Exact; [`Random::binomial_with`] can trade that
    /// for speed. To know *which* attempts succeeded, use [`Random::success_indices`].
    /// See [`Binomial`].
    pub fn binomial(&mut self, trials: u64, chance: Probability) -> u64 {
        self.sample(&Binomial::new(trials, chance))
    }

    /// The exact binomial. See [`BinomialRatio`].
    /// Panics for a ratio above one.
    ///
    /// # Question
    ///
    /// "Out of N attempts at an exact fraction, how many succeed?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::units::Ratio;
    /// let hits = random.binomial_ratio(1_000, Ratio::one_in(3).unwrap());
    /// assert!(hits <= 1_000);
    /// ```
    pub fn binomial_ratio(&mut self, trials: u64, chance: Ratio) -> u64 {
        self.sample(&BinomialRatio::new(trials, chance).expect("a probability must be at most one"))
    }

    /// How many events occur when the average count is `rate`.
    /// # Question
    ///
    /// "How many events happen in this region or interval, on average lambda?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Rate;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // How many rocks in this chunk, if chunks average 4.2?
    /// let rocks = random.poisson(Rate::new(4.2).unwrap());
    /// # let _ = rocks;
    /// ```
    ///
    /// The count that goes with [`Random::exponential`]'s waits. See [`Poisson`].
    pub fn poisson(&mut self, rate: Rate) -> u64 {
        self.sample(&Poisson::new(rate))
    }

    /// The exact Poisson. See [`PoissonRatio`].
    ///
    /// # Question
    ///
    /// "How many events at an exact average, with no floating point?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::units::Ratio;
    /// let events = random.poisson_ratio(Ratio::new(7, 2).unwrap());
    /// # let _ = events;
    /// ```
    pub fn poisson_ratio(&mut self, mean: Ratio) -> u64 {
        self.sample(&PoissonRatio::new(mean))
    }

    /// Integers around zero, thinning out with distance. See
    /// [`DiscreteLaplace`]. Panics for a zero scale.
    ///
    /// # Question
    ///
    /// "What is a whole-number offset around zero, mostly small?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::units::Ratio;
    /// // Jitter a grid position without leaving the grid.
    /// let offset = random.discrete_laplace(Ratio::new(3, 1).unwrap());
    /// # let _ = offset;
    /// ```
    pub fn discrete_laplace(&mut self, scale: Ratio) -> i64 {
        self.sample(&DiscreteLaplace::new(scale).expect("the scale must be above zero"))
    }

    /// The exact bell curve on integers. See [`DiscreteGaussian`].
    /// Panics for a variance too large to hold exactly.
    ///
    /// # Question
    ///
    /// "What is a whole number from a bell curve, exactly?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::units::Ratio;
    /// let value = random.discrete_gaussian(0, Ratio::new(4, 1).unwrap());
    /// # let _ = value;
    /// ```
    pub fn discrete_gaussian(&mut self, mean: i64, variance: Ratio) -> i64 {
        self.sample(&DiscreteGaussian::new(mean, variance).expect("variance too large"))
    }

    /// Zipf over `1..=elements`. See [`Zipf`], and keep one around for
    /// repeated sampling: its setup is not free.
    ///
    /// # Question
    ///
    /// "Which item, when a few are chosen far more often than the rest?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Word frequencies, popular spawns, common loot.
    /// let rank = random.zipf(100, 1.0);
    /// assert!((1..=100).contains(&rank));
    /// ```
    pub fn zipf(&mut self, elements: u64, exponent: f64) -> u64 {
        self.sample(&Zipf::new(elements, exponent).expect("invalid Zipf parameters"))
    }

    /// Failures before the first success. See [`Geometric`].
    ///
    /// # Question
    ///
    /// "How many attempts fail before the next success?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::units::Probability;
    /// // Skip straight to the next success instead of testing every position.
    /// let gap = random.geometric(Probability::new(0.01).unwrap());
    /// # let _ = gap;
    /// ```
    pub fn geometric(&mut self, chance: Probability) -> u64 {
        self.sample(&Geometric::new(chance))
    }

    /// Failures before the first success at a rational chance. See
    /// [`GeometricRatio`], and cache one for repeated draws at a rare chance.
    /// Panics for a ratio above one.
    ///
    /// # Question
    ///
    /// "The same, at an exact fraction?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::units::Ratio;
    /// let gap = random.geometric_ratio(Ratio::one_in(100).unwrap());
    /// # let _ = gap;
    /// ```
    pub fn geometric_ratio(&mut self, chance: Ratio) -> u64 {
        distributions::geometric_ratio(self, chance)
    }

    /// Failures before the first success at one in `count`: the one to reach
    /// for when scattering something every so many voxels or ticks. A `count`
    /// of zero never succeeds and gives `u64::MAX`. See [`GeometricRatio`].
    ///
    /// # Question
    ///
    /// "How far to the next one-in-N success?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Walk a row of voxels, landing only on the ore.
    /// let mut index = 0u64;
    /// index += random.geometric_one_in(1_000) + 1;
    /// # let _ = index;
    /// ```
    pub fn geometric_one_in(&mut self, count: u64) -> u64 {
        distributions::geometric_one_in(self, count)
    }

    /// Rounds up or down so the average is the value; 0 for a non-finite value.
    /// Rounds to a whole number, up or down in proportion to the fraction.
    /// # Question
    ///
    /// "How do I round repeatedly without the error piling up?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // 2.3 rounds to 2 seven times in ten and to 3 three times,
    /// // so a long run averages 2.3 rather than drifting to 2.
    /// let whole = random.stochastic_round(2.3);
    /// assert!(whole == 2 || whole == 3);
    /// ```
    ///
    /// For accumulating quantities — resource yields, damage over time — where always
    /// rounding the same way would bias the total. See [`StochasticRound`].
    pub fn stochastic_round(&mut self, value: f64) -> i64 {
        StochasticRound::new(value).map_or(0, |rounding| self.sample(&rounding))
    }

    /// `count` distinct indices below `length`, in no particular order.
    ///
    /// For choosing several things at once without repeats: three ore types,
    /// five spawn points in a chunk. Takes everything when `count` reaches
    /// `length`.
    ///
    /// Unbiased. Uses O(min(count, length)) time and auxiliary space, choosing
    /// a sparse swap map for small samples and a contiguous pool for dense ones.
    ///
    /// # Question
    ///
    /// "Which K different positions should I choose, uniformly?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Five distinct spawn slots out of a thousand.
    /// let slots = random.sample_distinct(5, 1_000);
    /// assert_eq!(slots.len(), 5);
    /// ```
    pub fn sample_distinct(&mut self, count: usize, length: usize) -> Vec<usize> {
        crate::structures::sampling::uniform_indices(length, count, Some(self))
    }

    /// A random set of weights; `None` when no concentration is positive and
    /// finite. See [`Dirichlet`].
    ///
    /// # Question
    ///
    /// "What is a random set of weights that sum to one?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // A random biome mix, three parts, none favoured.
    /// let mix = random.dirichlet(&[1.0, 1.0, 1.0]).expect("positive");
    /// # let _ = mix;
    /// ```
    pub fn dirichlet(&mut self, concentrations: &[f64]) -> Option<Weights> {
        Dirichlet::new(concentrations).and_then(|dirichlet| self.sample(&dirichlet))
    }

    /// Index drawn in proportion to each weight, O(n) per call. Use
    /// [`Categorical`] for O(1) repeated sampling of one distribution.
    ///
    /// # Question
    ///
    /// "Which category, given these weights?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::units::Weights;
    /// let table = Weights::new([10.0, 3.0, 1.0]).expect("positive");
    /// let picked = random.weighted_index(&table);
    /// assert!(picked < 3);
    /// ```
    pub fn weighted_index(&mut self, weights: &Weights) -> usize {
        self.sample(weights)
    }

    /// A point spread evenly over a filled circle of radius one.
    /// # Question
    ///
    /// "Where in this circular area should I place something?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Scatter within a blast radius.
    /// let point = random.point_in_disc();
    /// assert!(point.norm() <= 1.0);
    /// ```
    ///
    /// Evenly by *area*, so it does not bunch at the centre the way scaling a random
    /// radius would. See [`UnitDisc`].
    pub fn point_in_disc(&mut self) -> Vector2<f64> {
        self.sample(&UnitDisc)
    }

    /// A point spread evenly through a filled sphere of radius one.
    /// # Question
    ///
    /// "Where in this spherical volume should I place something?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let point = random.point_in_ball();
    /// assert!(point.norm() <= 1.0);
    /// ```
    ///
    /// Evenly by *volume*. See [`UnitBall`].
    pub fn point_in_ball(&mut self) -> Vector3<f64> {
        self.sample(&UnitBall)
    }

    /// A random direction in the plane.
    /// # Question
    ///
    /// "Which way should this face, in two dimensions?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let facing = random.unit_vector_2d();
    /// assert!((facing.norm() - 1.0).abs() < 1e-12);
    /// ```
    ///
    /// See [`UnitCircle`].
    pub fn unit_vector_2d(&mut self) -> Vector2<f64> {
        self.sample(&UnitCircle)
    }

    /// A random direction in space.
    /// # Question
    ///
    /// "Which way should this face, or which way should it be thrown?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Scatter debris in every direction equally.
    /// let direction = random.unit_vector_3d();
    /// assert!((direction.norm() - 1.0).abs() < 1e-12);
    /// ```
    ///
    /// Evenly over the sphere, which picking two uniform angles would not be — that
    /// bunches at the poles. For an orientation rather than a direction, use
    /// [`Random::rotation`]. See [`UnitSphere`].
    pub fn unit_vector_3d(&mut self) -> Vector3<f64> {
        self.sample(&UnitSphere)
    }

    /// A random point on the unit sphere in four dimensions.
    /// # Question
    ///
    /// "What is a uniform point on the 3-sphere?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let point = random.unit_vector_4d();
    /// assert!((point.norm() - 1.0).abs() < 1e-12);
    /// ```
    ///
    /// Mostly useful as a uniform rotation, which [`Random::rotation`] wraps it as.
    /// See [`UnitHypersphere`].
    pub fn unit_vector_4d(&mut self) -> Vector4<f64> {
        self.sample(&UnitHypersphere)
    }
}

// ---------------------------------------------------------------------------
// Asking about many things at once
// ---------------------------------------------------------------------------

/// Bulk questions: one call where the obvious code would loop.
///
/// Each of these answers a question about a whole population with a handful of
/// draws. See [`approximation`] for when a sampler may trade exactness for speed,
/// and [`distributions::bulk`](crate::random::distributions) for the algorithms.
impl Random {
    /// How many of `count` positions succeed, and which ones.
    ///
    /// # Question
    ///
    /// "Which of these N positions succeed at probability P?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Unit;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // Which of 10,000 voxels get a random tick this frame?
    /// let ticked = random.success_indices(10_000, Unit::one_in(1_000));
    ///
    /// for index in &ticked {
    ///     assert!(*index < 10_000);
    /// }
    /// ```
    ///
    /// Ascending, and costs one draw per *success* rather than per position — see
    /// [`SparseSuccesses`]. For the count alone, [`Random::binomial`] is cheaper
    /// still.
    pub fn success_indices(&mut self, count: u64, chance: Unit) -> Vec<u64> {
        self.sample(&SparseSuccesses::new(count, chance)).collect()
    }

    /// Sixty-four independent coins at once, as bits.
    ///
    /// # Question
    ///
    /// "Which of these 64 slots succeed, packed into one word?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Ratio;
    /// # let mut random = Random::new(Seed::from_integer(2u64));
    /// // Bit `i` is set when slot `i` succeeded.
    /// let mask = random.success_mask(Ratio::one_in(4).unwrap());
    /// let how_many = mask.count_ones();
    /// # let _ = how_many;
    /// ```
    ///
    /// Exact for any fraction, and a handful of words whatever the chance — the same
    /// machinery as [`Random::chance_mask`], which this is a clearer name for.
    pub fn success_mask(&mut self, chance: Ratio) -> u64 {
        self.chance_mask(chance)
    }

    /// How many of `count` items land in each category.
    ///
    /// # Question
    ///
    /// "Distribute N items among these categories by weight. How many in each?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Unit;
    /// # let mut random = Random::new(Seed::from_integer(3u64));
    /// let biomes = [
    ///     Unit::percent(40).unwrap(), // forest
    ///     Unit::percent(30).unwrap(), // grass
    ///     Unit::percent(20).unwrap(), // desert
    ///     Unit::percent(10).unwrap(), // tundra
    /// ];
    /// let counts = random.multinomial(10_000, &biomes).unwrap();
    ///
    /// assert_eq!(counts.iter().sum::<u64>(), 10_000);
    /// ```
    ///
    /// The counts always sum to exactly `count`. `None` for no categories or all-zero
    /// weights; see [`Multinomial`] for how the shares are normalised.
    pub fn multinomial(&mut self, count: u64, shares: &[Unit]) -> Option<Vec<u64>> {
        let split = Multinomial::new(count, shares)?;

        Some(self.sample(&split))
    }

    /// How many successes are drawn from a finite population, without replacement.
    ///
    /// # Question
    ///
    /// "Drawing K from a population of S successes and the rest failures, how many
    /// successes do I get?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(4u64));
    /// // Ten cards from a deck of 30 rare and 70 common.
    /// let rares = random.hypergeometric(100, 30, 10).unwrap();
    ///
    /// assert!(rares <= 10);
    /// ```
    ///
    /// Unlike [`Random::binomial`], each draw changes what is left. `None` when there
    /// are more successes than population.
    pub fn hypergeometric(&mut self, population: u64, successes: u64, draws: u64) -> Option<u64> {
        let deck = Hypergeometric::new(population, successes, draws)?;

        Some(self.sample(&deck))
    }

    /// How many failures occur before the `successes`-th success.
    ///
    /// # Question
    ///
    /// "How many attempts fail before I get R successes?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Unit;
    /// # let mut random = Random::new(Seed::from_integer(5u64));
    /// // How many spots are rejected before three are placed?
    /// let rejected = random.negative_binomial(3, Unit::one_in(8));
    /// let inspected = rejected + 3;
    /// # let _ = inspected;
    /// ```
    ///
    /// **Failures only** — the successes are not counted, so total trials is
    /// `result + successes`.
    pub fn negative_binomial(&mut self, successes: u64, chance: Unit) -> u64 {
        self.sample(&NegativeBinomial::new(successes, chance))
    }

    /// How many succeed when every trial has its own chance.
    ///
    /// # Question
    ///
    /// "Each candidate has a different chance. How many succeed altogether?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::units::Unit;
    /// # let mut random = Random::new(Seed::from_integer(6u64));
    /// let soils = [
    ///     Unit::percent(80).unwrap(),
    ///     Unit::percent(50).unwrap(),
    ///     Unit::percent(5).unwrap(),
    /// ];
    /// let germinated = random.poisson_binomial(&soils);
    ///
    /// assert!(germinated <= 3);
    /// ```
    ///
    /// One draw per trial, since the trials genuinely differ. Averaging the chances
    /// and passing that to [`Random::binomial`] gets the mean right and the spread
    /// wrong.
    pub fn poisson_binomial(&mut self, chances: &[Unit]) -> u64 {
        self.sample(&PoissonBinomial::new(chances))
    }

    /// `count` distinct items chosen in proportion to their weights.
    ///
    /// # Question
    ///
    /// "Which K distinct items should I choose, weighted?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(7u64));
    /// // Three distinct loot items from a weighted table.
    /// let chosen = random.choose_weighted_n(&[10.0, 5.0, 1.0, 0.5], 3);
    ///
    /// assert_eq!(chosen.len(), 3);
    /// let mut sorted = chosen.clone();
    /// sorted.sort_unstable();
    /// sorted.dedup();
    /// assert_eq!(sorted.len(), 3, "no duplicates");
    /// ```
    ///
    /// Nothing is chosen twice. Uses exponential keys (Efraimidis–Spirakis) rather
    /// than drawing and rejecting duplicates, so the cost does not blow up when
    /// `count` approaches the number of items. Weights that are zero, negative or
    /// not finite are never chosen, so fewer than `count` may come back.
    pub fn choose_weighted_n(&mut self, weights: &[f64], count: usize) -> Vec<usize> {
        crate::structures::sampling::weighted_indices(weights, count, self)
    }

    /// `count` items chosen uniformly from a stream of unknown length.
    ///
    /// # Question
    ///
    /// "Which K items should I keep, seeing the stream once and never knowing how
    /// long it is?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(8u64));
    /// // Five samples from a sequence that is generated, not stored.
    /// let kept = random.reservoir_sample((0..1_000_000).filter(|n| n % 7 == 0), 5);
    ///
    /// assert_eq!(kept.len(), 5);
    /// ```
    ///
    /// Holds only the items it might keep: **O(count) memory however long the stream
    /// is**. Fewer than `count` come back only if the stream was shorter.
    pub fn reservoir_sample<T, I>(&mut self, stream: I, count: usize) -> Vec<T>
    where
        I: IntoIterator<Item = T>,
    {
        crate::structures::sampling::reservoir_sample(stream, count, self)
    }

    /// Puts these items in a random order.
    ///
    /// # Question
    ///
    /// "What is a random ordering of these items?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(9u64));
    /// let mut deck: Vec<u32> = (0..52).collect();
    /// random.shuffle(&mut deck);
    ///
    /// assert_eq!(deck.len(), 52);
    /// ```
    ///
    /// Every ordering equally likely, in place, one draw per item.
    ///
    /// # Anything that can be shuffled
    ///
    /// The argument is any [`Shuffle`](crate::structures::traits::Shuffle), not only a
    /// slice, so a collection that keeps an index or splits its storage reorders itself
    /// correctly without this method knowing how:
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::structures::collections::sequences::ring_buffer::RingBuffer;
    /// # let mut random = Random::new(Seed::from_integer(3u64));
    /// let mut recent: RingBuffer<u32> = RingBuffer::new(8);
    /// for value in 0..8 {
    ///     recent.push_back(value);
    /// }
    ///
    /// random.shuffle(&mut recent);
    ///
    /// assert_eq!(recent.len(), 8);
    /// ```
    ///
    /// The counterpart is [`Seed::shuffle`](crate::random::seed::Seed::shuffle), which
    /// gives one fixed ordering instead of a new one each call. This advances the
    /// stream; that one takes a temporary cursor from a seed.
    /// Everything one weighted table produces, this time.
    ///
    /// # Question
    ///
    /// "What is in this chest, on this occasion?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// use voxel_world::random::bulk_pick::{BulkPickEntry, BulkPickTable, Count, Quantity};
    ///
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let spawns = BulkPickTable {
    ///     count: Count::Uniform { low: 0, high: 4 },
    ///     entries: vec![
    ///         BulkPickEntry::new("wolf", 70, Quantity::Fixed(1)),
    ///         BulkPickEntry::new("bear", 30, Quantity::Fixed(1)),
    ///     ],
    /// };
    ///
    /// let group = random.bulk_pick(&spawns).expect("a valid table");
    ///
    /// // Nought to four animals, and possibly none at all.
    /// assert!(group.iter().map(|one| one.selections).sum::<u64>() <= 4);
    /// ```
    ///
    /// Advances the stream, so consecutive calls generally differ. For the same answer
    /// every time — a chest that holds what it held before — use
    /// [`Seed::bulk_pick`](crate::random::seed::Seed::bulk_pick).
    ///
    /// The three stages and what each costs are documented on
    /// [`bulk_pick`](crate::random::bulk_pick::bulk_pick), which this calls.
    pub fn bulk_pick<T, C, Q>(
        &mut self,
        table: &BulkPickTable<T, C, Q>,
    ) -> Result<Vec<BulkPickResult<T>>, BulkError>
    where
        T: Clone,
        C: CountDistribution,
        Q: QuantityDistribution,
    {
        bulk_pick(table, self)
    }

    pub fn shuffle<T: Shuffle + ?Sized>(&mut self, items: &mut T) {
        items.shuffle(self);
    }

    /// Fills just the first `count` places with a random selection.
    ///
    /// # Question
    ///
    /// "Which K unique items should I pick, without bothering to order the rest?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(10u64));
    /// let mut candidates: Vec<u32> = (0..1_000).collect();
    /// let taken = random.partial_shuffle(&mut candidates, 3);
    ///
    /// assert_eq!(taken, 3);
    /// let chosen = &candidates[..taken];
    /// # let _ = chosen;
    /// ```
    ///
    /// One draw per *chosen* item rather than per item. Only the first `count` places
    /// are a valid random selection; the rest are left disturbed, not shuffled.
    pub fn partial_shuffle<T: Shuffle + ?Sized>(&mut self, items: &mut T, count: usize) -> usize {
        items.partial_shuffle(count, self)
    }
}

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
        let drift: f64 = approximation::binomial_odds_drift_for_hypergeometric(population, deck.draws());

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


