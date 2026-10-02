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
}
