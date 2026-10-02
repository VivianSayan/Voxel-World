impl Random {
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
