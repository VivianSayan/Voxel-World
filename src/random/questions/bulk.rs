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
