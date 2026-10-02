impl Seed {
    /// The seed as it is stored, for saving or for passing to the noise fields.
    pub const fn value(self) -> u128 {
        self.0
    }

    /// Folded to 64 bits, for callers that have no room for more.
    pub fn as_u64(self) -> u64 {
        fold_u128(self.0)
    }

    /// Folded to 32 bits.
    pub fn as_u32(self) -> u32 {
        let folded: u64 = self.as_u64();

        (folded ^ (folded >> 32)) as u32
    }

    /// Uniform in `[0, 1)`, for a one-off fraction with no generator to make.
    ///
    /// [`Seed::unit`] is the integer-backed version, and the better choice unless
    /// the result is headed for floating-point maths.
    pub fn unit_f64(self) -> f64 {
        self.stream().unit_f64()
    }

    /// Uniform in `[0, 1)` on the `2^-63` grid, for a one-off fraction.
    ///
    /// The same word as [`Seed::unit_f64`], read at full precision instead of 53
    /// bits.
    pub fn unit(self) -> Unit {
        self.stream().unit()
    }

    /// The one [`Fixed`] in `[0, 1)` this seed stands for.
    ///
    /// # Question
    ///
    /// "What fixed-point fraction belongs to this place, for ever?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::math::{Fixed, Vector3};
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// let world = Seed::from_integer(1u64).child("terrain");
    /// let here = world.at_position(VoxelPosition3::new(Vector3::new(8, 0, 3)));
    ///
    /// let jitter = here.fixed();
    ///
    /// assert_eq!(jitter, here.fixed(), "the same place, the same answer");
    /// assert!(jitter < Fixed::ONE);
    /// ```
    ///
    /// Reads the same word as [`Seed::unit`], kept to this layout's width, so the two
    /// are one value at two precisions rather than two unrelated draws.
    pub fn fixed(self) -> Fixed {
        Fixed::fraction_from_seed(self)
    }

    /// The one [`Fixed`] in `[low, high)` this seed stands for.
    ///
    /// # Question
    ///
    /// "What value between these bounds belongs to this place?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::math::{Fixed, Vector3};
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// let world = Seed::from_integer(2u64).child("trees");
    /// let here = world.at_position(VoxelPosition3::new(Vector3::new(1, 0, 1)));
    ///
    /// let height = here.fixed_range(Fixed::from_integer(4), Fixed::from_integer(9));
    ///
    /// assert!(height >= Fixed::from_integer(4) && height < Fixed::from_integer(9));
    /// assert_eq!(height, here.fixed_range(Fixed::from_integer(4), Fixed::from_integer(9)));
    /// ```
    ///
    /// # Panics
    ///
    /// If the range is empty or inverted, as [`Random::fixed_range`] does.
    pub fn fixed_range(self, low: Fixed, high: Fixed) -> Fixed {
        let range = crate::random::fixed_point::UniformFixedRange::new(low, high)
            .expect("a fixed-point range must hold at least one value");

        self.sample(&range)
    }

    /// A uniformly random [`Probability`] in `[0, 1)`, for a one-off chance.
    ///
    /// The same word as [`Seed::unit`], narrowed to what an `f64` holds. Prefer
    /// [`Seed::unit`] where the value will be compared or stored; this is for
    /// handing a drawn chance to something that wants a [`Probability`].
    pub fn probability(self) -> Probability {
        self.unit().to_probability()
    }

    /// A uniformly random [`UnitValue`] in `[0, 1)`.
    pub fn unit_value(self) -> UnitValue {
        UnitValue::from_seed(self)
    }

    /// A uniformly random [`NoiseValue`] in `[-1, 1)`.
    ///
    /// A uniform sample of the range, not a sample of a noise field: for jitter,
    /// dithering and test data rather than terrain.
    pub fn noise_value(self) -> NoiseValue {
        self.stream().sample(&UniformNoise)
    }

    /// A uniformly random orientation.
    ///
    /// Uniform over rotations, not over Euler angles — see
    /// [`UniformRotation`] for why
    /// those differ.
    pub fn rotation(self) -> UnitQuaternion {
        self.sample(&UniformRotation)
    }

    /// True with probability `chance`, for a one-off decision.
    ///
    /// The same seed always decides the same way, which is what makes this
    /// usable for world content rather than only for effects.
    pub fn chance(self, chance: Probability) -> bool {
        self.sample(&Bernoulli::new(chance))
    }

    /// True with probability `chance`, for a one-off decision.
    ///
    /// Exact: the chance of a yes is the [`Unit`] to its last bit, for every value
    /// including zero and one. [`Seed::chance`] rounds the effective probability to
    /// the 53-bit grid instead, so this is the one to reach for when the number
    /// matters — particularly near one, where a `Unit` holds values an `f64` cannot
    /// tell from certainty.
    pub fn chance_unit(self, chance: Unit) -> bool {
        chance.decide(self)
    }

    /// True with exactly a [`Ratio`]'s chance; `None` for a ratio above one.
    pub fn chance_ratio(self, chance: Ratio) -> Option<bool> {
        Some(self.sample(&BernoulliRatio::new(chance)?))
    }

    /// True one time in `count`; never for a `count` of zero.
    pub fn one_in(self, count: u64) -> bool {
        self.sample(&BernoulliRatio::one_in(count))
    }

    /// Uniform in `0..length`, or `None` when `length` is zero.
    ///
    /// Exactly uniform (Lemire's method, see
    /// [`StochasticSource::bounded_u64`]), and still a pure function of the seed:
    /// the same seed always answers the same, rejection or not.
    pub fn below(self, length: u64) -> Option<u64> {
        (length > 0).then(|| self.stream().bounded_u64(length))
    }

    /// Everything one weighted table produces for this seed — the same, for ever.
    ///
    /// # Question
    ///
    /// "What is in *this* chest? And what will be in it when the next player opens
    /// it?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::fixed;
    /// use voxel_world::math::Vector3;
    /// use voxel_world::random::bulk_pick::{BulkPickEntry, BulkPickTable, Count, Quantity};
    /// use voxel_world::random::seed::Seed;
    /// use voxel_world::spatial::VoxelPosition3;
    /// use voxel_world::unit;
    ///
    /// const LOOT_VERSION: u32 = 1;
    ///
    /// let loot = BulkPickTable {
    ///     count: Count::Poisson { mean: fixed!(3.0) },
    ///     entries: vec![
    ///         BulkPickEntry::new("iron", 40, Quantity::Binomial { trials: 8, chance: unit!(0.5) }),
    ///         BulkPickEntry::new("arrows", 25, Quantity::Uniform { low: 4, high: 12 }),
    ///         BulkPickEntry::new("coins", 30, Quantity::Poisson { mean: fixed!(8.0) }),
    ///         BulkPickEntry::new("shard", 5, Quantity::Geometric { chance: unit!(0.4) }),
    ///     ],
    /// };
    ///
    /// let world = Seed::from_integer(1u64);
    /// let here = VoxelPosition3::new(Vector3::new(12, 0, 7));
    ///
    /// let chest = world.child("loot").at_position(here).at_version(LOOT_VERSION);
    /// let contents = chest.bulk_pick(&loot).expect("a valid table");
    ///
    /// // Opened twice, the same chest.
    /// assert_eq!(contents, chest.bulk_pick(&loot).unwrap());
    /// ```
    ///
    /// The result may well be empty, because a Poisson count can be zero. That is an
    /// ordinary chest with nothing in it, not a failure, and it needs no `"Empty"` row
    /// in the table to express.
    ///
    /// # One cursor for the whole question
    ///
    /// The count, the weighted selection and every quantity are drawn from a single
    /// temporary cursor, which is then discarded. One table plus one seed is one
    /// question with one answer — bumping [`at_version`](Seed::at_version) is how to
    /// change it deliberately.
    pub fn bulk_pick<T, C, Q>(
        self,
        table: &crate::random::bulk_pick::BulkPickTable<T, C, Q>,
    ) -> Result<Vec<crate::random::bulk_pick::BulkPickResult<T>>, crate::random::bulk_pick::BulkError>
    where
        T: Clone,
        C: crate::random::bulk_pick::CountDistribution,
        Q: crate::random::bulk_pick::QuantityDistribution,
    {
        let mut cursor: SeedCursor = self.cursor();

        crate::random::bulk_pick::bulk_pick(table, &mut cursor)
    }

    /// Puts a collection into the one random order this seed stands for.
    ///
    /// # Question
    ///
    /// "What order does this seed put these in — the same order, every time?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// let chest = Seed::from_integer(4u64).child("chest").index(17);
    ///
    /// let mut loot = ["sword", "shield", "potion", "rope"];
    /// chest.shuffle(&mut loot);
    ///
    /// // Reopening the same chest finds the same arrangement.
    /// let mut again = ["sword", "shield", "potion", "rope"];
    /// chest.shuffle(&mut again);
    ///
    /// assert_eq!(loot, again);
    /// ```
    ///
    /// # Anything that can be shuffled, not just a slice
    ///
    /// The argument is any [`Shuffle`], so an
    /// array, a `Vec`, an `OrderedSet`, a `RingBuffer` or a collection written later
    /// all work without this method knowing anything about them:
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::structures::collections::sequences::ordered_set::OrderedSet;
    /// let seed = Seed::from_integer(7u64);
    ///
    /// let mut set: OrderedSet<&str> = OrderedSet::new();
    /// for name in ["a", "b", "c", "d"] {
    ///     set.push(name);
    /// }
    ///
    /// seed.shuffle(&mut set);
    ///
    /// // The set reordered itself *and* rebuilt the index that finds its members.
    /// assert!(set.contains(&"c"));
    /// ```
    ///
    /// # Where the responsibility lies
    ///
    /// ```text
    /// Seed        provides the randomness
    /// Shuffle     defines what a random reordering means for the type
    /// collection  owns its representation and its invariants
    /// ```
    ///
    /// A seed cannot know that an `OrderedSet` has a member-to-position index to
    /// rebuild, or that a `RingBuffer` is split in two inside its deque. It does not
    /// need to: it hands over a source of random words and the collection does the
    /// rest. A new collection joins in by implementing `Shuffle`, with no change here.
    ///
    /// # Why a seed can shuffle at all
    ///
    /// Every ordering is equally likely, as from a stream — the difference is only
    /// where the words come from. A [`Random`] moves on with each draw, so shuffling
    /// twice gives two orders; a seed is a fixed question, so it gives one order for
    /// ever. That is what a generated world needs: a chest whose contents are laid out
    /// the same way for every player who opens it.
    pub fn shuffle<T: Shuffle + ?Sized>(self, items: &mut T) {
        // One cursor for the whole call, so a single shuffle is a single deterministic
        // stream starting at this seed rather than a cursor remade per draw.
        let mut cursor: SeedCursor = self.cursor();

        items.shuffle(&mut cursor);
    }

    /// The same items, in the one random order this seed stands for.
    ///
    /// # Question
    ///
    /// "What is this list, as this seed would arrange it?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// let seed = Seed::from_integer(8u64);
    /// let order = seed.shuffled((0..10).collect::<Vec<u32>>());
    ///
    /// assert_eq!(order.len(), 10);
    /// assert_eq!(order, seed.shuffled((0..10).collect::<Vec<u32>>()));
    /// ```
    ///
    /// The owning form of [`Seed::shuffle`], for when there is no list to borrow yet.
    pub fn shuffled<T>(self, mut items: Vec<T>) -> Vec<T> {
        self.shuffle(&mut items);

        items
    }

    /// Fills just the first `count` places with the selection this seed stands for.
    ///
    /// # Question
    ///
    /// "Which `count` of these does this seed choose, without ordering the rest?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// let region = Seed::from_integer(2u64).child("spawns");
    ///
    /// let mut candidates: Vec<u32> = (0..1_000).collect();
    /// let filled = region.partial_shuffle(&mut candidates, 3);
    ///
    /// assert_eq!(filled, 3);
    ///
    /// // Three draws, not a thousand — and the same three every time.
    /// let chosen = candidates[..3].to_vec();
    /// let mut again: Vec<u32> = (0..1_000).collect();
    /// region.partial_shuffle(&mut again, 3);
    ///
    /// assert_eq!(chosen, again[..3]);
    /// ```
    ///
    /// As with [`Seed::shuffle`], the argument is any
    /// [`Shuffle`] and the collection decides what
    /// "the first `count` places" means for its own order.
    ///
    /// # What is and is not guaranteed
    ///
    /// Returns `count`, or the length if that is smaller. Only those first places are
    /// a uniform selection drawn without replacement; what follows them is left in
    /// whatever order the partial swaps happened to leave it, and is **not** a shuffle
    /// of the remainder.
    pub fn partial_shuffle<T: Shuffle + ?Sized>(self, items: &mut T, count: usize) -> usize {
        let mut cursor: SeedCursor = self.cursor();

        items.partial_shuffle(count, &mut cursor)
    }

    /// Picks one element, or `None` when the slice is empty.
    pub fn pick<T>(self, items: &[T]) -> Option<&T> {
        self.below(items.len() as u64)
            .map(|index| &items[index as usize])
    }

    /// One draw from any distribution, the same draw every time for the same
    /// seed.
    ///
    /// ```ignore
    /// let trunk = Triangular::new(4.0, 6.0, 11.0)?;
    /// let height = trees.at_voxel(position).sample(&trunk);
    /// ```
    ///
    /// Every word is a domain-separated permutation of one position in a
    /// full-cycle seed cursor. That whitening applies even to [`Seed::from_raw`]
    /// values, so small or adjacent stored seeds do not expose small or
    /// adjacent first draws. The seed itself is never changed. For a value that
    /// must match on every machine, require a
    /// [`PortableDistribution`](crate::random::PortableDistribution).
    pub fn sample<D: Distribution>(self, distribution: &D) -> D::Output {
        distribution.sample(&mut self.stream())
    }

    /// Picks one of a [`WeightedDiscrete`]'s values, the same one every time
    /// for the same seed.
    ///
    /// Separate from [`Seed::sample`] only because the table hands back a
    /// reference into itself, which a [`Distribution`] cannot.
    pub fn sample_weighted<T>(self, table: &WeightedDiscrete<T>) -> &T {
        table.sample(&mut self.stream())
    }

    /// Creates the compact advancing source rooted at this seed.
    pub const fn cursor(self) -> SeedCursor {
        SeedCursor::new(self)
    }

    /// The temporary cursor a one-shot draw reads from.
    fn stream(self) -> SeedCursor {
        self.cursor()
    }

    /// A generator started from this seed, for anything that needs a stream
    /// rather than a single value.
    pub fn to_random(self) -> Random {
        Random::new(self)
    }
}

// ---------------------------------------------------------------------------
// Writing one down
// ---------------------------------------------------------------------------
