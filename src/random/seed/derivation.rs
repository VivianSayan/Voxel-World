impl Seed {
    /// A named branch: the seed for one subsystem, layer or feature.
    ///
    /// The child is a deterministic function of the parent, not independent of
    /// it; what the label buys is that the two are decorrelated. Knowing one
    /// branch tells you nothing usable about another or about the parent
    /// without inverting the mixing, so terrain and caves under one world show
    /// no shared structure even though both come from the same number.
    ///
    /// The name is part of the world's identity once anything has generated
    /// from it: renaming `"caves"` changes every cave.
    pub fn child(self, label: &str) -> Self {
        Self(absorb128(
            absorb128(self.0 ^ CHILD_TAG, hash_bytes(label.as_bytes())),
            label.len() as u128,
        ))
    }

    /// Follows several labels at once, the same as calling [`Seed::child`] for
    /// each in turn.
    pub fn path<'a>(self, labels: impl IntoIterator<Item = &'a str>) -> Self {
        labels
            .into_iter()
            .fold(self, |seed, label| seed.child(label))
    }

    /// A numbered branch, for one of a run of things: the nth attempt, the nth
    /// entity, the nth octave.
    pub fn index(self, index: u64) -> Self {
        Self(absorb128(self.0 ^ INDEX_TAG, expand_u64(index)))
    }

    /// A branch for a point in space, in as many dimensions as are given.
    ///
    /// The coordinates go in one at a time rather than being folded together,
    /// so permuting them gives different seeds and no two axes can cancel.
    /// Takes the array from any of the vector types: `seed.at(p.to_array())`.
    pub fn at<const N: usize>(self, coordinates: [i128; N]) -> Self {
        let mut state: u128 = self.0 ^ POSITION_TAG ^ (N as u128);

        for coordinate in coordinates {
            state = absorb128(state, coordinate as u128);
        }

        Self(state)
    }

    /// A branch for a voxel, which is [`Seed::at`] without spelling out the
    /// array.
    pub fn at_voxel(self, position: VoxelPosition3) -> Self {
        self.at(position.to_array())
    }

    /// A branch for a point at one octree depth, so the same coordinates at
    /// different depths are unrelated.
    /// Prefer [`Seed::at_position`] with a
    /// [`NodePosition3`]: a node position carries its
    /// own depth, so there is no pair of arguments to transpose. This remains for
    /// loose coordinates that are not yet a position.
    pub fn at_depth<const N: usize>(self, depth: Depth, coordinates: [i128; N]) -> Self {
        self.at(coordinates).at_level(depth)
    }

    /// Folds an octree level in, on a channel of its own.
    ///
    /// # Question
    ///
    /// "Which level of the tree is this?"
    ///
    /// # Why a level is not an index
    ///
    /// This used to go through [`Seed::index`], and that was a bug: it made "the
    /// fifth thing at this position" and "the depth-five node at this position" the
    /// same seed. A tree's own levels then correlated with whatever else counted from
    /// the same place — a fifth tree, a fifth ore vein — for no reason anybody could
    /// trace.
    ///
    /// # Why the level has to be folded at all
    ///
    /// Because the addresses alone are *not* distinct. An octree address at depth `d`
    /// is the voxel position shifted right by `tree_depth − d`, so climbing a level
    /// drops a bit:
    ///
    /// ```text
    /// voxel (8, 0, 0) at the deepest level
    ///       (4, 0, 0) one level up
    ///       (2, 0, 0) two levels up
    /// ```
    ///
    /// The address `(4, 0, 0)` therefore names a real node at *many* different
    /// depths, and "address `a` at depth `d`" collides with "address `a << 1` at
    /// depth `d + 1`" on the bits alone. Only the level tells them apart, which is
    /// why it is folded and why there are tests for exactly that alignment.
    pub fn at_level(self, depth: Depth) -> Self {
        Self(absorb128(
            self.0 ^ DEPTH_TAG,
            expand_u64(depth.level() as i64 as u64),
        ))
    }

    /// The seed for one octave of a layered field.
    ///
    /// # Question
    ///
    /// "What seed does octave `n` of this noise use?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// let terrain = Seed::from_integer(5u64).child("terrain");
    /// let here = VoxelPosition3::new(Vector3::new(100, 0, 100));
    ///
    /// // Each octave samples the same place at its own frequency, and must not
    /// // reuse another octave's randomness or the layers correlate visibly.
    /// let layers: Vec<Seed> = (0..6).map(|n| terrain.at_octave(n).at_position(here)).collect();
    ///
    /// assert_eq!(layers.len(), 6);
    /// ```
    ///
    /// # Why not [`Seed::index`] or [`Seed::at_level`]
    ///
    /// All three count small integers, and all three would collide if they shared a
    /// channel. They mean different things: an index picks one of several siblings, a
    /// level names a depth in a tree, an octave names a frequency band over the same
    /// space. A field's third octave and a tree's third level are unrelated, and
    /// should stay that way.
    ///
    /// Octaves are the case where a collision is easiest to *see*: two octaves
    /// sharing randomness makes the second a scaled copy of the first, and the result
    /// looks like a repeating pattern rather than noise.
    pub fn at_octave(self, octave: u32) -> Self {
        Self(absorb128(
            self.0 ^ OCTAVE_TAG,
            expand_u64(u64::from(octave)),
        ))
    }

    /// The seed for a position, whatever shape of position it is.
    ///
    /// # Question
    ///
    /// "What seed belongs to this place?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::{Depth, NodePosition3, VoxelPosition3};
    /// # use voxel_world::math::Vector3;
    /// let world = Seed::from_integer(1u64);
    ///
    /// // One voxel, and one whole node of the tree — the same call either way.
    /// let voxel = world.at_position(VoxelPosition3::new(Vector3::new(4, 64, -9)));
    /// let node = world.at_position(NodePosition3::new(Vector3::new(1, 2, 3), Depth::new(4)));
    ///
    /// assert_ne!(voxel, node, "a voxel and a node are different places");
    /// ```
    ///
    /// Takes any of [`VoxelPosition2`],
    /// `VoxelPosition3`, `VoxelPosition4` or the `NodePosition` family, so the two
    /// and four dimensional cases read the same as the three dimensional one. A
    /// `NodePosition` carries its own [`Depth`], so there is no second argument to
    /// pass or to get the wrong way round — which is what [`Seed::at_depth`] asked
    /// for and this does not.
    ///
    /// The dimension count and the depth are both folded in, so a voxel at `(1, 2)`
    /// and a voxel at `(1, 2, 0)` are different places, and so are the same address
    /// at two different depths.
    pub fn at_position<P: SeedablePosition>(self, position: P) -> Self {
        position.derive(self)
    }

    /// A seed two things agree on, whichever of them asks.
    ///
    /// # Question
    ///
    /// "What seed do these two share, without either one being in charge?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// let left = Seed::from_integer(10u64);
    /// let right = Seed::from_integer(20u64);
    ///
    /// // Order does not matter, which is the whole point.
    /// assert_eq!(left.between(right), right.between(left));
    /// ```
    ///
    /// # Why this is not [`Seed::combine`]
    ///
    /// `combine` is **ordered**: `a.combine(b)` and `b.combine(a)` are different
    /// seeds. That is right when one side owns the result — a world seed combined
    /// with a biome seed — and wrong whenever two peers have to reach the same
    /// answer independently.
    ///
    /// The voxel case: the face between two neighbouring voxels belongs to both, and
    /// each must decide what is on it without consulting the other. Written with
    /// `combine` the two disagree, and the seam shows. Written with this they cannot:
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// # let world = Seed::from_integer(1u64);
    /// let here = VoxelPosition3::new(Vector3::new(4, 0, 0));
    /// let next = VoxelPosition3::new(Vector3::new(5, 0, 0));
    ///
    /// // The shared face, derived from either side.
    /// let from_here = world.at_position(here).between(world.at_position(next));
    /// let from_next = world.at_position(next).between(world.at_position(here));
    ///
    /// assert_eq!(from_here, from_next);
    /// ```
    ///
    /// Also for anything symmetric between two parties: a trade, a shared border, a
    /// handshake.
    pub fn between(self, other: Self) -> Self {
        // Sorting the pair before folding is what makes it symmetric. Adding or
        // exclusive-oring the two would also be symmetric and would also collide:
        // every pair summing to the same total would share a seed.
        let (low, high): (u128, u128) = if self.0 <= other.0 {
            (self.0, other.0)
        } else {
            (other.0, self.0)
        };

        Self(absorb128(mix128(low ^ PAIR_TAG), high))
    }

    /// The seed for this thing at a particular tick.
    ///
    /// # Question
    ///
    /// "What seed does this use *this* tick?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::time::Tick;
    /// let spawner = Seed::from_integer(7u64);
    ///
    /// let now = spawner.at_tick(Tick::new(1_000));
    /// let later = spawner.at_tick(Tick::new(1_001));
    ///
    /// assert_ne!(now, later);
    /// ```
    ///
    /// # Why not just [`Seed::index`]
    ///
    /// Because `index` is already spoken for. A chunk's fifth child and tick five are
    /// different questions, and folding both through `index` would give them the same
    /// answer — content would correlate with the clock for no reason anybody could
    /// find. A tag of its own keeps the two apart.
    pub fn at_tick(self, tick: Tick) -> Self {
        Self(absorb128(self.0 ^ TICK_TAG, expand_u64(tick.count())))
    }

    /// The seed shared by everything in the same region of a given size.
    ///
    /// # Question
    ///
    /// "What seed does the whole area around this position share?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// let world = Seed::from_integer(3u64);
    /// let region = |x, y, z| {
    ///     world.at_region(VoxelPosition3::new(Vector3::new(x, y, z)), 64)
    /// };
    ///
    /// // Everything inside one 64-voxel cube agrees — that is what makes a biome.
    /// assert_eq!(region(0, 0, 0), region(63, 63, 63));
    /// assert_ne!(region(0, 0, 0), region(64, 0, 0));
    /// ```
    ///
    /// # The trap this exists to absorb
    ///
    /// Finding the containing region means dividing by the size and rounding *down*,
    /// and Rust's `/` rounds towards zero. So `-1 / 64` is `0`, putting the voxel at
    /// `-1` in the same region as the one at `0` while `-65` lands two regions away —
    /// a seam one voxel wide on the negative side of every axis, which is exactly the
    /// kind of bug that only shows up once a player walks west of the origin.
    ///
    /// This uses floor division, so the regions tile the world evenly through zero.
    /// A size of zero gives the position itself, there being no region to speak of.
    pub fn at_region(self, position: VoxelPosition3, size: u64) -> Self {
        if size == 0 {
            return self.at_position(position);
        }

        let size: i128 = size as i128;
        let region: [i128; 3] = position
            .to_array()
            .map(|coordinate| coordinate.div_euclid(size));

        Self(absorb128(self.0 ^ REGION_TAG, size as u128)).at(region)
    }

    /// Mixes two seeds into one that depends on both, in order.
    ///
    /// # Question
    ///
    /// "What seed belongs to this **ordered** pair — this one acting on that one?"
    ///
    /// The order is part of the question. A creature biting a wall is not a wall
    /// biting a creature, and the two should not share a seed.
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// let attacker = Seed::from_integer(1u64).child("wolf");
    /// let target = Seed::from_integer(1u64).child("door");
    ///
    /// // Who acts on whom matters.
    /// assert_ne!(attacker.combine(target), target.combine(attacker));
    ///
    /// // And a thing paired with itself is still its own question.
    /// assert_ne!(attacker.combine(attacker), attacker);
    /// ```
    ///
    /// # Why there is no separate `ordered_pair`
    ///
    /// This **is** the ordered pair, and [`Seed::between`] is the unordered one. The
    /// two together cover both cases, so a third name would only be a synonym for one
    /// of them — and a caller who reached for the wrong synonym would get a working
    /// program with a subtly wrong world.
    ///
    /// # Why not exclusive-or
    ///
    /// `self` is mixed before `other` is folded in, so the two are not
    /// interchangeable: `a.combine(b)` and `b.combine(a)` differ, and
    /// `a.combine(a)` is neither `a` nor a constant. Xor-ing them together
    /// instead would give all three away, since xor cannot tell an argument's
    /// position and cancels a value against itself.
    pub fn combine(self, other: Self) -> Self {
        Self(absorb128(mix128(self.0 ^ COMBINE_TAG), other.0))
    }

    /// Separates a field from its neighbours by a fixed tag, without mixing.
    ///
    /// A bare xor, so it costs nothing and can be applied inside a sampling
    /// loop. That is safe here only because everything downstream mixes: the
    /// tag's job is to make two fields disagree about where to start, and the
    /// hash that follows is what spreads the difference. Use [`Seed::mix_in`]
    /// for a value that has to stand on its own.
    pub const fn domain(self, tag: u128) -> Self {
        Self(self.0 ^ tag)
    }

    /// Mixes in any further value: a tag of your own, a version number, a
    /// checksum of the settings that shaped the world.
    pub fn mix_in(self, value: u128) -> Self {
        Self(absorb128(self.0, value))
    }

    /// The next seed in a run, for walking a sequence without holding an index.
    ///
    /// Adds a fixed odd Weyl increment modulo `2^128`, so the walk is proven to
    /// visit every 128-bit state exactly once before repeating.
    ///
    /// Order-dependent by nature, so prefer [`Seed::index`] or [`Seed::at`] for
    /// anything that has to survive a reload.
    pub fn advance(self) -> Self {
        Self(self.0.wrapping_add(GOLDEN_GAMMA_128))
    }

    /// The seed `steps` places further along the walk.
    ///
    /// # Question
    ///
    /// "What seed is this one, `n` steps on?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// let start = Seed::from_integer(1u64);
    ///
    /// assert_eq!(start.advance_by(3), start.advance().advance().advance());
    /// assert_eq!(start.advance_by(0), start);
    /// ```
    ///
    /// # Why this is one multiplication and not a loop
    ///
    /// [`Seed::advance`] adds a fixed stride, so `n` steps add `n` strides — and
    /// wrapping multiplication gives that directly. Skipping a million places costs
    /// the same as skipping one, which is what makes it usable for seeking into a
    /// stream rather than replaying it.
    pub fn advance_by(self, steps: u64) -> Self {
        Self(
            self.0
                .wrapping_add(GOLDEN_GAMMA_128.wrapping_mul(steps as u128)),
        )
    }

    /// The seed for one of the directions out of a place.
    ///
    /// # Question
    ///
    /// "What seed does the thing leaving this voxel *this way* use?"
    ///
    /// The number is a face or neighbour index, and this crate does not fix the
    /// convention: six axis-aligned faces, twenty-six including diagonals, or four
    /// compass directions all work, as long as one world uses one numbering.
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// let world = Seed::from_integer(1u64);
    /// let here = world.at_position(VoxelPosition3::new(Vector3::new(3, 4, 5)));
    ///
    /// // A corridor leaving east and one leaving west decide independently.
    /// assert_ne!(here.at_direction(0), here.at_direction(1));
    /// ```
    ///
    /// # Why not [`Seed::between`]
    ///
    /// Both concern a relationship between two neighbouring places, and the choice
    /// between them is whether the two sides should agree.
    ///
    /// [`Seed::between`] is symmetric: both voxels compute the same seed for the wall
    /// they share, which is what a shared feature needs — carve from one side and the
    /// other side sees the same hole. `at_direction` is not: it belongs to the voxel
    /// looking outward, so a thing growing east and its eastern neighbour's thing
    /// growing west are separate decisions. Use this one for what leaves a place, and
    /// [`Seed::between`] for what two places own jointly.
    pub fn at_direction(self, direction: u32) -> Self {
        Self(absorb128(
            self.0 ^ DIRECTION_TAG,
            expand_u64(u64::from(direction)),
        ))
    }

    /// The seed for one axis of a value that has one number per axis.
    ///
    /// # Question
    ///
    /// "What seed does the *x* component of this use, as against the *y*?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// let wind = Seed::from_integer(9u64).child("wind");
    ///
    /// // A displacement vector whose components must not move together.
    /// let offsets: Vec<Seed> = (0..3).map(|axis| wind.at_axis(axis)).collect();
    ///
    /// assert_ne!(offsets[0], offsets[1]);
    /// assert_ne!(offsets[1], offsets[2]);
    /// ```
    ///
    /// # Why not [`Seed::at_direction`] or [`Seed::index`]
    ///
    /// An axis is unsigned and a direction is not: *x* is one axis, while east and
    /// west are two directions along it. Deriving a vector's three components from
    /// `at_direction(0..3)` would work but would quietly spend half the direction
    /// numbering, and a later `at_direction(1)` for "west" would collide with the
    /// *y* component.
    ///
    /// Against [`Seed::index`]: an index picks one of several *things*, where an axis
    /// picks one part of a single thing. Keeping them apart means a vector's
    /// components cannot collide with a list's elements.
    pub fn at_axis(self, axis: u32) -> Self {
        Self(absorb128(self.0 ^ AXIS_TAG, expand_u64(u64::from(axis))))
    }

    /// The seed for one output field of a generator that produces several.
    ///
    /// # Question
    ///
    /// "What seed does the moisture field use, as against the height field?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// const HEIGHT: u32 = 0;
    /// const MOISTURE: u32 = 1;
    ///
    /// let world = Seed::from_integer(4u64);
    /// let here = VoxelPosition3::new(Vector3::new(12, 0, 7));
    ///
    /// // Two fields over the same ground, which must not be copies of each other.
    /// let height = world.at_channel(HEIGHT).at_position(here);
    /// let moisture = world.at_channel(MOISTURE).at_position(here);
    ///
    /// assert_ne!(height, moisture);
    /// ```
    ///
    /// # Why not [`Seed::child`]
    ///
    /// [`Seed::child`] takes a name and would do this perfectly well —
    /// `child("moisture")` is clearer than `at_channel(1)` and is the better choice
    /// when the fields are known when the code is written.
    ///
    /// This exists for when they are not: a shader, a table or a config file that
    /// names its outputs by number, or a loop over `0..channel_count`. Hashing a
    /// name per iteration to get an integer back is wasteful, and
    /// `child(&n.to_string())` makes the derivation depend on how the number was
    /// formatted.
    pub fn at_channel(self, channel: u32) -> Self {
        Self(absorb128(
            self.0 ^ CHANNEL_TAG,
            expand_u64(u64::from(channel)),
        ))
    }

    /// The seed for the `n`th try at something that may be rejected.
    ///
    /// # Question
    ///
    /// "The first placement did not fit. What seed does the retry use?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// let spot = Seed::from_integer(3u64).child("boulder");
    ///
    /// // Retrying with the same seed would propose the same rejected candidate for
    /// // ever, so each attempt has to ask a different question.
    /// let mut tries = (0..8).map(|attempt| spot.at_attempt(attempt));
    ///
    /// assert_ne!(tries.next(), tries.next());
    /// ```
    ///
    /// # Why this is not optional
    ///
    /// Rejection sampling is the one place where reusing a seed does not merely
    /// correlate two results, it hangs: a rejected candidate derived from a fixed
    /// seed is rejected again every time. Either the attempt number enters the
    /// derivation or the loop cannot terminate.
    ///
    /// # Why not [`Seed::advance`]
    ///
    /// [`Seed::advance`] also walks to a fresh seed and is cheaper. It is a good fit
    /// inside a single call that loops and then discards its position. Prefer this
    /// one when the attempt number is *known* — reconstructed from saved state, or
    /// needed again later — because a derivation from a number can be recomputed out
    /// of order, where a walk has to be replayed from its start.
    pub fn at_attempt(self, attempt: u32) -> Self {
        Self(absorb128(
            self.0 ^ ATTEMPT_TAG,
            expand_u64(u64::from(attempt)),
        ))
    }

    /// The seed for one step of an iterative process.
    ///
    /// # Question
    ///
    /// "What seed does step 400 of this walk, growth or relaxation use?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// let river = Seed::from_integer(11u64).child("river");
    ///
    /// // Jump straight to the thousandth step without walking there.
    /// assert_ne!(river.at_step(1_000), river.at_step(999));
    /// assert_eq!(river.at_step(1_000), river.at_step(1_000));
    /// ```
    ///
    /// # Why not [`Seed::at_tick`]
    ///
    /// A tick is a point in the world's clock, shared by everything alive in it. A
    /// step counts iterations inside one computation, and several of them usually
    /// happen within a single tick. A cellular automaton run for sixty steps during
    /// tick seven needs both, and they must not collide.
    ///
    /// # Why not [`Seed::advance_by`]
    ///
    /// [`Seed::advance_by`] reaches step `n` just as cheaply. The difference is that
    /// a walk is a sequence from one starting point, so `a.advance_by(2)` may equal
    /// `b.advance_by(1)` for a `b` further back along the same line, while
    /// `at_step` keeps separate processes separate regardless of their counters.
    pub fn at_step(self, step: u64) -> Self {
        Self(absorb128(self.0 ^ STEP_TAG, expand_u64(step)))
    }

    /// The seed for one pass of a multi-pass generator.
    ///
    /// # Question
    ///
    /// "What seed does the decoration pass use, as against the carving pass that ran
    /// over the same chunk before it?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// const CARVE: u32 = 0;
    /// const DECORATE: u32 = 1;
    ///
    /// let chunk = Seed::from_integer(6u64)
    ///     .at_position(VoxelPosition3::new(Vector3::new(0, 0, 0)));
    ///
    /// // The same ground, visited twice for different reasons.
    /// assert_ne!(chunk.at_pass(CARVE), chunk.at_pass(DECORATE));
    /// ```
    ///
    /// # On "phase"
    ///
    /// A phase of generation and a pass over the data are the same idea, and this is
    /// the method for both. A second name would add no separation that a second
    /// number does not already give, and would invite the mistake of using one name
    /// in the writer and the other in the reader.
    ///
    /// # Why not [`Seed::at_step`]
    ///
    /// Passes are a fixed, small, named list, and they usually differ in *kind*:
    /// carving and decorating are not two iterations of one procedure. Steps are a
    /// run of the same operation. Keeping them apart means a thirty-step erosion pass
    /// numbered 2 cannot collide with pass 2 of the pipeline.
    pub fn at_pass(self, pass: u32) -> Self {
        Self(absorb128(self.0 ^ PASS_TAG, expand_u64(u64::from(pass))))
    }

    /// The seed for one rule of a grammar or rewrite system.
    ///
    /// # Question
    ///
    /// "Rule 12 matched here. What seed do its random choices use?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// let site = Seed::from_integer(8u64)
    ///     .at_position(VoxelPosition3::new(Vector3::new(2, 9, 2)));
    ///
    /// // Two rules that could both fire here choose independently, so adding a rule
    /// // does not disturb what the others do.
    /// assert_ne!(site.at_rule(12), site.at_rule(13));
    /// ```
    ///
    /// # Why this helps a grammar in particular
    ///
    /// A rule set grows: rules are added, reordered and removed while a world is
    /// being tuned. Deriving each rule's randomness from its own identifier means an
    /// edit to one rule leaves the rest of the world alone, which is what makes the
    /// tuning loop usable at all. Deriving from position alone would make every rule
    /// at one site share a stream, so inserting a rule would shift all of them.
    ///
    /// Prefer a stable identifier over the rule's index in a list, for the same
    /// reason: an index changes when something is inserted above it.
    pub fn at_rule(self, rule: u32) -> Self {
        Self(absorb128(self.0 ^ RULE_TAG, expand_u64(u64::from(rule))))
    }

    /// The seed for one level of detail.
    ///
    /// # Question
    ///
    /// "What seed does this region use when it is being built coarsely, for a distant
    /// view?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// let region = Seed::from_integer(13u64)
    ///     .at_position(VoxelPosition3::new(Vector3::new(64, 0, 64)));
    ///
    /// assert_ne!(region.at_detail(0), region.at_detail(3));
    /// ```
    ///
    /// # Why not [`Seed::at_level`]
    ///
    /// These are easy to confuse because they are usually both small integers
    /// counting the same way, and in many worlds they even coincide.
    ///
    /// They answer different questions. [`Seed::at_level`] names a **depth in the
    /// octree**: where a node sits in the structure, which is a fact about the world.
    /// Detail names **how finely something is being rendered or generated right
    /// now**, which is a fact about the viewer. One node can be built at several
    /// levels of detail over its life without ever changing depth, and two nodes at
    /// different depths can be drawn at the same detail. Deriving both from one lane
    /// would tie a camera distance to a tree position.
    ///
    /// If detail in your world *is* octree depth, use [`Seed::at_level`] and let this
    /// one alone — a single lane is better than two that must be kept in step.
    pub fn at_detail(self, detail: u32) -> Self {
        Self(absorb128(
            self.0 ^ DETAIL_TAG,
            expand_u64(u64::from(detail)),
        ))
    }

    /// The seed for one epoch: an era of the world, or a regeneration of it.
    ///
    /// # Question
    ///
    /// "The world has been reshaped for the third time. What seed does this place use
    /// now?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// let world = Seed::from_integer(21u64);
    /// let here = VoxelPosition3::new(Vector3::new(5, 0, 5));
    ///
    /// // The same ground, two ages apart.
    /// assert_ne!(
    ///     world.at_epoch(0).at_position(here),
    ///     world.at_epoch(1).at_position(here),
    /// );
    /// ```
    ///
    /// # Why not [`Seed::at_tick`]
    ///
    /// Both are time, at wildly different scales, and the scale is the point. A tick
    /// is one update; an epoch spans however many ticks an age lasts. Advancing an
    /// epoch is meant to be a *visible* reshaping of everything, where advancing a
    /// tick should leave the world recognisable. Separate lanes let a thing ask "what
    /// age is this?" and "what moment is this?" independently.
    ///
    /// # Why not [`Seed::at_version`]
    ///
    /// An epoch changes **inside** one world's story: the players saw it happen.
    /// A version changes because the *code* changed, which no character in the world
    /// experienced. Keeping them apart means a content update does not look like an
    /// in-world cataclysm, and vice versa.
    pub fn at_epoch(self, epoch: u64) -> Self {
        Self(absorb128(self.0 ^ EPOCH_TAG, expand_u64(epoch)))
    }

    /// The seed for one version of the thing generating from it.
    ///
    /// # Question
    ///
    /// "The cave algorithm changed. How do I get new caves without disturbing the
    /// terrain the old one sat in?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// const CAVE_VERSION: u32 = 2;
    ///
    /// let world = Seed::from_integer(31u64);
    /// let here = VoxelPosition3::new(Vector3::new(8, 8, 8));
    ///
    /// let caves = world.child("caves").at_version(CAVE_VERSION).at_position(here);
    /// let terrain = world.child("terrain").at_position(here);
    ///
    /// // Bumping CAVE_VERSION moves the caves and leaves `terrain` exactly as it was.
    /// assert_ne!(caves, terrain);
    /// ```
    ///
    /// # Why a lane of its own
    ///
    /// Without it, the only ways to reshuffle one generator are to change the world
    /// seed, which moves *everything*, or to rename it, which is a source edit that
    /// reads as a mistake later. A version lane makes "this part of the world is
    /// generated differently now" an ordinary, local, documented change.
    ///
    /// This is a version of **your generator**, chosen by you, and is unrelated to
    /// [`SEED_ALGORITHM_VERSION`], which versions the derivation machinery underneath
    /// and is not yours to set.
    pub fn at_version(self, version: u32) -> Self {
        Self(absorb128(
            self.0 ^ VERSION_TAG,
            expand_u64(u64::from(version)),
        ))
    }

    /// The seed for a value in a lane of your own.
    ///
    /// # Question
    ///
    /// "My world has a kind of thing this crate never heard of. Where do I put it?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::{Seed, domain_tag};
    /// const WEATHER_FRONT: u128 = domain_tag("my_crate::weather_front");
    ///
    /// let world = Seed::from_integer(1u64);
    ///
    /// assert_ne!(world.derive_tagged(WEATHER_FRONT, 4), world.index(4));
    /// assert_eq!(
    ///     world.derive_tagged(WEATHER_FRONT, 4),
    ///     world.derive_tagged(WEATHER_FRONT, 4),
    /// );
    /// ```
    ///
    /// Build `tag` with [`domain_tag`]; a zero tag leaves the lane untagged, which
    /// merges it with the plain derivations.
    ///
    /// # Why not [`Seed::domain`] followed by [`Seed::index`]
    ///
    /// That composition is nearly this method, and is fine. The difference is that
    /// [`Seed::index`]'s lane is shared with every other indexed thing, so the tag is
    /// doing all the separating on its own, through a bare exclusive-or that a
    /// carefully chosen index could partly undo. Here the tag and the value are
    /// folded by the same hash, so neither can be unpicked from the other.
    ///
    /// # Why not [`Seed::mix_in`]
    ///
    /// [`Seed::mix_in`] folds in a value with no lane at all, which is right for
    /// something there is only one of — a checksum of the world settings, say. Use
    /// this when the value is one of many of a kind, and the kind needs to stay clear
    /// of other kinds.
    pub fn derive_tagged(self, tag: u128, value: u64) -> Self {
        Self(absorb128(self.0 ^ tag, expand_u64(value)))
    }

    /// The seed for one value of a type that names its own lane.
    ///
    /// # Question
    ///
    /// "What seed does *this variant* of my enum use?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::random::seed::{Seed, SeedDomain};
    /// use voxel_world::seed_domain;
    ///
    /// #[derive(Copy, Clone)]
    /// enum Ore {
    ///     Iron,
    ///     Copper,
    /// }
    ///
    /// seed_domain!(Ore => "my_crate::ore");
    ///
    /// let world = Seed::from_integer(5u64);
    ///
    /// assert_ne!(world.derive(Ore::Iron), world.derive(Ore::Copper));
    /// ```
    ///
    /// See [`SeedDomain`] for how a type claims a lane, and [`Seed::derive_tagged`]
    /// for the same thing without defining a type.
    pub fn derive<D: SeedDomain>(self, domain: D) -> Self {
        Self(absorb128(self.0 ^ D::TAG, expand_u64(domain.payload())))
    }
}

// ---------------------------------------------------------------------------
// Using one
// ---------------------------------------------------------------------------
