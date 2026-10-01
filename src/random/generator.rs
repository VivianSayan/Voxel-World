//! The generator itself: its state, how it is started, and the raw draws.
//!
//! Everything here is about producing uniform words and nothing about what to ask
//! of them — [`questions`](super::questions) holds that. A fixed algorithm that
//! should rarely need to change.

use crate::math::UnitQuaternion;
use crate::random::distributions::UniformRotation;
use crate::random::mixing::mix128;
use crate::random::seed::Seed;
use crate::random::source::RandomSource;
use crate::spatial::VoxelPosition3;
use crate::units::{NoiseValue, Probability, UniformNoise, UniformProbability, Unit, UnitValue};

/// A seeded xoshiro256** stream for simulation, not cryptography.
///
/// Integer samplers avoid platform transcendental differences. Samplers using
/// `ln`, `exp`, `powf` or trigonometry do not promise cross-platform bit-for-bit
/// equality. Reproducibility and distribution accuracy are separate properties:
/// for example, rare-event rational geometric sampling is reproducible but uses
/// a fixed-point approximation. See each sampler's contract.
///
/// The state is four words, expanded from the complete 128-bit seed by two
/// domain-separated permutations, plus the second of the last pair of normals:
/// the polar method makes them two at a time, so one is kept for the following
/// call. `Clone` gives an independent stream that replays the same words from
/// this point.
#[derive(Clone, Debug)]
pub struct Random {
    seed: Seed,
    state: [u64; 4],
    spare_normal: Option<f64>,
}

/// Version of the xoshiro transition and 128-bit seed-expansion format.
///
/// Increment this before intentionally changing either algorithm. Saved
/// [`RandomState`] values reject versions they do not understand rather than
/// silently replaying a different stream.
///
/// # What this does *not* cover
///
/// Only the stream of words. Changing a **sampler** — how those words become a
/// binomial count, a direction or a gamma variate — leaves this alone, because the
/// words themselves are unchanged and a saved [`RandomState`] still resumes exactly
/// where it left off. That belongs to
/// [`DISTRIBUTION_ALGORITHM_VERSION`](crate::random::DISTRIBUTION_ALGORITHM_VERSION)
/// instead.
///
/// Bumping this one for a sampler change would be worse than doing nothing: every
/// stored stream would be rejected, to describe a change that did not affect any of
/// them.
pub const RANDOM_ALGORITHM_VERSION: u32 = 1;

/// Byte length of [`RandomState::to_bytes`].
pub const RANDOM_STATE_BYTES: usize = 61;

/// A versioned, portable snapshot of a [`Random`] stream.
///
/// It includes the cached second normal because omitting it would restore the
/// integer stream correctly while changing the next normal draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomState {
    version: u32,
    seed: Seed,
    words: [u64; 4],
    spare_normal: Option<f64>,
}

impl RandomState {
    /// Builds a checked state from decoded storage fields.
    ///
    /// Rejects unknown versions, xoshiro's locked all-zero state, and a
    /// non-finite cached normal.
    ///
    /// # Question
    ///
    /// "How do I rebuild a generator's state from fields I stored myself?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::seed::Seed;
    /// // Rejected unless the version and fields are consistent.
    /// # use voxel_world::random::RandomState;
    /// let restored = RandomState::from_parts(0, Seed::from_integer(1u64), [1, 2, 3, 4], None);
    /// # let _ = restored;
    /// ```
    pub fn from_parts(
        version: u32,
        seed: Seed,
        words: [u64; 4],
        spare_normal: Option<f64>,
    ) -> Option<Self> {
        (version == RANDOM_ALGORITHM_VERSION
            && words != [0; 4]
            && spare_normal.is_none_or(f64::is_finite))
        .then_some(Self {
            version,
            seed,
            words,
            spare_normal,
        })
    }

    /// Random algorithm/data-format version carried by this state.
    pub const fn version(self) -> u32 {
        self.version
    }

    /// Seed from which the stream was originally constructed.
    pub const fn seed(self) -> Seed {
        self.seed
    }

    /// Four current xoshiro state words.
    pub const fn words(self) -> [u64; 4] {
        self.words
    }

    /// Cached second normal, if the last normal draw produced a pair.
    pub const fn spare_normal(self) -> Option<f64> {
        self.spare_normal
    }

    /// Encodes the state as fixed-layout little-endian bytes.
    ///
    /// Layout: version (4), seed (16), four words (32), spare-present flag
    /// (1), and spare bits (8). The spare bits are zero when absent.
    ///
    /// # Question
    ///
    /// "How do I write this generator into a save file?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let saved = random.snapshot().to_bytes();
    /// assert!(!saved.is_empty());
    /// ```
    pub fn to_bytes(self) -> [u8; RANDOM_STATE_BYTES] {
        let mut bytes = [0u8; RANDOM_STATE_BYTES];
        bytes[0..4].copy_from_slice(&self.version.to_le_bytes());
        bytes[4..20].copy_from_slice(&self.seed.value().to_le_bytes());
        for (index, word) in self.words.into_iter().enumerate() {
            let start = 20 + index * 8;
            bytes[start..start + 8].copy_from_slice(&word.to_le_bytes());
        }
        if let Some(spare) = self.spare_normal {
            bytes[52] = 1;
            bytes[53..61].copy_from_slice(&spare.to_bits().to_le_bytes());
        }
        bytes
    }

    /// Decodes bytes written by [`RandomState::to_bytes`].
    ///
    /// # Question
    ///
    /// "How do I read a generator back out of a save file?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let saved = random.snapshot().to_bytes();
    /// # use voxel_world::random::RandomState;
    /// let restored = RandomState::from_bytes(&saved).expect("our own bytes");
    /// # let _ = restored;
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes: &[u8; RANDOM_STATE_BYTES] = bytes.try_into().ok()?;
        let version = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let seed = Seed::from_raw(u128::from_le_bytes(bytes[4..20].try_into().ok()?));
        let mut words = [0u64; 4];
        for (index, word) in words.iter_mut().enumerate() {
            let start = 20 + index * 8;
            *word = u64::from_le_bytes(bytes[start..start + 8].try_into().ok()?);
        }
        let spare_normal = match bytes[52] {
            0 if bytes[53..61] == [0; 8] => None,
            1 => Some(f64::from_bits(u64::from_le_bytes(
                bytes[53..61].try_into().ok()?,
            ))),
            _ => return None,
        };
        Self::from_parts(version, seed, words, spare_normal)
    }
}

impl Random {
    /// A generator started from a seed, at the beginning of its stream.
    ///
    /// # Question
    ///
    /// "How do I start a generator from a seed?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// let mut fresh = Random::new(Seed::from_integer(42u64));
    /// let first = fresh.next_u64();
    /// // The same seed always starts the same stream.
    /// assert_eq!(Random::new(Seed::from_integer(42u64)).next_u64(), first);
    /// ```
    pub fn new(seed: Seed) -> Self {
        let mut random = Self {
            seed,
            state: [0; 4],
            spare_normal: None,
        };

        random.reseed(seed);
        random
    }

    /// Restarts the generator from a new seed, discarding where it had got to.
    ///
    /// Two domain-separated 128-bit permutations expand the complete seed into
    /// the four words xoshiro256** needs. The first permutation alone is
    /// bijective, so two distinct seeds cannot initialize the same stream.
    /// An all-zero state is the one state xoshiro cannot leave and is guarded
    /// against defensively.
    ///
    /// # Question
    ///
    /// "How do I point this generator at a different seed without making a new one?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// random.reseed(Seed::from_integer(7u64));
    /// assert_eq!(random.next_u64(), Random::new(Seed::from_integer(7u64)).next_u64());
    /// ```
    pub fn reseed(&mut self, seed: Seed) {
        self.seed = seed;
        self.spare_normal = None;

        // Deliberately distinct from every seed-derivation and cursor tag.
        const FIRST_DOMAIN: u128 = 0xCBBB_9D5D_C105_9ED8_629A_292A_367C_D507;
        const SECOND_DOMAIN: u128 = 0x9159_015A_3070_DD17_152F_ECD8_F70E_5939;
        let first = mix128(seed.value() ^ FIRST_DOMAIN);
        let second = mix128(seed.value() ^ SECOND_DOMAIN);
        self.state = [
            first as u64,
            (first >> 64) as u64,
            second as u64,
            (second >> 64) as u64,
        ];

        if self.state == [0; 4] {
            self.state[0] = 1;
        }
    }

    /// Restores a stream from a checked, versioned snapshot.
    ///
    /// # Question
    ///
    /// "How do I resume a generator exactly where a save left off?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let saved = random.snapshot();
    /// let expected = random.next_u64();
    /// let mut resumed = Random::from_state(saved).expect("our own state");
    /// assert_eq!(resumed.next_u64(), expected, "the replay matches");
    /// ```
    pub fn from_state(state: RandomState) -> Option<Self> {
        (state.version == RANDOM_ALGORITHM_VERSION && state.words != [0; 4]).then_some(Self {
            seed: state.seed,
            state: state.words,
            spare_normal: state.spare_normal,
        })
    }

    /// Captures everything required to resume this exact stream.
    pub const fn snapshot(&self) -> RandomState {
        RandomState {
            version: RANDOM_ALGORITHM_VERSION,
            seed: self.seed,
            words: self.state,
            spare_normal: self.spare_normal,
        }
    }

    /// The seed this generator was started from, however far along it now is.
    ///
    /// # Question
    ///
    /// "Which seed was this generator started from?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let started = random.seed();
    /// random.next_u64();
    /// assert_eq!(random.seed(), started, "drawing does not change it");
    /// ```
    pub fn seed(&self) -> Seed {
        self.seed
    }

    /// Builds a new generator for a single voxel position.
    ///
    /// The result depends only on this generator's seed and the position, so a
    /// given world seed always produces the same stream for the same voxel,
    /// whatever order positions are visited in. This generator's own state is
    /// left untouched, which is what makes chunk generation order independent.
    ///
    /// # Question
    ///
    /// "What generator belongs to this one voxel, independently of every other?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// let here = VoxelPosition3::new(Vector3::new(10, 64, -3));
    /// let mut local = random.random_from_position(here);
    /// // Depends only on the world seed and the coordinate, so neighbouring
    /// // chunks agree without talking to each other.
    /// let value = local.next_u64();
    /// assert_eq!(random.random_from_position(here).next_u64(), value);
    /// ```
    pub fn random_from_position(&self, position: VoxelPosition3) -> Random {
        Random::new(self.seed_from_position(position))
    }

    /// As [`Random::random_from_position`], from loose coordinates.
    ///
    /// # Question
    ///
    /// "The same, when I have loose x, y, z rather than a position type?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let mut local = random.random_from_coordinates(10, 64, -3);
    /// # let _ = local.next_u64();
    /// ```
    pub fn random_from_coordinates(&self, x: i128, y: i128, z: i128) -> Random {
        Random::new(self.seed_from_coordinates(x, y, z))
    }

    /// The seed [`Random::random_from_position`] would use, without building
    /// the generator.
    ///
    /// # Question
    ///
    /// "What seed does this voxel have, when I only want the seed?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// # use voxel_world::spatial::VoxelPosition3;
    /// # use voxel_world::math::Vector3;
    /// let seed = random.seed_from_position(VoxelPosition3::new(Vector3::new(1, 2, 3)));
    /// // Cheaper than building a whole generator for one decision.
    /// # let _ = seed;
    /// ```
    pub fn seed_from_position(&self, position: VoxelPosition3) -> Seed {
        self.seed.at(position.to_array())
    }

    /// As [`Random::seed_from_position`], from loose coordinates.
    ///
    /// Both defer to [`Seed::at`], so a position seeded through a generator and
    /// one seeded straight from a [`Seed`] agree, and there is only one place
    /// that decides how a coordinate reaches a seed.
    ///
    /// # Question
    ///
    /// "The same, from loose coordinates?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let seed = random.seed_from_coordinates(1, 2, 3);
    /// # let _ = seed;
    /// ```
    pub fn seed_from_coordinates(&self, x: i128, y: i128, z: i128) -> Seed {
        self.seed.at([x, y, z])
    }

    /// An even coin, from one bit of a draw.
    ///
    /// # Question
    ///
    /// "Heads or tails?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// if random.next_bool() { /* heads */ }
    /// ```
    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() & 1) == 1
    }

    /// Uniform over every `u16`, taken from the top of a draw, where
    /// xoshiro256**'s bits are strongest.
    ///
    /// # Question
    ///
    /// "What is a random `u16`?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let value = random.next_u16();
    /// # let _ = value;
    /// ```
    pub fn next_u16(&mut self) -> u16 {
        (self.next_u64() >> 48) as u16
    }

    /// Uniform over every `u32`, taken from the top half of a draw.
    ///
    /// # Question
    ///
    /// "What is a random `u32`?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let value = random.next_u32();
    /// # let _ = value;
    /// ```
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform over every `u64`: one step of xoshiro256**, and the draw every
    /// other method here is built from.
    ///
    /// The output is the `**` scrambler, a multiply, rotate and multiply of one
    /// state word, computed before the state advances. The period is
    /// `2^256 - 1`.
    ///
    /// # Question
    ///
    /// "What is a random `u64`? (the draw every other method is built on)"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let value = random.next_u64();
    /// # let _ = value;
    /// ```
    pub fn next_u64(&mut self) -> u64 {
        let result: u64 = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);

        let temporary: u64 = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= temporary;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Uniform over the evenly spaced 24-bit grid in `[0, 1)`.
    ///
    /// # Question
    ///
    /// "What is a random fraction between zero and one, as an `f32`?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let fraction = random.next_f32();
    /// assert!((0.0..1.0).contains(&fraction));
    /// ```
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform over the evenly spaced 53-bit grid in `[0, 1)`.
    ///
    /// # Question
    ///
    /// "What is a random fraction between zero and one?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let fraction = random.next_f64();
    /// assert!((0.0..1.0).contains(&fraction));
    /// ```
    pub fn next_f64(&mut self) -> f64 {
        self.unit_f64()
    }

    /// Uniform in `[0, 1)` on the `2^-63` grid, integer-backed.
    ///
    /// Ten bits finer than [`Random::next_f64`], totally ordered, and exact to
    /// compare a chance against. One word, the same as every other fraction here.
    ///
    /// # Question
    ///
    /// "What is a random fraction, held exactly rather than as a float?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let fraction = random.unit();
    /// // Ten bits finer than `next_f64`, and exact to compare a chance against.
    /// assert!(!fraction.is_one(), "a draw is half-open");
    /// ```
    pub fn unit(&mut self) -> Unit {
        <Self as RandomSource>::unit(self)
    }

    /// A uniformly random [`Probability`] in `[0, 1)`.
    ///
    /// The same draw as [`Random::unit`], narrowed to what an `f64` holds. Prefer
    /// `unit` where the value will be compared or stored.
    ///
    /// # Question
    ///
    /// "What is a random chance?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let chance = random.probability();
    /// assert!(chance.value() >= 0.0 && chance.value() < 1.0);
    /// ```
    pub fn probability(&mut self) -> Probability {
        self.sample(&UniformProbability)
    }

    /// A uniformly random [`NoiseValue`] in `[-1, 1)`.
    ///
    /// A uniform sample of the range, not a sample of a noise field.
    ///
    /// # Question
    ///
    /// "What is a random value between minus one and one?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let jitter = random.noise_value();
    /// assert!(jitter.value() >= -1.0 && jitter.value() < 1.0);
    /// ```
    pub fn noise_value(&mut self) -> NoiseValue {
        self.sample(&UniformNoise)
    }

    /// A uniformly random orientation.
    ///
    /// Uniform over rotations rather than over Euler angles, which are not the same
    /// thing; see [`UniformRotation`].
    ///
    /// # Question
    ///
    /// "Which way should this be oriented?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// // A uniformly random orientation, for scattering props.
    /// let facing = random.rotation();
    /// # let _ = facing;
    /// ```
    pub fn rotation(&mut self) -> UnitQuaternion {
        self.sample(&UniformRotation)
    }

    /// A uniform fraction carrying its half-open range in its type, so it can
    /// be scaled or used as an index without a further check.
    ///
    /// # Question
    ///
    /// "What is a random fraction I can index a table with directly?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let fraction = random.unit_value();
    /// let index = fraction.index_of(8);
    /// assert!(index < 8, "always in range, no bounds check needed");
    /// ```
    pub fn unit_value(&mut self) -> UnitValue {
        UnitValue::new(self.unit_f64()).unwrap()
    }

    /// Uniform over every `u128`, from two draws: the first is the high half.
    ///
    /// # Question
    ///
    /// "What is a random `u128`?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let value = random.next_u128();
    /// # let _ = value;
    /// ```
    pub fn next_u128(&mut self) -> u128 {
        let upper: u128 = self.next_u64() as u128;
        let lower: u128 = self.next_u64() as u128;

        (upper << 64) | lower
    }

    /// Uniform over every `i128`, the same bits as
    /// [`next_u128`](Random::next_u128) read as signed.
    ///
    /// # Question
    ///
    /// "What is a random `i128`, positive or negative?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let value = random.next_i128();
    /// # let _ = value;
    /// ```
    pub fn next_i128(&mut self) -> i128 {
        self.next_u128() as i128
    }

    /// Uniform over every `i64`, the same bits as
    /// [`next_u64`](Random::next_u64) read as signed.
    ///
    /// # Question
    ///
    /// "What is a random `i64`, positive or negative?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let value = random.next_i64();
    /// # let _ = value;
    /// ```
    pub fn next_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    /// Uniform over every `i32`, the same bits as
    /// [`next_u32`](Random::next_u32) read as signed.
    ///
    /// # Question
    ///
    /// "What is a random `i32`, positive or negative?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let value = random.next_i32();
    /// # let _ = value;
    /// ```
    pub fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }

    /// Uniform over every `i16`, the same bits as
    /// [`next_u16`](Random::next_u16) read as signed.
    ///
    /// # Question
    ///
    /// "What is a random `i16`, positive or negative?"
    ///
    /// # Example
    ///
    /// ```
    /// # use voxel_world::random::Random;
    /// # use voxel_world::random::seed::Seed;
    /// # let mut random = Random::new(Seed::from_integer(1u64));
    /// let value = random.next_i16();
    /// # let _ = value;
    /// ```
    pub fn next_i16(&mut self) -> i16 {
        self.next_u16() as i16
    }
}

/// The stream as a source for the samplers, keeping the spare normal so that
/// a pair of them costs one rejection loop rather than two.
impl RandomSource for Random {
    fn next_u64(&mut self) -> u64 {
        Random::next_u64(self)
    }

    fn take_spare_normal(&mut self) -> Option<f64> {
        self.spare_normal.take()
    }

    fn keep_spare_normal(&mut self, spare: f64) {
        self.spare_normal = Some(spare);
    }
}
