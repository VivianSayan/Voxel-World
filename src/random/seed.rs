//! Seeds: making one 128-bit value out of whatever the world was given, and
//! branching it into as many independent sub-seeds as the world needs.
//!
//! A world is reproducible only if every random decision traces back to one
//! number by a path that never varies. [`Seed`] is that number, and the methods
//! here are the paths.
//!
//! Two habits keep a seeded world honest, and this type is built around both.
//!
//! **Derive, never reuse.** Handing the same seed to the terrain, the caves and
//! the ore placement correlates them: rivers follow ridges because both read
//! the same stream. Give each its own branch instead, with [`Seed::child`]:
//!
//! ```ignore
//! let world = Seed::parse("Wandering Hill");
//! let terrain = world.child("terrain");
//! let caves = world.child("caves");
//! ```
//!
//! **Derive from data, never from order.** A seed reached by counting chunks as
//! they load depends on which order the player walked. One reached from the
//! chunk's coordinates does not, so it survives saving, reloading and a
//! different route through the world:
//!
//! ```ignore
//! let chunk = terrain.at(position.to_array());
//! ```
//!
//! Every derivation is a pure function of what went into it. There is no
//! counter and no hidden state, so the same call anywhere in the program, in
//! any order, on any thread, gives the same answer.
//!
//! Everything here is integer arithmetic on explicitly little-endian bytes, so
//! a seed derived on one machine matches the same derivation on another.

use crate::math::Fixed;
use crate::structures::traits::Shuffle;
use crate::math::linear::{Vector2, Vector3, Vector4};
use crate::random::Random;
use crate::random::distributions::{Bernoulli, BernoulliRatio, Distribution, WeightedDiscrete};
use crate::random::mixing::{
    GOLDEN_GAMMA, GOLDEN_GAMMA_128, absorb128, expand_u64, fold_u128, hash_bytes, mix128, mix64,
};
use crate::random::source::StochasticSource;
use crate::spatial::{
    Depth, NodePosition2, NodePosition3, NodePosition4, VoxelPosition2, VoxelPosition3,
    VoxelPosition4,
};
use crate::time::Tick;
use crate::units::{Probability, Ratio};
use crate::units::Unit;
use crate::math::UnitQuaternion;
use crate::random::distributions::UniformRotation;
use crate::units::{NoiseValue, UniformNoise, UnitValue};

/// Keeps the ways of deriving a seed from colliding with one another, so that
/// `child("7")`, `index(7)` and `at([7])` are three different seeds.
const CHILD_TAG: u128 = 0x7B7F_AEB8_9C7A_2D31_C4CE_B9FE_1A85_EC53;
const INDEX_TAG: u128 = 0x1F83_D9AB_FB41_BD6B_5BE0_CD19_137E_2179;
const POSITION_TAG: u128 = 0x510E_527F_ADE6_82D1_9B05_688C_2B3E_6C1F;
const COMBINE_TAG: u128 = 0x6A09_E667_F3BC_C908_BB67_AE85_84CA_A73B;
const DRAW_TAG: u128 = 0x3C6E_F372_FE94_F82B_A54F_F53A_5F1D_36F1;
const PAIR_TAG: u128 = 0x2B7E_1516_28AE_D2A6_ABF7_1588_09CF_4F3C;
const TICK_TAG: u128 = 0xA409_3822_299F_31D0_082E_FA98_EC4E_6C89;
const REGION_TAG: u128 = 0x243F_6A88_85A3_08D3_1319_8A2E_0370_7344;
const DEPTH_TAG: u128 = 0xD1B5_4A32_D192_ED03_A4A3_0B20_7C77_8A45;
const OCTAVE_TAG: u128 = 0x4528_21E6_38D0_1377_BE54_66CF_34E9_0C6C;

/// The tags for the semantic lanes below.
///
/// These are built with [`domain_tag`] rather than written out as literals, so the
/// built-in lanes and the ones you define yourself come from the same function. A
/// name is easier to audit than sixteen bytes of hex, and the compile-time check
/// further down proves the function kept them apart.
const DIRECTION_TAG: u128 = domain_tag("voxel_world::seed::direction");
const AXIS_TAG: u128 = domain_tag("voxel_world::seed::axis");
const CHANNEL_TAG: u128 = domain_tag("voxel_world::seed::channel");
const ATTEMPT_TAG: u128 = domain_tag("voxel_world::seed::attempt");
const STEP_TAG: u128 = domain_tag("voxel_world::seed::step");
const PASS_TAG: u128 = domain_tag("voxel_world::seed::pass");
const RULE_TAG: u128 = domain_tag("voxel_world::seed::rule");
const DETAIL_TAG: u128 = domain_tag("voxel_world::seed::detail");
const EPOCH_TAG: u128 = domain_tag("voxel_world::seed::epoch");
const VERSION_TAG: u128 = domain_tag("voxel_world::seed::version");

/// Turns a name into a stable 128-bit domain tag, at compile time.
///
/// # Question
///
/// "What tag should *my* kind of derivation use, so it cannot land on anybody
/// else's?"
///
/// A tag's only job is to keep one kind of question out of another's space. Any
/// distinct 128-bit number does that, but inventing them by hand is a poor job for a
/// person: the numbers carry no meaning, two crates can pick the same one, and
/// nothing reminds you which is taken. Hashing a name solves all three — the name
/// says what the lane is for, namespacing it by crate makes a clash somebody else's
/// impossibility rather than your risk, and the value is derived rather than
/// remembered.
///
/// # Example
///
/// ```
/// use voxel_world::random::seed::{Seed, domain_tag};
///
/// const VEGETATION: u128 = domain_tag("my_crate::vegetation");
/// const MINERALS: u128 = domain_tag("my_crate::minerals");
///
/// assert_ne!(VEGETATION, MINERALS);
///
/// let world = Seed::from_integer(7u64);
///
/// // Two kinds of thing asking about index 3 get two different answers.
/// assert_ne!(
///     world.derive_tagged(VEGETATION, 3),
///     world.derive_tagged(MINERALS, 3),
/// );
/// ```
///
/// # Stability
///
/// The mapping from name to tag is part of the derivation algorithm and moves with
/// [`SEED_ALGORITHM_VERSION`]. The same name gives the same tag on every platform and
/// every build, so a tag may be stored beside a generated world — but renaming the
/// string reshuffles that lane, exactly as changing any other seed input would.
///
/// # Why not a counter
///
/// A registry of small numbers would be shorter, but it only works while one crate
/// owns every lane. Two crates both reaching for `4` would collide, and neither could
/// see it. Names cannot be deduplicated centrally either, but a 128-bit hash of a
/// namespaced name makes an accidental clash far less likely than a mistake in a
/// hand-kept list.
pub const fn domain_tag(name: &str) -> u128 {
    let bytes: &[u8] = name.as_bytes();

    // Two independent accumulators, so the two halves of the result do not agree
    // about what they saw. The low lane is plain FNV-1a; the high lane folds the
    // byte's position in as well, which is what separates anagrams.
    let mut low: u64 = 0xCBF2_9CE4_8422_2325;
    let mut high: u64 = GOLDEN_GAMMA;
    let mut index: usize = 0;

    while index < bytes.len() {
        let byte: u64 = bytes[index] as u64;

        low = (low ^ byte).wrapping_mul(0x0000_0100_0000_01B3);
        high = (high ^ byte.wrapping_add(index as u64)).wrapping_mul(0x0000_0100_0000_01B3);

        index += 1;
    }

    // Folding the length in distinguishes a name from the same name padded, and
    // mix64 is what turns these counters into something with no visible structure.
    let low: u64 = mix64(low ^ (bytes.len() as u64));
    let high: u64 = mix64(high.wrapping_add(bytes.len() as u64));

    ((high as u128) << 64) | (low as u128)
}

/// A named lane for seed derivation, so a set of related kinds can share one.
///
/// # Question
///
/// "I have an enum of kinds of thing. How do I give each of them its own
/// randomness, without writing a tag per variant?"
///
/// The answer is that the *enum* is the lane and the *variant* is the value. One tag
/// covers the whole type, and [`Seed::derive`] separates the variants within it, the
/// same way [`Seed::index`] separates siblings under one parent.
///
/// # Example
///
/// ```
/// use voxel_world::random::seed::{Seed, SeedDomain, domain_tag};
///
/// #[derive(Copy, Clone)]
/// enum Biome {
///     Tundra,
///     Desert,
///     Forest,
/// }
///
/// impl SeedDomain for Biome {
///     const TAG: u128 = domain_tag("my_crate::biome");
///
///     fn payload(&self) -> u64 {
///         *self as u64
///     }
/// }
///
/// let world = Seed::from_integer(1u64);
///
/// assert_ne!(world.derive(Biome::Tundra), world.derive(Biome::Desert));
/// assert_eq!(world.derive(Biome::Forest), world.derive(Biome::Forest));
/// ```
///
/// The [`seed_domain!`](crate::seed_domain) macro writes that impl for a plain
/// C-like enum.
///
/// # Why a trait rather than passing a tag each time
///
/// [`Seed::derive_tagged`] already takes a tag and a value, and is the right tool for
/// a one-off. The trait earns its place when the lane belongs to a *type*: the tag is
/// then written once, next to the type it describes, instead of at every call site
/// where somebody could reach for the wrong constant. It also makes the lane part of
/// the type's public contract, so a caller cannot derive from your enum in the wrong
/// space even by accident.
pub trait SeedDomain {
    /// The tag separating this type's values from every other derivation.
    ///
    /// Build it with [`domain_tag`] from a namespaced name. It must not be zero: a
    /// zero tag leaves the lane untagged and merges it with the plain derivations.
    const TAG: u128;

    /// The value identifying *this* one within the lane.
    ///
    /// Distinct values must give distinct numbers, or they share a seed. For a
    /// C-like enum the discriminant does that for free.
    fn payload(&self) -> u64;
}

/// Implements [`SeedDomain`] for a C-like enum, given a name for its lane.
///
/// # Example
///
/// ```
/// use voxel_world::random::seed::{Seed, SeedDomain};
/// use voxel_world::seed_domain;
///
/// #[derive(Copy, Clone)]
/// enum Decoration {
///     Rocks,
///     Shrubs,
/// }
///
/// seed_domain!(Decoration => "my_crate::decoration");
///
/// let world = Seed::from_integer(2u64);
///
/// assert_ne!(world.derive(Decoration::Rocks), world.derive(Decoration::Shrubs));
/// ```
///
/// The type must be `Copy` and castable with `as u64`, which a C-like enum is. For
/// anything carrying data, write the impl by hand and decide there how the payload is
/// built — that is a choice about which differences matter, and a macro should not
/// make it for you.
#[macro_export]
macro_rules! seed_domain {
    ($kind:ty => $name:literal) => {
        impl $crate::random::seed::SeedDomain for $kind {
            const TAG: u128 = $crate::random::seed::domain_tag($name);

            fn payload(&self) -> u64 {
                *self as u64
            }
        }
    };
}

/// Fails the build if any two derivation tags are equal.
///
/// A tag's only job is to keep one kind of derivation out of another's space, so two
/// of them sharing a value would silently merge two questions into one.
/// `GOLDEN_GAMMA_128` is in the list because [`Seed::advance`] folds it in and it must
/// not match a tag either — `TICK_TAG` was accidentally given its value once, and no
/// behavioural test could see it, because the two are combined by different
/// operations. A compile-time check cannot miss it.
const _TAGS_ARE_DISTINCT: () = {
    let tags: [u128; 21] = [
        CHILD_TAG,
        INDEX_TAG,
        POSITION_TAG,
        COMBINE_TAG,
        DRAW_TAG,
        PAIR_TAG,
        TICK_TAG,
        REGION_TAG,
        DEPTH_TAG,
        OCTAVE_TAG,
        DIRECTION_TAG,
        AXIS_TAG,
        CHANNEL_TAG,
        ATTEMPT_TAG,
        STEP_TAG,
        PASS_TAG,
        RULE_TAG,
        DETAIL_TAG,
        EPOCH_TAG,
        VERSION_TAG,
        GOLDEN_GAMMA_128,
    ];

    let mut outer: usize = 0;

    while outer < tags.len() {
        // A zero tag is not a tag: `state ^ 0` leaves the state alone, so that lane
        // would quietly become the untagged one.
        assert!(
            tags[outer] != 0,
            "a seed derivation tag is zero, which would leave its lane untagged"
        );

        let mut inner: usize = outer + 1;

        while inner < tags.len() {
            assert!(
                tags[outer] != tags[inner],
                "two seed derivation tags share a value, which would merge two \
                 different questions into one"
            );
            inner += 1;
        }

        outer += 1;
    }
};

/// Version of deterministic seed derivation and [`SeedCursor`] word mapping.
///
/// Persist this beside generated-world metadata when exact replay across
/// releases matters, and increment it before intentionally changing a tag,
/// mixer, derivation rule, cursor step, or cursor output mapping.
pub const SEED_ALGORITHM_VERSION: u32 = 2;

/// The alphabet [`Seed::to_code`] writes in: Crockford base32, which leaves out
/// `I`, `L`, `O` and `U` so that nothing in a written-down seed can be mistaken
/// for a digit or for another letter.
const CODE_ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// The integer widths [`Seed::from_integer`] accepts, which is all of them.
///
/// Every width maps to the same bit pattern for the same value, so the type a
/// number arrived in never shows up in the seed. Signed values sign-extend, so
/// `-1` and `u128::MAX` agree; that is the one place two different values meet.
mod sealed {
    pub trait SeedInteger {}
}

/// A primitive integer that can be converted into a [`Seed`] without losing
/// its value or depending on its width.
///
/// This trait is sealed: the crate implements it for every Rust integer type,
/// and external implementations cannot accidentally introduce a different
/// conversion rule.
pub trait SeedInteger: Copy + sealed::SeedInteger {
    /// The value as the 128 bits a seed is built from.
    fn to_seed_bits(self) -> u128;
}

macro_rules! implement_seed_integer {
    ($($integer:ty),+) => {
        $(
            impl SeedInteger for $integer {
                fn to_seed_bits(self) -> u128 {
                    self as i128 as u128
                }
            }

            impl sealed::SeedInteger for $integer {}
        )+
    };
}

// Everything that fits in an `i128` without losing its value, sign-extended so
// that narrow negatives agree with wide ones.
implement_seed_integer!(u8, u16, u32, u64, i8, i16, i32, i64, i128);

impl SeedInteger for u128 {
    /// Taken whole rather than through `i128`, since it does not fit. A value
    /// small enough for both still agrees with the narrower types.
    fn to_seed_bits(self) -> u128 {
        self
    }
}

impl sealed::SeedInteger for u128 {}

impl SeedInteger for usize {
    fn to_seed_bits(self) -> u128 {
        self as u128
    }
}

impl sealed::SeedInteger for usize {}

impl SeedInteger for isize {
    fn to_seed_bits(self) -> u128 {
        self as i128 as u128
    }
}

impl sealed::SeedInteger for isize {}

/// A world seed, and the root of every seed derived from it.
///
/// One `u128` and nothing else: cheap to copy, cheap to compare, and cheap to
/// write down. It carries no cursor and no state, so the same derivation gives
/// the same seed however often it is made, in whatever order, on whatever
/// thread.
///
/// The three groups of methods are the three things a seed is for. *Building*
/// one from whatever the world was given: a number, a name, a file, the clock.
/// *Branching* it, with [`child`](Seed::child), [`at`](Seed::at) and the rest,
/// into as many decorrelated seeds as the world needs. *Using* one, either for
/// a single decision through [`chance`](Seed::chance),
/// [`below`](Seed::below) or [`sample`](Seed::sample), or as the start of a
/// stream through [`to_random`](Seed::to_random).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Seed(u128);

/// A compact, deterministic cursor through words derived from one [`Seed`].
///
/// This is the advancing counterpart to an immutable seed. It retains the
/// origin for inspection while its current position walks a proven full-cycle
/// 128-bit Weyl sequence. Each position is whitened before being folded to a
/// word, so raw or neighboring seeds do not expose structured first draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SeedCursor {
    origin: Seed,
    next: Seed,
}

impl SeedCursor {
    /// Starts at the first word derived from `seed`.
    pub const fn new(seed: Seed) -> Self {
        Self {
            origin: seed,
            next: seed,
        }
    }

    /// Restores a cursor from its original seed and current position.
    ///
    /// Both values are needed because distributions may consume a variable
    /// number of words, so a draw count is not enough to restore every cursor.
    pub const fn from_parts(seed: Seed, position: Seed) -> Self {
        Self {
            origin: seed,
            next: position,
        }
    }

    /// The immutable seed from which this cursor started.
    pub const fn seed(self) -> Seed {
        self.origin
    }

    /// The state from which the next word will be derived.
    pub const fn position(self) -> Seed {
        self.next
    }

    /// Encodes the origin followed by the current position as little-endian
    /// bytes, suitable for a save file carrying
    /// [`SEED_ALGORITHM_VERSION`].
    pub fn to_bytes(self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&self.origin.value().to_le_bytes());
        bytes[16..].copy_from_slice(&self.next.value().to_le_bytes());
        bytes
    }

    /// Decodes bytes written by [`SeedCursor::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes: &[u8; 32] = bytes.try_into().ok()?;
        let origin = Seed::from_raw(u128::from_le_bytes(bytes[..16].try_into().ok()?));
        let position = Seed::from_raw(u128::from_le_bytes(bytes[16..].try_into().ok()?));
        Some(Self::from_parts(origin, position))
    }

    /// Samples one distribution and advances past every word it consumed.
    pub fn sample<D: Distribution>(&mut self, distribution: &D) -> D::Output {
        distribution.sample(self)
    }
}

impl From<Seed> for SeedCursor {
    fn from(seed: Seed) -> Self {
        Self::new(seed)
    }
}

impl StochasticSource for SeedCursor {
    fn next_u64(&mut self) -> u64 {
        let word = fold_u128(mix128(self.next.0 ^ DRAW_TAG));
        self.next = self.next.advance();
        word
    }
}

// ---------------------------------------------------------------------------
// Building one
// ---------------------------------------------------------------------------

impl Seed {
    /// Takes a 128-bit value exactly as given, with no mixing.
    ///
    /// For round-tripping a seed that was already derived, such as one loaded
    /// from a save. To turn a number a player typed into a seed, prefer
    /// [`Seed::from_integer`], which spreads a small or structured value over
    /// the whole width first.
    pub const fn from_raw(value: u128) -> Self {
        Self(value)
    }

    /// Spreads any integer over the full width.
    ///
    /// Takes every integer width, including `u128`, `usize` and `isize`. The
    /// value is what matters, not the type it arrived in: `1u8`, `1i64` and
    /// `1u128` all give the same seed. Negative values are taken as their
    /// two's-complement pattern, so `-1` and `u128::MAX` agree.
    pub fn from_integer(value: impl SeedInteger) -> Self {
        Self(mix128(value.to_seed_bits()))
    }

    /// Hashes text, whatever it says.
    ///
    /// Always hashes, so `from_text("42")` is unrelated to `from_integer(42)`.
    /// Use [`Seed::parse`] for input a player typed, where a number should mean
    /// the number.
    pub fn from_text(text: &str) -> Self {
        Self(hash_bytes(text.as_bytes()))
    }

    /// Hashes raw bytes: a file, a key, anything already in binary.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(hash_bytes(bytes))
    }

    /// Reads whatever a player typed into a seed box.
    ///
    /// A value that parses as a number is used as that number, so typing the
    /// seed printed by another copy of the game reproduces its world. Anything
    /// else is hashed as text. Surrounding whitespace is ignored, and an empty
    /// input hashes as empty text rather than being treated as zero.
    ///
    /// Both a plain decimal number and a [`Seed::to_code`] string are
    /// recognised, so a seed can be shared in either form.
    pub fn parse(input: &str) -> Self {
        let trimmed: &str = input.trim();

        if let Ok(value) = trimmed.parse::<i128>() {
            return Self::from_integer(value);
        }

        if let Ok(value) = trimmed.parse::<u128>() {
            return Self(mix128(value));
        }

        if let Some(seed) = Self::from_code(trimmed) {
            return seed;
        }

        Self::from_text(trimmed)
    }

    /// Hashes a floating point value.
    ///
    /// Zero and negative zero give the same seed, and every NaN gives the same
    /// seed as every other, so a value that compares equal always seeds equal.
    pub fn from_f64(value: f64) -> Self {
        let bits: u64 = if value.is_nan() {
            f64::NAN.to_bits()
        } else if value == 0.0 {
            0
        } else {
            value.to_bits()
        };

        Self(expand_u64(bits))
    }

    /// Hashes a floating point value, widened from `f32` first so that a number
    /// exactly representable in both seeds the same either way.
    pub fn from_f32(value: f32) -> Self {
        Self::from_f64(value as f64)
    }

    /// A seed from the operating system, for a world that was not given one.
    ///
    /// Falls back to [`Seed::best_effort_entropy`] only if the platform's
    /// random source fails. Call this once, store the returned seed, and derive
    /// reproducible children from it. The stream algorithms in this module are
    /// still non-cryptographic even when their starting seed came from the OS.
    pub fn entropy() -> Self {
        Self::try_entropy().unwrap_or_else(|_| Self::best_effort_entropy())
    }

    /// Reads all 128 seed bits from the operating system.
    pub fn try_entropy() -> Result<Self, getrandom::Error> {
        let mut bytes = [0u8; 16];
        getrandom::fill(&mut bytes)?;
        Ok(Self::from_raw(u128::from_le_bytes(bytes)))
    }

    /// Best-effort fallback when operating-system randomness is unavailable.
    ///
    /// This combines time, an allocation address, and a process-local counter.
    /// It is intended only to make ordinary worlds differ and must never be
    /// treated as secret or unpredictable.
    pub fn best_effort_entropy() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let nanoseconds: u128 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);

        let marker: Box<u8> = Box::new(0);
        let address: u128 = (&*marker as *const u8) as usize as u128;

        let ticket: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);

        Self(absorb128(
            absorb128(mix128(nanoseconds), address),
            expand_u64(ticket),
        ))
    }
}

// ---------------------------------------------------------------------------
// Branching
// ---------------------------------------------------------------------------

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
        Self(absorb128(self.0 ^ DETAIL_TAG, expand_u64(u64::from(detail))))
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
    /// The argument is any [`Shuffle`](crate::structures::traits::Shuffle), so an
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
    /// [`Shuffle`](crate::structures::traits::Shuffle) and the collection decides what
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

impl Seed {
    /// The seed as a code a player can write down and type back in.
    ///
    /// Crockford base32 in four groups of six characters and one of two: 26
    /// digits of five bits, which covers the 128 bits with three to spare. The
    /// alphabet has no `I`, `L`, `O` or `U`, so there is no digit a letter can
    /// be confused with and no accidental word.
    ///
    /// The digits are cut from the low end first, so the grouping is the same
    /// whatever the value's magnitude, and written out most significant first,
    /// which is the order [`Seed::from_code`] reads when it walks the string
    /// backwards.
    pub fn to_code(self) -> String {
        let mut characters: [u8; 26] = [b'0'; 26];

        let mut value: u128 = self.0;

        for slot in characters.iter_mut() {
            *slot = CODE_ALPHABET[(value & 0x1F) as usize];
            value >>= 5;
        }

        let mut code: String = String::with_capacity(30);

        for (position, character) in characters.into_iter().rev().enumerate() {
            if position > 0 && position % 6 == 0 {
                code.push('-');
            }

            code.push(character as char);
        }

        code
    }

    /// Reads a code back, or `None` when it is not one.
    ///
    /// Forgiving about how it was written down: case is ignored, dashes,
    /// spaces and underscores may be anywhere or absent, and the letters `I`
    /// and `L` are read as `1` and `O` as `0`, which is how they are most often
    /// mistyped.
    ///
    /// Strict about shape, though. A code is exactly the 26 digits
    /// [`Seed::to_code`] writes, and the leading one carries only the three
    /// bits left over from 128, so it is never above `7`. Without those two
    /// checks this would accept most ordinary words, since the alphabet is
    /// nearly all of the letters: `"Wandering Hill"` would read as a code
    /// rather than as a name, and [`Seed::parse`] would hand back a world
    /// nobody asked for.
    pub fn from_code(code: &str) -> Option<Self> {
        let mut value: u128 = 0;
        let mut digits: u32 = 0;

        for character in code.chars().rev() {
            if character == '-' || character == ' ' || character == '_' {
                continue;
            }

            let upper: char = character.to_ascii_uppercase();
            let digit: u8 = match upper {
                'I' | 'L' => 1,
                'O' => 0,
                _ => CODE_ALPHABET
                    .iter()
                    .position(|&candidate| candidate == upper as u8)? as u8,
            };

            if digits == 25 && digit > 7 {
                return None;
            }

            if digits == 26 {
                return None;
            }

            value |= (digit as u128) << (digits * 5);
            digits += 1;
        }

        if digits != 26 {
            return None;
        }

        Some(Self(value))
    }

    /// The seed as 32 hexadecimal digits, for logs and save files.
    pub fn to_hex(self) -> String {
        format!("{:032x}", self.0)
    }

    /// Reads back what [`Seed::to_hex`] wrote. Underscores and a leading `0x`
    /// are allowed.
    pub fn from_hex(text: &str) -> Option<Self> {
        let cleaned: String = text
            .trim()
            .trim_start_matches("0x")
            .trim_start_matches("0X")
            .chars()
            .filter(|character| *character != '_')
            .collect();

        u128::from_str_radix(&cleaned, 16).ok().map(Self)
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// Every integer type seeds by value, through the same path as
/// [`Seed::from_integer`], so the two can never drift apart.
macro_rules! implement_integer_seed {
    ($($integer:ty),+) => {
        $(
            impl From<$integer> for Seed {
                fn from(value: $integer) -> Self {
                    Self::from_integer(value)
                }
            }
        )+
    };
}

implement_integer_seed!(
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize
);

impl From<bool> for Seed {
    fn from(value: bool) -> Self {
        Self::from(value as u8)
    }
}

impl From<char> for Seed {
    fn from(value: char) -> Self {
        Self::from(value as u32)
    }
}

impl From<f64> for Seed {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<f32> for Seed {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<&str> for Seed {
    fn from(value: &str) -> Self {
        Self::from_text(value)
    }
}

impl From<&String> for Seed {
    fn from(value: &String) -> Self {
        Self::from_text(value)
    }
}

impl From<String> for Seed {
    fn from(value: String) -> Self {
        Self::from_text(&value)
    }
}

impl From<&[u8]> for Seed {
    fn from(value: &[u8]) -> Self {
        Self::from_bytes(value)
    }
}

impl<const N: usize> From<[i128; N]> for Seed {
    fn from(value: [i128; N]) -> Self {
        Self::from_raw(0).at(value)
    }
}

impl From<Vector2<i128>> for Seed {
    fn from(value: Vector2<i128>) -> Self {
        Self::from(value.to_array())
    }
}

impl From<Vector3<i128>> for Seed {
    fn from(value: Vector3<i128>) -> Self {
        Self::from(value.to_array())
    }
}

impl From<Vector4<i128>> for Seed {
    fn from(value: Vector4<i128>) -> Self {
        Self::from(value.to_array())
    }
}

impl From<Seed> for u128 {
    fn from(value: Seed) -> Self {
        value.0
    }
}

impl From<Seed> for Random {
    fn from(value: Seed) -> Self {
        value.to_random()
    }
}

impl std::fmt::Display for Seed {
    /// The shareable code, which is what a player should see.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_code())
    }
}

impl std::fmt::Debug for Seed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Seed({} / {:#034x})", self.to_code(), self.0)
    }
}


// ---------------------------------------------------------------------------
// Positions a seed can be derived at
// ---------------------------------------------------------------------------

/// A place a [`Seed`] can be derived for.
///
/// Implemented for every [`VoxelPosition`](crate::spatial::VoxelPosition3) and
/// [`NodePosition`](crate::spatial::NodePosition3), in two, three and four
/// dimensions, so [`Seed::at_position`] reads the same whatever shape of position is
/// to hand.
///
/// # Why the position folds itself
///
/// The obvious signature would hand back the coordinates for `Seed` to absorb, but
/// the shapes differ — three coordinates here, four there, and a depth on some of
/// them — and a trait cannot return differently sized arrays without either
/// allocating or reaching for const generics that do not compile on stable. Letting
/// each position fold *itself* into the seed sidesteps all of it: no allocation, and
/// each type states its own convention in one place.
pub trait SeedablePosition {
    /// Folds this position into `seed`.
    fn derive(&self, seed: Seed) -> Seed;
}

/// Generates the impls, which differ only in dimension and in whether a depth comes
/// along with the address.
macro_rules! seedable_positions {
    (voxel: $($voxel:ty),*; node: $($node:ty),*) => {
        $(
            /// Folds the coordinates. The dimension count goes in too, so a voxel at
            /// `(1, 2)` is not the same place as one at `(1, 2, 0)`.
            impl SeedablePosition for $voxel {
                fn derive(&self, seed: Seed) -> Seed {
                    seed.at(self.to_array())
                }
            }
        )*
        $(
            /// Folds the address and then the depth, so one address at two depths
            /// gives two places. Matches what [`Seed::at_depth`] did with the two
            /// passed separately.
            impl SeedablePosition for $node {
                fn derive(&self, seed: Seed) -> Seed {
                    seed.at(self.to_array()).at_level(self.depth())
                }
            }
        )*
    };
}

seedable_positions! {
    voxel: VoxelPosition2, VoxelPosition3, VoxelPosition4;
    node: NodePosition2, NodePosition3, NodePosition4
}
