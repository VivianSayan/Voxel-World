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
use crate::math::UnitQuaternion;
use crate::math::linear::{Vector2, Vector3, Vector4};
use crate::random::Random;
use crate::random::distributions::UniformRotation;
use crate::random::distributions::{Bernoulli, BernoulliRatio, Distribution, WeightedDiscrete};
use crate::random::mixing::{
    GOLDEN_GAMMA, GOLDEN_GAMMA_128, absorb128, expand_u64, fold_u128, hash_bytes, mix64, mix128,
};
use crate::random::source::StochasticSource;
use crate::spatial::{
    Depth, NodePosition2, NodePosition3, NodePosition4, VoxelPosition2, VoxelPosition3,
    VoxelPosition4,
};
use crate::structures::traits::Shuffle;
use crate::time::Tick;
use crate::units::Unit;
use crate::units::{NoiseValue, UniformNoise, UnitValue};
use crate::units::{Probability, Ratio};

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

include!("seed/derivation.rs");
include!("seed/sampling.rs");
include!("seed/serialization.rs");
include!("seed/conversions.rs");
include!("seed/positions.rs");
