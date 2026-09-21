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

use crate::misc::linear::{Vector2, Vector3, Vector4};
use crate::misc::mixing::{absorb128, expand_u64, fold_u128, hash_bytes, mix128};
use crate::misc::random::Random;

/// Keeps the ways of deriving a seed from colliding with one another, so that
/// `child("7")`, `index(7)` and `at([7])` are three different seeds.
const CHILD_TAG: u128 = 0x7B7F_AEB8_9C7A_2D31_C4CE_B9FE_1A85_EC53;
const INDEX_TAG: u128 = 0x1F83_D9AB_FB41_BD6B_5BE0_CD19_137E_2179;
const POSITION_TAG: u128 = 0x510E_527F_ADE6_82D1_9B05_688C_2B3E_6C1F;
const COMBINE_TAG: u128 = 0x6A09_E667_F3BC_C908_BB67_AE85_84CA_A73B;
const STREAM_TAG: u128 = 0x3C6E_F372_FE94_F82B_A54F_F53A_5F1D_36F1;

/// The alphabet [`Seed::to_code`] writes in: Crockford base32, which leaves out
/// `I`, `L`, `O` and `U` so that nothing in a written-down seed can be mistaken
/// for a digit or for another letter.
const CODE_ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// The integer widths [`Seed::from_integer`] accepts, which is all of them.
///
/// Every width maps to the same bit pattern for the same value, so the type a
/// number arrived in never shows up in the seed. Signed values sign-extend, so
/// `-1` and `u128::MAX` agree; that is the one place two different values meet.
pub trait SeedInteger: Copy {
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

impl SeedInteger for usize {
    fn to_seed_bits(self) -> u128 {
        self as u128
    }
}

impl SeedInteger for isize {
    fn to_seed_bits(self) -> u128 {
        self as i128 as u128
    }
}

/// A world seed, and the root of every seed derived from it.
///
/// Cheap to copy, and comparing two is comparing two integers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Seed(u128);

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

    /// A seed from the clock and this process's own addresses, for a world that
    /// was not given one.
    ///
    /// The only thing here that is not reproducible, and the only one that
    /// should be: call it once when a world is created, store what it returned,
    /// and derive everything else from that.
    ///
    /// Best effort, and only good enough to make two worlds differ. The clock
    /// is public, a heap address leaks little and is often guessable, and the
    /// counter starts at zero every run, so anyone who knows roughly when a
    /// world was made can search the space. Do not use it for anything that has
    /// to stay secret: it is not a cryptographic random source, and this module
    /// has none. Take those from the operating system.
    pub fn entropy() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        /// Two calls in the same clock tick would otherwise agree, and a
        /// short-lived allocation tends to come back at the same address, so
        /// neither of those alone separates them.
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let nanoseconds: u128 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);

        // The clock alone repeats if two worlds are made in the same tick, and
        // is guessable besides. A heap address adds whatever the allocator and
        // address-space randomisation contribute.
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
        labels.into_iter().fold(self, |seed, label| seed.child(label))
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

    /// A branch for a point at one octree depth, so the same coordinates at
    /// different depths are unrelated.
    pub fn at_depth<const N: usize>(self, depth: i8, coordinates: [i128; N]) -> Self {
        self.at(coordinates).index(depth as i64 as u64)
    }

    /// Mixes two seeds into one that depends on both.
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
    /// The step is bijective, so the walk is a permutation of the whole 128-bit
    /// space: it can never run into a value it has already passed without
    /// returning to where it started, and there is no shortening tail before
    /// the loop closes. How long that loop is, though, is not proven. A random
    /// permutation's cycle through a given point averages about `2^127`, and no
    /// short cycle has turned up in testing, but treat the period as unknown
    /// rather than guaranteed.
    ///
    /// Order-dependent by nature, so prefer [`Seed::index`] or [`Seed::at`] for
    /// anything that has to survive a reload.
    pub fn advance(self) -> Self {
        Self(mix128(self.0 ^ STREAM_TAG))
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
    pub fn unit_f64(self) -> f64 {
        (self.as_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// True with probability `chance`, for a one-off decision.
    ///
    /// The same seed always decides the same way, which is what makes this
    /// usable for world content rather than only for effects.
    pub fn chance(self, chance: f64) -> bool {
        self.unit_f64() < chance
    }

    /// Uniform in `0..length`, or `None` when `length` is zero.
    ///
    /// Exactly uniform, not merely close to it. Taking the high half of a
    /// widening multiply avoids the bias that `%` would introduce, but leaves
    /// its own: the buckets it cuts `2^64` into differ in size by one, so short
    /// values are over-represented by up to one part in `2^64 / length`. That
    /// is nothing for a game, and removing it costs nothing either, so the
    /// remainder is checked against Lemire's threshold and the seed advanced
    /// for fresh bits on the rare rejection.
    ///
    /// Still a pure function of the seed: the same seed always answers the
    /// same, rejection or not.
    pub fn below(self, length: u64) -> Option<u64> {
        if length == 0 {
            return None;
        }

        let mut source: Seed = self;
        let mut product: u128 = source.as_u64() as u128 * length as u128;

        // Only the low half can be short-changed, and only when it falls below
        // the count of leftover values.
        if (product as u64) < length {
            let threshold: u64 = length.wrapping_neg() % length;

            while (product as u64) < threshold {
                source = source.advance();
                product = source.as_u64() as u128 * length as u128;
            }
        }

        Some((product >> 64) as u64)
    }

    /// Picks one element, or `None` when the slice is empty.
    pub fn pick<T>(self, items: &[T]) -> Option<&T> {
        self.below(items.len() as u64)
            .map(|index| &items[index as usize])
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
    /// Crockford base32 in four groups of six characters and one of two. The
    /// alphabet has no `I`, `L`, `O` or `U`, so there is no digit a letter can
    /// be confused with and no accidental word.
    pub fn to_code(self) -> String {
        let mut characters: [u8; 26] = [b'0'; 26];

        // Little end first, so the groups read in a fixed order whatever the
        // value's magnitude.
        let mut value: u128 = self.0;

        for slot in characters.iter_mut() {
            *slot = CODE_ALPHABET[(value & 0x1F) as usize];
            value >>= 5;
        }

        let mut code: String = String::with_capacity(30);

        // Written most significant first, which is the order `from_code` reads
        // when it walks the string backwards.
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
                    .position(|&candidate| candidate == upper as u8)?
                    as u8,
            };

            // 26 groups of five bits is 130, so the leading digit carries only
            // the three bits that are left and cannot be above 7.
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

implement_integer_seed!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize);

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
