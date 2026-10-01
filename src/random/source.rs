//! Where random bits come from, kept apart from what is done with them.
//!
//! Every sampler in [`distributions`](super::distributions) asks only for a
//! [`RandomSource`]: something that hands out 64 uniform bits at a time. Two
//! things do:
//!
//! - [`Random`], a stream that moves on after every draw, for
//!   anything that wants a run of different values.
//! - a [`Seed`], through [`Seed::sample`]: a temporary cursor that
//!   starts at the seed and lasts one call, so the same seed always answers the
//!   same, however many words the answer takes.
//!
//! [`Seed::sample`]: super::seed::Seed::sample
//!
//! The helpers below are the only places the raw bits are turned into
//! fractions and bounded integers, so the two sources cannot drift apart on
//! how that is done.
//!
//! [`DrawSource`] is the other half of the story: randomness that something
//! *keeps* and advances itself, rather than randomness a sampler borrows.

use crate::random::Random;
use crate::random::distributions::Distribution;
use crate::random::seed::{Seed, SeedCursor};
use crate::units::Unit;

/// Anything that produces uniform 64-bit words.
///
/// Only [`RandomSource::next_u64`] has to be written; the rest are built on it
/// and should be left alone, since every sampler relies on them consuming
/// exactly the draws they do.
pub trait RandomSource {
    /// Uniform over every `u64`.
    fn next_u64(&mut self) -> u64;

    /// Uniform in `[0, 1)` on the `2^-63` grid.
    ///
    /// The integer-backed fraction, and the one to prefer for a decision: comparing
    /// against a [`Unit`] chance is exact to the last bit, where the 53-bit
    /// [`RandomSource::unit_f64`] grid rounds the effective probability by up to
    /// `2^-53`. It also costs no more — one word either way.
    ///
    /// [`RandomSource::unit_f64`] remains the right call when the fraction is about
    /// to be fed to `ln`, `powf` or the like, which is most of the continuous
    /// samplers: those need a float in the end and going through a `Unit` would only
    /// add a conversion.
    fn unit(&mut self) -> Unit {
        Unit::from_word(self.next_u64())
    }

    /// Uniform in `[0, 1)` with 53 bits of precision.
    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `(0, 1)`, open at both ends. Safe to pass to `ln` or use as
    /// a divisor.
    ///
    /// Takes 52 bits and sits the value at the midpoint of its interval rather
    /// than its lower edge, which lifts it off zero. 52 rather than 53 because
    /// the midpoint is then exactly representable even at the top of the range;
    /// with 53 bits, adding a half could round the result to one.
    fn open_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 12) as f64 + 0.5) * (1.0 / (1u64 << 52) as f64)
    }

    /// Uniform in `[0, range)` without bias (Lemire's method). `range` must
    /// not be zero.
    ///
    /// Taking the high half of a widening multiply avoids the bias `%` would
    /// bring, but cuts `2^64` into buckets that differ in size by one; the
    /// remainder check against Lemire's threshold removes that too, drawing
    /// again on the rare rejection.
    fn bounded_u64(&mut self, range: u64) -> u64 {
        assert!(range > 0, "a bounded random range must be non-zero");

        let mut product: u128 = self.next_u64() as u128 * range as u128;
        let mut low: u64 = product as u64;

        if low < range {
            let threshold: u64 = range.wrapping_neg() % range;
            while low < threshold {
                product = self.next_u64() as u128 * range as u128;
                low = product as u64;
            }
        }

        (product >> 64) as u64
    }

    /// The second normal of the last pair, if the source keeps one.
    ///
    /// The polar method makes normals two at a time. A long-lived stream keeps
    /// the second for the next call; a one-shot source has no next call, so the
    /// default keeps nothing.
    fn take_spare_normal(&mut self) -> Option<f64> {
        None
    }

    /// Offers the second normal of a pair for later. Ignored by default.
    fn keep_spare_normal(&mut self, _spare: f64) {}
}

// ---------------------------------------------------------------------------
// Randomness an owner keeps
// ---------------------------------------------------------------------------

/// Randomness that something stores and advances itself, one draw at a time.
///
/// [`RandomSource`] is what a sampler *borrows* to read words. This is what a
/// long-lived owner *keeps* between draws: an entry in a schedule, a component
/// on an entity, anything that has to come back later and roll again without a
/// generator being passed in.
///
/// Both kinds of randomness in the crate qualify, and they differ in what they
/// cost and what they promise:
///
/// | | Size | Replay | Saving |
/// |---|---|---|---|
/// | [`SeedCursor`] | origin plus current seed position | a pure function of where its seed came from | store the cursor, or derive it again |
/// | [`Random`] | four-word xoshiro state plus metadata | depends on that stream's own history | store [`Random::snapshot`](crate::random::Random::snapshot) |
///
/// A [`SeedCursor`] moves past every word a draw reads. A [`Random`] simply
/// carries on where its xoshiro state left off. An immutable [`Seed`] itself
/// deliberately does not implement this trait.
pub trait DrawSource {
    /// One draw, advancing this source so the next call gives a different
    /// answer.
    fn draw<D: Distribution>(&mut self, distribution: &D) -> D::Output;

    /// A source of this kind derived from a seed, for a holder that wants one
    /// made rather than supplied.
    fn from_seed(seed: Seed) -> Self
    where
        Self: Sized;
}

/// Carries on where the stream left off.
impl DrawSource for Random {
    fn draw<D: Distribution>(&mut self, distribution: &D) -> D::Output {
        distribution.sample(self)
    }

    fn from_seed(seed: Seed) -> Self {
        Random::new(seed)
    }
}

/// Draws from a seed-derived cursor and carries on after every word consumed.
impl DrawSource for SeedCursor {
    fn draw<D: Distribution>(&mut self, distribution: &D) -> D::Output {
        self.sample(distribution)
    }

    fn from_seed(seed: Seed) -> Self {
        seed.cursor()
    }
}

/// Either kind of randomness, for a holder whose entries do not all want the
/// same one.
///
/// Storing this instead of a [`Seed`] or a [`Random`] costs a tag and a pointer
/// for the streaming case, and is only worth it when the choice genuinely
/// varies: most collections are better off generic over [`DrawSource`] and
/// homogeneous.
#[derive(Clone, Debug)]
pub enum EventRandom {
    /// A position in the chain a seed determines: small, and replayable from
    /// the world seed.
    Seeded(SeedCursor),
    /// A stream of its own: larger, and its state is part of the world's.
    Stream(Box<Random>),
}

impl EventRandom {
    /// The seed this source started from, for the seeded case.
    pub fn seed(&self) -> Seed {
        match self {
            Self::Seeded(cursor) => cursor.seed(),
            Self::Stream(random) => random.seed(),
        }
    }
}

/// Whichever kind it holds.
impl DrawSource for EventRandom {
    fn draw<D: Distribution>(&mut self, distribution: &D) -> D::Output {
        match self {
            Self::Seeded(seed) => seed.draw(distribution),
            Self::Stream(random) => random.draw(distribution),
        }
    }

    /// The seeded kind, which is the cheaper of the two; build
    /// [`EventRandom::Stream`] directly where a stream is wanted.
    fn from_seed(seed: Seed) -> Self {
        Self::Seeded(seed.cursor())
    }
}

impl From<Seed> for EventRandom {
    fn from(seed: Seed) -> Self {
        Self::Seeded(seed.cursor())
    }
}
