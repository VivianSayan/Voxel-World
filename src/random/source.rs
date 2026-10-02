//! Where stochastic draws come from, kept apart from what is done with them.
//!
//! # Two traits, and the difference between them
//!
//! | | [`StochasticSource`] | [`StochasticStream`] |
//! |---|---|---|
//! | Who holds it | an algorithm **borrows** it for one call | an owner **keeps** it between calls |
//! | What it answers | "give me words, bounded integers, fractions" | "draw me a value from this distribution" |
//! | Lifetime | as long as the `&mut` borrow | as long as the thing that owns it |
//! | Can be built from a seed | no, it already exists | yes, [`StochasticStream::from_seed`] |
//!
//! Everything in [`distributions`](super::distributions), every shuffle and the
//! bulk picker ask only for a [`StochasticSource`]: 64 uniform bits at a time, plus
//! the few helpers below that turn those bits into fractions and bounded integers.
//! Keeping those helpers in one place is what stops two sources drifting apart on how
//! it is done.
//!
//! [`StochasticStream`] is a different capability, not a bigger one. A scheduler
//! entry, a component on an entity, anything that has to come back later and roll
//! again without a generator being handed to it — those **store** their randomness,
//! and that is what this trait describes. It is also why it carries `from_seed`: a
//! holder usually wants one made rather than supplied.
//!
//! # What implements which
//!
//! - [`Random`] — a stream that moves on after every draw. Both traits.
//! - [`SeedCursor`] — a deterministic position that advances past each word it
//!   reads. Both traits.
//! - [`Seed`] — **neither**, deliberately. A seed is an immutable question, not a
//!   stream; it hands out a temporary cursor through [`Seed::cursor`] and keeps no
//!   state of its own. That is what makes the same seed answer the same, for ever.
//!
//! [`Seed::cursor`]: super::seed::Seed::cursor
//!
//! # Why these names
//!
//! `StochasticSource` was `RandomSource`, which read as though it belonged to
//! [`Random`] — yet [`SeedCursor`] implements it just as fully, and a seeded world
//! uses that path far more. `Stochastic` names the capability rather than one of its
//! two providers.
//!
//! `StochasticStream` was `DrawSource`, which said nothing about how it differed from
//! the other one. *Source* against *stream* carries the distinction in the names: a
//! source is borrowed, a stream is owned and advances. It is also the word the crate
//! already used in prose for exactly this.

use crate::math::fixed::Fixed;
use crate::random::Random;
use crate::random::distributions::Distribution;
use crate::random::seed::{Seed, SeedCursor};
use crate::units::Unit;

/// Anything that produces uniform 64-bit words.
///
/// Only [`StochasticSource::next_u64`] has to be written; the rest are built on it
/// and should be left alone, since every sampler relies on them consuming
/// exactly the draws they do.
pub trait StochasticSource {
    /// Uniform over every `u64`.
    fn next_u64(&mut self) -> u64;

    /// Uniform in `[0, 1)` on the `2^-63` grid.
    ///
    /// The integer-backed fraction, and the one to prefer for a decision: comparing
    /// against a [`Unit`] chance is exact to the last bit, where the 53-bit
    /// [`StochasticSource::unit_f64`] grid rounds the effective probability by up to
    /// `2^-53`. It also costs no more — one word either way.
    ///
    /// [`StochasticSource::unit_f64`] remains the right call when the fraction is about
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

    /// A uniform `u128` below `range`, with no bias. `range` must not be zero.
    ///
    /// # Why this is not [`StochasticSource::bounded_u64`] widened
    ///
    /// Lemire's method needs the high half of a `range × word` product, which at 128
    /// bits means a 256-bit multiply. The crate has one, but it lives inside the
    /// fixed-point implementation as a detail of that type, and reaching into it from
    /// here would tie two unrelated parts of the crate together for a function that
    /// is rarely hot.
    ///
    /// So this masks instead: it draws the fewest bits that can hold `range - 1` and
    /// tries again on a value at or above it. That is unbiased for the same reason
    /// rejection always is — every accepted value was equally likely — and it accepts
    /// at least half the time, so it averages under two attempts.
    fn bounded_u128(&mut self, range: u128) -> u128 {
        assert!(range > 0, "a bounded random range must be non-zero");

        if range == 1 {
            // Only one possible answer, and no words spent discovering it.
            return 0;
        }

        let bits: u32 = u128::BITS - (range - 1).leading_zeros();

        loop {
            let candidate: u128 = if bits <= 64 {
                u128::from(self.next_u64()) >> (64 - bits)
            } else {
                let high: u128 = u128::from(self.next_u64());
                let low: u128 = u128::from(self.next_u64());

                ((high << 64) | low) >> (128 - bits)
            };

            if candidate < range {
                return candidate;
            }
        }
    }

    /// A uniform index below `length`, which must not be zero.
    ///
    /// # Question
    ///
    /// "Which of these `length` positions?"
    ///
    /// The one place anything here turns a word into a position, so every collection
    /// that picks or shuffles shares one unbiased bounded draw. It lives on the trait
    /// rather than on [`Random`] so that a [`Seed`]'s cursor can do it too — which is
    /// what lets a seeded world shuffle a deck and get the same order every time.
    ///
    /// # Panics
    ///
    /// On a zero length, which has no index to return.
    fn index_below(&mut self, length: usize) -> usize {
        assert!(length > 0, "cannot draw an index from an empty range");

        self.bounded_u64(length as u64) as usize
    }

    /// A uniformly random [`Fixed`] in `[0, 1)`, on the `2^-FRACTION_BITS` grid.
    ///
    /// The same word as [`StochasticSource::unit`], keeping its top
    /// `FRACTION_BITS` bits, so the two are one draw at two precisions rather than
    /// two unrelated ones.
    ///
    /// # Why the bits are truncated rather than rounded
    ///
    /// `self.unit().to_fixed()` looks like the obvious implementation and is wrong:
    /// that conversion rounds, so a draw near the top carries up to exactly one and
    /// the range stops being half-open. Keeping the high bits cannot do that — the
    /// result is a count below `2^FRACTION_BITS`, so it is always under one.
    ///
    /// Half-open matters for the same reason it does for [`Unit`]: a comparison
    /// against a chance is only exact when one is unreachable.
    fn fixed(&mut self) -> Fixed {
        Fixed::fraction_from_word(self.next_u64())
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
/// Randomness something **keeps**, as against randomness an algorithm borrows.
///
/// # Question
///
/// "This thing has to roll again later, on its own. What does it store?"
///
/// [`StochasticSource`] is what a sampler *borrows* to read words. This is what a
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
pub trait StochasticStream {
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
impl StochasticStream for Random {
    fn draw<D: Distribution>(&mut self, distribution: &D) -> D::Output {
        distribution.sample(self)
    }

    fn from_seed(seed: Seed) -> Self {
        Random::new(seed)
    }
}

/// Draws from a seed-derived cursor and carries on after every word consumed.
impl StochasticStream for SeedCursor {
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
/// varies: most collections are better off generic over [`StochasticStream`] and
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
impl StochasticStream for EventRandom {
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
