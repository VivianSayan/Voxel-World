//! Integer-only samplers over [`Ratio`] parameters: exact probabilities, and
//! the same result from the same seed on every platform.
//!
//! Everything here rests on one idea. A uniform draw `u` in `[0, 1)` is an
//! endless string of random bits, and `u < p` is decided at the first bit where
//! that string and the binary expansion of `p` differ. For a fraction `p`, long
//! division produces the expansion one bit at a time, so the comparison needs
//! no floating point and stops, on average, after two bits. That gives an exact
//! coin for any fraction ([`chance_fraction`]), and, by comparing 64 draws
//! against the same expansion in parallel, 64 coins in a handful of words
//! ([`BernoulliMask`]).
//!
//! Exact coins for fractions also give exact coins for `exp(-p/q)`, by
//! Canonne, Kamath and Steinke's method ("The Discrete Gaussian for
//! Differential Privacy", 2020), and those give exact discrete Laplace and
//! discrete Gaussian samplers. The Poisson sampler uses the same coins with a
//! rejection step of its own.

use super::{Distribution, PortableDistribution};
use crate::random::source::RandomSource;
use crate::units::Ratio;

// ---------------------------------------------------------------------------
// Coins
// ---------------------------------------------------------------------------

/// The binary digits of `numerator / denominator`, most significant first,
/// by long division.
struct Digits {
    remainder: u128,
    denominator: u128,
}

impl Digits {
    /// The expansion of `numerator / denominator`, which must be a proper
    /// fraction. Nothing is produced until [`Digits::next`] is called.
    fn new(numerator: u128, denominator: u128) -> Self {
        debug_assert!(numerator < denominator);
        Self {
            remainder: numerator,
            denominator,
        }
    }

    /// The next digit: doubles the remainder and takes the whole part, which
    /// is long division in base two.
    ///
    /// The doubling can carry out of 128 bits. The true value is then at least
    /// `2^128`, so it is above any denominator and the digit is a one, and the
    /// wrapped subtraction still lands on the right remainder, which is below
    /// the denominator again.
    fn next(&mut self) -> bool {
        let (doubled, carried): (u128, bool) = self.remainder.overflowing_add(self.remainder);
        let digit: bool = carried || doubled >= self.denominator;
        self.remainder = if digit {
            doubled.wrapping_sub(self.denominator)
        } else {
            doubled
        };
        digit
    }

    /// Whether every digit from here on is zero, which is how a fraction with
    /// a short expansion, such as one eighth, is finished early.
    fn ended(&self) -> bool {
        self.remainder == 0
    }
}

/// True with probability exactly `numerator / denominator`. Anything at or
/// above one is always true; a zero numerator is never true.
///
/// Compares the draw with the fraction one bit at a time, most significant
/// first, and stops at the first bit where they differ: a zero drawn against a
/// one in the fraction puts the draw below it, and a one against a zero puts it
/// above. Matching all the way to the end of a short expansion means the draw
/// is at or above the fraction, since nothing is left for it to be below.
///
/// Reads one word, and another only in the `2^-64` case that all its bits
/// matched the expansion.
pub(super) fn chance_fraction<S: RandomSource + ?Sized>(
    source: &mut S,
    numerator: u128,
    denominator: u128,
) -> bool {
    if numerator >= denominator {
        return true;
    }
    if numerator == 0 {
        return false;
    }

    let mut digits: Digits = Digits::new(numerator, denominator);

    loop {
        let word: u64 = source.next_u64();

        for position in (0..u64::BITS).rev() {
            let drawn: bool = (word >> position) & 1 == 1;
            let digit: bool = digits.next();

            if drawn != digit {
                return digit;
            }

            if digits.ended() {
                return false;
            }
        }
    }
}

/// True with probability exactly `numerator / (denominator * divisor)`, without
/// the product overflowing: when it would, the chance is split into two
/// independent coins.
fn chance_fraction_divided<S: RandomSource + ?Sized>(
    source: &mut S,
    numerator: u128,
    denominator: u128,
    divisor: u128,
) -> bool {
    match denominator.checked_mul(divisor) {
        Some(product) => chance_fraction(source, numerator, product),
        None => {
            chance_fraction(source, numerator, denominator) && chance_fraction(source, 1, divisor)
        }
    }
}

/// 64 independent coins at `numerator / denominator`, one per bit.
///
/// All 64 draws are compared against the fraction at once: each word supplies
/// the next bit of every draw, and a lane is settled as soon as its bit differs
/// from the fraction's. Every word settles about half of the lanes still open,
/// so about seven words decide all 64. A fraction with a short expansion, such
/// as one in two or three in eight, finishes in as many words as it has
/// digits: a single word for one half.
///
/// Each word supplies one bit of all 64 draws at once. Where the fraction's
/// digit is a one, the lanes that drew a zero are settled as successes and the
/// rest stay open; where it is a zero, the lanes that drew a one are settled as
/// failures. Any lane still level with the fraction when its digits run out is
/// at or above it, and so a failure.
fn chance_mask<S: RandomSource + ?Sized>(
    source: &mut S,
    numerator: u128,
    denominator: u128,
) -> u64 {
    if numerator >= denominator {
        return u64::MAX;
    }
    if numerator == 0 {
        return 0;
    }

    let mut digits: Digits = Digits::new(numerator, denominator);
    let mut open: u64 = u64::MAX;
    let mut successes: u64 = 0;

    while open != 0 {
        let word: u64 = source.next_u64();

        if digits.next() {
            successes |= open & !word;
            open &= word;
        } else {
            open &= !word;
        }

        if digits.ended() {
            break;
        }
    }

    successes
}

/// True with probability exactly `exp(-numerator / denominator)`.
///
/// Canonne, Kamath & Steinke (2020), Algorithm 1. For an exponent up to one,
/// coins at `x`, `x / 2`, `x / 3`, ... are tossed until one fails, and the
/// answer is whether the count of tosses was odd: the chance of that is the
/// alternating series for `exp(-x)`. A larger exponent is split into its whole
/// part, one `exp(-1)` coin per unit, and the rest.
fn chance_exp_minus<S: RandomSource + ?Sized>(
    source: &mut S,
    numerator: u128,
    denominator: u128,
) -> bool {
    let whole: u128 = numerator / denominator;

    for _ in 0..whole {
        if !chance_exp_minus_below_one(source, 1, 1) {
            return false;
        }
    }

    chance_exp_minus_below_one(source, numerator % denominator, denominator)
}

/// [`chance_exp_minus`] for an exponent of at most one.
fn chance_exp_minus_below_one<S: RandomSource + ?Sized>(
    source: &mut S,
    numerator: u128,
    denominator: u128,
) -> bool {
    let mut tosses: u128 = 1;

    while chance_fraction_divided(source, numerator, denominator, tosses) {
        tosses += 1;
    }

    tosses % 2 == 1
}

// ---------------------------------------------------------------------------
// Bernoulli
// ---------------------------------------------------------------------------

/// True with exactly the chance of a [`Ratio`].
///
/// The integer twin of [`Bernoulli`](super::Bernoulli), for chances no `f64`
/// holds: one in three is one in three, not 0.333...3. One word, a multiply
/// and a comparison.
///
/// Exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BernoulliRatio {
    numerator: u64,
    denominator: u64,
}

impl BernoulliRatio {
    /// `None` for a ratio above one.
    pub fn new(chance: Ratio) -> Option<Self> {
        chance.is_proper().then(|| Self {
            numerator: chance.numerator(),
            denominator: chance.denominator(),
        })
    }

    /// One in `count`. A `count` of zero never succeeds.
    pub fn one_in(count: u64) -> Self {
        Self {
            numerator: (count > 0) as u64,
            denominator: count.max(1),
        }
    }
}

impl Distribution for BernoulliRatio {
    type Output = bool;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> bool {
        if self.numerator == 0 {
            return false;
        }
        if self.numerator >= self.denominator {
            return true;
        }
        source.bounded_u64(self.denominator) < self.numerator
    }
}

/// 64 independent coins at a [`Ratio`], one per bit of a `u64`.
///
/// For deciding a whole row of voxels at once: which of these 64 columns get
/// grass, which of these 64 cells hold a crystal. About seven words for all
/// 64 at any chance, and fewer for chances with short binary expansions; one
/// word at one half.
///
/// Exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BernoulliMask {
    numerator: u64,
    denominator: u64,
}

impl BernoulliMask {
    /// `None` for a ratio above one.
    pub fn new(chance: Ratio) -> Option<Self> {
        chance.is_proper().then(|| Self {
            numerator: chance.numerator(),
            denominator: chance.denominator(),
        })
    }

    /// One in `count` for each bit. A `count` of zero never sets one.
    pub fn one_in(count: u64) -> Self {
        Self {
            numerator: (count > 0) as u64,
            denominator: count.max(1),
        }
    }
}

impl Distribution for BernoulliMask {
    type Output = u64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        chance_mask(source, self.numerator as u128, self.denominator as u128)
    }
}

// ---------------------------------------------------------------------------
// Binomial
// ---------------------------------------------------------------------------

/// Number of successes in `trials` independent trials at a [`Ratio`] chance.
///
/// Counts the set bits of [`BernoulliMask`]s, 64 trials at a time. The cost
/// grows with the number of trials, about seven words per 64, where the
/// floating-point [`Binomial`](super::Binomial) takes a handful of draws
/// whatever the size; the exchange is exactness. Right for the counts a voxel
/// world asks for, such as how many of a chunk's 4096 cells hold ore.
///
/// Splitting a count between the eight children of an octree node is a
/// sequence of these: each item goes to the first child at one in eight, to
/// the second at one in seven of those left, and so on. Seeding each node from
/// its position keeps the split the same however the tree is visited.
///
/// Exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinomialRatio {
    trials: u64,
    chance: BernoulliMask,
}

impl BinomialRatio {
    /// `None` for a ratio above one.
    pub fn new(trials: u64, chance: Ratio) -> Option<Self> {
        Some(Self {
            trials,
            chance: BernoulliMask::new(chance)?,
        })
    }
}

impl Distribution for BinomialRatio {
    type Output = u64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        let whole_words: u64 = self.trials / u64::BITS as u64;
        let leftover: u32 = (self.trials % u64::BITS as u64) as u32;

        let mut successes: u64 = 0;

        for _ in 0..whole_words {
            successes += self.chance.sample(source).count_ones() as u64;
        }

        if leftover > 0 {
            let lanes: u64 = (1u64 << leftover) - 1;
            successes += (self.chance.sample(source) & lanes).count_ones() as u64;
        }

        successes
    }
}

// ---------------------------------------------------------------------------
// Poisson
// ---------------------------------------------------------------------------

/// Poisson with a [`Ratio`] mean.
///
/// The mean is split into equal pieces of at most one half and a Poisson drawn
/// for each, since sums of Poissons are Poisson. For a piece `x`, coins at
/// `x`, `x / 2`, `x / 3`, ... are tossed until one fails; the count of
/// successes `k` has chance `x^k / k!` of being reached, and is kept with
/// chance `(1 - x)(k + 1) / (k + 1 - x)`, which leaves exactly the Poisson
/// probabilities. At most one half, a piece is kept about four times in five.
///
/// The cost grows with the mean: about three words per half unit of it. Fine
/// for the counts a chunk needs, trees or boulders or caves; for means in the
/// thousands, the floating-point [`Poisson`](super::Poisson) is far cheaper.
///
/// Exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoissonRatio {
    numerator: u64,
    denominator: u64,
}

impl PoissonRatio {
    /// Any ratio: the mean may be above one.
    pub fn new(mean: Ratio) -> Self {
        Self {
            numerator: mean.numerator(),
            denominator: mean.denominator(),
        }
    }
}

impl Distribution for PoissonRatio {
    type Output = u64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        if self.numerator == 0 {
            return 0;
        }

        let numerator: u128 = self.numerator as u128;
        let denominator: u128 = self.denominator as u128;

        let pieces: u128 = (2 * numerator).div_ceil(denominator);
        let piece_denominator: u128 = denominator * pieces;

        let mut total: u64 = 0;

        for _ in 0..pieces {
            total = total.saturating_add(poisson_piece(source, numerator, piece_denominator));
        }

        total
    }
}

/// Poisson with mean `numerator / denominator`, which is at most one half.
///
/// Coins at `x`, `x / 2`, `x / 3`, ... are tossed until one fails, so the count
/// of successes reaches `k` with chance `x^k / k!`. Keeping that count with
/// chance `(1 - x)(k + 1) / (k + 1 - x)`, written here over a common
/// denominator, leaves exactly the Poisson probabilities. Both sides of that
/// fraction stay far inside 128 bits for any count the coins can realistically
/// reach, as does the piece denominator, which stays below `2^67`.
fn poisson_piece<S: RandomSource + ?Sized>(
    source: &mut S,
    numerator: u128,
    denominator: u128,
) -> u64 {
    loop {
        let mut successes: u128 = 0;

        while chance_fraction_divided(source, numerator, denominator, successes + 1) {
            successes += 1;
        }

        let next: u128 = successes + 1;
        let keep_numerator: u128 = (denominator - numerator) * next;
        let keep_denominator: u128 = denominator * next - numerator;

        if chance_fraction(source, keep_numerator, keep_denominator) {
            return successes as u64;
        }
    }
}

// ---------------------------------------------------------------------------
// Laplace and Gaussian
// ---------------------------------------------------------------------------

/// Integers around zero with `P(k)` proportional to `exp(-|k| / scale)`.
///
/// The two-sided geometric: most results small, either sign equally likely, a
/// tail that thins steadily. For integer jitter, an offset from a centre, a
/// wander of a few voxels.
///
/// Canonne, Kamath & Steinke (2020), Algorithm 2.
///
/// Exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscreteLaplace {
    numerator: u64,
    denominator: u64,
}

impl DiscreteLaplace {
    /// `None` for a zero scale.
    pub fn new(scale: Ratio) -> Option<Self> {
        (!scale.is_zero()).then(|| Self {
            numerator: scale.numerator(),
            denominator: scale.denominator(),
        })
    }
}

impl Distribution for DiscreteLaplace {
    type Output = i64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> i64 {
        discrete_laplace(source, self.numerator, self.denominator)
    }
}

/// Integers around zero with `P(k)` proportional to
/// `exp(-|k| * denominator / numerator)`.
///
/// Builds the magnitude out of two parts: a remainder below the numerator, kept
/// with chance `exp(-remainder / numerator)`, and a count of whole numerators,
/// which is geometric at `exp(-1)`. Their combination has exactly the Laplace
/// weights. A sign is drawn last, and a negative zero is rejected so that zero
/// is not counted from both sides.
fn discrete_laplace<S: RandomSource + ?Sized>(
    source: &mut S,
    numerator: u64,
    denominator: u64,
) -> i64 {
    loop {
        let remainder: u64 = source.bounded_u64(numerator);

        if !chance_exp_minus(source, remainder as u128, numerator as u128) {
            continue;
        }

        let mut wholes: u128 = 0;
        while chance_exp_minus_below_one(source, 1, 1) {
            wholes += 1;
        }

        let magnitude: u128 =
            (remainder as u128 + numerator as u128 * wholes) / denominator as u128;
        let negative: bool = source.next_u64() >> 63 == 1;

        if negative && magnitude == 0 {
            continue;
        }

        let magnitude: i64 = magnitude.min(i64::MAX as u128) as i64;

        return if negative { -magnitude } else { magnitude };
    }
}

/// Integers around `mean` with `P(k)` proportional to
/// `exp(-(k - mean)^2 / (2 * variance))`: the bell curve on whole numbers.
///
/// Heights, depths and offsets that should cluster around a value, with the
/// cross-platform exactness the floating-point [`Normal`](super::Normal)
/// cannot give. Its variance is close to, and for variances above one almost
/// exactly, the one given.
///
/// Canonne, Kamath & Steinke (2020), Algorithm 3: a discrete Laplace proposal
/// with scale `floor(sigma) + 1`, kept with an exact `exp(-x)` coin. The
/// exponent is held as a fraction of 128-bit integers; a proposal so far out
/// that it does not fit, beyond 64 scales, is rejected rather than tested.
/// Those are also ones the test would reject with chance above `1 - e^-1984`,
/// so the distribution differs from exact by less than `2^-2900`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscreteGaussian {
    mean: i64,
    /// The variance as `numerator / denominator`.
    numerator: u128,
    /// The Laplace proposal's scale.
    scale: u64,
    /// `denominator * scale`, the unit the exponent's numerator is counted in.
    unit: u128,
    /// `2 * numerator * denominator * scale^2`, the exponent's denominator.
    exponent_denominator: u128,
}

impl DiscreteGaussian {
    /// `None` when the variance is too large for the exponent to be held
    /// exactly: numerators past about `2^56`. A zero variance always gives the
    /// mean.
    ///
    /// The proposal's scale is `floor(sigma) + 1`, computed as the integer
    /// square root of the variance's whole part, since `floor(sqrt(n / d))` is
    /// `isqrt(floor(n / d))`. The two cached denominators are what the
    /// acceptance test's exponent, `(|y| d t - n)^2 / (2 n d t^2)`, is written
    /// over; the check that proposals out to 64 scales still fit is what bounds
    /// the error quoted on the type.
    pub fn new(mean: i64, variance: Ratio) -> Option<Self> {
        let numerator: u128 = variance.numerator() as u128;
        let denominator: u128 = variance.denominator() as u128;

        let scale: u64 = (variance.numerator() / variance.denominator()).isqrt() + 1;
        let unit: u128 = denominator * scale as u128;

        if unit.checked_mul(64 * scale as u128)? > u64::MAX as u128 {
            return None;
        }

        let exponent_denominator: u128 = (2 * numerator)
            .checked_mul(denominator)?
            .checked_mul(scale as u128)?
            .checked_mul(scale as u128)?;

        Some(Self {
            mean,
            numerator,
            scale,
            unit,
            exponent_denominator,
        })
    }
}

impl Distribution for DiscreteGaussian {
    type Output = i64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> i64 {
        if self.numerator == 0 {
            return self.mean;
        }

        loop {
            let offset: i64 = discrete_laplace(source, self.scale, 1);

            let Some(scaled) = (offset.unsigned_abs() as u128).checked_mul(self.unit) else {
                continue;
            };
            let Some(exponent) = scaled
                .abs_diff(self.numerator)
                .checked_mul(scaled.abs_diff(self.numerator))
            else {
                continue;
            };

            if chance_exp_minus(source, exponent, self.exponent_denominator) {
                return self.mean.saturating_add(offset);
            }
        }
    }
}

impl PortableDistribution for BernoulliRatio {}
impl PortableDistribution for BernoulliMask {}
impl PortableDistribution for BinomialRatio {}
impl PortableDistribution for PoissonRatio {}
impl PortableDistribution for DiscreteLaplace {}
impl PortableDistribution for DiscreteGaussian {}
