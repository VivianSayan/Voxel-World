//! Integer-valued distributions over floating-point parameters, and the
//! geometric samplers over [`Ratio`]s.
//!
//! The fully integer versions of the Bernoulli, binomial and Poisson
//! distributions are in [`exact`](super::exact).

use super::{Distribution, PortableDistribution};
use crate::random::source::RandomSource;
use crate::units::{Probability, Rate, Ratio};
use crate::units::Unit;
use std::f64::consts::PI;

/// Cost heuristic for switching from exact trials to approximate Q64 inversion.
/// See the sampling benchmark before tuning this for a particular workload.
const COUNTING_LIMIT: u64 = 256;

// ---------------------------------------------------------------------------
// Uniform integers
// ---------------------------------------------------------------------------

/// Uniform integer in `[low, high]`, both inclusive.
///
/// Exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformU64 {
    low: u64,
    span: u64,
}

impl UniformU64 {
    /// `None` when `low > high`.
    pub fn new(low: u64, high: u64) -> Option<Self> {
        (low <= high).then(|| Self {
            low,
            span: high - low,
        })
    }
}

impl Distribution for UniformU64 {
    type Output = u64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        if self.span == u64::MAX {
            return source.next_u64();
        }
        self.low + source.bounded_u64(self.span + 1)
    }
}

/// Uniform integer in `[low, high]`, both inclusive.
///
/// Exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformI64 {
    low: i64,
    span: u64,
}

impl UniformI64 {
    /// `None` when `low > high`.
    pub fn new(low: i64, high: i64) -> Option<Self> {
        (low <= high).then(|| Self {
            low,
            span: (high as u64).wrapping_sub(low as u64),
        })
    }
}

impl Distribution for UniformI64 {
    type Output = i64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> i64 {
        if self.span == u64::MAX {
            return source.next_u64() as i64;
        }
        self.low
            .wrapping_add(source.bounded_u64(self.span + 1) as i64)
    }
}

impl PortableDistribution for UniformU64 {}
impl PortableDistribution for UniformI64 {}

// ---------------------------------------------------------------------------
// Bernoulli, binomial, Poisson
// ---------------------------------------------------------------------------

/// True with the given chance.
///
/// One draw and one comparison, no transcendentals. The comparison is `<` against
/// a half-open draw, so [`Probability::NEVER`] never fires and
/// [`Probability::ALWAYS`] always does.
///
/// # Precision
///
/// The chance is held as a [`Unit`], on the `2^-63` grid, and the draw is taken at
/// the same width. So the realised frequency is the stated chance to within
/// `2^-64` rather than the `2^-53` a 53-bit draw would give — a thousandfold
/// tighter, at no extra cost, since both read one word.
///
/// Building one from a [`Probability`] still rounds that `f64` onto the grid, so a
/// chance below `2^-64` becomes never. That was already true of the 53-bit draw
/// this replaced, and at a coarser threshold: `2^-53`.
///
/// For a chance that is a fraction such as one in three, which no `f64` holds
/// exactly at all, see [`BernoulliRatio`](super::BernoulliRatio).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bernoulli {
    chance: Unit,
}

impl Bernoulli {
    /// A coin at the given chance. Always valid, since a [`Probability`] is
    /// already in `[0, 1]`.
    pub fn new(chance: Probability) -> Self {
        Self {
            chance: Unit::from_probability(chance),
        }
    }

    /// A coin at a chance given as a [`Unit`], which is the exact form.
    ///
    /// Preferred where the number matters: it skips the rounding `new` does, and a
    /// `Unit` can express chances near one that an `f64` cannot separate from
    /// certainty.
    pub const fn at(chance: Unit) -> Self {
        Self { chance }
    }

    /// The chance this coin fires at.
    pub const fn chance(self) -> Unit {
        self.chance
    }
}

impl Distribution for Bernoulli {
    type Output = bool;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> bool {
        self.chance.decide_from(source)
    }
}

impl PortableDistribution for Bernoulli {}

/// Number of successes in `trials` independent trials at the given chance.
///
/// Sequential inversion when few successes are expected, transformed
/// rejection (Hörmann, 1993) otherwise: a handful of draws whatever the size.
///
/// A chance above one half is sampled as its complement and the count
/// mirrored, since both algorithms assume the smaller side.
///
/// Platform-dependent: uses `ln` and `exp`. The exact version, for a
/// [`Ratio`] chance, is [`BinomialRatio`](super::BinomialRatio).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binomial {
    trials: u64,
    chance: Probability,
}

impl Binomial {
    /// A count of successes over `trials` independent trials at this chance.
    /// Always valid; zero trials or a zero chance simply give zero.
    pub fn new(trials: u64, chance: Probability) -> Self {
        Self { trials, chance }
    }
}

impl Distribution for Binomial {
    type Output = u64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        let trials: u64 = self.trials;
        let p: f64 = self.chance.value();

        if trials == 0 || p == 0.0 {
            return 0;
        }
        if p == 1.0 {
            return trials;
        }

        let flipped: bool = p > 0.5;
        let p: f64 = if flipped { 1.0 - p } else { p };

        let successes: u64 = if trials as f64 * p < 10.0 {
            binomial_inversion(source, trials, p)
        } else {
            binomial_btrs(source, trials, p)
        };

        if flipped {
            trials - successes
        } else {
            successes
        }
    }
}

/// Sequential inversion, fast when trials * p is small.
///
/// Walks the distribution from zero, subtracting each outcome's probability
/// from the draw until it runs out. The whole walk restarts on the rare
/// occasion that accumulated rounding carries it past the last outcome.
fn binomial_inversion<S: RandomSource + ?Sized>(source: &mut S, trials: u64, p: f64) -> u64 {
    let q: f64 = 1.0 - p;
    let ratio: f64 = p / q;
    let a: f64 = (trials + 1) as f64 * ratio;
    let start: f64 = (trials as f64 * (-p).ln_1p()).exp();

    'retry: loop {
        let mut u: f64 = source.unit_f64();
        let mut probability: f64 = start;
        let mut x: u64 = 0;

        while u > probability {
            u -= probability;
            x += 1;
            if x > trials {
                continue 'retry;
            }
            probability *= a / x as f64 - ratio;
        }

        return x;
    }
}

/// Transformed rejection with squeeze (Hörmann, 1993), for trials * p >= 10.
fn binomial_btrs<S: RandomSource + ?Sized>(source: &mut S, trials: u64, p: f64) -> u64 {
    let n: f64 = trials as f64;
    let q: f64 = 1.0 - p;
    let spq: f64 = (n * p * q).sqrt();

    let b: f64 = 1.15 + 2.53 * spq;
    let a: f64 = -0.0873 + 0.0248 * b + 0.01 * p;
    let c: f64 = n * p + 0.5;
    let v_r: f64 = 0.92 - 4.2 / b;
    let alpha: f64 = (2.83 + 5.1 / b) * spq;
    let log_ratio: f64 = (p / q).ln();
    let mode: f64 = ((n + 1.0) * p).floor();
    let h: f64 = ln_gamma(mode + 1.0) + ln_gamma(n - mode + 1.0);

    loop {
        let u: f64 = source.unit_f64() - 0.5;
        let v: f64 = source.open_unit_f64();
        let us: f64 = 0.5 - u.abs();
        let k: f64 = ((2.0 * a / us + b) * u + c).floor();

        if k < 0.0 || k > n {
            continue;
        }
        if us >= 0.07 && v <= v_r {
            return k as u64;
        }

        let lhs: f64 = (v * alpha / (a / (us * us) + b)).ln();
        let rhs: f64 = h - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0) + (k - mode) * log_ratio;

        if lhs <= rhs {
            return k as u64;
        }
    }
}

/// Poisson with the given mean.
///
/// Inversion below a mean of 10, transformed rejection (Hörmann, 1993) above.
///
/// Platform-dependent: uses `exp` and `ln`. The exact version, for a
/// [`Ratio`] mean, is [`PoissonRatio`](super::PoissonRatio).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poisson {
    mean: Rate,
}

impl Poisson {
    /// A Poisson count with this mean. Always valid, since a [`Rate`] is
    /// already positive and finite.
    pub fn new(mean: Rate) -> Self {
        Self { mean }
    }
}

impl Distribution for Poisson {
    type Output = u64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        let lambda: f64 = self.mean.value();

        if lambda < 10.0 {
            poisson_inversion(source, lambda)
        } else {
            poisson_ptrs(source, lambda)
        }
    }
}

fn poisson_inversion<S: RandomSource + ?Sized>(source: &mut S, lambda: f64) -> u64 {
    'retry: loop {
        let u: f64 = source.unit_f64();
        let mut probability: f64 = (-lambda).exp();
        let mut cumulative: f64 = probability;
        let mut k: u64 = 0;

        while u > cumulative {
            k += 1;
            if k > 1000 {
                continue 'retry;
            }
            probability *= lambda / k as f64;
            cumulative += probability;
        }

        return k;
    }
}

/// Transformed rejection (Hörmann, 1993), for lambda >= 10.
fn poisson_ptrs<S: RandomSource + ?Sized>(source: &mut S, lambda: f64) -> u64 {
    let sqrt_lambda: f64 = lambda.sqrt();
    let ln_lambda: f64 = lambda.ln();
    let b: f64 = 0.931 + 2.53 * sqrt_lambda;
    let a: f64 = -0.059 + 0.02483 * b;
    let ln_inv_alpha: f64 = (1.1239 + 1.1328 / (b - 3.4)).ln();
    let v_r: f64 = 0.9277 - 3.6224 / (b - 2.0);

    loop {
        let u: f64 = source.unit_f64() - 0.5;
        let v: f64 = source.open_unit_f64();
        let us: f64 = 0.5 - u.abs();
        let k: f64 = ((2.0 * a / us + b) * u + lambda + 0.43).floor();

        if us >= 0.07 && v <= v_r {
            return k as u64;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }

        let lhs: f64 = v.ln() + ln_inv_alpha - (a / (us * us) + b).ln();
        let rhs: f64 = -lambda + k * ln_lambda - ln_gamma(k + 1.0);

        if lhs <= rhs {
            return k as u64;
        }
    }
}

// ---------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------

/// Rounds a value to a whole number, up or down, so that the average over many
/// draws is the value itself.
///
/// For a quantity that is meaningful in aggregate but has to be whole each
/// time: 0.3 trees per chunk becomes a tree in roughly three chunks out of
/// ten, rather than none in every chunk or one in every chunk.
///
/// Exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StochasticRound {
    value: f64,
}

impl StochasticRound {
    /// `None` for a non-finite value.
    pub fn new(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self { value })
    }
}

impl Distribution for StochasticRound {
    type Output = i64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> i64 {
        let floor: f64 = self.value.floor();
        let fraction: f64 = self.value - floor;
        // The fractional part is exactly the chance of rounding up, so this is a
        // coin — decided on the `2^-63` grid rather than the 53-bit one.
        let rounded: f64 = if Unit::clamped(fraction).decide_from(source) {
            floor + 1.0
        } else {
            floor
        };

        rounded.clamp(i64::MIN as f64, i64::MAX as f64) as i64
    }
}

impl PortableDistribution for StochasticRound {}

// ---------------------------------------------------------------------------
// Geometric
// ---------------------------------------------------------------------------

/// How many failures before the first success, at the given chance.
///
/// The discrete twin of [`Exponential`](super::Exponential): how many steps
/// until the next event, rather than how long. Use it for the spacing between
/// things scattered along a line, a row of voxels, or a run of ticks.
///
/// A chance of zero never succeeds, so it gives `u64::MAX` rather than looping
/// forever.
///
/// Platform-dependent: uses `ln`. [`GeometricRatio`] is the integer version.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometric {
    chance: Probability,
}

impl Geometric {
    /// A count of failures before the first success at this chance. Always
    /// valid; a chance of zero gives `u64::MAX` rather than looping.
    pub fn new(chance: Probability) -> Self {
        Self { chance }
    }
}

impl Distribution for Geometric {
    type Output = u64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        let p: f64 = self.chance.value();

        if p >= 1.0 {
            return 0;
        }

        if p <= 0.0 {
            return u64::MAX;
        }

        let failures: f64 = source.open_unit_f64().ln() / (-p).ln_1p();

        if failures >= u64::MAX as f64 {
            u64::MAX
        } else {
            failures as u64
        }
    }
}

/// Failures before the first success at a rational chance, with the inverse
/// table for rare chances built once rather than on every draw.
///
/// Integer-only and reproducible. Counting is unbiased when
/// `floor(denominator / numerator) <= 256`. Rarer events use a Q64 inverse
/// approximation: truncation slightly favors shorter runs, and its error grows
/// for probabilities near `2^-64`. It is not an exact sampler there. Zero
/// gives `u64::MAX`; unrepresentable waits saturate at `u64::MAX`.
#[derive(Clone, Debug)]
pub struct GeometricRatio {
    chance: Ratio,
    inverse: Option<GeometricInverse>,
}

impl GeometricRatio {
    /// Rejects ratios greater than one.
    pub fn new(chance: Ratio) -> Option<Self> {
        if !chance.is_proper() {
            return None;
        }
        let numerator: u64 = chance.numerator();
        let denominator: u64 = chance.denominator();
        let inverse: Option<GeometricInverse> = (numerator > 0
            && denominator / numerator > COUNTING_LIMIT)
            .then(|| GeometricInverse::new(numerator, denominator));
        Some(Self { chance, inverse })
    }

    /// A zero count means never; a count of one succeeds immediately.
    pub fn one_in(count: u64) -> Self {
        Self::new(Ratio::one_in(count).unwrap_or(Ratio::ZERO)).unwrap()
    }

    /// The chance this was built with, reduced.
    pub fn chance(&self) -> Ratio {
        self.chance
    }
}

impl Distribution for GeometricRatio {
    type Output = u64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        match &self.inverse {
            Some(inverse) => inverse.sample(source),
            None => geometric_ratio(source, self.chance),
        }
    }
}

impl PortableDistribution for GeometricRatio {}

/// [`GeometricRatio`] without the cached table. Panics for an improper ratio.
///
/// Picks between three paths. A chance of one in a power of two is answered by
/// [`geometric_in_power_of_two`], which reads whole words of trials at a time.
/// Any other chance of one in at most [`COUNTING_LIMIT`] counts trials with
/// [`geometric_by_chunks`], which is exact and, while the expected run is
/// short, cheaper than inverting: a handful of trials beats sixty squarings.
/// Past that limit the two swap places and never swap back, since counting
/// grows with the answer and inverting does not, so a rarer chance builds a
/// table and inverts.
pub(in crate::random) fn geometric_ratio<S: RandomSource + ?Sized>(
    source: &mut S,
    chance: Ratio,
) -> u64 {
    assert!(chance.is_proper(), "a probability must be at most one");
    let numerator: u64 = chance.numerator();
    let denominator: u64 = chance.denominator();

    if numerator == 0 {
        return u64::MAX;
    }

    if numerator >= denominator {
        return 0;
    }

    if denominator / numerator <= COUNTING_LIMIT {
        if numerator == 1 && denominator.is_power_of_two() {
            return geometric_in_power_of_two(source, denominator.trailing_zeros());
        }

        return geometric_by_chunks(source, numerator, denominator);
    }

    GeometricInverse::new(numerator, denominator).sample(source)
}

/// [`geometric_ratio`] at one in `count`, skipping the ratio's checks.
pub(in crate::random) fn geometric_one_in<S: RandomSource + ?Sized>(
    source: &mut S,
    count: u64,
) -> u64 {
    match count {
        0 => u64::MAX,
        1 => 0,
        _ if count > COUNTING_LIMIT => GeometricInverse::new(1, count).sample(source),
        _ if count.is_power_of_two() => geometric_in_power_of_two(source, count.trailing_zeros()),
        _ => geometric_by_chunks(source, 1, count),
    }
}

/// Failures before a run of `bits` zero bits comes up, which is a chance of
/// one in `2^bits`.
///
/// Every `bits`-wide slice of a uniform word is itself uniform and they are
/// independent, so one word carries `64 / bits` trials. Rather than walk those
/// slices, the whole word is reduced at once: or-ing the word into itself
/// `bits - 1` times marks every slice that holds a one, and the first slice
/// left unmarked is the first success. A run of a million failures costs a
/// million divided by the slices per word, and each of those words costs a
/// handful of instructions rather than a loop.
///
/// The marker pattern holds one bit at the bottom of each whole slice. Where
/// the slice divides the word evenly that pattern is exactly `u64::MAX / mask`,
/// which is worth the special case: the widths that divide evenly are also the
/// ones with the most slices, and at one bit per slice building the pattern by
/// loop would cost sixty-four iterations to save a single draw.
///
/// Folding a word means or-ing it onto itself shifted right, less than a full
/// slice at a time, so no slice can ever be marked by its neighbour's bits. A
/// slice left clear is a success, and the lowest clear one is the first.
fn geometric_in_power_of_two<S: RandomSource + ?Sized>(source: &mut S, bits: u32) -> u64 {
    debug_assert!((1..64).contains(&bits));

    let per_word: u32 = u64::BITS / bits;
    let mask: u64 = (1u64 << bits) - 1;

    let starts: u64 = if u64::BITS % bits == 0 {
        u64::MAX / mask
    } else {
        (0..per_word).fold(0u64, |pattern, slice| pattern | (1u64 << (slice * bits)))
    };

    let mut failures: u64 = 0;

    loop {
        let word: u64 = source.next_u64();

        let mut marked: u64 = word;
        for shift in 1..bits {
            marked |= word >> shift;
        }

        let clear: u64 = !marked & starts;

        if clear != 0 {
            return failures.saturating_add((clear.trailing_zeros() / bits) as u64);
        }

        failures = failures.saturating_add(per_word as u64);

        if failures == u64::MAX {
            return u64::MAX;
        }
    }
}

/// Failures before a draw below `denominator` lands under `numerator`,
/// reading several trials from each word.
///
/// A trial only needs as many bits as it takes to write `denominator`, not a
/// whole word, so one word carries several. Slices that land at or above
/// `denominator` are thrown away rather than folded back in, which is what
/// keeps the surviving slices uniform below `denominator` and independent of
/// one another. The waste is bounded: the slice is the smallest power of two
/// that fits, so at worst just under half of them are discarded, and for a
/// denominator that is already a power of two none are.
///
/// A denominator past half the range leaves no room to pack more than one trial
/// per word, so that case falls back to a whole bounded draw per trial.
fn geometric_by_chunks<S: RandomSource + ?Sized>(
    source: &mut S,
    numerator: u64,
    denominator: u64,
) -> u64 {
    let bits: u32 = u64::BITS - (denominator - 1).leading_zeros();

    if bits >= u64::BITS {
        let mut failures: u64 = 0;

        while source.bounded_u64(denominator) >= numerator {
            failures = failures.saturating_add(1);

            if failures == u64::MAX {
                return u64::MAX;
            }
        }

        return failures;
    }

    let mask: u64 = (1u64 << bits) - 1;
    let per_word: u32 = u64::BITS / bits;
    let mut failures: u64 = 0;

    loop {
        let mut word: u64 = source.next_u64();

        for _ in 0..per_word {
            let value: u64 = word & mask;

            word >>= bits;

            if value < numerator {
                return failures;
            }

            if value < denominator {
                failures = failures.saturating_add(1);

                if failures == u64::MAX {
                    return u64::MAX;
                }
            }
        }
    }
}

/// Q64 survival powers. Each is below one, so only the products need u128.
#[derive(Clone, Debug)]
struct GeometricInverse {
    powers: [u64; 64],
    levels: usize,
}

impl GeometricInverse {
    fn new(numerator: u64, denominator: u64) -> Self {
        debug_assert!(numerator > 0 && numerator < denominator);
        let mut powers: [u64; 64] = [0; 64];
        powers[0] = ((((denominator - numerator) as u128) << 64) / denominator as u128) as u64;
        let mut levels: usize = 1;
        while levels < powers.len() {
            let previous: u128 = powers[levels - 1] as u128;
            let squared: u64 = ((previous * previous) >> 64) as u64;
            if squared == 0 {
                break;
            }
            powers[levels] = squared;
            levels += 1;
        }
        Self { powers, levels }
    }

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        let draw: u128 = source.next_u64() as u128;
        let mut survived: u128 = 1u128 << 64;
        let mut failures: u64 = 0;
        for level in (0..self.levels).rev() {
            let reached: u128 = (survived * self.powers[level] as u128) >> 64;
            if reached > draw {
                survived = reached;
                failures |= 1u64 << level;
            }
        }
        failures
    }
}

// ---------------------------------------------------------------------------
// Zipf
// ---------------------------------------------------------------------------

/// Zipf over `1..=elements` with `P(k)` proportional to `k^-exponent`.
///
/// Rejection-inversion (Hörmann & Derflinger, 1996): O(1) per sample, no
/// tables.
///
/// Platform-dependent: uses `ln` and `exp`.
#[derive(Clone, Debug)]
pub struct Zipf {
    elements: f64,
    exponent: f64,
    h_integral_x1: f64,
    h_integral_n: f64,
    s: f64,
}

impl Zipf {
    /// A Zipf distribution over `1..=elements` at this exponent, with the
    /// constants the rejection method needs computed once here.
    ///
    /// `None` for no elements, for an exponent that is not positive and finite,
    /// and for parameters whose constants overflow, which takes an enormous
    /// element count at a small exponent.
    pub fn new(elements: u64, exponent: f64) -> Option<Self> {
        if elements == 0 || !exponent.is_finite() || exponent <= 0.0 {
            return None;
        }

        let mut zipf: Self = Self {
            elements: elements as f64,
            exponent,
            h_integral_x1: 0.0,
            h_integral_n: 0.0,
            s: 0.0,
        };

        zipf.h_integral_x1 = zipf.h_integral(1.5) - 1.0;
        zipf.h_integral_n = zipf.h_integral(zipf.elements + 0.5);
        zipf.s = 2.0 - zipf.h_integral_inverse(zipf.h_integral(2.5) - zipf.h(2.0));
        (zipf.h_integral_x1.is_finite() && zipf.h_integral_n.is_finite() && zipf.s.is_finite())
            .then_some(zipf)
    }

    fn h(&self, x: f64) -> f64 {
        (-self.exponent * x.ln()).exp()
    }

    /// Integral of h, (x^(1 - s) - 1) / (1 - s), stable near s = 1.
    fn h_integral(&self, x: f64) -> f64 {
        let ln_x: f64 = x.ln();
        exp_m1_over_x((1.0 - self.exponent) * ln_x) * ln_x
    }

    fn h_integral_inverse(&self, x: f64) -> f64 {
        let t: f64 = (x * (1.0 - self.exponent)).max(-1.0);
        (ln_1p_over_x(t) * x).exp()
    }
}

impl Distribution for Zipf {
    type Output = u64;

    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> u64 {
        loop {
            let u: f64 =
                self.h_integral_n + source.unit_f64() * (self.h_integral_x1 - self.h_integral_n);
            let x: f64 = self.h_integral_inverse(u);
            let k: f64 = (x + 0.5).floor().clamp(1.0, self.elements);

            if k - x <= self.s || u >= self.h_integral(k + 0.5) - self.h(k) {
                return k as u64;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// ln(Gamma(x)) for x > 0 (Lanczos approximation, g = 7).
///
/// Accurate to about fifteen digits over the range the samplers use it for.
/// Arguments below one half are reflected through
/// `Gamma(x) Gamma(1 - x) = pi / sin(pi x)`, since the series converges only to
/// the right of that.
pub(super) fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x: f64 = x - 1.0;
    let mut sum: f64 = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        sum += coefficient / (x + index as f64);
    }

    let t: f64 = x + 7.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// ln(1 + x) / x, continuous at 0.
fn ln_1p_over_x(x: f64) -> f64 {
    if x.abs() > 1e-8 {
        x.ln_1p() / x
    } else {
        1.0 - x * (0.5 - x * (1.0 / 3.0 - 0.25 * x))
    }
}

/// (e^x - 1) / x, continuous at 0.
fn exp_m1_over_x(x: f64) -> f64 {
    if x.abs() > 1e-8 {
        x.exp_m1() / x
    } else {
        1.0 + x * 0.5 * (1.0 + x / 3.0 * (1.0 + 0.25 * x))
    }
}
