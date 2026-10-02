//! Numbers that mean something only inside a range.
//!
//! All three of these are `f64` underneath, and all three are ranges that code
//! keeps assuming and never checking: a probability outside `[0, 1]` is a bug
//! that shows up as a decision that always goes one way, and a noise value
//! outside `[-1, 1]` is one that shows up as terrain clipping through its own
//! limits.
//!
//! There is no way to build any of them outside its range, so nothing
//! downstream has to check and nothing has to say what it assumes. Each has two
//! constructors for the two ways a value arrives: `new`, which returns `None`
//! for anything out of range and suits values from configuration or a save, and
//! `clamped`, which folds the value into range and suits results of arithmetic
//! that may drift a fraction outside it.

use crate::math::rational::Ratio;
use crate::random::seed::Seed;
use crate::math::Unit;
use crate::random::distributions::{Distribution, PortableDistribution};
use crate::random::source::StochasticSource;
use std::fmt;

/// The largest `f64` below 1.0, which is where a value at the top of a
/// half-open range lands.
const JUST_BELOW_ONE: f64 = f64::from_bits(0x3FEF_FFFF_FFFF_FFFF);

/// A probability in `[0, 1]`.
///
/// Closed at both ends, unlike [`UnitValue`], because never and always are both
/// things worth being able to say: a voxel type that is switched off entirely,
/// a check that must always pass.
///
/// The combining methods ([`Probability::and`], [`Probability::or`],
/// [`Probability::in_any_of`]) all assume the events are independent, which is
/// the common case for world generation, where each decision reads its own
/// seed. They are not valid for events that share a cause.
///
/// # This type, or [`Unit`]?
///
/// [`Unit`] holds the same range as an integer count of
/// `2^-63`. Prefer it for **decisions and stored chances**: its complement is
/// exact, where `1.0 - 1e-20` in floating point is simply `1.0`, so chains of
/// `and`, `or` and `in_any_of` keep what this loses. It is also [`Ord`], [`Eq`] and
/// [`Hash`], which this cannot be while an `f64` carries NaN.
///
/// Prefer **this** type for a **continuous parameter**: the shape of a
/// [`Geometric`](crate::random::distributions::Geometric) or a
/// [`Binomial`](crate::random::distributions::Binomial), where the value is fed to
/// `ln` or multiplied by a large count and a very small magnitude is meaningful. A
/// chance of `10^-30` gives a geometric mean of `10^30` failures, which an `f64`
/// expresses and a `Unit` would floor to zero.
///
/// # What the small-value range does *not* buy
///
/// It does **not** make tiny chances samplable, and it never did. A coin compares a
/// uniform draw against the chance, and the draw has a grid of its own — so the
/// realised frequency has a floor no matter how small the stored number is.
///
/// Before [`Probability::decide`] was routed through a [`Unit`], that floor was
/// `2^-53`: a chance of `10^-20` did not fire at `10^-20`, it fired at about
/// `1.1 × 10^-16`, four orders of magnitude too often, because only the single draw
/// `0` could satisfy the comparison. It now floors at `2^-64` instead, and a chance
/// below that fires never rather than at a spuriously high rate.
///
/// So the `10^-308` an `f64` reaches is real for *arithmetic and storage*, and
/// fiction for *deciding*. Where a genuinely tiny chance has to be sampled, use
/// [`Seed::chance_ratio`](crate::random::seed::Seed::chance_ratio) or
/// [`BernoulliRatio`](crate::random::distributions::BernoulliRatio), which are
/// exact for any fraction and have no floor at all.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Debug)]
pub struct Probability(f64);

impl Probability {
    /// Never: the chance zero. [`Probability::decide`] always says no.
    pub const NEVER: Self = Self(0.0);
    /// Always: the chance one. [`Probability::decide`] always says yes.
    pub const ALWAYS: Self = Self(1.0);
    /// A coin: the chance one half.
    pub const EVEN: Self = Self(0.5);

    /// The chance as given, or `None` if it is NaN or outside `[0, 1]`.
    ///
    /// The constructor for a value that came from outside the program, such as
    /// configuration or a save file, where a bad value is the caller's problem
    /// and should be reported rather than quietly corrected.
    pub fn new(chance: f64) -> Option<Self> {
        if chance.is_nan() || !(0.0..=1.0).contains(&chance) {
            return None;
        }

        Some(Self(chance))
    }

    /// The chance brought into `[0, 1]`: anything below zero becomes
    /// [`Self::NEVER`], anything above one becomes [`Self::ALWAYS`], and NaN
    /// becomes [`Self::NEVER`].
    ///
    /// The constructor for a value that came out of arithmetic which can drift
    /// a little past an end, where landing exactly on that end is the right
    /// answer rather than an error.
    pub fn clamped(chance: f64) -> Self {
        if chance.is_nan() {
            Self(0.0)
        } else {
            Self(chance.clamp(0.0, 1.0))
        }
    }

    /// One chance in `count`, or [`Self::NEVER`] when `count` is zero.
    ///
    /// Reads the way the design reads: one ore vein in 400 voxels is
    /// `Probability::one_in(400)`. The division is inexact for most counts; for
    /// a chance that must be exactly one in `count`, use
    /// [`Ratio::one_in`](crate::units::Ratio::one_in) and the exact samplers
    /// built on it.
    pub fn one_in(count: u64) -> Self {
        if count == 0 {
            return Self::NEVER;
        }

        Self(1.0 / count as f64)
    }

    /// `hits` out of `total`, or [`Self::NEVER`] when `total` is zero.
    ///
    /// Clamped, so more hits than the total gives [`Self::ALWAYS`] rather than
    /// a value above one. As with [`Probability::one_in`], the division rounds;
    /// [`Ratio::out_of`](crate::units::Ratio::out_of) keeps the fraction whole.
    pub fn ratio(hits: u64, total: u64) -> Self {
        if total == 0 {
            return Self::NEVER;
        }

        Self::clamped(hits as f64 / total as f64)
    }

    /// The chance as a plain `f64` in `[0, 1]`.
    pub const fn value(self) -> f64 {
        self.0
    }

    /// The chance of this not happening, `1 - p`.
    ///
    /// Exact at both ends, and for probabilities near one it loses precision in
    /// the same way `1 - p` always does: the complement of a chance within
    /// `2^-53` of one is zero.
    pub fn complement(self) -> Self {
        Self(1.0 - self.0)
    }

    /// The chance of this and `other` both happening, `p * q`, treating them as
    /// independent.
    pub fn and(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    /// The chance of this or `other` happening, or both, treating them as
    /// independent.
    ///
    /// Evaluated as `larger + (1 - larger) * smaller` rather than the usual
    /// `p + q - p * q`. The two agree mathematically, but ordering the operands
    /// keeps the subtraction away from the small value, so a large chance
    /// combined with a tiny one keeps the tiny one's contribution instead of
    /// rounding it away.
    pub fn or(self, other: Self) -> Self {
        let larger = self.0.max(other.0);
        let smaller = self.0.min(other.0);
        Self::clamped(larger + (1.0 - larger) * smaller)
    }

    /// The chance of this happening at least once in `tries` independent
    /// attempts: mathematically `1 - (1 - p)^tries`.
    ///
    /// Computed by binary exponentiation in probability space, doubling with
    /// [`Probability::or`] rather than raising `1 - p` to a power. Two reasons:
    /// `powi` would need the exponent as a signed integer, and `1 - p` for a
    /// tiny `p` rounds straight to one, which would report an impossible run of
    /// attempts as impossible. Costs about `log2(tries)` multiplications.
    pub fn in_any_of(self, tries: u32) -> Self {
        let mut remaining = tries;
        let mut block = self;
        let mut result = Self::NEVER;
        while remaining != 0 {
            if remaining & 1 != 0 {
                result = result.or(block);
            }
            remaining >>= 1;
            if remaining != 0 {
                block = block.or(block);
            }
        }
        result
    }

    /// Whether it happens, for a given seed.
    ///
    /// The same seed always decides the same way, which is what makes this
    /// usable for world content rather than only for effects. The seed's
    /// fraction is compared with `<`, and that fraction is half-open, so
    /// [`Self::NEVER`] never fires and [`Self::ALWAYS`] always does.
    ///
    /// The comparison itself is exact, but the chance often is not: a
    /// probability written as a fraction, such as one in three, is stored as
    /// the nearest `f64`. Where that matters, use
    /// [`Seed::one_in`](crate::random::seed::Seed::one_in) or
    /// [`Seed::chance_ratio`](crate::random::seed::Seed::chance_ratio).
    ///
    /// Decided through a [`Unit`], so the realised frequency matches the stored
    /// chance to within `2^-64`. A chance below that never fires — the type can
    /// hold such a number but no comparison against a uniform draw can realise it;
    /// see the note on this type about what the small-value range does not buy.
    pub fn decide(self, seed: Seed) -> bool {
        // Through `Unit`, so the realised frequency matches the stated chance to
        // `2^-64` rather than the `2^-53` a 53-bit draw gives — and so this agrees
        // with `Bernoulli`, which decides the same way.
        Unit::from_probability(self).decide(seed)
    }
}

impl fmt::Display for Probability {
    /// As a percentage with one decimal, such as `12.5%`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.1}%", self.0 * 100.0)
    }
}

/// A noise sample in `[-1, 1]`.
///
/// What every field in [`crate::spatial::noise`] produces. Keeping it apart
/// from a bare `f64` matters most where noise is mixed with heights, densities
/// and thresholds, which are all `f64` too and none of which are in this range.
///
/// The methods here are the shaping steps a generator applies between sampling
/// a field and using it: rescaling it onto a range, folding it into ridges,
/// blending two fields, thresholding it into solid and empty. Each one either
/// cannot leave the range or clamps back into it, so a chain of them is still a
/// noise value at the end.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Debug)]
pub struct NoiseValue(f64);

impl NoiseValue {
    /// The middle of the range, which is where a smooth field sits on average.
    pub const ZERO: Self = Self(0.0);
    /// The bottom of the range.
    pub const LOWEST: Self = Self(-1.0);
    /// The top of the range.
    pub const HIGHEST: Self = Self(1.0);

    /// The sample as given, or `None` if it is NaN or outside `[-1, 1]`.
    pub fn new(value: f64) -> Option<Self> {
        if value.is_nan() || !(-1.0..=1.0).contains(&value) {
            return None;
        }

        Some(Self(value))
    }

    /// The sample brought into `[-1, 1]`, with NaN becoming [`Self::ZERO`].
    ///
    /// The constructor the noise fields themselves use: summing octaves and
    /// dividing by their total amplitude can land a fraction of a bit outside
    /// the range, and the honest answer there is the end of the range.
    pub fn clamped(value: f64) -> Self {
        if value.is_nan() {
            Self(0.0)
        } else {
            Self(value.clamp(-1.0, 1.0))
        }
    }

    /// The sample as a plain `f64` in `[-1, 1]`.
    pub const fn value(self) -> f64 {
        self.0
    }

    /// The same sample rescaled from `[-1, 1]` to `[0, 1]`.
    pub fn to_unit(self) -> f64 {
        self.0 * 0.5 + 0.5
    }

    /// The same sample as a [`Probability`], by the rescaling in
    /// [`NoiseValue::to_unit`].
    ///
    /// Turns a field into a chance that varies over space: denser ore where the
    /// field runs high, none where it runs low.
    pub fn to_probability(self) -> Probability {
        Probability::clamped(self.to_unit())
    }

    /// The sample mapped linearly onto `[low, high]`, which is what turns a
    /// field into a height, a density or a temperature.
    ///
    /// The bounds may run either way: passing a `high` below `low` inverts the
    /// field.
    pub fn remap(self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.to_unit()
    }

    /// Distance from zero, which folds the bottom half of the range up onto the
    /// top.
    ///
    /// The building block of ridged terrain: what was a smooth crossing of zero
    /// becomes a crease. [`NoiseValue::ridged`] is this, recentred.
    pub fn magnitude(self) -> Self {
        Self(self.0.abs())
    }

    /// The sample turned inside out, `1 - 2|v|`, so that what was near zero is
    /// now near the top of the range and what was near either end is now near
    /// the bottom.
    ///
    /// The usual way of making ridges out of a smooth field: the sharp line is
    /// where the original field crossed zero. Stays in range because `|v|` is
    /// in `[0, 1]`.
    pub fn ridged(self) -> Self {
        Self(1.0 - 2.0 * self.0.abs())
    }

    /// The two samples mixed, `weight` of the way from this one to `other`.
    ///
    /// A weight of [`Unit::ZERO`] gives this sample and [`Unit::ONE`] gives `other`.
    /// Written as `self + (other - self) * weight`, which is exact at both ends,
    /// unlike `self * (1 - weight) + other * weight`. Stays in range because both
    /// ends are in range.
    ///
    /// The weight is a [`Unit`] rather than a [`Probability`]: it is a position
    /// along an interval, not the chance of anything, and saying so keeps the two
    /// from being passed to each other by accident.
    pub fn blend(self, other: Self, weight: Unit) -> Self {
        Self::clamped(self.0 + (other.0 - self.0) * weight.to_f64())
    }

    /// Whether the sample is strictly above `threshold`, the usual way a
    /// density field becomes solid or empty.
    pub fn exceeds(self, threshold: Self) -> bool {
        self.0 > threshold.0
    }

    /// The sample scaled towards zero by `amount`.
    ///
    /// How an octave's amplitude is applied. Amplitudes only ever shrink, since a
    /// [`Unit`] is at most one, so the range holds.
    ///
    /// A [`Unit`] rather than a [`Probability`] for the same reason as
    /// [`NoiseValue::blend`]: an amplitude is a fraction of a range, not a chance.
    pub fn scaled(self, amount: Unit) -> Self {
        Self(self.0 * amount.to_f64())
    }
}

impl fmt::Display for NoiseValue {
    /// With a sign and four decimals, such as `-0.3125`, so that a column of
    /// samples lines up.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:+.4}", self.0)
    }
}

/// A value in `[0, 1)`, which is what a raw hash becomes and what
/// [`crate::spatial::position::LocalPosition3`] is made of per axis.
///
/// Half-open at the top, which is the property that makes it safe to scale: a
/// list of `n` things can be indexed by multiplying by `n` without ever landing
/// on `n` itself. That is why the generators produce this rather than a
/// [`Probability`], which includes its top end.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Debug)]
pub struct UnitValue(f64);

impl UnitValue {
    /// The bottom of the range. There is no constant for the top, because the
    /// range excludes it.
    pub const ZERO: Self = Self(0.0);

    /// The value as given, or `None` if it is NaN or outside `[0, 1)`. One is
    /// rejected along with everything above it.
    pub fn new(value: f64) -> Option<Self> {
        (0.0..1.0).contains(&value).then_some(Self(value))
    }

    /// The value brought into `[0, 1)`: anything at or above one becomes the
    /// largest `f64` below one, and anything below zero, NaN included, becomes
    /// [`Self::ZERO`].
    pub fn clamped(value: f64) -> Self {
        if value.is_nan() || value < 0.0 {
            Self(0.0)
        } else if value >= 1.0 {
            Self(JUST_BELOW_ONE)
        } else {
            Self(value)
        }
    }

    /// The fraction a seed stands for: its top 53 bits over `2^53`.
    ///
    /// The same seed always gives the same fraction, on any machine, since the
    /// conversion is a shift and one exact multiplication.
    pub fn from_seed(seed: Seed) -> Self {
        Self(seed.unit_f64())
    }

    /// The value as a plain `f64` in `[0, 1)`.
    pub const fn value(self) -> f64 {
        self.0
    }

    /// Which of `count` equal buckets the value falls in, from `0` to
    /// `count - 1`, or `0` when `count` is zero.
    ///
    /// The multiplication cannot reach `count`, since the value is below one,
    /// but the result is clamped anyway so that no rounding can produce an
    /// out-of-bounds index.
    pub fn index_of(self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }

        ((self.0 * count as f64) as usize).min(count - 1)
    }

    /// The value spread across `[low, high)`, half-open at the top like the
    /// value itself.
    ///
    /// The result is held below `high` even when rounding in
    /// `low + (high - low) * value` would reach it, so it is safe to use as a
    /// coordinate inside a cell. Equal bounds give that bound.
    ///
    /// Panics if the bounds are not ordered and finite, or if their span
    /// overflows to infinity, since none of those can produce a value in the
    /// range asked for.
    pub fn between(self, low: f64, high: f64) -> f64 {
        assert!(
            low.is_finite() && high.is_finite() && low <= high && (high - low).is_finite(),
            "bounds must be ordered, finite, and have a finite span"
        );
        if low == high {
            return low;
        }
        (low + (high - low) * self.0).min(high.next_down()).max(low)
    }

    /// The same number as a [`Probability`], which holds because `[0, 1)` sits
    /// inside `[0, 1]`.
    pub fn to_probability(self) -> Probability {
        Probability(self.0)
    }

    /// The same number as a [`Unit`], on the `2^-63` grid.
    ///
    /// Exact: a `UnitValue` is an `f64` in `[0, 1)` and the grid is finer than any
    /// `f64` at or above `2^-11`. Below that the low bits round away, as they do
    /// for any `f64` — see [`Unit::from_probability`].
    pub fn to_unit(self) -> Unit {
        Unit::clamped(self.0)
    }

    /// The value stretched to `[-1, 1)`, the range a noise field works in.
    pub fn to_noise_value(self) -> NoiseValue {
        NoiseValue(self.0 * 2.0 - 1.0)
    }
}

impl fmt::Display for UnitValue {
    /// With six decimals, such as `0.376289`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.6}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Exact fractions as probabilities
// ---------------------------------------------------------------------------
//
// [`Ratio`] is a general rational number and lives in `math`, which knows
// nothing about probabilities; the bridge between the two belongs here.

impl Ratio {
    /// The same value as a [`Probability`], through
    /// [`Ratio::to_f64`](crate::math::Ratio::to_f64), clamped to one when
    /// improper rather than refused.
    ///
    /// This is where a ratio stops being exact, so the samplers that take a
    /// ratio directly are the better route where the frequency has to be
    /// exactly the fraction given.
    pub fn to_probability(self) -> Probability {
        Probability::clamped(self.to_f64())
    }
}

impl Probability {
    /// The chance a fraction states, clamped to one when the fraction is above
    /// it; see [`Ratio::to_probability`].
    pub fn from_ratio(ratio: Ratio) -> Self {
        ratio.to_probability()
    }
}

/// Clamped to one when improper; see [`Ratio::to_probability`].
impl From<Ratio> for Probability {
    fn from(ratio: Ratio) -> Self {
        ratio.to_probability()
    }
}

impl Unit {
    /// The same number as a [`UnitValue`], or `None` for [`Unit::ONE`].
    ///
    /// A `UnitValue` is half-open, so certainty has nowhere to go in it. That is the
    /// whole difference between the two types, and why this can fail at all.
    ///
    /// Values *just below* one need care rather than refusal: they convert to the
    /// `f64` `1.0`, which a `UnitValue` cannot hold either, so they land on the
    /// largest `f64` below one. They are genuinely below one, and the half-open
    /// range has a value for that — it simply cannot say which one.
    pub fn to_unit_value(self) -> Option<UnitValue> {
        if self.is_one() {
            return None;
        }

        Some(UnitValue(self.to_f64().min(JUST_BELOW_ONE)))
    }
}

// ---------------------------------------------------------------------------
// Drawing these types
// ---------------------------------------------------------------------------

/// A uniformly random [`Probability`].
///
/// Half-open in `[0, 1)`, like every uniform draw here — a drawn chance is never
/// exactly certain. Built from a [`Unit`] and converted, so it agrees bit for bit
/// with [`UniformUnit`](crate::units::UniformUnit) on the same word, at the 53 bits
/// an `f64` can hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniformProbability;

impl Distribution for UniformProbability {
    type Output = Probability;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> Probability {
        source.unit().to_probability()
    }
}

/// Integer-only, so the same seed gives the same draw on every target.
impl PortableDistribution for UniformProbability {}

/// A uniformly random [`NoiseValue`] in `[-1, 1)`.
///
/// Half-open at the top, inheriting the half-open unit draw it is stretched from.
/// This is a *uniform* sample of the range, not a sample of any noise field: it is
/// what to use for jitter, dithering and test data, not for terrain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniformNoise;

impl Distribution for UniformNoise {
    type Output = NoiseValue;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> NoiseValue {
        // `to_unit_value` cannot fail: the draw is half-open, so never one.
        source
            .unit()
            .to_unit_value()
            .expect("a uniform draw is below one")
            .to_noise_value()
    }
}

/// Integer-only, so the same seed gives the same draw on every target.
impl PortableDistribution for UniformNoise {}

// ---------------------------------------------------------------------------
// The bridge between the two unit-interval types
// ---------------------------------------------------------------------------

impl Unit {
    /// The same number as a [`Probability`], rounded to the nearest `f64`.
    ///
    /// **Lossy in general.** A `Unit` has 63 fractional bits; an `f64` has 53
    /// significant ones. Near the top of the range that leaves about a thousand
    /// `Unit` values sharing each `f64`, so neighbours collapse together —
    /// `ALMOST_ONE` converts to exactly `1.0`, as does the value below it. Use
    /// [`Unit::to_bits`] or [`fmt::Display`] when the exact value must survive.
    ///
    /// Deterministic: the same `Unit` always gives the same `f64`.
    pub fn to_probability(self) -> Probability {
        Probability::clamped(self.to_f64())
    }

    /// A [`Probability`] as a value here, rounded to the nearest representable
    /// `Unit`.
    ///
    /// # This is not exact, despite the bit counts
    ///
    /// It is tempting to reason that an `f64` carries at most 53 significant bits
    /// and there are 63 fractional ones here to put them in, so nothing can be
    /// lost. That argument is wrong, and it is worth being clear about why: it
    /// confuses *significant* bits with *absolute* position.
    ///
    /// A `Unit` has a fixed resolution of `2^-63 ≈ 1.08 × 10^-19` across the whole
    /// interval. An `f64`'s 53 bits sit wherever its exponent puts them, and for a
    /// small value that is far below the grid. `1e-30` has every one of its
    /// significant bits beneath `2^-63`, so it rounds to [`Unit::ZERO`] — not
    /// approximately zero, exactly zero.
    ///
    /// # What actually holds
    ///
    /// The conversion rounds to the nearest representable value, so the error never
    /// exceeds half a step, `2^-64`. Whether anything is lost depends on where the
    /// value sits:
    ///
    /// - **At or above `2^-11`** (about `4.9 × 10^-4`) an `f64`'s least significant
    ///   bit is at or above `2^-63`, so the conversion is exact and round-trips.
    /// - **Below that**, the low bits fall off the grid and are lost. The result is
    ///   still the nearest `Unit`, but it is not the same number.
    /// - **Below `2^-64`**, everything is lost and the result is [`Unit::ZERO`].
    /// - **Within half a step of one**, the result is [`Unit::ONE`].
    ///
    /// This is the trade the module documentation describes: uniform precision
    /// across `[0, 1]`, bought by giving up the very small values an `f64` reaches
    /// through its exponent. It is deterministic — the same `Probability` always
    /// gives the same `Unit` — but it is not universally exact.
    pub fn from_probability(chance: Probability) -> Self {
        Self::clamped(chance.value())
    }
}

impl From<Probability> for Unit {
    fn from(chance: Probability) -> Self {
        Self::from_probability(chance)
    }
}

impl From<Unit> for Probability {
    fn from(value: Unit) -> Self {
        value.to_probability()
    }
}
