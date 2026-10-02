//! Real-valued distributions.

use super::{Distribution, PortableDistribution};
use crate::random::source::StochasticSource;
use crate::units::Rate;
use std::f64::consts::{E, PI};

const SQRT_TAU: f64 = 2.506_628_274_631_000_5;

/// A distribution over `f64` parameters: the struct, a `new` that rejects
/// non-finite or invalid parameters, and the sampler. Inside `$body` each
/// parameter is bound by name and `$source` is the [`StochasticSource`].
macro_rules! distribution {
    (
        $(#[$meta:meta])*
        $name:ident [$($parameter:ident),+] valid $valid:expr;
        sample($source:ident) $body:block
    ) => {
        $(#[$meta])*
        ///
        /// Built by `new`, which checks the parameters once; sampling it after
        /// that cannot fail.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name { $($parameter: f64),+ }

        impl $name {
            /// `None` unless every parameter is finite and valid together.
            pub fn new($($parameter: f64),+) -> Option<Self> {
                let valid: bool = $valid;
                if !($($parameter.is_finite())&&+) || !valid {
                    return None;
                }
                Some(Self { $($parameter),+ })
            }
        }

        impl Distribution for $name {
            type Output = f64;

            fn sample<S: StochasticSource + ?Sized>(&self, $source: &mut S) -> f64 {
                let Self { $($parameter),+ } = *self;
                $body
            }
        }
    };
}

distribution! {
    /// Uniform in `[low, high)`, or the constant `low` when the bounds meet.
    ///
    /// Exact.
    Uniform [low, high] valid low <= high && (high - low).is_finite();
    sample(source) {
        uniform(source, low, high)
    }
}

impl PortableDistribution for Uniform {}
impl PortableDistribution for Triangular {}

distribution! {
    /// Normal with the given mean and non-negative standard deviation
    /// (Marsaglia's polar method).
    ///
    /// Platform-dependent: uses `ln`.
    Normal [mean, std_dev] valid std_dev >= 0.0;
    sample(source) {
        mean + std_dev * standard_normal(source)
    }
}

distribution! {
    /// Normal restricted to `[low, high]` (Robert, 1995). Stays efficient even
    /// when the interval lies far out in a tail.
    ///
    /// Platform-dependent: uses `ln` and `exp`.
    TruncatedNormal [mean, std_dev, low, high] valid std_dev > 0.0 && low <= high && {
        let a: f64 = (low - mean) / std_dev;
        let b: f64 = (high - mean) / std_dev;
        (a * a + 4.0).is_finite() && (b * b + 4.0).is_finite()
    };
    sample(source) {
        let a: f64 = (low - mean) / std_dev;
        let b: f64 = (high - mean) / std_dev;

        let z: f64 = if a == b {
            a
        } else if a <= 0.0 && b >= 0.0 {
            truncated_normal_centered(source, a, b)
        } else if a > 0.0 {
            truncated_normal_tail(source, a, b)
        } else {
            -truncated_normal_tail(source, -b, -a)
        };

        (mean + std_dev * z).clamp(low, high)
    }
}

distribution! {
    /// Log-normal: `exp` of a normal with location `mu` and deviation `sigma`.
    ///
    /// Platform-dependent: uses `ln` and `exp`.
    LogNormal [mu, sigma] valid sigma >= 0.0;
    sample(source) {
        (mu + sigma * standard_normal(source)).exp()
    }
}

distribution! {
    /// Triangular: between `low` and `high`, peaking at `mode`.
    ///
    /// The distribution to reach for when a value should sit around somewhere
    /// without a hard edge and without the unbounded tails of a normal. Far
    /// easier to author than a mean and a deviation, because all three numbers
    /// are the thing being described rather than a summary of it.
    ///
    /// Exact: the inverse of the distribution, so one draw and a `sqrt`, no
    /// rejection and no transcendentals. The draw is split at the fraction of
    /// the span the mode sits at, and each half of the distribution is inverted
    /// separately. The two halves meet exactly at the mode: approaching that
    /// fraction from below the first gives `low + span * peak`, and from above
    /// the second gives `high - span * (1 - peak)`, which is the same point.
    Triangular [low, mode, high] valid low <= mode && mode <= high && (high - low).is_finite();
    sample(source) {
        let span: f64 = high - low;

        if span <= 0.0 {
            return low;
        }

        let peak: f64 = (mode - low) / span;
        let draw: f64 = source.unit_f64();

        if draw < peak {
            low + span * (draw * peak).sqrt()
        } else {
            high - span * ((1.0 - draw) * (1.0 - peak)).sqrt()
        }
    }
}

distribution! {
    /// Gamma with the given shape (k) and scale (theta). Mean is
    /// `shape * scale`.
    ///
    /// Platform-dependent: uses `ln` and `exp`.
    Gamma [shape, scale] valid shape > 0.0 && scale > 0.0;
    sample(source) {
        ln_standard_gamma(source, shape).exp() * scale
    }
}

distribution! {
    /// Beta on `[0, 1]`, built from two Gamma samples as `x / (x + y)`.
    ///
    /// Both gammas are drawn in log space and the division is evaluated there
    /// too, as `1 / (1 + exp(ln y - ln x))`, so a very small alpha or beta
    /// cannot underflow to `0 / 0` on the way.
    ///
    /// Platform-dependent: uses `ln` and `exp`.
    Beta [alpha, beta] valid alpha > 0.0 && beta > 0.0;
    sample(source) {
        let ln_x: f64 = ln_standard_gamma(source, alpha);
        let ln_y: f64 = ln_standard_gamma(source, beta);

        1.0 / (1.0 + (ln_y - ln_x).exp())
    }
}

distribution! {
    /// Power function distribution on `[0, 1]` with density
    /// `alpha * x^(alpha - 1)`.
    ///
    /// Platform-dependent: uses `powf`.
    Power [alpha] valid alpha > 0.0;
    sample(source) {
        source.open_unit_f64().powf(1.0 / alpha)
    }
}

distribution! {
    /// Cauchy with the given location and positive scale. Has no mean: expect
    /// occasional enormous values.
    ///
    /// Platform-dependent: uses `tan`.
    Cauchy [location, scale] valid scale > 0.0;
    sample(source) {
        location + scale * (PI * (source.open_unit_f64() - 0.5)).tan()
    }
}

distribution! {
    /// Pareto (type I) with minimum value `scale` and tail index `shape`.
    ///
    /// Platform-dependent: uses `powf`.
    Pareto [scale, shape] valid scale > 0.0 && shape > 0.0;
    sample(source) {
        scale * source.open_unit_f64().powf(-1.0 / shape)
    }
}

/// Exponential with the given rate, whose mean is the rate's reciprocal.
///
/// The continuous version of [`Geometric`](super::Geometric): how long until
/// the next event, rather than how many steps.
///
/// Platform-dependent: uses `ln`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponential {
    rate: Rate,
}

impl Exponential {
    /// Always valid: a [`Rate`] is already positive and finite.
    pub fn new(rate: Rate) -> Self {
        Self { rate }
    }
}

impl Distribution for Exponential {
    type Output = f64;

    fn sample<S: StochasticSource + ?Sized>(&self, source: &mut S) -> f64 {
        -source.open_unit_f64().ln() / self.rate.value()
    }
}

// ---------------------------------------------------------------------------
// Shared algorithms
// ---------------------------------------------------------------------------

fn uniform<S: StochasticSource + ?Sized>(source: &mut S, low: f64, high: f64) -> f64 {
    if low == high {
        return low;
    }
    (low + (high - low) * source.unit_f64())
        .min(high.next_down())
        .max(low)
}

/// Standard normal, mean 0 and deviation 1 (Marsaglia polar method).
///
/// Normals come in pairs; the second goes to the source to keep, if it keeps
/// one, and is handed back by the next call.
pub(in crate::random) fn standard_normal<S: StochasticSource + ?Sized>(source: &mut S) -> f64 {
    if let Some(spare) = source.take_spare_normal() {
        return spare;
    }

    loop {
        let x: f64 = 2.0 * source.unit_f64() - 1.0;
        let y: f64 = 2.0 * source.unit_f64() - 1.0;
        let s: f64 = x * x + y * y;

        if s > 0.0 && s < 1.0 {
            let factor: f64 = (-2.0 * s.ln() / s).sqrt();
            source.keep_spare_normal(y * factor);
            return x * factor;
        }
    }
}

/// Standard normal truncated to `[a, b]` where `a <= 0 <= b`.
///
/// Two strategies. A wide interval holds at least about half the distribution's
/// mass, so plain rejection, drawing normals until one lands inside, is cheap.
/// A narrow one would reject far too often, so it draws uniformly across the
/// interval and accepts against the normal's own density instead.
fn truncated_normal_centered<S: StochasticSource + ?Sized>(source: &mut S, a: f64, b: f64) -> f64 {
    if b - a >= SQRT_TAU {
        loop {
            let z: f64 = standard_normal(source);
            if z >= a && z <= b {
                return z;
            }
        }
    }

    loop {
        let z: f64 = uniform(source, a, b);
        if source.open_unit_f64().ln() <= -0.5 * z * z {
            return z;
        }
    }
}

/// Standard normal truncated to `[a, b]` where `0 < a < b` (`b` may be
/// infinite).
///
/// Robert's method. For an interval short enough that the density hardly
/// varies across it, a uniform proposal accepted against the density is
/// efficient. For a longer or unbounded one, the proposal is an exponential
/// started at `a` with the rate that maximises the acceptance rate, which is
/// what keeps the cost flat however far out the interval lies.
fn truncated_normal_tail<S: StochasticSource + ?Sized>(source: &mut S, a: f64, b: f64) -> f64 {
    let root: f64 = (a * a + 4.0).sqrt();
    let uniform_limit: f64 = a + 2.0 * E.sqrt() / (a + root) * ((a * a - a * root) / 4.0).exp();

    if b <= uniform_limit {
        loop {
            let z: f64 = uniform(source, a, b);
            if source.open_unit_f64().ln() <= 0.5 * (a * a - z * z) {
                return z;
            }
        }
    }

    let alpha: f64 = 0.5 * (a + root);
    loop {
        let z: f64 = a - source.open_unit_f64().ln() / alpha;
        if z > b {
            continue;
        }
        let offset: f64 = z - alpha;
        if source.open_unit_f64().ln() <= -0.5 * offset * offset {
            return z;
        }
    }
}

/// Natural log of a Gamma(shape, 1) sample.
///
/// Marsaglia and Tsang's method needs a shape of at least one. A smaller shape
/// uses the identity `Gamma(a) = Gamma(a + 1) * U^(1/a)`, which is why a
/// uniform draw is raised to a power here. Everything is kept in log space so
/// that a very small shape, where that power is enormous, cannot underflow the
/// sample to zero before it is used.
fn ln_standard_gamma<S: StochasticSource + ?Sized>(source: &mut S, shape: f64) -> f64 {
    if shape < 1.0 {
        return standard_gamma_large(source, shape + 1.0).ln()
            + source.open_unit_f64().ln() / shape;
    }
    standard_gamma_large(source, shape).ln()
}

/// Gamma(shape, 1) for shape >= 1 (Marsaglia & Tsang, 2000).
///
/// Squeezes a normal draw through a cubed transform whose shape matches the
/// gamma's, then accepts against the remaining difference. Accepts well over
/// nine times in ten for any shape, so the loop almost never runs twice.
fn standard_gamma_large<S: StochasticSource + ?Sized>(source: &mut S, shape: f64) -> f64 {
    let d: f64 = shape - 1.0 / 3.0;
    let spread: f64 = 1.0 / (3.0 * d.sqrt());

    loop {
        let x: f64 = standard_normal(source);
        let v: f64 = 1.0 + x * spread;
        if v <= 0.0 {
            continue;
        }

        let v3: f64 = v * v * v;
        let log_accept: f64 = 0.5 * x * x + d * (1.0 - v3 + 3.0 * v.ln());

        if source.open_unit_f64().ln() <= log_accept {
            return d * v3;
        }
    }
}
