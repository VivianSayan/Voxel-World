//! Samplers, written once and usable from any [`RandomSource`].
//!
//! Each distribution is a small value: `new` checks its parameters once and
//! [`Distribution::sample`] draws from it. The same value serves both kinds of
//! randomness in the crate:
//!
//! ```ignore
//! let height = Triangular::new(4.0, 6.0, 11.0)?;
//!
//! // A run of different values, from a stream.
//! let a = random.sample(&height);
//! let b = random.sample(&height);
//!
//! // One fixed value per voxel, from its seed: same voxel, same answer.
//! let here = world.child("trees").at_voxel(position).sample(&height);
//! ```
//!
//! [`Random`](super::Random)'s named methods (`random.gamma(..)` and the rest)
//! are shorthands that build the distribution and sample it once.
//!
//! # Exact, reproducible, or neither
//!
//! Every sampler is deterministic: the same source gives the same result on the
//! same build. Two further properties are listed on each type.
//!
//! - **Exact** samplers use only integer arithmetic, or floating-point steps
//!   IEEE-754 rounds the same way everywhere (`+ - * /`, `sqrt`). The same
//!   seed gives the same result on every platform, and the probabilities are
//!   the stated ones, not approximations of them.
//! - **Platform-dependent** samplers call `ln`, `exp`, `powf` or a
//!   trigonometric function. Those are not correctly rounded, so a different
//!   platform or libm may disagree in the last bit, and a rejection loop can
//!   then consume a different number of draws and diverge. Fine for effects;
//!   for world content that must match across machines, prefer an exact one.
//!
//! The exact family exists for that second case: integer versions of the
//! Bernoulli, binomial, Poisson, Laplace and Gaussian distributions whose
//! parameters are [`Ratio`](crate::units::Ratio)s.

mod continuous;
mod directions;
mod discrete;
mod exact;
mod bulk;
mod tables;

/// Version of the mapping from random words to sampled values.
///
/// # Question
///
/// "A world was generated with an older build. Will its terrain come out the same
/// today?"
///
/// [`SEED_ALGORITHM_VERSION`](crate::random::SEED_ALGORITHM_VERSION) answers that for
/// seed derivation and
/// [`RANDOM_ALGORITHM_VERSION`](crate::random::RANDOM_ALGORITHM_VERSION) for the word
/// stream, but neither covers the step between: the samplers here, which turn those
/// words into counts, directions and variates. Two builds can agree on every word and
/// still disagree on every value.
///
/// Persist this beside generated-world metadata along with the other two, and
/// increment it before intentionally changing how any sampler consumes words — a new
/// algorithm, a different rejection condition, a changed approximation threshold, or
/// a different number of words drawn per value. Fixing a sampler that was simply
/// wrong counts: the world still changes.
///
/// # Why it starts at one rather than counting past changes
///
/// The samplers have moved more than once already, but nothing has been released from
/// this crate and no world outside it was generated with the earlier behaviour. There
/// is nothing for an older number to distinguish, so this begins where the record
/// begins.
pub const DISTRIBUTION_ALGORITHM_VERSION: u32 = 1;

pub use bulk::{
    Hypergeometric, Multinomial, NegativeBinomial, PoissonBinomial, SparseSuccesses,
};
pub use continuous::{
    Beta, Cauchy, Exponential, Gamma, LogNormal, Normal, Pareto, Power, Triangular,
    TruncatedNormal, Uniform,
};
pub use directions::{
    UniformRotation, UnitBall, UnitCircle, UnitDisc, UnitHypersphere, UnitSphere,
};
pub use discrete::{
    Bernoulli, Binomial, Geometric, GeometricRatio, Poisson, StochasticRound, UniformI64,
    UniformU64, Zipf,
};
pub use exact::{
    BernoulliMask, BernoulliRatio, BinomialRatio, DiscreteGaussian, DiscreteLaplace, PoissonRatio,
};
pub use tables::{Categorical, Dirichlet, IntegerCategorical, WeightedDiscrete};

pub(super) use continuous::standard_normal;
pub(super) use discrete::{geometric_one_in, geometric_ratio};

use super::source::RandomSource;

/// Something that can be sampled.
pub trait Distribution {
    type Output;

    /// One draw. Consumes as many words from `source` as the algorithm needs,
    /// which for rejection samplers varies from call to call.
    fn sample<S: RandomSource + ?Sized>(&self, source: &mut S) -> Self::Output;
}

/// A distribution whose draw is bit-for-bit portable across supported targets.
///
/// Persistent world generation can require this marker at compile time rather
/// than relying on callers to remember which samplers avoid platform-dependent
/// `ln`, `exp`, `powf`, `cbrt`, and trigonometric implementations. Portability
/// describes replay, not necessarily mathematical exactness: for example,
/// [`GeometricRatio`] uses a documented fixed-point approximation for very
/// rare probabilities but implements this trait because that approximation is
/// integer-only and reproducible.
pub trait PortableDistribution: Distribution {}
