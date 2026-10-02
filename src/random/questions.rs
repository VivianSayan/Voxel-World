//! What [`Random`] can be asked.
//!
//! Every method here is a shorthand: it builds the distribution of the same name and
//! samples it once, panicking on parameters its `new` would refuse. The algorithm,
//! and whether it is exact or platform-dependent, is documented on the distribution.
//! For many draws from one distribution, build it once and use
//! [`Random::sample`].
//!
//! Split from the generator itself because the two change for different reasons: the
//! generator is a fixed algorithm that should almost never move, while this grows
//! every time there is a new question worth asking.

use crate::math::Fixed;
use crate::math::linear::{Vector2, Vector3, Vector4};
use crate::random::approximation::{
    Approximation, BinomialAlgorithm, HypergeometricAlgorithm, PoissonAlgorithm,
    PoissonBinomialAlgorithm,
};
use crate::random::bulk_pick::{
    BulkError, BulkPickResult, BulkPickTable, CountDistribution, QuantityDistribution, bulk_pick,
};
use crate::random::distributions::*;
use crate::random::fixed_point::UniformFixedRange;
use crate::random::source::StochasticSource;
use crate::random::{Random, approximation, distributions};
use crate::structures::traits::Shuffle;
use crate::units::{Probability, Rate, Ratio, Unit, Weights};

include!("questions/continuous.rs");
include!("questions/discrete_and_geometry.rs");
include!("questions/bulk.rs");
include!("questions/approximation.rs");
