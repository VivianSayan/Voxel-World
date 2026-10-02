//! Pseudorandom streams, seed derivation, and reusable distributions.
//!
//! [`Random`] is a stream: every draw moves it on. [`Seed`] is an immutable
//! derivation value: the same seed always gives the same answer. A
//! [`SeedCursor`] is the compact advancing counterpart. Every sampler in
//! [`distributions`] works with these sources through [`Random::sample`],
//! [`Seed::sample`], or [`StochasticStream::draw`].

pub mod bit_permuter;
pub mod bulk_pick;
pub mod fixed_point;
mod generator;
pub mod unit;
mod questions;
pub mod approximation;
pub mod distributions;
pub(crate) mod mixing;
pub mod seed;
pub mod source;

pub use approximation::{
    Approximation, BinomialAlgorithm, HypergeometricAlgorithm, PoissonAlgorithm,
    PoissonBinomialAlgorithm,
};
pub use bulk_pick::{
    BulkError, BulkPickEntry, BulkPickResult, BulkPickTable, Count, CountDistribution, Quantity,
    QuantityDistribution, bulk_pick,
};
pub use distributions::*;
pub use fixed_point::{UniformFixed, UniformFixedRange};
pub use seed::{
    SEED_ALGORITHM_VERSION, Seed, SeedCursor, SeedDomain, SeedInteger, SeedablePosition,
    domain_tag,
};
pub use source::{StochasticStream, EventRandom, StochasticSource};


pub use generator::{
    RANDOM_ALGORITHM_VERSION, RANDOM_STATE_BYTES, Random, RandomState,
};
