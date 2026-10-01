//! Pseudorandom streams, seed derivation, and reusable distributions.
//!
//! [`Random`] is a stream: every draw moves it on. [`Seed`] is an immutable
//! derivation value: the same seed always gives the same answer. A
//! [`SeedCursor`] is the compact advancing counterpart. Every sampler in
//! [`distributions`] works with these sources through [`Random::sample`],
//! [`Seed::sample`], or [`DrawSource::draw`].

pub mod bit_permuter;
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
pub use distributions::*;
pub use seed::{
    SEED_ALGORITHM_VERSION, Seed, SeedCursor, SeedDomain, SeedInteger, SeedablePosition,
    domain_tag,
};
pub use source::{DrawSource, EventRandom, RandomSource};


pub use generator::{
    RANDOM_ALGORITHM_VERSION, RANDOM_STATE_BYTES, Random, RandomState,
};
