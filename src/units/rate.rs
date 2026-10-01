//! Positive finite event rates.
//!
//! The parameter of the samplers that answer "how often", and the reciprocal of
//! the ones that answer "how long between": an [`Exponential`] takes a rate and
//! gives a waiting time, a [`Poisson`] takes a rate and gives a count.
//!
//! [`Exponential`]: crate::random::Exponential
//! [`Poisson`]: crate::random::Poisson

use std::fmt;

/// A rate: how often something happens per unit of whatever is being counted.
///
/// Strictly positive and finite, which is what `exponential` and `poisson` both
/// assumed and checked only in debug builds. A rate of zero means an event that
/// never happens, and dividing by it gives an infinite waiting time rather than
/// an error, which is why it is excluded here rather than clamped.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct Rate(f64);

impl Rate {
    /// One event per unit, which is the standard form every other rate is a
    /// scaling of.
    pub const UNIT: Self = Self(1.0);

    /// The rate as given, or `None` unless it is finite and strictly above
    /// zero.
    pub fn new(per_unit: f64) -> Option<Self> {
        if !per_unit.is_finite() || per_unit <= 0.0 {
            return None;
        }

        Some(Self(per_unit))
    }

    /// The rate whose average spacing between events is `mean`, which is its
    /// reciprocal.
    ///
    /// The constructor to reach for when the design says "a boulder every 40
    /// voxels" rather than "0.025 boulders per voxel". `None` for a spacing of
    /// zero, for a negative one, or for one so large that its reciprocal
    /// underflows to zero.
    pub fn from_mean(mean: f64) -> Option<Self> {
        Self::new(1.0 / mean)
    }

    /// The rate as a plain `f64`, events per unit.
    pub const fn value(self) -> f64 {
        self.0
    }

    /// The average gap between events, the reciprocal of the rate. Finite and
    /// above zero, since the rate is.
    pub fn mean(self) -> f64 {
        1.0 / self.0
    }
}

impl fmt::Display for Rate {
    /// As `<rate> per unit`, the unit being whatever the caller is counting.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} per unit", self.0)
    }
}
