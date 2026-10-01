//! Arithmetic on ranges rather than on numbers, and on unions of ranges.
//!
//! [`Interval`] is a closed range `[low, high]` that adds, subtracts and
//! multiplies as a whole: the result is the range of every answer the operands
//! could have given. [`IntervalSet`] is a union of disjoint ranges, which keeps
//! the gaps that a single interval has to swallow.
//!
//! # It is not a ring, and that is not a shortcoming
//!
//! The name "interval ring" is common, and wrong twice over. Neither law holds:
//!
//! - **No additive inverse.** `[1, 2] - [1, 2]` is `[-1, 1]`, not `[0, 0]`.
//!   Subtracting a range from itself does not cancel, because the two ends are
//!   free to differ — that is the whole point of the arithmetic.
//! - **No distributivity, only subdistributivity.** With `X = [-1, 1]`,
//!   `Y = [1, 1]` and `Z = [-1, -1]`, `X(Y + Z)` is `[0, 0]` while `XY + XZ` is
//!   `[-2, 2]`. The two are not equal; the first is contained in the second.
//!
//! So these types implement [`Zero`] and [`One`] and stop there: not
//! [`Semiring`](crate::math::traits::Semiring), which needs distributivity, and
//! not [`Ring`], which needs inverses. Claiming
//! either would let generic code assume a cancellation that silently does not
//! happen.
//!
//! What the arithmetic *does* promise is containment: every result holds every
//! value the operation could have produced. That is what makes it useful for
//! bounding error, and it is weaker than a ring on purpose.

use crate::math::traits::{Field, One, Ring, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The smaller of two values, keeping the first where they do not compare.
///
/// [`PartialOrd`] rather than [`Ord`] throughout, because `f64` is not `Ord` and
/// bounding floating-point error is what interval arithmetic is mostly for. A
/// value that compares with nothing — a `NaN` — cannot reach these, since
/// [`Interval::new`] refuses to build an interval out of one.
fn smaller<T: PartialOrd>(first: T, second: T) -> T {
    if second < first { second } else { first }
}

/// The larger of two values, keeping the first where they do not compare.
fn larger<T: PartialOrd>(first: T, second: T) -> T {
    if second > first { second } else { first }
}

// ---------------------------------------------------------------------------
// A single interval
// ---------------------------------------------------------------------------

/// A closed range `[low, high]`, treated as one value.
///
/// # Invariant
///
/// `low <= high`, so every interval is non-empty and holds at least its own
/// endpoints. An empty range is represented by an [`IntervalSet`] with no parts
/// rather than by a reversed interval, which keeps this type's arithmetic total.
///
/// # Example
///
/// ```
/// use voxel_world::math::Interval;
///
/// let a: Interval<i64> = Interval::new(1, 3).expect("ordered");
/// let b: Interval<i64> = Interval::new(-2, 1).expect("ordered");
///
/// // Addition adds the ends.
/// assert_eq!(a + b, Interval::new(-1, 4).unwrap());
///
/// // Multiplication has to try all four corners, since a negative end can
/// // become the largest product.
/// assert_eq!(a * b, Interval::new(-6, 3).unwrap());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Interval<T> {
    low: T,
    high: T,
}

impl<T: PartialOrd> Interval<T> {
    /// The range from `low` to `high`, or `None` if they are the wrong way round
    /// or do not compare at all.
    ///
    /// Written as `!(low <= high)` rather than `low > high` so that a value
    /// comparing with nothing is refused: for a `NaN` both comparisons are false,
    /// and an interval built from one would break every ordering below it.
    pub fn new(low: T, high: T) -> Option<Self> {
        // `partial_cmp` rather than a negated comparison, so that "they do not
        // compare" is a case in its own right rather than something inferred
        // from two falsehoods.
        match low.partial_cmp(&high) {
            Some(Ordering::Less | Ordering::Equal) => Some(Self { low, high }),
            _ => None,
        }
    }

    /// The range between two values, whichever order they arrive in.
    pub fn between(first: T, second: T) -> Self {
        if first <= second {
            Self {
                low: first,
                high: second,
            }
        } else {
            Self {
                low: second,
                high: first,
            }
        }
    }

    /// The degenerate range holding one value.
    pub fn point(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            low: value.clone(),
            high: value,
        }
    }

    /// The lower end.
    pub fn low(&self) -> &T {
        &self.low
    }

    /// The upper end.
    pub fn high(&self) -> &T {
        &self.high
    }

    /// Whether the range holds one value.
    pub fn contains(&self, value: &T) -> bool {
        self.low <= *value && *value <= self.high
    }

    /// Whether the range holds only one value, so that the arithmetic on it is
    /// ordinary arithmetic.
    pub fn is_point(&self) -> bool {
        self.low == self.high
    }

    /// Whether the two ranges share any value.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.low <= other.high && other.low <= self.high
    }

    /// Whether the two ranges share a value or sit end to end, which is when
    /// their union is again one range.
    pub fn touches(&self, other: &Self) -> bool {
        self.overlaps(other)
    }

    /// Whether every value of `other` is also in this range.
    pub fn encloses(&self, other: &Self) -> bool {
        self.low <= other.low && other.high <= self.high
    }

    /// The values in both ranges, or `None` when they are disjoint.
    pub fn intersect(&self, other: &Self) -> Option<Self>
    where
        T: Clone,
    {
        let low: T = larger(self.low.clone(), other.low.clone());
        let high: T = smaller(self.high.clone(), other.high.clone());

        Self::new(low, high)
    }

    /// The smallest range holding both, which swallows any gap between them.
    ///
    /// [`IntervalSet::union`] keeps the gap instead.
    pub fn hull(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        Self {
            low: smaller(self.low.clone(), other.low.clone()),
            high: larger(self.high.clone(), other.high.clone()),
        }
    }
}

impl<T: Ring + PartialOrd> Interval<T> {
    /// How wide the range is.
    pub fn width(&self) -> T {
        self.high.clone() - self.low.clone()
    }

    /// Whether the range straddles zero, which is what stops it being divided
    /// by.
    pub fn contains_zero(&self) -> bool {
        self.contains(&T::zero())
    }
}

impl<T: Ring + PartialOrd> Add for Interval<T> {
    type Output = Self;

    /// The ends add: the smallest sum is of the two smallest, the largest of the
    /// two largest.
    fn add(self, other: Self) -> Self {
        Self {
            low: self.low + other.low,
            high: self.high + other.high,
        }
    }
}

impl<T: Ring + PartialOrd> Neg for Interval<T> {
    type Output = Self;

    /// The ends swap as well as change sign, since negating reverses the order.
    fn neg(self) -> Self {
        Self {
            low: -self.high,
            high: -self.low,
        }
    }
}

impl<T: Ring + PartialOrd> Sub for Interval<T> {
    type Output = Self;

    /// The largest difference is the largest minus the smallest, so the ends
    /// cross over.
    ///
    /// This is why there is no additive inverse: `x - x` is `[low - high,
    /// high - low]`, which is zero only for a single point.
    fn sub(self, other: Self) -> Self {
        Self {
            low: self.low - other.high,
            high: self.high - other.low,
        }
    }
}

impl<T: Ring + PartialOrd> Mul for Interval<T> {
    type Output = Self;

    /// All four corner products, then the smallest and largest of them.
    ///
    /// Nothing cheaper is correct in general: a negative end times a negative
    /// end is positive, so which corner is largest depends on the signs, and
    /// checking those is more cases than simply comparing four products.
    fn mul(self, other: Self) -> Self {
        let corners: [T; 4] = [
            self.low.clone() * other.low.clone(),
            self.low * other.high.clone(),
            self.high.clone() * other.low,
            self.high * other.high,
        ];

        let [first, second, third, fourth] = corners;

        Self {
            low: smaller(
                smaller(first.clone(), second.clone()),
                smaller(third.clone(), fourth.clone()),
            ),
            high: larger(larger(first, second), larger(third, fourth)),
        }
    }
}

impl<T: Ring + PartialOrd> AddAssign for Interval<T> {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<T: Ring + PartialOrd> SubAssign for Interval<T> {
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl<T: Ring + PartialOrd> MulAssign for Interval<T> {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<T: Field + PartialOrd> Interval<T> {
    /// The reciprocal range, or `None` when the range straddles zero.
    ///
    /// A range holding zero has no reciprocal that is a single range: the
    /// reciprocals of `[-1, 1]` are everything outside `(-1, 1)`, which is two
    /// pieces. [`IntervalSet::reciprocal`] gives them.
    pub fn reciprocal(&self) -> Option<Self> {
        if self.contains_zero() {
            return None;
        }

        let low: T = self.high.clone().inverse()?;
        let high: T = self.low.clone().inverse()?;

        Self::new(low, high)
    }

    /// This divided by `divisor`, or `None` when the divisor straddles zero.
    pub fn divide(&self, divisor: &Self) -> Option<Self> {
        Some(self.clone() * divisor.reciprocal()?)
    }
}

impl<T: Ring + PartialOrd> Zero for Interval<T> {
    /// The single point zero, which is the only interval that adds nothing.
    fn zero() -> Self {
        Self::point(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.low.is_zero() && self.high.is_zero()
    }
}

impl<T: Ring + PartialOrd> One for Interval<T> {
    /// The single point one.
    fn one() -> Self {
        Self::point(T::one())
    }

    fn is_one(&self) -> bool {
        self.low.is_one() && self.high.is_one()
    }
}

impl<T: fmt::Display> fmt::Display for Interval<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "[{}, {}]", self.low, self.high)
    }
}

// ---------------------------------------------------------------------------
// A union of intervals
// ---------------------------------------------------------------------------

/// A union of disjoint ranges, kept sorted and merged.
///
/// # What it is for
///
/// Keeping the holes. A single [`Interval`] can only widen: adding `[0, 1]` to
/// `[0, 1] ∪ [5, 6]` should give `[0, 2] ∪ [5, 7]`, and a single interval has to
/// answer `[0, 7]` instead — claiming every value in between is reachable when
/// none of `2..5` is.
///
/// The cost is that the number of pieces grows: multiplying an `n`-piece set by
/// an `m`-piece one considers `n × m` products before merging. Where the pieces
/// stay few that is nothing; where they do not, [`IntervalSet::hull`] collapses
/// back to the single range that a plain interval would have given.
///
/// # Invariant
///
/// The parts are sorted by their lower end, none is empty, and no two overlap or
/// touch — touching ones are merged, since `[0, 1] ∪ [1, 2]` is `[0, 2]`. So the
/// representation of a set of values is unique, which is what lets equality be
/// derived.
///
/// # Example
///
/// ```
/// use voxel_world::math::{Interval, IntervalSet};
///
/// let split: IntervalSet<i64> = IntervalSet::new(vec![
///     Interval::new(0, 1).unwrap(),
///     Interval::new(5, 6).unwrap(),
/// ]);
///
/// let shift: IntervalSet<i64> = IntervalSet::from(Interval::new(0, 1).unwrap());
///
/// // The gap survives, where a single interval would have swallowed it.
/// assert_eq!(
///     split.clone() + shift,
///     IntervalSet::new(vec![
///         Interval::new(0, 2).unwrap(),
///         Interval::new(5, 7).unwrap(),
///     ])
/// );
///
/// assert_eq!(split.hull(), Interval::new(0, 6));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct IntervalSet<T> {
    /// Sorted by lower end, disjoint, non-touching.
    parts: Vec<Interval<T>>,
}

impl<T: PartialOrd + Clone> IntervalSet<T> {
    /// The union of these ranges, sorted and merged.
    pub fn new(parts: Vec<Interval<T>>) -> Self {
        let mut set = Self { parts };
        set.normalise();

        set
    }

    /// The empty set, which holds no values at all.
    pub fn empty() -> Self {
        Self { parts: Vec::new() }
    }

    /// The pieces, in order.
    pub fn parts(&self) -> &[Interval<T>] {
        &self.parts
    }

    /// How many pieces there are, which is one more than the number of gaps.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Whether the set holds no values.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Whether any piece holds the value.
    pub fn contains(&self, value: &T) -> bool {
        self.parts.iter().any(|part| part.contains(value))
    }

    /// The smallest single range holding everything, or `None` when empty.
    ///
    /// What a plain [`Interval`] would have given: the gaps are lost.
    pub fn hull(&self) -> Option<Interval<T>> {
        let first: &Interval<T> = self.parts.first()?;
        let last: &Interval<T> = self.parts.last()?;

        Interval::new(first.low.clone(), last.high.clone())
    }

    /// Everything in either set.
    pub fn union(&self, other: &Self) -> Self {
        let mut parts: Vec<Interval<T>> = self.parts.clone();
        parts.extend(other.parts.iter().cloned());

        Self::new(parts)
    }

    /// Everything in both sets.
    ///
    /// Every pair of pieces is intersected, which is all that is needed: a value
    /// is in both sets exactly when some piece of each holds it.
    pub fn intersect(&self, other: &Self) -> Self {
        let mut parts: Vec<Interval<T>> = Vec::new();

        for left in &self.parts {
            for right in &other.parts {
                if let Some(shared) = left.intersect(right) {
                    parts.push(shared);
                }
            }
        }

        Self::new(parts)
    }

    /// Sorts the pieces and merges the ones that meet.
    fn normalise(&mut self) {
        // `partial_cmp` because the ends need only be partially ordered; nothing
        // incomparable can be here, since `Interval::new` refuses it.
        self.parts.sort_by(|left, right| {
            left.low
                .partial_cmp(&right.low)
                .unwrap_or(Ordering::Equal)
                .then(
                    left.high
                        .partial_cmp(&right.high)
                        .unwrap_or(Ordering::Equal),
                )
        });

        let mut merged: Vec<Interval<T>> = Vec::with_capacity(self.parts.len());

        for part in self.parts.drain(..) {
            match merged.last_mut() {
                // Sorted, so only the last piece can meet this one.
                Some(last) if last.touches(&part) => {
                    last.high = larger(last.high.clone(), part.high);
                }
                _ => merged.push(part),
            }
        }

        self.parts = merged;
    }
}

impl<T: Ring + PartialOrd> IntervalSet<T> {
    /// Every piecewise sum, merged.
    fn combine(
        &self,
        other: &Self,
        operation: impl Fn(Interval<T>, Interval<T>) -> Interval<T>,
    ) -> Self {
        let mut parts: Vec<Interval<T>> = Vec::with_capacity(self.parts.len() * other.parts.len());

        for left in &self.parts {
            for right in &other.parts {
                parts.push(operation(left.clone(), right.clone()));
            }
        }

        Self::new(parts)
    }
}

impl<T: Ring + PartialOrd> Add for IntervalSet<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.combine(&other, |left, right| left + right)
    }
}

impl<T: Ring + PartialOrd> Sub for IntervalSet<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.combine(&other, |left, right| left - right)
    }
}

impl<T: Ring + PartialOrd> Mul for IntervalSet<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.combine(&other, |left, right| left * right)
    }
}

impl<T: Ring + PartialOrd> Neg for IntervalSet<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(self.parts.into_iter().map(|part| -part).collect())
    }
}

impl<T: Field + PartialOrd> IntervalSet<T> {
    /// The reciprocals of everything in the set.
    ///
    /// This is where the sparse form earns its place: a piece straddling zero is
    /// split at zero into the parts that *can* be inverted, and the reciprocals
    /// of those are kept. A single interval has to give up entirely.
    ///
    /// Zero itself has no reciprocal, so it is dropped — which means the result
    /// may hold fewer values than the input did, unlike every other operation
    /// here.
    ///
    /// The two open-ended pieces this ought to produce — the reciprocals of
    /// values arbitrarily near zero run off to infinity — cannot be represented
    /// without an unbounded end, so each is cut off at the reciprocal of the
    /// nearest representable end. The result is therefore a *subset* of the true
    /// answer where a piece straddles zero, and exact everywhere else.
    pub fn reciprocal(&self) -> Self {
        let mut parts: Vec<Interval<T>> = Vec::new();

        for part in &self.parts {
            if !part.contains_zero() {
                if let Some(inverted) = part.reciprocal() {
                    parts.push(inverted);
                }

                continue;
            }

            // Split at zero and invert whichever sides have any width.
            for side in [
                Interval::new(part.low.clone(), T::zero()),
                Interval::new(T::zero(), part.high.clone()),
            ]
            .into_iter()
            .flatten()
            {
                if side.is_point() {
                    continue;
                }

                // The end at zero cannot be inverted, so the other end bounds
                // the piece and this is a subset of the true reciprocal.
                if let Some(bound) = if side.high.is_zero() {
                    side.low.clone().inverse()
                } else {
                    side.high.clone().inverse()
                } {
                    parts.push(Interval::between(bound.clone(), bound));
                }
            }
        }

        Self::new(parts)
    }
}

impl<T: Ring + PartialOrd> Zero for IntervalSet<T> {
    fn zero() -> Self {
        Self::from(Interval::point(T::zero()))
    }

    fn is_zero(&self) -> bool {
        matches!(self.parts.as_slice(), [only] if only.is_zero())
    }
}

impl<T: Ring + PartialOrd> One for IntervalSet<T> {
    fn one() -> Self {
        Self::from(Interval::point(T::one()))
    }

    fn is_one(&self) -> bool {
        matches!(self.parts.as_slice(), [only] if only.is_one())
    }
}

impl<T: PartialOrd + Clone> From<Interval<T>> for IntervalSet<T> {
    fn from(interval: Interval<T>) -> Self {
        Self {
            parts: vec![interval],
        }
    }
}

impl<T: PartialOrd + Clone> FromIterator<Interval<T>> for IntervalSet<T> {
    fn from_iter<I: IntoIterator<Item = Interval<T>>>(parts: I) -> Self {
        Self::new(parts.into_iter().collect())
    }
}

impl<T: fmt::Display> fmt::Display for IntervalSet<T> {
    /// The pieces joined by unions, or an empty-set sign for nothing.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.parts.is_empty() {
            return formatter.write_str("\u{2205}");
        }

        for (index, part) in self.parts.iter().enumerate() {
            if index > 0 {
                formatter.write_str(" \u{222a} ")?;
            }

            write!(formatter, "{part}")?;
        }

        Ok(())
    }
}
