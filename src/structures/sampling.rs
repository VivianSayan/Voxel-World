//! Random selection helpers shared by the collections.
//!
//! # Any source, not just a stream
//!
//! Each of these takes anything implementing
//! [`StochasticSource`](crate::random::StochasticSource), so the same helper serves a
//! [`Random`](crate::random::Random) stream and a
//! [`Seed`](crate::random::seed::Seed)'s cursor alike. They were written for `Random`
//! alone, which meant a seed — the thing a world uses to get the *same* answer every
//! time — could not shuffle anything at all.

use crate::random::distributions::{Distribution, Exponential};
use crate::random::source::StochasticSource;
use crate::structures::hashing::FastHashMap;
use crate::units::Rate;

/// Fisher-Yates shuffle.
///
/// # Question
///
/// "What is a random ordering of these items?"
///
/// Every ordering is equally likely. One draw per item, in place, no allocation.
pub fn shuffle<T, S: StochasticSource + ?Sized>(items: &mut [T], source: &mut S) {
    for last in (1..items.len()).rev() {
        let other = index_below(last + 1, source);
        items.swap(last, other);
    }
}

/// A uniform index below `length`, which must not be zero.
///
/// The one place the helpers here turn a word into a position, so they all share a
/// single unbiased bounded draw rather than each reaching for its own.
fn index_below<S: StochasticSource + ?Sized>(length: usize, source: &mut S) -> usize {
    debug_assert!(length > 0, "cannot draw an index from an empty range");

    source.bounded_u64(length as u64) as usize
}

/// Shuffles just enough to fill the first `count` places, and reports how many
/// that was.
///
/// # Question
///
/// "Which `count` unique items should I pick, and I do not care about the order of
/// the rest?"
///
/// # Why not shuffle the whole thing
///
/// A full shuffle costs one draw per item. This costs one per *chosen* item: taking
/// three spawn points out of a million candidates is three draws, not a million.
/// The first `count` places end up holding a uniformly random selection, in random
/// order; everything after them is left in whatever order the partial swaps put it,
/// which is not a valid shuffle and should not be treated as one.
///
/// Returns how many places were actually filled, which is `count` unless the slice
/// is shorter.
pub fn partial_shuffle<T, S: StochasticSource + ?Sized>(
    items: &mut [T],
    count: usize,
    source: &mut S,
) -> usize {
    let wanted: usize = count.min(items.len());

    for position in 0..wanted {
        // Pick from this position onwards, so nothing already chosen is disturbed.
        let other: usize = position + index_below(items.len() - position, source);
        items.swap(position, other);
    }

    wanted
}

/// `count` items chosen uniformly from a stream of unknown length.
///
/// # Question
///
/// "Which `count` items should I keep, seeing the stream only once and never
/// knowing how long it is?"
///
/// # Why this is worth having
///
/// The obvious approach collects the whole stream and then samples it, which costs
/// memory proportional to the stream. This holds only the `count` items it might
/// keep — **O(count) storage however long the stream turns out to be** — which is
/// what makes it usable on a generated or lazy sequence that does not fit in memory.
///
/// Every item that went past has the same chance of being in the result, and fewer
/// than `count` come back only if the stream was shorter than that.
///
/// # Which algorithm, and why
///
/// Vitter's Algorithm R: keep the first `count`, then for the `i`-th item after
/// them, replace a random one of the kept with probability `count / i`. One draw per
/// item.
///
/// Algorithm L is asymptotically better — it computes how far to skip instead of
/// asking about every item — but the skip needs `ln` and `exp`, which are
/// platform-dependent. Algorithm R uses only integer draws, so the same seed keeps
/// the same items on every target. For a crate whose worlds have to replay, that is
/// worth more than the constant factor.
pub fn reservoir_sample<T, I, S>(stream: I, count: usize, source: &mut S) -> Vec<T>
where
    I: IntoIterator<Item = T>,
    S: StochasticSource + ?Sized,
{
    if count == 0 {
        return Vec::new();
    }

    let mut kept: Vec<T> = Vec::with_capacity(count);

    for (position, item) in stream.into_iter().enumerate() {
        if position < count {
            kept.push(item);
            continue;
        }

        // The `position`-th item (counting from zero) is the `position + 1`-th seen,
        // and belongs in the reservoir with chance `count / (position + 1)`.
        // One of the `position + 1` items seen so far, so the chance of landing in
        // the reservoir is `count / (position + 1)`.
        let candidate: usize = index_below(position + 1, source);

        if candidate < count {
            kept[candidate] = item;
        }
    }

    kept
}

/// `k` distinct indices from `0..len`, uniformly, in random order.
///
/// Two strategies, both a partial Fisher-Yates shuffle and both unbiased. When
/// the sample is dense, more than a quarter of the population, a contiguous
/// pool of every index is cheaper than a map, and since the population is then
/// below `4k` it still costs O(k) time and space. When it is sparse, only the
/// positions the first `k` swaps actually touch are recorded, in a map, so the
/// cost stays proportional to `k` however large the population is: sampling
/// three indices out of `usize::MAX` allocates three entries.
///
/// Without a source, returns the first `k` indices, which is what the
/// collections use when they have no generator to hand.
pub fn uniform_indices<S: StochasticSource + ?Sized>(
    len: usize,
    k: usize,
    source: Option<&mut S>,
) -> Vec<usize> {
    let k = k.min(len);
    if k == 0 {
        return Vec::new();
    }
    let Some(source) = source else {
        return (0..k).collect();
    };

    if k > len / 4 {
        let mut pool: Vec<usize> = (0..len).collect();
        for position in 0..k {
            let other = position + index_below(len - position, source);
            pool.swap(position, other);
        }
        pool.truncate(k);
        return pool;
    }

    let mut moved = FastHashMap::with_capacity_and_hasher(k, Default::default());
    let mut indices = Vec::with_capacity(k);
    for position in 0..k {
        let other = position + index_below(len - position, source);
        let at_position = moved.remove(&position).unwrap_or(position);
        let at_other = if position == other {
            at_position
        } else {
            moved.remove(&other).unwrap_or(other)
        };
        if position != other {
            moved.insert(other, at_position);
        }
        indices.push(at_other);
    }
    indices
}

/// `k` distinct indices, each drawn in proportion to its weight among those
/// not yet drawn. Indices with a non-positive or non-finite weight are never
/// drawn, so fewer than `k` may come back.
///
/// Uses exponential keys (Efraimidis-Spirakis): each index gets
/// `Exp(1) / weight` and the `k` smallest keys win. Selection is O(n), followed
/// by O(k log k) to return them in draw order.
pub fn weighted_indices<S: StochasticSource + ?Sized>(
    weights: &[f64],
    k: usize,
    source: &mut S,
) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let mut keyed: Vec<(f64, usize)> = weights
        .iter()
        .enumerate()
        .filter(|(_, weight)| weight.is_finite() && **weight > 0.0)
        .map(|(index, weight)| (Exponential::new(Rate::UNIT).sample(source) / weight, index))
        .collect();

    let k = k.min(keyed.len());
    if k == 0 {
        return Vec::new();
    }

    if k < keyed.len() {
        keyed.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        keyed.truncate(k);
    }
    keyed.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    keyed.into_iter().map(|(_, index)| index).collect()
}

/// Turns log-weights into linear weights without overflow, by shifting the
/// largest to zero first.
pub fn weights_from_logs(log_weights: &mut [f64]) {
    let max = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    for weight in log_weights.iter_mut() {
        *weight = (*weight - max).exp();
    }
}
