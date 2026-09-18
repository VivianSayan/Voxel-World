//! Random selection helpers shared by the collections.

use crate::misc::random::Random;
use crate::misc::structures::hashing::FastHashMap;

/// Fisher-Yates shuffle.
pub fn shuffle<T>(items: &mut [T], random: &mut Random) {
    for last in (1..items.len()).rev() {
        let other = random.uniform_index(last + 1);
        items.swap(last, other);
    }
}

/// `k` distinct indices from `0..len`, uniformly, in random order.
/// Without a `Random`, returns the first `k` indices.
pub fn uniform_indices(len: usize, k: usize, random: Option<&mut Random>) -> Vec<usize> {
    let k = k.min(len);
    let Some(random) = random else {
        return (0..k).collect();
    };

    // A lazy partial Fisher-Yates. Only positions touched by the first k
    // swaps are materialized, keeping both time and memory proportional to k.
    let mut moved = FastHashMap::default();
    let mut indices = Vec::with_capacity(k);
    for position in 0..k {
        let other = position + random.uniform_index(len - position);
        let at_position = moved.remove(&position).unwrap_or(position);
        let at_other = moved.remove(&other).unwrap_or(other);
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
/// `Exp(1) / weight` and the `k` smallest keys win. That is O(n), against
/// O(n * k) for repeated weighted draws.
pub fn weighted_indices(weights: &[f64], k: usize, random: &mut Random) -> Vec<usize> {
    let mut keyed: Vec<(f64, usize)> = weights
        .iter()
        .enumerate()
        .filter(|(_, weight)| weight.is_finite() && **weight > 0.0)
        .map(|(index, weight)| (random.exponential(1.0) / weight, index))
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
