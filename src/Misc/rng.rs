use std::f64::consts::{E, PI};

const SQRT_TAU: f64 = 2.506_628_274_631_000_5;

#[derive(Clone, Debug)]
pub struct Random {
    seed: u128,
    state: [u64; 4],
    // Normal samples are generated in pairs; the second one is kept here.
    spare_normal: Option<f64>,
}

impl Random {
    pub fn new(seed: u128) -> Self {
        let mut random = Self {
            seed,
            state: [0; 4],
            spare_normal: None,
        };

        random.reseed(seed);
        random
    }

    pub fn reseed(&mut self, seed: u128) {
        self.seed = seed;
        self.spare_normal = None;

        // Collapse the 128-bit seed into a starting 64-bit value.
        // Both halves influence the resulting state.
        let lower: u64 = seed as u64;
        let upper: u64 = (seed >> 64) as u64;

        let mut seed_state: u64 =
            lower ^ upper.rotate_left(32) ^ 0x9E3779B97F4A7C15;

        // SplitMix64 is used here to expand the seed into the
        // four independent state words required by xoshiro256**.
        for index in 0..4 {
            self.state[index] = Self::splitmix64(&mut seed_state);
        }

        // xoshiro must never have an entirely zero state.
        if self.state == [0; 4] {
            self.state[0] = 0x9E3779B97F4A7C15;
        }
    }

    pub fn seed(&self) -> u128 {
        self.seed
    }

    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() & 1) == 1
    }

    pub fn next_u16(&mut self) -> u16 {
        // The high bits of xoshiro256** are the strongest.
        (self.next_u64() >> 48) as u16
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    pub fn next_u64(&mut self) -> u64 {
        // xoshiro256** output transformation.
        let result: u64 = self.state[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);

        // State transition.
        let temporary: u64 = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= temporary;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    pub fn next_f32(&mut self) -> f32 {
        let result: u32 = self.next_u32();
        (result as f32) / (u32::MAX as f32)
    }

    pub fn next_f64(&mut self) -> f64 {
        let result: u64 = self.next_u64();
        (result as f64) / (u64::MAX as f64)
    }

    pub fn next_u128(&mut self) -> u128 {
        let upper: u128 = self.next_u64() as u128;
        let lower: u128 = self.next_u64() as u128;

        (upper << 64) | lower
    }

    pub fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);

        let mut value: u64 = *state;

        value = (value ^ (value >> 30))
            .wrapping_mul(0xBF58476D1CE4E5B9);

        value = (value ^ (value >> 27))
            .wrapping_mul(0x94D049BB133111EB);

        value ^ (value >> 31)
    }

    pub fn nex_i128(&mut self) -> i128 {
        self.next_u128() as i128
    }

    pub fn nex_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    pub fn nex_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }

    pub fn nex_i16(&mut self) -> i16 {
        self.next_u16() as i16
    }
}

// ---------------------------------------------------------------------------
// Unit interval helpers
// ---------------------------------------------------------------------------

impl Random {
    /// Uniform in [0, 1) with 53 bits of precision.
    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in (0, 1). Safe to pass to `ln` or use as a divisor.
    fn open_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, range) without modulo bias (Lemire's method).
    fn bounded_u64(&mut self, range: u64) -> u64 {
        debug_assert!(range > 0);

        let mut product: u128 = self.next_u64() as u128 * range as u128;
        let mut low: u64 = product as u64;

        if low < range {
            let threshold: u64 = range.wrapping_neg() % range;
            while low < threshold {
                product = self.next_u64() as u128 * range as u128;
                low = product as u64;
            }
        }

        (product >> 64) as u64
    }
}

// ---------------------------------------------------------------------------
// Continuous distributions
// ---------------------------------------------------------------------------

impl Random {
    /// Uniform in [low, high).
    pub fn uniform_f64(&mut self, low: f64, high: f64) -> f64 {
        debug_assert!(low <= high);
        low + (high - low) * self.unit_f64()
    }

    /// Uniform in [low, high).
    pub fn uniform_f32(&mut self, low: f32, high: f32) -> f32 {
        self.uniform_f64(low as f64, high as f64) as f32
    }

    /// Standard normal, mean 0 and standard deviation 1 (Marsaglia polar method).
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }

        loop {
            let x: f64 = 2.0 * self.unit_f64() - 1.0;
            let y: f64 = 2.0 * self.unit_f64() - 1.0;
            let s: f64 = x * x + y * y;

            if s > 0.0 && s < 1.0 {
                let factor: f64 = (-2.0 * s.ln() / s).sqrt();
                self.spare_normal = Some(y * factor);
                return x * factor;
            }
        }
    }

    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        debug_assert!(std_dev >= 0.0);
        mean + std_dev * self.standard_normal()
    }

    /// Normal restricted to [low, high] (Robert, 1995).
    /// Stays efficient even when the interval lies far out in a tail.
    pub fn truncated_normal(&mut self, mean: f64, std_dev: f64, low: f64, high: f64) -> f64 {
        debug_assert!(std_dev > 0.0 && low <= high);

        let a: f64 = (low - mean) / std_dev;
        let b: f64 = (high - mean) / std_dev;

        let z: f64 = if a == b {
            a
        } else if a <= 0.0 && b >= 0.0 {
            self.truncated_normal_centered(a, b)
        } else if a > 0.0 {
            self.truncated_normal_tail(a, b)
        } else {
            -self.truncated_normal_tail(-b, -a)
        };

        (mean + std_dev * z).clamp(low, high)
    }

    /// Standard normal truncated to [a, b] where a <= 0 <= b.
    fn truncated_normal_centered(&mut self, a: f64, b: f64) -> f64 {
        if b - a >= SQRT_TAU {
            // At least ~half the mass is inside, plain rejection is cheap.
            loop {
                let z: f64 = self.standard_normal();
                if z >= a && z <= b {
                    return z;
                }
            }
        }

        loop {
            let z: f64 = self.uniform_f64(a, b);
            if self.open_unit_f64().ln() <= -0.5 * z * z {
                return z;
            }
        }
    }

    /// Standard normal truncated to [a, b] where 0 < a < b (b may be infinite).
    fn truncated_normal_tail(&mut self, a: f64, b: f64) -> f64 {
        let root: f64 = (a * a + 4.0).sqrt();
        let uniform_limit: f64 =
            a + 2.0 * E.sqrt() / (a + root) * ((a * a - a * root) / 4.0).exp();

        if b <= uniform_limit {
            loop {
                let z: f64 = self.uniform_f64(a, b);
                if self.open_unit_f64().ln() <= 0.5 * (a * a - z * z) {
                    return z;
                }
            }
        }

        // Exponential proposal with the optimal rate.
        let alpha: f64 = 0.5 * (a + root);
        loop {
            let z: f64 = a - self.open_unit_f64().ln() / alpha;
            if z > b {
                continue;
            }
            let offset: f64 = z - alpha;
            if self.open_unit_f64().ln() <= -0.5 * offset * offset {
                return z;
            }
        }
    }

    pub fn log_normal(&mut self, mu: f64, sigma: f64) -> f64 {
        self.normal(mu, sigma).exp()
    }

    /// Exponential with rate `lambda` (mean 1 / lambda).
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        debug_assert!(lambda > 0.0);
        -self.open_unit_f64().ln() / lambda
    }

    /// Gamma with the given shape (k) and scale (theta). Mean is shape * scale.
    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        debug_assert!(shape > 0.0 && scale > 0.0);
        self.ln_standard_gamma(shape).exp() * scale
    }

    /// Natural log of a Gamma(shape, 1) sample. Working in log space keeps
    /// very small shapes from underflowing to zero.
    fn ln_standard_gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            // Gamma(a) = Gamma(a + 1) * U^(1 / a)
            return self.standard_gamma_large(shape + 1.0).ln()
                + self.open_unit_f64().ln() / shape;
        }
        self.standard_gamma_large(shape).ln()
    }

    /// Gamma(shape, 1) for shape >= 1 (Marsaglia & Tsang, 2000).
    fn standard_gamma_large(&mut self, shape: f64) -> f64 {
        let d: f64 = shape - 1.0 / 3.0;
        let spread: f64 = 1.0 / (3.0 * d.sqrt());

        loop {
            let x: f64 = self.standard_normal();
            let v: f64 = 1.0 + x * spread;
            if v <= 0.0 {
                continue;
            }

            let v3: f64 = v * v * v;
            let log_accept: f64 = 0.5 * x * x + d * (1.0 - v3 + 3.0 * v.ln());

            if self.open_unit_f64().ln() <= log_accept {
                return d * v3;
            }
        }
    }

    /// Beta on [0, 1], built from two Gamma samples.
    pub fn beta(&mut self, alpha: f64, beta: f64) -> f64 {
        debug_assert!(alpha > 0.0 && beta > 0.0);

        let ln_x: f64 = self.ln_standard_gamma(alpha);
        let ln_y: f64 = self.ln_standard_gamma(beta);

        // x / (x + y) evaluated in log space.
        1.0 / (1.0 + (ln_y - ln_x).exp())
    }

    /// Power function distribution on [0, 1] with density alpha * x^(alpha - 1).
    pub fn power(&mut self, alpha: f64) -> f64 {
        debug_assert!(alpha > 0.0);
        self.open_unit_f64().powf(1.0 / alpha)
    }

    pub fn cauchy(&mut self, location: f64, scale: f64) -> f64 {
        debug_assert!(scale > 0.0);
        location + scale * (PI * (self.open_unit_f64() - 0.5)).tan()
    }

    /// Pareto (type I) with minimum value `scale` and tail index `shape`.
    pub fn pareto(&mut self, scale: f64, shape: f64) -> f64 {
        debug_assert!(scale > 0.0 && shape > 0.0);
        scale * self.open_unit_f64().powf(-1.0 / shape)
    }
}

// ---------------------------------------------------------------------------
// Discrete distributions
// ---------------------------------------------------------------------------

impl Random {
    /// Uniform integer in [low, high] (inclusive).
    pub fn uniform_u64(&mut self, low: u64, high: u64) -> u64 {
        debug_assert!(low <= high);

        let span: u64 = high - low;
        if span == u64::MAX {
            return self.next_u64();
        }
        low + self.bounded_u64(span + 1)
    }

    /// Uniform integer in [low, high] (inclusive).
    pub fn uniform_i64(&mut self, low: i64, high: i64) -> i64 {
        debug_assert!(low <= high);

        let span: u64 = (high as u64).wrapping_sub(low as u64);
        if span == u64::MAX {
            return self.next_u64() as i64;
        }
        low.wrapping_add(self.bounded_u64(span + 1) as i64)
    }

    /// Uniform index in [0, length).
    pub fn uniform_index(&mut self, length: usize) -> usize {
        debug_assert!(length > 0);
        self.bounded_u64(length as u64) as usize
    }

    /// True with probability `p`.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        debug_assert!((0.0..=1.0).contains(&p));
        self.unit_f64() < p
    }

    /// Number of successes in `trials` independent trials with success chance `p`.
    pub fn binomial(&mut self, trials: u64, p: f64) -> u64 {
        debug_assert!((0.0..=1.0).contains(&p));

        if trials == 0 || p == 0.0 {
            return 0;
        }
        if p == 1.0 {
            return trials;
        }

        // Sample with p <= 0.5 and mirror, both algorithms assume it.
        let flipped: bool = p > 0.5;
        let p: f64 = if flipped { 1.0 - p } else { p };

        let successes: u64 = if trials as f64 * p < 10.0 {
            self.binomial_inversion(trials, p)
        } else {
            self.binomial_btrs(trials, p)
        };

        if flipped { trials - successes } else { successes }
    }

    /// Sequential inversion, fast when trials * p is small.
    fn binomial_inversion(&mut self, trials: u64, p: f64) -> u64 {
        let q: f64 = 1.0 - p;
        let ratio: f64 = p / q;
        let a: f64 = (trials + 1) as f64 * ratio;
        let start: f64 = (trials as f64 * (-p).ln_1p()).exp();

        'retry: loop {
            let mut u: f64 = self.unit_f64();
            let mut probability: f64 = start;
            let mut x: u64 = 0;

            while u > probability {
                u -= probability;
                x += 1;
                if x > trials {
                    // Floating point leftovers ran past the support.
                    continue 'retry;
                }
                probability *= a / x as f64 - ratio;
            }

            return x;
        }
    }

    /// Transformed rejection with squeeze (Hörmann, 1993), for trials * p >= 10.
    fn binomial_btrs(&mut self, trials: u64, p: f64) -> u64 {
        let n: f64 = trials as f64;
        let q: f64 = 1.0 - p;
        let spq: f64 = (n * p * q).sqrt();

        let b: f64 = 1.15 + 2.53 * spq;
        let a: f64 = -0.0873 + 0.0248 * b + 0.01 * p;
        let c: f64 = n * p + 0.5;
        let v_r: f64 = 0.92 - 4.2 / b;
        let alpha: f64 = (2.83 + 5.1 / b) * spq;
        let log_ratio: f64 = (p / q).ln();
        let mode: f64 = ((n + 1.0) * p).floor();
        let h: f64 = ln_gamma(mode + 1.0) + ln_gamma(n - mode + 1.0);

        loop {
            let u: f64 = self.unit_f64() - 0.5;
            let v: f64 = self.open_unit_f64();
            let us: f64 = 0.5 - u.abs();
            let k: f64 = ((2.0 * a / us + b) * u + c).floor();

            if k < 0.0 || k > n {
                continue;
            }
            if us >= 0.07 && v <= v_r {
                return k as u64;
            }

            let lhs: f64 = (v * alpha / (a / (us * us) + b)).ln();
            let rhs: f64 =
                h - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0) + (k - mode) * log_ratio;

            if lhs <= rhs {
                return k as u64;
            }
        }
    }

    /// Poisson with mean `lambda`.
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        debug_assert!(lambda >= 0.0);

        if lambda == 0.0 {
            0
        } else if lambda < 10.0 {
            self.poisson_inversion(lambda)
        } else {
            self.poisson_ptrs(lambda)
        }
    }

    fn poisson_inversion(&mut self, lambda: f64) -> u64 {
        'retry: loop {
            let u: f64 = self.unit_f64();
            let mut probability: f64 = (-lambda).exp();
            let mut cumulative: f64 = probability;
            let mut k: u64 = 0;

            while u > cumulative {
                k += 1;
                if k > 1000 {
                    continue 'retry;
                }
                probability *= lambda / k as f64;
                cumulative += probability;
            }

            return k;
        }
    }

    /// Transformed rejection (Hörmann, 1993), for lambda >= 10.
    fn poisson_ptrs(&mut self, lambda: f64) -> u64 {
        let sqrt_lambda: f64 = lambda.sqrt();
        let ln_lambda: f64 = lambda.ln();
        let b: f64 = 0.931 + 2.53 * sqrt_lambda;
        let a: f64 = -0.059 + 0.02483 * b;
        let ln_inv_alpha: f64 = (1.1239 + 1.1328 / (b - 3.4)).ln();
        let v_r: f64 = 0.9277 - 3.6224 / (b - 2.0);

        loop {
            let u: f64 = self.unit_f64() - 0.5;
            let v: f64 = self.open_unit_f64();
            let us: f64 = 0.5 - u.abs();
            let k: f64 = ((2.0 * a / us + b) * u + lambda + 0.43).floor();

            if us >= 0.07 && v <= v_r {
                return k as u64;
            }
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }

            let lhs: f64 = v.ln() + ln_inv_alpha - (a / (us * us) + b).ln();
            let rhs: f64 = -lambda + k * ln_lambda - ln_gamma(k + 1.0);

            if lhs <= rhs {
                return k as u64;
            }
        }
    }

    /// Zipf over 1..=elements with the given exponent.
    /// For repeated sampling with fixed parameters, keep a `Zipf` around instead.
    pub fn zipf(&mut self, elements: u64, exponent: f64) -> u64 {
        Zipf::new(elements, exponent).sample(self)
    }

    /// Index drawn from `weights` in proportion to each weight. O(n) per call,
    /// no allocation. Use `Categorical` for O(1) repeated sampling.
    pub fn weighted_index(&mut self, weights: &[f64]) -> usize {
        debug_assert!(!weights.is_empty());

        let total: f64 = weights.iter().sum();
        debug_assert!(total > 0.0);

        let mut target: f64 = self.unit_f64() * total;
        let mut last_positive: usize = 0;

        for (index, &weight) in weights.iter().enumerate() {
            if weight > 0.0 {
                if target < weight {
                    return index;
                }
                target -= weight;
                last_positive = index;
            }
        }

        // Only reached through rounding error.
        last_positive
    }
}

// ---------------------------------------------------------------------------
// Precomputed samplers
// ---------------------------------------------------------------------------

/// Zipf distribution over 1..=elements with P(k) proportional to k^-exponent.
/// Rejection-inversion (Hörmann & Derflinger, 1996): O(1) per sample, no tables.
#[derive(Clone, Debug)]
pub struct Zipf {
    elements: f64,
    exponent: f64,
    h_integral_x1: f64,
    h_integral_n: f64,
    s: f64,
}

impl Zipf {
    pub fn new(elements: u64, exponent: f64) -> Self {
        assert!(elements > 0, "Zipf requires at least one element");
        assert!(exponent > 0.0, "Zipf exponent must be positive");

        let mut zipf = Self {
            elements: elements as f64,
            exponent,
            h_integral_x1: 0.0,
            h_integral_n: 0.0,
            s: 0.0,
        };

        zipf.h_integral_x1 = zipf.h_integral(1.5) - 1.0;
        zipf.h_integral_n = zipf.h_integral(zipf.elements + 0.5);
        zipf.s = 2.0 - zipf.h_integral_inverse(zipf.h_integral(2.5) - zipf.h(2.0));
        zipf
    }

    pub fn sample(&self, rng: &mut Random) -> u64 {
        loop {
            let u: f64 = self.h_integral_n
                + rng.unit_f64() * (self.h_integral_x1 - self.h_integral_n);
            let x: f64 = self.h_integral_inverse(u);
            let k: f64 = (x + 0.5).floor().clamp(1.0, self.elements);

            if k - x <= self.s || u >= self.h_integral(k + 0.5) - self.h(k) {
                return k as u64;
            }
        }
    }

    fn h(&self, x: f64) -> f64 {
        (-self.exponent * x.ln()).exp()
    }

    /// Integral of h, (x^(1 - s) - 1) / (1 - s), stable near s = 1.
    fn h_integral(&self, x: f64) -> f64 {
        let ln_x: f64 = x.ln();
        exp_m1_over_x((1.0 - self.exponent) * ln_x) * ln_x
    }

    fn h_integral_inverse(&self, x: f64) -> f64 {
        let t: f64 = (x * (1.0 - self.exponent)).max(-1.0);
        (ln_1p_over_x(t) * x).exp()
    }
}

/// Categorical distribution over indices 0..n. Probabilities are normalized,
/// so any non-negative weights work. O(n) setup, O(1) sampling (alias method).
#[derive(Clone, Debug)]
pub struct Categorical {
    probability: Vec<f64>,
    alias: Vec<usize>,
}

impl Categorical {
    pub fn new(probabilities: &[f64]) -> Self {
        let count: usize = probabilities.len();
        assert!(count > 0, "Categorical requires at least one outcome");
        assert!(
            probabilities.iter().all(|p| p.is_finite() && *p >= 0.0),
            "Categorical probabilities must be finite and non-negative"
        );

        let total: f64 = probabilities.iter().sum();
        assert!(total > 0.0, "Categorical probabilities must not all be zero");

        // Vose's alias method.
        let mut scaled: Vec<f64> = probabilities
            .iter()
            .map(|p| p * count as f64 / total)
            .collect();
        let mut probability: Vec<f64> = vec![1.0; count];
        let mut alias: Vec<usize> = (0..count).collect();

        let (mut small, mut large): (Vec<usize>, Vec<usize>) =
            (0..count).partition(|&index| scaled[index] < 1.0);

        while let (Some(&less), Some(&more)) = (small.last(), large.last()) {
            small.pop();
            probability[less] = scaled[less];
            alias[less] = more;

            scaled[more] -= 1.0 - scaled[less];
            if scaled[more] < 1.0 {
                large.pop();
                small.push(more);
            }
        }
        // Whatever remains in either list is 1.0 up to rounding, already set.

        Self { probability, alias }
    }

    pub fn len(&self) -> usize {
        self.probability.len()
    }

    pub fn sample(&self, rng: &mut Random) -> usize {
        let column: usize = rng.uniform_index(self.probability.len());
        if rng.unit_f64() < self.probability[column] {
            column
        } else {
            self.alias[column]
        }
    }
}

/// Picks one of a set of values in proportion to its weight.
#[derive(Clone, Debug)]
pub struct WeightedDiscrete<T> {
    values: Vec<T>,
    table: Categorical,
}

impl<T> WeightedDiscrete<T> {
    pub fn new(entries: impl IntoIterator<Item = (T, f64)>) -> Self {
        let (values, weights): (Vec<T>, Vec<f64>) = entries.into_iter().unzip();
        let table: Categorical = Categorical::new(&weights);
        Self { values, table }
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn sample_index(&self, rng: &mut Random) -> usize {
        self.table.sample(rng)
    }

    pub fn sample(&self, rng: &mut Random) -> &T {
        &self.values[self.table.sample(rng)]
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// ln(Gamma(x)) for x > 0 (Lanczos approximation, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula.
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x: f64 = x - 1.0;
    let mut sum: f64 = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        sum += coefficient / (x + index as f64);
    }

    let t: f64 = x + 7.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// ln(1 + x) / x, continuous at 0.
fn ln_1p_over_x(x: f64) -> f64 {
    if x.abs() > 1e-8 {
        x.ln_1p() / x
    } else {
        1.0 - x * (0.5 - x * (1.0 / 3.0 - 0.25 * x))
    }
}

/// (e^x - 1) / x, continuous at 0.
fn exp_m1_over_x(x: f64) -> f64 {
    if x.abs() > 1e-8 {
        x.exp_m1() / x
    } else {
        1.0 + x * 0.5 * (1.0 + x / 3.0 * (1.0 + 0.25 * x))
    }
}
