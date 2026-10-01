//! Small dependency-free comparisons, not a substitute for workload profiling.
//! Run with `cargo bench --bench sampling` (release optimizations).

use std::collections::VecDeque;
use std::hint::black_box;
use std::time::Instant;
use voxel_world::random::seed::Seed;
use voxel_world::random::{
    Binomial, BinomialRatio, Categorical, DiscreteGaussian, Distribution, GeometricRatio,
    IntegerCategorical, Normal, Poisson, PoissonRatio, Random, Triangular,
};
use voxel_world::structures::traits::Choose;
use voxel_world::units::{Probability, Rate, Ratio, Weights};

/// One boxed draw, so float and exact samplers of different types share a table.
type Sampler = Box<dyn FnMut() -> i64>;

fn random() -> Random {
    Random::new(Seed::from_raw(42))
}

fn measure<T>(name: &str, iterations: u32, mut operation: impl FnMut() -> T) -> f64 {
    for _ in 0..32 {
        black_box(operation());
    }
    let mut trials = [0.0; 7];
    for elapsed in &mut trials {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(operation());
        }
        *elapsed = start.elapsed().as_secs_f64() * 1e9 / iterations as f64;
    }
    trials.sort_by(f64::total_cmp);
    let median = trials[trials.len() / 2];
    println!("{name:44} {median:12.1} ns/op");
    median
}

fn compare(baseline: f64, optimized: f64) {
    println!("  baseline / optimized: {:.2}x\n", baseline / optimized);
}

fn main() {
    println!("Median of 7 trials; fixed seeds; construction excluded unless stated.\n");
    for count in [257, 1_000_000, 10_000_000_000] {
        let sampler = GeometricRatio::one_in(count);
        let mut one_shot = random();
        let mut cached = random();
        let baseline = measure(
            &format!("geometric 1/{count}: build + sample"),
            100_000,
            || one_shot.geometric_one_in(black_box(count)),
        );
        let optimized = measure(
            &format!("geometric 1/{count}: cached sample"),
            100_000,
            || black_box(&sampler).sample(&mut cached),
        );
        compare(baseline, optimized);
    }

    let queue: VecDeque<_> = (0..4096).collect();
    let mut reservoir = random();
    let mut indexed = random();
    let baseline = measure("choose from 4096: reservoir reference", 2000, || {
        let mut chosen = None;
        for (seen, item) in black_box(&queue).iter().enumerate() {
            if reservoir.uniform_index(seen + 1) == 0 {
                chosen = Some(item);
            }
        }
        chosen
    });
    let optimized = measure("choose from 4096: direct index", 1_000_000, || {
        black_box(&queue).choose(&mut indexed)
    });
    compare(baseline, optimized);

    let mut full_pool = random();
    let mut sparse = random();
    let population = 100_000;
    let count = 16;
    let baseline = measure("16 of 100000: full-pool partial shuffle", 1000, || {
        let mut pool: Vec<_> = (0..black_box(population)).collect();
        for position in 0..count {
            let other = position + full_pool.uniform_index(population - position);
            pool.swap(position, other);
        }
        pool.truncate(count);
        pool
    });
    let optimized = measure("16 of 100000: sparse partial shuffle", 10_000, || {
        sparse.sample_distinct(black_box(count), black_box(population))
    });
    compare(baseline, optimized);

    println!(
        "Exact integer samplers against their floating-point versions (below 1x: exact is slower):\n"
    );
    let mut float = random();
    let mut exact = random();
    let pairs: [(&str, u32, Sampler, Sampler); 4] = [
        (
            "binomial 4096 at 1/64",
            100_000,
            {
                let d = Binomial::new(4096, Probability::new(1.0 / 64.0).unwrap());
                Box::new(move || d.sample(&mut float) as i64)
            },
            {
                let d = BinomialRatio::new(4096, Ratio::one_in(64).unwrap()).unwrap();
                Box::new(move || d.sample(&mut exact) as i64)
            },
        ),
        (
            "poisson at 7/2",
            1_000_000,
            {
                let d = Poisson::new(Rate::new(3.5).unwrap());
                let mut r = random();
                Box::new(move || d.sample(&mut r) as i64)
            },
            {
                let d = PoissonRatio::new(Ratio::new(7, 2).unwrap());
                let mut r = random();
                Box::new(move || d.sample(&mut r) as i64)
            },
        ),
        (
            "normal / discrete gaussian, variance 25",
            1_000_000,
            {
                let d = Normal::new(0.0, 5.0).unwrap();
                let mut r = random();
                Box::new(move || d.sample(&mut r).round() as i64)
            },
            {
                let d = DiscreteGaussian::new(0, Ratio::new(25, 1).unwrap()).unwrap();
                let mut r = random();
                Box::new(move || d.sample(&mut r))
            },
        ),
        (
            "categorical over 8 weights",
            1_000_000,
            {
                let d = Categorical::new(
                    &Weights::new([70.0, 25.0, 5.0, 1.0, 9.0, 3.0, 2.0, 8.0]).unwrap(),
                );
                let mut r = random();
                Box::new(move || d.sample(&mut r) as i64)
            },
            {
                let d = IntegerCategorical::new(&[70, 25, 5, 1, 9, 3, 2, 8]).unwrap();
                let mut r = random();
                Box::new(move || d.sample(&mut r) as i64)
            },
        ),
    ];
    for (name, iterations, mut floating, mut integer) in pairs {
        let baseline = measure(&format!("{name}: float"), iterations, &mut floating);
        let optimized = measure(&format!("{name}: exact"), iterations, &mut integer);
        compare(baseline, optimized);
    }

    let triangle = Triangular::new(0.0, 2.0, 5.0).unwrap();
    let mut stream = random();
    let baseline = measure("triangular: from a Random", 1_000_000, || {
        stream.sample(&triangle)
    });
    let mut index = 0u64;
    let world = Seed::from_raw(42);
    let optimized = measure("triangular: from a Seed, new each time", 1_000_000, || {
        index += 1;
        world.index(index).sample(&triangle)
    });
    compare(baseline, optimized);
}
