use voxel_world::math::{Quaternion, UnitQuaternion, Vector3};
use voxel_world::random::seed::Seed;
use voxel_world::spatial::{Depth, LocalPosition3, NodePosition3, TreeDepth, VoxelPosition3};
use voxel_world::structures::collections::TickScheduler;
use voxel_world::time::{Seconds, Tick, TickDuration, TickRate};
use voxel_world::units::{NoiseValue, Probability, Unit, UnitValue, Weights};

#[test]
fn probability_operations_preserve_the_range() {
    let values = [
        0.0,
        f64::MIN_POSITIVE,
        1e-18,
        0.01,
        0.5,
        1.0 - f64::EPSILON,
        1.0,
    ];
    for a in values {
        let p = Probability::new(a).unwrap();
        for b in values {
            let q = Probability::new(b).unwrap();
            for result in [p.and(q), p.or(q), p.complement()] {
                assert!(Probability::new(result.value()).is_some());
            }
        }
        for tries in [0, 1, 2, 1_000, i32::MAX as u32, u32::MAX] {
            assert!(Probability::new(p.in_any_of(tries).value()).is_some());
        }
    }
    assert_eq!(Probability::EVEN.in_any_of(u32::MAX), Probability::ALWAYS);
    assert_eq!(Probability::ALWAYS.in_any_of(0), Probability::NEVER);
    assert_eq!(Probability::EVEN.in_any_of(2).value(), 0.75);
    let accumulated = Probability::new(1e-18)
        .unwrap()
        .in_any_of(1_000_000_000)
        .value();
    assert!((accumulated - 1e-9).abs() < 1e-18, "{accumulated}");
}

#[test]
fn constructors_reject_invalid_values_and_repair_is_explicit() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 2.0] {
        assert!(Probability::new(invalid).is_none());
        assert!(UnitValue::new(invalid).is_none());
        assert!(LocalPosition3::new(Vector3::splat(invalid)).is_none());
        assert!(Probability::new(Probability::clamped(invalid).value()).is_some());
        assert!(UnitValue::new(UnitValue::clamped(invalid).value()).is_some());
    }
    assert!(UnitValue::new(1.0).is_none());
    assert!(LocalPosition3::new(Vector3::splat(1.0)).is_none());
    assert_eq!(
        NoiseValue::LOWEST.blend(NoiseValue::HIGHEST, Unit::ONE),
        NoiseValue::HIGHEST
    );
    assert_eq!(UnitValue::clamped(1.0).between(1.0, 1.0f64.next_up()), 1.0);
}

#[test]
fn weights_require_a_finite_total_and_never_pick_zero_weight() {
    assert!(Weights::new([f64::MAX, f64::MAX]).is_none());
    assert!(Weights::new([]).is_none());
    assert!(Weights::new([0.0, 0.0]).is_none());
    for weights in [[0.0, 1.0, 0.0], [0.0, f64::MIN_POSITIVE, 0.0]] {
        let weights = Weights::new(weights).unwrap();
        for unit in [0.0, 0.5, 1.0f64.next_down()] {
            assert_eq!(weights.index_for(UnitValue::new(unit).unwrap()), 1);
        }
        assert_eq!(
            weights.cumulative(),
            [Probability::NEVER, Probability::ALWAYS, Probability::ALWAYS]
        );
    }
}

#[test]
fn coordinate_conversions_hold_at_negative_and_extreme_positions() {
    for levels in [0, 8, 53, 54, 100, 127] {
        let tree = TreeDepth::new(levels).unwrap();
        assert_eq!(tree.node_width(tree.floor()), Some(1));
        for depth in [tree.ceiling(), Depth::ROOT, tree.floor()] {
            for x in [i128::MIN, i128::MIN + 1, -1025, -1, 0, 1, 1025, i128::MAX] {
                let position = VoxelPosition3::new(Vector3::new(x, x, x));
                let node = position.node_at(tree, depth).unwrap();
                assert!(node.contains(tree, position));
                let local = position.local_at(tree, depth).unwrap();
                assert!(local.fraction().all(|value| (0.0..1.0).contains(&value)));
                let origin = node.origin(tree).unwrap();
                assert_eq!(origin.node_at(tree, depth), Some(node));
                assert!(
                    origin
                        .local_at(tree, depth)
                        .unwrap()
                        .fraction()
                        .all(|value| value == 0.0)
                );
            }
        }
    }
    assert!(TreeDepth::new(128).is_none());
    assert!(TreeDepth::new(255).is_none());
    let tree = TreeDepth::new(4).unwrap();
    let position = VoxelPosition3::default();
    assert!(position.node_at(tree, Depth::new(5)).is_none());
    assert!(position.node_at(tree, Depth::HIGHEST).is_none());
    let oversized = NodePosition3::new(Vector3::splat(i128::MAX), Depth::ROOT);
    assert!(oversized.origin(tree).is_none());
    assert!(
        NodePosition3::new(Vector3::splat(0i128), Depth::HIGHEST)
            .parent()
            .is_none()
    );
}

#[test]
fn typed_time_and_scheduler_agree() {
    assert!(TickRate::new(0).is_none());
    assert!(Seconds::new(-1.0).is_none());
    assert!(Seconds::new(f64::INFINITY).is_none());
    let rate = TickRate::new(20).unwrap();
    assert_eq!(
        rate.ticks_in(Seconds::new(2.5).unwrap()),
        TickDuration::new(50)
    );
    assert_eq!(rate.seconds_in(TickDuration::new(50)).value(), 2.5);
    let start = Tick::new(100);
    assert_eq!((start + TickDuration::new(3)) - start, TickDuration::new(3));
    let mut schedule = TickScheduler::starting_at(start);
    schedule.schedule(TickDuration::new(3), "relative");
    schedule.schedule_at(Tick::new(101), "absolute");
    assert_eq!(schedule.advance(), ["absolute"]);
    assert_eq!(schedule.advance_by(TickDuration::new(2)), ["relative"]);
    assert!(schedule.is_empty());
    let mut ending = TickScheduler::starting_at(Tick::new(u64::MAX));
    ending.schedule(TickDuration::ZERO, "last");
    assert_eq!(ending.take_due(), ["last"]);
    assert!(ending.is_empty());
}

#[test]
fn rotations_reject_degenerate_inputs_and_preserve_lengths() {
    assert!(UnitQuaternion::new(Quaternion::ZERO).is_none());
    assert!(UnitQuaternion::new(Quaternion::new(f64::NAN, 0.0, 0.0, 0.0)).is_none());
    assert!(UnitQuaternion::from_axis_angle(Vector3::ZERO, 1.0).is_none());
    assert!(UnitQuaternion::from_axis_angle(Vector3::X, f64::INFINITY).is_none());
    let quarter_turn =
        UnitQuaternion::from_axis_angle(Vector3::Z, std::f64::consts::FRAC_PI_2).unwrap();
    assert!(quarter_turn.rotate(Vector3::X).distance(Vector3::Y) < 1e-14);
    let vector = Vector3::new(1.0, 2.0, -3.0);
    for fraction in [0.0, 0.1, 0.5, 0.9, 1.0] {
        let rotation =
            UnitQuaternion::IDENTITY.slerp(quarter_turn, Unit::new(fraction).unwrap());
        assert!((rotation.rotate(vector).norm() - vector.norm()).abs() < 1e-14);
        assert!(
            rotation
                .inverse()
                .rotate(rotation.rotate(vector))
                .distance(vector)
                < 1e-14
        );
    }
    for scale in [f64::MIN_POSITIVE, f64::MAX] {
        let rotation = UnitQuaternion::new(Quaternion::new(scale, scale, 0.0, 0.0)).unwrap();
        assert!((rotation.quaternion().norm() - 1.0).abs() < 1e-14);
    }
}

#[test]
fn seed_helpers_use_the_same_typed_identity() {
    let seed = Seed::from_raw(123);
    let position = VoxelPosition3::new(Vector3::new(-12, 34, 56));
    assert_eq!(seed.at_voxel(position), seed.at(position.to_array()));
    assert_ne!(
        seed.at_depth(Depth::new(2), position.to_array()),
        seed.at_depth(Depth::new(3), position.to_array())
    );
    assert!(!seed.chance(Probability::NEVER));
    assert!(seed.chance(Probability::ALWAYS));
}
