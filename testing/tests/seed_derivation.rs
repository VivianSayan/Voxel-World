//! Deriving one seed from another.
//!
//! Every derivation has to satisfy two things, and they pull in opposite directions:
//!
//! - **It must be stable.** The same question always gives the same seed, so a world
//!   replays.
//! - **It must not collide.** Two *different* questions must not give the same seed,
//!   or unrelated content correlates for no reason anybody can find.
//!
//! The collision tests are the valuable half. A derivation that merely compiles will
//! pass the stability ones.

use std::collections::HashSet;
use voxel_world::math::{Vector2, Vector3, Vector4};
use voxel_world::random::seed::{Seed, domain_tag};
use voxel_world::seed_domain;
use voxel_world::spatial::{
    Depth, NodePosition2, NodePosition3, NodePosition4, VoxelPosition2, VoxelPosition3,
    VoxelPosition4,
};
use voxel_world::time::Tick;

fn world() -> Seed {
    Seed::from_integer(0xC0FFEEu64)
}

fn voxel(x: i128, y: i128, z: i128) -> VoxelPosition3 {
    VoxelPosition3::new(Vector3::new(x, y, z))
}

// ===========================================================================
// advance_by
// ===========================================================================

#[test]
fn advancing_by_n_is_advancing_n_times() {
    let start: Seed = world();

    assert_eq!(start.advance_by(0), start, "no steps changes nothing");

    let mut walked: Seed = start;
    for steps in 1..=64u64 {
        walked = walked.advance();

        assert_eq!(
            start.advance_by(steps),
            walked,
            "advance_by({steps}) should match {steps} single steps"
        );
    }

    // The point of it: a huge skip costs the same as a small one, and still lands
    // where the walk would.
    let far: Seed = start.advance_by(1_000_000);
    assert_eq!(far, start.advance_by(999_999).advance());

    // And it composes.
    assert_eq!(start.advance_by(300).advance_by(700), start.advance_by(1_000));
}

#[test]
fn advancing_does_not_revisit() {
    let mut seen: HashSet<Seed> = HashSet::new();
    let start: Seed = world();

    for steps in 0..20_000u64 {
        assert!(
            seen.insert(start.advance_by(steps)),
            "advance_by({steps}) repeated an earlier seed"
        );
    }
}

// ===========================================================================
// at_position: uniform across shapes and dimensions
// ===========================================================================

#[test]
fn positions_of_every_shape_derive_a_seed() {
    let world: Seed = world();

    // The whole point: the same call for every shape.
    let two = world.at_position(VoxelPosition2::new(Vector2::new(1, 2)));
    let three = world.at_position(voxel(1, 2, 3));
    let four = world.at_position(VoxelPosition4::new(Vector4::new(1, 2, 3, 4)));
    let node2 = world.at_position(NodePosition2::new(Vector2::new(1, 2), Depth::new(3)));
    let node3 = world.at_position(NodePosition3::new(Vector3::new(1, 2, 3), Depth::new(3)));
    let node4 = world.at_position(NodePosition4::new(Vector4::new(1, 2, 3, 4), Depth::new(3)));

    let all = [two, three, four, node2, node3, node4];
    let unique: HashSet<Seed> = all.iter().copied().collect();

    assert_eq!(unique.len(), all.len(), "six different places, six seeds");
}

/// A voxel at `(1, 2)` must not be the same place as one at `(1, 2, 0)`.
#[test]
fn dimension_is_part_of_the_place() {
    let world: Seed = world();

    assert_ne!(
        world.at_position(VoxelPosition2::new(Vector2::new(1, 2))),
        world.at_position(voxel(1, 2, 0)),
        "padding with a zero must not land on the lower-dimensional place"
    );
    assert_ne!(
        world.at_position(voxel(1, 2, 3)),
        world.at_position(VoxelPosition4::new(Vector4::new(1, 2, 3, 0)))
    );
}

/// A node carries its depth, so the same address at two depths is two places.
#[test]
fn depth_is_part_of_the_place() {
    let world: Seed = world();
    let address = Vector3::new(1, 2, 3);

    let shallow = world.at_position(NodePosition3::new(address, Depth::new(2)));
    let deep = world.at_position(NodePosition3::new(address, Depth::new(3)));

    assert_ne!(shallow, deep, "two depths, two places");

    // And it agrees with what `at_depth` did with the arguments passed separately,
    // so the replacement did not move anything.
    assert_eq!(
        shallow,
        world.at_depth(Depth::new(2), [1i128, 2, 3]),
        "at_position must match the older two-argument form"
    );

    // A node is also not the bare voxel at the same address.
    assert_ne!(shallow, world.at_position(voxel(1, 2, 3)));
}

#[test]
fn positions_are_stable_and_distinct() {
    let world: Seed = world();
    let mut seen: HashSet<Seed> = HashSet::new();

    for x in -8i128..8 {
        for y in -8i128..8 {
            for z in -8i128..8 {
                let place = voxel(x, y, z);
                let derived = world.at_position(place);

                assert_eq!(derived, world.at_position(place), "stable");
                assert!(seen.insert(derived), "({x}, {y}, {z}) collided");
            }
        }
    }
}

// ===========================================================================
// between: the shared seed
// ===========================================================================

#[test]
fn a_shared_seed_does_not_depend_on_who_asks() {
    let mut left: Seed = Seed::from_integer(1u64);

    for _ in 0..2_000 {
        let right: Seed = left.advance_by(7);

        assert_eq!(left.between(right), right.between(left), "order must not matter");
        left = left.advance();
    }

    // Unlike `combine`, which is ordered on purpose.
    let a: Seed = Seed::from_integer(10u64);
    let b: Seed = Seed::from_integer(20u64);
    assert_ne!(a.combine(b), b.combine(a), "combine stays ordered");

    // A seed paired with itself is fine, and is not the seed itself.
    assert_eq!(a.between(a), a.between(a));
    assert_ne!(a.between(a), a);
}

/// The voxel case this exists for: two neighbours agreeing on the face between them.
#[test]
fn neighbours_agree_on_the_face_between_them() {
    let world: Seed = world();

    for x in 0i128..64 {
        let here = world.at_position(voxel(x, 0, 0));
        let next = world.at_position(voxel(x + 1, 0, 0));

        assert_eq!(
            here.between(next),
            next.between(here),
            "the face at x = {x} must look the same from both sides"
        );
    }

    // And different faces are different: a pair-wise seed that collapsed to a sum or
    // an exclusive-or would give every pair with the same total one seed.
    let mut faces: HashSet<Seed> = HashSet::new();
    for x in 0i128..40 {
        for y in 0i128..40 {
            let a = world.at_position(voxel(x, 0, 0));
            let b = world.at_position(voxel(0, y, 0));

            assert!(faces.insert(a.between(b)), "pair ({x}, {y}) collided");
        }
    }
}

// ===========================================================================
// at_tick
// ===========================================================================

#[test]
fn ticks_do_not_collide_with_indices() {
    let spawner: Seed = world();

    // The bug this prevents: "child five" and "tick five" sharing a seed, which would
    // tie content to the clock.
    for n in 0..256u64 {
        assert_ne!(
            spawner.at_tick(Tick::new(n)),
            spawner.index(n),
            "tick {n} must not be index {n}"
        );
    }

    let mut seen: HashSet<Seed> = HashSet::new();
    for n in 0..20_000u64 {
        assert!(seen.insert(spawner.at_tick(Tick::new(n))), "tick {n} collided");
    }

    assert_eq!(spawner.at_tick(Tick::new(5)), spawner.at_tick(Tick::new(5)));
}

// ===========================================================================
// at_region
// ===========================================================================

/// Regions must tile evenly *through zero*. Truncating division would put the voxel
/// at −1 in the same region as 0 and leave a one-voxel seam on every negative axis.
#[test]
fn regions_tile_evenly_across_the_origin() {
    let world: Seed = world();
    const SIZE: u64 = 64;
    let region = |x: i128, y: i128, z: i128| world.at_region(voxel(x, y, z), SIZE);

    // Inside one cube, everything agrees.
    assert_eq!(region(0, 0, 0), region(63, 63, 63));
    assert_eq!(region(1, 2, 3), region(60, 61, 62));

    // The next cube over is a different region.
    assert_ne!(region(0, 0, 0), region(64, 0, 0));
    assert_ne!(region(0, 0, 0), region(0, 64, 0));

    // And the negative side: −1 belongs with −64, not with 0. This is the assertion
    // that fails under `/` instead of `div_euclid`.
    assert_eq!(region(-1, -1, -1), region(-64, -64, -64));
    assert_ne!(region(-1, 0, 0), region(0, 0, 0), "the seam must not exist");
    assert_ne!(region(-65, 0, 0), region(-1, 0, 0));

    // Every region of a run is distinct, both sides of zero.
    let mut seen: HashSet<Seed> = HashSet::new();
    for step in -20i128..20 {
        assert!(
            seen.insert(region(step * SIZE as i128, 0, 0)),
            "region at step {step} collided"
        );
    }
}

#[test]
fn region_size_is_part_of_the_question() {
    let world: Seed = world();
    let place = voxel(100, 200, 300);

    // Different sizes are different questions, even about the same voxel.
    let sizes: Vec<Seed> = [4u64, 8, 16, 64, 256]
        .iter()
        .map(|size| world.at_region(place, *size))
        .collect();
    let unique: HashSet<Seed> = sizes.iter().copied().collect();

    assert_eq!(unique.len(), sizes.len(), "each size is its own partition");

    // A size of one is a region per voxel, so neighbours differ.
    assert_ne!(
        world.at_region(voxel(0, 0, 0), 1),
        world.at_region(voxel(1, 0, 0), 1)
    );

    // A size of zero has no region to speak of and falls back to the position.
    assert_eq!(world.at_region(place, 0), world.at_position(place));
}

// ===========================================================================
// Nothing collides across the whole family
// ===========================================================================

/// Every derivation must stake out its own space. One shared tag between two of them
/// would tie unrelated content together.
#[test]
fn the_derivations_do_not_collide_with_each_other() {
    let base: Seed = world();
    let other: Seed = Seed::from_integer(999u64);

    let derived: Vec<(&str, Seed)> = vec![
        ("child", base.child("trees")),
        ("index", base.index(3)),
        // `at` is not in this list: `at_position` on a `VoxelPosition3` *is*
        // `at(to_array())`, deliberately — the same question in two spellings, not
        // two questions. That aliasing is asserted on its own below.
        ("at_position", base.at_position(voxel(3, 0, 0))),
        ("at_depth", base.at_depth(Depth::new(3), [3i128, 0, 0])),
        ("at_tick", base.at_tick(Tick::new(3))),
        ("at_level", base.at_level(Depth::new(3))),
        ("at_octave", base.at_octave(3)),
        ("at_region", base.at_region(voxel(3, 0, 0), 8)),
        ("combine", base.combine(other)),
        ("between", base.between(other)),
        ("mix_in", base.mix_in(3)),
        ("domain", base.domain(3)),
        ("advance", base.advance()),
        ("advance_by", base.advance_by(3)),
        ("itself", base),
    ];

    // The intended aliases, stated so they are not mistaken for collisions.
    assert_eq!(
        base.at_position(voxel(3, 0, 0)),
        base.at([3i128, 0, 0]),
        "a voxel position is its coordinates"
    );
    assert_eq!(
        base.at_position(voxel(3, 0, 0)),
        base.at_voxel(voxel(3, 0, 0)),
        "and `at_voxel` is the same shorthand"
    );

    for (index, (left_name, left)) in derived.iter().enumerate() {
        for (right_name, right) in derived.iter().skip(index + 1) {
            assert_ne!(
                left, right,
                "`{left_name}` and `{right_name}` derive the same seed"
            );
        }
    }
}

// ===========================================================================
// Octree levels: the bit-alignment hazard
// ===========================================================================

fn node(x: i128, y: i128, z: i128, level: i8) -> NodePosition3 {
    NodePosition3::new(Vector3::new(x, y, z), Depth::new(level))
}

/// An octree address at depth `d` is the voxel position shifted right by
/// `tree_depth − d`, so climbing a level drops a bit. The address `(4, 0, 0)`
/// therefore names a real node at many depths, and "address `a` at depth `d`" has the
/// *same bits* as "address `a << 1` at depth `d + 1`".
///
/// Nothing but the level distinguishes them, so this checks the level is actually
/// doing that work.
#[test]
fn shifted_addresses_at_shifted_depths_do_not_collide() {
    let world: Seed = world();

    // The direct statement of the hazard: the same bits, one level apart.
    for level in 1i8..40 {
        for base in [1i128, 3, 5, 7, 1_000_003] {
            let lower = world.at_position(node(base, 0, 0, level));
            let higher = world.at_position(node(base << 1, 0, 0, level + 1));

            assert_ne!(
                lower, higher,
                "address {base} at depth {level} collided with {} at depth {}",
                base << 1,
                level + 1
            );
        }
    }

    // And the whole ladder at once: one voxel's address all the way up the tree,
    // which is exactly the sequence of shifts the octree walks.
    let deepest: i128 = 0x5A5A_5A5A;
    let mut seen: HashSet<Seed> = HashSet::new();

    for level in 0i8..32 {
        let shift: u32 = level as u32;
        let address: i128 = deepest >> shift;
        let derived = world.at_position(node(address, address, address, 31 - level));

        assert!(
            seen.insert(derived),
            "level {level} of the ancestor chain repeated an earlier level"
        );
    }
}

/// Every `(address, level)` pair in a block must be its own seed. A sweep catches the
/// alignments a hand-picked case would miss.
#[test]
fn addresses_and_levels_are_jointly_distinct() {
    let world: Seed = world();
    let mut seen: HashSet<Seed> = HashSet::new();

    for level in 0i8..24 {
        for x in -6i128..6 {
            for y in -6i128..6 {
                let derived = world.at_position(node(x, y, 0, level));

                assert!(
                    seen.insert(derived),
                    "({x}, {y}) at depth {level} collided with an earlier pair"
                );
            }
        }
    }

    assert_eq!(seen.len(), 24 * 12 * 12, "every pair accounted for");
}

/// A level is not an index. Before this had a channel of its own, "the fifth thing at
/// this position" and "the depth-five node at this position" were the same seed.
#[test]
fn a_level_is_not_an_index_or_an_octave() {
    let world: Seed = world();
    let here = world.at([1i128, 2, 3]);

    for n in 0i8..64 {
        let as_level = here.at_level(Depth::new(n));
        let as_index = here.index(n as u64);
        let as_octave = here.at_octave(n as u32);

        assert_ne!(as_level, as_index, "level {n} must not be index {n}");
        assert_ne!(as_level, as_octave, "level {n} must not be octave {n}");
        assert_ne!(as_index, as_octave, "index {n} must not be octave {n}");
    }

    // The specific pairing that used to collide.
    assert_ne!(
        world.at_depth(Depth::new(5), [1i128, 2, 3]),
        world.at([1i128, 2, 3]).index(5),
        "a depth-five node is not the fifth thing at that position"
    );

    // Negative levels work too — `Depth` is signed, for levels above the root.
    let mut seen: HashSet<Seed> = HashSet::new();
    for n in -40i8..40 {
        assert!(seen.insert(here.at_level(Depth::new(n))), "level {n} collided");
    }
}

// ===========================================================================
// Octaves
// ===========================================================================

/// Two octaves sharing randomness makes the second a scaled copy of the first, which
/// reads as a repeating pattern rather than noise — so this is the collision most
/// visible to a player.
#[test]
fn octaves_are_independent_of_each_other() {
    let terrain: Seed = world().child("terrain");
    let mut seen: HashSet<Seed> = HashSet::new();

    for octave in 0..10_000u32 {
        assert!(
            seen.insert(terrain.at_octave(octave)),
            "octave {octave} repeated an earlier one"
        );
    }

    // And an octave at a place is distinct per octave *and* per place.
    let mut places: HashSet<Seed> = HashSet::new();
    for octave in 0..8u32 {
        for x in 0i128..32 {
            let derived = terrain.at_octave(octave).at_position(voxel(x, 0, 0));

            assert!(places.insert(derived), "octave {octave} at x = {x} collided");
        }
    }

    // Order matters and is the caller's choice, but both orders must be stable.
    let a = terrain.at_octave(3).at_position(voxel(1, 2, 3));
    let b = terrain.at_position(voxel(1, 2, 3)).at_octave(3);
    assert_ne!(a, b, "the two orders are different questions");
    assert_eq!(a, terrain.at_octave(3).at_position(voxel(1, 2, 3)), "stable");
    assert_eq!(b, terrain.at_position(voxel(1, 2, 3)).at_octave(3), "stable");
}

// ---------------------------------------------------------------------------
// The semantic lanes
//
// Each of these answers a different question about the same small integer. The
// point of having many is that a world asks several of them at once, so the tests
// that matter most are the ones checking they stay out of each other's way.
// ---------------------------------------------------------------------------

#[test]
fn every_lane_is_a_different_question_about_the_same_number() {
    let world = Seed::from_integer(12345u64);

    // Every derivation that takes a small integer, asked the same one. If any two
    // shared a channel, two unrelated parts of a world would move together.
    for value in [0u64, 1, 2, 7, 63, 1000, u64::from(u32::MAX)] {
        let small = value as u32;

        let derived: Vec<(&str, Seed)> = vec![
            ("index", world.index(value)),
            ("at", world.at([value as i128])),
            ("level", world.at_level(Depth::new(small.min(20) as i8))),
            ("octave", world.at_octave(small)),
            ("tick", world.at_tick(Tick::new(value))),
            ("direction", world.at_direction(small)),
            ("axis", world.at_axis(small)),
            ("channel", world.at_channel(small)),
            ("attempt", world.at_attempt(small)),
            ("step", world.at_step(value)),
            ("pass", world.at_pass(small)),
            ("rule", world.at_rule(small)),
            ("detail", world.at_detail(small)),
            ("epoch", world.at_epoch(value)),
            ("version", world.at_version(small)),
            ("advance", world.advance_by(value)),
        ];

        let unique: HashSet<Seed> = derived.iter().map(|(_, seed)| *seed).collect();

        assert_eq!(
            unique.len(),
            derived.len(),
            "two lanes collided on {value}: {:?}",
            derived
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<&str>>()
        );
    }
}

/// One derivation, named, so the table below reads as a list rather than a type.
type Lane = fn(Seed, u32) -> Seed;

#[test]
fn each_lane_separates_its_own_values() {
    let world = Seed::from_integer(99u64);

    // Within a lane, neighbouring values must not share a seed either — this is the
    // ordinary case, and the one that would make a pattern visible in a world.
    let lanes: [(&str, Lane); 10] = [
        ("direction", |seed, n| seed.at_direction(n)),
        ("axis", |seed, n| seed.at_axis(n)),
        ("channel", |seed, n| seed.at_channel(n)),
        ("attempt", |seed, n| seed.at_attempt(n)),
        ("step", |seed, n| seed.at_step(u64::from(n))),
        ("pass", |seed, n| seed.at_pass(n)),
        ("rule", |seed, n| seed.at_rule(n)),
        ("detail", |seed, n| seed.at_detail(n)),
        ("epoch", |seed, n| seed.at_epoch(u64::from(n))),
        ("version", |seed, n| seed.at_version(n)),
    ];

    for (name, lane) in lanes {
        let seeds: HashSet<Seed> = (0..512).map(|n| lane(world, n)).collect();

        assert_eq!(seeds.len(), 512, "{name} reused a seed within its own lane");
    }
}

#[test]
fn the_lanes_are_deterministic() {
    let world = Seed::from_integer(7u64);

    // The whole contract: the same question always has the same answer.
    assert_eq!(world.at_direction(3), world.at_direction(3));
    assert_eq!(world.at_epoch(1 << 40), world.at_epoch(1 << 40));
    assert_eq!(world.at_version(2).at_channel(1), world.at_version(2).at_channel(1));

    // And order within a chain is part of the question, not incidental.
    assert_ne!(
        world.at_version(2).at_channel(1),
        world.at_channel(1).at_version(2),
    );
}

#[test]
fn a_named_tag_gives_a_lane_of_its_own() {
    const VEGETATION: u128 = domain_tag("test::vegetation");
    const MINERALS: u128 = domain_tag("test::minerals");

    assert_ne!(VEGETATION, MINERALS);
    assert_ne!(VEGETATION, 0, "a zero tag would leave the lane untagged");

    let world = Seed::from_integer(4u64);

    assert_ne!(world.derive_tagged(VEGETATION, 9), world.derive_tagged(MINERALS, 9));
    assert_ne!(world.derive_tagged(VEGETATION, 9), world.index(9));
    assert_eq!(world.derive_tagged(VEGETATION, 9), world.derive_tagged(VEGETATION, 9));
}

#[test]
fn similar_names_do_not_give_similar_tags() {
    // The hash has to separate names that differ by one byte, by their order, and by
    // length — a weak mixer would leave neighbouring names sharing a lane.
    let names: [&str; 8] = [
        "a",
        "b",
        "ab",
        "ba",
        "test::ore",
        "test::ord",
        "test::or",
        "test::ore ",
    ];

    let tags: HashSet<u128> = names.iter().map(|name| domain_tag(name)).collect();

    assert_eq!(tags.len(), names.len(), "two names shared a tag");

    // Anagrams in particular, which a position-blind hash would merge.
    assert_ne!(domain_tag("stone"), domain_tag("tones"));
}

#[test]
fn a_type_can_claim_a_lane_for_its_values() {
    #[derive(Copy, Clone)]
    enum Ore {
        Iron,
        Copper,
        Tin,
    }

    seed_domain!(Ore => "test::ore_kind");

    #[derive(Copy, Clone)]
    enum Tree {
        Pine,
        Birch,
    }

    seed_domain!(Tree => "test::tree_kind");

    let world = Seed::from_integer(21u64);

    // Variants within a type are separate.
    let within: HashSet<Seed> = [Ore::Iron, Ore::Copper, Ore::Tin]
        .into_iter()
        .map(|ore| world.derive(ore))
        .collect();

    assert_eq!(within.len(), 3);

    // And the first variant of one type is not the first variant of another, which
    // is the mistake a shared lane would make.
    assert_ne!(world.derive(Ore::Iron), world.derive(Tree::Pine));
    assert_ne!(world.derive(Ore::Copper), world.derive(Tree::Birch));

    assert_eq!(world.derive(Ore::Tin), world.derive(Ore::Tin));
}

#[test]
fn octree_levels_stay_apart_when_their_addresses_align() {
    // The original reason these lanes exist. An octree address is the position
    // shifted right once per level, so a node high in the tree has the same small
    // address as a node near the origin lower down. Without the level in the
    // derivation, those two would be the same question.
    let world = Seed::from_integer(1u64);
    let mut seen: HashSet<Seed> = HashSet::new();

    for level in 0..16i8 {
        for address in 0..8i128 {
            let seed = world
                .at_level(Depth::new(level))
                .at([address, address, address]);

            assert!(
                seen.insert(seed),
                "level {level} address {address} reused an earlier seed",
            );
        }
    }

    assert_eq!(seen.len(), 16 * 8);
}

#[test]
fn the_version_numbers_are_pinned_so_a_bump_is_deliberate() {
    use voxel_world::random::{
        DISTRIBUTION_ALGORITHM_VERSION, RANDOM_ALGORITHM_VERSION, SEED_ALGORITHM_VERSION,
    };

    // Each covers one stage of turning a world seed into a value: the derivation, the
    // word stream, and the samplers that read it. A world's metadata needs all three,
    // because a change to any one of them changes what gets generated.
    //
    // They are pinned here for the same reason the trigonometry has golden bits: a
    // bump is a statement that saved worlds will not replay, and it should fail a
    // test on its way in rather than arrive unnoticed.
    assert_eq!(SEED_ALGORITHM_VERSION, 2, "seed derivation");
    assert_eq!(RANDOM_ALGORITHM_VERSION, 1, "xoshiro stream and seed expansion");
    assert_eq!(DISTRIBUTION_ALGORITHM_VERSION, 1, "samplers");
}
