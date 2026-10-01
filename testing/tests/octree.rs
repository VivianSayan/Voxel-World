//! Reading and writing an octree by position: descent, subdivision, packing,
//! collapse, and what it refuses.

use voxel_world::random::seed::Seed;
use voxel_world::units::Weights;
use voxel_world::world::{Octree, VoxelType, VoxelTypeIndex};

/// Three registered types to build with.
fn types() -> (VoxelTypeIndex, [voxel_world::world::VoxelTypeId; 3]) {
    let mut registry = VoxelTypeIndex::new();
    let air = registry.add_voxel_type(VoxelType::new("air")).unwrap();
    let stone = registry.add_voxel_type(VoxelType::new("stone")).unwrap();
    let dirt = registry.add_voxel_type(VoxelType::new("dirt")).unwrap();

    (registry, [air, stone, dirt])
}

fn seed() -> Seed {
    Seed::from_text("octree test")
}

#[test]
fn a_uniform_node_answers_the_same_everywhere_in_it() {
    let (_registry, [air, _stone, _dirt]) = types();
    let tree = Octree::new_uniform(air);

    // Two levels: a 4-by-4-by-4 region, every corner of it the same.
    for position in [[0, 0, 0], [3, 3, 3], [1, 2, 3], [3, 0, 1]] {
        assert_eq!(tree.voxel(position, 2, seed()), Some(air));
        assert!(tree.is_decided(position, 2));
    }

    // Outside the node answers nothing rather than wrapping round.
    assert_eq!(tree.voxel([4, 0, 0], 2, seed()), None);
    assert!(!tree.is_decided([0, 0, 4], 2));
}

#[test]
fn an_undecided_node_reads_as_nothing_and_refuses_a_write() {
    let (_registry, [air, _stone, _dirt]) = types();

    let mut procedural = Octree::new_procedural();
    assert_eq!(procedural.voxel([0, 0, 0], 2, seed()), None);
    assert!(!procedural.is_decided([0, 0, 0], 2));
    assert!(
        !procedural.set_voxel([0, 0, 0], 2, 4, air),
        "writing would have to invent the voxels around it"
    );

    // A branch with no children is undecided in the same way.
    let mut empty = Octree::new_children();
    assert_eq!(empty.voxel([1, 1, 1], 2, seed()), None);
    assert!(!empty.is_decided([1, 1, 1], 2));
    assert!(!empty.set_voxel([1, 1, 1], 2, 4, air));

    // Filling it outright is the way to decide a region.
    procedural.fill(air);
    assert_eq!(procedural.voxel([0, 0, 0], 2, seed()), Some(air));
    assert!(procedural.is_decided([3, 3, 3], 2));
}

#[test]
fn writing_into_a_uniform_node_packs_it_at_the_bottom() {
    let (_registry, [air, stone, _dirt]) = types();
    let mut tree = Octree::new_uniform(air);

    // levels 2 (a 4-cube) with pack_at 4, so this node packs rather than splits.
    assert!(tree.set_voxel([1, 2, 3], 2, 4, stone));

    assert!(tree.packed().is_some(), "it became a packed block");
    assert!(tree.children().is_none(), "and did not subdivide");
    assert_eq!(tree.packed().unwrap().len(), 64, "4 * 4 * 4");

    // The written voxel changed and nothing else did.
    assert_eq!(tree.voxel([1, 2, 3], 2, seed()), Some(stone));
    for position in [[0, 0, 0], [3, 3, 3], [1, 2, 2]] {
        assert_eq!(tree.voxel(position, 2, seed()), Some(air));
    }
}

#[test]
fn writing_high_up_subdivides_instead_of_packing() {
    let (_registry, [air, stone, _dirt]) = types();
    let mut tree = Octree::new_uniform(air);

    // levels 6 with pack_at 4: the top two levels split, then a packed block.
    assert!(tree.set_voxel([0, 0, 0], 6, 4, stone));

    let children = tree.children().expect("it subdivided");
    assert_eq!(children.len(), 8);

    // Only the child holding the write changed; the other seven are still the
    // single uniform node they were split into.
    assert_eq!(
        children
            .iter()
            .filter(|c| c.uniform_id() == Some(air))
            .count(),
        7
    );

    assert_eq!(tree.voxel([0, 0, 0], 6, seed()), Some(stone));
    assert_eq!(tree.voxel([63, 63, 63], 6, seed()), Some(air));
    assert_eq!(tree.voxel([1, 0, 0], 6, seed()), Some(air));
}

#[test]
fn descent_reaches_every_octant() {
    let (_registry, [air, stone, dirt]) = types();

    // Write a different value in each of the eight corners of a 2-cube, which
    // is one voxel per octant, and read them all back.
    let mut tree = Octree::new_uniform(air);

    let corners = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ];

    // pack_at 0 forces subdivision all the way to single voxels.
    for (index, corner) in corners.iter().enumerate() {
        let id = if index % 2 == 0 { stone } else { dirt };
        assert!(tree.set_voxel(*corner, 1, 0, id), "writing {corner:?}");
    }

    for (index, corner) in corners.iter().enumerate() {
        let expected = if index % 2 == 0 { stone } else { dirt };
        assert_eq!(tree.voxel(*corner, 1, seed()), Some(expected), "{corner:?}");
    }
}

#[test]
fn agreeing_children_collapse_back_to_one_node() {
    let (_registry, [air, stone, _dirt]) = types();
    let mut tree = Octree::new_uniform(air);

    // Force subdivision to single voxels, then overwrite every one of them with
    // the same type: the tree should fold back up rather than keep eight nodes.
    for corner in [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ] {
        assert!(tree.set_voxel(corner, 1, 0, stone));
    }

    assert_eq!(
        tree.uniform_id(),
        Some(stone),
        "eight agreeing children became one uniform node"
    );
    assert!(tree.children().is_none());
    assert_eq!(tree.voxel([1, 1, 1], 1, seed()), Some(stone));
}

#[test]
fn a_weighted_node_must_be_materialised_before_it_can_be_written() {
    let (_registry, [air, stone, dirt]) = types();
    let weights = Weights::new([1.0, 1.0]).unwrap();
    let mut tree = Octree::new_probability(&weights, vec![air, stone]).unwrap();

    // It reads: every position resolves to one of its types, from the seed.
    let read = tree.voxel([1, 1, 1], 2, seed());
    assert!(read == Some(air) || read == Some(stone));
    assert!(tree.is_decided([1, 1, 1], 2));

    // But it cannot be written, because realising it needs a seed per voxel.
    assert!(!tree.set_voxel([1, 1, 1], 2, 4, dirt));

    // Materialising draws them all, keeping what the node held.
    let before: Vec<Option<_>> = (0..4)
        .map(|x| tree.voxel([x, 0, 0], 2, seed().at([x as i128, 0, 0])))
        .collect();

    assert!(tree.materialise(2, |local| {
        seed().at([local[0] as i128, local[1] as i128, local[2] as i128])
    }));
    assert!(tree.packed().is_some());

    let after: Vec<Option<_>> = (0..4).map(|x| tree.voxel([x, 0, 0], 2, seed())).collect();
    assert_eq!(before, after, "storage changed, contents did not");

    // And now it takes a write.
    assert!(tree.set_voxel([1, 1, 1], 2, 4, dirt));
    assert_eq!(tree.voxel([1, 1, 1], 2, seed()), Some(dirt));

    // Materialising anything else does nothing.
    assert!(!tree.materialise(2, |_| seed()));
    assert!(!Octree::new_uniform(air).materialise(2, |_| seed()));
}

#[test]
fn a_weighted_node_gives_neighbouring_voxels_different_types() {
    let (_registry, [air, stone, _dirt]) = types();
    let weights = Weights::new([1.0, 1.0]).unwrap();
    let tree = Octree::new_probability(&weights, vec![air, stone]).unwrap();

    // The point of passing the voxel's own seed: a weighted region is a mix,
    // not one type repeated.
    let world = seed();
    let mut seen_air = false;
    let mut seen_stone = false;

    for x in 0..16i128 {
        match tree.voxel([x as u32, 0, 0], 4, world.at([x, 0, 0])) {
            Some(id) if id == air => seen_air = true,
            Some(id) if id == stone => seen_stone = true,
            other => panic!("unexpected {other:?}"),
        }
    }

    assert!(seen_air && seen_stone, "a mix, not one type");

    // And it is the same mix every time it is asked.
    for x in 0..16i128 {
        assert_eq!(
            tree.voxel([x as u32, 0, 0], 4, world.at([x, 0, 0])),
            tree.voxel([x as u32, 0, 0], 4, world.at([x, 0, 0]))
        );
    }
}

#[test]
fn a_packed_node_reads_and_writes_by_position() {
    let (_registry, [air, stone, dirt]) = types();

    // 2 levels is a 4-cube: 64 voxels, x fastest then y then z.
    let mut tree = Octree::new_packed_uniform(64, air);

    assert!(tree.set_voxel([0, 0, 0], 2, 4, stone));
    assert!(tree.set_voxel([3, 3, 3], 2, 4, dirt));

    assert_eq!(tree.voxel([0, 0, 0], 2, seed()), Some(stone));
    assert_eq!(tree.voxel([3, 3, 3], 2, seed()), Some(dirt));
    assert_eq!(tree.voxel([1, 0, 0], 2, seed()), Some(air));

    // The index ordering matches Chunk's: x fastest, then y, then z.
    let packed = tree.packed().unwrap();
    assert_eq!(packed.get(0), Some(&stone), "[0,0,0] is index 0");
    assert_eq!(packed.get(63), Some(&dirt), "[3,3,3] is the last index");
}

#[test]
fn a_single_voxel_node_ignores_the_position_it_has_none() {
    let (_registry, [air, stone, _dirt]) = types();
    let mut tree = Octree::new_uniform(air);

    assert_eq!(tree.voxel([0, 0, 0], 0, seed()), Some(air));
    assert!(tree.is_decided([0, 0, 0], 0));

    // At zero levels the only position is the origin.
    assert_eq!(tree.voxel([1, 0, 0], 0, seed()), None);

    assert!(tree.set_voxel([0, 0, 0], 0, 4, stone));
    assert_eq!(tree.uniform_id(), Some(stone), "it just becomes that type");
}

#[test]
fn a_depth_beyond_a_coordinate_is_refused_rather_than_wrapping() {
    let (_registry, [air, stone, _dirt]) = types();
    let mut tree = Octree::new_uniform(air);

    assert_eq!(Octree::MAX_LEVELS, 31);
    assert_eq!(tree.voxel([0, 0, 0], 32, seed()), None);
    assert!(!tree.is_decided([0, 0, 0], 32));
    assert!(!tree.set_voxel([0, 0, 0], 32, 4, stone));

    // And a pack that would not fit in memory is refused, not attempted.
    assert!(!tree.set_voxel([0, 0, 0], 31, 31, stone));
    assert_eq!(tree.uniform_id(), Some(air), "nothing changed");
}

#[test]
fn a_whole_region_can_be_set_without_touching_its_voxels() {
    let (_registry, [air, stone, _dirt]) = types();
    let mut tree = Octree::new_uniform(air);

    // A 6-level node is 64 voxels a side. Set one 16-cube of it: region_levels
    // 4, so the coordinate counts in 16-cubes and runs 0..4.
    assert!(tree.set_region([1, 0, 0], 6, 4, 4, stone));

    // Everything in that 16-cube is stone...
    for x in 16..32u32 {
        for y in [0, 7, 15] {
            assert_eq!(tree.voxel([x, y, 0], 6, seed()), Some(stone), "{x},{y}");
        }
    }
    assert_eq!(tree.voxel([31, 15, 15], 6, seed()), Some(stone));

    // ...and nothing outside it is.
    assert_eq!(tree.voxel([15, 0, 0], 6, seed()), Some(air));
    assert_eq!(tree.voxel([32, 0, 0], 6, seed()), Some(air));
    assert_eq!(tree.voxel([16, 16, 0], 6, seed()), Some(air));

    // The filled region is one uniform node, not 4096 voxels of storage.
    assert_eq!(tree.region_type([1, 0, 0], 6, 4, seed()), Some(stone));
    assert_eq!(tree.region_type([0, 0, 0], 6, 4, seed()), Some(air));
}

#[test]
fn a_region_covering_the_whole_node_replaces_it() {
    let (_registry, [air, stone, dirt]) = types();

    // Whatever the node was, including packed storage, it becomes one type.
    let mut tree = Octree::new_packed_uniform(64, air);
    assert!(tree.set_voxel([1, 1, 1], 2, 4, dirt));
    assert!(tree.packed().is_some());

    assert!(tree.set_region([0, 0, 0], 2, 2, 4, stone));
    assert_eq!(tree.uniform_id(), Some(stone), "storage dropped with it");
    assert!(tree.packed().is_none());

    // Even an undecided node takes a whole-node region, which is what `fill`
    // does: there is nothing below to invent.
    let mut procedural = Octree::new_procedural();
    assert!(procedural.set_region([0, 0, 0], 3, 3, 4, stone));
    assert_eq!(procedural.uniform_id(), Some(stone));
}

#[test]
fn the_position_shifts_with_the_region_size() {
    let (_registry, [air, stone, _dirt]) = types();

    // The voxel at [20, 5, 33] sits in these regions:
    assert_eq!(Octree::region_of([20, 5, 33], 0), [20, 5, 33], "itself");
    assert_eq!(Octree::region_of([20, 5, 33], 2), [5, 1, 8], "4-cubes");
    assert_eq!(Octree::region_of([20, 5, 33], 4), [1, 0, 2], "16-cubes");

    // And the shift undoes to the region's low corner.
    assert_eq!(Octree::region_origin([1, 0, 2], 4), [16, 0, 32]);
    assert_eq!(Octree::region_origin([5, 1, 8], 2), [20, 4, 32]);

    // Setting the 16-cube that holds a voxel, found by the shift, contains it.
    let mut tree = Octree::new_uniform(air);
    let voxel = [20u32, 5, 33];
    let region = Octree::region_of(voxel, 4);

    assert!(tree.set_region(region, 6, 4, 4, stone));
    assert_eq!(tree.voxel(voxel, 6, seed()), Some(stone));
    assert_eq!(
        tree.voxel([16, 0, 32], 6, seed()),
        Some(stone),
        "its corner"
    );
    assert_eq!(
        tree.voxel([15, 0, 32], 6, seed()),
        Some(air),
        "just outside"
    );
}

#[test]
fn a_region_inside_a_packed_block_fills_a_cube_not_a_run() {
    let (_registry, [air, stone, _dirt]) = types();

    // A 4-level node with pack_at 4 packs rather than subdividing, so this
    // exercises the row-by-row fill: a cube is not contiguous in a linear block.
    let mut tree = Octree::new_uniform(air);
    assert!(tree.set_region([0, 0, 0], 4, 2, 4, stone));

    assert!(tree.packed().is_some(), "it packed rather than subdividing");

    // The 4-cube at the origin is stone, and only it.
    for z in 0..16u32 {
        for y in 0..16u32 {
            for x in 0..16u32 {
                let inside = x < 4 && y < 4 && z < 4;
                let expected = if inside { stone } else { air };

                assert_eq!(
                    tree.voxel([x, y, z], 4, seed()),
                    Some(expected),
                    "at {x},{y},{z}"
                );
            }
        }
    }
}

#[test]
fn filling_a_region_in_pieces_leaves_the_same_tree_as_filling_it_at_once() {
    let (_registry, [air, stone, _dirt]) = types();

    // Eight 8-cubes make the 16-cube they sit in, so filling all eight should
    // collapse to exactly what filling the parent does.
    let mut in_pieces = Octree::new_uniform(air);
    for region in [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ] {
        assert!(in_pieces.set_region(region, 6, 3, 2, stone));
    }

    // levels 6, region 4 addresses 16-cubes; [0,0,0] is the one just filled.
    assert_eq!(
        in_pieces.region_type([0, 0, 0], 6, 4, seed()),
        Some(stone),
        "the eight pieces collapsed into the 16-cube"
    );

    let mut at_once = Octree::new_uniform(air);
    assert!(at_once.set_region([0, 0, 0], 6, 4, 2, stone));
    assert_eq!(at_once.region_type([0, 0, 0], 6, 4, seed()), Some(stone));

    // Both read alike, inside the filled region and outside it.
    for position in [[0, 0, 0], [15, 15, 15], [16, 0, 0], [63, 63, 63]] {
        assert_eq!(
            in_pieces.voxel(position, 6, seed()),
            at_once.voxel(position, 6, seed()),
            "at {position:?}"
        );
    }
}

#[test]
fn a_region_larger_than_the_node_or_outside_it_is_refused() {
    let (_registry, [air, stone, _dirt]) = types();
    let mut tree = Octree::new_uniform(air);

    // Bigger than the node.
    assert!(!tree.set_region([0, 0, 0], 2, 3, 4, stone));
    assert_eq!(tree.region_type([0, 0, 0], 2, 3, seed()), None);

    // Outside it: at levels 4 with 2-level regions the coordinate runs 0..4.
    assert!(!tree.set_region([4, 0, 0], 4, 2, 4, stone));
    assert!(tree.set_region([3, 0, 0], 4, 2, 4, stone));

    // And the refusals that apply to a single voxel apply to a region too,
    // unless the region is the whole node.
    let mut weighted =
        Octree::new_probability(&Weights::new([1.0, 1.0]).unwrap(), vec![air, stone]).unwrap();
    assert!(!weighted.set_region([0, 0, 0], 4, 2, 4, stone));
    assert!(
        weighted.set_region([0, 0, 0], 4, 4, 4, stone),
        "the whole node"
    );
}

#[test]
fn a_region_read_inherits_a_coarser_uniform_node() {
    let (_registry, [air, stone, _dirt]) = types();
    let mut tree = Octree::new_uniform(air);

    // Fill a 16-cube, then ask about a 4-cube inside it: a uniform node is
    // uniform throughout, so the smaller region inherits its answer.
    assert!(tree.set_region([0, 0, 0], 6, 4, 2, stone));

    assert_eq!(tree.region_type([0, 0, 0], 6, 2, seed()), Some(stone));
    assert_eq!(tree.region_type([3, 3, 3], 6, 2, seed()), Some(stone));
    assert_eq!(
        tree.region_type([4, 0, 0], 6, 2, seed()),
        Some(air),
        "outside"
    );

    // A region spanning a branch has no single answer.
    assert!(tree.set_voxel([0, 0, 0], 6, 4, air));
    assert_eq!(
        tree.region_type([0, 0, 0], 6, 4, seed()),
        None,
        "the 16-cube now varies"
    );
}
