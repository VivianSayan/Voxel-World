use voxel_world::math::{Vector2, Vector3};
use voxel_world::random::seed::Seed;
use voxel_world::spatial::{
    Depth, TreeDepth, VoxelPosition2, VoxelPosition3,
    noise::{cellular::get_2d_cellular, value_noise::get_3d_value, white_noise::get_3d_white},
};
use voxel_world::world::{Chunk, ChunkGenerator, VoxelType, VoxelTypeIndex, octree::Octree};

#[test]
fn generation_is_independent_of_palette_registry_and_chunk_order() {
    let mut first_registry = VoxelTypeIndex::new();
    let air = first_registry
        .add_voxel_type(VoxelType::new("air"))
        .unwrap();
    let stone = first_registry
        .add_voxel_type(VoxelType::new("stone"))
        .unwrap();
    let mut second_registry = VoxelTypeIndex::new();
    let other_stone = second_registry
        .add_voxel_type(VoxelType::new("stone"))
        .unwrap();
    let other_air = second_registry
        .add_voxel_type(VoxelType::new("air"))
        .unwrap();
    let seed = Seed::from_text("order independence");
    let generator =
        ChunkGenerator::new(seed, &first_registry, &[(stone, 3.0), (air, 1.0)]).unwrap();
    let reordered = ChunkGenerator::new(
        seed,
        &second_registry,
        &[(other_air, 1.0), (other_stone, 3.0)],
    )
    .unwrap();
    let origins = [
        VoxelPosition3::new(Vector3::new(-16, -16, -16)),
        VoxelPosition3::default(),
    ];
    let chunks: Vec<_> = origins
        .map(|origin| generator.generate(origin).unwrap())
        .into();
    for (index, origin) in origins.into_iter().enumerate().rev() {
        let chunk = &chunks[index];
        assert_eq!(*chunk, generator.generate(origin).unwrap());
        let other = reordered.generate(origin).unwrap();
        assert_eq!(chunk.voxels().len(), Chunk::VOLUME);
        for (&id, &other_id) in chunk.voxels().iter().zip(other.voxels()) {
            assert_eq!(
                first_registry.get_voxel_type(id).unwrap().name(),
                second_registry.get_voxel_type(other_id).unwrap().name()
            );
        }
        assert_eq!(chunk.voxel(origin), Some(generator.voxel(origin)));
        assert_eq!(chunk.voxel(origin.offset(Vector3::new(-1, 0, 0))), None);
        assert_eq!(chunk.voxel(origin.offset(Vector3::new(16, 0, 0))), None);
    }
    let changed_seed = ChunkGenerator::new(
        seed.child("different"),
        &first_registry,
        &[(air, 1.0), (stone, 3.0)],
    )
    .unwrap();
    assert_ne!(
        chunks[1].voxels(),
        changed_seed
            .generate(VoxelPosition3::default())
            .unwrap()
            .voxels()
    );
}

#[test]
fn registry_palette_and_chunk_bounds_are_checked() {
    let mut registry = VoxelTypeIndex::new();
    assert!(registry.add_voxel_type(VoxelType::new("")).is_none());
    let id = registry.add_voxel_type(VoxelType::new("stone")).unwrap();
    assert!(registry.add_voxel_type(VoxelType::new("stone")).is_none());
    assert!(ChunkGenerator::new(Seed::from_raw(0), &registry, &[]).is_none());
    assert!(ChunkGenerator::new(Seed::from_raw(0), &registry, &[(id, 0.0)]).is_none());
    assert!(ChunkGenerator::new(Seed::from_raw(0), &registry, &[(id, 1.0), (id, 2.0)]).is_none());
    assert!(
        ChunkGenerator::new(
            Seed::from_raw(0),
            &registry,
            &[(voxel_world::world::VoxelTypeId::NONE, 1.0)]
        )
        .is_none()
    );
    let generator = ChunkGenerator::new(Seed::from_raw(0), &registry, &[(id, 1.0)]).unwrap();
    assert!(
        generator
            .generate(VoxelPosition3::new(Vector3::splat(i128::MAX)))
            .is_none()
    );
    let at_min = generator
        .generate(VoxelPosition3::new(Vector3::splat(i128::MIN)))
        .unwrap();
    assert!(at_min.voxels().iter().all(|held| *held == id));
    let children = Octree::from_children(std::array::from_fn(|_| Octree::new_uniform(id)));
    assert_eq!(children.children().unwrap().len(), 8);
    assert!(
        children
            .children()
            .unwrap()
            .iter()
            .all(|node| node.resolve(Seed::from_raw(0)) == Some(id))
    );
}

#[test]
fn packed_nodes_choose_compact_voxel_storage() {
    use voxel_world::structures::storage::{CompactKind, Palette};
    use voxel_world::units::Weights;

    let mut registry = VoxelTypeIndex::new();
    let stone = registry.add_voxel_type(VoxelType::new("stone")).unwrap();
    let dirt = registry.add_voxel_type(VoxelType::new("dirt")).unwrap();
    let seed = Seed::from_raw(5);

    // A packed leaf starts as a single shared value with no per-voxel indices.
    let mut node = Octree::new_packed_uniform(4096, stone);
    assert!(node.is_resolvable());
    assert_eq!(node.packed().unwrap().len(), 4096);
    assert_eq!(node.packed().unwrap().index_bits(), 0);
    assert_eq!(node.packed().unwrap().packed_bytes(), 0);
    assert_eq!(node.packed().unwrap().kind(), CompactKind::Uniform);

    // It answers per position rather than for the whole node.
    assert_eq!(
        node.resolve(seed),
        None,
        "a packed node holds no single type"
    );
    assert_eq!(node.resolve_at(0, seed), Some(stone));
    assert_eq!(node.resolve_at(9999, seed), None, "past the block");

    // Writing one voxel is a masked write, not a subdivision.
    node.packed_mut().unwrap().set(17, dirt);
    assert_eq!(node.resolve_at(17, seed), Some(dirt));
    assert_eq!(node.resolve_at(18, seed), Some(stone));
    assert_eq!(node.packed().unwrap().index_bits(), 1);
    assert_eq!(node.packed().unwrap().kind(), CompactKind::Palette);
    assert!(node.packed().unwrap().packed_bytes() < 4096 * size_of::<u64>());

    // A node built from a palette directly.
    let built = Octree::new_packed(Palette::filled(8, dirt));
    assert_eq!(built.resolve_at(7, seed), Some(dirt));
    assert!(built.packed().is_some() && built.children().is_none());

    // The other kinds still behave, and resolve_at agrees with resolve for them.
    let uniform = Octree::new_uniform(stone);
    assert_eq!(uniform.resolve_at(123, seed), Some(stone));
    assert!(uniform.packed().is_none());

    let weights = Weights::new([3.0, 1.0]).unwrap();
    let weighted = Octree::new_probability(&weights, vec![stone, dirt]).unwrap();
    assert!(weighted.is_resolvable());
    assert_eq!(weighted.resolve_at(0, seed), weighted.resolve(seed));

    let branch = Octree::new_children();
    assert!(!branch.is_resolvable());
    assert_eq!(branch.resolve_at(0, seed), None);
    assert!(!Octree::new_procedural().is_resolvable());

    // Every node is one pointer plus its tag, whichever kind it is.
    assert_eq!(size_of::<Octree>(), 16);
}

#[test]
fn noise_clamps_named_levels_consistently_and_stays_in_range() {
    let tree = TreeDepth::new(8).unwrap();
    let seed = Seed::from_raw(17);
    let point = VoxelPosition3::new(Vector3::new(-1, -45, 301));
    assert_eq!(
        get_3d_white(seed, tree, Depth::new(100), point),
        get_3d_white(seed, tree, tree.floor(), point)
    );
    assert_eq!(
        get_3d_white(seed, tree, Depth::HIGHEST, point),
        get_3d_white(seed, tree, tree.ceiling(), point)
    );
    for x in -32..32 {
        let point = VoxelPosition3::new(Vector3::new(x, -x, x * 2));
        assert!((-1.0..=1.0).contains(&get_3d_value(seed, tree, Depth::ROOT, 8, point).value()));
        let cell = get_2d_cellular(
            seed,
            tree,
            Depth::new(4),
            VoxelPosition2::new(Vector2::new(x, -x)),
        );
        assert!(cell.nearest >= 0.0 && cell.second >= cell.nearest);
    }
}
