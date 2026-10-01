//! The structures and traits added to `structures`: the new containers, and the
//! promises the existing ones now make.

use voxel_world::define_id_kinds;
use voxel_world::random::seed::Seed;
use voxel_world::random::{BernoulliMask, Distribution, Random};
use voxel_world::spatial::MortonKey;
use voxel_world::structures::collections::{
    BitSet, BoundedOrderedSet, BucketQueue, DisjointSets, FuzzySet, MultiSet, OrderedSet,
    PriorityQueue, RunLengthSequence, Scheduler, Set, UniqueScheduler, WeightedSet,
};
use voxel_world::structures::mappings::{IntervalMap, LruCache};
use voxel_world::structures::storage::{
    BitGrid2, BitGrid3, BitGrid4, CompactKind, CompactSequence, Grid2, Grid3, Grid4, Palette,
    SlotMap,
};
use voxel_world::structures::traits::{
    Bounded, Capacity, ContentHashable, Grouping, HandleStore, MapMut, PriorityQueueLike,
    RangeQuery, ValueCollection, WeightedChoose,
};
use voxel_world::units::{Id, Probability, Ratio};

define_id_kinds! {
    TestEntityKind => "test entity",
}

fn rng(seed: u128) -> Random {
    Random::new(Seed::from_raw(seed))
}

#[test]
fn bit_sets_pack_members_and_combine_by_word() {
    let mut set = BitSet::new();
    assert!(set.insert(3) && set.insert(64) && set.insert(200));
    assert!(!set.insert(3));
    assert_eq!(set.len(), 3);
    assert!(set.contains(64) && !set.contains(65));
    assert_eq!(set.iter().collect::<Vec<usize>>(), [3, 64, 200]);
    assert_eq!((set.first(), set.last()), (Some(3), Some(200)));
    assert!(set.remove(64) && !set.remove(64));
    assert_eq!(set.len(), 2);

    let left: BitSet = [1usize, 2, 3].into_iter().collect();
    let right: BitSet = [3usize, 4].into_iter().collect();
    assert_eq!(
        (&left | &right).iter().collect::<Vec<usize>>(),
        [1, 2, 3, 4]
    );
    assert_eq!((&left & &right).iter().collect::<Vec<usize>>(), [3]);
    assert_eq!((&left - &right).iter().collect::<Vec<usize>>(), [1, 2]);
    assert_eq!((&left ^ &right).iter().collect::<Vec<usize>>(), [1, 2, 4]);
    assert!(BitSet::from_iter([1usize, 2]).is_subset(&left));
    assert!(left.is_disjoint(&BitSet::from_iter([9usize])));

    // A sampler's mask becomes a set without unpacking it.
    let mask = BernoulliMask::new(Ratio::new(1, 2).unwrap()).unwrap();
    let word = mask.sample(&mut rng(7));
    let from_mask = BitSet::from_word(word);
    assert_eq!(from_mask.len(), word.count_ones() as usize);
    assert_eq!(from_mask.word(0), word);

    // Value-based collection view, since members are computed not stored.
    assert_eq!(ValueCollection::len(&left), 3);
    assert!(left.contains_value(2));
    assert_eq!(left.values().collect::<Vec<usize>>(), [1, 2, 3]);

    // Choice is uniform over the members held.
    let mut random = rng(11);
    let mut seen = Set::new();
    for _ in 0..200 {
        seen.insert(left.choose_member(&mut random).unwrap());
    }
    assert_eq!(seen.len(), 3);
    assert_eq!(BitSet::new().choose_member(&mut random), None);
}

#[test]
fn disjoint_sets_merge_and_answer_connectivity() {
    let mut caves: DisjointSets<&str> = DisjointSets::new();
    assert!(caves.join("a", "b"));
    assert!(caves.join("c", "d"));
    assert!(!caves.join("a", "b"));
    assert_eq!(caves.group_count(), 2);
    assert!(caves.joined(&"a", &"b") && !caves.joined(&"a", &"c"));
    assert_eq!(caves.group_size(&"a"), 2);

    assert!(caves.join("b", "c"));
    assert_eq!(caves.group_count(), 1);
    assert!(caves.joined(&"a", &"d"));
    assert_eq!(caves.group_size(&"d"), 4);
    assert_eq!(DisjointSets::groups(&mut caves).len(), 1);

    caves.insert("lonely");
    assert_eq!(caves.group_count(), 2);
    assert_eq!(caves.len(), 5);
    assert!(!caves.joined(&"lonely", &"a"));
    assert_eq!(
        DisjointSets::groups(&mut caves)
            .iter()
            .map(Set::len)
            .sum::<usize>(),
        5
    );
}

#[test]
fn bucket_queues_serve_lowest_priority_first_in_arrival_order() {
    let mut queue: BucketQueue<&str> = BucketQueue::new(16);
    assert!(queue.push(15, "dim"));
    assert!(queue.push(0, "bright"));
    assert!(queue.push(0, "also bright"));
    assert!(!queue.push(16, "out of range"));
    assert_eq!(queue.len(), 3);

    assert_eq!(queue.pop(), Some((0, "bright")));
    assert_eq!(queue.pop(), Some((0, "also bright")));
    // Work discovered at a lower priority after the cursor moved is still served.
    queue.push(2, "found later");
    assert_eq!(queue.pop(), Some((2, "found later")));
    assert_eq!(queue.pop(), Some((15, "dim")));
    assert_eq!(queue.pop(), None);
    assert!(queue.is_empty());
}

#[test]
fn run_length_sequences_merge_and_split_runs() {
    let mut column: RunLengthSequence<&str> = RunLengthSequence::filled(320, "air");
    assert_eq!(column.run_count(), 1);
    assert!(column.is_uniform());

    column.fill(0, 62, "stone");
    column.fill(62, 64, "dirt");
    assert_eq!(column.run_count(), 3);
    assert_eq!(column.get(0), Some(&"stone"));
    assert_eq!(column.get(63), Some(&"dirt"));
    assert_eq!(column.get(64), Some(&"air"));
    assert_eq!(column.get(320), None);
    assert!(!column.is_uniform());

    // Writing a value a position already holds changes nothing.
    assert!(!column.set(0, "stone"));
    assert_eq!(column.run_count(), 3);
    // Writing into the middle of a run splits it.
    assert!(column.set(30, "ore"));
    assert_eq!(column.run_count(), 5);
    // And writing it back merges the neighbours again.
    assert!(column.set(30, "stone"));
    assert_eq!(column.run_count(), 3);

    assert_eq!(
        column.values().filter(|value| **value == "stone").count(),
        62
    );
    assert_eq!(column.runs().count(), 3);
    assert_eq!(RangeQuery::range(&column, 60..70).count(), 3);

    let built: RunLengthSequence<u8> = [1u8, 1, 1, 2, 2, 3].into_iter().collect();
    assert_eq!(built.run_count(), 3);
    assert_eq!(built.len(), 6);
}

#[test]
fn interval_maps_override_and_merge_spans() {
    let mut bands: IntervalMap<i32, &str> = IntervalMap::new();
    bands.insert(0, 64, "stone");
    bands.insert(64, 96, "dirt");
    assert_eq!(bands.get(&0), Some(&"stone"));
    assert_eq!(bands.get(&63), Some(&"stone"));
    assert_eq!(bands.get(&64), Some(&"dirt"));
    assert_eq!(bands.get(&200), None);
    assert_eq!(bands.len(), 2);

    // A later span cuts the earlier one back.
    bands.insert(32, 40, "ore");
    assert_eq!(bands.len(), 4);
    assert_eq!(bands.get(&31), Some(&"stone"));
    assert_eq!(bands.get(&35), Some(&"ore"));
    assert_eq!(bands.get(&45), Some(&"stone"));

    // Equal neighbours merge back together.
    bands.insert(32, 40, "stone");
    assert_eq!(bands.len(), 2);

    bands.remove(0, 16);
    assert_eq!(bands.get(&8), None);
    assert_eq!(RangeQuery::range(&bands, 20..70).count(), 2);
}

#[test]
fn caches_evict_the_least_recently_used() {
    let mut cache: LruCache<&str, u32> = LruCache::new(2);
    assert!(!cache.insert("a", 1).evicted_something());
    assert!(!cache.insert("b", 2).evicted_something());
    // Using "a" makes "b" the next to go.
    assert_eq!(cache.get(&"a"), Some(&1));
    assert_eq!(cache.next_eviction(), Some(&"b"));

    let evicted = cache.insert("c", 3);
    assert_eq!(evicted.dropped, Some(("b", 2)));
    assert!(cache.contains_key(&"a") && cache.contains_key(&"c"));
    assert_eq!(cache.peek(&"b"), None);
    assert_eq!(cache.len(), 2);
    assert!(cache.is_full());

    // Peeking does not change what goes next.
    let before = cache.next_eviction().copied();
    cache.peek(&"a");
    assert_eq!(cache.next_eviction().copied(), before);
    assert_eq!(cache.insert("a", 9).replaced, Some(1));
    assert_eq!(cache.remove(&"a"), Some(9));
}

#[test]
fn slot_maps_retire_ids_with_their_values() {
    let mut entities: SlotMap<TestEntityKind, &str> = SlotMap::new();
    let first: Id<TestEntityKind> = entities.insert("player");
    let second = entities.insert("boulder");
    assert_eq!(entities.get(first), Some(&"player"));
    assert_eq!(entities.len(), 2);

    assert_eq!(entities.remove(first), Some("player"));
    assert_eq!(entities.get(first), None);
    assert!(!entities.contains(first));

    // The freed slot is reused, and the old id still does not resolve.
    let third = entities.insert("arrow");
    assert_eq!(entities.slot_count(), 2);
    assert_eq!(entities.get(third), Some(&"arrow"));
    assert_eq!(entities.get(first), None);
    assert_ne!(third, first);

    *entities.get_mut(second).unwrap() = "rock";
    assert_eq!(entities.get(second), Some(&"rock"));
    assert_eq!(entities.iter().count(), 2);

    entities.retain(|id, _| id == second);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities.get(third), None);

    entities.clear();
    assert!(entities.is_empty());
    assert_eq!(entities.get(second), None);
    assert_eq!(SlotMap::<TestEntityKind, u8>::new().get(Id::NONE), None);
}

#[test]
fn palettes_pack_repeating_values_and_widen_as_needed() {
    let mut chunk: Palette<u16> = Palette::filled(4096, 0);
    assert_eq!(chunk.len(), 4096);
    assert!(chunk.is_uniform());
    assert_eq!(chunk.index_bits(), 0);
    assert_eq!(chunk.packed_bytes(), 0, "a uniform chunk stores no indices");

    assert!(chunk.set(10, 1));
    assert_eq!(chunk.index_bits(), 1, "two values need one bit each");
    assert_eq!(chunk.packed_bytes(), 4096 / 8);
    assert_eq!(chunk.get(10), Some(&1));
    assert_eq!(chunk.get(11), Some(&0));
    assert_eq!(chunk.get(4096), None);
    assert!(!chunk.is_uniform());

    chunk.set(11, 2);
    assert_eq!(chunk.index_bits(), 2);
    chunk.set(12, 3);
    chunk.set(13, 4);
    assert_eq!(chunk.index_bits(), 4, "five values need four bits each");
    assert_eq!(chunk.distinct_len(), 5);
    assert_eq!(chunk.get(10), Some(&1), "widening keeps every value");
    assert_eq!(chunk.get(13), Some(&4));
    assert_eq!(chunk.count_of(&0), 4092);

    // Against one word per position.
    assert!(chunk.packed_bytes() < 4096 * size_of::<u16>());

    chunk.fill(0, 4096, 7);
    assert!(chunk.is_uniform());
    chunk.compact();
    assert_eq!(chunk.distinct_len(), 1);
    assert_eq!(chunk.index_bits(), 0);
    assert!(chunk.iter().all(|value| *value == 7));

    let built: Palette<u8> = [1u8, 1, 2, 1].into_iter().collect();
    assert_eq!(built.iter().copied().collect::<Vec<u8>>(), [1, 1, 2, 1]);
    assert_eq!(built.distinct_len(), 2);
}

#[test]
fn morton_keys_interleave_and_walk_the_octree() {
    let key = MortonKey::new(1, 2, 3).unwrap();
    assert_eq!(key.coordinates().to_array(), [1, 2, 3]);
    assert_eq!(MortonKey::ORIGIN.coordinates().to_array(), [0, 0, 0]);
    assert_eq!(
        MortonKey::new(5, 7, 9).unwrap().coordinates().to_array(),
        [5, 7, 9]
    );
    assert_eq!(MortonKey::new(1 << 21, 0, 0), None);

    // Ordering follows the curve, so a sorted walk stays local.
    assert!(MortonKey::new(0, 0, 0).unwrap() < MortonKey::new(1, 0, 0).unwrap());
    assert!(MortonKey::new(1, 1, 1).unwrap() < MortonKey::new(2, 0, 0).unwrap());

    // The eight children of a node are consecutive, and each is inside it.
    let parent = MortonKey::new(4, 4, 4).unwrap();
    let children = parent.children().unwrap();
    assert_eq!(children[0].parent(), parent);
    assert_eq!(children[7].child_index(), 7);
    assert!(children.iter().all(|child| child.is_inside(parent, 1)));
    assert_eq!(children[3].ancestor(1), parent);
    // Child 3 is binary 011: offset on x and y, not on z.
    assert_eq!(children[3].coordinates().to_array(), [9, 9, 8]);
    assert_eq!(children[4].coordinates().to_array(), [8, 8, 9]);

    let mut values: Vec<u64> = children.iter().map(|child| child.value()).collect();
    values.sort_unstable();
    assert!(values.windows(2).all(|pair| pair[1] == pair[0] + 1));

    // A key that already fills the width has no level below it.
    let deepest = MortonKey::new((1 << 21) - 1, (1 << 21) - 1, (1 << 21) - 1).unwrap();
    assert_eq!(deepest.children(), None);
    assert_eq!(deepest.ancestor(21), MortonKey::ORIGIN);
}

#[test]
fn the_new_traits_hold_across_the_collections() {
    // Weighted choice is now a contract rather than a per-type method.
    let mut loot: WeightedSet<&str> = WeightedSet::new();
    loot.set_weight("common", 90.0);
    loot.set_weight("rare", 10.0);
    assert!((loot.weighted_chance_of(&"common").value() - 0.9).abs() < 1e-9);

    let mut counts: MultiSet<&str> = MultiSet::new();
    counts.insert_times("stone", 3);
    counts.insert_times("ore", 1);
    assert!((counts.weighted_chance_of(&"stone").value() - 0.75).abs() < 1e-9);
    assert_eq!(counts.weighted_chance_of(&"absent").value(), 0.0);

    let mut biome: FuzzySet<&str> = FuzzySet::new();
    biome.set_membership("warm", Probability::new(0.5).unwrap());
    biome.set_membership("wet", Probability::new(0.5).unwrap());
    assert!((biome.weighted_chance_of(&"warm").value() - 0.5).abs() < 1e-9);

    let mut random = rng(3);
    let picks: Vec<&&str> = loot.choose_multiple_weighted(&mut random, 2);
    assert_eq!(picks.len(), 2);

    // Reservable allocation and hard limits are separate contracts.
    let mut set: Set<u32> = Capacity::with_capacity(64);
    assert!(set.capacity() >= 64);
    set.insert(1);
    assert!(set.spare_capacity() >= 1);
    set.shrink_to_fit();

    let mut ordered: OrderedSet<u32> = Capacity::with_capacity(10);
    assert!(ordered.capacity() >= 10);
    ordered.reserve(100);
    assert!(ordered.capacity() >= 100);

    let cache: LruCache<u8, u8> = LruCache::new(4);
    assert_eq!(Bounded::limit(&cache), 4);
    assert_eq!(Bounded::bounded_len(&cache), 0);

    // Content hashing: order matters for a sequence, not for a set.
    let first: Set<u32> = [1u32, 2, 3].into_iter().collect();
    let second: Set<u32> = [3u32, 2, 1].into_iter().collect();
    assert_eq!(first.content_hash(), second.content_hash());
    assert!(first.content_matches(&second));
    assert_ne!(
        vec![1u32, 2, 3].content_hash(),
        vec![3u32, 2, 1].content_hash()
    );
    assert_ne!(first.content_hash(), Set::<u32>::new().content_hash());

    let bits: BitSet = [1usize, 5].into_iter().collect();
    assert_eq!(
        bits.content_hash(),
        BitSet::from_iter([5usize, 1]).content_hash()
    );

    // The entry API inserts once and reads back in the same call.
    let mut grouped: voxel_world::structures::mappings::MultiMap<&str, u8> =
        voxel_world::structures::mappings::MultiMap::new();
    grouped.insert_pair("ores", 1);
    assert_eq!(grouped.get(&"ores").map(|values| values.len()), Some(1));
}

#[test]
fn owned_iteration_and_display_are_available() {
    let counts: MultiSet<&str> = [("stone", 2usize), ("ore", 1)].into_iter().collect();
    let mut drained: Vec<(&str, usize)> = counts.into_iter().collect();
    drained.sort();
    assert_eq!(drained, [("ore", 1), ("stone", 2)]);

    let weights: WeightedSet<&str> = [("a", 1.0f64)].into_iter().collect();
    assert_eq!(
        weights.into_iter().collect::<Vec<(&str, f64)>>(),
        [("a", 1.0)]
    );

    let sparse: voxel_world::structures::collections::SparseSequence<u8> =
        [(0i64, 1u8), (5, 2)].into_iter().collect();
    assert_eq!(
        sparse.into_iter().collect::<Vec<(i64, u8)>>(),
        [(0, 1), (5, 2)]
    );

    let buffer: voxel_world::structures::collections::RingBuffer<u8> =
        [1u8, 2, 3].into_iter().collect();
    assert_eq!(buffer.len(), 3);

    let set: Set<&str> = ["only"].into_iter().collect();
    assert_eq!(set.to_string(), "{only}");
    let ordered: OrderedSet<u8> = [1u8, 2].into_iter().collect();
    assert_eq!(ordered.to_string(), "[1, 2]");
    let column: RunLengthSequence<u8> = [1u8, 1, 2].into_iter().collect();
    assert_eq!(column.to_string(), "[0..2 = 1, 2..3 = 2]");
}

#[test]
fn logical_identity_ignores_internal_slack_and_preserves_partitions() {
    let mut grown = BitSet::new();
    grown.insert(4096);
    grown.remove(4096);
    let empty = BitSet::new();
    assert_eq!(grown, empty);
    assert_eq!(grown.content_hash(), empty.content_hash());

    let mut joined: DisjointSets<u32> = DisjointSets::new();
    joined.join(1, 2);
    joined.join(3, 4);
    let representative = *Grouping::labels(&joined).next().unwrap();
    assert_eq!(Grouping::group(&joined, &representative).unwrap().len(), 2);

    let mut separate: DisjointSets<u32> = DisjointSets::new();
    separate.insert(1);
    separate.insert(2);
    separate.insert(3);
    separate.insert(4);
    assert_ne!(joined.content_hash(), separate.content_hash());
}

#[test]
fn range_queries_use_their_ordered_indices_and_respect_edge_bounds() {
    let mut schedule = Scheduler::new();
    schedule.schedule_at(5, "a");
    schedule.schedule_at(8, "b");
    schedule.schedule_at(8, "c");
    schedule.schedule_at(13, "d");
    assert_eq!(
        RangeQuery::range(&schedule, 6..=8).collect::<Vec<_>>(),
        [(8, &"b"), (8, &"c")]
    );

    let runs: RunLengthSequence<u8> = [1, 1, 2, 2, 3].into_iter().collect();
    assert_eq!(RangeQuery::range(&runs, ..=usize::MAX).count(), 3);
    assert_eq!(
        RangeQuery::range(
            &runs,
            (
                std::ops::Bound::Excluded(usize::MAX),
                std::ops::Bound::Unbounded
            )
        )
        .count(),
        0
    );

    let mut spans = IntervalMap::new();
    spans.insert(0, 10, 'a');
    spans.insert(10, 20, 'b');
    assert_eq!(RangeQuery::range(&spans, ..10).count(), 1);
    assert_eq!(RangeQuery::range(&spans, ..=10).count(), 2);
}

#[test]
fn generic_storage_and_bounded_work_structures_compose() {
    let mut image = Grid2::filled([3, 2], 0u8).unwrap();
    *image.get_mut([2, 1]).unwrap() = 5;
    assert_eq!(image.index_of([2, 1]), Some(5));
    assert_eq!(image.get([2, 1]), Some(&5));

    let mut grid = Grid3::filled([2, 3, 4], 0u8).unwrap();
    *grid.get_mut([1, 2, 3]).unwrap() = 9;
    assert_eq!(grid.get([1, 2, 3]), Some(&9));
    assert_eq!(grid.index_of([2, 0, 0]), None);

    let mut mask = BitGrid3::new([2, 3, 4]).unwrap();
    assert_eq!(mask.set([1, 2, 3], true), Some(false));
    assert_eq!(mask.get([1, 2, 3]), Some(true));
    assert_eq!(mask.count_ones(), 1);

    let mut timeline = Grid4::filled([2, 3, 4, 5], 0u8).unwrap();
    *timeline.get_mut([1, 2, 3, 4]).unwrap() = 11;
    assert_eq!(timeline.index_of([1, 2, 3, 4]), Some(119));
    assert_eq!(timeline.get([1, 2, 3, 4]), Some(&11));

    let mut image_mask = BitGrid2::new([3, 2]).unwrap();
    let mut timeline_mask = BitGrid4::new([2, 3, 4, 5]).unwrap();
    assert_eq!(image_mask.set([2, 1], true), Some(false));
    assert_eq!(timeline_mask.set([1, 2, 3, 4], true), Some(false));
    assert_eq!(image_mask.index_of([2, 1]), Some(5));
    assert_eq!(timeline_mask.index_of([1, 2, 3, 4]), Some(119));

    let uniform = CompactSequence::filled(100, 7u64);
    assert_eq!(uniform.kind(), CompactKind::Uniform);
    let runs: CompactSequence<u64> = (0..1000).map(|i| if i < 500 { 1 } else { 2 }).collect();
    assert_eq!(runs.kind(), CompactKind::Runs);
    let palette: CompactSequence<u8> = (0..100).map(|i| (i % 2) as u8).collect();
    assert_eq!(palette.kind(), CompactKind::Palette);
    let dense: CompactSequence<u64> = (0..100).collect();
    assert_eq!(dense.kind(), CompactKind::Dense);

    let mut recent = BoundedOrderedSet::new(2);
    assert_eq!(recent.insert("a"), None);
    assert_eq!(recent.insert("b"), None);
    assert_eq!(recent.insert("c"), Some("a"));
    assert_eq!(recent.iter().copied().collect::<Vec<_>>(), ["b", "c"]);

    let mut unique = UniqueScheduler::new();
    assert_eq!(unique.schedule_at(10, "job"), None);
    assert_eq!(unique.schedule_at(4, "job"), Some(10));
    assert_eq!(unique.len(), 1);
    assert_eq!(unique.advance_by(4), ["job"]);
}

#[test]
fn capability_traits_join_related_implementations() {
    let mut tree: PriorityQueue<&str, i32> = PriorityQueue::new();
    assert!(PriorityQueueLike::push_priority(&mut tree, 3, "tree"));
    assert_eq!(
        PriorityQueueLike::peek_priority_item(&tree),
        Some((3, &"tree"))
    );
    assert_eq!(
        PriorityQueueLike::pop_priority(&mut tree),
        Some((3, "tree"))
    );

    let mut buckets = BucketQueue::new(4);
    assert!(PriorityQueueLike::push_priority(&mut buckets, 2, "bucket"));
    assert_eq!(
        PriorityQueueLike::pop_priority(&mut buckets),
        Some((2, "bucket"))
    );

    let mut slots: SlotMap<TestEntityKind, u8> = SlotMap::new();
    let handle = HandleStore::insert(&mut slots, 7);
    assert_eq!(HandleStore::get(&slots, handle), Some(&7));
    assert_eq!(HandleStore::remove(&mut slots, handle), Some(7));
}
