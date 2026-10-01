//! The rota family: even division into a fixed number of groups.

use voxel_world::random::Random;
use voxel_world::random::seed::Seed;
use voxel_world::structures::collections::{
    BucketQueue, MultiRota, OrderedMultiRota, OrderedRota, Rota,
};
use voxel_world::structures::traits::{
    BalancedPartition, Capacity, Choose, Collection, CollectionInsert, CollectionRemove,
    ContentHashable, Measured, MeasuredMut, Partitioned, SetAlgebra, WeightedChoose,
};

/// Asserts the promise every rota makes: no two groups differ by more than one.
fn assert_balanced<P: BalancedPartition>(partition: &P, total: usize) {
    let sizes: Vec<usize> = partition.group_sizes().collect();
    let smallest = *sizes.iter().min().unwrap();
    let largest = *sizes.iter().max().unwrap();

    assert!(
        largest - smallest <= 1,
        "groups {sizes:?} differ by more than one"
    );
    assert!(partition.is_balanced());
    assert_eq!(sizes.iter().sum::<usize>(), total, "shares must add up");
    assert_eq!(partition.spread(), largest - smallest);
}

#[test]
fn a_rota_spreads_and_stays_spread_through_any_sequence_of_changes() {
    let mut rota: Rota<u32, 4> = Rota::new();

    for value in 0..13 {
        assert!(rota.insert(value));
        assert_balanced(&rota, rota.len());
    }
    assert_eq!(rota.len(), 13);
    assert_eq!(rota.group_sizes(), &[4, 3, 3, 3]);
    assert!(!rota.insert(0), "already held");

    // Removing from the smallest group pulls one across from the largest.
    for value in [0, 5, 9, 12, 3] {
        assert!(rota.remove(&value));
        assert_balanced(&rota, rota.len());
    }
    assert!(!rota.remove(&0));
    assert_eq!(rota.len(), 8);
    assert_eq!(rota.group_sizes(), &[2, 2, 2, 2]);

    // Every element is in exactly one group, and the groups cover all of them.
    let mut seen: Vec<u32> = rota.groups().flatten().copied().collect();
    seen.sort_unstable();
    let mut held: Vec<u32> = rota.iter().copied().collect();
    held.sort_unstable();
    assert_eq!(seen, held);
    for value in &held {
        let group = rota.group_of(value).unwrap();
        assert!(rota.group(group).contains(value));
    }
}

#[test]
fn a_rota_covers_everything_once_per_round() {
    // The point of the structure: one group per turn visits every element
    // exactly once per GROUPS turns, whatever the population size.
    let mut rota: Rota<u32, 8> = (0..100).collect();
    rota.remove(&17);
    rota.insert(200);

    let mut visited: Vec<u32> = Vec::new();
    for turn in 0..8 {
        visited.extend(rota.group(turn).iter().copied());
    }
    visited.sort_unstable();

    let mut expected: Vec<u32> = rota.iter().copied().collect();
    expected.sort_unstable();
    assert_eq!(visited, expected);
    assert_eq!(visited.len(), 100);

    // And no turn is more than one element heavier than another.
    assert_balanced(&rota, 100);
    assert!(
        rota.group_sizes()
            .iter()
            .all(|size| (12..=13).contains(size))
    );
}

#[test]
fn rotas_report_their_division_through_the_partition_trait() {
    let rota: Rota<u32, 3> = (0..7).collect();

    assert_eq!(Partitioned::group_count(&rota), 3);
    assert_eq!(rota.group_sizes().to_vec(), vec![3, 2, 2]);
    assert_eq!(Partitioned::group_len(&rota, 0), 3);
    assert_eq!(Partitioned::group_len(&rota, 99), 0, "no such group");
    assert_eq!(rota.group_members(0).count(), 3);
    assert_eq!(rota.group_members(99).count(), 0);
    assert_eq!(rota.fullest_group(), Some(0));
    assert_eq!(rota.emptiest_group(), Some(1));
    assert_eq!(Partitioned::spread(&rota), 1);
    assert!(!rota.is_group_empty(0));

    // A bucket queue is partitioned too, but deliberately unbalanced.
    let mut queue: BucketQueue<&str> = BucketQueue::new(4);
    queue.push(0, "bright");
    queue.push(0, "also");
    queue.push(3, "dim");
    assert_eq!(Partitioned::group_count(&queue), 4);
    assert_eq!(Partitioned::group_len(&queue, 0), 2);
    assert_eq!(queue.group_members(3).collect::<Vec<&&str>>(), [&"dim"]);
    assert!(!queue.is_balanced(), "priorities are not shares");
}

#[test]
fn a_multi_rota_gives_every_copy_its_own_turn() {
    let mut rota: MultiRota<&str, 3> = MultiRota::new();

    rota.insert_times("tree", 5);
    rota.insert("rock");
    assert_eq!(rota.len(), 6);
    assert_eq!(rota.distinct_len(), 2);
    assert_eq!(rota.count_of(&"tree"), 5);
    assert_balanced(&rota, 6);

    // Copies of one value are spread rather than piled together.
    assert!(rota.groups_of(&"tree").len() == 5);
    assert!(rota.group_sizes().iter().all(|size| *size == 2));

    assert!(rota.remove_one(&"tree"));
    assert_eq!(rota.count_of(&"tree"), 4);
    assert_balanced(&rota, 5);
    assert_eq!(rota.remove_all(&"tree"), 4);
    assert_eq!(rota.len(), 1);
    assert!(!rota.contains(&"tree"));
    assert_balanced(&rota, 1);

    // Measured, as a multiset is.
    rota.set_measure("rock", 4);
    assert_eq!(rota.measure_of(&"rock"), 4);
    assert_eq!(rota.total_measure(), 4);
    rota.subtract_measure(&"rock", 2);
    assert_eq!(rota.count_of(&"rock"), 2);
    assert!((rota.weighted_chance_of(&"rock").value() - 1.0).abs() < 1e-9);
    assert_balanced(&rota, rota.len());
}

#[test]
fn an_ordered_rota_keeps_both_orders_in_agreement() {
    let mut rota: OrderedRota<u32, 3> = OrderedRota::new();

    for value in 0..10 {
        rota.insert(value);
    }
    assert_balanced(&rota, 10);

    // The whole reads back in insertion order.
    assert_eq!(
        rota.iter().copied().collect::<Vec<u32>>(),
        (0..10).collect::<Vec<u32>>()
    );
    assert_eq!(rota.first(), Some(&0));
    assert_eq!(rota.last(), Some(&9));

    // Each group is a subsequence of that order.
    for group in 0..3 {
        let members: Vec<u32> = rota.group(group).copied().collect();
        assert!(
            members.windows(2).all(|pair| pair[0] < pair[1]),
            "{members:?}"
        );
        assert!(
            members
                .iter()
                .all(|value| rota.group_of(value) == Some(group))
        );
    }

    // Removing keeps the order of everything else, and the balance.
    assert!(rota.remove(&0));
    assert!(rota.remove(&5));
    assert!(!rota.remove(&5));
    assert_balanced(&rota, 8);
    assert_eq!(
        rota.iter().copied().collect::<Vec<u32>>(),
        [1, 2, 3, 4, 6, 7, 8, 9]
    );

    // A late arrival goes to the end of the order, not the middle.
    rota.insert(100);
    assert_eq!(rota.last(), Some(&100));
    assert_balanced(&rota, 9);

    // Reading every group in turn covers the rota exactly once.
    let mut covered: Vec<u32> = (0..3)
        .flat_map(|group| rota.group(group).copied())
        .collect();
    covered.sort_unstable();
    assert_eq!(covered, [1, 2, 3, 4, 6, 7, 8, 9, 100]);
}

#[test]
fn an_ordered_multi_rota_queues_repeats_in_order() {
    let mut rota: OrderedMultiRota<&str, 2> = OrderedMultiRota::new();

    rota.insert("a");
    rota.insert("b");
    rota.insert("a");
    rota.insert("c");
    assert_eq!(rota.len(), 4);
    assert_eq!(rota.distinct_len(), 3);
    assert_eq!(rota.count_of(&"a"), 2);
    assert_eq!(
        rota.iter().copied().collect::<Vec<&str>>(),
        ["a", "b", "a", "c"]
    );
    assert_balanced(&rota, 4);

    // Removing takes the earliest copy, so the later one keeps its place.
    assert!(rota.remove_one(&"a"));
    assert_eq!(rota.iter().copied().collect::<Vec<&str>>(), ["b", "a", "c"]);
    assert_eq!(rota.count_of(&"a"), 1);
    assert_balanced(&rota, 3);

    assert_eq!(rota.remove_all(&"a"), 1);
    assert_eq!(rota.remove_all(&"missing"), 0);
    assert_eq!(rota.iter().copied().collect::<Vec<&str>>(), ["b", "c"]);
    assert_balanced(&rota, 2);
}

#[test]
fn rotas_carry_the_shared_collection_vocabulary() {
    let mut rota: Rota<u32, 4> = Rota::new();

    // Collection, CollectionInsert, CollectionRemove and the blanket mutable
    // and random-choice traits.
    assert!(CollectionInsert::insert(&mut rota, 1));
    assert!(CollectionInsert::insert(&mut rota, 2));
    assert_eq!(Collection::len(&rota), 2);
    assert!(Collection::contains(&rota, &1));
    assert_eq!(rota.elements().count(), 2);

    let mut random = Random::new(Seed::from_raw(9));
    assert!(rota.choose(&mut random).is_some());
    assert!(Rota::<u32, 4>::new().choose(&mut random).is_none());

    CollectionRemove::retain(&mut rota, |value| *value != 1);
    assert_eq!(rota.len(), 1);
    assert_balanced(&rota, 1);
    CollectionRemove::clear(&mut rota);
    assert!(rota.is_empty());

    // Capacity, on the list-backed variants.
    let mut sized: Rota<u32, 4> = Capacity::with_capacity(64);
    assert!(sized.capacity() >= 64);
    sized.reserve(128);
    assert!(sized.capacity() >= 128);
    sized.insert(1);
    sized.shrink_to_fit();

    // Set algebra over the elements, ignoring how each rota divided them.
    let left: Rota<u32, 4> = (0..6).collect();
    let right: Rota<u32, 4> = (4..10).collect();
    assert_eq!(left.union(&right).len(), 10);
    assert_eq!(left.intersection(&right).len(), 2);
    assert_eq!(left.difference(&right).len(), 4);
    assert!(Rota::<u32, 4>::from_iter([0, 1]).is_subset(&left));
    assert_balanced(&left.union(&right), 10);

    // Content hashing covers the division as well as the elements.
    let same: Rota<u32, 4> = (0..6).collect();
    assert_eq!(left.content_hash(), same.content_hash());
    assert_ne!(left.content_hash(), right.content_hash());

    // Iteration by value and by reference.
    let owned: Vec<u32> = (0..6).collect::<Rota<u32, 4>>().into_iter().collect();
    assert_eq!(owned.len(), 6);
    assert_eq!((&left).into_iter().count(), 6);
    assert_eq!(left.to_string(), "6 over [2, 2, 1, 1]");
}

#[test]
fn a_single_group_rota_is_a_plain_set() {
    let mut rota: Rota<u32, 1> = Rota::new();

    rota.insert(1);
    rota.insert(2);
    assert_eq!(rota.group_sizes(), &[2]);
    assert_eq!(rota.group(0).len(), 2);
    assert_balanced(&rota, 2);
    assert!(rota.remove(&1));
    assert_balanced(&rota, 1);

    // And an empty rota is balanced by definition.
    let empty: Rota<u32, 5> = Rota::new();
    assert_balanced(&empty, 0);
    assert_eq!(empty.fullest_group(), Some(0));
}
