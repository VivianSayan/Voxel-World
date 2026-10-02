//! The [`Shuffle`] trait, and shuffling from a seed rather than a stream.
//!
//! # What changed, and why these tests exist
//!
//! Shuffling and random selection used to be available only from a [`Random`] stream.
//! That left the one case a generated world most needs unreachable: a [`Seed`] is the
//! thing that gives the *same* answer every time, and it could not shuffle anything.
//!
//! So the tests here come in pairs. One half checks that a shuffle is a real shuffle —
//! every ordering, nothing lost. The other half checks that a seed's shuffle is the
//! same arrangement every time it is asked, which is the property a replayable world
//! rests on.

use std::collections::{HashMap, HashSet, VecDeque};
use voxel_world::random::Random;
use voxel_world::random::source::StochasticSource;
use voxel_world::random::seed::Seed;
use voxel_world::structures::collections::sequences::labeled_ordered_set::LabeledOrderedSet;
use voxel_world::structures::collections::sequences::ordered_set::OrderedSet;
use voxel_world::structures::collections::sequences::ring_buffer::RingBuffer;
use voxel_world::structures::collections::sparse::sparse_sequence::SparseSequence;
use voxel_world::structures::traits::{Choose, Shuffle};

fn stream(seed: u64) -> Random {
    Random::new(Seed::from_integer(seed))
}

// ---------------------------------------------------------------------------
// A shuffle is a permutation
// ---------------------------------------------------------------------------

#[test]
fn shuffling_keeps_every_element() {
    let mut random = stream(1);

    for _ in 0..500 {
        let mut items: Vec<u32> = (0..40).collect();
        items.shuffle(&mut random);

        let mut sorted = items.clone();
        sorted.sort_unstable();

        assert_eq!(sorted, (0..40).collect::<Vec<u32>>(), "an element was lost");
    }
}

#[test]
fn every_ordering_turns_up() {
    // Four elements have 24 orderings. A correct shuffle reaches all of them; one with
    // an off-by-one in its index range systematically misses some.
    let mut random = stream(2);
    let mut seen: HashMap<Vec<u8>, u32> = HashMap::new();

    for _ in 0..48_000 {
        let mut items: Vec<u8> = vec![0, 1, 2, 3];
        items.shuffle(&mut random);
        *seen.entry(items).or_default() += 1;
    }

    assert_eq!(seen.len(), 24, "only {} of 24 orderings appeared", seen.len());

    // And roughly evenly: 2000 expected each, so five standard deviations is generous.
    let expected: f64 = 48_000.0 / 24.0;
    let allowed: f64 = 5.0 * expected.sqrt();

    for (ordering, count) in &seen {
        assert!(
            (f64::from(*count) - expected).abs() < allowed,
            "{ordering:?} appeared {count} times, expected about {expected}"
        );
    }
}

#[test]
fn an_empty_or_single_slice_shuffles_without_complaint() {
    let mut random = stream(3);

    let mut nothing: Vec<u32> = Vec::new();
    nothing.shuffle(&mut random);
    assert!(nothing.is_empty());

    let mut one = [7u32];
    one.shuffle(&mut random);
    assert_eq!(one, [7]);
}

#[test]
fn slices_and_arrays_shuffle_directly() {
    // `[T]` is what implements the trait, so anything that derefs or coerces to a
    // slice is covered without an impl of its own.
    let mut random = stream(4);

    let mut array: [u32; 16] = std::array::from_fn(|index| index as u32);
    array.shuffle(&mut random);

    let mut owned: Vec<u32> = (0..16).collect();
    owned.shuffle(&mut random);

    let slice: &mut [u32] = &mut owned[4..12];
    slice.shuffle(&mut random);

    assert_eq!(array.iter().sum::<u32>(), 120);
    assert_eq!(owned.iter().sum::<u32>(), 120);
}

// ---------------------------------------------------------------------------
// From a seed
// ---------------------------------------------------------------------------

#[test]
fn a_seed_gives_one_ordering_for_ever() {
    let chest = Seed::from_integer(5u64).child("chest").index(17);

    let first = chest.shuffled((0..32u32).collect::<Vec<u32>>());

    for _ in 0..20 {
        assert_eq!(
            chest.shuffled((0..32u32).collect::<Vec<u32>>()),
            first,
            "a seed changed its mind about an ordering"
        );
    }
}

#[test]
fn different_seeds_give_different_orderings() {
    let world = Seed::from_integer(6u64).child("chests");

    let orderings: HashSet<Vec<u32>> = (0..200u64)
        .map(|index| world.index(index).shuffled((0..12u32).collect::<Vec<u32>>()))
        .collect();

    assert!(
        orderings.len() > 190,
        "only {} distinct orderings from 200 seeds",
        orderings.len()
    );
}

#[test]
fn a_seeds_shuffle_matches_its_cursor() {
    // `Seed::shuffle` is a shorthand, not a second algorithm: it must agree with
    // shuffling through the seed's own cursor.
    let seed = Seed::from_integer(7u64).child("deck");

    let mut viaseed: Vec<u32> = (0..52).collect();
    seed.shuffle(&mut viaseed);

    let mut via_cursor: Vec<u32> = (0..52).collect();
    via_cursor.shuffle(&mut seed.cursor());

    assert_eq!(viaseed, via_cursor);
}

#[test]
fn a_seeds_shuffle_is_a_real_shuffle() {
    // Every ordering of three elements, drawn from many seeds rather than many draws.
    let world = Seed::from_integer(8u64);
    let mut seen: HashSet<Vec<u8>> = HashSet::new();

    for index in 0..2_000u64 {
        seen.insert(world.index(index).shuffled(vec![0u8, 1, 2]));
    }

    assert_eq!(seen.len(), 6, "a seed-driven shuffle missed an ordering");
}

// ---------------------------------------------------------------------------
// Partial shuffles
// ---------------------------------------------------------------------------

#[test]
fn a_partial_shuffle_fills_what_it_promises() {
    let mut random = stream(9);
    let mut items: Vec<u32> = (0..1_000).collect();

    assert_eq!(items.partial_shuffle(3, &mut random), 3);

    // Asking for more than there is fills everything and says so.
    let mut few: Vec<u32> = (0..4).collect();
    assert_eq!(few.partial_shuffle(10, &mut random), 4);
}

#[test]
fn a_partial_shuffle_chooses_uniformly() {
    // The first `count` places must be a uniform selection, which is the whole point
    // of paying for only `count` draws.
    const DRAWS: u32 = 60_000;
    let mut random = stream(10);
    let mut counts = [0u32; 6];

    for _ in 0..DRAWS {
        let mut items: Vec<usize> = (0..6).collect();
        items.partial_shuffle(2, &mut random);

        counts[items[0]] += 1;
        counts[items[1]] += 1;
    }

    // Each of the six is chosen in two of six places, so twice DRAWS/6 each.
    let expected: f64 = 2.0 * f64::from(DRAWS) / 6.0;
    let allowed: f64 = 5.0 * expected.sqrt();

    for (item, count) in counts.iter().enumerate() {
        assert!(
            (f64::from(*count) - expected).abs() < allowed,
            "item {item} chosen {count} times, expected about {expected}"
        );
    }
}

#[test]
fn a_seeds_partial_shuffle_is_stable() {
    let region = Seed::from_integer(11u64).child("spawns");

    let mut items: Vec<u32> = (0..1_000).collect();
    assert_eq!(region.partial_shuffle(&mut items, 3), 3);

    let chosen: Vec<u32> = items[..3].to_vec();

    let mut again: Vec<u32> = (0..1_000).collect();
    region.partial_shuffle(&mut again, 3);

    assert_eq!(chosen, again[..3]);
}

// ---------------------------------------------------------------------------
// The collections that implement it
// ---------------------------------------------------------------------------

#[test]
fn a_deque_shuffles_across_both_halves() {
    // A deque is a ring, so a shuffle that forgot to make it contiguous would only
    // ever move elements within one half. Pushing from both ends guarantees two.
    let mut deque: VecDeque<u32> = VecDeque::new();

    for value in 0..10 {
        deque.push_back(value);
        deque.push_front(value + 100);
    }

    let before: Vec<u32> = deque.iter().copied().collect();
    let mut random = stream(12);

    let mut moved = false;

    for _ in 0..20 {
        deque.shuffle(&mut random);

        if deque.iter().copied().collect::<Vec<u32>>() != before {
            moved = true;
        }

        let mut sorted: Vec<u32> = deque.iter().copied().collect();
        sorted.sort_unstable();
        let mut expected = before.clone();
        expected.sort_unstable();

        assert_eq!(sorted, expected, "a deque lost an element");
    }

    assert!(moved, "the deque never changed order");
}

#[test]
fn a_ring_buffer_shuffles() {
    let mut buffer: RingBuffer<u32> = RingBuffer::new(8);

    for value in 0..12 {
        buffer.push_back(value);
    }

    let before: Vec<u32> = buffer.iter().copied().collect();
    let mut random = stream(13);

    buffer.shuffle(&mut random);

    let mut sorted: Vec<u32> = buffer.iter().copied().collect();
    sorted.sort_unstable();
    let mut expected = before.clone();
    expected.sort_unstable();

    assert_eq!(sorted, expected, "the buffer lost a value");
    assert_eq!(buffer.len(), before.len());
}

#[test]
fn an_ordered_set_shuffles_and_stays_findable() {
    // The set keeps an index from member to position, so a shuffle has to rebuild it.
    // If it did not, the set would still hold everything and be unable to find any
    // of it — which `contains` is here to catch.
    let mut set: OrderedSet<u32> = OrderedSet::new();

    for value in 0..50 {
        set.push(value);
    }

    let mut random = stream(14);
    set.shuffle(&mut random);

    assert_eq!(set.len(), 50);

    for value in 0..50 {
        assert!(set.contains(&value), "{value} became unreachable after shuffling");
    }
}

#[test]
fn a_labeled_ordered_set_keeps_labels_with_their_elements() {
    let mut set: LabeledOrderedSet<u32, u32> = LabeledOrderedSet::new();

    for value in 0..30 {
        set.push(value, value * 10);
    }

    let mut random = stream(15);
    set.shuffle(&mut random);

    assert_eq!(set.len(), 30);

    // A label belongs to its element, not to the position it was in.
    for value in 0..30 {
        assert_eq!(set.label_of(&value), Some(&(value * 10)), "label lost for {value}");
        assert_eq!(set.get_by_label(&(value * 10)), Some(&value), "reverse lookup lost");
    }
}

#[test]
fn a_sparse_sequence_shuffles() {
    let mut sequence: SparseSequence<u32> = SparseSequence::new();

    for value in 0..40 {
        sequence.push(value);
    }

    let mut random = stream(16);
    sequence.shuffle(&mut random);

    assert_eq!(sequence.len(), 40);
}

// ---------------------------------------------------------------------------
// The selection traits, now reachable from a seed
// ---------------------------------------------------------------------------

#[test]
fn choosing_works_from_a_seeds_cursor() {
    // `Choose` was tied to `Random`, so this could not be written at all before.
    let items: Vec<u32> = (0..20).collect();
    let world = Seed::from_integer(17u64);

    for index in 0..100u64 {
        let seed = world.index(index);

        let one = *items.choose(&mut seed.cursor()).expect("non-empty");
        let again = *items.choose(&mut seed.cursor()).expect("non-empty");

        assert_eq!(one, again, "a seed chose differently the second time");
        assert!(items.contains(&one));
    }

    // And across seeds it really does vary.
    let chosen: HashSet<u32> = (0..200u64)
        .map(|index| *items.choose(&mut world.index(index).cursor()).unwrap())
        .collect();

    assert!(chosen.len() > 15, "only {} of 20 items ever chosen", chosen.len());
}

#[test]
fn choosing_several_works_from_a_seeds_cursor() {
    let items: Vec<u32> = (0..50).collect();
    let seed = Seed::from_integer(18u64).child("loot");

    let picked: Vec<u32> = items
        .choose_multiple(&mut seed.cursor(), 5)
        .into_iter()
        .copied()
        .collect();

    assert_eq!(picked.len(), 5);
    assert_eq!(
        picked.iter().collect::<HashSet<&u32>>().len(),
        5,
        "choose_multiple repeated an element"
    );

    let again: Vec<u32> = items
        .choose_multiple(&mut seed.cursor(), 5)
        .into_iter()
        .copied()
        .collect();

    assert_eq!(picked, again, "not stable for one seed");
}

// ---------------------------------------------------------------------------
// Picking one element
// ---------------------------------------------------------------------------

#[test]
fn picking_reports_an_empty_slice_rather_than_panicking() {
    let mut random = stream(19);

    assert_eq!(random.pick::<u8>(&[]), None);
    assert_eq!(random.pick_mut::<u8>(&mut []), None);
    assert_eq!(Seed::from_integer(1u64).pick::<u8>(&[]), None);
}

#[test]
fn picking_reaches_every_element() {
    let items: Vec<u32> = (0..10).collect();
    let mut random = stream(20);

    let seen: HashSet<u32> = (0..2_000).map(|_| *random.pick(&items).unwrap()).collect();

    assert_eq!(seen.len(), 10, "pick never reached some elements");
}

#[test]
fn picking_mutably_changes_one_element() {
    let mut health = [10u32; 4];
    let mut random = stream(21);

    for _ in 0..6 {
        *random.pick_mut(&mut health).unwrap() -= 1;
    }

    assert_eq!(health.iter().sum::<u32>(), 34, "exactly six points removed");
}

// ---------------------------------------------------------------------------
// Streams and seeds are different questions
// ---------------------------------------------------------------------------

#[test]
fn a_stream_moves_on_where_a_seed_does_not() {
    // Shuffling twice from a stream gives two orderings; from a seed, one. Both are
    // correct, and the difference is the reason both exist.
    let mut random = stream(22);

    let mut first: Vec<u32> = (0..30).collect();
    let mut second: Vec<u32> = (0..30).collect();

    first.shuffle(&mut random);
    second.shuffle(&mut random);

    assert_ne!(first, second, "a stream repeated itself");

    let seed = Seed::from_integer(23u64);

    assert_eq!(
        seed.shuffled((0..30u32).collect::<Vec<u32>>()),
        seed.shuffled((0..30u32).collect::<Vec<u32>>()),
    );
}

// ===========================================================================
// The seed API knows only the `Shuffle` trait
//
// Each test below hands a different representation to the *same* `Seed::shuffle`
// call. Nothing in `Seed` distinguishes them: the collection decides what a random
// reordering means for it and keeps its own invariants while doing it.
// ===========================================================================

/// Builds an `OrderedSet` of `0..count`.
fn ordered_set(count: u32) -> OrderedSet<u32> {
    let mut set = OrderedSet::new();

    for value in 0..count {
        set.push(value);
    }

    set
}

/// Builds a `SparseSequence` of `0..count`.
fn sparse_sequence(count: u32) -> SparseSequence<u32> {
    let mut sequence = SparseSequence::new();

    for value in 0..count {
        sequence.push(value);
    }

    sequence
}

/// Builds a `RingBuffer` of `0..count`, filled to capacity.
fn ring_buffer(count: u32) -> RingBuffer<u32> {
    let mut buffer = RingBuffer::new(count as usize);

    for value in 0..count {
        buffer.push_back(value);
    }

    buffer
}

/// Builds a `LabeledOrderedSet` of `0..count`, each labelled ten times its value.
fn labeled(count: u32) -> LabeledOrderedSet<u32, u32> {
    let mut set = LabeledOrderedSet::new();

    for value in 0..count {
        set.push(value, value * 10);
    }

    set
}

#[test]
fn a_seed_shuffles_an_array_and_a_vec() {
    let seed = Seed::from_integer(100u64);

    let mut array: [u32; 16] = std::array::from_fn(|index| index as u32);
    let mut again: [u32; 16] = std::array::from_fn(|index| index as u32);

    seed.shuffle(&mut array);
    seed.shuffle(&mut again);

    assert_eq!(array, again, "same seed, same order");

    let mut owned: Vec<u32> = (0..16).collect();
    seed.shuffle(&mut owned);

    assert_eq!(owned, array.to_vec(), "a Vec and an array agree");

    // A slice too, which is the unsized case the generic parameter has to accept.
    let mut backing: Vec<u32> = (0..16).collect();
    let slice: &mut [u32] = backing.as_mut_slice();
    seed.shuffle(slice);

    assert_eq!(backing, owned);
}

#[test]
fn a_seed_shuffles_an_ordered_set_and_leaves_it_valid() {
    let seed = Seed::from_integer(101u64);

    let mut set = ordered_set(60);
    seed.shuffle(&mut set);

    // 4: the collection's own invariant checker, which is what catches a reordering
    // that forgot to rebuild the member-to-position index.
    assert!(set.check_invariants(), "the set's invariants broke");

    // 3 and 5: every member still present exactly once, and still findable.
    assert_eq!(set.len(), 60);

    for value in 0..60u32 {
        assert!(set.contains(&value), "{value} went missing");
        assert_eq!(
            set.iter().nth(set.index_of(&value).expect("indexed")),
            Some(&value),
            "the index for {value} points at the wrong place",
        );
    }

    // 1: the same seed gives the same order.
    let mut again = ordered_set(60);
    seed.shuffle(&mut again);

    assert_eq!(
        set.iter().copied().collect::<Vec<u32>>(),
        again.iter().copied().collect::<Vec<u32>>(),
    );

    // 2: a different seed normally gives a different one.
    let mut other = ordered_set(60);
    Seed::from_integer(102u64).shuffle(&mut other);

    assert_ne!(
        set.iter().copied().collect::<Vec<u32>>(),
        other.iter().copied().collect::<Vec<u32>>(),
    );
}

#[test]
fn a_seed_shuffles_a_sparse_sequence_and_keeps_its_indices_occupied() {
    let seed = Seed::from_integer(103u64);

    let before = sparse_sequence(40);
    let occupied: Vec<i64> = before.iter().map(|(index, _)| index).collect();

    let mut sequence = sparse_sequence(40);
    seed.shuffle(&mut sequence);

    // The documented meaning: values are redistributed over the indices that were
    // already occupied, rather than compacted, renumbered or removed.
    assert_eq!(
        sequence.iter().map(|(index, _)| index).collect::<Vec<i64>>(),
        occupied,
        "the occupied indices changed",
    );

    // Every value still present exactly once, and the reverse lookup still works.
    let mut values: Vec<u32> = sequence.values().copied().collect();
    values.sort_unstable();

    assert_eq!(values, (0..40).collect::<Vec<u32>>());

    for value in 0..40u32 {
        assert!(sequence.contains(&value), "{value} became unreachable");
    }

    // And `get` agrees with what iteration reports, which it would not if the
    // element index had gone stale.
    for (index, value) in sequence.iter() {
        assert_eq!(sequence.get(index), Some(value));
    }

    let mut again = sparse_sequence(40);
    seed.shuffle(&mut again);

    assert_eq!(
        sequence.values().copied().collect::<Vec<u32>>(),
        again.values().copied().collect::<Vec<u32>>(),
    );
}

#[test]
fn a_seed_shuffles_a_ring_buffer() {
    let seed = Seed::from_integer(104u64);

    let mut buffer = ring_buffer(32);
    seed.shuffle(&mut buffer);

    assert_eq!(buffer.len(), 32);

    let mut values: Vec<u32> = buffer.iter().copied().collect();
    values.sort_unstable();

    assert_eq!(values, (0..32).collect::<Vec<u32>>());

    let mut again = ring_buffer(32);
    seed.shuffle(&mut again);

    assert_eq!(
        buffer.iter().copied().collect::<Vec<u32>>(),
        again.iter().copied().collect::<Vec<u32>>(),
    );

    let mut other = ring_buffer(32);
    Seed::from_integer(105u64).shuffle(&mut other);

    assert_ne!(
        buffer.iter().copied().collect::<Vec<u32>>(),
        other.iter().copied().collect::<Vec<u32>>(),
    );
}

#[test]
fn a_seed_shuffles_a_labeled_set_and_labels_follow_their_elements() {
    let seed = Seed::from_integer(106u64);

    let mut set = labeled(30);
    seed.shuffle(&mut set);

    assert_eq!(set.len(), 30);

    // A label belongs to its element, not to the position that element was in, so
    // every pairing must survive a reordering untouched.
    for value in 0..30u32 {
        assert_eq!(set.label_of(&value), Some(&(value * 10)), "label lost for {value}");
        assert_eq!(set.get_by_label(&(value * 10)), Some(&value), "reverse lookup lost");
    }

    // And the pairs that iteration reports are the same pairs, in the new order.
    for (element, label) in set.iter() {
        assert_eq!(*label, element * 10, "a label moved to another element");
    }

    let mut again = labeled(30);
    seed.shuffle(&mut again);

    assert_eq!(
        set.iter().map(|(e, _)| *e).collect::<Vec<u32>>(),
        again.iter().map(|(e, _)| *e).collect::<Vec<u32>>(),
    );
}

// ---------------------------------------------------------------------------
// The extension point
// ---------------------------------------------------------------------------

#[test]
fn a_collection_written_here_joins_in_by_implementing_the_trait() {
    // This is the architecture under test. `Seed` and `Random` were not changed to
    // know about this type; it takes part purely by implementing `Shuffle`, and it
    // reorders its own two parallel halves in step — something no generic slice
    // algorithm could have done for it.
    #[derive(Debug, PartialEq)]
    struct Paired {
        names: Vec<char>,
        counts: Vec<u32>,
    }

    impl Shuffle for Paired {
        fn shuffle<S: StochasticSource + ?Sized>(&mut self, source: &mut S) {
            // Shuffle an index list, then apply that one permutation to both halves,
            // which is what keeps a name with its own count.
            let mut order: Vec<usize> = (0..self.names.len()).collect();
            order.shuffle(source);

            self.names = order.iter().map(|at| self.names[*at]).collect();
            self.counts = order.iter().map(|at| self.counts[*at]).collect();
        }

        fn partial_shuffle<S: StochasticSource + ?Sized>(
            &mut self,
            count: usize,
            source: &mut S,
        ) -> usize {
            let mut order: Vec<usize> = (0..self.names.len()).collect();
            let filled = order.partial_shuffle(count, source);

            self.names = order.iter().map(|at| self.names[*at]).collect();
            self.counts = order.iter().map(|at| self.counts[*at]).collect();

            filled
        }
    }

    fn build() -> Paired {
        Paired {
            names: ('a'..='j').collect(),
            counts: (0..10).collect(),
        }
    }

    let seed = Seed::from_integer(107u64);

    let mut paired = build();
    seed.shuffle(&mut paired);

    // Each name kept its own count across the reordering.
    for (position, name) in paired.names.iter().enumerate() {
        let expected = u32::from(*name as u8 - b'a');

        assert_eq!(paired.counts[position], expected, "a pair came apart");
    }

    // Deterministic, like every other collection.
    let mut again = build();
    seed.shuffle(&mut again);

    assert_eq!(paired, again);

    // Actually reordered, rather than left alone.
    assert_ne!(paired, build());

    // And the same type works through `Random`, with no extra code either side.
    let mut streamed = build();
    streamed.shuffle(&mut stream(108));

    assert_eq!(streamed.names.len(), 10);

    // partial_shuffle too, through the seed's generic method.
    let mut few = build();

    assert_eq!(seed.partial_shuffle(&mut few, 4), 4);
}

// ---------------------------------------------------------------------------
// Partial shuffling across representations
// ---------------------------------------------------------------------------

#[test]
fn a_partial_shuffle_reports_what_it_filled_on_every_representation() {
    let seed = Seed::from_integer(109u64);

    // The count is `min(requested, size)` whatever the collection is.
    assert_eq!(seed.partial_shuffle(&mut (0..20u32).collect::<Vec<u32>>(), 5), 5);
    assert_eq!(seed.partial_shuffle(&mut (0..3u32).collect::<Vec<u32>>(), 5), 3);
    assert_eq!(seed.partial_shuffle(&mut ordered_set(20), 5), 5);
    assert_eq!(seed.partial_shuffle(&mut ordered_set(3), 5), 3);
    assert_eq!(seed.partial_shuffle(&mut ring_buffer(20), 5), 5);
    assert_eq!(seed.partial_shuffle(&mut sparse_sequence(20), 5), 5);
    assert_eq!(seed.partial_shuffle(&mut labeled(20), 5), 5);

    let mut array: [u32; 8] = std::array::from_fn(|index| index as u32);
    assert_eq!(seed.partial_shuffle(&mut array, 3), 3);
}

#[test]
fn a_partial_shuffles_prefix_is_distinct_and_from_the_collection() {
    // Drawn without replacement, so the chosen prefix never repeats an element. The
    // suffix is deliberately not checked: it is not a shuffle of the remainder.
    let world = Seed::from_integer(110u64);

    for index in 0..200u64 {
        let mut items: Vec<u32> = (0..50).collect();
        let filled = world.index(index).partial_shuffle(&mut items, 6);

        assert_eq!(filled, 6);

        let prefix: Vec<u32> = items[..6].to_vec();
        let distinct: HashSet<u32> = prefix.iter().copied().collect();

        assert_eq!(distinct.len(), 6, "the prefix repeated an element: {prefix:?}");

        for value in &prefix {
            assert!(*value < 50, "{value} was not one of the originals");
        }

        // Nothing was lost overall either.
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<u32>>());
    }
}

#[test]
fn a_partial_shuffle_keeps_collection_invariants_too() {
    let seed = Seed::from_integer(111u64);

    let mut set = ordered_set(40);
    seed.partial_shuffle(&mut set, 5);

    assert!(set.check_invariants(), "a partial shuffle broke the set");
    assert_eq!(set.len(), 40);

    for value in 0..40u32 {
        assert!(set.contains(&value));
    }

    let mut pairs = labeled(20);
    seed.partial_shuffle(&mut pairs, 5);

    for value in 0..20u32 {
        assert_eq!(pairs.label_of(&value), Some(&(value * 10)));
    }

    let before: Vec<i64> = sparse_sequence(20).iter().map(|(index, _)| index).collect();
    let mut sequence = sparse_sequence(20);
    seed.partial_shuffle(&mut sequence, 5);

    assert_eq!(
        sequence.iter().map(|(index, _)| index).collect::<Vec<i64>>(),
        before,
        "a partial shuffle moved the occupied indices",
    );
}
