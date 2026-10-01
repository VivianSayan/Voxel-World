//! Precedence, shadowing, and what happens to the layers underneath.

use voxel_world::structures::hashing::FastHashMap;
use voxel_world::structures::mappings::LayeredMap;
use voxel_world::structures::traits::{ContentHashable, Map};

fn layer(pairs: &[(&'static str, u32)]) -> FastHashMap<&'static str, u32> {
    pairs.iter().copied().collect()
}

fn settings() -> LayeredMap<&'static str, u32> {
    LayeredMap::from_layers([
        layer(&[("volume", 80)]),
        layer(&[("volume", 50), ("fullscreen", 0)]),
        layer(&[("volume", 30), ("fullscreen", 1), ("distance", 8)]),
    ])
}

#[test]
fn the_earliest_layer_holding_a_key_decides_its_value() {
    let map = settings();

    assert_eq!(map.get(&"volume"), Some(&80), "layer 0 outranks the rest");
    assert_eq!(
        map.get(&"fullscreen"),
        Some(&0),
        "layer 1, layer 0 has none"
    );
    assert_eq!(map.get(&"distance"), Some(&8), "only layer 2 has it");
    assert_eq!(map.get(&"missing"), None);

    assert_eq!(map.find_layer(&"volume"), Some(0));
    assert_eq!(map.find_layer(&"fullscreen"), Some(1));
    assert_eq!(map.find_layer(&"distance"), Some(2));
    assert_eq!(map.find_layer(&"missing"), None);

    assert!(map.contains_key(&"distance"));
    assert!(!map.contains_key(&"missing"));
}

#[test]
fn a_write_finds_the_layer_that_already_holds_the_key() {
    // The distinction from Python's ChainMap, which would write to layer 0.
    let mut map = LayeredMap::from_layers([layer(&[]), layer(&[("x", 10)]), layer(&[("x", 20)])]);

    assert_eq!(map.insert("x", 15), Some(10), "the replaced value");

    assert!(map.layer(0).unwrap().is_empty(), "layer 0 is untouched");
    assert_eq!(map.get_from_layer(1, &"x"), Some(&15), "changed in place");
    assert_eq!(map.get_from_layer(2, &"x"), Some(&20), "still shadowed");
    assert_eq!(map.get(&"x"), Some(&15));
    assert_eq!(map.find_layer(&"x"), Some(1));

    // In-place mutation finds the same layer and moves nothing.
    *map.get_mut(&"x").unwrap() = 16;
    assert_eq!(map.get_from_layer(1, &"x"), Some(&16));
    assert_eq!(map.find_layer(&"x"), Some(1));
    assert!(map.layer(0).unwrap().is_empty());
}

#[test]
fn a_key_no_layer_holds_goes_into_the_first_layer() {
    let mut map = settings();

    assert_eq!(map.insert("new", 1), None);
    assert_eq!(map.find_layer(&"new"), Some(0));
    assert_eq!(map.get(&"new"), Some(&1));

    // Even when lower layers exist and are the natural home of similar keys.
    let mut single: LayeredMap<&str, u32> = LayeredMap::new();
    assert_eq!(single.layer_count(), 1);
    assert_eq!(single.insert("only", 7), None);
    assert_eq!(single.find_layer(&"only"), Some(0));
}

#[test]
fn removing_the_visible_value_uncovers_the_one_below_it() {
    let mut map = LayeredMap::from_layers([layer(&[("x", 10)]), layer(&[("x", 20)])]);

    assert_eq!(map.get(&"x"), Some(&10));
    assert!(map.is_shadowed(&"x"));

    assert_eq!(map.remove(&"x"), Some(10));
    assert_eq!(
        map.get(&"x"),
        Some(&20),
        "the lower value was never touched"
    );
    assert!(map.contains_key(&"x"), "the key is still present");
    assert!(!map.is_shadowed(&"x"), "nothing left underneath");

    assert_eq!(map.remove(&"x"), Some(20));
    assert_eq!(map.get(&"x"), None);
    assert!(!map.contains_key(&"x"));
    assert_eq!(map.remove(&"x"), None);
}

#[test]
fn a_key_can_be_taken_out_of_every_layer_at_once() {
    let mut map = settings();
    assert_eq!(map.occurrences(&"volume").count(), 3);

    assert_eq!(map.remove_everywhere(&"volume"), 3);
    assert!(!map.contains_key(&"volume"));
    assert_eq!(map.occurrences(&"volume").count(), 0);
    assert_eq!(map.remove_everywhere(&"volume"), 0);

    // The other keys are untouched.
    assert_eq!(map.get(&"fullscreen"), Some(&0));
    assert_eq!(map.get(&"distance"), Some(&8));
}

#[test]
fn every_occurrence_of_a_key_can_be_inspected_in_precedence_order() {
    let map = settings();

    let found: Vec<(usize, u32)> = map
        .occurrences(&"volume")
        .map(|(index, value)| (index, *value))
        .collect();
    assert_eq!(found, vec![(0, 80), (1, 50), (2, 30)]);

    assert!(map.is_shadowed(&"volume"));
    assert!(map.is_shadowed(&"fullscreen"));
    assert!(!map.is_shadowed(&"distance"), "only one layer holds it");

    // What a removal would uncover, in turn.
    let hidden: Vec<u32> = map
        .occurrences(&"volume")
        .skip(1)
        .map(|(_, v)| *v)
        .collect();
    assert_eq!(hidden, vec![50, 30]);
}

#[test]
fn visible_iteration_shows_each_key_once() {
    let map = LayeredMap::from_layers([layer(&[("a", 1), ("b", 2)]), layer(&[("b", 3), ("c", 4)])]);

    let mut seen: Vec<(&str, u32)> = map.iter().map(|(k, v)| (*k, *v)).collect();
    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![("a", 1), ("b", 2), ("c", 4)],
        "b comes from layer 0"
    );

    assert_eq!(map.len(), 3, "three visible keys");
    assert_eq!(map.total_len(), 4, "four pairs stored");

    let mut keys: Vec<&str> = map.keys().copied().collect();
    keys.sort_unstable();
    assert_eq!(keys, vec!["a", "b", "c"]);

    let mut values: Vec<u32> = map.values().copied().collect();
    values.sort_unstable();
    assert_eq!(values, vec![1, 2, 4]);

    // Flattening keeps exactly the visible view.
    let flat = map.flatten();
    assert_eq!(flat.len(), 3);
    assert_eq!(flat.get("b"), Some(&2));
    assert_eq!(flat.get("c"), Some(&4));
}

#[test]
fn layers_can_be_added_removed_and_reordered() {
    let mut map = LayeredMap::from_layers([layer(&[("x", 1)]), layer(&[("x", 2)])]);
    assert_eq!(map.layer_count(), 2);
    assert_eq!(map.get(&"x"), Some(&1));

    // Reordering changes which value wins, and nothing else.
    assert!(map.swap_layers(0, 1));
    assert_eq!(map.get(&"x"), Some(&2));
    assert!(!map.swap_layers(0, 9));

    // A new layer on top takes precedence; below, it is the last resort.
    assert!(map.insert_layer(0, layer(&[("x", 3)])));
    assert_eq!(map.get(&"x"), Some(&3));
    assert_eq!(map.layer_count(), 3);
    assert!(!map.insert_layer(99, layer(&[])));

    map.push_layer(layer(&[("x", 4), ("only_low", 5)]));
    assert_eq!(map.get(&"x"), Some(&3), "still the top layer");
    assert_eq!(map.get(&"only_low"), Some(&5));

    // Moving keeps the relative order of the others, unlike a swap.
    assert!(map.move_layer(0, 2));
    assert_eq!(
        map.get(&"x"),
        Some(&2),
        "layer 0 moved down past the others"
    );
    assert!(!map.move_layer(0, 99));

    // Removing a layer hands it back and uncovers what it hid.
    let taken = map.remove_layer(0).expect("more than one layer");
    assert_eq!(taken.get("x"), Some(&2));
    assert_eq!(map.layer_count(), 3);

    // Layers can be worked on directly.
    map.layer_mut(0).expect("a layer").insert("direct", 9);
    assert_eq!(map.get(&"direct"), Some(&9));
    assert!(map.layer(9).is_none());
    assert!(map.layer_mut(9).is_none());
    assert_eq!(map.layers().count(), 3);
}

#[test]
fn there_is_always_a_layer_to_insert_into() {
    let mut map = LayeredMap::from_layers([layer(&[("a", 1)]), layer(&[("b", 2)])]);

    assert!(map.remove_layer(1).is_some());
    assert_eq!(map.layer_count(), 1);

    // The last layer cannot be removed, so insertion can never have nowhere to
    // go.
    assert!(map.remove_layer(0).is_none());
    assert_eq!(map.layer_count(), 1);
    assert_eq!(map.insert("c", 3), None);
    assert_eq!(map.get(&"c"), Some(&3));

    // An empty list of layers still gives one.
    let empty: LayeredMap<&str, u32> = LayeredMap::from_layers([]);
    assert_eq!(empty.layer_count(), 1);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(LayeredMap::<&str, u32>::with_layers(0).layer_count(), 1);
    assert_eq!(LayeredMap::<&str, u32>::with_layers(4).layer_count(), 4);
}

#[test]
fn one_layer_behaves_like_an_ordinary_map() {
    let mut map: LayeredMap<&str, u32> = [("a", 1), ("b", 2)].into_iter().collect();

    assert_eq!(map.layer_count(), 1);
    assert_eq!(map.len(), 2);
    assert_eq!(map.total_len(), 2);
    assert_eq!(map.get(&"a"), Some(&1));
    assert_eq!(map.insert("a", 5), Some(1));
    assert_eq!(map.remove(&"a"), Some(5));
    assert_eq!(map.get(&"a"), None);
    assert!(!map.is_shadowed(&"b"));

    map.clear();
    assert!(map.is_empty());
    assert_eq!(map.layer_count(), 1, "clearing keeps the layers");
}

#[test]
fn a_layer_can_be_chosen_explicitly_for_reading_and_writing() {
    let mut map = settings();

    // Writing into a chosen layer, which is what Python's ChainMap always does.
    assert_eq!(map.insert_into(0, "distance", 16), Ok(None));
    assert_eq!(map.get(&"distance"), Some(&16), "now overridden");
    assert_eq!(map.get_from_layer(2, &"distance"), Some(&8), "and shadowed");
    assert_eq!(map.insert_into(0, "distance", 32), Ok(Some(16)));

    // A layer that does not exist gives the pair back rather than losing it.
    assert_eq!(map.insert_into(9, "lost", 1), Err(("lost", 1)));
    assert!(!map.contains_key(&"lost"));

    // Removing from a chosen layer, including one that is already shadowed.
    assert_eq!(map.remove_from_layer(2, &"volume"), Some(30));
    assert_eq!(
        map.get(&"volume"),
        Some(&80),
        "the visible one is untouched"
    );
    assert_eq!(map.occurrences(&"volume").count(), 2);
    assert_eq!(map.remove_from_layer(9, &"volume"), None);
    assert_eq!(map.get_from_layer(9, &"volume"), None);
}

#[test]
fn a_value_can_be_promoted_into_a_higher_layer() {
    let mut map = LayeredMap::from_layers([layer(&[]), layer(&[]), layer(&[("x", 20)])]);

    assert_eq!(map.find_layer(&"x"), Some(2));
    assert!(map.promote(&"x", 0));
    assert_eq!(map.find_layer(&"x"), Some(0));
    assert_eq!(map.get(&"x"), Some(&20), "the value came with it");
    assert!(
        map.layer(2).unwrap().is_empty(),
        "it moved rather than being copied"
    );

    // Promoting to where it already is does nothing and reports success.
    assert!(map.promote(&"x", 0));
    assert_eq!(map.find_layer(&"x"), Some(0));

    // A demotion is refused, since it could change what is visible.
    assert!(!map.promote(&"x", 2));
    assert_eq!(map.find_layer(&"x"), Some(0));

    // As are a missing key and a layer that does not exist.
    assert!(!map.promote(&"missing", 0));
    assert!(!map.promote(&"x", 9));

    // Promoting never disturbs what was shadowed.
    let mut shadowed =
        LayeredMap::from_layers([layer(&[]), layer(&[("y", 1)]), layer(&[("y", 2)])]);
    assert!(shadowed.promote(&"y", 0));
    assert_eq!(shadowed.get(&"y"), Some(&1));
    assert_eq!(shadowed.get_from_layer(2, &"y"), Some(&2));
    assert_eq!(shadowed.occurrences(&"y").count(), 2);
}

#[test]
fn the_map_trait_describes_the_visible_view() {
    let map = settings();

    assert_eq!(Map::len(&map), 3, "three distinct visible keys");
    assert!(!Map::is_empty(&map));
    assert!(Map::contains_key(&map, &"volume"));
    assert_eq!(Map::get(&map, &"volume"), Some(&80));

    // Pairs are the visible ones, so a shadowed value is not one of them.
    assert_eq!(Map::pairs(&map).count(), 3);
    assert_eq!(Map::pair_count(&map), 3);
    assert!(map.contains_pair(&"volume", &80));
    assert!(
        !map.contains_pair(&"volume", &50),
        "shadowed, so not a pair"
    );
    assert_eq!(Map::keys(&map).count(), 3);

    // Submap comparison follows the visible view too.
    let flat: LayeredMap<&str, u32> = map.flatten().into_iter().collect();
    assert!(map.is_submap(&flat));
    assert!(map.is_supermap(&flat));
}

#[test]
fn the_content_hash_follows_the_visible_view_not_the_layering() {
    // The same visible pairs arranged differently hash alike.
    let layered = LayeredMap::from_layers([layer(&[("a", 1)]), layer(&[("a", 9), ("b", 2)])]);
    let flat: LayeredMap<&str, u32> = [("a", 1), ("b", 2)].into_iter().collect();

    assert_eq!(layered.content_hash(), flat.content_hash());
    assert!(layered.content_matches(&flat));

    // A change to what is visible changes the hash.
    let mut changed = layered.clone();
    changed.insert("a", 5);
    assert_ne!(changed.content_hash(), layered.content_hash());

    // A change only to what is shadowed does not.
    let mut shadowed = layered.clone();
    assert_eq!(shadowed.insert_into(1, "a", 100), Ok(Some(9)));
    assert_eq!(shadowed.content_hash(), layered.content_hash());
}

#[test]
fn iteration_repeats_and_reserving_leaves_the_view_alone() {
    let mut map = settings();

    // Deterministic order: the same map iterates the same way each time.
    let first: Vec<(&str, u32)> = map.iter().map(|(k, v)| (*k, *v)).collect();
    let second: Vec<(&str, u32)> = map.iter().map(|(k, v)| (*k, *v)).collect();
    assert_eq!(first, second);

    // Layer order decides precedence, never hash order: every visible pair
    // comes from the earliest layer holding its key.
    for (key, value) in map.iter() {
        let layer = map.find_layer(key).expect("a visible key is somewhere");
        assert_eq!(map.get_from_layer(layer, key), Some(value));
    }

    map.reserve(16);
    map.shrink_to_fit();
    assert_eq!(map.get(&"volume"), Some(&80));
    assert_eq!(map.len(), 3);
}
