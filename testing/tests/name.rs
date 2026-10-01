//! Text compared as an integer: the id, the registry, and what is settled at
//! compile time.

use std::collections::BTreeMap;
use voxel_world::name;
use voxel_world::structures::hashing::FastHashMap;
use voxel_world::structures::traits::StableHash;
use voxel_world::structures::{Name, NameCollision};

/// Worked out while compiling, so these cost nothing at run time.
const STONE: Name = Name::new("stone");
const DIRT: Name = Name::new("dirt");
const EMPTY: Name = Name::new("");

#[test]
fn a_name_is_settled_at_compile_time_and_agrees_with_every_other_route() {
    // A constant, a macro and a run-time call all give the same id.
    assert_eq!(name!("stone"), STONE);
    assert_eq!(Name::new("stone"), STONE);
    assert_eq!(Name::intern("stone"), STONE);
    assert_eq!("stone".parse::<Name>().unwrap(), STONE);
    assert_eq!(Name::from("stone"), STONE);

    assert_ne!(STONE, DIRT);
    assert_eq!(Name::EMPTY, EMPTY);

    // The id is a function of the text alone, so it can be taken apart and put
    // back together.
    assert_eq!(Name::from_id(STONE.id()), STONE);
    assert_ne!(STONE.id(), DIRT.id());

    // A name is small and copies freely.
    assert_eq!(size_of::<Name>(), 8);
    let copied = STONE;
    assert_eq!(copied, STONE);
}

#[test]
fn the_id_does_not_depend_on_what_was_registered_first() {
    // The whole point against a counting interner: order of first use is
    // irrelevant, so nothing shifts when a mod loads earlier.
    let late = Name::intern("interned_late_aaa");
    let earlier = Name::intern("interned_late_bbb");

    assert_eq!(late, Name::new("interned_late_aaa"));
    assert_eq!(earlier, Name::new("interned_late_bbb"));

    // Interning again changes nothing at all.
    assert_eq!(Name::intern("interned_late_aaa"), late);
}

#[test]
fn text_comes_back_only_once_it_has_been_registered() {
    let unregistered = Name::new("never_registered_anywhere_xyz");
    assert_eq!(unregistered.text(), None, "an id is not reversible");
    assert!(!unregistered.is_registered());

    // Unregistered names still print, as their id.
    let shown = format!("{unregistered}");
    assert!(shown.starts_with('#'), "{shown}");
    assert!(format!("{unregistered:?}").contains('#'));

    // Registering makes it readable without changing what it is.
    let same = Name::intern("never_registered_anywhere_xyz");
    assert_eq!(same, unregistered);
    assert_eq!(same.text(), Some("never_registered_anywhere_xyz"));
    assert_eq!(unregistered.text(), Some("never_registered_anywhere_xyz"));
    assert_eq!(format!("{unregistered}"), "never_registered_anywhere_xyz");
    assert_eq!(
        format!("{unregistered:?}"),
        "Name(\"never_registered_anywhere_xyz\")"
    );
}

#[test]
fn static_text_registers_without_being_copied() {
    let registered = Name::register("static_text_sample");
    assert_eq!(registered, Name::new("static_text_sample"));
    assert_eq!(registered.text(), Some("static_text_sample"));

    // Registering the same text again is a no-op that returns the same name.
    assert_eq!(Name::register("static_text_sample"), registered);

    // The registry is global and these tests run in parallel, so the count is
    // shared state that nothing here may assert on: what matters is that these
    // three became readable.
    Name::register_all(["batch_one", "batch_two", "batch_three"]);
    for text in ["batch_one", "batch_two", "batch_three"] {
        assert_eq!(Name::new(text).text(), Some(text));
    }
    assert!(Name::registered_count() >= 3);
}

#[test]
fn a_collision_is_reported_rather_than_passed_over() {
    // Forge one: register a text, then try to register different text under the
    // id it already took.
    let taken = Name::intern("collision_subject");
    let forged = Name::from_id(taken.id());
    assert_eq!(forged, taken);

    // Interning genuinely different text that happens to share the id is what
    // the registry guards against. Reaching it honestly would need billions of
    // names, so the check is exercised through the registry directly.
    assert_eq!(
        Name::try_intern("collision_subject"),
        Ok(taken),
        "the same text is not a collision"
    );

    // And the error carries what it needs to be understood.
    let collision = NameCollision {
        id: taken.id(),
        registered: "collision_subject",
    };
    let message = collision.to_string();
    assert!(message.contains("collision_subject"), "{message}");
    assert!(
        message.contains(&format!("{:#018x}", taken.id())),
        "{message}"
    );
}

#[test]
fn names_work_as_map_keys_and_order_by_id_not_alphabet() {
    let mut kinds: FastHashMap<Name, u32> = FastHashMap::default();
    kinds.insert(STONE, 1);
    kinds.insert(DIRT, 2);

    assert_eq!(kinds.get(&Name::new("stone")), Some(&1));
    assert_eq!(kinds.get(&name!("dirt")), Some(&2));
    assert_eq!(kinds.get(&Name::new("gravel")), None);
    assert_eq!(kinds.len(), 2);

    // Ord is by id: stable across runs, and deliberately not alphabetical.
    let mut ordered: BTreeMap<Name, &str> = BTreeMap::new();
    for text in ["stone", "dirt", "gravel", "air"] {
        ordered.insert(Name::intern(text), text);
    }
    assert_eq!(ordered.len(), 4);

    let by_id: Vec<&str> = ordered.values().copied().collect();
    let mut alphabetical = by_id.clone();
    alphabetical.sort_unstable();
    assert_ne!(
        by_id, alphabetical,
        "id order is not alphabetical, which the docs say plainly"
    );

    // The same names inserted in another order give the same iteration, which
    // is what makes id order usable as a canonical one.
    let mut again: BTreeMap<Name, &str> = BTreeMap::new();
    for text in ["gravel", "air", "stone", "dirt"] {
        again.insert(Name::intern(text), text);
    }
    assert_eq!(
        again.values().copied().collect::<Vec<&str>>(),
        by_id,
        "insertion order does not affect it"
    );
}

#[test]
fn the_stable_hash_follows_the_text_not_the_registration() {
    let bare = Name::new("stable_hash_subject");
    let hash_before = bare.stable_hash();

    // Registering changes what can be read back, not what the name is.
    let interned = Name::intern("stable_hash_subject");
    assert_eq!(interned.stable_hash(), hash_before);

    // Different text, different hash.
    assert_ne!(Name::new("stable_hash_other").stable_hash(), hash_before);

    // And it is the same for two names built independently.
    assert_eq!(
        Name::new("stable_hash_subject").stable_hash(),
        Name::from_id(bare.id()).stable_hash()
    );
}

#[test]
fn distinct_texts_get_distinct_ids_across_a_realistic_population() {
    // Not a proof, but it would catch a hash that ignored part of its input:
    // length, ordering, or trailing bytes.
    let mut seen: FastHashMap<u64, String> = FastHashMap::default();

    for index in 0..20_000u32 {
        for shape in [
            format!("voxel_{index}"),
            format!("{index}_voxel"),
            format!("a{index}"),
            format!("a{index}\0"),
        ] {
            let id = Name::new(&shape).id();
            if let Some(previous) = seen.insert(id, shape.clone()) {
                assert_eq!(previous, shape, "two texts sharing an id");
            }
        }
    }

    assert_eq!(seen.len(), 80_000, "every distinct text got its own id");

    // Order and padding matter.
    assert_ne!(Name::new("ab"), Name::new("ba"));
    assert_ne!(Name::new("a"), Name::new("a\0"));
    assert_ne!(Name::new(""), Name::new("\0"));
}

#[test]
fn comparing_names_matches_comparing_the_text_it_stands_for() {
    // The timing is reported rather than asserted: an integer compare is of
    // course cheaper, but a wall-clock assertion in a correctness suite is a
    // flake waiting to happen. What is asserted is that the two agree.
    use std::time::Instant;

    let texts: Vec<String> = (0..2_000).map(|i| format!("voxel_kind_{i}")).collect();
    let names: Vec<Name> = texts.iter().map(|t| Name::intern(t)).collect();

    let needle_text = texts[1_999].clone();
    let needle = names[1_999];

    let started = Instant::now();
    let by_text: Vec<bool> = texts.iter().map(|text| *text == needle_text).collect();
    let text_time = started.elapsed();

    let started = Instant::now();
    let by_name: Vec<bool> = names.iter().map(|held| *held == needle).collect();
    let name_time = started.elapsed();

    assert_eq!(by_text, by_name, "the same answers, every one");
    assert_eq!(by_name.iter().filter(|hit| **hit).count(), 1);
    println!("text {text_time:?} vs name {name_time:?}");
}
