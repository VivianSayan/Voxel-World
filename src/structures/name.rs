//! Text compared as an integer.

use crate::random::mixing::mix64;
use crate::structures::hashing::FastHashMap;
use crate::structures::traits::StableHash;
use crate::units::ContentHash;
use std::fmt;
use std::str::FromStr;
use std::sync::{OnceLock, RwLock};

/// The offset basis and prime of 64-bit FNV-1a, which is short enough to run in
/// a `const fn` and good enough before the avalanche that follows it.
const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

/// A name held as the hash of its text, so that comparing two is comparing two
/// integers.
///
/// # What it is for
///
/// Text used as an identity rather than as content: voxel kinds, property
/// names, tag segments, anything read from data files and then compared a great
/// many times. Godot's `StringName` solves the same problem. Measured on this
/// crate's own hashing benchmark, the same 200,000 inserts and look-ups cost
/// 14.8 ms keyed by short strings and 2.7 ms keyed by an integer; a name moves
/// the second column's cost onto the first column's ergonomics.
///
/// It is **not** for text that is content: a sign's inscription, a player's
/// message, anything shown or edited. Those want a `String`.
///
/// # The id is the hash, not a registration number
///
/// A name's id is a pure function of its text, worked out by the same code
/// every time. Nothing has to have been registered first, no table is consulted
/// to make one, and two processes that never spoke agree on what `"stone"` is.
///
/// That is the difference from an interner that hands out 0, 1, 2… in the order
/// it first saw each string. Those ids depend on load order, so they change when
/// a mod loads earlier, cannot be written to a save, and make any map keyed by
/// them iterate differently between runs. A hashed id has none of those
/// problems.
///
/// # Compile time
///
/// [`Name::new`] is a `const fn`, so a name can be worked out while compiling:
///
/// ```
/// use voxel_world::structures::Name;
///
/// const STONE: Name = Name::new("stone");
///
/// // In an expression, `name!` forces the same thing rather than leaving it to
/// // the optimiser.
/// use voxel_world::name;
/// assert_eq!(name!("stone"), STONE);
/// ```
///
/// A comparison against a constant name is then one integer compare against an
/// immediate, with no hashing and no allocation at run time.
///
/// # Reading one back
///
/// An id cannot be turned back into its text — that is what hashing means. A
/// global table maps the ids that have been *registered* back to their text, for
/// printing and debugging, and [`Name::text`] consults it. A name that was only
/// ever built by [`Name::new`] is unregistered and prints as its id.
///
/// Registering is what [`Name::intern`] and [`Name::register`] do, and it is
/// worth doing once at start-up for every name a person might have to read in a
/// log. Comparison never needs it.
///
/// ```
/// use voxel_world::structures::Name;
///
/// const STONE: Name = Name::new("stone");
///
/// assert_eq!(Name::intern("stone"), STONE, "the same id either way");
/// assert_eq!(STONE.text(), Some("stone"), "and now it can be read back");
/// ```
///
/// # Collisions
///
/// Two different strings could in principle land on the same id. With 64 bits
/// that needs about five billion distinct names before it is likely even once;
/// a world with a hundred thousand of them sits at roughly one chance in four
/// billion. The registry checks anyway, because the failure would be silent and
/// total — two names indistinguishable everywhere — and [`Name::intern`] panics
/// rather than let it pass. [`Name::try_intern`] reports it instead.
///
/// The check only covers registered names, which is another reason to register
/// everything at start-up.
///
/// # Ordering
///
/// [`Ord`] compares ids, so it is stable across runs and machines but **not
/// alphabetical**. It is the right order for a `BTreeMap<Name, _>` whose
/// iteration should be canonical; it is the wrong one for anything shown to a
/// person, which should sort on [`Name::text`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Name(u64);

/// Two different strings that want the same id.
///
/// Vanishingly unlikely, and unrecoverable if ignored, so it is reported rather
/// than papered over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NameCollision {
    /// The id both strings hash to.
    pub id: u64,
    /// The text already registered for it.
    pub registered: &'static str,
}

impl fmt::Display for NameCollision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "name id {:#018x} is already registered for {:?}",
            self.id, self.registered
        )
    }
}

impl std::error::Error for NameCollision {}

impl Name {
    /// The name of the empty string.
    pub const EMPTY: Self = Self::new("");

    /// The name for this text, worked out from the text alone.
    ///
    /// A `const fn`, so `const KIND: Name = Name::new("stone")` costs nothing at
    /// run time. Nothing is registered, so [`Name::text`] will not find it
    /// unless the same text is interned somewhere.
    pub const fn new(text: &str) -> Self {
        let bytes: &[u8] = text.as_bytes();
        let mut hash: u64 = OFFSET_BASIS;
        let mut index: usize = 0;

        while index < bytes.len() {
            hash ^= bytes[index] as u64;
            hash = hash.wrapping_mul(PRIME);
            index += 1;
        }

        // FNV alone leaves the high bits sluggish, and a hash map indexes with
        // the low ones; the avalanche spreads both.
        Self(mix64(hash))
    }

    /// The name for this text, registered so that it can be read back.
    ///
    /// The text is copied and kept for the life of the process, which is why
    /// [`Name::register`] is better where the text is already `'static`.
    ///
    /// # Panics
    ///
    /// If another string is already registered for this id. See
    /// [`Name::try_intern`] to handle that instead.
    pub fn intern(text: &str) -> Self {
        match Self::try_intern(text) {
            Ok(name) => name,
            Err(collision) => panic!("{collision}, not {text:?}"),
        }
    }

    /// The name for this text, registered, reporting a collision rather than
    /// panicking on one.
    pub fn try_intern(text: &str) -> Result<Self, NameCollision> {
        let name: Self = Self::new(text);

        if let Some(known) = name.text() {
            return if known == text {
                Ok(name)
            } else {
                Err(NameCollision {
                    id: name.0,
                    registered: known,
                })
            };
        }

        Self::record(name, || {
            Box::leak(text.to_owned().into_boxed_str()) as &'static str
        })
    }

    /// The name for text that already lives for the whole program, registered
    /// without copying it.
    ///
    /// # Panics
    ///
    /// If another string is already registered for this id.
    pub fn register(text: &'static str) -> Self {
        let name: Self = Self::new(text);

        if let Some(known) = name.text() {
            assert!(
                known == text,
                "name id {:#018x} is already registered for {known:?}, not {text:?}",
                name.0
            );

            return name;
        }

        Self::record(name, || text).expect("the id was free a moment ago")
    }

    /// Registers several names at once, which is what a start-up table wants.
    ///
    /// # Panics
    ///
    /// On the first collision, naming both strings.
    pub fn register_all<I: IntoIterator<Item = &'static str>>(texts: I) {
        for text in texts {
            Self::register(text);
        }
    }

    /// The text this name was made from, if it was ever registered.
    ///
    /// `None` is not an error: a name built by [`Name::new`] compares perfectly
    /// well without anything knowing what it says.
    pub fn text(self) -> Option<&'static str> {
        registry()
            .read()
            .expect("the name registry is never held across a panic")
            .get(&self.0)
            .copied()
    }

    /// Whether this name's text is known.
    pub fn is_registered(self) -> bool {
        self.text().is_some()
    }

    /// The id itself, for storing or for a switch.
    ///
    /// Stable across runs and machines, since it is a fixed function of the
    /// text. Prefer writing the text to a save all the same: an id cannot be
    /// read by a person, and cannot be recovered if the text is lost.
    pub const fn id(self) -> u64 {
        self.0
    }

    /// A name from an id that came from [`Name::id`].
    pub const fn from_id(id: u64) -> Self {
        Self(id)
    }

    /// How many names have been registered.
    pub fn registered_count() -> usize {
        registry()
            .read()
            .expect("the name registry is never held across a panic")
            .len()
    }

    /// Puts a name in the registry, taking the text only if the id is free.
    fn record(name: Self, text: impl FnOnce() -> &'static str) -> Result<Self, NameCollision> {
        let mut table = registry()
            .write()
            .expect("the name registry is never held across a panic");

        match table.get(&name.0) {
            // Another thread got there between the read and the write.
            Some(known) => {
                let known: &'static str = known;

                if known == text() {
                    Ok(name)
                } else {
                    Err(NameCollision {
                        id: name.0,
                        registered: known,
                    })
                }
            }
            None => {
                table.insert(name.0, text());

                Ok(name)
            }
        }
    }
}

/// The text of every registered name, kept for the life of the process.
///
/// Read far more than written, and written once per distinct name, so a lock
/// costs nothing that matters: comparing names never touches it.
fn registry() -> &'static RwLock<FastHashMap<u64, &'static str>> {
    static REGISTRY: OnceLock<RwLock<FastHashMap<u64, &'static str>>> = OnceLock::new();

    REGISTRY.get_or_init(|| RwLock::new(FastHashMap::default()))
}

/// A name worked out while compiling.
///
/// [`Name::new`] is a `const fn`, so the optimiser will usually fold it away on
/// a literal; this makes it a certainty by evaluating in a constant block, and
/// it fails to compile rather than silently hashing at run time.
///
/// ```
/// use voxel_world::name;
/// use voxel_world::structures::Name;
///
/// assert_eq!(name!("stone"), Name::new("stone"));
/// ```
#[macro_export]
macro_rules! name {
    ($text:literal) => {
        const { $crate::structures::Name::new($text) }
    };
}

impl From<&str> for Name {
    /// Interns, so the text can be read back.
    fn from(text: &str) -> Self {
        Self::intern(text)
    }
}

impl FromStr for Name {
    type Err = NameCollision;

    fn from_str(text: &str) -> Result<Self, NameCollision> {
        Self::try_intern(text)
    }
}

impl fmt::Display for Name {
    /// The text if it is known, and the id if it is not.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.text() {
            Some(text) => formatter.write_str(text),
            None => write!(formatter, "#{:016x}", self.0),
        }
    }
}

impl fmt::Debug for Name {
    /// As the text it was made from, quoted, or as its id when unregistered.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.text() {
            Some(text) => write!(formatter, "Name({text:?})"),
            None => write!(formatter, "Name(#{:016x})", self.0),
        }
    }
}

/// The id, which is already a function of the text alone, so two runs agree.
impl StableHash for Name {
    fn stable_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&self.0.to_le_bytes())
    }
}
