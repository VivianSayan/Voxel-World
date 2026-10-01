//! Identifiers, one kind per thing being identified.
//!
//! Every id in a program tends to be a `u64`, so a voxel type id fits perfectly
//! into a slot expecting an entity id and the compiler says nothing. [`Id<K>`]
//! is a `u64` tagged at compile time with what it identifies, so those two are
//! different types that cannot be swapped, assigned or compared.
//!
//! The tag costs nothing: `Id<K>` holds a [`PhantomData`], so it is still eight
//! bytes and still `Copy`.
//!
//! ```ignore
//! define_id_kinds! {
//!     VoxelTypeKind => "voxel type",
//!     EntityKind => "entity",
//! }
//!
//! pub type VoxelTypeId = Id<VoxelTypeKind>;
//! pub type EntityId = Id<EntityKind>;
//! ```
//!
//! Tags come from the kind's name, so kinds can be declared in any order and in
//! separate modules without colliding or renumbering each other.
//!
//! [`AnyId`] is the universal one, for the places that genuinely have to hold
//! ids of mixed kinds: a save file's index, a debug overlay, an event queue
//! carrying references to whatever raised it. It keeps the kind as a value
//! rather than a type, so two ids of different kinds still never compare equal,
//! and converting back to a typed id checks the kind.

use crate::random::seed::Seed;
use std::fmt;
use std::marker::PhantomData;

/// The tag a kind gets, derived from its name at compile time.
///
/// Deriving it from the name rather than from declaration order means a kind
/// keeps its tag wherever it is declared and whatever is declared beside it, so
/// kinds can be added anywhere and split across modules without renumbering.
/// Renaming one does change its tag, so treat a kind's name as part of the save
/// format once ids have been written down.
///
/// FNV-1a, because it is short enough to run in a `const` and the only thing
/// being asked of it is that a few dozen names land on a few dozen numbers.
pub const fn tag_for_name(name: &str) -> u32 {
    let bytes: &[u8] = name.as_bytes();
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut index: usize = 0;

    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        index += 1;
    }

    (hash ^ (hash >> 32)) as u32
}

/// What an [`Id`] identifies. Implemented by marker types, usually through
/// [`define_id_kinds!`](crate::define_id_kinds).
pub trait IdKind: Copy + 'static {
    /// A number no other kind shares, which is what keeps an [`AnyId`] of one
    /// kind from equalling another. Derived from [`IdKind::NAME`] by
    /// [`tag_for_name`].
    const TAG: u32;

    /// What to call this kind when printing one.
    const NAME: &'static str;
}

/// Declares id kinds.
///
/// Each kind's tag comes from its name, not from where it sits in the list, so
/// kinds may be added anywhere, reordered, or declared in separate invocations
/// in different modules without any of them changing number. Renaming one does
/// change its tag, so a kind's name is part of the save format once ids have
/// been written down.
#[macro_export]
macro_rules! define_id_kinds {
    ($($kind:ident => $name:literal),+ $(,)?) => {
        $(
            #[doc = concat!("The kind marking an [`Id`](", stringify!($crate), "::units::Id) as identifying a ", $name, ".")]
            ///
            /// A marker type with no value: it exists only in the type of an
            /// [`Id`], and carries the tag derived from its name.
            #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
            pub struct $kind;

            impl $crate::units::identifier::IdKind for $kind {
                const TAG: u32 =
                    $crate::units::identifier::tag_for_name($name);
                const NAME: &'static str = $name;
            }
        )+
    };
}

/// An identifier for one kind of thing.
///
/// A `u64` with a marker type attached. Two ids of different kinds are
/// different types, so nothing can assign, compare or pass one where the other
/// belongs; two of the same kind compare, order and hash as the plain numbers
/// they are. The marker is a [`PhantomData`], so this is still eight bytes and
/// still `Copy`.
///
/// `u64::MAX` is reserved for [`Id::NONE`] and is what [`Default`] gives, so an
/// absent id needs no `Option` in a struct or array that cannot afford one.
pub struct Id<K: IdKind> {
    value: u64,
    kind: PhantomData<K>,
}

impl<K: IdKind> Id<K> {
    /// The id reserved for "nothing", so that an absent id needs no `Option`
    /// in the places that cannot afford one.
    pub const NONE: Self = Self {
        value: u64::MAX,
        kind: PhantomData,
    };

    /// An id at a given number. Nothing is checked: `u64::MAX` gives
    /// [`Id::NONE`], and numbering is the caller's business.
    pub const fn new(value: u64) -> Self {
        Self {
            value,
            kind: PhantomData,
        }
    }

    /// The number underneath, for indexing or writing down.
    pub const fn value(self) -> u64 {
        self.value
    }

    /// Whether this is [`Id::NONE`].
    pub const fn is_none(self) -> bool {
        self.value == u64::MAX
    }

    /// An id derived from a seed, for ids a world generates rather than counts
    /// out.
    ///
    /// The kind's tag is the index the seed is derived through, so the same
    /// seed gives unrelated ids for different kinds, and the same seed and kind
    /// always give the same id. Can in principle land on `u64::MAX` and so on
    /// [`Id::NONE`], with probability `2^-64`.
    pub fn from_seed(seed: Seed) -> Self {
        Self::new(seed.index(K::TAG as u64).as_u64())
    }

    /// The same id with its kind moved from the type to a field, for a
    /// container that holds several kinds. Reversed by
    /// [`AnyId::downcast`].
    pub fn to_any(self) -> AnyId {
        AnyId {
            tag: K::TAG,
            value: self.value,
        }
    }
}

// The traits below are written out rather than derived. A derive would add a
// `K: Clone` style bound on the marker type, which never exists at runtime and
// has no business being asked for; each one forwards to the `u64` instead.
impl<K: IdKind> Clone for Id<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: IdKind> Copy for Id<K> {}

impl<K: IdKind> PartialEq for Id<K> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<K: IdKind> Eq for Id<K> {}

impl<K: IdKind> PartialOrd for Id<K> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: IdKind> Ord for Id<K> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<K: IdKind> std::hash::Hash for Id<K> {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        self.value.hash(hasher);
    }
}

impl<K: IdKind> Default for Id<K> {
    /// [`Id::NONE`], so that a defaulted id is absent rather than pointing at
    /// whatever happens to be numbered zero.
    fn default() -> Self {
        Self::NONE
    }
}

impl<K: IdKind> fmt::Debug for Id<K> {
    /// As `<kind name>(<number>)`, or `<kind name>(none)` for [`Id::NONE`].
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            return write!(formatter, "{}(none)", K::NAME);
        }

        write!(formatter, "{}({})", K::NAME, self.value)
    }
}

impl<K: IdKind> fmt::Display for Id<K> {
    /// The same as [`Debug`](fmt::Debug): the kind's name is worth showing
    /// either way.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, formatter)
    }
}

/// An identifier of any kind, carrying which kind as a value.
///
/// For the places that have to hold a mixture: a save index, an event that
/// refers to whatever raised it, a debug view. Two ids never compare equal
/// across kinds, even at the same number, so nothing is lost by the type
/// disappearing.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AnyId {
    tag: u32,
    value: u64,
}

impl AnyId {
    /// An id from a tag and a number, for reading one back from storage. Prefer
    /// [`Id::to_any`], which takes the tag from the kind itself.
    pub const fn new(tag: u32, value: u64) -> Self {
        Self { tag, value }
    }

    /// Which kind this is, as [`IdKind::TAG`].
    pub const fn tag(self) -> u32 {
        self.tag
    }

    /// The number underneath.
    pub const fn value(self) -> u64 {
        self.value
    }

    /// Whether this is an absent id; see [`Id::NONE`].
    pub const fn is_none(self) -> bool {
        self.value == u64::MAX
    }

    /// Whether this is an id of that kind.
    pub fn is<K: IdKind>(self) -> bool {
        self.tag == K::TAG
    }

    /// The typed id back, or `None` when it is of some other kind. The one way
    /// to leave [`AnyId`], and it checks the tag first.
    pub fn downcast<K: IdKind>(self) -> Option<Id<K>> {
        if !self.is::<K>() {
            return None;
        }

        Some(Id::new(self.value))
    }
}

impl<K: IdKind> From<Id<K>> for AnyId {
    fn from(id: Id<K>) -> Self {
        id.to_any()
    }
}

impl fmt::Display for AnyId {
    /// As `kind <tag>(<number>)`. The tag is a number here rather than a name,
    /// since the kind is no longer in the type to ask.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            return write!(formatter, "kind {}(none)", self.tag);
        }

        write!(formatter, "kind {}({})", self.tag, self.value)
    }
}
