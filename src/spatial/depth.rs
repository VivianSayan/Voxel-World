//! Octree levels, and how deep a tree goes.
//!
//! These are two different numbers that both get called "depth", which is most
//! of the reason for separating them. [`TreeDepth`] is a property of the world:
//! how many times the root subdivides before a node is one voxel. [`Depth`] is
//! a level within that, and may sit above the root at a negative value, where
//! nodes span whole trees.
//!
//! Deeper means a larger number and a smaller node. Depth 0 is the root,
//! [`TreeDepth`] is the floor where a node is a single voxel.

use std::fmt;

/// A level of the octree. Larger is deeper, and deeper is smaller.
///
/// Negative levels sit above the root, each one doubling the node again, which
/// is how a field gets features larger than a single tree.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct Depth(i8);

impl Depth {
    /// The root of one tree, where a node is the whole tree.
    pub const ROOT: Self = Self(0);

    /// The shallowest level that can be named: nodes here span `2^127` trees.
    pub const HIGHEST: Self = Self(i8::MIN);
    /// The deepest level that can be named. A given tree's own floor is
    /// [`TreeDepth::floor`], which is usually far above this.
    pub const LOWEST: Self = Self(i8::MAX);

    /// A level by number: zero is the root, positive is inside a tree, negative
    /// is above it.
    pub const fn new(level: i8) -> Self {
        Self(level)
    }

    /// The level as a number, for arithmetic this type does not offer.
    pub const fn level(self) -> i8 {
        self.0
    }

    /// `steps` levels further down, saturating rather than wrapping so that a
    /// walk off the bottom stays at the bottom.
    pub const fn deeper(self, steps: i8) -> Self {
        Self(self.0.saturating_add(steps))
    }

    /// `steps` levels back up.
    pub const fn shallower(self, steps: i8) -> Self {
        Self(self.0.saturating_sub(steps))
    }

    /// The level one step down, which is where this node's children live.
    pub const fn child(self) -> Self {
        self.deeper(1)
    }

    /// The level one step up, which is where this node's parent lives.
    pub const fn parent(self) -> Self {
        self.shallower(1)
    }

    /// Whether this level is above the root, where a node covers more than one
    /// tree.
    pub const fn is_above_root(self) -> bool {
        self.0 < 0
    }

    /// How many levels separate two, as a count that cannot be negative.
    ///
    /// Computed in 16 bits, so even the two extremes are a true count rather
    /// than an overflow.
    pub const fn distance_to(self, other: Self) -> u16 {
        (other.0 as i16 - self.0 as i16).unsigned_abs()
    }
}

impl fmt::Display for Depth {
    /// As `depth <level>`, such as `depth 7`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "depth {}", self.0)
    }
}

/// Any level number names a depth, so this cannot fail.
impl From<i8> for Depth {
    fn from(level: i8) -> Self {
        Self(level)
    }
}

/// How deep a tree goes: the level at which one node is one voxel.
///
/// Never negative, which is why it is a separate type rather than a [`Depth`],
/// and never above 127, so that every level it names fits in a [`Depth`].
///
/// Positions are given at this level, so a node here needs no shifting at all;
/// every shallower level shifts a coordinate right by the difference, which is
/// what [`TreeDepth::shift_for`] computes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct TreeDepth(u8);

impl TreeDepth {
    /// A tree of this many levels, or `None` above 127, which no [`Depth`]
    /// could name.
    pub const fn new(levels: u8) -> Option<Self> {
        if levels <= i8::MAX as u8 {
            Some(Self(levels))
        } else {
            None
        }
    }

    /// How many times the root subdivides before a node is one voxel.
    pub const fn levels(self) -> u8 {
        self.0
    }

    /// The deepest level this tree has, where a node is one voxel and a
    /// coordinate needs no shifting.
    pub const fn floor(self) -> Depth {
        Depth::new(self.0 as i8)
    }

    /// The shallowest level this tree can address: the one whose nodes are
    /// `2^127` voxels across, which is the widest shift a `u128` coordinate
    /// allows. Usually negative, since levels above the root are legal.
    pub const fn ceiling(self) -> Depth {
        Depth::new((self.0 as i16 - 127) as i8)
    }

    /// How many bits to shift a voxel coordinate right to get a node address
    /// at `depth`, which is also the width of that node as a power of two.
    ///
    /// The one place the conversion from a level to a shift is written, so
    /// nothing else has to remember that it is `tree_depth - depth` and that it
    /// can go wrong in two directions. `None` for a level below the tree's
    /// floor, which would be narrower than a voxel, and for one so far above
    /// the root that the shift would pass 127. Call [`TreeDepth::clamp`] first
    /// where saturating a requested level is what is wanted.
    pub const fn shift_for(self, depth: Depth) -> Option<u32> {
        let difference: i16 = self.0 as i16 - depth.level() as i16;
        if difference < 0 || difference > 127 {
            None
        } else {
            Some(difference as u32)
        }
    }

    /// How many voxels across a node at `depth` is: one shifted left by
    /// [`TreeDepth::shift_for`]. `None` for a level this tree cannot
    /// address.
    pub const fn node_width(self, depth: Depth) -> Option<u128> {
        match self.shift_for(depth) {
            Some(shift) => Some(1u128 << shift),
            None => None,
        }
    }

    /// Whether `depth` names a level this tree can address.
    ///
    /// Levels above the root count, as long as their node width still fits.
    pub const fn contains(self, depth: Depth) -> bool {
        self.shift_for(depth).is_some()
    }

    /// `depth` brought inside the tree: no deeper than
    /// [`TreeDepth::floor`], where a node is one voxel, and no shallower than
    /// [`TreeDepth::ceiling`], where the width stops fitting.
    pub const fn clamp(self, depth: Depth) -> Depth {
        if depth.level() > self.floor().level() {
            self.floor()
        } else if depth.level() < self.ceiling().level() {
            self.ceiling()
        } else {
            depth
        }
    }
}

impl fmt::Display for TreeDepth {
    /// As `<n> levels`, such as `12 levels`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} levels", self.0)
    }
}

/// As [`TreeDepth::new`], with a message instead of `None`.
impl TryFrom<u8> for TreeDepth {
    type Error = &'static str;
    fn try_from(levels: u8) -> Result<Self, Self::Error> {
        Self::new(levels).ok_or("tree depth must be in 0..=127")
    }
}
