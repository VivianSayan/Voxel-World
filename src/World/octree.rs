//! Sparse voxel storage: a node is one type, a weighted choice, a packed block
//! of types, eight children, or a promise that a generator will supply it.
//!
//! The tree compresses *coherence*: a region whose voxels agree collapses to
//! one node, and nothing else turns a million voxels of sky into a single
//! entry. What it cannot compress is a region where neighbours disagree, since
//! there is no agreement left to exploit — subdividing to the last voxel costs
//! far more than storing them plainly.
//!
//! A packed node is where that stops. Below the depth at which the world is
//! still coherent, a node holds a [`CompactSequence`] of its voxels instead of
//! children. It can use one value, a palette, runs, or direct storage according
//! to the data it actually contains.

use crate::random::seed::Seed;
use crate::structures::storage::CompactSequence;
use crate::units::{Unit, Weights};
use crate::world::ids::VoxelTypeId;

/// The eight children in x-fastest order, as [`Octree::descend`] indexes them.
const CHILDREN: usize = 8;

/// What a node actually holds.
///
/// Every payload larger than a pointer is boxed. A node is whichever variant is
/// biggest, so an unboxed weighted list or packed block would be paid for by
/// every uniform node in the tree, and uniform nodes are most of a world.
enum Octype {
    Children { children: Option<Box<[Octree; 8]>> },
    Uniform { id: VoxelTypeId },
    Probability(Box<WeightedTypes>),
    Packed(Box<CompactSequence<VoxelTypeId>>),
    Procedural,
}

/// The payload of a weighted node: which types it may hold, and the thresholds
/// a draw is compared against.
struct WeightedTypes {
    /// Running totals rather than the weights they were built from: entry `i`
    /// is the chance of landing on `ids[0..=i]`, so the last one is always 1.
    /// Picking is then one comparison per entry against a single draw, with no
    /// summing at pick time.
    cumulative: Vec<Unit>,
    ids: Vec<VoxelTypeId>,
}

/// One node of the sparse voxel tree.
///
/// A node is uniform where a whole region is one type, which is most of a
/// world; it holds children only where the region actually varies. A weighted
/// node stands for a region that is one type per voxel but mixed overall, such
/// as gravel through stone, without storing each voxel: the type is resolved on
/// demand from the voxel's own seed, so it costs nothing to store and comes out
/// the same every time it is asked. A packed node stores its voxels outright,
/// for the depths where neither of those holds.
pub struct Octree {
    octype: Octype,
}

impl Octree {
    /// A node that is one type throughout.
    pub fn new_uniform(id: VoxelTypeId) -> Self {
        Octree {
            octype: Octype::Uniform { id },
        }
    }

    /// A node that will have children, none of them allocated yet.
    pub fn new_children() -> Self {
        Octree {
            octype: Octype::Children { children: None },
        }
    }

    /// A node with all eight children given at once, kept together in a single
    /// allocation rather than eight.
    pub fn from_children(children: [Octree; 8]) -> Self {
        Self {
            octype: Octype::Children {
                children: Some(Box::new(children)),
            },
        }
    }

    /// This node's eight children, or `None` when it has none or is not a
    /// branch at all.
    pub fn children(&self) -> Option<&[Octree; 8]> {
        match &self.octype {
            Octype::Children { children } => children.as_deref(),
            _ => None,
        }
    }

    /// A node nothing has decided yet: the generator supplies it, not the tree.
    ///
    /// This is the tree's *undecided* state. It resolves to `None` and reports
    /// `false` from [`Octree::is_resolvable`] and [`Octree::is_decided`], so a
    /// caller that reaches one knows the region has no contents yet and must ask
    /// the generator. Writing into one is refused rather than guessed at: see
    /// [`Octree::set_voxel`].
    ///
    /// A branch whose children have not been allocated —
    /// [`Octree::new_children`] — is undecided in the same way and answers
    /// alike.
    pub fn new_procedural() -> Self {
        Self {
            octype: Octype::Procedural,
        }
    }

    /// A node holding several voxel types, chosen by relative weight.
    ///
    /// [`Weights`] has already established that these make a usable
    /// distribution, so all that is left here is to turn them into the running
    /// totals the pick walks.
    ///
    /// Returns `None` when the two lists disagree in length, since that is the
    /// one thing the weights alone cannot tell.
    pub fn new_probability(weights: &Weights, ids: Vec<VoxelTypeId>) -> Option<Self> {
        if weights.len() != ids.len() {
            return None;
        }

        Some(Octree {
            octype: Octype::Probability(Box::new(WeightedTypes {
                // On the `2^-63` grid once, at build time, so picking is an
                // integer comparison and the realised shares match the weights to
                // `2^-64` rather than to a 53-bit draw's `2^-53`.
                cumulative: weights
                    .cumulative()
                    .into_iter()
                    .map(Unit::from_probability)
                    .collect(),
                ids,
            })),
        })
    }

    /// A leaf storing every one of its voxels in a compact dense sequence.
    ///
    /// What to reach for at the depth where subdividing stops paying: the tree
    /// above keeps its sparseness, while this block chooses uniform, palette,
    /// run-length, or direct storage instead of paying for a node per voxel.
    /// The voxels are indexed by position within the node, in whatever order
    /// the caller fills them; the usual one is x fastest, then y, then z, matching
    /// [`Chunk`](crate::world::Chunk).
    ///
    /// Resolving one needs the position, so it answers through
    /// [`Octree::resolve_at`] rather than [`Octree::resolve`].
    pub fn new_packed(voxels: impl Into<CompactSequence<VoxelTypeId>>) -> Self {
        Self {
            octype: Octype::Packed(Box::new(voxels.into())),
        }
    }

    /// A packed node whose voxels are all one type, ready to be written into.
    ///
    /// This starts in the uniform representation and allocates no per-position
    /// indices until the first differing voxel is written.
    pub fn new_packed_uniform(count: usize, id: VoxelTypeId) -> Self {
        Self::new_packed(CompactSequence::filled(count, id))
    }

    /// The packed block this node holds, or `None` when it is not a packed
    /// node.
    pub fn packed(&self) -> Option<&CompactSequence<VoxelTypeId>> {
        match &self.octype {
            Octype::Packed(voxels) => Some(voxels),
            _ => None,
        }
    }

    /// The packed block, for writing into.
    ///
    /// The one way a node's contents change in place, and the reason a packed
    /// node is worth having: setting a voxel updates its compact leaf storage
    /// rather than subdividing down to a new node.
    pub fn packed_mut(&mut self) -> Option<&mut CompactSequence<VoxelTypeId>> {
        match &mut self.octype {
            Octype::Packed(voxels) => Some(voxels),
            _ => None,
        }
    }

    /// Whether this node holds a type for every point in it without descending:
    /// uniform, weighted or packed.
    pub fn is_resolvable(&self) -> bool {
        !matches!(self.octype, Octype::Children { .. } | Octype::Procedural)
    }

    /// The voxel type this node holds throughout, or `None` where it varies
    /// within the node or is not resolvable at all.
    ///
    /// `seed` decides the weighted case, so the same point in the same world
    /// always resolves the same way, whatever order nodes are visited in. It is
    /// read as a fraction and compared against the running totals in order,
    /// which is inverse-transform sampling over the node's types.
    ///
    /// Branches, generator nodes and packed nodes resolve to `None`: none of
    /// them holds one type for the whole node, and the caller has to descend,
    /// generate, or ask [`Octree::resolve_at`] for a particular position.
    pub fn resolve(&self, seed: Seed) -> Option<VoxelTypeId> {
        match &self.octype {
            Octype::Uniform { id } => Some(*id),
            Octype::Probability(weighted) => {
                let draw: Unit = seed.unit();

                weighted
                    .cumulative
                    .iter()
                    .position(|threshold| draw < *threshold)
                    .and_then(|index| weighted.ids.get(index).copied())
            }
            Octype::Packed(_) | Octype::Children { .. } | Octype::Procedural => None,
        }
    }

    /// The voxel type at one position within this node, or `None` where the
    /// node cannot answer for itself.
    ///
    /// The general form of [`Octree::resolve`]: a uniform node ignores the
    /// position, a weighted one takes its draw from the seed as before, and a
    /// packed one reads the position out of its block. Branch and generator
    /// nodes still answer `None`, since the caller has to descend or generate.
    ///
    /// `position` is the index within the node's own block, not a world
    /// coordinate; the caller works it out from the node's extent, which the
    /// node does not know.
    pub fn resolve_at(&self, position: usize, seed: Seed) -> Option<VoxelTypeId> {
        match &self.octype {
            Octype::Packed(voxels) => voxels.get(position).copied(),
            _ => self.resolve(seed),
        }
    }

    /// The type this node holds throughout, when it is a plain uniform node.
    ///
    /// Narrower than [`Octree::resolve`]: a weighted node needs a seed and a
    /// packed one needs a position, so neither answers here.
    pub fn uniform_id(&self) -> Option<VoxelTypeId> {
        match self.octype {
            Octype::Uniform { id } => Some(id),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Reading and writing by position
// ---------------------------------------------------------------------------

/// A node's own coordinate space.
///
/// `levels` is how many times a node may still be halved: a node spanning
/// `levels` is `1 << levels` voxels along each axis, so `levels == 0` is one
/// voxel. A node does not know its own size — the tree above it does — so every
/// call below is told.
///
/// Positions are local to the node, x fastest then y then z, the order
/// [`Chunk`](crate::world::Chunk) uses and the order
/// [`Octree::new_packed`] expects.
impl Octree {
    /// The deepest `levels` a node can span, limited by a `u32` coordinate.
    pub const MAX_LEVELS: u8 = 31;

    /// The voxel type at one position, descending as far as it must.
    ///
    /// `None` where the region is undecided: a [procedural](Octree::new_procedural)
    /// node, a branch with no children, or a position outside the node.
    ///
    /// `seed` decides a weighted node, and should be the seed of the *voxel*
    /// rather than of the tree, so that neighbouring voxels in one weighted
    /// region differ. The caller derives it, since only the caller knows the
    /// world coordinate — `world_seed.at(position.to_array())` is the usual
    /// form.
    pub fn voxel(&self, local: [u32; 3], levels: u8, seed: Seed) -> Option<VoxelTypeId> {
        if !Self::within(local, levels) {
            return None;
        }

        match &self.octype {
            Octype::Children { children } => {
                let children: &[Octree; CHILDREN] = children.as_deref()?;
                let (index, inner) = Self::descend(local, levels)?;

                children[index].voxel(inner, levels - 1, seed)
            }
            Octype::Packed(voxels) => voxels.get(Self::packed_index(local, levels)?).copied(),
            _ => self.resolve(seed),
        }
    }

    /// Whether anything has decided the voxel at this position.
    ///
    /// The check that comes before a read or a write: `false` means the region
    /// is undecided and the generator has not run, so [`Octree::voxel`] would
    /// answer `None` and [`Octree::set_voxel`] would refuse.
    pub fn is_decided(&self, local: [u32; 3], levels: u8) -> bool {
        if !Self::within(local, levels) {
            return false;
        }

        match &self.octype {
            Octype::Children { children } => match children.as_deref() {
                Some(children) => match Self::descend(local, levels) {
                    Some((index, inner)) => children[index].is_decided(inner, levels - 1),
                    None => false,
                },
                None => false,
            },
            Octype::Procedural => false,
            _ => true,
        }
    }

    /// Sets the voxel at one position, giving the node per-voxel storage if it
    /// had none.
    ///
    /// The single-voxel case of [`Octree::set_region`], which is where the
    /// behaviour is described.
    pub fn set_voxel(&mut self, local: [u32; 3], levels: u8, pack_at: u8, id: VoxelTypeId) -> bool {
        self.set_region(local, levels, 0, pack_at, id)
    }

    /// Sets a whole region to one type, without touching its voxels one by one.
    ///
    /// Returns whether it was written.
    ///
    /// # Addressing a region
    ///
    /// `region_levels` is the size of the region: it spans `1 << region_levels`
    /// voxels along each axis, so zero is a single voxel and `levels` is the
    /// whole node. A region is always aligned to its own size, so `local`
    /// counts *in regions*, not in voxels, and runs `0..1 << (levels -
    /// region_levels)` along each axis.
    ///
    /// That is the same position shifted right by `region_levels`: the deepest
    /// level uses every bit of a voxel coordinate, and each level above it uses
    /// one fewer. [`Octree::region_of`] does the shift, and
    /// [`Octree::region_origin`] undoes it.
    ///
    /// ```text
    /// levels = 6   (a 64-voxel cube)
    ///
    /// region_levels = 0   local in 0..64   one voxel
    /// region_levels = 2   local in 0..16   a 4-voxel cube
    /// region_levels = 4   local in 0..4    a 16-voxel cube
    /// region_levels = 6   local == [0,0,0] the whole node
    /// ```
    ///
    /// # What a node becomes
    ///
    /// Setting a region never needs per-voxel storage for the region itself: the
    /// node covering it simply becomes uniform. Only the nodes *above* it have
    /// to change, and they change as they do for a single voxel:
    ///
    /// - a region filling the whole node replaces it outright, dropping whatever
    ///   storage it had.
    /// - above `pack_at` levels, a uniform node becomes eight children of its
    ///   own type and the write descends into one of them.
    /// - at or below `pack_at`, it becomes a [packed](Octree::new_packed) block,
    ///   and the region is filled as a range of it — one run per row, since a
    ///   cube is not contiguous in a linear block.
    ///
    /// Eight children that end up uniform and agreeing collapse back into one,
    /// so filling a region in pieces leaves the same tree as filling it at once.
    ///
    /// # What is refused
    ///
    /// `false`, with nothing changed, for a region larger than the node, a
    /// position outside it, and the nodes that [`Octree::set_voxel`] refuses:
    /// [procedural](Octree::new_procedural) ones, branches with no children, and
    /// [weighted](Octree::new_probability) ones.
    pub fn set_region(
        &mut self,
        local: [u32; 3],
        levels: u8,
        region_levels: u8,
        pack_at: u8,
        id: VoxelTypeId,
    ) -> bool {
        let Some(steps) = levels.checked_sub(region_levels) else {
            return false;
        };

        if !Self::within(local, steps) {
            return false;
        }

        // The region is the whole node, so nothing below matters.
        if steps == 0 {
            self.fill(id);

            return true;
        }

        match &self.octype {
            Octype::Procedural | Octype::Probability(_) | Octype::Children { children: None } => {
                return false;
            }
            _ => {}
        }

        if let Octype::Uniform { id: held } = self.octype {
            self.octype = if levels <= pack_at {
                match Self::volume(levels) {
                    Some(volume) => Octype::Packed(Box::new(CompactSequence::filled(volume, held))),
                    None => return false,
                }
            } else {
                Octype::Children {
                    children: Some(Box::new(std::array::from_fn(|_| Self::new_uniform(held)))),
                }
            };
        }

        match &mut self.octype {
            Octype::Packed(voxels) => Self::fill_packed(voxels, local, levels, region_levels, id),
            Octype::Children {
                children: Some(children),
            } => {
                let Some((index, inner)) = Self::descend(local, steps) else {
                    return false;
                };

                let written: bool =
                    children[index].set_region(inner, levels - 1, region_levels, pack_at, id);

                if written {
                    self.collapse();
                }

                written
            }
            _ => false,
        }
    }

    /// Fills a region's rows inside a packed block.
    ///
    /// A cube is not contiguous in a linear block, so this fills one run per
    /// row: the voxels along x are adjacent, and each `(y, z)` pair starts a new
    /// run.
    fn fill_packed(
        voxels: &mut CompactSequence<VoxelTypeId>,
        local: [u32; 3],
        levels: u8,
        region_levels: u8,
        id: VoxelTypeId,
    ) -> bool {
        let [x, y, z] = Self::region_origin(local, region_levels);
        let span: u32 = 1 << region_levels;

        for depth in z..z + span {
            for row in y..y + span {
                let Some(start) = Self::packed_index([x, row, depth], levels) else {
                    return false;
                };

                voxels.fill(start, start + span as usize, id);
            }
        }

        true
    }

    /// The region of a given size that holds a voxel, as
    /// [`Octree::set_region`] addresses it.
    ///
    /// The shift itself: a coarser region uses fewer bits of the coordinate.
    pub fn region_of(local: [u32; 3], region_levels: u8) -> [u32; 3] {
        local.map(|coordinate| coordinate >> region_levels)
    }

    /// The voxel at the low corner of a region, which undoes
    /// [`Octree::region_of`].
    pub fn region_origin(region: [u32; 3], region_levels: u8) -> [u32; 3] {
        region.map(|coordinate| coordinate << region_levels)
    }

    /// The one type a whole region holds, or `None` where it varies or is
    /// undecided.
    ///
    /// The read counterpart to [`Octree::set_region`], and cheaper than reading
    /// every voxel to find out: it descends to the node covering the region and
    /// asks whether that node answers for itself. A region smaller than the node
    /// it lands in inherits that node's answer, since a uniform node is uniform
    /// throughout.
    ///
    /// `None` where the region spans a branch, a packed block, or anything
    /// undecided — in which case the voxels have to be read individually.
    pub fn region_type(
        &self,
        local: [u32; 3],
        levels: u8,
        region_levels: u8,
        seed: Seed,
    ) -> Option<VoxelTypeId> {
        let steps: u8 = levels.checked_sub(region_levels)?;

        if !Self::within(local, steps) {
            return None;
        }

        if steps == 0 {
            return self.resolve(seed);
        }

        match &self.octype {
            Octype::Children { children } => {
                let children: &[Octree; CHILDREN] = children.as_deref()?;
                let (index, inner) = Self::descend(local, steps)?;

                children[index].region_type(inner, levels - 1, region_levels, seed)
            }
            // A packed block varies in general, and proving otherwise would mean
            // reading all of it.
            Octype::Packed(_) => None,
            _ => self.resolve(seed),
        }
    }

    /// Replaces everything this node holds with one type.
    ///
    /// The cheap way to clear a region: whatever the node was — a branch, a
    /// packed block, undecided — it becomes one uniform node, and its storage
    /// goes with it.
    pub fn fill(&mut self, id: VoxelTypeId) {
        self.octype = Octype::Uniform { id };
    }

    /// Turns a weighted node into a packed block by drawing every voxel, so
    /// that it can be written into.
    ///
    /// `seed_of` supplies the seed for each local position; it must be derived
    /// from the world coordinate, or every voxel in the node draws alike. The
    /// values are the same ones [`Octree::voxel`] would have answered, so this
    /// changes how the node is stored and not what it holds.
    ///
    /// Returns whether anything was converted: `false` for a node that is not
    /// weighted, and for one too large to pack.
    pub fn materialise(&mut self, levels: u8, seed_of: impl Fn([u32; 3]) -> Seed) -> bool {
        if !matches!(self.octype, Octype::Probability(_)) {
            return false;
        }

        let Some(volume) = Self::volume(levels) else {
            return false;
        };

        let edge: u32 = 1 << levels;
        let mut voxels: Vec<VoxelTypeId> = Vec::with_capacity(volume);

        // x fastest, then y, then z, matching `packed_index`.
        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    let local: [u32; 3] = [x, y, z];

                    match self.resolve(seed_of(local)) {
                        Some(id) => voxels.push(id),
                        None => return false,
                    }
                }
            }
        }

        self.octype = Octype::Packed(Box::new(CompactSequence::from(voxels)));

        true
    }

    /// Collapses eight uniform children of one type back into one uniform node.
    ///
    /// Returns whether it collapsed. Called after every descending write, since
    /// filling a region in voxel by voxel would otherwise leave a tree of
    /// agreeing children where one node would do.
    ///
    /// Only branches collapse. A packed block that has become uniform keeps its
    /// storage, because noticing would mean reading every voxel on every write;
    /// [`CompactSequence::compact`] is the deliberate way to reclaim it.
    pub fn collapse(&mut self) -> bool {
        let Octype::Children {
            children: Some(children),
        } = &self.octype
        else {
            return false;
        };

        let Some(id) = children[0].uniform_id() else {
            return false;
        };

        if !children.iter().all(|child| child.uniform_id() == Some(id)) {
            return false;
        }

        self.octype = Octype::Uniform { id };

        true
    }

    /// Whether a position lies inside a node spanning `levels`.
    fn within(local: [u32; 3], levels: u8) -> bool {
        if levels > Self::MAX_LEVELS {
            return false;
        }

        let edge: u32 = 1 << levels;

        local.iter().all(|coordinate| *coordinate < edge)
    }

    /// Which child holds this position, and where the position sits inside it.
    ///
    /// The child is picked by the top bit of each coordinate, x in bit 0, y in
    /// bit 1 and z in bit 2, so children run x fastest. The position keeps the
    /// bits below it. `None` for a node that cannot be halved.
    fn descend(local: [u32; 3], levels: u8) -> Option<(usize, [u32; 3])> {
        if levels == 0 {
            return None;
        }

        let shift: u32 = u32::from(levels) - 1;
        let mask: u32 = (1 << shift) - 1;

        let index: usize = (0..3)
            .map(|axis| (((local[axis] >> shift) & 1) as usize) << axis)
            .sum();

        Some((index, [local[0] & mask, local[1] & mask, local[2] & mask]))
    }

    /// Where a position sits in a packed block spanning `levels`.
    fn packed_index(local: [u32; 3], levels: u8) -> Option<usize> {
        let edge: usize = 1usize.checked_shl(u32::from(levels))?;
        let [x, y, z] = local.map(|coordinate| coordinate as usize);

        Some(x + edge * (y + edge * z))
    }

    /// How many voxels a node spanning `levels` holds, or `None` when that
    /// many would not fit in a `usize`.
    fn volume(levels: u8) -> Option<usize> {
        1usize.checked_shl(u32::from(levels).checked_mul(3)?)
    }
}
