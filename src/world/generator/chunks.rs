use crate::math::Vector3;
use crate::random::seed::Seed;
use crate::spatial::VoxelPosition3;
use crate::units::{Unit, Weights};
use crate::world::{Chunk, VoxelTypeId, VoxelTypeIndex};

/// A minimal weighted voxel generator: each voxel is drawn independently from
/// a fixed palette of types and weights.
///
/// The palette is sorted by registered name and the weights turned into running
/// totals once, at construction. Sorting by name is what makes the world
/// independent of the registry: two runs that declared the same types in a
/// different order, and so handed out different ids, still generate the same
/// world from the same seed.
///
/// Holds no random stream. One generator can serve every chunk, on any thread,
/// and generating a chunk changes nothing.
pub struct ChunkGenerator {
    seed: Seed,
    ids: Vec<VoxelTypeId>,
    cumulative: Vec<Unit>,
}

impl ChunkGenerator {
    /// A generator over a palette of types and their relative weights.
    ///
    /// `None` when a palette entry names a type the registry does not hold,
    /// when two entries name the same type, or when the weights do not make a
    /// usable distribution. The seed is branched once here, so a world's
    /// material choices are decorrelated from everything else derived from the
    /// same world seed.
    pub fn new(
        seed: Seed,
        registry: &VoxelTypeIndex,
        palette: &[(VoxelTypeId, f64)],
    ) -> Option<Self> {
        let mut entries = palette
            .iter()
            .map(|&(id, weight)| Some((registry.get_voxel_type(id)?.name(), id, weight)))
            .collect::<Option<Vec<_>>>()?;
        entries.sort_by(|left, right| left.0.cmp(right.0));
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return None;
        }
        let weights = Weights::new(entries.iter().map(|entry| entry.2))?;
        Some(Self {
            seed: seed.child("voxel-material-v1"),
            ids: entries.iter().map(|entry| entry.1).collect(),
            cumulative: weights
                .cumulative()
                .into_iter()
                .map(Unit::from_probability)
                .collect(),
        })
    }

    /// The type at one global voxel coordinate.
    ///
    /// The voxel's own seed is read as a fraction and located among the running
    /// totals by binary search, which is inverse-transform sampling over the
    /// palette. Depends on nothing but the seed, the coordinate and the
    /// palette, so neighbouring chunks agree at their shared faces without
    /// consulting one another.
    pub fn voxel(&self, position: VoxelPosition3) -> VoxelTypeId {
        let unit: Unit = self.seed.at_voxel(position).unit();
        let index = self
            .cumulative
            .partition_point(|threshold| *threshold <= unit);
        self.ids[index]
    }

    /// Fills one dense chunk whose lowest corner is `origin`, x fastest then y
    /// then z.
    ///
    /// `None` when the chunk's far corner would run past the end of the
    /// coordinate range, which is checked before any work is done.
    pub fn generate(&self, origin: VoxelPosition3) -> Option<Chunk> {
        let start = origin.coordinates();
        for coordinate in start.to_array() {
            coordinate.checked_add(Chunk::EDGE as i128 - 1)?;
        }
        let mut voxels = Vec::with_capacity(Chunk::VOLUME);
        for z in 0..Chunk::EDGE {
            for y in 0..Chunk::EDGE {
                for x in 0..Chunk::EDGE {
                    let position =
                        VoxelPosition3::new(start + Vector3::new(x as i128, y as i128, z as i128));
                    voxels.push(self.voxel(position));
                }
            }
        }
        Some(Chunk {
            origin,
            voxels: crate::structures::storage::Grid3::from_vec([Chunk::EDGE; 3], voxels)
                .expect("chunk dimensions match its generated volume"),
        })
    }
}
