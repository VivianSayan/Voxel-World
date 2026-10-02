// Positions a seed can be derived at
// ---------------------------------------------------------------------------

/// A place a [`Seed`] can be derived for.
///
/// Implemented for every [`VoxelPosition`](crate::spatial::VoxelPosition3) and
/// [`NodePosition`](crate::spatial::NodePosition3), in two, three and four
/// dimensions, so [`Seed::at_position`] reads the same whatever shape of position is
/// to hand.
///
/// # Why the position folds itself
///
/// The obvious signature would hand back the coordinates for `Seed` to absorb, but
/// the shapes differ — three coordinates here, four there, and a depth on some of
/// them — and a trait cannot return differently sized arrays without either
/// allocating or reaching for const generics that do not compile on stable. Letting
/// each position fold *itself* into the seed sidesteps all of it: no allocation, and
/// each type states its own convention in one place.
pub trait SeedablePosition {
    /// Folds this position into `seed`.
    fn derive(&self, seed: Seed) -> Seed;
}

/// Generates the impls, which differ only in dimension and in whether a depth comes
/// along with the address.
macro_rules! seedable_positions {
    (voxel: $($voxel:ty),*; node: $($node:ty),*) => {
        $(
            /// Folds the coordinates. The dimension count goes in too, so a voxel at
            /// `(1, 2)` is not the same place as one at `(1, 2, 0)`.
            impl SeedablePosition for $voxel {
                fn derive(&self, seed: Seed) -> Seed {
                    seed.at(self.to_array())
                }
            }
        )*
        $(
            /// Folds the address and then the depth, so one address at two depths
            /// gives two places. Matches what [`Seed::at_depth`] did with the two
            /// passed separately.
            impl SeedablePosition for $node {
                fn derive(&self, seed: Seed) -> Seed {
                    seed.at(self.to_array()).at_level(self.depth())
                }
            }
        )*
    };
}

seedable_positions! {
    voxel: VoxelPosition2, VoxelPosition3, VoxelPosition4;
    node: NodePosition2, NodePosition3, NodePosition4
}
