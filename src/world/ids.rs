//! The kinds of thing the world hands out identifiers for.
//!
//! Every kind declared here becomes its own type, so a voxel type id cannot be
//! passed where an entity id belongs even though both are numbers underneath.
//! See [`crate::units::identifier`] for what that buys and what [`AnyId`]
//! is for.
//!
//! [`AnyId`]: crate::units::AnyId
//!
//! A kind's tag comes from its name, so new kinds can be added anywhere in this
//! list, or declared in another module entirely, without disturbing the ones
//! already here. Renaming a kind does change its tag, so once ids are in a save
//! file the names are part of the format.

use crate::units::Id;

crate::define_id_kinds! {
    VoxelTypeKind => "voxel type",
}

/// Identifies an entry in the world's [`VoxelTypeIndex`].
///
/// [`VoxelTypeIndex`]: crate::world::voxel_type_index::VoxelTypeIndex
pub type VoxelTypeId = Id<VoxelTypeKind>;
