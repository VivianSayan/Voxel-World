//! Structures that own their data and hand out a way of reaching it.
//!
//! The rest of the module is about relationships between values that already
//! exist somewhere. These two are about the values themselves: where they live,
//! how they are addressed, and how little room they can be made to take.
//!
//! - [`SlotMap`] stores values at reusable slots and gives
//!   out [`Id`](crate::units::Id)s that stop working when the value is removed,
//!   so a stale handle is caught rather than silently pointing at whatever took
//!   the slot.
//! - [`Palette`] stores a large run of values that repeat by
//!   keeping each distinct one once and packing indices into it, which is how a
//!   chunk of voxels fits in a fraction of the space one word each would take.
//! - [`CompactSequence`] chooses among uniform, palette, run-length and dense
//!   representations after a batch of changes.
//! - [`Grid`]/[`BitGrid`] and their 2D, 3D and 4D aliases centralize checked
//!   coordinate indexing.

pub mod compact_sequence;
pub mod grid;
pub mod palette;
pub mod slot_map;

pub use compact_sequence::{CompactKind, CompactSequence};
pub use grid::{BitGrid, BitGrid2, BitGrid3, BitGrid4, Grid, Grid2, Grid3, Grid4};
pub use palette::Palette;
pub use slot_map::SlotMap;
