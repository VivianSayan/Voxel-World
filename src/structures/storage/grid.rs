//! Dense and bit-packed rectangular grids with checked multidimensional
//! indexing.
//!
//! [`Grid<T, N>`] stores one `T` per cell. [`BitGrid<N>`] represents boolean
//! cells with one bit per set position. The const parameter `N` is the number
//! of axes; the common [`Grid2`], [`Grid3`], [`Grid4`], [`BitGrid2`],
//! [`BitGrid3`], and [`BitGrid4`] aliases give descriptive names to the usual
//! dimensions.
//!
//! # Memory layout
//!
//! Values are stored in one flat allocation. Axis zero changes fastest, then
//! axis one, and so on. For a 3D coordinate `[x, y, z]` and dimensions
//! `[width, height, depth]`, the flat index is:
//!
//! ```text
//! x + width * (y + height * z)
//! ```
//!
//! The same rule extends to every `N`. This makes `as_slice()` suitable for
//! serialization, upload buffers, and loops that do not need coordinates.
//!
//! # Checked construction and access
//!
//! Constructors return `None` if multiplying the axis lengths would overflow
//! `usize`; [`Grid::from_vec`] also requires exactly one value per cell. Access
//! methods return `None` for any coordinate outside its corresponding axis.
//! A zero-length axis gives the grid zero cells, so every coordinate is out of
//! range.
//!
//! # Examples
//!
//! ```
//! use voxel_world::structures::storage::{BitGrid3, Grid2, Grid4};
//!
//! let mut image = Grid2::filled([3, 2], 0u8).unwrap();
//! *image.get_mut([2, 1]).unwrap() = 9;
//! assert_eq!(image.index_of([2, 1]), Some(5));
//! assert_eq!(image.as_slice(), &[0, 0, 0, 0, 0, 9]);
//!
//! let samples = Grid4::from_vec([2, 1, 1, 2], vec![10, 11, 12, 13]).unwrap();
//! assert_eq!(samples.get([1, 0, 0, 1]), Some(&13));
//!
//! let mut occupied = BitGrid3::new([16, 16, 16]).unwrap();
//! occupied.set([3, 4, 5], true);
//! assert_eq!(occupied.get([3, 4, 5]), Some(true));
//! assert_eq!(occupied.count_ones(), 1);
//! ```

use crate::structures::collections::BitSet;

fn flatten_index<const N: usize>(size: &[usize; N], position: [usize; N]) -> Option<usize> {
    let mut index = 0usize;
    let mut stride = 1usize;

    for (&coordinate, &side) in position.iter().zip(size) {
        if coordinate >= side {
            return None;
        }
        index = index.checked_add(coordinate.checked_mul(stride)?)?;
        stride = stride.checked_mul(side)?;
    }

    Some(index)
}

/// A dense rectangular grid with `N` axes and one stored value per cell.
///
/// Axis zero changes fastest. A `[x, y, z]` grid therefore uses x-major
/// storage, and `[x, y, z, w]` adds w as the outermost axis.
///
/// The dimensions and backing vector are kept consistent by construction.
/// Coordinate access is O(N), which is a small fixed number for the provided
/// aliases, while flat slice access has the same cost as a `Vec`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Grid<T, const N: usize> {
    size: [usize; N],
    values: Vec<T>,
}

impl<T, const N: usize> Grid<T, N> {
    /// Builds a grid from flat axis-zero-major storage.
    ///
    /// Returns `None` if the product of the axis lengths overflows `usize` or
    /// differs from `values.len()`. An empty vector is valid exactly when at
    /// least one axis has length zero.
    pub fn from_vec(size: [usize; N], values: Vec<T>) -> Option<Self> {
        (Self::volume_of(size)? == values.len()).then_some(Self { size, values })
    }

    /// Returns the length of every axis in coordinate order.
    ///
    /// For [`Grid3`], this is `[width, height, depth]`; for [`Grid4`] it is
    /// `[width, height, depth, frames]` or whichever meaning the caller assigns
    /// to the fourth axis.
    pub const fn size(&self) -> [usize; N] {
        self.size
    }

    /// Returns the total number of cells, equal to the product of [`size`](Self::size).
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the grid has zero cells.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Borrows all cells as flat storage with axis zero changing fastest.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    /// Mutably borrows all cells as flat storage with axis zero changing
    /// fastest.
    ///
    /// The slice length cannot change, so the grid's dimensional invariant is
    /// preserved.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Consumes the grid and returns its flat backing vector.
    pub fn into_vec(self) -> Vec<T> {
        self.values
    }

    /// Converts a coordinate into its flat axis-zero-major index.
    ///
    /// Returns `None` when any coordinate is greater than or equal to the
    /// corresponding axis length. Arithmetic remains checked even though a
    /// successfully constructed grid has already established a valid volume.
    pub fn index_of(&self, position: [usize; N]) -> Option<usize> {
        flatten_index(&self.size, position)
    }

    /// Borrows the value at a coordinate, or returns `None` outside the grid.
    pub fn get(&self, position: [usize; N]) -> Option<&T> {
        self.values.get(self.index_of(position)?)
    }

    /// Mutably borrows the value at a coordinate, or returns `None` outside
    /// the grid.
    pub fn get_mut(&mut self, position: [usize; N]) -> Option<&mut T> {
        let index = self.index_of(position)?;
        self.values.get_mut(index)
    }

    fn volume_of(size: [usize; N]) -> Option<usize> {
        size.into_iter()
            .try_fold(1usize, |volume, side| volume.checked_mul(side))
    }
}

impl<T: Clone, const N: usize> Grid<T, N> {
    /// Builds a grid whose cells are all clones of `value`.
    ///
    /// Returns `None` if the product of the dimensions overflows `usize`.
    /// Otherwise this allocates exactly one `T` per cell and is O(volume).
    pub fn filled(size: [usize; N], value: T) -> Option<Self> {
        Some(Self {
            size,
            values: vec![value; Self::volume_of(size)?],
        })
    }
}

/// A rectangular boolean grid that stores set cells in a [`BitSet`].
///
/// Clear cells require no individual allocation; the bit storage grows only as
/// high as the greatest set flat index. This is particularly useful for sparse
/// occupancy, visibility, masks, and visited-state grids. Coordinates and flat
/// indices use exactly the same axis-zero-major layout as [`Grid`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitGrid<const N: usize> {
    size: [usize; N],
    volume: usize,
    bits: BitSet,
}

impl<const N: usize> BitGrid<N> {
    /// Creates a grid with every cell clear.
    ///
    /// Returns `None` if the product of the axis lengths overflows `usize`.
    /// No per-cell bit storage is allocated until a cell is set.
    pub fn new(size: [usize; N]) -> Option<Self> {
        Some(Self {
            size,
            volume: Grid::<(), N>::volume_of(size)?,
            bits: BitSet::new(),
        })
    }

    /// Returns the length of every axis in coordinate order.
    pub const fn size(&self) -> [usize; N] {
        self.size
    }

    /// Returns the total number of addressable cells, set or clear.
    pub const fn len(&self) -> usize {
        self.volume
    }

    /// Returns whether the grid has zero addressable cells.
    ///
    /// This is different from [`count_ones`](Self::count_ones) being zero: a
    /// non-empty grid starts with every bit clear.
    pub const fn is_empty(&self) -> bool {
        self.volume == 0
    }

    /// Reads a cell, returning `None` when the coordinate is outside the grid.
    ///
    /// A valid cell that has never been set returns `Some(false)`.
    pub fn get(&self, position: [usize; N]) -> Option<bool> {
        Some(self.bits.contains(self.index_of(position)?))
    }

    /// Sets or clears one cell.
    ///
    /// Returns its previous boolean value, or `None` when the coordinate is
    /// outside the grid. Setting a high flat index may grow the underlying bit
    /// storage; clearing trailing high bits does not shrink allocation.
    pub fn set(&mut self, position: [usize; N], value: bool) -> Option<bool> {
        let index = self.index_of(position)?;
        let previous = self.bits.contains(index);
        if value {
            self.bits.insert(index);
        } else {
            self.bits.remove(index);
        }
        Some(previous)
    }

    /// Returns the number of cells currently set to `true`.
    ///
    /// This is O(1), maintained by [`BitSet`] as cells change.
    pub fn count_ones(&self) -> usize {
        self.bits.len()
    }

    /// Converts a coordinate into its flat axis-zero-major bit index.
    ///
    /// Returns `None` when any coordinate is outside its axis.
    pub fn index_of(&self, position: [usize; N]) -> Option<usize> {
        flatten_index(&self.size, position)
    }

    /// Clears every set cell while retaining the allocated bit storage.
    ///
    /// Dimensions and [`len`](Self::len) are unchanged.
    pub fn clear(&mut self) {
        self.bits.clear();
    }
}

/// A dense two-dimensional grid, indexed as `[x, y]` with x changing fastest.
pub type Grid2<T> = Grid<T, 2>;
/// A dense three-dimensional grid, indexed as `[x, y, z]` with x changing
/// fastest.
pub type Grid3<T> = Grid<T, 3>;
/// A dense four-dimensional grid, indexed as `[x, y, z, w]` with x changing
/// fastest and w slowest.
pub type Grid4<T> = Grid<T, 4>;

/// A bit-packed two-dimensional boolean grid indexed as `[x, y]`.
pub type BitGrid2 = BitGrid<2>;
/// A bit-packed three-dimensional boolean grid indexed as `[x, y, z]`.
pub type BitGrid3 = BitGrid<3>;
/// A bit-packed four-dimensional boolean grid indexed as `[x, y, z, w]`.
pub type BitGrid4 = BitGrid<4>;
