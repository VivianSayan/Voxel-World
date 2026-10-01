//! Storage shared by `SparseSequence` and `SparseSetSequence`: slots at
//! arbitrary integer indices, a reverse index from element to the slots
//! holding it, and a run-length record of occupied indices so the first
//! free index is found in O(log n).

use crate::structures::collections::sets::set::Set;
use crate::structures::hashing::FastHashMap;
use crate::structures::traits::Element;
use std::collections::{BTreeMap, BTreeSet};

/// What a slot holds, as far as the reverse index is concerned.
///
/// The two sparse collections differ only in this: one slot holds a single
/// element, the other a whole set. Everything else about them, including the
/// element-to-indices index, is written once against this trait.
pub trait SlotContents<T> {
    /// Calls `visit` once for every element in the slot, in whatever order the
    /// slot keeps them.
    fn for_each_element(&self, visit: impl FnMut(&T));
}

/// A slot holding exactly one element, which is what a sparse sequence stores
/// at each index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct One<T>(pub T);

impl<T> SlotContents<T> for One<T> {
    fn for_each_element(&self, mut visit: impl FnMut(&T)) {
        visit(&self.0);
    }
}

impl<T: Element> SlotContents<T> for Set<T> {
    fn for_each_element(&self, visit: impl FnMut(&T)) {
        self.iter().for_each(visit);
    }
}

/// Occupied indices as maximal runs `start -> end`, both inclusive.
///
/// A sparse collection may be occupied anywhere in the `i64` range, so the free
/// indices cannot be listed. Storing the occupied ones as runs instead keeps
/// the map small when they are contiguous, which is the usual case, and makes
/// [`IndexRuns::first_free`] one lookup rather than a walk.
///
/// Runs never touch: inserting an index that closes the gap between two merges
/// them, and removing one from the middle of a run splits it.
#[derive(Clone, Debug, Default)]
pub struct IndexRuns {
    runs: BTreeMap<i64, i64>,
}

impl IndexRuns {
    /// The run covering `index`, if any: the last run starting at or before
    /// it, when that run also reaches it.
    fn run_containing(&self, index: i64) -> Option<(i64, i64)> {
        let (&start, &end) = self.runs.range(..=index).next_back()?;
        (end >= index).then_some((start, end))
    }

    /// Marks an index occupied, merging with the runs on either side where
    /// they meet it. Does nothing if it is occupied already.
    pub fn insert(&mut self, index: i64) {
        if self.run_containing(index).is_some() {
            return;
        }
        let mut start = index;
        let mut end = index;
        if let Some((before_start, _)) = index
            .checked_sub(1)
            .and_then(|before| self.run_containing(before))
        {
            start = before_start;
        }
        if let Some(after_end) = index
            .checked_add(1)
            .and_then(|after| self.runs.remove(&after))
        {
            end = after_end;
        }
        self.runs.insert(start, end);
    }

    /// Marks an index free, splitting its run in two if it was in the middle
    /// of one. Does nothing if it was free already.
    pub fn remove(&mut self, index: i64) {
        let Some((start, end)) = self.run_containing(index) else {
            return;
        };
        self.runs.remove(&start);
        if start < index {
            self.runs.insert(start, index - 1);
        }
        if index < end {
            self.runs.insert(index + 1, end);
        }
    }

    /// The smallest unoccupied index that is zero or above.
    ///
    /// Zero when nothing occupies it, and otherwise one past the end of the run
    /// that starts there, since runs never touch. Panics only if every
    /// non-negative index is occupied, which needs `2^63` slots.
    pub fn first_free(&self) -> i64 {
        self.run_containing(0).map_or(0, |(_, end)| {
            end.checked_add(1)
                .expect("no non-negative sparse index remains")
        })
    }

    /// Forgets every run, leaving nothing occupied.
    pub fn clear(&mut self) {
        self.runs.clear();
    }
}

/// The storage behind both sparse collections: slots at arbitrary `i64`
/// indices, an index from element to the slots holding it, and the run record
/// of which indices are occupied.
///
/// The three are kept in step by going through this type: [`SparseCore::put`]
/// and [`SparseCore::take`] update all of them, and the operations that move
/// slots about en masse call [`SparseCore::rebuild`] afterwards rather than
/// trying to patch the index.
///
/// The fields are public because the two collections built on it read the slots
/// directly; the run record is not, since it is only correct when maintained
/// here.
#[derive(Clone, Debug)]
pub struct SparseCore<T, S> {
    pub slots: BTreeMap<i64, S>,
    pub positions: FastHashMap<T, BTreeSet<i64>>,
    runs: IndexRuns,
}

impl<T, S> Default for SparseCore<T, S> {
    fn default() -> Self {
        Self {
            slots: BTreeMap::new(),
            positions: FastHashMap::default(),
            runs: IndexRuns::default(),
        }
    }
}

impl<T: Element, S: SlotContents<T>> SparseCore<T, S> {
    /// The smallest unoccupied index that is zero or above, in O(log n).
    pub fn first_free(&self) -> i64 {
        self.runs.first_free()
    }

    /// The lowest occupied index, or `None` when nothing is stored.
    pub fn min_index(&self) -> Option<i64> {
        self.slots.keys().next().copied()
    }

    /// The highest occupied index, or `None` when nothing is stored.
    pub fn max_index(&self) -> Option<i64> {
        self.slots.keys().next_back().copied()
    }

    /// Records that `item` appears at `index`, for a caller that changed a
    /// slot's contents in place rather than replacing the slot.
    pub fn index_element(&mut self, item: &T, index: i64) {
        match self.positions.get_mut(item) {
            Some(indices) => {
                indices.insert(index);
            }
            None => {
                self.positions.insert(item.clone(), BTreeSet::from([index]));
            }
        }
    }

    /// Records that `item` no longer appears at `index`, dropping the item
    /// from the index entirely once it appears nowhere.
    pub fn unindex_element(&mut self, item: &T, index: i64) {
        if let Some(indices) = self.positions.get_mut(item) {
            indices.remove(&index);
            if indices.is_empty() {
                self.positions.remove(item);
            }
        }
    }

    /// Stores a slot at an index, returning the one it replaced.
    ///
    /// Removes the old slot first, so its elements leave the index, then adds
    /// the new slot's elements and marks the index occupied.
    pub fn put(&mut self, index: i64, slot: S) -> Option<S> {
        let old = self.take(index);
        let Self { positions, .. } = self;
        slot.for_each_element(|item| match positions.get_mut(item) {
            Some(indices) => {
                indices.insert(index);
            }
            None => {
                positions.insert(item.clone(), BTreeSet::from([index]));
            }
        });
        self.slots.insert(index, slot);
        self.runs.insert(index);
        old
    }

    /// Removes the slot at an index and returns it, taking its elements out of
    /// the index and marking the index free. `None` if nothing was there.
    pub fn take(&mut self, index: i64) -> Option<S> {
        let slot = self.slots.remove(&index)?;
        let Self { positions, .. } = self;
        slot.for_each_element(|item| {
            if let Some(indices) = positions.get_mut(item) {
                indices.remove(&index);
                if indices.is_empty() {
                    positions.remove(item);
                }
            }
        });
        self.runs.remove(index);
        Some(slot)
    }

    /// Marks an index occupied after a slot was created in place.
    pub fn occupy(&mut self, index: i64) {
        self.runs.insert(index);
    }

    /// Marks an index free after its slot was removed in place.
    pub fn vacate(&mut self, index: i64) {
        self.runs.remove(index);
    }

    /// Moves every slot at or after `place` one index up, making room for an
    /// insertion there.
    ///
    /// Splits the map at `place` and reinserts the tail one index higher, then
    /// rebuilds, since every moved slot's recorded positions have changed.
    /// Panics rather than wrapping if a slot sits at `i64::MAX` and has nowhere
    /// to go.
    pub fn shift_from(&mut self, place: i64) {
        let moved = self.slots.split_off(&place);
        if moved.is_empty() {
            return;
        }
        assert!(
            moved
                .last_key_value()
                .is_none_or(|(index, _)| *index < i64::MAX),
            "cannot shift i64::MAX"
        );
        for (index, slot) in moved {
            self.slots.insert(index + 1, slot);
        }
        self.rebuild();
    }

    /// Reorders the slot contents while keeping the same occupied indices: the
    /// contents are handed to `reorder` as a list, and put back in the order it
    /// leaves them.
    ///
    /// What sorting and shuffling a sparse sequence are built on. Rebuilds the
    /// element index afterwards, since the contents have moved.
    pub fn reorder(&mut self, reorder: impl FnOnce(&mut Vec<S>)) {
        let indices: Vec<i64> = self.slots.keys().copied().collect();
        let mut contents: Vec<S> = std::mem::take(&mut self.slots).into_values().collect();
        reorder(&mut contents);
        self.slots = indices.into_iter().zip(contents).collect();
        self.rebuild();
    }

    /// Renumbers the slots to `0..len`, keeping their order and closing every
    /// gap between them. Rebuilds the element index afterwards.
    pub fn compact(&mut self) {
        let slots = std::mem::take(&mut self.slots);
        self.slots = slots
            .into_values()
            .enumerate()
            .map(|(index, slot)| (index as i64, slot))
            .collect();
        self.rebuild();
    }

    /// Rebuilds the element index and the run record from the slots
    /// themselves, in O(n).
    ///
    /// What the bulk operations use instead of patching: after a shift, a
    /// reorder or a compaction, every recorded position is stale.
    pub fn rebuild(&mut self) {
        self.positions.clear();
        self.runs.clear();
        let Self {
            slots,
            positions,
            runs,
        } = self;
        for (&index, slot) in slots.iter() {
            runs.insert(index);
            slot.for_each_element(|item| {
                positions.entry(item.clone()).or_default().insert(index);
            });
        }
    }

    /// Empties the slots, the element index and the run record.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.positions.clear();
        self.runs.clear();
    }

    /// The lowest index holding `item`, or `None` if it is not stored. O(1)
    /// in the number of slots, since the element index answers it.
    pub fn first_index_of(&self, item: &T) -> Option<i64> {
        self.positions.get(item)?.first().copied()
    }

    /// The highest index holding `item`, or `None` if it is not stored.
    pub fn last_index_of(&self, item: &T) -> Option<i64> {
        self.positions.get(item)?.last().copied()
    }

    /// Every index holding `item`, in increasing order, and empty if it is not
    /// stored.
    pub fn indices_of(&self, item: &T) -> impl DoubleEndedIterator<Item = i64> + '_ {
        self.positions.get(item).into_iter().flatten().copied()
    }
}
