//! Storage shared by `SparseSequence` and `SparseSetSequence`: slots at
//! arbitrary integer indices, a reverse index from element to the slots
//! holding it, and a run-length record of occupied indices so the first
//! free index is found in O(log n).

use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::hashing::FastHashMap;
use crate::misc::structures::traits::Element;
use std::collections::{BTreeMap, BTreeSet};

/// What a slot holds, as far as the reverse index is concerned.
pub trait SlotContents<T> {
    fn for_each_element(&self, visit: impl FnMut(&T));
}

/// A slot holding exactly one element.
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

/// Occupied indices as maximal runs `start -> end` (inclusive).
#[derive(Clone, Debug, Default)]
pub struct IndexRuns {
    runs: BTreeMap<i64, i64>,
}

impl IndexRuns {
    fn run_containing(&self, index: i64) -> Option<(i64, i64)> {
        let (&start, &end) = self.runs.range(..=index).next_back()?;
        (end >= index).then_some((start, end))
    }

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
    pub fn first_free(&self) -> i64 {
        self.run_containing(0).map_or(0, |(_, end)| {
            end.checked_add(1)
                .expect("no non-negative sparse index remains")
        })
    }

    pub fn clear(&mut self) {
        self.runs.clear();
    }
}

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
    pub fn first_free(&self) -> i64 {
        self.runs.first_free()
    }

    pub fn min_index(&self) -> Option<i64> {
        self.slots.keys().next().copied()
    }

    pub fn max_index(&self) -> Option<i64> {
        self.slots.keys().next_back().copied()
    }

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

    pub fn unindex_element(&mut self, item: &T, index: i64) {
        if let Some(indices) = self.positions.get_mut(item) {
            indices.remove(&index);
            if indices.is_empty() {
                self.positions.remove(item);
            }
        }
    }

    /// Stores a slot, returning the one it replaced.
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

    /// Moves every slot at or after `place` one index up.
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

    /// Reorders the slot contents while keeping the occupied indices.
    pub fn reorder(&mut self, reorder: impl FnOnce(&mut Vec<S>)) {
        let indices: Vec<i64> = self.slots.keys().copied().collect();
        let mut contents: Vec<S> = std::mem::take(&mut self.slots).into_values().collect();
        reorder(&mut contents);
        self.slots = indices.into_iter().zip(contents).collect();
        self.rebuild();
    }

    /// Renumbers the slots to `0..len`, keeping their order.
    pub fn compact(&mut self) {
        let slots = std::mem::take(&mut self.slots);
        self.slots = slots
            .into_values()
            .enumerate()
            .map(|(index, slot)| (index as i64, slot))
            .collect();
        self.rebuild();
    }

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

    pub fn clear(&mut self) {
        self.slots.clear();
        self.positions.clear();
        self.runs.clear();
    }

    pub fn first_index_of(&self, item: &T) -> Option<i64> {
        self.positions.get(item)?.first().copied()
    }

    pub fn last_index_of(&self, item: &T) -> Option<i64> {
        self.positions.get(item)?.last().copied()
    }

    pub fn indices_of(&self, item: &T) -> impl DoubleEndedIterator<Item = i64> + '_ {
        self.positions.get(item).into_iter().flatten().copied()
    }
}
