//! Which group takes the next element, and which move restores the balance
//! after one leaves.
//!
//! Shared by every rota in this family. It tracks nothing but the group sizes,
//! so the collections above decide how their elements are stored while the
//! invariant lives in one place.

/// Group sizes, kept within one of each other.
///
/// `GROUPS` is the number of groups, fixed at compile time. Every operation is
/// a scan of that many counters, which is a handful of comparisons for the
/// sizes a rota is meant for; nothing here allocates.
#[derive(Clone, Debug)]
pub(super) struct Balance<const GROUPS: usize> {
    /// How many elements each group currently holds.
    sizes: [usize; GROUPS],
    /// Where the next search for an emptiest group begins, so that equally
    /// empty groups are filled in turn rather than always the first of them.
    cursor: usize,
    /// The total, kept rather than summed.
    len: usize,
}

impl<const GROUPS: usize> Balance<GROUPS> {
    /// All groups empty.
    pub(super) const fn new() -> Self {
        const { assert!(GROUPS > 0, "a rota needs at least one group") };

        Self {
            sizes: [0; GROUPS],
            cursor: 0,
            len: 0,
        }
    }

    /// How many elements are held across every group.
    pub(super) const fn len(&self) -> usize {
        self.len
    }

    /// How many a single group holds.
    pub(super) const fn size(&self, group: usize) -> usize {
        self.sizes[group]
    }

    /// Every group's size, in group order.
    pub(super) const fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// The group the next element belongs in: the emptiest, and among equally
    /// empty ones the next in turn.
    ///
    /// Advancing the cursor past the chosen group is what spreads a run of
    /// insertions evenly rather than piling them into the lowest-numbered
    /// group each time.
    pub(super) fn next_group(&mut self) -> usize {
        let mut chosen: usize = self.cursor;
        let mut fewest: usize = self.sizes[chosen];

        for step in 1..GROUPS {
            let candidate: usize = (self.cursor + step) % GROUPS;

            if self.sizes[candidate] < fewest {
                chosen = candidate;
                fewest = self.sizes[candidate];
            }
        }

        self.cursor = (chosen + 1) % GROUPS;

        chosen
    }

    /// Records that a group gained an element.
    pub(super) const fn record_insert(&mut self, group: usize) {
        self.sizes[group] += 1;
        self.len += 1;
    }

    /// Records that a group lost an element.
    pub(super) const fn record_remove(&mut self, group: usize) {
        self.sizes[group] -= 1;
        self.len -= 1;
    }

    /// Records that one element moved between groups, which leaves the total
    /// alone.
    pub(super) const fn record_move(&mut self, from: usize, to: usize) {
        self.sizes[from] -= 1;
        self.sizes[to] += 1;
    }

    /// The move that would restore the invariant: take one element from the
    /// fullest group and give it to the emptiest.
    ///
    /// `None` once no group is more than one ahead of another, which is the
    /// state every public operation leaves behind. A single removal can only
    /// open a gap of two, so a caller normally finds at most one move waiting;
    /// bulk changes may need several, which is why it is worth asking until it
    /// answers `None`.
    pub(super) fn transfer(&self) -> Option<(usize, usize)> {
        let mut fullest: usize = 0;
        let mut emptiest: usize = 0;

        for group in 1..GROUPS {
            if self.sizes[group] > self.sizes[fullest] {
                fullest = group;
            }

            if self.sizes[group] < self.sizes[emptiest] {
                emptiest = group;
            }
        }

        (self.sizes[fullest] >= self.sizes[emptiest] + 2).then_some((fullest, emptiest))
    }

    /// Back to every group empty.
    pub(super) const fn clear(&mut self) {
        self.sizes = [0; GROUPS];
        self.cursor = 0;
        self.len = 0;
    }
}

impl<const GROUPS: usize> Default for Balance<GROUPS> {
    fn default() -> Self {
        Self::new()
    }
}
