//! Structures whose members are divided into a fixed set of numbered groups.

/// Read access to a structure that keeps its members in numbered groups.
///
/// The groups are addressed by index, from `0` to
/// [`group_count`](Partitioned::group_count) exclusive, and every member is in
/// exactly one of them. That is the difference from
/// [`Grouping`](super::Grouping), where groups are named by a label, may be
/// absent, and a member may carry several: here the groups are a fixed
/// partition, so their number is known in advance and the shares add up to the
/// whole.
///
/// What a group *means* is the structure's business. A
/// [`Rota`](crate::structures::collections::Rota) divides its members evenly so
/// that each group is one turn's worth of work; a
/// [`BucketQueue`](crate::structures::collections::BucketQueue) divides its
/// members by priority, so its groups are deliberately uneven. The trait says
/// only that the division exists and can be read;
/// [`BalancedPartition`] is what promises the shares are even.
pub trait Partitioned {
    /// What the groups hold.
    type Member;

    /// How many groups there are. Fixed for the life of the structure.
    fn group_count(&self) -> usize;

    /// How many members one group holds, or zero for a group that does not
    /// exist.
    fn group_len(&self, group: usize) -> usize;

    /// The members of one group, in whatever order the structure defines, and
    /// nothing at all for a group that does not exist.
    fn group_members(&self, group: usize) -> impl Iterator<Item = &Self::Member>;

    /// Every group's size, in group order.
    ///
    /// The rotas also expose the same numbers as a slice through an inherent
    /// method of this name, which takes precedence when the type is known; this
    /// one is what generic code over [`Partitioned`] sees.
    fn group_sizes(&self) -> impl Iterator<Item = usize> {
        (0..self.group_count()).map(|group| self.group_len(group))
    }

    /// Whether one group holds nothing.
    fn is_group_empty(&self, group: usize) -> bool {
        self.group_len(group) == 0
    }

    /// The index of a group holding the most members, or `None` when there are
    /// no groups. Ties go to the lowest index.
    fn fullest_group(&self) -> Option<usize> {
        (0..self.group_count()).reduce(|fullest, group| {
            if self.group_len(group) > self.group_len(fullest) {
                group
            } else {
                fullest
            }
        })
    }

    /// The index of a group holding the fewest members, or `None` when there
    /// are no groups. Ties go to the lowest index.
    fn emptiest_group(&self) -> Option<usize> {
        (0..self.group_count()).reduce(|emptiest, group| {
            if self.group_len(group) < self.group_len(emptiest) {
                group
            } else {
                emptiest
            }
        })
    }

    /// How far apart the fullest and emptiest groups are, which is zero when
    /// every group holds the same number.
    fn spread(&self) -> usize {
        let fullest: usize = self
            .fullest_group()
            .map_or(0, |group| self.group_len(group));
        let emptiest: usize = self
            .emptiest_group()
            .map_or(0, |group| self.group_len(group));

        fullest - emptiest
    }

    /// Whether no two groups differ by more than one member.
    ///
    /// Always true of a [`BalancedPartition`]; worth asking of anything else.
    fn is_balanced(&self) -> bool {
        self.spread() <= 1
    }
}

/// A partition that keeps its groups within one member of each other at all
/// times.
///
/// The promise holds after every operation, not merely after a rebalance: an
/// insertion goes to a group that is currently smallest, and a removal that
/// would open a gap of two moves one member across to close it. A caller may
/// therefore take one group per turn and know that no turn carries meaningfully
/// more than another, and [`Partitioned::is_balanced`] is true whenever it is
/// asked.
///
/// The cost of that promise is that a member does not always stay in the group
/// it first landed in. Anything that remembers where a member was should ask
/// again rather than assume.
pub trait BalancedPartition: Partitioned {}
