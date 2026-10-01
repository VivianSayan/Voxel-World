//! Traits for structures that sort their members into labeled groups.

use crate::structures::traits::collection::Element;

/// Read access to members partitioned or indexed under labels.
pub trait Grouping {
    /// Type stored in each group.
    type Member: Element;
    /// Type used to identify groups.
    type Label: Element;

    /// A borrowed view of one group. It may be a stored set, a slice, or an
    /// iterator that computes membership from another representation.
    type Group<'a>: IntoIterator<Item = &'a Self::Member>
    where
        Self: 'a;

    /// The members carrying `label`.
    fn group(&self, label: &Self::Label) -> Option<Self::Group<'_>>;

    /// Iterates over every label that currently has a non-empty group.
    fn labels(&self) -> impl Iterator<Item = &Self::Label>;

    /// Returns the number of non-empty labels.
    fn label_count(&self) -> usize;

    /// Returns the number of members carrying `label`, or zero if absent.
    fn group_len(&self, label: &Self::Label) -> usize {
        self.group(label)
            .map_or(0, |group| group.into_iter().count())
    }

    /// Returns whether `label` has a non-empty group.
    fn contains_label(&self, label: &Self::Label) -> bool {
        self.group(label).is_some()
    }

    /// Returns whether `member` belongs to the group identified by `label`.
    fn is_in_group(&self, member: &Self::Member, label: &Self::Label) -> bool {
        self.group(label)
            .is_some_and(|group| group.into_iter().any(|held| held == member))
    }

    /// Each label with its group.
    fn groups(&self) -> impl Iterator<Item = (&Self::Label, Self::Group<'_>)> {
        self.labels()
            .filter_map(|label| Some((label, self.group(label)?)))
    }
}

/// Groupings that keep their groups indexed by size.
pub trait GroupSizes: Grouping {
    /// Labels whose group has exactly `len` members.
    fn labels_with_len(&self, len: usize) -> impl Iterator<Item = &Self::Label>;

    /// Returns one arbitrary largest group, or `None` when there are no groups.
    fn largest_group(&self) -> Option<(&Self::Label, Self::Group<'_>)>;

    /// Returns one arbitrary smallest group, or `None` when there are no groups.
    fn smallest_group(&self) -> Option<(&Self::Label, Self::Group<'_>)>;

    /// Every group tied for largest. The older `largest_group` method picks
    /// one arbitrary label when several groups have the same size.
    fn largest_groups(&self) -> impl Iterator<Item = (&Self::Label, Self::Group<'_>)> {
        let len = self
            .largest_group()
            .map_or(0, |(_, group)| group.into_iter().count());
        self.labels_with_len(len)
            .filter_map(|label| Some((label, self.group(label)?)))
    }

    /// Every group tied for smallest. The older `smallest_group` method picks
    /// one arbitrary label when several groups have the same size.
    fn smallest_groups(&self) -> impl Iterator<Item = (&Self::Label, Self::Group<'_>)> {
        let len = self
            .smallest_group()
            .map_or(0, |(_, group)| group.into_iter().count());
        self.labels_with_len(len)
            .filter_map(|label| Some((label, self.group(label)?)))
    }

    /// Labels with exactly one member, paired with that member.
    fn singleton_groups(&self) -> impl Iterator<Item = (&Self::Label, &Self::Member)> {
        self.labels_with_len(1)
            .filter_map(|label| Some((label, self.group(label)?.into_iter().next()?)))
    }
}
