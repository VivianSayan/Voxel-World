//! Asking a structure for everything between two bounds.

use std::ops::RangeBounds;

/// A structure whose elements are kept in the order of some key, and can be
/// asked for the ones inside a range of it.
///
/// The key is whatever the structure orders by, which is not always the
/// element: a priority queue ranges over priorities, a sparse sequence over its
/// indices, a scheduler over the steps work is due on. Each of those had the
/// method already; the trait is what lets one piece of code take any of them.
///
/// The cost is expected to be O(log n) to find the first match and then one
/// step per match, rather than a walk of everything. A structure that can only
/// filter should not implement this, since the point of the bound is that the
/// caller is asking for a cheap window rather than a scan.
pub trait RangeQuery {
    /// What the structure is ordered by.
    type Key: Ord;

    /// What a match hands back.
    type Item<'a>
    where
        Self: 'a;

    /// Everything whose key lies in `range`, in key order.
    ///
    /// Double-ended, so the same call serves "the first few above" and "the
    /// last few below" without a second method.
    fn range<'a, R: RangeBounds<Self::Key>>(
        &'a self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = Self::Item<'a>>;

    /// The first entry at or after `key`, if there is one.
    fn first_from<'a>(&'a self, key: Self::Key) -> Option<Self::Item<'a>>
    where
        Self::Key: Clone,
    {
        self.range(key..).next()
    }

    /// The last entry at or before `key`, if there is one.
    fn last_until<'a>(&'a self, key: Self::Key) -> Option<Self::Item<'a>>
    where
        Self::Key: Clone,
    {
        self.range(..=key).next_back()
    }
}
