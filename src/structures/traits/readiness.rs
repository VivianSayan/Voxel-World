//! Holding work back until something releases it.

/// Structures that hold items until something makes them ready, and then hand
/// the ready ones over together.
///
/// What makes an item ready is the implementor's business and is deliberately
/// not part of this trait: a [`Scheduler`](crate::structures::collections::Scheduler)
/// releases what the clock has reached, a
/// [`ConditionIndex`](crate::structures::indices::ConditionIndex) releases what
/// a change to its state has satisfied. What they share is the shape at the
/// other end — work accumulates, and a caller collects it in a batch at a
/// moment of its own choosing rather than being called back mid-write.
///
/// That shape is worth naming because the same two questions are otherwise
/// asked under different names in different places: whether anything is waiting,
/// and take it.
///
/// # Held and ready are separate counts
///
/// [`Pending::pending_len`] counts what is still being held and
/// [`Pending::ready_len`] what is waiting to be collected. They are two counts,
/// not a split of one total: an implementor may hold an item and have it ready
/// at the same time, or release an item from its keeping as it becomes ready.
///
/// # Collecting is destructive
///
/// [`Pending::take_ready`] hands the items over and does not keep them. A caller
/// that drops the returned batch has dropped the work.
pub trait Pending {
    /// What comes out when ready items are collected.
    ///
    /// Often the held item itself, but not always: a structure that records
    /// something about the release describes it here instead.
    type Ready;

    /// How many items are being held.
    fn pending_len(&self) -> usize;

    /// Whether anything is being held.
    fn has_pending(&self) -> bool {
        self.pending_len() > 0
    }

    /// How many items are ready to be collected as things stand.
    fn ready_len(&self) -> usize;

    /// Whether anything is ready to be collected as things stand.
    fn has_ready(&self) -> bool {
        self.ready_len() > 0
    }

    /// Collects everything ready, leaving the rest held.
    ///
    /// The order is the implementor's canonical one, and empty when nothing is
    /// ready.
    fn take_ready(&mut self) -> Vec<Self::Ready>;
}
