//! A common interface for queues that serve values by priority.

/// A queue that associates every item with a priority and serves the lowest
/// priority first.
pub trait PriorityQueueLike {
    /// The queued value.
    type Item;
    /// The value used to order work.
    type Priority: Ord;

    /// Queues an item, returning `false` when the queue cannot accept it.
    fn push_priority(&mut self, priority: Self::Priority, item: Self::Item) -> bool;

    /// The next item and its priority without removing it.
    fn peek_priority_item(&self) -> Option<(Self::Priority, &Self::Item)>;

    /// Removes the next item and returns its priority with it.
    fn pop_priority(&mut self) -> Option<(Self::Priority, Self::Item)>;
}
