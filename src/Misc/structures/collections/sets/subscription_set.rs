//! A set of subscribers that all receive the same callback, with each
//! subscriber passed in as the first argument.
//!
//! This replaces the Godot `CallableSubscriptionSet`, which connected one
//! callable (bound to each object) to a signal on every member object. Rust
//! has no signal registry, so the owner calls `notify` wherever the signal
//! would have been emitted.
//!
//! ```ignore
//! let mut watchers = SubscriptionSet::new(|chunk: &ChunkId, event: &Edit| mark_dirty(*chunk, event));
//! watchers.insert(chunk_id);
//! watchers.notify(&edit);
//! ```

use crate::misc::structures::collections::sets::set::Set;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, UniqueCollection,
};
use std::fmt;

#[derive(Clone)]
/// Unique subscribers of type `S` paired with a callback `F` invoked for each
/// subscriber when an event is published.
pub struct SubscriptionSet<S, F> {
    subscribers: Set<S>,
    callback: F,
}

impl<S: Element, F> SubscriptionSet<S, F> {
    /// Creates an empty subscription set using `callback` for notifications.
    pub fn new(callback: F) -> Self {
        Self {
            subscribers: Set::new(),
            callback,
        }
    }

    /// Returns the number of subscribers.
    pub fn len(&self) -> usize {
        self.subscribers.len()
    }

    /// Returns whether there are no subscribers.
    pub fn is_empty(&self) -> bool {
        self.subscribers.is_empty()
    }

    /// Returns whether `subscriber` is registered.
    pub fn contains(&self, subscriber: &S) -> bool {
        self.subscribers.contains(subscriber)
    }

    /// Registers `subscriber`, returning whether it was new.
    pub fn insert(&mut self, subscriber: S) -> bool {
        self.subscribers.insert(subscriber)
    }

    /// Unregisters `subscriber`, returning whether it was present.
    pub fn remove(&mut self, subscriber: &S) -> bool {
        self.subscribers.remove(subscriber)
    }

    /// Removes every subscriber while retaining allocated storage.
    pub fn clear(&mut self) {
        self.subscribers.clear();
    }

    /// Retains subscribers for which `keep` returns `true`.
    pub fn retain(&mut self, keep: impl FnMut(&S) -> bool) {
        self.subscribers.retain(keep);
    }

    /// Iterates over subscribers.
    pub fn iter(&self) -> std::collections::hash_set::Iter<'_, S> {
        self.subscribers.iter()
    }

    /// Borrows the subscriber set.
    pub fn subscribers(&self) -> &Set<S> {
        &self.subscribers
    }

    /// Borrows the callback without invoking it.
    pub fn callback(&self) -> &F {
        &self.callback
    }

    /// Calls the callback once per subscriber.
    pub fn notify<E: ?Sized>(&self, event: &E)
    where
        F: Fn(&S, &E),
    {
        for subscriber in &self.subscribers {
            (self.callback)(subscriber, event);
        }
    }

    /// Calls a callback that can mutate its own state.
    pub fn notify_mut<E: ?Sized>(&mut self, event: &E)
    where
        F: FnMut(&S, &E),
    {
        let Self {
            subscribers,
            callback,
        } = self;
        for subscriber in subscribers.iter() {
            callback(subscriber, event);
        }
    }
}

impl<S: fmt::Debug, F> fmt::Debug for SubscriptionSet<S, F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SubscriptionSet")
            .field("subscribers", &self.subscribers)
            .finish_non_exhaustive()
    }
}

impl<S: Element, F> Extend<S> for SubscriptionSet<S, F> {
    fn extend<I: IntoIterator<Item = S>>(&mut self, subscribers: I) {
        for subscriber in subscribers {
            self.insert(subscriber);
        }
    }
}

impl<'a, S, F> IntoIterator for &'a SubscriptionSet<S, F> {
    type Item = &'a S;
    type IntoIter = std::collections::hash_set::Iter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.subscribers).into_iter()
    }
}

impl<S: Element, F> Collection for SubscriptionSet<S, F> {
    type Item = S;

    fn len(&self) -> usize {
        self.subscribers.len()
    }

    fn contains(&self, subscriber: &S) -> bool {
        self.subscribers.contains(subscriber)
    }

    fn elements(&self) -> impl Iterator<Item = &S> {
        self.subscribers.iter()
    }
}

impl<S: Element, F> CollectionInsert for SubscriptionSet<S, F> {
    fn insert(&mut self, subscriber: S) -> bool {
        self.subscribers.insert(subscriber)
    }
}

impl<S: Element, F> CollectionRemove for SubscriptionSet<S, F> {
    fn remove(&mut self, subscriber: &S) -> bool {
        self.subscribers.remove(subscriber)
    }

    fn clear(&mut self) {
        self.subscribers.clear();
    }

    fn retain<P: FnMut(&S) -> bool>(&mut self, keep: P) {
        self.subscribers.retain(keep);
    }
}

impl<S: Element, F> UniqueCollection for SubscriptionSet<S, F> {}

impl<S: Element, F> Choose for SubscriptionSet<S, F> {}
