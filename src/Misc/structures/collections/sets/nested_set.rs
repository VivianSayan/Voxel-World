//! A set stored as a hash trie: each level of the tree consumes five bits
//! of the element's hash, so it holds 32 buckets. The Godot version built
//! this from nested dictionaries keyed by hash characters; here each node
//! is a 32-bit occupancy bitmap plus a compact array of only the occupied
//! buckets (a hash array mapped trie), so sparse levels stay small.
//!
//! Elements whose full 64-bit hashes collide share a leaf bucket.

use crate::misc::structures::hashing::hash_one;
use crate::misc::structures::traits::operators::impl_set_operators;
use crate::misc::structures::traits::{
    Choose, Collection, CollectionInsert, CollectionRemove, Element, SetAlgebra, UniqueCollection,
};

const BITS_PER_LEVEL: u32 = 5;
const LEVEL_MASK: u64 = (1 << BITS_PER_LEVEL) - 1;

#[derive(Clone, Debug)]
enum Entry<T> {
    Leaf(u64, T),
    /// Elements whose full hashes are equal.
    Collision(u64, Vec<T>),
    Branch(Box<Node<T>>),
}

#[derive(Clone, Debug)]
struct Node<T> {
    bitmap: u32,
    entries: Vec<Entry<T>>,
}

impl<T> Default for Node<T> {
    fn default() -> Self {
        Self {
            bitmap: 0,
            entries: Vec::new(),
        }
    }
}

#[inline]
fn bucket_of(hash: u64, shift: u32) -> u32 {
    ((hash >> shift) & LEVEL_MASK) as u32
}

impl<T: Element> Node<T> {
    #[inline]
    fn slot(&self, bucket: u32) -> Option<usize> {
        let bit = 1u32 << bucket;
        (self.bitmap & bit != 0).then(|| (self.bitmap & (bit - 1)).count_ones() as usize)
    }

    fn contains(&self, hash: u64, item: &T, shift: u32) -> bool {
        let Some(slot) = self.slot(bucket_of(hash, shift)) else {
            return false;
        };
        match &self.entries[slot] {
            Entry::Leaf(leaf_hash, leaf) => *leaf_hash == hash && leaf == item,
            Entry::Collision(bucket_hash, items) => *bucket_hash == hash && items.contains(item),
            Entry::Branch(node) => node.contains(hash, item, shift + BITS_PER_LEVEL),
        }
    }

    fn insert(&mut self, hash: u64, item: T, shift: u32) -> bool {
        let bucket = bucket_of(hash, shift);
        let Some(slot) = self.slot(bucket) else {
            let bit = 1u32 << bucket;
            let slot = (self.bitmap & (bit - 1)).count_ones() as usize;
            self.bitmap |= bit;
            self.entries.insert(slot, Entry::Leaf(hash, item));
            return true;
        };

        match &mut self.entries[slot] {
            Entry::Branch(node) => node.insert(hash, item, shift + BITS_PER_LEVEL),
            Entry::Collision(bucket_hash, items) if *bucket_hash == hash => {
                if items.contains(&item) {
                    return false;
                }
                items.push(item);
                true
            }
            Entry::Leaf(leaf_hash, leaf) if *leaf_hash == hash => {
                if *leaf == item {
                    return false;
                }
                let existing =
                    std::mem::replace(&mut self.entries[slot], Entry::Branch(Box::default()));
                let Entry::Leaf(_, existing) = existing else {
                    unreachable!()
                };
                self.entries[slot] = Entry::Collision(hash, vec![existing, item]);
                true
            }
            _ => {
                // Different hashes share this bucket: push the existing entry
                // one level down, next to the new one.
                let existing =
                    std::mem::replace(&mut self.entries[slot], Entry::Branch(Box::default()));
                let existing_hash = match &existing {
                    Entry::Leaf(existing_hash, _) | Entry::Collision(existing_hash, _) => {
                        *existing_hash
                    }
                    Entry::Branch(_) => unreachable!(),
                };
                let Entry::Branch(node) = &mut self.entries[slot] else {
                    unreachable!()
                };
                node.place(existing_hash, existing, shift + BITS_PER_LEVEL);
                node.insert(hash, item, shift + BITS_PER_LEVEL)
            }
        }
    }

    /// Puts an already-built leaf or collision entry into this empty-ish node.
    fn place(&mut self, hash: u64, entry: Entry<T>, shift: u32) {
        let bucket = bucket_of(hash, shift);
        let bit = 1u32 << bucket;
        let slot = (self.bitmap & (bit - 1)).count_ones() as usize;
        self.bitmap |= bit;
        self.entries.insert(slot, entry);
    }

    fn remove(&mut self, hash: u64, item: &T, shift: u32) -> Option<T> {
        let bucket = bucket_of(hash, shift);
        let slot = self.slot(bucket)?;

        let removed = match &mut self.entries[slot] {
            Entry::Leaf(leaf_hash, leaf) if *leaf_hash == hash && *leaf == *item => {
                self.bitmap &= !(1u32 << bucket);
                let Entry::Leaf(_, leaf) = self.entries.remove(slot) else {
                    unreachable!()
                };
                return Some(leaf);
            }
            Entry::Leaf(..) => return None,
            Entry::Collision(bucket_hash, items) => {
                if *bucket_hash != hash {
                    return None;
                }
                let position = items.iter().position(|stored| stored == item)?;
                let removed = items.swap_remove(position);
                if items.len() == 1 {
                    let last = items.pop().unwrap();
                    self.entries[slot] = Entry::Leaf(hash, last);
                }
                removed
            }
            Entry::Branch(node) => {
                let removed = node.remove(hash, item, shift + BITS_PER_LEVEL)?;
                if node.entries.is_empty() {
                    self.bitmap &= !(1u32 << bucket);
                    self.entries.remove(slot);
                } else if node.entries.len() == 1 && !matches!(node.entries[0], Entry::Branch(_)) {
                    // Pull a lone leaf back up so chains of single branches do
                    // not linger after removals.
                    let lone = node.entries.pop().unwrap();
                    self.entries[slot] = lone;
                }
                removed
            }
        };
        Some(removed)
    }

    fn depth(&self) -> usize {
        1 + self
            .entries
            .iter()
            .map(|entry| match entry {
                Entry::Branch(node) => node.depth(),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
/// Hash-array-mapped-trie set storing unique elements of type `T`.
///
/// This is useful when inspecting or sharing trie-shaped hash storage matters;
/// for ordinary hash-set use, [`Set`](super::set::Set) is simpler.
pub struct NestedSet<T> {
    root: Node<T>,
    len: usize,
}

impl<T> Default for NestedSet<T> {
    fn default() -> Self {
        Self {
            root: Node::default(),
            len: 0,
        }
    }
}

impl<T> NestedSet<T> {
    /// Creates an empty trie set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of members.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether no members are stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Removes every member and resets the trie.
    pub fn clear(&mut self) {
        self.root = Node::default();
        self.len = 0;
    }

    /// Iterates over members in depth-first trie order.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            stack: vec![self.root.entries.iter()],
            bucket: [].iter(),
            remaining: self.len,
        }
    }
}

impl<T: Element> NestedSet<T> {
    /// Returns whether `item` is present.
    pub fn contains(&self, item: &T) -> bool {
        self.root.contains(hash_one(item), item, 0)
    }

    /// Returns whether the item was new.
    pub fn insert(&mut self, item: T) -> bool {
        let inserted = self.root.insert(hash_one(&item), item, 0);
        self.len += inserted as usize;
        inserted
    }

    /// Removes `item`, returning whether it was present.
    pub fn remove(&mut self, item: &T) -> bool {
        self.take(item).is_some()
    }

    /// Removes `item` and returns the stored copy.
    pub fn take(&mut self, item: &T) -> Option<T> {
        let taken = self.root.remove(hash_one(item), item, 0)?;
        self.len -= 1;
        Some(taken)
    }

    /// Retains members for which `keep` returns `true`.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let removed: Vec<T> = self.iter().filter(|item| !keep(item)).cloned().collect();
        for item in &removed {
            self.remove(item);
        }
    }

    /// Number of trie levels in use.
    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    fn filtered(&self, mut keep: impl FnMut(&T) -> bool) -> Self {
        self.iter().filter(|item| keep(item)).cloned().collect()
    }
}

/// Walks the trie depth-first.
pub struct Iter<'a, T> {
    stack: Vec<std::slice::Iter<'a, Entry<T>>>,
    bucket: std::slice::Iter<'a, T>,
    remaining: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            if let Some(item) = self.bucket.next() {
                self.remaining -= 1;
                return Some(item);
            }
            match self.stack.last_mut()?.next() {
                None => {
                    self.stack.pop();
                }
                Some(Entry::Leaf(_, item)) => {
                    self.remaining -= 1;
                    return Some(item);
                }
                Some(Entry::Collision(_, items)) => self.bucket = items.iter(),
                Some(Entry::Branch(node)) => self.stack.push(node.entries.iter()),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T: Element> Collection for NestedSet<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.len
    }

    fn contains(&self, item: &T) -> bool {
        Self::contains(self, item)
    }

    fn elements(&self) -> impl Iterator<Item = &T> {
        self.iter()
    }
}

impl<T: Element> CollectionInsert for NestedSet<T> {
    fn insert(&mut self, item: T) -> bool {
        Self::insert(self, item)
    }
}

impl<T: Element> CollectionRemove for NestedSet<T> {
    fn remove(&mut self, item: &T) -> bool {
        Self::remove(self, item)
    }

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn retain<F: FnMut(&T) -> bool>(&mut self, keep: F) {
        Self::retain(self, keep)
    }
}

impl<T: Element> UniqueCollection for NestedSet<T> {}

impl<T: Element> Choose for NestedSet<T> {}

impl<T: Element> SetAlgebra for NestedSet<T> {
    fn union(&self, other: &Self) -> Self {
        let (larger, smaller) = if self.len >= other.len {
            (self, other)
        } else {
            (other, self)
        };
        let mut output = larger.clone();
        output.extend(smaller.iter().cloned());
        output
    }

    fn intersection(&self, other: &Self) -> Self {
        let (smaller, larger) = if self.len <= other.len {
            (self, other)
        } else {
            (other, self)
        };
        smaller.filtered(|item| larger.contains(item))
    }

    fn difference(&self, other: &Self) -> Self {
        self.filtered(|item| !other.contains(item))
    }

    fn is_subset(&self, other: &Self) -> bool {
        self.len <= other.len && self.iter().all(|item| other.contains(item))
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        let (smaller, larger) = if self.len <= other.len {
            (self, other)
        } else {
            (other, self)
        };
        !smaller.iter().any(|item| larger.contains(item))
    }
}

impl_set_operators!([T: Element] NestedSet<T>);

impl<T: Element> PartialEq for NestedSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.is_subset(other)
    }
}

impl<T: Element> Eq for NestedSet<T> {}

impl<T: Element> FromIterator<T> for NestedSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        let mut set = Self::new();
        set.extend(items);
        set
    }
}

impl<T: Element> Extend<T> for NestedSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        for item in items {
            self.insert(item);
        }
    }
}

impl<'a, T> IntoIterator for &'a NestedSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}
