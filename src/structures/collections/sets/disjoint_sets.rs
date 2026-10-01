//! Elements grouped by connection, where the groups are discovered rather than
//! declared.
//!
//! The question it answers is "are these two part of the same thing?" for a
//! relation that arrives one link at a time: two cave cells that turned out to
//! be joined, two ore veins that touched, two regions a river ran between. Every
//! join is near-constant time, and so is every question, without ever building
//! the groups explicitly.
//!
//! What it cannot do is separate anything again. Links only ever merge groups,
//! so a structure that has to forget one has to be rebuilt.

use crate::structures::collections::sets::set::Set;
use crate::structures::hashing::FastHashMap;
use crate::structures::traits::{
    Capacity, Collection, CollectionInsert, ContentHashable, DeterministicOrder, Element, Grouping,
    StableHash, stable_hash_unordered,
};
use crate::units::digest::ContentHash;
use std::hash::Hash;

/// A disjoint-set forest: elements partitioned into groups that only ever
/// merge.
///
/// Each element points at another in its group until one points at itself, and
/// that one names the group. Two refinements keep the chains short enough that
/// both operations are near-constant in practice: a join attaches the smaller
/// group under the larger, and every lookup repoints the elements it passes
/// straight at the root it found.
///
/// The representative of a group is whichever element the structure settled on,
/// which is not the first one added and is not stable across further joins.
/// Where a group needs a name of its own, keep it beside the structure.
#[derive(Clone, Debug, Default)]
pub struct DisjointSets<T> {
    /// Each element's parent, by position; a root points at itself.
    parents: Vec<usize>,
    /// How many elements each root's group holds, meaningful only at a root.
    sizes: Vec<usize>,
    /// The elements themselves, in the order they were added.
    elements: Vec<T>,
    /// Where each element sits in the vectors above.
    positions: FastHashMap<T, usize>,
    /// How many groups there are, kept as joins happen.
    groups: usize,
}

/// Borrowed members of one discovered component.
pub struct DisjointGroup<'a, T> {
    forest: &'a DisjointSets<T>,
    root: usize,
    next: usize,
    remaining: usize,
}

impl<'a, T: Element> Iterator for DisjointGroup<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.forest.elements.len() {
            let position = self.next;
            self.next += 1;
            if self.forest.root_readonly(position) == self.root {
                self.remaining -= 1;
                return Some(&self.forest.elements[position]);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Element> ExactSizeIterator for DisjointGroup<'_, T> {}

impl<T: Element> DisjointSets<T> {
    /// An empty forest.
    pub fn new() -> Self {
        Self {
            parents: Vec::new(),
            sizes: Vec::new(),
            elements: Vec::new(),
            positions: FastHashMap::default(),
            groups: 0,
        }
    }

    /// Adds an element as a group of its own, or does nothing if it is already
    /// known. Returns whether it was new.
    pub fn insert(&mut self, element: T) -> bool {
        if self.positions.contains_key(&element) {
            return false;
        }

        let position: usize = self.elements.len();

        self.positions.insert(element.clone(), position);
        self.elements.push(element);
        self.parents.push(position);
        self.sizes.push(1);
        self.groups += 1;

        true
    }

    /// Whether the element is known at all.
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, element: &Q) -> bool
    where
        T: std::borrow::Borrow<Q>,
    {
        self.positions.contains_key(element)
    }

    /// The element standing for `element`'s group, or `None` if it is unknown.
    ///
    /// Two elements are in the same group exactly when this returns the same
    /// one for both, which is what [`DisjointSets::joined`] asks.
    pub fn representative(&mut self, element: &T) -> Option<&T> {
        let position: usize = *self.positions.get(element)?;
        let root: usize = self.root_of(position);

        Some(&self.elements[root])
    }

    /// Joins the two elements' groups, adding either if it is unknown.
    ///
    /// Returns whether anything changed: `false` means they were already in the
    /// same group, which is the common answer once a region has been walked.
    pub fn join(&mut self, first: T, second: T) -> bool {
        self.insert(first.clone());
        self.insert(second.clone());

        let (left, right): (usize, usize) = (self.positions[&first], self.positions[&second]);
        let (mut left_root, mut right_root): (usize, usize) =
            (self.root_of(left), self.root_of(right));

        if left_root == right_root {
            return false;
        }

        // The smaller group goes under the larger, which is what bounds the
        // depth of the chains.
        if self.sizes[left_root] < self.sizes[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }

        self.parents[right_root] = left_root;
        self.sizes[left_root] += self.sizes[right_root];
        self.groups -= 1;

        true
    }

    /// Whether two elements are in the same group. `false` if either is
    /// unknown.
    pub fn joined(&mut self, first: &T, second: &T) -> bool {
        let (Some(left), Some(right)) = (
            self.positions.get(first).copied(),
            self.positions.get(second).copied(),
        ) else {
            return false;
        };

        self.root_of(left) == self.root_of(right)
    }

    /// How many elements share `element`'s group, including it. Zero if it is
    /// unknown.
    pub fn group_size(&mut self, element: &T) -> usize {
        let Some(position) = self.positions.get(element).copied() else {
            return 0;
        };

        let root: usize = self.root_of(position);

        self.sizes[root]
    }

    /// How many groups there are.
    pub const fn group_count(&self) -> usize {
        self.groups
    }

    /// How many elements are known.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Whether nothing is known.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Every element, in the order they were added.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    /// The groups, each as a set, in the order their first element was added.
    ///
    /// The one operation that is not near-constant: it walks every element and
    /// builds the sets, which is O(n). Ask for it once at the end rather than
    /// inside a loop.
    pub fn groups(&mut self) -> Vec<Set<T>> {
        let mut roots: FastHashMap<usize, usize> = FastHashMap::default();
        let mut collected: Vec<Set<T>> = Vec::with_capacity(self.groups);

        for position in 0..self.elements.len() {
            let root: usize = self.root_of(position);
            let slot: usize = match roots.get(&root) {
                Some(slot) => *slot,
                None => {
                    roots.insert(root, collected.len());
                    collected.push(Set::new());
                    collected.len() - 1
                }
            };

            collected[slot].insert(self.elements[position].clone());
        }

        collected
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        self.parents.clear();
        self.sizes.clear();
        self.elements.clear();
        self.positions.clear();
        self.groups = 0;
    }

    /// The root of the chain `position` is in, repointing everything on the way
    /// straight at it so the next walk is shorter.
    fn root_of(&mut self, position: usize) -> usize {
        let mut current: usize = position;

        while self.parents[current] != current {
            let grandparent: usize = self.parents[self.parents[current]];

            self.parents[current] = grandparent;
            current = grandparent;
        }

        current
    }

    /// Finds a root without path compression, for borrowed group views.
    fn root_readonly(&self, position: usize) -> usize {
        let mut current = position;
        while self.parents[current] != current {
            current = self.parents[current];
        }
        current
    }
}

impl<T: Element> Collection for DisjointSets<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn contains(&self, item: &T) -> bool {
        self.positions.contains_key(item)
    }

    /// In the order elements were added, which is what makes this
    /// [`DeterministicOrder`]: the groups they fell into do not affect it.
    fn elements(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }
}

impl<T: Element> CollectionInsert for DisjointSets<T> {
    /// Adds the element as a group of its own; joining is
    /// [`DisjointSets::join`].
    fn insert(&mut self, item: T) -> bool {
        DisjointSets::insert(self, item)
    }
}

impl<T: Element> Capacity for DisjointSets<T> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            parents: Vec::with_capacity(capacity),
            sizes: Vec::with_capacity(capacity),
            elements: Vec::with_capacity(capacity),
            positions: FastHashMap::with_capacity_and_hasher(capacity, Default::default()),
            groups: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.parents.reserve(additional);
        self.sizes.reserve(additional);
        self.elements.reserve(additional);
        self.positions.reserve(additional);
    }

    fn shrink_to_fit(&mut self) {
        self.parents.shrink_to_fit();
        self.sizes.shrink_to_fit();
        self.elements.shrink_to_fit();
        self.positions.shrink_to_fit();
    }
}

impl<T: Element> DeterministicOrder for DisjointSets<T> {}

/// The partition itself, independently of representatives and insertion order.
impl<T: Element + StableHash> ContentHashable for DisjointSets<T> {
    fn content_hash(&self) -> ContentHash {
        let mut groups: FastHashMap<usize, Vec<&T>> = FastHashMap::default();
        for (position, element) in self.elements.iter().enumerate() {
            groups
                .entry(self.root_readonly(position))
                .or_default()
                .push(element);
        }
        let mut hashes: Vec<u128> = groups
            .values()
            .map(|group| stable_hash_unordered(group.iter().copied()).value())
            .collect();
        hashes.sort_unstable();
        hashes
            .into_iter()
            .fold(ContentHash::EMPTY, |state, hash| state.and_value(hash))
    }
}

impl<T: Element> FromIterator<(T, T)> for DisjointSets<T> {
    /// From a list of links, each joining two elements.
    fn from_iter<I: IntoIterator<Item = (T, T)>>(links: I) -> Self {
        let mut forest: Self = Self::new();

        for (first, second) in links {
            forest.join(first, second);
        }

        forest
    }
}

impl<T: Element> Extend<(T, T)> for DisjointSets<T> {
    fn extend<I: IntoIterator<Item = (T, T)>>(&mut self, links: I) {
        for (first, second) in links {
            self.join(first, second);
        }
    }
}

/// Groups are found rather than labelled, so the label is the representative
/// element. A label that is not a current representative has no group.
impl<T: Element> Grouping for DisjointSets<T> {
    type Member = T;
    type Label = T;
    type Group<'a>
        = DisjointGroup<'a, T>
    where
        Self: 'a;

    fn group(&self, label: &T) -> Option<Self::Group<'_>> {
        let position = *self.positions.get(label)?;
        let root = self.root_readonly(position);
        if root != position {
            return None;
        }
        Some(DisjointGroup {
            forest: self,
            root,
            next: 0,
            remaining: self.sizes[root],
        })
    }

    /// The representatives, which are the elements that point at themselves.
    fn labels(&self) -> impl Iterator<Item = &T> {
        self.parents
            .iter()
            .enumerate()
            .filter(|(position, parent)| position == *parent)
            .map(|(position, _)| &self.elements[position])
    }

    fn label_count(&self) -> usize {
        self.groups
    }
}

/// As how much is known and how it is divided, such as
/// `12 elements in 3 groups`, since the groups themselves take a walk to build.
impl<T: Element> std::fmt::Display for DisjointSets<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} elements in {} groups",
            self.elements.len(),
            self.groups
        )
    }
}
