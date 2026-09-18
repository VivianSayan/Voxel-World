//! A hierarchical index of typed tag paths.
//!
//! A path is a `/`-separated list of segments. Each segment is either
//! `type:tag` or a bare `tag`:
//!
//! ```text
//! phont:euphont/express:clothes/dress
//! ```
//!
//! Every segment becomes a node, so the path above also makes
//! `phont:euphont` and `phont:euphont/express:clothes` present. Removing a
//! path removes its whole subtree, and prefix nodes that only existed to
//! hold it are pruned.
//!
//! Each node keeps counts of the tags and types anywhere beneath it, so
//! "does this tag occur at any depth" is O(1).

use crate::misc::structures::hashing::{FastHashMap, FastHashSet};
use std::sync::Arc;

type Name = Arc<str>;

/// The type used for bare tags.
const UNTYPED: &str = "";

struct Segment<'a> {
    kind: &'a str,
    tag: &'a str,
    rest: Option<&'a str>,
}

fn split_first(path: &str) -> Option<Segment<'_>> {
    let (segment, rest) = match path.split_once('/') {
        Some((segment, rest)) => (segment, (!rest.is_empty()).then_some(rest)),
        None => (path, None),
    };
    let (kind, tag) = segment.split_once(':').unwrap_or((UNTYPED, segment));
    (!tag.is_empty()).then_some(Segment { kind, tag, rest })
}

fn is_valid_path(path: &str) -> bool {
    let mut rest = Some(path);
    while let Some(path) = rest {
        match split_first(path) {
            Some(segment) => rest = segment.rest,
            None => return false,
        }
    }
    true
}

fn format_segment(kind: &str, tag: &str) -> String {
    if kind.is_empty() {
        tag.to_owned()
    } else {
        format!("{kind}:{tag}")
    }
}

/// What a removal took out of a subtree, so ancestors can update counts.
#[derive(Default)]
struct Removed {
    paths: usize,
    tags: FastHashMap<Name, usize>,
    types: FastHashMap<Name, usize>,
}

impl Removed {
    fn add_node(&mut self, kind: &Name, tag: &Name) {
        *self.tags.entry(tag.clone()).or_insert(0) += 1;
        if !kind.is_empty() {
            *self.types.entry(kind.clone()).or_insert(0) += 1;
        }
    }
}

fn add_count(counts: &mut FastHashMap<Name, usize>, name: &Name, amount: usize) {
    *counts.entry(name.clone()).or_insert(0) += amount;
}

fn subtract_counts(counts: &mut FastHashMap<Name, usize>, removed: &FastHashMap<Name, usize>) {
    for (name, amount) in removed {
        if let Some(count) = counts.get_mut(name) {
            *count -= amount;
            if *count == 0 {
                counts.remove(name);
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Hierarchical index of slash-separated tag paths.
///
/// Each segment may be a bare tag or a typed `type:tag` segment. The index
/// tracks exact path endings, subtrees, and tag/type occurrence counts at
/// arbitrary depth.
pub struct TagIndex {
    /// type -> tag -> subtree
    entries: FastHashMap<Name, FastHashMap<Name, TagIndex>>,
    /// tag -> the types it appears under at this level
    tag_types: FastHashMap<Name, FastHashSet<Name>>,
    /// Whether a path ends at this node.
    terminal: bool,
    /// Paths ending at or below this node.
    path_count: usize,
    /// Tag and type occurrences strictly below this level.
    deep_tags: FastHashMap<Name, usize>,
    deep_types: FastHashMap<Name, usize>,
}

impl TagIndex {
    /// Creates an empty tag-path index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a path. Returns whether it was new. Paths with an empty tag in
    /// any segment are rejected.
    pub fn add(&mut self, path: &str) -> bool {
        if !is_valid_path(path) {
            return false;
        }
        match self.add_inner(path) {
            Some((added, _)) => added,
            None => false,
        }
    }

    /// Returns whether a path was added, and the (type, tag) nodes created
    /// at this level or below.
    fn add_inner(&mut self, path: &str) -> Option<(bool, Vec<(Name, Name)>)> {
        let segment = split_first(path)?;

        let mut created: Vec<(Name, Name)> = Vec::new();
        let exists = self
            .entries
            .get(segment.kind)
            .is_some_and(|tags| tags.contains_key(segment.tag));
        if !exists {
            // Reuse the stored names where they exist, so each name is
            // allocated once per level.
            let kind: Name = match self.entries.get_key_value(segment.kind) {
                Some((kind, _)) => kind.clone(),
                None => Name::from(segment.kind),
            };
            let tag: Name = match self.tag_types.get_key_value(segment.tag) {
                Some((tag, _)) => tag.clone(),
                None => Name::from(segment.tag),
            };
            self.tag_types
                .entry(tag.clone())
                .or_default()
                .insert(kind.clone());
            self.entries
                .entry(kind.clone())
                .or_default()
                .insert(tag.clone(), TagIndex::default());
            created.push((kind, tag));
        }
        let child = self
            .entries
            .get_mut(segment.kind)
            .unwrap()
            .get_mut(segment.tag)
            .unwrap();

        let added = match segment.rest {
            None => {
                let added = !child.terminal;
                child.terminal = true;
                child.path_count += added as usize;
                added
            }
            Some(rest) => match child.add_inner(rest) {
                Some((added, deeper)) => {
                    for (kind, tag) in &deeper {
                        add_count(&mut self.deep_tags, tag, 1);
                        if !kind.is_empty() {
                            add_count(&mut self.deep_types, kind, 1);
                        }
                    }
                    created.extend(deeper);
                    added
                }
                None => false,
            },
        };

        self.path_count += added as usize;
        Some((added, created))
    }

    /// Adds every path from `paths`, ignoring invalid and duplicate paths.
    pub fn add_all<'a>(&mut self, paths: impl IntoIterator<Item = &'a str>) {
        for path in paths {
            self.add(path);
        }
    }

    /// Removes a path and everything beneath it. Returns whether it existed.
    pub fn remove(&mut self, path: &str) -> bool {
        self.remove_inner(path).is_some()
    }

    fn remove_inner(&mut self, path: &str) -> Option<Removed> {
        let segment = split_first(path)?;
        let Some(rest) = segment.rest else {
            return self.remove_entry(segment.kind, segment.tag);
        };

        let child = self.entries.get_mut(segment.kind)?.get_mut(segment.tag)?;
        let mut removed = child.remove_inner(rest)?;
        let prune = !child.terminal && child.entries.is_empty();

        subtract_counts(&mut self.deep_tags, &removed.tags);
        subtract_counts(&mut self.deep_types, &removed.types);
        self.path_count -= removed.paths;

        if prune {
            let own = self.remove_entry(segment.kind, segment.tag)?;
            // The pruned node had nothing beneath it, so `own` is just itself.
            for (tag, count) in own.tags {
                add_count(&mut removed.tags, &tag, count);
            }
            for (kind, count) in own.types {
                add_count(&mut removed.types, &kind, count);
            }
        }
        Some(removed)
    }

    /// Removes one entry at this level with its subtree.
    fn remove_entry(&mut self, kind: &str, tag: &str) -> Option<Removed> {
        let tags = self.entries.get_mut(kind)?;
        let (tag_name, child) = tags.remove_entry(tag)?;
        let kind_name = self.entries.get_key_value(kind).unwrap().0.clone();
        if self.entries[kind].is_empty() {
            self.entries.remove(kind);
        }
        if let Some(kinds) = self.tag_types.get_mut(tag) {
            kinds.remove(kind);
            if kinds.is_empty() {
                self.tag_types.remove(tag);
            }
        }

        // The child's own entries and everything below them were below
        // this level too.
        let mut removed = Removed {
            paths: child.path_count,
            tags: child.deep_tags,
            types: child.deep_types,
        };
        for (child_kind, child_tags) in &child.entries {
            for child_tag in child_tags.keys() {
                removed.add_node(child_kind, child_tag);
            }
        }
        subtract_counts(&mut self.deep_tags, &removed.tags);
        subtract_counts(&mut self.deep_types, &removed.types);
        self.path_count -= removed.paths;

        removed.add_node(&kind_name, &tag_name);
        Some(removed)
    }

    /// Removes every entry at this level with `tag`, under any type.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        let Some(kinds) = self.tag_types.get(tag) else {
            return false;
        };
        let kinds: Vec<Name> = kinds.iter().cloned().collect();
        for kind in kinds {
            self.remove_entry(&kind, tag);
        }
        true
    }

    /// Removes every entry at this level of type `kind`.
    pub fn remove_type(&mut self, kind: &str) -> bool {
        let Some(tags) = self.entries.get(kind) else {
            return false;
        };
        let tags: Vec<Name> = tags.keys().cloned().collect();
        for tag in tags {
            self.remove_entry(kind, &tag);
        }
        true
    }

    /// Removes all paths and resets every derived count.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Returns whether the index contains no entries at its root level.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of paths added (and not removed) in this index.
    pub fn path_count(&self) -> usize {
        self.path_count
    }

    /// Number of distinct tags at this level.
    pub fn tag_count(&self) -> usize {
        self.tag_types.len()
    }

    /// Number of distinct types at this level.
    pub fn type_count(&self) -> usize {
        self.entries.len() - self.entries.contains_key(UNTYPED) as usize
    }

    /// The node at `path`, as an index of its own.
    pub fn sub_index(&self, path: &str) -> Option<&TagIndex> {
        let segment = split_first(path)?;
        let child = self.entries.get(segment.kind)?.get(segment.tag)?;
        match segment.rest {
            Some(rest) => child.sub_index(rest),
            None => Some(child),
        }
    }

    /// True when a node exists at `path`, whether or not a path ended there.
    pub fn has(&self, path: &str) -> bool {
        self.sub_index(path).is_some()
    }

    /// True when exactly this path was added.
    pub fn has_exact(&self, path: &str) -> bool {
        self.sub_index(path).is_some_and(|node| node.terminal)
    }

    /// Returns whether `tag` exists at the current level, typed or untyped.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tag_types.contains_key(tag)
    }

    /// Returns whether non-empty type `kind` exists at the current level.
    pub fn has_type(&self, kind: &str) -> bool {
        !kind.is_empty() && self.entries.contains_key(kind)
    }

    /// Returns whether `tag` occurs at the current level or any descendant.
    pub fn has_tag_any_depth(&self, tag: &str) -> bool {
        self.has_tag(tag) || self.deep_tags.contains_key(tag)
    }

    /// Returns whether type `kind` occurs at this level or any descendant.
    pub fn has_type_any_depth(&self, kind: &str) -> bool {
        self.has_type(kind) || self.deep_types.contains_key(kind)
    }

    /// Tags at this level, bare or typed.
    pub fn tags(&self) -> impl Iterator<Item = &str> {
        self.tag_types.keys().map(|tag| &**tag)
    }

    /// Types at this level.
    pub fn types(&self) -> impl Iterator<Item = &str> {
        self.entries
            .keys()
            .filter(|kind| !kind.is_empty())
            .map(|kind| &**kind)
    }

    /// Iterates over tags directly under type `kind` at the current level.
    pub fn tags_of_type<'a>(&'a self, kind: &str) -> impl Iterator<Item = &'a str> + 'a {
        self.entries
            .get(kind)
            .into_iter()
            .flat_map(|tags| tags.keys())
            .map(|tag| &**tag)
    }

    /// Iterates over non-empty types under which `tag` occurs at this level.
    pub fn types_of_tag<'a>(&'a self, tag: &str) -> impl Iterator<Item = &'a str> + 'a {
        self.tag_types
            .get(tag)
            .into_iter()
            .flatten()
            .filter(|kind| !kind.is_empty())
            .map(|kind| &**kind)
    }

    /// `(type, tag, subtree)` for each entry at this level. Bare tags have
    /// an empty type.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str, &TagIndex)> {
        self.entries.iter().flat_map(|(kind, tags)| {
            tags.iter()
                .map(move |(tag, child)| (&**kind, &**tag, child))
        })
    }

    /// Every path in this index.
    pub fn paths(&self) -> Vec<String> {
        let mut output = Vec::with_capacity(self.path_count);
        self.collect_paths("", &mut output);
        output
    }

    /// Every path beneath `path`. With `full`, the results include `path`
    /// itself as a prefix; otherwise they are relative to it.
    pub fn paths_with_prefix(&self, path: &str, full: bool) -> Vec<String> {
        let Some(node) = self.sub_index(path) else {
            return Vec::new();
        };
        let prefix = if full { path.trim_end_matches('/') } else { "" };
        let mut output = Vec::with_capacity(node.path_count);
        for (kind, tag, child) in node.entries() {
            child.collect_paths(&join(prefix, &format_segment(kind, tag)), &mut output);
        }
        output
    }

    fn collect_paths(&self, prefix: &str, output: &mut Vec<String>) {
        if self.terminal && !prefix.is_empty() {
            output.push(prefix.to_owned());
        }
        for (kind, tag, child) in self.entries() {
            child.collect_paths(&join(prefix, &format_segment(kind, tag)), output);
        }
    }
}

fn join(prefix: &str, segment: &str) -> String {
    if prefix.is_empty() {
        segment.to_owned()
    } else {
        format!("{prefix}/{segment}")
    }
}

impl<'a> FromIterator<&'a str> for TagIndex {
    fn from_iter<I: IntoIterator<Item = &'a str>>(paths: I) -> Self {
        let mut index = Self::new();
        index.add_all(paths);
        index
    }
}

impl<'a> Extend<&'a str> for TagIndex {
    fn extend<I: IntoIterator<Item = &'a str>>(&mut self, paths: I) {
        self.add_all(paths);
    }
}
