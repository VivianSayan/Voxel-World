// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Borrowed values or elements: none, one, or a whole set.
pub enum RefIter<'a, T> {
    /// Iterator with no values.
    Empty,
    /// Iterator over zero or one borrowed value.
    One(Option<&'a T>),
    /// Iterator over a borrowed set of values.
    Set(std::collections::hash_set::Iter<'a, T>),
}

impl<'a, T> RefIter<'a, T> {
    fn from_set(set: Option<&'a Set<T>>) -> Self {
        match set {
            Some(set) => Self::Set(set.into_iter()),
            None => Self::Empty,
        }
    }
}

impl<'a, T> Iterator for RefIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match self {
            Self::Empty => None,
            Self::One(item) => item.take(),
            Self::Set(items) => items.next(),
        }
    }
}

#[derive(Clone, Debug)]
enum Storage<E, V> {
    Flag(Set<E>),
    SingleShared(GroupedSingleMap<E, V>),
    SingleUnique(BiMap<E, V>),
    MultiShared(GroupedMultiMap<E, V>),
    MultiUnique(UniqueMultiMap<E, V>),
}

impl<E: Element, V: Element> Storage<E, V> {
    fn new(kind: PropertyKind) -> Self {
        match kind {
            PropertyKind::Flag => Self::Flag(Set::new()),
            PropertyKind::Single => Self::SingleShared(GroupedSingleMap::new()),
            PropertyKind::UniqueSingle => Self::SingleUnique(BiMap::new()),
            PropertyKind::Multi => Self::MultiShared(GroupedMultiMap::new()),
            PropertyKind::UniqueMulti => Self::MultiUnique(UniqueMultiMap::new()),
        }
    }

    fn kind(&self) -> PropertyKind {
        match self {
            Self::Flag(_) => PropertyKind::FLAG,
            Self::SingleShared(_) => PropertyKind::SINGLE,
            Self::SingleUnique(_) => PropertyKind::UNIQUE_SINGLE,
            Self::MultiShared(_) => PropertyKind::MULTI,
            Self::MultiUnique(_) => PropertyKind::UNIQUE_MULTI,
        }
    }

    fn member_count(&self) -> usize {
        match self {
            Self::Flag(set) => set.len(),
            Self::SingleShared(map) => map.len(),
            Self::SingleUnique(map) => map.len(),
            Self::MultiShared(map) => map.len(),
            Self::MultiUnique(map) => map.len(),
        }
    }

    fn contains_member(&self, element: &E) -> bool {
        match self {
            Self::Flag(set) => set.contains(element),
            Self::SingleShared(map) => map.contains(element),
            Self::SingleUnique(map) => map.contains_left(element),
            Self::MultiShared(map) => map.contains(element),
            Self::MultiUnique(map) => map.contains_key(element),
        }
    }

    fn has_value(&self, value: &V) -> bool {
        match self {
            Self::Flag(_) => false,
            Self::SingleShared(map) => map.contains_label(value),
            Self::SingleUnique(map) => map.contains_right(value),
            Self::MultiShared(map) => map.contains_label(value),
            Self::MultiUnique(map) => map.contains_value(value),
        }
    }

    fn has_pair(&self, element: &E, value: &V) -> bool {
        match self {
            Self::Flag(_) => false,
            Self::SingleShared(map) => map.contains_pair(element, value),
            Self::SingleUnique(map) => map.contains_pair(element, value),
            Self::MultiShared(map) => map.contains_pair(element, value),
            Self::MultiUnique(map) => map.contains_pair(element, value),
        }
    }

    fn get(&self, element: &E) -> Option<&V> {
        match self {
            Self::SingleShared(map) => map.get(element),
            Self::SingleUnique(map) => map.get_by_left(element),
            _ => None,
        }
    }

    fn values(&self, element: &E) -> RefIter<'_, V> {
        match self {
            Self::Flag(_) => RefIter::Empty,
            Self::SingleShared(map) => RefIter::One(map.get(element)),
            Self::SingleUnique(map) => RefIter::One(map.get_by_left(element)),
            Self::MultiShared(map) => RefIter::from_set(map.get(element)),
            Self::MultiUnique(map) => RefIter::from_set(map.get(element)),
        }
    }

    fn members_with(&self, value: &V) -> RefIter<'_, E> {
        match self {
            Self::Flag(_) => RefIter::Empty,
            Self::SingleShared(map) => RefIter::from_set(map.group(value)),
            Self::SingleUnique(map) => RefIter::One(map.get_by_right(value)),
            Self::MultiShared(map) => RefIter::from_set(map.group(value)),
            Self::MultiUnique(map) => RefIter::One(map.key_of(value)),
        }
    }

    fn value_member_count(&self, value: &V) -> usize {
        match self {
            Self::SingleShared(map) => map.group_len(value),
            Self::MultiShared(map) => map.group_len(value),
            _ => self.has_value(value) as usize,
        }
    }

    fn members(&self) -> Box<dyn Iterator<Item = &E> + '_> {
        match self {
            Self::Flag(set) => Box::new(set.iter()),
            Self::SingleShared(map) => Box::new(map.elements()),
            Self::SingleUnique(map) => Box::new(map.left_values()),
            Self::MultiShared(map) => Box::new(map.elements()),
            Self::MultiUnique(map) => Box::new(map.keys()),
        }
    }

    /// The element currently holding `value`, for unique properties.
    fn owner_of(&self, value: &V) -> Option<&E> {
        match self {
            Self::SingleUnique(map) => map.get_by_right(value),
            Self::MultiUnique(map) => map.key_of(value),
            _ => None,
        }
    }

    fn remove_member(&mut self, element: &E) -> bool {
        match self {
            Self::Flag(set) => set.remove(element),
            Self::SingleShared(map) => map.remove(element).is_some(),
            Self::SingleUnique(map) => map.remove_by_left(element).is_some(),
            Self::MultiShared(map) => map.remove(element).is_some(),
            Self::MultiUnique(map) => map.remove(element).is_some(),
        }
    }

    fn remove_value(&mut self, element: &E, value: &V) -> bool {
        match self {
            Self::Flag(_) => false,
            Self::SingleShared(map) => map.remove_pair(element, value),
            Self::SingleUnique(map) => map.remove_pair(element, value),
            Self::MultiShared(map) => map.remove_pair(element, value),
            Self::MultiUnique(map) => map.remove_pair(element, value),
        }
    }

    fn set_flag(&mut self, element: E) {
        if let Self::Flag(set) = self {
            set.insert(element);
        }
    }

    /// Makes `value` the element's only value, taking it from any other
    /// owner of a unique property.
    fn put_single(&mut self, element: E, value: V) {
        match self {
            Self::Flag(set) => {
                set.insert(element);
            }
            Self::SingleShared(map) => {
                map.insert(element, value);
            }
            Self::SingleUnique(map) => {
                map.insert(element, value);
            }
            Self::MultiShared(map) => map.replace_labels(&element, &Set::from([value])),
            Self::MultiUnique(map) => {
                map.remove(&element);
                map.insert_or_move(element, value);
            }
        }
    }

    /// Adds `value` to a multi property; replaces the value of a single one.
    fn put_value(&mut self, element: E, value: V) {
        match self {
            Self::MultiShared(map) => {
                map.insert(element, value);
            }
            Self::MultiUnique(map) => {
                map.insert_or_move(element, value);
            }
            _ => self.put_single(element, value),
        }
    }
}

impl<E: Element, V: Element> PartialEq for Storage<E, V> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Flag(a), Self::Flag(b)) => a == b,
            (Self::SingleShared(a), Self::SingleShared(b)) => a == b,
            (Self::SingleUnique(a), Self::SingleUnique(b)) => a == b,
            (Self::MultiShared(a), Self::MultiShared(b)) => a == b,
            (Self::MultiUnique(a), Self::MultiUnique(b)) => a == b,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
