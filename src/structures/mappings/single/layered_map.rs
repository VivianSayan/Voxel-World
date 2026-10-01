//! Several maps read as one, the earliest holding a key winning.

use crate::structures::hashing::FastHashMap;
use crate::structures::traits::{
    ContentHashable, DeterministicMapOrder, Element, Map, StableHash, stable_hash_unordered,
};
use crate::units::ContentHash;
use std::fmt;

/// An ordered stack of maps read as one, where the earliest layer holding a key
/// decides its value.
///
/// # What it is for
///
/// Layered settings, where a value comes from whichever source outranks the
/// rest and the sources stay separate so any of them can be replaced on its
/// own:
///
/// ```text
/// layer 0   runtime override
/// layer 1   world or module configuration
/// layer 2   engine defaults
/// ```
///
/// or inherited properties, where an instance falls back on what it was made
/// from:
///
/// ```text
/// layer 0   this creature
/// layer 1   its archetype
/// layer 2   base defaults
/// ```
///
/// Nothing is merged or copied. The layers stay as they are, and dropping the
/// override layer restores the configuration underneath exactly, which is the
/// point: a merged map cannot be unmerged.
///
/// # Precedence
///
/// A look-up walks the layers from first to last and stops at the first one
/// holding the key. Precedence is decided by layer order and never by anything
/// else — in particular never by the hash order within a layer.
///
/// ```text
/// layer 0: { volume: 80 }
/// layer 1: { volume: 50, fullscreen: false }
///
/// get("volume")     -> 80      (layer 0 outranks layer 1)
/// get("fullscreen") -> false   (only layer 1 has it)
/// ```
///
/// # Shadowing
///
/// A key held by more than one layer is visible only at the earliest of them.
/// The rest still exist, untouched, and come back into view when the one above
/// them goes:
///
/// ```text
/// layer 0: { x: 10 }
/// layer 1: { x: 20 }
///
/// get(x)    -> 10
/// remove(x)            removes layer 0's occurrence only
/// get(x)    -> 20      layer 1's was never touched
/// ```
///
/// That is the invariant the whole structure rests on: **the value for a key is
/// the one in the earliest layer holding it, and lower occurrences stay intact
/// and become visible again when the ones above them are removed.**
/// [`LayeredMap::remove_everywhere`] is for when a key really should be gone.
///
/// # How this differs from Python's `ChainMap`
///
/// Python's `ChainMap` writes everything to the first map, so assigning to a
/// key that lives in a lower map shadows it with a new entry in the top one.
/// Here a write to an existing key goes to **the layer that already holds it**:
///
/// ```text
/// layer 0: {}
/// layer 1: { x: 10 }
/// layer 2: { x: 20 }
///
/// insert(x, 15)
///
/// layer 0: {}
/// layer 1: { x: 15 }     <- changed in place
/// layer 2: { x: 20 }     <- still shadowed
/// ```
///
/// A key held by no layer is new, so it goes into layer 0. The difference
/// matters when layers are saved separately: editing a configuration value
/// should change the configuration, not pile a runtime override on top of it
/// that outlives the edit. [`LayeredMap::insert_into`] writes to a chosen layer
/// where the Python behaviour is what is wanted.
///
/// # When a plain map is simpler
///
/// Almost always, if the layers are never taken apart. A `FastHashMap` built by
/// merging the sources is smaller, and every operation on it is a single hash
/// look-up rather than one per layer. Reach for this only when the layers have
/// to stay separable — because one of them is reloaded, swapped, or saved by
/// itself — or when what is shadowed has to remain recoverable.
///
/// # What it costs
///
/// Everything here is proportional to the number of layers, which is expected
/// to be small. A look-up is at most one hash look-up per layer, and stops at
/// the first hit. Nothing is cached: [`LayeredMap::layer_mut`] hands out the
/// backing maps, so any index this structure kept could be invalidated behind
/// its back, and an index that is sometimes wrong is worse than none.
///
/// The consequence is that [`LayeredMap::len`] is **not** O(1) — it counts
/// distinct visible keys, which means walking every pair in every layer, though
/// it takes the obvious shortcut when there is only one layer. Visible
/// iteration is lazy and allocates nothing: a pair is visible when no earlier
/// layer holds its key, which costs a look-up per earlier layer. Prefer
/// [`LayeredMap::contains_key`] to `len() > 0` on a key, and
/// [`LayeredMap::is_empty`], which stops at the first pair it finds.
///
/// # Invariant
///
/// There is always at least one layer, so there is always somewhere for a new
/// key to go and insertion cannot fail. [`LayeredMap::remove_layer`] refuses to
/// remove the last one.
///
/// # Example
///
/// ```
/// use voxel_world::structures::hashing::FastHashMap;
/// use voxel_world::structures::mappings::LayeredMap;
///
/// let mut defaults: FastHashMap<&str, u32> = FastHashMap::default();
/// defaults.insert("volume", 50);
/// defaults.insert("draw_distance", 8);
///
/// let mut settings: LayeredMap<&str, u32> = LayeredMap::new();
/// settings.push_layer(defaults);
///
/// // The override layer is on top and starts empty.
/// assert_eq!(settings.get(&"volume"), Some(&50));
/// settings.insert("volume", 80);
/// assert_eq!(settings.get(&"volume"), Some(&80));
///
/// // Written to the layer that already held it, not piled on top.
/// assert_eq!(settings.find_layer(&"volume"), Some(1));
/// assert_eq!(settings.layer_count(), 2);
///
/// // A key nothing holds is new, so it goes to the top layer.
/// settings.insert("debug_overlay", 1);
/// assert_eq!(settings.find_layer(&"debug_overlay"), Some(0));
/// ```
#[derive(Clone)]
pub struct LayeredMap<K, V> {
    /// Highest precedence first. Never empty.
    layers: Vec<FastHashMap<K, V>>,
}

impl<K, V> Default for LayeredMap<K, V> {
    /// One empty layer.
    fn default() -> Self {
        Self {
            layers: vec![FastHashMap::default()],
        }
    }
}

// ---------------------------------------------------------------------------
// Building one
// ---------------------------------------------------------------------------

impl<K: Element, V> LayeredMap<K, V> {
    /// A map with one empty layer.
    pub fn new() -> Self {
        Self::default()
    }

    /// A map with `count` empty layers, or one if `count` is zero.
    pub fn with_layers(count: usize) -> Self {
        Self {
            layers: (0..count.max(1)).map(|_| FastHashMap::default()).collect(),
        }
    }

    /// A map over these layers, highest precedence first.
    ///
    /// An empty sequence gives one empty layer, since there is always at least
    /// one.
    pub fn from_layers(layers: impl IntoIterator<Item = FastHashMap<K, V>>) -> Self {
        let layers: Vec<FastHashMap<K, V>> = layers.into_iter().collect();

        if layers.is_empty() {
            return Self::new();
        }

        Self { layers }
    }
}

// ---------------------------------------------------------------------------
// The layers themselves
// ---------------------------------------------------------------------------

impl<K: Element, V> LayeredMap<K, V> {
    /// How many layers there are, always at least one.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// One layer as an ordinary map, or `None` for an index past the last.
    pub fn layer(&self, index: usize) -> Option<&FastHashMap<K, V>> {
        self.layers.get(index)
    }

    /// One layer to change directly, for work this structure has no opinion
    /// about: reloading a configuration file into its layer, or clearing the
    /// override layer wholesale.
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut FastHashMap<K, V>> {
        self.layers.get_mut(index)
    }

    /// Every layer, highest precedence first.
    pub fn layers(&self) -> impl ExactSizeIterator<Item = &FastHashMap<K, V>> {
        self.layers.iter()
    }

    /// Adds a layer below every existing one, as the new last resort.
    pub fn push_layer(&mut self, layer: FastHashMap<K, V>) {
        self.layers.push(layer);
    }

    /// Adds a layer at a position, pushing the ones at or after it down.
    ///
    /// An index equal to [`LayeredMap::layer_count`] appends. Returns whether
    /// the index was usable.
    pub fn insert_layer(&mut self, index: usize, layer: FastHashMap<K, V>) -> bool {
        if index > self.layers.len() {
            return false;
        }

        self.layers.insert(index, layer);

        true
    }

    /// Takes a layer out and hands it back, pulling the ones below it up.
    ///
    /// Returns `None` for an index past the last, and for the only layer, since
    /// a map with nowhere to put a new key would have to start failing
    /// insertions.
    pub fn remove_layer(&mut self, index: usize) -> Option<FastHashMap<K, V>> {
        if self.layers.len() == 1 || index >= self.layers.len() {
            return None;
        }

        Some(self.layers.remove(index))
    }

    /// Exchanges the precedence of two layers. Returns whether both existed.
    pub fn swap_layers(&mut self, first: usize, second: usize) -> bool {
        if first >= self.layers.len() || second >= self.layers.len() {
            return false;
        }

        self.layers.swap(first, second);

        true
    }

    /// Moves a layer to another position, sliding the layers in between along.
    ///
    /// Unlike [`LayeredMap::swap_layers`] this keeps the relative order of
    /// everything else, which is what reordering a precedence chain usually
    /// means. Returns whether both positions existed.
    pub fn move_layer(&mut self, from: usize, to: usize) -> bool {
        if from >= self.layers.len() || to >= self.layers.len() {
            return false;
        }

        let layer: FastHashMap<K, V> = self.layers.remove(from);
        self.layers.insert(to, layer);

        true
    }
}

// ---------------------------------------------------------------------------
// Reading through the layers
// ---------------------------------------------------------------------------

impl<K: Element, V> LayeredMap<K, V> {
    /// The value for a key: the one in the earliest layer holding it.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.layers.iter().find_map(|layer| layer.get(key))
    }

    /// The value for a key, to change in place, in the earliest layer holding
    /// it.
    ///
    /// Changing it here does not move it between layers, so what was shadowed
    /// stays shadowed.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.layers.iter_mut().find_map(|layer| layer.get_mut(key))
    }

    /// Whether any layer holds the key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.layers.iter().any(|layer| layer.contains_key(key))
    }

    /// Which layer a key's value comes from.
    pub fn find_layer(&self, key: &K) -> Option<usize> {
        self.layers.iter().position(|layer| layer.contains_key(key))
    }

    /// The value one particular layer holds for a key, ignoring precedence.
    pub fn get_from_layer(&self, index: usize, key: &K) -> Option<&V> {
        self.layers.get(index)?.get(key)
    }

    /// Every layer holding the key and what it holds, highest precedence first.
    ///
    /// The first is the visible one and the rest are shadowed, so
    /// `occurrences(key).skip(1)` is what a removal would uncover in turn.
    pub fn occurrences<'a>(&'a self, key: &'a K) -> impl Iterator<Item = (usize, &'a V)> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(move |(index, layer)| layer.get(key).map(|value| (index, value)))
    }

    /// Whether the key's value is hiding another one below it.
    ///
    /// True when more than one layer holds the key, which means removing the
    /// visible value uncovers another rather than removing the key.
    pub fn is_shadowed(&self, key: &K) -> bool {
        self.occurrences(key).nth(1).is_some()
    }

    /// How many distinct keys are visible.
    ///
    /// Walks every pair in every layer to discount shadowed ones, so this is
    /// not the constant-time `len` an ordinary map has; see
    /// [`LayeredMap::total_len`] for the cheap count and
    /// [`LayeredMap::is_empty`] for the cheap emptiness test.
    pub fn len(&self) -> usize {
        match self.layers.as_slice() {
            [single] => single.len(),
            _ => self.iter().count(),
        }
    }

    /// Whether no layer holds anything.
    ///
    /// Constant time in the number of layers: a visible key exists exactly when
    /// any layer is non-empty.
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(FastHashMap::is_empty)
    }

    /// How many pairs are stored across every layer, shadowed ones included.
    ///
    /// The cost of holding the layers separately, against
    /// [`LayeredMap::len`]'s count of what is actually visible.
    pub fn total_len(&self) -> usize {
        self.layers.iter().map(FastHashMap::len).sum()
    }

    /// Every visible pair, highest precedence layer first, each key once.
    ///
    /// Lazy and allocation-free: a pair is visible when no earlier layer holds
    /// its key, which costs a look-up per earlier layer. Within one layer the
    /// order is that layer's own, which is settled but not canonical — see
    /// [`DeterministicMapOrder`].
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(move |(index, layer)| {
                layer
                    .iter()
                    .filter(move |(key, _)| !self.covered_before(index, key))
            })
    }

    /// Every visible key, each once.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(key, _)| key)
    }

    /// Every visible value, one per visible key.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, value)| value)
    }

    /// The visible view as one ordinary map, leaving this one as it is.
    ///
    /// What to build once the layers have settled and the separation is no
    /// longer wanted: every later look-up is then a single hash look-up, and
    /// everything shadowed is dropped.
    pub fn flatten(&self) -> FastHashMap<K, V>
    where
        K: Clone,
        V: Clone,
    {
        self.iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    /// Whether an earlier layer than `index` holds the key.
    fn covered_before(&self, index: usize, key: &K) -> bool {
        self.layers[..index]
            .iter()
            .any(|layer| layer.contains_key(key))
    }
}

// ---------------------------------------------------------------------------
// Writing through the layers
// ---------------------------------------------------------------------------

impl<K: Element, V> LayeredMap<K, V> {
    /// Sets a key's value in the earliest layer holding it, or puts it in layer
    /// zero when no layer holds it.
    ///
    /// Returns the value it replaced. This is where the structure parts company
    /// with Python's `ChainMap`, which would always write to layer zero; see
    /// the type's documentation, and [`LayeredMap::insert_into`] for writing to
    /// a layer of the caller's choosing.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.find_layer(&key) {
            Some(index) => self.layers[index].insert(key, value),
            None => self.layers[0].insert(key, value),
        }
    }

    /// Sets a key's value in one particular layer, whatever the others hold.
    ///
    /// For putting an override on top of a value that already exists lower
    /// down, or for filling a layer that stands for one source. Returns the
    /// value it replaced in that layer, or gives the pair back untouched when
    /// there is no such layer.
    pub fn insert_into(&mut self, index: usize, key: K, value: V) -> Result<Option<V>, (K, V)> {
        match self.layers.get_mut(index) {
            Some(layer) => Ok(layer.insert(key, value)),
            None => Err((key, value)),
        }
    }

    /// Removes a key's visible occurrence, uncovering whatever was under it.
    ///
    /// Returns the value removed. The key remains present if a lower layer
    /// holds it, which is the point; [`LayeredMap::remove_everywhere`] is for
    /// making it gone.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let index: usize = self.find_layer(key)?;

        self.layers[index].remove(key)
    }

    /// Removes a key from one particular layer, whatever the others hold.
    pub fn remove_from_layer(&mut self, index: usize, key: &K) -> Option<V> {
        self.layers.get_mut(index)?.remove(key)
    }

    /// Removes a key from every layer, and returns how many went.
    ///
    /// Leaves nothing to uncover, so the key is absent afterwards.
    pub fn remove_everywhere(&mut self, key: &K) -> usize {
        self.layers
            .iter_mut()
            .filter(|layer| layer.contains_key(key))
            .map(|layer| layer.remove(key))
            .count()
    }

    /// Moves a key's visible value up into a higher-precedence layer.
    ///
    /// The value leaves the layer it was in, so nothing is shadowed by the move
    /// and what was under it stays under it. Returns whether it moved: `false`
    /// when no layer holds the key, when `layer` is past the last, or when
    /// `layer` is below where the key already is, which would be a demotion and
    /// could change what is visible.
    pub fn promote(&mut self, key: &K, layer: usize) -> bool {
        let Some(from) = self.find_layer(key) else {
            return false;
        };

        if layer >= self.layers.len() || layer > from {
            return false;
        }

        if layer == from {
            return true;
        }

        let Some(value) = self.layers[from].remove(key) else {
            return false;
        };

        self.layers[layer].insert(key.clone(), value);

        true
    }

    /// Empties every layer, keeping the layers themselves.
    ///
    /// The precedence chain is a structure the caller set up, so clearing the
    /// contents leaves it standing; use [`LayeredMap::remove_layer`] to take a
    /// layer out.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Makes room in layer zero for `additional` more new keys.
    ///
    /// Layer zero because that is where a key no layer holds goes. Reserving in
    /// any other layer is [`LayeredMap::layer_mut`] plus that map's own
    /// `reserve`.
    pub fn reserve(&mut self, additional: usize) {
        self.layers[0].reserve(additional);
    }

    /// Gives back the room no layer is using.
    pub fn shrink_to_fit(&mut self) {
        for layer in &mut self.layers {
            layer.shrink_to_fit();
        }
    }
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// The visible view: one value per key, from the earliest layer holding it.
impl<K: Element, V: Element> Map for LayeredMap<K, V> {
    type Key = K;
    type Value = V;
    type Mapped = V;

    /// Distinct visible keys, which is not constant time here; see
    /// [`LayeredMap::len`].
    fn len(&self) -> usize {
        LayeredMap::len(self)
    }

    fn is_empty(&self) -> bool {
        LayeredMap::is_empty(self)
    }

    fn contains_key(&self, key: &K) -> bool {
        LayeredMap::contains_key(self, key)
    }

    fn get(&self, key: &K) -> Option<&V> {
        LayeredMap::get(self, key)
    }

    fn contains_pair(&self, key: &K, value: &V) -> bool {
        LayeredMap::get(self, key) == Some(value)
    }

    fn keys(&self) -> impl Iterator<Item = &K> {
        LayeredMap::keys(self)
    }

    fn pairs(&self) -> impl Iterator<Item = (&K, &V)> {
        self.iter()
    }
}

/// Layer order fixes precedence, and each layer is hashed with a fixed hasher,
/// so the same sequence of operations iterates the same way every run.
///
/// Deliberately not [`CanonicalMapOrder`](crate::structures::traits::CanonicalMapOrder):
/// within a layer the order still follows that map's insertion history, so two
/// layered maps holding the same pairs can iterate differently.
impl<K: Element, V> DeterministicMapOrder for LayeredMap<K, V> {}

/// Over the visible pairs, so two layered maps with the same view agree however
/// their layers are arranged.
impl<K: Element + StableHash, V: StableHash> ContentHashable for LayeredMap<K, V> {
    fn content_hash(&self) -> ContentHash {
        stable_hash_unordered(self.iter())
    }
}

impl<K: Element, V> FromIterator<(K, V)> for LayeredMap<K, V> {
    /// One layer holding these pairs.
    fn from_iter<I: IntoIterator<Item = (K, V)>>(pairs: I) -> Self {
        Self {
            layers: vec![pairs.into_iter().collect()],
        }
    }
}

impl<K: Element + fmt::Debug, V: fmt::Debug> fmt::Debug for LayeredMap<K, V> {
    /// The visible view, as an ordinary map would print.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Element, V> fmt::Display for LayeredMap<K, V> {
    /// How much is visible, and how much is being kept underneath.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total: usize = self.total_len();
        let visible: usize = self.len();

        write!(
            formatter,
            "{visible} visible of {total} across {} layers",
            self.layers.len()
        )
    }
}
