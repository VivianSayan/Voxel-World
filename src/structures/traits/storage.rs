//! Common interfaces for handle-addressed storage.

/// A store that owns values and returns opaque handles used to reach them.
pub trait HandleStore {
    /// A handle issued by the store.
    type Handle: Copy;
    /// The stored value.
    type Value;

    /// Number of live values.
    fn len(&self) -> usize;
    /// Stores a value and returns its handle.
    fn insert(&mut self, value: Self::Value) -> Self::Handle;
    /// Resolves a handle.
    fn get(&self, handle: Self::Handle) -> Option<&Self::Value>;
    /// Resolves a handle for mutation.
    fn get_mut(&mut self, handle: Self::Handle) -> Option<&mut Self::Value>;
    /// Removes and returns the value addressed by a handle.
    fn remove(&mut self, handle: Self::Handle) -> Option<Self::Value>;

    /// Whether no live values are held.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
