//! Values that report which logical kind they are.

use super::collection::Element;
use crate::structures::hashing::hash_one;
use std::fmt;

/// A value type that can say which of its logical kinds a value is.
///
/// A property store is generic over one value type `V`. When `V` is a single
/// concrete type — a string, an integer — the compiler already guarantees that
/// every value in a property is the same kind of thing, and there is nothing to
/// check. The moment `V` becomes a sum type so that different properties can
/// hold different things, that guarantee is gone: `Health` and `Weather` have
/// the same Rust type, so nothing stops a weather being written to a health.
///
/// This trait is what closes that gap. A value type names its own kind type and
/// reports the kind of each value, and a schema can then say which kind a
/// property accepts and reject the rest.
///
/// # Why a trait rather than an enum in this crate
///
/// The kinds belong to the caller's world, not to this crate: `Weather` and
/// `Phase` mean nothing here. Naming the kind type as an associated type keeps
/// the check fully general — any value type, any set of kinds — while staying
/// type safe, because a property's declared kind and a value's reported kind are
/// the same Rust type and cannot be crossed between two different value types.
///
/// Rust cannot require that `V` *be* an enum. Requiring that it can report a
/// kind is the enforceable form of the same idea.
///
/// # Equality is guaranteed, not assumed
///
/// A property store holds values in hashed sets and compares them to answer
/// `Is` and to find owners, so equality on a value has to be total and
/// meaningful. That is not left to convention: a value type is [`Element`],
/// which is `Eq + Hash + Clone`, and deriving `Eq` on an enum asserts `Eq` for
/// every variant's payload in turn. A payload that has no total equality
/// therefore keeps the whole value type out of a store, at compile time:
///
/// ```compile_fail
/// use voxel_world::structures::indices::PropertyQuery;
///
/// // `f64` has no total equality, so this cannot derive `Eq` ...
/// #[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// enum Value {
///     Depth(f64),
/// }
///
/// // ... and therefore cannot be a property store's value type.
/// let _: PropertyQuery<u32, &str, Value> = PropertyQuery::new();
/// ```
///
/// So `a == b` is always safe to write on a value type a store accepts, and
/// always means what it says about the payloads.
///
/// # Contract
///
/// The compiler guarantees the bound; these two are on the implementor, and
/// matter only for a hand-written `Eq` or `kind`:
///
/// 1. **A kind is stable.** `value.kind()` gives the same answer every time for
///    the same value.
/// 2. **Equal values have equal kinds.** If `a == b` then
///    `a.kind() == b.kind()`.
///
/// The second is what ties equality to the schema. Storage looks values up by
/// equality while the schema checks them by kind, so a pair that is equal but
/// differently kinded could be stored under one kind and found under another —
/// exactly the state the schema exists to prevent. A derived `Eq` satisfies both
/// for free, since it compares the variant before the payload.
///
/// [`check_kinds`] verifies both over a sample, for a value type whose `Eq` or
/// `kind` is written by hand.
///
/// # The single-kind case
///
/// A type with only one logical kind implements this with `Kind = ()`. The
/// check still happens and is always satisfied, which costs a comparison of two
/// zero-sized values — nothing at run time. It is implemented here for the
/// standard types that are usable as values, so a store over `String` or `u64`
/// needs no work from the caller.
///
/// ```
/// use voxel_world::structures::traits::Kinded;
///
/// #[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// enum Value {
///     Count(u64),
///     Name(String),
/// }
///
/// #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
/// enum ValueKind {
///     Count,
///     Name,
/// }
///
/// impl Kinded for Value {
///     type Kind = ValueKind;
///
///     fn kind(&self) -> ValueKind {
///         match self {
///             Self::Count(_) => ValueKind::Count,
///             Self::Name(_) => ValueKind::Name,
///         }
///     }
/// }
/// ```
pub trait Kinded {
    /// The set of kinds values of this type come in.
    ///
    /// [`Debug`](fmt::Debug) because a rejected write reports the kind it
    /// expected and the kind it was given, and a schema error that cannot name
    /// them is of little use.
    type Kind: Element + fmt::Debug;

    /// Which kind this value is.
    fn kind(&self) -> Self::Kind;
}

/// Types with a single logical kind, for which the schema check is a formality
/// the compiler has already made.
macro_rules! implement_single_kind {
    ($($type:ty),* $(,)?) => {
        $(
            impl Kinded for $type {
                type Kind = ();

                fn kind(&self) -> Self::Kind {}
            }
        )*
    };
}

implement_single_kind!(
    bool, char, String, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize,
);

impl Kinded for &str {
    type Kind = ();

    fn kind(&self) -> Self::Kind {}
}

// ---------------------------------------------------------------------------
// Checking a hand-written implementation
// ---------------------------------------------------------------------------

/// A way a value type broke the [`Kinded`] contract.
///
/// Values are named by their position in the sample that was checked, so that
/// reporting a violation does not require the value type to be printable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KindContract {
    /// A value did not equal itself, so its equality is not reflexive and
    /// nothing can be found once it is stored.
    NotReflexive {
        /// Where the value sat in the sample.
        index: usize,
    },
    /// A value reported two different kinds on two calls.
    UnstableKind {
        /// Where the value sat in the sample.
        index: usize,
    },
    /// Two equal values reported different kinds, so one could be stored under
    /// a kind the other would be found under.
    KindDiffers {
        /// Where the first value sat in the sample.
        left: usize,
        /// Where the second sat.
        right: usize,
    },
    /// Two equal values hashed differently, which breaks every hashed
    /// collection holding them.
    HashDiffers {
        /// Where the first value sat in the sample.
        left: usize,
        /// Where the second sat.
        right: usize,
    },
}

/// Checks a sample of values against the [`Kinded`] contract.
///
/// For a value type whose `Eq`, `Hash` or [`Kinded::kind`] is written by hand
/// rather than derived: give it a sample covering every variant, including
/// pairs that ought to be equal and pairs that ought not, and call it from a
/// test. A derived implementation cannot fail this, so there is no reason to
/// check one.
///
/// Every pair is compared, so this is quadratic in the sample and belongs in a
/// test rather than in a hot path.
///
/// ```
/// use voxel_world::structures::traits::{Kinded, check_kinds};
///
/// #[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// enum Value {
///     Count(u64),
///     Name(&'static str),
/// }
///
/// impl Kinded for Value {
///     type Kind = bool;
///
///     fn kind(&self) -> bool {
///         matches!(self, Self::Count(_))
///     }
/// }
///
/// assert_eq!(
///     check_kinds(&[Value::Count(1), Value::Count(1), Value::Name("a")]),
///     Ok(()),
/// );
/// ```
pub fn check_kinds<V: Element + Kinded>(values: &[V]) -> Result<(), KindContract> {
    // Comparing a value with itself is the reflexivity check, not a mistake.
    #[allow(clippy::eq_op)]
    for (index, value) in values.iter().enumerate() {
        if value != value {
            return Err(KindContract::NotReflexive { index });
        }

        if value.kind() != value.kind() {
            return Err(KindContract::UnstableKind { index });
        }
    }

    for (left, first) in values.iter().enumerate() {
        for (right, second) in values.iter().enumerate().skip(left + 1) {
            if first != second {
                continue;
            }

            if first.kind() != second.kind() {
                return Err(KindContract::KindDiffers { left, right });
            }

            if hash_one(first) != hash_one(second) {
                return Err(KindContract::HashDiffers { left, right });
            }
        }
    }

    Ok(())
}
