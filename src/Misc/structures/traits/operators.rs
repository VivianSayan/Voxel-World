//! Operator overloads derived from `SetAlgebra`, matching the ones std
//! provides for `&HashSet`: `|` union, `&` intersection, `-` difference,
//! `^` symmetric difference.

macro_rules! impl_set_operators {
    ([$($generics:tt)*] $ty:ty) => {
        impl<$($generics)*> std::ops::BitOr<&$ty> for &$ty {
            type Output = $ty;

            fn bitor(self, other: &$ty) -> $ty {
                $crate::misc::structures::traits::SetAlgebra::union(self, other)
            }
        }

        impl<$($generics)*> std::ops::BitAnd<&$ty> for &$ty {
            type Output = $ty;

            fn bitand(self, other: &$ty) -> $ty {
                $crate::misc::structures::traits::SetAlgebra::intersection(self, other)
            }
        }

        impl<$($generics)*> std::ops::Sub<&$ty> for &$ty {
            type Output = $ty;

            fn sub(self, other: &$ty) -> $ty {
                $crate::misc::structures::traits::SetAlgebra::difference(self, other)
            }
        }

        impl<$($generics)*> std::ops::BitXor<&$ty> for &$ty {
            type Output = $ty;

            fn bitxor(self, other: &$ty) -> $ty {
                $crate::misc::structures::traits::SetAlgebra::symmetric_difference(self, other)
            }
        }
    };
}

pub(crate) use impl_set_operators;
