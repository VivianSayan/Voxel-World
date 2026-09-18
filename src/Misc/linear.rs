//! Linear algebra for the voxel world.
//!
//! `Vector2`, `Vector3` and `Vector4` are generic over their component type,
//! so the same vector carries voxel coordinates (`i128`), directions and
//! gradients (`f64`), or anything else the world needs. Operations that only
//! make sense for real numbers, such as `norm` and `normalize`, are
//! implemented for `f64` only.
//!
//! All three types come from `implement_vector!`, so an operator added there
//! applies to every dimension at once. Whatever differs per dimension - the
//! 3D cross product, the axis constants - is written out after the macro
//! invocations.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shl, ShlAssign, Shr,
    ShrAssign, Sub, SubAssign,
};

/// Adds a list of expressions together. Folding the list this way avoids
/// starting from a zero value, which would force an extra bound on the
/// component type.
macro_rules! sum_terms {
    ($term:expr) => {
        $term
    };
    ($term:expr, $($rest:expr),+) => {
        $term + sum_terms!($($rest),+)
    };
}

macro_rules! implement_vector {
    ($name:ident, $count:literal, $tuple:ty, [$($component:ident),+]) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name<T>
        {
            $(pub $component: T,)+
        }

        impl<T> $name<T>
        {
            pub const fn new($($component: T),+) -> Self {
                Self { $($component),+ }
            }
        }

        impl<T> $name<T>
        where
            T: Copy,
        {
            /// The same value in all components.
            pub fn splat(value: T) -> Self {
                Self { $($component: value),+ }
            }
        }

        impl<T> $name<T>
        {
            /// The components in axis order. This consumes the vector rather
            /// than copying it, so it works for component types that are not
            /// `Copy` as well.
            pub fn to_array(self) -> [T; $count] {
                [$(self.$component),+]
            }
        }

        impl<T> From<[T; $count]> for $name<T>
        {
            fn from([$($component),+]: [T; $count]) -> Self {
                Self { $($component),+ }
            }
        }

        impl<T> From<$tuple> for $name<T>
        {
            fn from(($($component),+): $tuple) -> Self {
                Self { $($component),+ }
            }
        }

        // -------------------------------------------------------------------
        // Component-wise combinators
        // -------------------------------------------------------------------
        //
        // A vector of fixed width supports the part of the iterator vocabulary
        // that keeps its shape or collapses it entirely. It deliberately does
        // not implement `IntoIterator`: `filter` and friends have nowhere to
        // put their result, and a second `map` returning a lazy iterator rather
        // than a vector would be a trap. Reach for `to_array` when an actual
        // iterator is what is wanted.

        impl<T> $name<T>
        {
            /// Applies `f` to every component. The component type may change,
            /// so this is the general form of the casts further down.
            pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> $name<U> {
                $name { $($component: f(self.$component)),+ }
            }

            /// Applies `f` to matching components of two vectors. Component-wise
            /// minimum, maximum and comparison are all this with the right `f`.
            pub fn zip_map<U, V>(
                self,
                rhs: $name<U>,
                mut f: impl FnMut(T, U) -> V,
            ) -> $name<V> {
                $name { $($component: f(self.$component, rhs.$component)),+ }
            }

            /// Whether `predicate` holds for every component. Short-circuits.
            pub fn all(self, mut predicate: impl FnMut(T) -> bool) -> bool {
                $(predicate(self.$component))&&+
            }

            /// Whether `predicate` holds for at least one component.
            /// Short-circuits.
            pub fn any(self, mut predicate: impl FnMut(T) -> bool) -> bool {
                $(predicate(self.$component))||+
            }

            /// Folds the components in axis order, starting from x.
            pub fn fold<A>(self, initial: A, mut f: impl FnMut(A, T) -> A) -> A {
                let accumulator: A = initial;
                $(let accumulator: A = f(accumulator, self.$component);)+
                accumulator
            }
        }

        // -------------------------------------------------------------------
        // Bounds
        // -------------------------------------------------------------------
        //
        // `PartialOrd` rather than `Ord`, so that one set of definitions serves
        // both the integers the world uses for coordinates and `f64`. Two
        // inherent methods of the same name may not coexist even when their
        // bounds are mutually exclusive, so a separate float version is not an
        // option anyway.
        //
        // Every comparison against NaN is false, so a NaN component survives
        // here when it is in `self` and is dropped when it is in `other`.
        // Coordinates should never be NaN; `is_finite` is there to check.

        impl<T> $name<T>
        where
            T: PartialOrd,
        {
            /// The corner where each axis takes the smaller of the two values.
            /// With `component_max`, this grows an axis-aligned bounding box to
            /// cover a point.
            pub fn component_min(self, other: Self) -> Self {
                self.zip_map(other, |left, right| {
                    if right < left { right } else { left }
                })
            }

            /// The corner where each axis takes the larger of the two values.
            pub fn component_max(self, other: Self) -> Self {
                self.zip_map(other, |left, right| {
                    if right > left { right } else { left }
                })
            }

            /// The smallest component.
            pub fn min_component(self) -> T {
                self.fold_components(|left, right| {
                    if right < left { right } else { left }
                })
            }

            /// The largest component.
            pub fn max_component(self) -> T {
                self.fold_components(|left, right| {
                    if right > left { right } else { left }
                })
            }

            /// Confines every component to the box between `low` and `high`.
            pub fn clamp(self, low: Self, high: Self) -> Self {
                self.component_max(low).component_min(high)
            }
        }

        impl<T> $name<T>
        where
            T: PartialOrd + Copy,
        {
            /// Whether every component lies in the box that runs from `low` up
            /// to but not including `high`, which is the convention a voxel
            /// occupies its own coordinate and not the next one along.
            pub fn is_within(self, low: Self, high: Self) -> bool {
                $(
                    self.$component >= low.$component
                        && self.$component < high.$component
                )&&+
            }
        }

        impl<T> $name<T>
        {
            /// Reduces the components with `f`, in axis order. Private because
            /// it only exists to spell the four functions above once each.
            fn fold_components(self, mut f: impl FnMut(T, T) -> T) -> T {
                let [first, rest @ ..]: [T; $count] = self.to_array();
                let mut accumulator: T = first;

                for component in rest {
                    accumulator = f(accumulator, component);
                }

                accumulator
            }
        }

        // -------------------------------------------------------------------
        // Products
        // -------------------------------------------------------------------

        impl<T> $name<T>
        where
            T: Add<Output = T> + Mul<Output = T> + Copy,
        {
            pub fn dot(self, rhs: Self) -> T {
                sum_terms!($(self.$component * rhs.$component),+)
            }

            /// Squared length. Prefer this to `norm` when comparing distances:
            /// it needs no square root and stays exact for integer components.
            pub fn norm_squared(self) -> T {
                self.dot(self)
            }
        }

        // -------------------------------------------------------------------
        // Arithmetic
        // -------------------------------------------------------------------

        impl<T> Add for $name<T>
        where
            T: Add<Output = T>,
        {
            type Output = $name<T>;

            fn add(self, rhs: Self) -> Self::Output {
                $name {
                    $($component: self.$component + rhs.$component),+
                }
            }
        }

        impl<T> Sub for $name<T>
        where
            T: Sub<Output = T>,
        {
            type Output = $name<T>;

            fn sub(self, rhs: Self) -> Self::Output {
                $name {
                    $($component: self.$component - rhs.$component),+
                }
            }
        }

        impl<T> Neg for $name<T>
        where
            T: Neg<Output = T>,
        {
            type Output = $name<T>;

            fn neg(self) -> Self::Output {
                $name {
                    $($component: -self.$component),+
                }
            }
        }

        impl<T> Mul<T> for $name<T>
        where
            T: Mul<Output = T> + Copy,
        {
            type Output = $name<T>;

            fn mul(self, scalar: T) -> Self::Output {
                $name {
                    $($component: self.$component * scalar),+
                }
            }
        }

        impl<T> Div<T> for $name<T>
        where
            T: Div<Output = T> + Copy,
        {
            type Output = $name<T>;

            fn div(self, scalar: T) -> Self::Output {
                $name {
                    $($component: self.$component / scalar),+
                }
            }
        }

        impl<T> AddAssign for $name<T>
        where
            T: Add<Output = T> + Copy,
        {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<T> SubAssign for $name<T>
        where
            T: Sub<Output = T> + Copy,
        {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<T> MulAssign<T> for $name<T>
        where
            T: Mul<Output = T> + Copy,
        {
            fn mul_assign(&mut self, scalar: T) {
                *self = *self * scalar;
            }
        }

        impl<T> DivAssign<T> for $name<T>
        where
            T: Div<Output = T> + Copy,
        {
            fn div_assign(&mut self, scalar: T) {
                *self = *self / scalar;
            }
        }

        // -------------------------------------------------------------------
        // Bit shifts
        // -------------------------------------------------------------------

        impl<T> Shl<u32> for $name<T>
        where
            T: Shl<u32, Output = T>,
        {
            type Output = $name<T>;

            fn shl(self, rhs: u32) -> Self::Output {
                $name {
                    $($component: self.$component << rhs),+
                }
            }
        }

        impl<T> Shr<u32> for $name<T>
        where
            T: Shr<u32, Output = T>,
        {
            type Output = $name<T>;

            fn shr(self, rhs: u32) -> Self::Output {
                $name {
                    $($component: self.$component >> rhs),+
                }
            }
        }

        impl<T> ShlAssign<u32> for $name<T>
        where
            T: Shl<u32, Output = T> + Copy,
        {
            fn shl_assign(&mut self, rhs: u32) {
                *self = *self << rhs;
            }
        }

        impl<T> ShrAssign<u32> for $name<T>
        where
            T: Shr<u32, Output = T> + Copy,
        {
            fn shr_assign(&mut self, rhs: u32) {
                *self = *self >> rhs;
            }
        }
    };
}

implement_vector!(Vector2, 2, (T, T), [x, y]);
implement_vector!(Vector3, 3, (T, T, T), [x, y, z]);
implement_vector!(Vector4, 4, (T, T, T, T), [x, y, z, w]);

// ---------------------------------------------------------------------------
// Three dimensions only
// ---------------------------------------------------------------------------

impl<T> Vector3<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
{
    /// The vector perpendicular to both inputs, right-handed.
    pub fn cross(self, rhs: Self) -> Self {
        Vector3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

// ---------------------------------------------------------------------------
// Real-valued components
// ---------------------------------------------------------------------------

macro_rules! implement_real_vector {
    ($name:ident, $scalar:ty, [$($component:ident),+]) => {
        impl $name<$scalar> {
            pub const ZERO: Self = Self { $($component: 0.0),+ };
            pub const ONE: Self = Self { $($component: 1.0),+ };

            pub fn norm(self) -> $scalar {
                self.norm_squared().sqrt()
            }

            /// Scales the vector to length 1. The result is not finite when
            /// the vector is zero, so check `norm_squared` first if the input
            /// may be degenerate.
            pub fn normalize(self) -> Self {
                self / self.norm()
            }

            pub fn distance(self, other: Self) -> $scalar {
                (self - other).norm()
            }

            pub fn distance_squared(self, other: Self) -> $scalar {
                (self - other).norm_squared()
            }

            /// Moves from `self` towards `other` by a fraction `t`.
            pub fn lerp(self, other: Self, t: $scalar) -> Self {
                self + (other - self) * t
            }

            /// The absolute value of every component.
            pub fn abs(self) -> Self {
                self.map(<$scalar>::abs)
            }

            /// The part of `self` that lies along `other`: the shadow `self`
            /// casts on that direction. Not finite when `other` is zero.
            pub fn project_onto(self, other: Self) -> Self {
                other * (self.dot(other) / other.norm_squared())
            }

            /// What projection leaves behind: the part of `self` at right
            /// angles to `other`. Adding this to `project_onto` gives `self`.
            pub fn reject_from(self, other: Self) -> Self {
                self - self.project_onto(other)
            }

            /// How far `self` reaches along `other`, as a signed length.
            /// Negative when `self` points against `other`.
            pub fn scalar_projection(self, other: Self) -> $scalar {
                self.dot(other) / other.norm()
            }

            pub fn is_finite(self) -> bool {
                self.to_array().into_iter().all(|value| value.is_finite())
            }
        }
    };
}

// Only f64 is enabled. With an f32 impl as well, a bare literal vector such
// as `Vector3::new(1.0, 2.0, 3.0).norm()` becomes ambiguous: the literal
// matches both impls, and method lookup fails before f64 fallback applies.
// Add the f32 invocations if f32 vectors are ever needed, and expect to
// annotate literals at the call sites.
implement_real_vector!(Vector2, f64, [x, y]);
implement_real_vector!(Vector3, f64, [x, y, z]);
implement_real_vector!(Vector4, f64, [x, y, z, w]);

// ---------------------------------------------------------------------------
// Axis constants
// ---------------------------------------------------------------------------

impl Vector2<f64> {
    pub const X: Self = Self::new(1.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0);
}

impl Vector3<f64> {
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);
}

impl Vector4<f64> {
    pub const X: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);
    pub const Z: Self = Self::new(0.0, 0.0, 1.0, 0.0);
    pub const W: Self = Self::new(0.0, 0.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Casting between dimensions
// ---------------------------------------------------------------------------
//
// Growing keeps the components there are and fills the rest with
// `T::default()`, which is zero for every numeric type. Shrinking drops the
// trailing components.

impl<T> From<Vector2<T>> for Vector3<T>
where
    T: Default,
{
    fn from(value: Vector2<T>) -> Self {
        Self::new(value.x, value.y, T::default())
    }
}

impl<T> From<Vector2<T>> for Vector4<T>
where
    T: Default,
{
    fn from(value: Vector2<T>) -> Self {
        Self::new(value.x, value.y, T::default(), T::default())
    }
}

impl<T> From<Vector3<T>> for Vector4<T>
where
    T: Default,
{
    fn from(value: Vector3<T>) -> Self {
        Self::new(value.x, value.y, value.z, T::default())
    }
}

impl<T> From<Vector3<T>> for Vector2<T>
{
    fn from(value: Vector3<T>) -> Self {
        Self::new(value.x, value.y)
    }
}

impl<T> From<Vector4<T>> for Vector3<T>
{
    fn from(value: Vector4<T>) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}

impl<T> From<Vector4<T>> for Vector2<T>
{
    fn from(value: Vector4<T>) -> Self {
        Self::new(value.x, value.y)
    }
}

// ---------------------------------------------------------------------------
// Casting between component types
// ---------------------------------------------------------------------------
//
// These cannot be generic, because `as` is an operator rather than a trait,
// so each pair is spelled out. The conversion is exactly what `as` does on
// the components: float to integer truncates towards zero and saturates at
// the integer's limits, so -2.7 becomes -2 and 1e30 becomes i32::MAX.

macro_rules! implement_component_cast {
    ($name:ident, [$($component:ident),+], $from:ty, $to:ty) => {
        impl From<$name<$from>> for $name<$to> {
            fn from(value: $name<$from>) -> Self {
                Self {
                    $($component: value.$component as $to),+
                }
            }
        }
    };
}

macro_rules! implement_component_cast_both_ways {
    ($left:ty, $right:ty) => {
        implement_component_cast!(Vector2, [x, y], $left, $right);
        implement_component_cast!(Vector2, [x, y], $right, $left);
        implement_component_cast!(Vector3, [x, y, z], $left, $right);
        implement_component_cast!(Vector3, [x, y, z], $right, $left);
        implement_component_cast!(Vector4, [x, y, z, w], $left, $right);
        implement_component_cast!(Vector4, [x, y, z, w], $right, $left);
    };
}

implement_component_cast_both_ways!(f64, f32);

implement_component_cast_both_ways!(f64, i32);
implement_component_cast_both_ways!(f64, i64);
implement_component_cast_both_ways!(f64, i128);
implement_component_cast_both_ways!(f64, u32);
implement_component_cast_both_ways!(f64, u64);
implement_component_cast_both_ways!(f64, usize);

implement_component_cast_both_ways!(f32, i32);
implement_component_cast_both_ways!(f32, i64);
implement_component_cast_both_ways!(f32, i128);
implement_component_cast_both_ways!(f32, u32);
implement_component_cast_both_ways!(f32, u64);
implement_component_cast_both_ways!(f32, usize);
