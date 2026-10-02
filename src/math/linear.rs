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
//!
//! # What these are not
//!
//! None of them implement [`IntoIterator`]. A vector has a fixed width, so the
//! part of the iterator vocabulary that keeps its shape or collapses it
//! entirely is provided by hand ([`map`](Vector3::map),
//! [`zip_map`](Vector3::zip_map), [`all`](Vector3::all), [`any`](Vector3::any),
//! [`fold`](Vector3::fold)), while `filter` and friends have nowhere to put
//! their result. Call [`to_array`](Vector3::to_array) where a real iterator is
//! wanted.
//!
//! The comparison methods are bounded by [`PartialOrd`] rather than [`Ord`], so
//! that one set of definitions serves both the integers the world uses for
//! coordinates and `f64`. Every comparison against NaN is false, which means a
//! NaN component survives a component-wise minimum or maximum when it is in
//! `self` and is dropped when it is in the other vector. Coordinates should
//! never be NaN; [`is_finite`](Vector3::is_finite) is there to check.

use crate::math::fixed::Fixed;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign, Sub,
    SubAssign,
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
        #[doc = concat!("A ", stringify!($count), "-component vector, generic over its component type.")]
        ///
        /// Components are public and in axis order. `Eq` and `Hash` come from
        /// the component type, so integer vectors can key a map while `f64`
        /// ones cannot.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name<T>
        {
            $(pub $component: T,)+
        }

        impl<T> $name<T>
        {
            /// A vector from its components, in axis order.
            pub const fn new($($component: T),+) -> Self {
                Self { $($component),+ }
            }
        }

        impl<T> $name<T>
        where
            T: Copy,
        {
            /// The same value in every component, such as a cube's extent or a
            /// uniform scale.
            pub fn splat(value: T) -> Self {
                Self { $($component: value),+ }
            }
        }

        impl<T> $name<T>
        {
            /// The components in axis order, as an array.
            ///
            /// Consumes the vector rather than copying it, so it works for
            /// component types that are not `Copy`, and it is the way to reach
            /// a real iterator over the components.
            pub fn to_array(self) -> [T; $count] {
                [$(self.$component),+]
            }
        }

        /// From an array in axis order, the reverse of
        #[doc = concat!("[`", stringify!($name), "::to_array`].")]
        impl<T> From<[T; $count]> for $name<T>
        {
            fn from([$($component),+]: [T; $count]) -> Self {
                Self { $($component),+ }
            }
        }

        /// From a tuple in axis order.
        impl<T> From<$tuple> for $name<T>
        {
            fn from(($($component),+): $tuple) -> Self {
                Self { $($component),+ }
            }
        }

        // -------------------------------------------------------------------
        // Component-wise combinators
        // -------------------------------------------------------------------

        impl<T> $name<T>
        {
            /// Applies `f` to every component, in axis order.
            ///
            /// The component type may change, so this is the general form of
            /// the casts at the end of the file: `position.map(|v| v as f64)`
            /// is what those spell out for the common pairs.
            pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> $name<U> {
                $name { $($component: f(self.$component)),+ }
            }

            /// Applies `f` to matching components of two vectors, in axis
            /// order.
            ///
            /// Component-wise minimum, maximum and comparison are all this with
            /// the right `f`, and so is any arithmetic the operators do not
            /// already cover.
            pub fn zip_map<U, V>(
                self,
                rhs: $name<U>,
                mut f: impl FnMut(T, U) -> V,
            ) -> $name<V> {
                $name { $($component: f(self.$component, rhs.$component)),+ }
            }

            /// Whether `predicate` holds for every component.
            ///
            /// Short-circuits on the first component that fails, in axis
            /// order.
            pub fn all(self, mut predicate: impl FnMut(T) -> bool) -> bool {
                $(predicate(self.$component))&&+
            }

            /// Whether `predicate` holds for at least one component.
            ///
            /// Short-circuits on the first component that passes, in axis
            /// order.
            pub fn any(self, mut predicate: impl FnMut(T) -> bool) -> bool {
                $(predicate(self.$component))||+
            }

            /// Folds the components into a single value, in axis order,
            /// starting from `initial`.
            ///
            /// The general form of [`min_component`](Self::min_component) and
            /// the rest: those fold with no starting value, this one takes
            /// one, so the result may be of any type.
            pub fn fold<A>(self, initial: A, mut f: impl FnMut(A, T) -> A) -> A {
                let accumulator: A = initial;
                $(let accumulator: A = f(accumulator, self.$component);)+
                accumulator
            }
        }

        // -------------------------------------------------------------------
        // Bounds
        // -------------------------------------------------------------------

        impl<T> $name<T>
        where
            T: PartialOrd,
        {
            /// The corner where each axis takes the smaller of the two
            /// values.
            ///
            /// With [`component_max`](Self::component_max), this grows an
            /// axis-aligned bounding box to cover a point: the pair of corners
            /// is the box. A NaN in `self` survives and a NaN in `other` is
            /// dropped, since every comparison against NaN is false.
            pub fn component_min(self, other: Self) -> Self {
                self.zip_map(other, |left, right| {
                    if right < left { right } else { left }
                })
            }

            /// The corner where each axis takes the larger of the two values;
            /// see [`component_min`](Self::component_min).
            pub fn component_max(self, other: Self) -> Self {
                self.zip_map(other, |left, right| {
                    if right > left { right } else { left }
                })
            }

            /// The smallest component, comparing in axis order.
            pub fn min_component(self) -> T {
                self.fold_components(|left, right| {
                    if right < left { right } else { left }
                })
            }

            /// The largest component, comparing in axis order.
            pub fn max_component(self) -> T {
                self.fold_components(|left, right| {
                    if right > left { right } else { left }
                })
            }

            /// Every component confined to the box between `low` and `high`,
            /// inclusive at both ends.
            ///
            /// Bounds that cross on an axis give `high` on that axis, since the
            /// maximum is applied first.
            pub fn clamp(self, low: Self, high: Self) -> Self {
                self.component_max(low).component_min(high)
            }
        }

        impl<T> $name<T>
        where
            T: PartialOrd + Copy,
        {
            /// Whether every component lies in the box that runs from `low` up
            /// to but not including `high`.
            ///
            /// Half-open at the top, which is the convention throughout the
            /// world: a voxel occupies its own coordinate and not the next one
            /// along, so a chunk from `low` to `low + size` holds exactly
            /// `size` voxels per axis.
            pub fn is_within(self, low: Self, high: Self) -> bool {
                $(
                    self.$component >= low.$component
                        && self.$component < high.$component
                )&&+
            }
        }

        impl<T> $name<T>
        {
            /// Reduces the components with `f`, in axis order, starting from
            /// the first rather than from a supplied value.
            ///
            /// Private: it exists so the four methods above are each written
            /// once. [`fold`](Self::fold) is the public form.
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
            /// The dot product: the components multiplied pairwise and added.
            ///
            /// Zero when the two are at right angles, positive when they point
            /// the same way, and equal to the product of their lengths when
            /// they are parallel.
            pub fn dot(self, rhs: Self) -> T {
                sum_terms!($(self.$component * rhs.$component),+)
            }

            /// The squared length, the vector dotted with itself.
            ///
            /// Prefer this to [`norm`](Self::norm) when comparing distances or
            /// testing a radius: it needs no square root, and for integer
            /// components it stays exact where a length would not.
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

        /// Scales every component by one value.
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

        /// Divides every component by one value.
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
        //
        // Shifting every component at once is how a position moves between
        // octree levels: one level coarser is one bit right, one finer is one
        // bit left.

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
    /// The cross product: the vector at right angles to both inputs,
    /// right-handed, with length equal to the area of the parallelogram they
    /// span.
    ///
    /// Zero when the two are parallel, which is the usual way of testing for
    /// that. Defined only in three dimensions, so it sits outside the macro.
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
            /// Every component zero: the origin, and the additive identity.
            pub const ZERO: Self = Self { $($component: 0.0),+ };
            /// Every component one. Not a unit vector; see the axis constants
            /// for those.
            pub const ONE: Self = Self { $($component: 1.0),+ };

            /// The length, the square root of
            /// [`norm_squared`](Self::norm_squared).
            pub fn norm(self) -> $scalar {
                self.norm_squared().sqrt()
            }

            /// The same direction at length 1, the vector divided by its own
            /// length.
            ///
            /// Not finite for a zero vector, which has no direction to keep:
            /// check [`norm_squared`](Self::norm_squared) first where the input
            /// may be degenerate. The generators avoid the question by drawing
            /// directions that are already unit length; see
            /// [`UnitSphere`](crate::random::UnitSphere).
            pub fn normalize(self) -> Self {
                self / self.norm()
            }

            /// The distance between two points: the length of the difference.
            pub fn distance(self, other: Self) -> $scalar {
                (self - other).norm()
            }

            /// The squared distance between two points, which is what to
            /// compare against a squared radius.
            pub fn distance_squared(self, other: Self) -> $scalar {
                (self - other).norm_squared()
            }

            /// The point a fraction `t` of the way from `self` to `other`.
            ///
            /// Exact at both ends: `t` of 0 gives `self` and 1 gives `other`.
            /// Values outside `[0, 1]` extrapolate past the ends.
            pub fn lerp(self, other: Self, t: $scalar) -> Self {
                self + (other - self) * t
            }

            /// The absolute value of every component, which mirrors the vector
            /// into the all-positive corner.
            pub fn abs(self) -> Self {
                self.map(<$scalar>::abs)
            }

            /// The part of `self` that lies along `other`: the shadow `self`
            /// casts on that direction, as a vector.
            ///
            /// Not finite when `other` is zero, which names no direction.
            pub fn project_onto(self, other: Self) -> Self {
                other * (self.dot(other) / other.norm_squared())
            }

            /// What projection leaves behind: the part of `self` at right
            /// angles to `other`.
            ///
            /// Adding this to [`project_onto`](Self::project_onto) gives
            /// `self` back.
            pub fn reject_from(self, other: Self) -> Self {
                self - self.project_onto(other)
            }

            /// How far `self` reaches along `other`, as a signed length.
            ///
            /// Negative when `self` points against `other`, and equal to the
            /// length of [`project_onto`](Self::project_onto) otherwise. Not
            /// finite when `other` is zero.
            pub fn scalar_projection(self, other: Self) -> $scalar {
                self.dot(other) / other.norm()
            }

            /// Whether every component is finite: no infinity and no NaN.
            ///
            /// The check to make before trusting a vector that came out of a
            /// division, a normalisation or a file.
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
// Fixed-point components
// ---------------------------------------------------------------------------
//
// Addition, subtraction, scaling, negation, `dot`, `norm_squared` and the
// component-wise combinators already work for any component type that
// implements the operators, which `Fixed` does. What is written out here is the
// rest of what `implement_real_vector!` gives an `f64` vector, in the shape
// fixed point needs it: every operation that can overflow returns `None`
// instead of a value, and lengths come back as `Option` because a square root
// can leave the range.

macro_rules! implement_fixed_vector {
    ($name:ident, [$($component:ident),+]) => {
        impl $name<Fixed> {
            // No `ZERO` or `ONE` here on purpose. An associated constant cannot
            // be disambiguated by inference, so defining them would make the
            // plain `Vector3::ZERO` that f64 code writes ambiguous. Reach for
            // `splat(Fixed::ZERO)` instead.

            /// The absolute value of every component, which mirrors the vector
            /// into the all-positive corner.
            pub fn abs(self) -> Self {
                self.map(Fixed::abs)
            }

            /// The sum, or `None` if any component overflows. The operator
            /// panics instead; this is for callers that would rather ask.
            pub fn checked_add(self, other: Self) -> Option<Self> {
                Some(Self { $($component: self.$component.checked_add(other.$component)?),+ })
            }

            /// The difference, or `None` if any component overflows.
            pub fn checked_sub(self, other: Self) -> Option<Self> {
                Some(Self { $($component: self.$component.checked_sub(other.$component)?),+ })
            }

            /// Every component scaled by one value, or `None` on overflow.
            pub fn checked_scale(self, factor: Fixed) -> Option<Self> {
                Some(Self { $($component: self.$component.checked_mul(factor)?),+ })
            }

            /// The dot product, or `None` if any product or the running total
            /// overflows.
            pub fn checked_dot(self, other: Self) -> Option<Fixed> {
                let mut total: Fixed = Fixed::ZERO;

                $(
                    total = total.checked_add(self.$component.checked_mul(other.$component)?)?;
                )+

                Some(total)
            }

            /// The squared length, or `None` on overflow. What to compare
            /// against a squared radius, since it needs no square root.
            pub fn checked_norm_squared(self) -> Option<Fixed> {
                self.checked_dot(self)
            }

            /// The length, truncated to a step, or `None` on overflow.
            ///
            /// The square root is the largest value whose square does not
            /// exceed the squared length, so it is exact in the only sense a
            /// fixed-point root can be, and identical on every machine.
            pub fn norm(self) -> Option<Fixed> {
                self.checked_norm_squared()?.sqrt()
            }

            /// The squared distance between two points, or `None` on overflow.
            pub fn distance_squared(self, other: Self) -> Option<Fixed> {
                self.checked_sub(other)?.checked_norm_squared()
            }

            /// The distance between two points, or `None` on overflow.
            pub fn distance(self, other: Self) -> Option<Fixed> {
                self.checked_sub(other)?.norm()
            }

            /// The same direction at length one, or `None` for a zero vector,
            /// which names no direction, or on overflow.
            ///
            /// Each component is divided by the length, so the result is within
            /// a step of unit length rather than exactly on it.
            pub fn normalize(self) -> Option<Self> {
                let length: Fixed = self.norm()?;

                if length.is_zero() {
                    return None;
                }

                Some(Self { $($component: self.$component.checked_div(length)?),+ })
            }

            /// The point a fraction `t` of the way from this one to `other`, or
            /// `None` on overflow. Exact at both ends.
            /// Every component divided by one value, or `None` for a zero divisor or
            /// on overflow.
            ///
            /// # Why this is not `checked_scale` by a reciprocal
            ///
            /// Because a reciprocal rounds. `1/3` is not representable, so scaling by it
            /// loses accuracy that dividing directly keeps — dividing a vector of
            /// `6, -12, 30` by three gives exactly `2, -4, 10` here, where the
            /// reciprocal route gives `1.9999999995` and its neighbours.
            pub fn checked_divide(self, divisor: Fixed) -> Option<Self> {
                Some(Self { $($component: self.$component.checked_div(divisor)?),+ })
            }

            pub fn lerp(self, other: Self, t: Fixed) -> Option<Self> {
                Some(Self { $($component: self.$component.lerp(other.$component, t)?),+ })
            }

            /// The same vector in floating point, for rendering or for handing
            /// to something that works in `f64`.
            ///
            /// Far from the origin this is where the precision goes: subtract a
            /// nearby reference point first, in fixed point, and convert the
            /// offset instead.
            pub fn to_f64(self) -> $name<f64> {
                $name::new($(self.$component.to_f64()),+)
            }
        }
    };
}

implement_fixed_vector!(Vector2, [x, y]);
implement_fixed_vector!(Vector3, [x, y, z]);
implement_fixed_vector!(Vector4, [x, y, z, w]);

// Casting a fixed-point vector, component by component, the same way the casts
// at the end of this file work for the other component types. The difference is
// that `Fixed` has a range and a step of its own, so a conversion that can leave
// the range is a `TryFrom` rather than a silently saturating `as`.
//
// The component list travels between these macros as one opaque group, since a
// repetition over the integer widths cannot enclose one over the components.

macro_rules! implement_fixed_vector_casts {
    ($name:ident, $components:tt, [$($integer:ty),+]) => {
        implement_fixed_vector_real_casts!($name, $components);

        $(
            implement_fixed_vector_integer_cast!($name, $components, $integer);
        )+
    };
}

macro_rules! implement_fixed_vector_real_casts {
    ($name:ident, [$($component:ident),+]) => {
        /// The nearest floating-point vector, rounding each component. What a
        /// renderer or a sampler wants; far from the origin, subtract a
        /// reference point first and convert the offset.
        impl From<$name<Fixed>> for $name<f64> {
            fn from(value: $name<Fixed>) -> Self {
                Self { $($component: value.$component.to_f64()),+ }
            }
        }

        /// The nearest `f32` vector, which rounds much sooner than `f64`.
        impl From<$name<Fixed>> for $name<f32> {
            fn from(value: $name<Fixed>) -> Self {
                Self { $($component: value.$component.to_f32()),+ }
            }
        }

        /// The whole part of each component, truncated towards zero. Every
        /// value fits an `i128`, so this cannot fail.
        impl From<$name<Fixed>> for $name<i128> {
            fn from(value: $name<Fixed>) -> Self {
                Self { $($component: value.$component.to_integer()),+ }
            }
        }

        /// Each component truncated to its step, or an error if any of them is
        /// not finite or is outside the fixed-point range.
        impl TryFrom<$name<f64>> for $name<Fixed> {
            type Error = &'static str;

            fn try_from(value: $name<f64>) -> Result<Self, Self::Error> {
                Ok(Self { $($component: Fixed::try_from(value.$component)?),+ })
            }
        }

        /// Each component widened to an `f64` and truncated to its step.
        impl TryFrom<$name<f32>> for $name<Fixed> {
            type Error = &'static str;

            fn try_from(value: $name<f32>) -> Result<Self, Self::Error> {
                Ok(Self { $($component: Fixed::try_from(value.$component)?),+ })
            }
        }

        /// Whole units, or an error if any component is too large to scale.
        impl TryFrom<$name<i128>> for $name<Fixed> {
            type Error = &'static str;

            fn try_from(value: $name<i128>) -> Result<Self, Self::Error> {
                Ok(Self { $($component: Fixed::try_from(value.$component)?),+ })
            }
        }
    };
}

macro_rules! implement_fixed_vector_integer_cast {
    ($name:ident, [$($component:ident),+], $integer:ty) => {
        #[doc = concat!("Whole units from a `", stringify!($integer), "` vector, every one of which fits.")]
        impl From<$name<$integer>> for $name<Fixed> {
            fn from(value: $name<$integer>) -> Self {
                Self { $($component: Fixed::from(value.$component)),+ }
            }
        }

        #[doc = concat!("The whole part of each component, truncated towards zero, or an error when one is outside `", stringify!($integer), "`.")]
        impl TryFrom<$name<Fixed>> for $name<$integer> {
            type Error = &'static str;

            fn try_from(value: $name<Fixed>) -> Result<Self, Self::Error> {
                Ok(Self { $($component: <$integer>::try_from(value.$component)?),+ })
            }
        }
    };
}

implement_fixed_vector_casts!(
    Vector2,
    [x, y],
    [i8, i16, i32, i64, u8, u16, u32, u64, usize, isize]
);
implement_fixed_vector_casts!(
    Vector3,
    [x, y, z],
    [i8, i16, i32, i64, u8, u16, u32, u64, usize, isize]
);
implement_fixed_vector_casts!(
    Vector4,
    [x, y, z, w],
    [i8, i16, i32, i64, u8, u16, u32, u64, usize, isize]
);

// ---------------------------------------------------------------------------
// Axis constants
// ---------------------------------------------------------------------------
//
// The unit vector along each axis: already normalized, and the basis every
// rotation and projection is written against.

impl Vector2<f64> {
    /// One along x, zero elsewhere.
    pub const X: Self = Self::new(1.0, 0.0);
    /// One along y, zero elsewhere.
    pub const Y: Self = Self::new(0.0, 1.0);
}

impl Vector3<f64> {
    /// One along x, zero elsewhere.
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    /// One along y, zero elsewhere.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    /// One along z, zero elsewhere.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);
}

impl Vector4<f64> {
    /// One along x, zero elsewhere.
    pub const X: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    /// One along y, zero elsewhere.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);
    /// One along z, zero elsewhere.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0, 0.0);
    /// One along w, zero elsewhere.
    pub const W: Self = Self::new(0.0, 0.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Casting between dimensions
// ---------------------------------------------------------------------------

/// Grows a vector, filling the new components with `T::default()`, which is
/// zero for every numeric type.
impl<T> From<Vector2<T>> for Vector3<T>
where
    T: Default,
{
    fn from(value: Vector2<T>) -> Self {
        Self::new(value.x, value.y, T::default())
    }
}

/// Grows a vector, filling the new components with `T::default()`.
impl<T> From<Vector2<T>> for Vector4<T>
where
    T: Default,
{
    fn from(value: Vector2<T>) -> Self {
        Self::new(value.x, value.y, T::default(), T::default())
    }
}

/// Grows a vector, filling the new component with `T::default()`.
impl<T> From<Vector3<T>> for Vector4<T>
where
    T: Default,
{
    fn from(value: Vector3<T>) -> Self {
        Self::new(value.x, value.y, value.z, T::default())
    }
}

/// Shrinks a vector by dropping the trailing component.
impl<T> From<Vector3<T>> for Vector2<T> {
    fn from(value: Vector3<T>) -> Self {
        Self::new(value.x, value.y)
    }
}

/// Shrinks a vector by dropping the trailing component.
impl<T> From<Vector4<T>> for Vector3<T> {
    fn from(value: Vector4<T>) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}

/// Shrinks a vector by dropping the two trailing components.
impl<T> From<Vector4<T>> for Vector2<T> {
    fn from(value: Vector4<T>) -> Self {
        Self::new(value.x, value.y)
    }
}

// ---------------------------------------------------------------------------
// Casting between component types
// ---------------------------------------------------------------------------
//
// These cannot be generic, because `as` is an operator rather than a trait, so
// each pair is spelled out below.

macro_rules! implement_component_cast {
    ($name:ident, [$($component:ident),+], $from:ty, $to:ty) => {
        /// Casts every component with `as`, so a float to an integer truncates
        /// towards zero and saturates at the integer's limits: -2.7 becomes -2,
        /// and 1e30 becomes `i32::MAX`.
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
