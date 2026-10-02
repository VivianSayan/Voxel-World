// Conversions
// ---------------------------------------------------------------------------

macro_rules! implement_from_integer {
    ($($integer:ty),+) => {
        $(
            #[doc = concat!("Every `", stringify!($integer), "` is a whole number of units, so this cannot fail.")]
            impl From<$integer> for Fixed {
                fn from(whole: $integer) -> Self {
                    Self(whole as i128 * SCALE)
                }
            }
        )+
    };
}

implement_from_integer!(i8, i16, i32, i64, u8, u16, u32);

/// A `u64` is a whole number of units, and still fits after scaling.
impl<const FRACTION_BITS: u32> From<u64> for FixedPoint<FRACTION_BITS> {
    fn from(whole: u64) -> Self {
        Self(whole as i128 * Self::SCALE)
    }
}

/// Whole units, or an error when they are too large to scale.
impl<const FRACTION_BITS: u32> TryFrom<i128> for FixedPoint<FRACTION_BITS> {
    type Error = &'static str;

    fn try_from(whole: i128) -> Result<Self, Self::Error> {
        Self::from_integer_i128(whole).ok_or("whole value is outside the fixed-point range")
    }
}

/// As [`Fixed::from_f64`], with a message instead of `None`.
impl<const FRACTION_BITS: u32> TryFrom<f64> for FixedPoint<FRACTION_BITS> {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_f64(value).ok_or("value is not finite, or outside the fixed-point range")
    }
}

/// The nearest `f64`; see [`Fixed::to_f64`].
impl<const FRACTION_BITS: u32> From<FixedPoint<FRACTION_BITS>> for f64 {
    fn from(value: FixedPoint<FRACTION_BITS>) -> Self {
        value.to_f64()
    }
}

/// The nearest `f32`; see [`Fixed::to_f32`].
impl<const FRACTION_BITS: u32> From<FixedPoint<FRACTION_BITS>> for f32 {
    fn from(value: FixedPoint<FRACTION_BITS>) -> Self {
        value.to_f32()
    }
}

/// As [`Fixed::from_f32`], with a message instead of `None`.
impl<const FRACTION_BITS: u32> TryFrom<f32> for FixedPoint<FRACTION_BITS> {
    type Error = &'static str;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::from_f32(value).ok_or("value is not finite, or outside the fixed-point range")
    }
}

/// Whole units from a `usize`, which is at most 64 bits on any target, so this
/// cannot fail.
impl<const FRACTION_BITS: u32> From<usize> for FixedPoint<FRACTION_BITS> {
    fn from(whole: usize) -> Self {
        Self(whole as i128 * Self::SCALE)
    }
}

/// Whole units from an `isize`, which is at most 64 bits on any target.
impl<const FRACTION_BITS: u32> From<isize> for FixedPoint<FRACTION_BITS> {
    fn from(whole: isize) -> Self {
        Self(whole as i128 * Self::SCALE)
    }
}

/// Whole units, or an error when they are too large to scale.
impl<const FRACTION_BITS: u32> TryFrom<u128> for FixedPoint<FRACTION_BITS> {
    type Error = &'static str;

    fn try_from(whole: u128) -> Result<Self, Self::Error> {
        i128::try_from(whole)
            .ok()
            .and_then(Self::from_integer_i128)
            .ok_or("whole value is outside the fixed-point range")
    }
}

/// The whole part, truncated towards zero; see [`Fixed::to_integer`]. Every
/// value fits, so this cannot fail.
impl<const FRACTION_BITS: u32> From<FixedPoint<FRACTION_BITS>> for i128 {
    fn from(value: FixedPoint<FRACTION_BITS>) -> Self {
        value.to_integer()
    }
}

/// The whole part of a [`Fixed`], truncated towards zero, for the integer
/// widths that cannot hold every one of them.
macro_rules! implement_to_integer {
    ($($integer:ty),+) => {
        $(
            #[doc = concat!("The whole part, truncated towards zero, or an error when it is outside `", stringify!($integer), "`.")]
            impl TryFrom<Fixed> for $integer {
                type Error = &'static str;

                fn try_from(value: Fixed) -> Result<Self, Self::Error> {
                    Self::try_from(value.to_integer())
                        .map_err(|_| concat!("value is outside ", stringify!($integer)))
                }
            }
        )+
    };
}

implement_to_integer!(i8, i16, i32, i64, u8, u16, u32, u64, u128, usize, isize);

// ---------------------------------------------------------------------------
