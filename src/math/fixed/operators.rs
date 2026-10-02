// Operators
// ---------------------------------------------------------------------------

/// Exact, and panics on overflow in every build profile.
impl<const FRACTION_BITS: u32> Add for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.checked_add(other)
            .expect("fixed-point overflow in add")
    }
}

/// Exact, and panics on overflow in every build profile.
impl<const FRACTION_BITS: u32> Sub for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.checked_sub(other)
            .expect("fixed-point overflow in sub")
    }
}

/// Panics on [`Fixed::MIN`], which has no positive counterpart.
impl<const FRACTION_BITS: u32> Neg for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn neg(self) -> Self {
        self.checked_neg().expect("fixed-point overflow in neg")
    }
}

/// Truncates towards zero, and panics on overflow in every build profile; see
/// [`Fixed::checked_mul`].
impl<const FRACTION_BITS: u32> Mul for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.checked_mul(other)
            .expect("fixed-point overflow in mul")
    }
}

/// Truncates towards zero, and panics on a zero divisor or on overflow; see
/// [`Fixed::checked_div`].
impl<const FRACTION_BITS: u32> Div for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert!(!other.is_zero(), "fixed-point division by zero");

        self.checked_div(other)
            .expect("fixed-point overflow in div")
    }
}

/// The remainder, carrying the sign of the left side. Panics on a zero divisor.
impl<const FRACTION_BITS: u32> Rem for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        assert!(!other.is_zero(), "fixed-point remainder by zero");

        self.checked_rem(other)
            .expect("fixed-point overflow in rem")
    }
}

/// Scales by a whole number, which needs no rescaling afterwards.
impl<const FRACTION_BITS: u32> Mul<i128> for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn mul(self, factor: i128) -> Self {
        Self(
            self.0
                .checked_mul(factor)
                .expect("fixed-point overflow in mul"),
        )
    }
}

/// Divides by a whole number, truncating towards zero.
impl<const FRACTION_BITS: u32> Div<i128> for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn div(self, divisor: i128) -> Self {
        assert!(divisor != 0, "fixed-point division by zero");

        Self(
            self.0
                .checked_div(divisor)
                .expect("fixed-point overflow in div"),
        )
    }
}

/// Doubles the value `places` times, which is exact until it overflows.
impl<const FRACTION_BITS: u32> Shl<u32> for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn shl(self, places: u32) -> Self {
        Self(
            self.0
                .checked_shl(places)
                .filter(|bits| bits >> places == self.0)
                .expect("fixed-point overflow in shl"),
        )
    }
}

/// Halves the value `places` times, rounding towards negative infinity.
impl<const FRACTION_BITS: u32> Shr<u32> for FixedPoint<FRACTION_BITS> {
    type Output = Self;

    fn shr(self, places: u32) -> Self {
        Self(
            self.0
                .checked_shr(places)
                .expect("fixed-point shift too far"),
        )
    }
}

/// Writes an assignment operator for every layout, in terms of the matching binary
/// operator — which is itself generic, so `x += y` works wherever `x + y` does.
///
/// These were written for the [`Fixed`] alias alone while the type was still fixed to
/// one width. Nothing about them depended on that width, so a `FixedPoint<16>` simply
/// had no `+=` for no reason.
macro_rules! implement_assign {
    ($trait:ident, $method:ident, $operator:tt, $rhs:ty) => {
        impl<const FRACTION_BITS: u32> $trait<$rhs> for FixedPoint<FRACTION_BITS> {
            fn $method(&mut self, other: $rhs) {
                *self = *self $operator other;
            }
        }
    };
}

implement_assign!(AddAssign, add_assign, +, Self);
implement_assign!(SubAssign, sub_assign, -, Self);
implement_assign!(MulAssign, mul_assign, *, Self);
implement_assign!(DivAssign, div_assign, /, Self);
implement_assign!(RemAssign, rem_assign, %, Self);
implement_assign!(MulAssign, mul_assign, *, i128);
implement_assign!(DivAssign, div_assign, /, i128);
implement_assign!(ShlAssign, shl_assign, <<, u32);
implement_assign!(ShrAssign, shr_assign, >>, u32);

/// Adds a sequence up, panicking on overflow as [`Add`] does.
impl<const FRACTION_BITS: u32> Sum for FixedPoint<FRACTION_BITS> {
    fn sum<I: Iterator<Item = Self>>(values: I) -> Self {
        values.fold(Self::ZERO, |total, value| total + value)
    }
}

/// Multiplies a sequence together, panicking on overflow as [`Mul`] does.
impl<const FRACTION_BITS: u32> Product for FixedPoint<FRACTION_BITS> {
    fn product<I: Iterator<Item = Self>>(values: I) -> Self {
        values.fold(Self::ONE, |total, value| total * value)
    }
}

// ---------------------------------------------------------------------------
