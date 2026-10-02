// 256-bit helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Trigonometry
//
// All of it runs on `cordic`, which is integer shifts and adds at 96 fractional
// bits. Nothing here touches a float, so every result is bit-identical on every
// platform — which is the whole reason this type exists.
// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// Narrows a value from the internal 96-bit working format, rounding once.
    ///
    /// Rounding here and nowhere else is what keeps the error analysis simple: the
    /// loop's own error is bounded far below the last bit of this layout, so this
    /// single rounding is the only one that reaches the result.
    fn from_internal(value: i128) -> Self {
        Self(round_shift(value, cordic::BITS - FRACTION_BITS))
    }

    /// The sine and cosine of an angle in radians, together.
    ///
    /// # Question
    ///
    /// "Where on the unit circle is this angle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// let (sine, cosine) = Fixed::FRAC_PI_2.sin_cos();
    ///
    /// assert!((sine.to_f64() - 1.0).abs() < 1e-9);
    /// assert!(cosine.to_f64().abs() < 1e-9);
    /// ```
    ///
    /// # Why both at once
    ///
    /// CORDIC produces the whole vector, so the cosine is already computed when the
    /// sine is. Asking for them separately does the identical work twice, and
    /// anything rotating a point needs both. This is the method to reach for; [`sin`]
    /// and [`cos`] discard half of it.
    ///
    /// [`sin`]: Self::sin
    /// [`cos`]: Self::cos
    ///
    /// # Accuracy, and the angle beyond which it degrades
    ///
    /// Measured within **half a step** for angles up to `10^5` radians, at 8, 16, 32
    /// and 64 fractional bits.
    ///
    /// The limit is the argument reduction, which folds the angle into `[0, τ)`
    /// against a 96-bit `1/τ`. Its error grows as `|angle| × 2⁻⁹³`, so the useful
    /// range depends on the width:
    ///
    /// | fractional bits | full accuracy up to | meaningless beyond |
    /// |---|---|---|
    /// | 32 | about `2⁶¹` radians | about `2⁹³` |
    /// | 64 | about `2²⁹` radians | about `2⁹³` |
    ///
    /// Past the right-hand column the reduction has lost every bit of the fraction
    /// and the answer is noise — but it is *deterministic* noise, and it does not
    /// overflow: the reduction runs in 256 bits and is defined for every `i128`.
    /// An angle that large has already lost the precision in its own representation,
    /// so nothing here is recoverable by working harder.
    pub fn sin_cos(self) -> (Self, Self) {
        let (cosine, sine): (i128, i128) = cordic::cosine_sine(self.0, FRACTION_BITS);

        (Self::from_internal(sine), Self::from_internal(cosine))
    }

    /// The sine of an angle in radians.
    ///
    /// # Question
    ///
    /// "How far above the axis is this angle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.sin(), Fixed::ZERO);
    /// assert!(Fixed::PI.sin().to_f64().abs() < 1e-9);
    /// ```
    ///
    /// Always in `[-1, 1]`, so it cannot fail and returns no [`Option`]. Prefer
    /// [`sin_cos`](Self::sin_cos) where both are wanted: it costs the same as this.
    pub fn sin(self) -> Self {
        self.sin_cos().0
    }

    /// The cosine of an angle in radians.
    ///
    /// # Question
    ///
    /// "How far along the axis is this angle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.cos(), Fixed::ONE);
    /// assert!((Fixed::PI.cos() + Fixed::ONE).to_f64().abs() < 1e-9);
    /// ```
    ///
    /// Always in `[-1, 1]`, so it cannot fail. Prefer [`sin_cos`](Self::sin_cos)
    /// where both are wanted.
    pub fn cos(self) -> Self {
        self.sin_cos().1
    }

    /// The tangent of an angle in radians, or [`None`] where it does not fit.
    ///
    /// # Question
    ///
    /// "What is the slope of this angle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// let eighth = Fixed::FRAC_PI_4.tan().unwrap();
    ///
    /// assert!((eighth.to_f64() - 1.0).abs() < 1e-9);
    /// ```
    ///
    /// # Why this one returns an [`Option`] when [`sin`](Self::sin) does not
    ///
    /// A tangent is unbounded. Near a quarter turn it exceeds anything this type can
    /// hold, and exactly at one it is undefined. Both come back as [`None`] rather
    /// than a saturated value, because a slope of "the largest number available" is
    /// not an answer and would quietly poison whatever used it.
    ///
    /// Reaching the exact quarter turn takes an angle whose cosine rounds to zero at
    /// 96 fractional bits, which an angle stored at 64 or fewer effectively cannot
    /// miss — so the undefined case is reported, not approximated.
    pub fn tan(self) -> Option<Self> {
        let (cosine, sine): (i128, i128) = cordic::cosine_sine(self.0, FRACTION_BITS);

        if cosine == 0 {
            return None;
        }

        let negative: bool = (sine < 0) != (cosine < 0);

        let quotient: u128 = wide::divide(
            wide::shift_left(sine.unsigned_abs(), FRACTION_BITS),
            cosine.unsigned_abs(),
        )?;

        wide::apply_sign(quotient, negative).map(Self)
    }

    /// The angle of the point `(x, self)`, over the whole circle `(-π, π]`.
    ///
    /// # Question
    ///
    /// "Which way does this vector point?"
    ///
    /// The receiver is the **y** component and the argument the **x**, matching
    /// `atan2(y, x)` everywhere else in mathematics and in the standard library.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// let north_east = Fixed::ONE.atan2(Fixed::ONE);
    ///
    /// assert!((north_east.to_f64() - Fixed::FRAC_PI_4.to_f64()).abs() < 1e-9);
    ///
    /// // Behind, which `atan` alone could never tell from ahead.
    /// let behind = Fixed::ZERO.atan2(Fixed::NEGATIVE_ONE);
    ///
    /// assert!((behind.to_f64() - Fixed::PI.to_f64()).abs() < 1e-9);
    /// ```
    ///
    /// # Why this and not [`atan`](Self::atan)
    ///
    /// `atan(y/x)` loses the quadrant, because `y/x` is the same for a direction and
    /// its opposite — and it divides, which fails when `x` is zero and loses
    /// precision when `x` is small. This takes both components, so it keeps the
    /// quadrant, handles every axis including `(0, 0)`, and never divides. For a
    /// direction in a world, this is almost always the one that was wanted.
    ///
    /// `(0, 0)` has no angle; zero is returned, as the standard library does.
    ///
    /// # Accuracy
    ///
    /// The pair is rescaled to use the full width before the loop, so a direction
    /// built from two small components is as accurate as one built from two large
    /// ones. No argument reduction is involved, so there is no large-input
    /// degradation here.
    pub fn atan2(self, x: Self) -> Self {
        Self::from_internal(cordic::arctangent2(self.0, x.0))
    }

    /// The angle whose tangent is this, in `(-π/2, π/2)`.
    ///
    /// # Question
    ///
    /// "What angle has this slope?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert!((Fixed::ONE.atan().to_f64() - Fixed::FRAC_PI_4.to_f64()).abs() < 1e-9);
    /// assert_eq!(Fixed::ZERO.atan(), Fixed::ZERO);
    /// ```
    ///
    /// Bounded by a quarter turn, so it cannot fail. Use
    /// [`atan2`](Self::atan2) when the two components are available separately: it
    /// keeps the quadrant that a slope alone has already lost.
    pub fn atan(self) -> Self {
        self.atan2(Self::ONE)
    }

    /// The angle whose sine is this, in `[-π/2, π/2]`, or [`None`] outside `[-1, 1]`.
    ///
    /// # Question
    ///
    /// "What angle has this height on the unit circle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ONE.asin(), Some(Fixed::FRAC_PI_2));
    /// assert_eq!(Fixed::from_integer(2).asin(), None);
    /// ```
    ///
    /// # How it is built
    ///
    /// `asin(v) = atan2(v, √(1 − v²))`, which is exact as a relationship and needs no
    /// approximation of its own. Going through [`atan2`](Self::atan2) rather than
    /// dividing keeps the `v = ±1` case working, where the root is zero and a
    /// division would not be.
    ///
    /// # Accuracy
    ///
    /// Worst near `±1`, where the square root's own error is magnified by the steep
    /// slope of `asin`. In the middle of the range it is accurate to the last step or
    /// so. Where that matters, prefer [`atan2`](Self::atan2) on the two components
    /// directly and never form the sine at all.
    pub fn asin(self) -> Option<Self> {
        Some(self.atan2(self.cosine_of_arc()?))
    }

    /// The angle whose cosine is this, in `[0, π]`, or [`None`] outside `[-1, 1]`.
    ///
    /// # Question
    ///
    /// "What angle has this horizontal extent on the unit circle?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ONE.acos(), Some(Fixed::ZERO));
    /// assert_eq!(Fixed::from_integer(-2).acos(), None);
    /// ```
    ///
    /// `acos(v) = atan2(√(1 − v²), v)`, the same construction as
    /// [`asin`](Self::asin) with the arguments the other way round, and with the same
    /// loss of accuracy near `±1`. This is the usual way to turn a dot product of two
    /// unit vectors into the angle between them.
    pub fn acos(self) -> Option<Self> {
        Some(self.cosine_of_arc()?.atan2(self))
    }

    /// `sqrt(1 - self^2)`, for [`asin`](Self::asin) and [`acos`](Self::acos).
    ///
    /// [`None`] exactly when `self` is outside `[-1, 1]`, and never because of an
    /// intermediate.
    ///
    /// # Why the domain is tested before the multiplication rather than after
    ///
    /// Leaving it to `sqrt` to notice a negative would conflate two different
    /// failures: a value genuinely outside the domain, and a value inside it whose
    /// square happened to round past one. Testing the raw magnitude first means only
    /// the first can happen, so a representable input in `[-1, 1]` always produces an
    /// answer.
    ///
    /// Within the domain the squaring truncates towards zero, so `self^2` never
    /// exceeds the true square and `1 - self^2` cannot go negative. The clamp below
    /// is therefore unreachable arithmetic rather than a correction — it is there so
    /// that a future change to the multiplication's rounding cannot turn a valid
    /// input into a `None` without anyone noticing.
    fn cosine_of_arc(self) -> Option<Self> {
        if self.0.unsigned_abs() > Self::ONE.0 as u128 {
            return None;
        }

        let square: Self = self.checked_mul(self)?;
        let remainder: i128 = Self::ONE.0 - square.0;

        Self(remainder.max(0)).sqrt()
    }

    /// The hyperbolic sine, or [`None`] where it does not fit.
    ///
    /// # Question
    ///
    /// "What is `(eˣ − e⁻ˣ)/2` here?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.sinh(), Some(Fixed::ZERO));
    /// ```
    ///
    /// Built from [`exp`](Self::exp), which is itself integer-only, so this inherits
    /// that accuracy and determinism. It grows as fast as `eˣ` does, so it returns
    /// [`None`] once the result leaves the layout's range.
    pub fn sinh(self) -> Option<Self> {
        // `sinh(-x) = -sinh(x)`, so the work is done on the positive side and the
        // sign put back afterwards. Negating the raw value directly would overflow
        // on MIN, whose positive counterpart is not representable — and MIN is far
        // past where a hyperbolic sine fits anyway, so `checked_abs` reporting
        // `None` there is the right answer rather than a dodge.
        let magnitude: Self = Self(self.0.checked_abs()?);

        let up: Self = magnitude.exp()?;
        let down: Self = Self(-magnitude.0).exp()?;

        let half: i128 = up.0.checked_sub(down.0)? / 2;

        Some(Self(if self.0 < 0 { -half } else { half }))
    }

    /// The hyperbolic cosine, or [`None`] where it does not fit.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.cosh(), Some(Fixed::ONE));
    /// ```
    ///
    /// `(eˣ + e⁻ˣ)/2`, built from [`exp`](Self::exp). Never below one, and growing as
    /// fast as `eˣ`.
    pub fn cosh(self) -> Option<Self> {
        // `cosh(-x) = cosh(x)`, so only the magnitude matters. As in `sinh`, MIN has
        // no positive counterpart and is far past where the result fits.
        let magnitude: Self = Self(self.0.checked_abs()?);

        let up: Self = magnitude.exp()?;
        let down: Self = Self(-magnitude.0).exp()?;

        Some(Self(up.0.checked_add(down.0)? / 2))
    }

    /// The hyperbolic tangent, always in `(-1, 1)`.
    ///
    /// # Question
    ///
    /// "What is this value, squashed into `(-1, 1)`?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::ZERO.tanh(), Fixed::ZERO);
    /// assert_eq!(Fixed::from_integer(100).tanh(), Fixed::ONE);
    /// ```
    ///
    /// # Why this cannot fail where [`sinh`](Self::sinh) can
    ///
    /// It is computed as `(e²ˣ − 1)/(e²ˣ + 1)` rather than as a ratio of the two
    /// above, so the intermediate that would overflow never appears on its own. Past
    /// the point where the result is within one step of `±1` — about 23, since
    /// `1 − tanh(x) ≈ 2e⁻²ˣ` — that bound is returned directly, which is the
    /// correctly rounded answer rather than a saturation.
    pub fn tanh(self) -> Self {
        // Beyond this the true value is nearer to ±1 than to any other representable
        // value, so returning the bound is exact, not a clamp.
        const SATURATION: i64 = 23;

        // `tanh(-x) = -tanh(x)`, so the work happens on the magnitude and the sign
        // goes back on at the end. Computing the two signs independently would give
        // a function that is *nearly* odd: each side rounds its own quotient, and the
        // two quotients do not agree in the last step. MIN has no magnitude, and is
        // far past saturation in any case.
        let negative: bool = self.0 < 0;
        let magnitude: i128 = self.0.checked_abs().unwrap_or(i128::MAX);

        let positive: Self = Self::tanh_of_magnitude(magnitude, SATURATION);

        // The result is in [0, 1], so negating it cannot leave the range.
        Self(if negative { -positive.0 } else { positive.0 })
    }

    /// `tanh` of a value already known to be at or above zero.
    fn tanh_of_magnitude(magnitude: i128, saturation: i64) -> Self {
        if magnitude >= Self::from_integer(saturation).0 {
            return Self::ONE;
        }

        // The guard above leaves `magnitude < 23 * 2^FRACTION_BITS`, so the doubling
        // is under `46 * 2^64` even at the widest layout — about `2^69`, far inside
        // an `i128`. The multiplication is safe *because* of the saturation, not
        // independently of it.
        let Some(doubled) = Self(magnitude * 2).exp() else {
            return Self::ONE;
        };

        let numerator: i128 = doubled.0 - Self::ONE.0;
        let denominator: i128 = doubled.0 + Self::ONE.0;

        let Some(quotient) = wide::divide(
            wide::shift_left(numerator.unsigned_abs(), FRACTION_BITS),
            denominator.unsigned_abs(),
        ) else {
            return Self::ONE;
        };

        Self(quotient.min(Self::ONE.0 as u128) as i128)
    }
}

/// The 256-bit arithmetic a 128-bit fixed-point type needs in the middle of a
/// multiplication, a division or a square root.
///
/// Nothing here is a general-purpose wide integer: each function does exactly
/// what one [`Fixed`] operation needs, on magnitudes, with the sign handled by
/// the caller.
///
/// # Why this is not [`WideUint<4>`](crate::math::WideUint)
///
/// The crate does have a general 256-bit integer, and it would remove every line
/// below. It is far too slow to put here. Measured over 200,000 pairs:
///
/// | operation | this module | `WideUint<4>` |
/// |---|---|---|
/// | 256-bit multiply | 144 µs | 1.18 ms (8.2×) |
/// | multiply then divide | 1.19 ms | 147 ms (124×) |
///
/// The gap is structural, not an accident of tuning. This module divides a
/// 256-bit value by walking 128 bits with native `u128` compares; `WideUint<4>`
/// walks 256 bits and does a four-limb compare and subtraction at each one. Since
/// [`Fixed`] exists to be the arithmetic used *instead of* `f64` on hot paths,
/// paying 124× for tidiness is the wrong trade.
///
/// What the general type is used for instead is checking this one: the tests
/// in `testing/tests/fixed_wide.rs` run both over the same inputs and require
/// them to agree, so the fast code has an independent oracle rather than only its
/// own expectations. That is why the items here are `pub` and hidden — an
/// integration test cannot reach a crate-private module.
#[doc(hidden)]
pub mod wide {
    use crate::math::WideUint;

    /// A 256-bit unsigned value, as two halves.
    ///
    /// Ordered by the high half first, which is the value's own order.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct U256 {
        pub high: u128,
        pub low: u128,
    }

    /// The exact 256-bit product of two 128-bit values, by the schoolbook
    /// method on 64-bit halves, carrying between the partial products.
    pub fn multiply(left: u128, right: u128) -> U256 {
        const HALF: u32 = 64;
        let mask: u128 = u64::MAX as u128;

        let (left_high, left_low): (u128, u128) = (left >> HALF, left & mask);
        let (right_high, right_low): (u128, u128) = (right >> HALF, right & mask);

        let low_low: u128 = left_low * right_low;
        let cross_one: u128 = left_high * right_low;
        let cross_two: u128 = left_low * right_high;
        let high_high: u128 = left_high * right_high;

        let (cross, crossed): (u128, bool) = cross_one.overflowing_add(cross_two);
        let (low, carried): (u128, bool) = low_low.overflowing_add(cross << HALF);

        let mut high: u128 = high_high + (cross >> HALF) + u128::from(carried);

        if crossed {
            high += 1 << HALF;
        }

        U256 { high, low }
    }

    /// A 128-bit value widened and scaled up by `places` bits.
    pub fn shift_left(value: u128, places: u32) -> U256 {
        if places == 0 {
            return U256 {
                high: 0,
                low: value,
            };
        }

        U256 {
            high: value >> (u128::BITS - places),
            low: value << places,
        }
    }

    /// The value scaled down by `places` bits, or `None` if what remains does
    /// not fit in 128 bits.
    pub fn shift_right(value: U256, places: u32) -> Option<u128> {
        (value.high >> places == 0).then(|| shift_right_wrapping(value, places))
    }

    /// The low 128 bits of the value scaled down by `places` bits.
    pub fn shift_right_wrapping(value: U256, places: u32) -> u128 {
        if places == 0 {
            return value.low;
        }

        (value.low >> places) | (value.high << (u128::BITS - places))
    }

    /// The quotient of a 256-bit value by a 128-bit one, or `None` for a zero
    /// divisor or a quotient too large to hold.
    ///
    /// # Two paths, for a measured reason
    ///
    /// A numerator that fits in 128 bits is divided directly, which is one
    /// instruction or one compiler-runtime call. That covers every ordinary
    /// [`Fixed`](super::Fixed): the numerator here is a magnitude shifted up by
    /// [`FRACTION_BITS`](super::FRACTION_BITS), so `high` stays zero for anything
    /// below `2⁹⁶`. Measured at 17 ns, against 23 ns to go through
    /// [`WideUint<4>`](crate::math::WideUint) — so this path earns its keep and stays.
    ///
    /// Anything wider hands the work to `WideUint<4>`, whose division is Knuth's
    /// Algorithm D. That replaced a bit-at-a-time loop which took **377 ns** where
    /// this takes **40 ns** — nine times faster, and one fewer implementation of
    /// long division for the crate to be right about.
    ///
    /// The guard above establishes `high < divisor`, so the quotient is below
    /// `2¹²⁸` and always fits.
    pub fn divide(numerator: U256, divisor: u128) -> Option<u128> {
        if divisor == 0 || numerator.high >= divisor {
            return None;
        }

        if numerator.high == 0 {
            return Some(numerator.low / divisor);
        }

        let wide: WideUint<4> = WideUint::from_limbs([
            numerator.low as u64,
            (numerator.low >> 64) as u64,
            numerator.high as u64,
            (numerator.high >> 64) as u64,
        ]);

        let (quotient, _) = wide.div_rem(&WideUint::from(divisor))?;

        quotient.to_u128()
    }

    /// The largest value whose square does not exceed this one.
    ///
    /// A value that fits in 128 bits goes straight to the standard library's
    /// integer square root, which covers every [`Fixed`](super::Fixed) below
    /// `2^64`. A wider one is found one bit at a time from the top, testing
    /// each candidate by squaring it back out to 256 bits.
    pub fn square_root(value: U256) -> u128 {
        if value.high == 0 {
            return value.low.isqrt();
        }

        // The root of the high half, shifted back up, is never above the true
        // root and never more than `2^64` below it, so only the low half of the
        // answer is left to find bit by bit.
        let mut root: u128 = value.high.isqrt() << (u128::BITS / 2);

        for place in (0..u128::BITS / 2).rev() {
            let candidate: u128 = root | (1 << place);

            if multiply(candidate, candidate) <= value {
                root = candidate;
            }
        }

        root
    }

    /// A magnitude given its sign, or `None` when the negative range cannot
    /// hold it.
    pub fn apply_sign(magnitude: u128, negative: bool) -> Option<i128> {
        if negative {
            (magnitude <= i128::MAX as u128 + 1).then(|| (magnitude as i128).wrapping_neg())
        } else {
            (magnitude <= i128::MAX as u128).then_some(magnitude as i128)
        }
    }
}
