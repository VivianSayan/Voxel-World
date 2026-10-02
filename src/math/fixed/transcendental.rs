// ---------------------------------------------------------------------------

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// The base-two logarithm, or `None` for a value at or below zero.
    ///
    /// The whole part comes free from the position of the highest set bit. The
    /// fraction is then found one bit at a time: normalise what is left into
    /// `[1, 2)` and square it, and if the square reaches two the bit is set and
    /// the square is halved. Each output bit costs one squaring.
    ///
    /// Accurate to half a step: the squarings are carried out with 96
    /// fractional bits, so their rounding stays far below the answer and only
    /// the final rounding to a [`Fixed`] shows.
    pub fn log2(self) -> Option<Self> {
        Some(Self(round_shift(
            self.log2_wide()?,
            Self::LOG_BITS - FRACTION_BITS,
        )))
    }

    /// The natural logarithm, or `None` for a value at or below zero.
    ///
    /// [`Fixed::log2`] scaled by `ln 2`, with the scaling done in the internal
    /// working format rather than in steps, so nothing is lost between the two.
    /// Accurate to one step.
    pub fn ln(self) -> Option<Self> {
        self.scaled_log(WORKING_LN_2)
    }

    /// The base-ten logarithm, or `None` for a value at or below zero.
    ///
    /// As [`Fixed::ln`], scaled by the base-ten logarithm of two. Accurate to
    /// one step.
    pub fn log10(self) -> Option<Self> {
        self.scaled_log(WORKING_LOG10_2)
    }

    /// The logarithm in any base, as `log2(self) / log2(base)`, or `None`
    /// unless both are above zero and the base is not one.
    pub fn log(self, base: Self) -> Option<Self> {
        let value: i128 = self.log2_wide()?;
        let divisor: i128 = base.log2_wide()?;

        if divisor == 0 {
            return None;
        }

        let sign: bool = (value < 0) != (divisor < 0);
        let scaled: wide::U256 = wide::shift_left(value.unsigned_abs(), FRACTION_BITS);
        let magnitude: u128 = wide::divide(scaled, divisor.unsigned_abs())?;

        wide::apply_sign(magnitude, sign).map(Self)
    }

    /// Two raised to this value, or `None` where the result no longer fits.
    ///
    /// The whole part of the exponent becomes a shift and only the fraction
    /// needs work, which is what keeps the series short.
    ///
    /// # Range
    ///
    /// The limits follow from the layout rather than being chosen: the largest
    /// value is just under `2^(127 - FRACTION_BITS)`, so anything at or above that
    /// exponent gives `None`, and anything below `-(FRACTION_BITS + 1)` is nearer
    /// zero than to the smallest positive step and gives [`FixedPoint::ZERO`]. For
    /// the default layout that is `None` above 95 and zero below -33.
    ///
    /// Zero is the rounded answer rather than a failure; `None` means the result
    /// genuinely does not fit.
    ///
    /// # Accuracy
    ///
    /// Measured against an independent high-precision evaluation at 8, 16, 32 and 64
    /// fractional bits: within **half a step** wherever the result is at most one.
    /// Above that the error grows in proportion to the result, because the step is a
    /// fixed absolute size while the algorithm's error is relative.
    ///
    /// Not "correctly rounded", which this once claimed: that would mean the nearest
    /// representable value for *every* input, and it does not hold for large results.
    pub fn exp2(self) -> Option<Self> {
        // The domain is settled first, because the rescaling below is a left shift
        // that would wrap silently on a large raw value — in release *and* in debug,
        // since a shift only traps on an out-of-range shift count, never on the bits
        // it pushes off the top.
        let whole: i128 = self.0 >> FRACTION_BITS;

        if whole >= Self::EXP2_TOO_LARGE {
            return None;
        }

        if whole < Self::EXP2_UNDERFLOWS {
            return Some(Self::ZERO);
        }

        // Past those guards the exponent is a small number of whole units, so the
        // shift has room: `|self.0|` is under `2^(FRACTION_BITS + 8)` and the shift
        // adds only LOG_GUARD_BITS more.
        exp2_wide::<FRACTION_BITS>(self.0 << (Self::LOG_BITS - FRACTION_BITS))
    }

    /// `e` raised to this value, or `None` where the result no longer fits.
    ///
    /// The exponent is split as `k * ln 2 + r` with `r` in `[0, ln 2)`, so the
    /// whole of it becomes a shift and the series only ever sees a small
    /// remainder, where it converges in a dozen terms.
    ///
    /// # Range
    ///
    /// The same layout limits as [`FixedPoint::exp2`], carried across by `ln 2`:
    /// `None` above about `(127 - FRACTION_BITS) * ln 2`, and zero below about
    /// `-(FRACTION_BITS + 1) * ln 2`. For the default layout that is `None` above
    /// roughly 65.85 and zero below roughly -22.9.
    ///
    /// Zero really is the rounded answer: `exp(-30)` is nearer zero than to any
    /// other value this type can hold, which is why a distribution's tail cannot be
    /// computed here.
    ///
    /// # Accuracy
    ///
    /// Within **half a step** wherever the result is at most one, measured as for
    /// [`FixedPoint::exp2`]. Beyond that the error grows with the result and is a
    /// little worse than `exp2`'s, because the argument reduction divides by `ln 2`
    /// and so carries that constant's own representation error once per halving:
    /// at 64 fractional bits a result near `7e10` was 100 steps out, a relative
    /// error below `2^-93`.
    pub fn exp(self) -> Option<Self> {
        // As in `exp2`, the domain has to be settled before the rescaling, which is
        // a left shift that would wrap quietly on a large raw value. The previous
        // guard admitted raw values up to `2^100`, which a shift of 64 turned into
        // `2^164` and wrapped.
        let whole: i128 = self.0 >> FRACTION_BITS;

        if whole > Self::EXP_TOO_LARGE {
            return None;
        }

        if whole < Self::EXP_UNDERFLOWS {
            return Some(Self::ZERO);
        }

        let exponent: i128 = self.0 << (WORKING_BITS - FRACTION_BITS);
        let halvings: i128 = exponent.div_euclid(WORKING_LN_2 as i128);
        let remainder: u128 = exponent.rem_euclid(WORKING_LN_2 as i128) as u128;

        place(exponential_series(remainder), halvings)
    }

    /// This value raised to another, as `exp2(log2(self) * exponent)`, or
    /// `None` unless the value is above zero and the result fits.
    ///
    /// Both halves round, so this is the least accurate function here: about
    /// one part in `10^10`, and worse for a large exponent, which multiplies
    /// the logarithm's error along with the logarithm. The exponent is carried
    /// at the logarithm's own precision rather than rounded to a step first,
    /// which is worth a factor of a thousand. [`Fixed::powi`] is closer still
    /// and should be preferred for a whole exponent.
    pub fn pow(self, exponent: Self) -> Option<Self> {
        let logarithm: i128 = self.log2_wide()?;
        let sign: bool = (logarithm < 0) != (exponent.0 < 0);
        let product: wide::U256 =
            wide::multiply(logarithm.unsigned_abs(), exponent.0.unsigned_abs());

        // The product carries LOG_BITS + FRACTION_BITS fractional bits, and the
        // exponential takes LOG_BITS, so the scaled logarithm keeps all the
        // precision it was found with rather than being rounded to a step
        // first.
        let magnitude: u128 = wide::shift_right(product, FRACTION_BITS)?;

        exp2_wide(wide::apply_sign(magnitude, sign)?)
    }

    /// This value raised to a whole power, by repeated squaring, or `None` on
    /// overflow.
    ///
    /// Every multiplication truncates, so the error grows with the number of
    /// squarings, but it never involves a logarithm: about one part in `10^10`
    /// across the range, and exact for small whole bases and exponents.
    ///
    /// A negative exponent inverts either the base or the result, whichever
    /// keeps the intermediate value large: raising a value below one to a
    /// negative power inverts first, since the truncation in a tiny
    /// intermediate would otherwise be magnified by the inversion.
    pub fn powi(self, exponent: i32) -> Option<Self> {
        // Anything to the zero is one, including zero and MIN. Settling it first
        // means the rest never has to reason about a base it will not look at.
        if exponent == 0 {
            return Some(Self::ONE);
        }

        // A magnitude comparison on the raw integer, because `abs` panics on MIN,
        // whose positive counterpart does not exist — and an `Option`-returning
        // power should report that it cannot answer, not abort.
        let below_one: bool = self.0.unsigned_abs() < Self::ONE.0 as u128;
        let invert_first: bool = exponent < 0 && below_one;
        let mut remaining: u32 = exponent.unsigned_abs();
        let mut base: Self = if invert_first {
            self.checked_recip()?
        } else {
            self
        };
        let mut total: Self = Self::ONE;

        while remaining != 0 {
            if remaining & 1 == 1 {
                total = total.checked_mul(base)?;
            }

            remaining >>= 1;

            if remaining != 0 {
                base = base.checked_mul(base)?;
            }
        }

        if exponent < 0 && !invert_first {
            return total.checked_recip();
        }

        Some(total)
    }

    /// The base-two logarithm with [`LOG_BITS`] fractional bits, which is what
    /// every logarithm here is built from.
    fn log2_wide(self) -> Option<i128> {
        if self.0 <= 0 {
            return None;
        }

        let magnitude: u128 = self.0 as u128;
        let highest: u32 = u128::BITS - 1 - magnitude.leading_zeros();

        // What is left once the highest bit is taken out, in `[1, 2)`.
        let mut mantissa: u128 = if highest <= WORKING_BITS {
            magnitude << (WORKING_BITS - highest)
        } else {
            magnitude >> (highest - WORKING_BITS)
        };

        let whole: i128 = highest as i128 - FRACTION_BITS as i128;
        let mut fraction: i128 = 0;

        for place in (0..Self::LOG_BITS).rev() {
            mantissa = wide::shift_right_wrapping(wide::multiply(mantissa, mantissa), WORKING_BITS);

            if mantissa >= WORKING_ONE << 1 {
                mantissa >>= 1;
                fraction |= 1 << place;
            }
        }

        Some((whole << Self::LOG_BITS) + fraction)
    }

    /// A logarithm in the base whose `ln` is `scale`, given in the internal
    /// working format.
    fn scaled_log(self, scale: u128) -> Option<Self> {
        let logarithm: i128 = self.log2_wide()?;
        let product: wide::U256 = wide::multiply(logarithm.unsigned_abs(), scale);

        // The product carries LOG_BITS + WORKING_BITS fractional bits.
        let magnitude: u128 =
            wide::shift_right(product, WORKING_BITS + Self::LOG_BITS - FRACTION_BITS)?;

        wide::apply_sign(magnitude, logarithm < 0).map(Self)
    }
}

/// Two raised to a value carrying [`LOG_BITS`] fractional bits.
///
/// Taking the exponent at the logarithm's own precision rather than at a step
/// is what keeps [`Fixed::pow`] accurate: a step of error in the exponent is a
/// relative error of `ln 2 * 2^-32` in the result, which at a large result is
/// many steps.
fn exp2_wide<const FRACTION_BITS: u32>(exponent: i128) -> Option<FixedPoint<FRACTION_BITS>> {
    let log_bits: u32 = FRACTION_BITS + LOG_GUARD_BITS;
    let whole: i128 = exponent >> log_bits;
    let fraction: u128 = (exponent & ((1 << log_bits) - 1)) as u128;

    // The fraction as an exponent of e, which is what the series takes.
    let scaled: u128 = wide::shift_right_wrapping(
        wide::multiply(fraction << (WORKING_BITS - log_bits), WORKING_LN_2),
        WORKING_BITS,
    );

    place(exponential_series(scaled), whole)
}

/// `e` raised to a value in `[0, ln 2)`, given and returned in the internal
/// working format.
///
/// The Taylor series, whose terms are each the one before times `x / n`. The
/// argument is below 0.694, so the terms fall away by more than a factor of two
/// each time and the sum settles in about a dozen of them; it runs until a term
/// rounds to nothing.
fn exponential_series(exponent: u128) -> u128 {
    let mut term: u128 = WORKING_ONE;
    let mut total: u128 = WORKING_ONE;
    let mut step: u128 = 1;

    while term != 0 {
        term = wide::shift_right_wrapping(wide::multiply(term, exponent), WORKING_BITS) / step;
        total += term;
        step += 1;
    }

    total
}

/// A mantissa in `[1, 2)`, given in the internal working format, doubled
/// `halvings` times and brought back to a [`Fixed`].
///
/// `None` when the result is too large to hold; zero when it is too small,
/// which is the correctly rounded answer rather than a failure.
fn place<const FRACTION_BITS: u32>(
    mantissa: u128,
    halvings: i128,
) -> Option<FixedPoint<FRACTION_BITS>> {
    let shift: i128 = halvings - (WORKING_BITS - FRACTION_BITS) as i128;

    if shift <= -(u128::BITS as i128) {
        return Some(FixedPoint::ZERO);
    }

    if shift < 0 {
        return Some(FixedPoint(round_shift(mantissa as i128, (-shift) as u32)));
    }

    let widened: wide::U256 = wide::shift_left(mantissa, shift as u32);

    (widened.high == 0 && widened.low <= i128::MAX as u128)
        .then_some(FixedPoint(widened.low as i128))
}

/// A signed value divided by `2^places`, rounded to nearest with ties to even.
///
/// # Why this rule, and why one helper
///
/// This is the rounding every narrowing step in the file goes through — the
/// logarithms, the exponentials, the trigonometry and the named constants — so that
/// they cannot drift apart. Ties to even matches [`Unit`](crate::math::Unit), which
/// makes it the crate's standard, and it is the only common rule that does not bias
/// a long run of roundings in one direction.
///
/// It is deliberately *not* what the arithmetic operators do: [`FixedPoint::checked_mul`]
/// and friends truncate towards zero, because a product is an exact value being cut
/// to the grid rather than a real number being named. Both are documented where they
/// are used.
///
/// # Why not `(value + half) >> places`
///
/// That form is what this replaced. It rounds halves towards positive infinity, so
/// it treats the two signs differently, and `value + half` overflows outright near
/// the top of the range. Here the shift happens first, so nothing can overflow: the
/// quotient is at most half the input and `quotient + 1` cannot leave the type.
const fn round_shift(value: i128, places: u32) -> i128 {
    if places == 0 {
        return value;
    }

    if places >= i128::BITS {
        // Everything has been shifted away; what is left is below half a step.
        return 0;
    }

    // An arithmetic shift floors, and masking gives a remainder that is never
    // negative, so one comparison decides the rounding for both signs alike.
    let quotient: i128 = value >> places;
    let remainder: i128 = value & ((1 << places) - 1);
    let half: i128 = 1 << (places - 1);

    if remainder > half || (remainder == half && (quotient & 1) == 1) {
        quotient + 1
    } else {
        quotient
    }
}

// ---------------------------------------------------------------------------
