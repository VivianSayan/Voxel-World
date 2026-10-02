// Reading and writing
// ---------------------------------------------------------------------------

/// The shortest decimal that reads back as this exact value, which is at most
/// ten fractional digits and none at all for a whole number.
///
/// A step of `2^-32` is a finite decimal, but writing it out in full takes 32
/// digits: the stored value nearest `0.1` is exactly
/// `0.09999999986030161380767822265625`. Printing the shortest decimal that
/// [`FromStr`] maps back to the same bits keeps text round-trips honest and
/// readable at once. [`Fixed::to_exact_string`] writes the full expansion where
/// that is what is wanted.
///
/// A precision, as in `{:.3}`, gives exactly that many fractional digits,
/// truncated rather than rounded, so every digit printed is one the value
/// really has.
impl<const FRACTION_BITS: u32> fmt::Display for FixedPoint<FRACTION_BITS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let negative: bool = self.0 < 0;
        let magnitude: u128 = self.0.unsigned_abs();
        let whole: u128 = magnitude >> FRACTION_BITS;
        let fraction: u128 = magnitude & (Self::SCALE as u128 - 1);

        let digits: String = match formatter.precision() {
            Some(wanted) => truncated_digits(fraction, wanted, FRACTION_BITS),
            None => shortest_digits(fraction, FRACTION_BITS),
        };

        let text: String = if digits.is_empty() {
            whole.to_string()
        } else {
            format!("{whole}.{digits}")
        };

        formatter.pad_integral(!negative, "", &text)
    }
}

impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// Every digit of the exact decimal value, which for a fraction is always
    /// 32 of them.
    ///
    /// Each step is `2^-32`, and a negative power of two is a finite decimal,
    /// so this terminates and is exact rather than rounded. Useful for a
    /// snapshot or a golden test, where the point is to pin the stored bits and
    /// not to be read.
    pub fn to_exact_string(self) -> String {
        let magnitude: u128 = self.0.unsigned_abs();
        let whole: u128 = magnitude >> FRACTION_BITS;
        let digits: String = truncated_digits(
            magnitude & (Self::SCALE as u128 - 1),
            FRACTION_BITS as usize,
            FRACTION_BITS,
        );
        let sign: &str = if self.0 < 0 { "-" } else { "" };

        if digits.is_empty() {
            return format!("{sign}{whole}");
        }

        format!("{sign}{whole}.{digits}")
    }
}

/// The first `wanted` digits of a fraction's decimal expansion, taken by
/// repeatedly multiplying by ten and lifting the digit that carries past the
/// point. Empty once the expansion has ended.
fn truncated_digits(fraction: u128, wanted: usize, fraction_bits: u32) -> String {
    let scale: u128 = 1u128 << fraction_bits;
    let mut remaining: u128 = fraction;
    let mut digits: String = String::new();

    while digits.len() < wanted {
        remaining *= 10;
        digits.push(char::from_digit((remaining >> fraction_bits) as u32, 10).unwrap());
        remaining &= scale - 1;
    }

    digits
}

/// The fewest digits, rounded to nearest, that read back as this exact
/// fraction.
///
/// Enough digits always suffice: a step of `2^-f` is coarser than `10^-ceil(f/3)`,
/// so a little over a third of the fractional bits is always a wide enough decimal.
/// At the default 32 bits that is the familiar ten digits.
fn shortest_digits(fraction: u128, fraction_bits: u32) -> String {
    if fraction == 0 {
        return String::new();
    }

    let scale: u128 = 1u128 << fraction_bits;
    // log10(2) is a little over 0.301, so this never under-counts.
    let limit: usize = (fraction_bits as usize * 302 / 1000) + 1;
    let mut power: u128 = 1;

    for places in 1..=limit {
        power *= 10;

        let rounded: u128 = ((fraction * power) + scale / 2) >> fraction_bits;

        if rounded < power && (rounded << fraction_bits) / power == fraction {
            return format!("{rounded:0>places$}");
        }
    }

    let rounded: u128 = ((fraction * power) + scale / 2) >> fraction_bits;

    format!("{rounded:0>limit$}")
}

/// As [`Display`](fmt::Display), tagged so a value is recognisable in a dump.
impl<const FRACTION_BITS: u32> fmt::Debug for FixedPoint<FRACTION_BITS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Fixed({self})")
    }
}

/// What went wrong while reading a [`Fixed`] from text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseFixedError {
    /// The text held no digits at all, or nothing before the point.
    Empty,
    /// The text held something that is not a digit, a sign or a single point.
    Invalid,
    /// The value is outside the range a [`Fixed`] can hold.
    OutOfRange,
}

impl fmt::Display for ParseFixedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "no digits to read",
            Self::Invalid => "not a fixed-point number",
            Self::OutOfRange => "outside the fixed-point range",
        })
    }
}

impl std::error::Error for ParseFixedError {}

impl From<ParseIntError> for ParseFixedError {
    fn from(_: ParseIntError) -> Self {
        Self::Invalid
    }
}

/// Reads decimal text such as `-12.25`, exactly.
///
/// The whole part is read as an integer and the fractional digits as a fraction
/// over a power of ten, then scaled and truncated towards zero, so the result
/// is the nearest representable value at or before the one written. No `f64` is
/// involved at any point, which is what keeps a written-down value and the one
/// read back from it in step.
impl<const FRACTION_BITS: u32> FixedPoint<FRACTION_BITS> {
    /// Reads decimal notation exactly, with no float anywhere in between.
    ///
    /// # Question
    ///
    /// "What value of this layout most closely represents the decimal number in this
    /// text?"
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    ///
    /// assert_eq!(Fixed::from_decimal_str("1.25").unwrap(), Fixed::ONE + Fixed::ONE / 4);
    /// assert_eq!(Fixed::from_decimal_str("1e3").unwrap(), Fixed::from_integer(1000));
    ///
    /// assert!(Fixed::from_decimal_str("1..2").is_err());
    /// ```
    ///
    /// # What it accepts
    ///
    /// The syntax [`decimal::parse`] documents: an
    /// optional sign, digits with at most one point, underscores anywhere, and an
    /// optional `e` or `E` exponent with its own sign.
    ///
    /// # Rounding, and how it differs from what this used to do
    ///
    /// Nearest, ties to even — the same rule [`fixed!`](crate::fixed) uses, so the
    /// literal and the parsed string agree bit for bit.
    ///
    /// The earlier implementation truncated towards zero and stopped after 38
    /// fractional digits. Both are gone: a long decimal is now carried through
    /// [`BigUint`](crate::math::BigUint) and rounded once, exactly.
    pub fn from_decimal_str(text: &str) -> Result<Self, ParseFixedError> {
        decimal::fixed_bits_exact(text, FRACTION_BITS)
            .map(Self)
            .map_err(ParseFixedError::from)
    }
}

impl From<decimal::DecimalError> for ParseFixedError {
    fn from(error: decimal::DecimalError) -> Self {
        match error {
            decimal::DecimalError::Empty => Self::Empty,
            decimal::DecimalError::Invalid => Self::Invalid,
            decimal::DecimalError::TooManyDigits
            | decimal::DecimalError::OutOfRange
            | decimal::DecimalError::NotAUnit => Self::OutOfRange,
        }
    }
}

/// Reads decimal notation through [`FixedPoint::from_decimal_str`], which never
/// involves a float.
impl<const FRACTION_BITS: u32> FromStr for FixedPoint<FRACTION_BITS> {
    type Err = ParseFixedError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::from_decimal_str(text)
    }
}

// ---------------------------------------------------------------------------
