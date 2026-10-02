//! Exact decimal notation, read with integers and never through a float.
//!
//! # The problem this solves
//!
//! ```ignore
//! let gravity = Fixed::from_f64(9.80665);
//! ```
//!
//! By the time `from_f64` runs, `9.80665` is gone. The compiler replaced it with the
//! nearest `f64`, which is `9.8066500000000004135358466533012688159942626953125`, and
//! no amount of care downstream can recover what was written. The fixed-point value
//! that comes out is the correctly rounded image of *that* number, not of the decimal
//! in the source.
//!
//! It is a small error, and it is also an avoidable one. A decimal literal is an
//! exact rational — `980665 / 100000` — and converting a rational to a fixed-point
//! grid is integer arithmetic. The only thing missing is a way to get at the digits
//! before the compiler turns them into a float, which is what [`fixed!`](crate::fixed)
//! and [`unit!`](crate::unit) do with `stringify!`.
//!
//! # The path
//!
//! ```text
//! source decimal tokens  ->  Decimal  ->  integer scaling  ->  rounding  ->  raw bits
//! ```
//!
//! No step is a float. The same [`Decimal`] serves the compile-time macros and the
//! runtime string parsers, so a constant and a parsed configuration value that spell
//! the same number land on the same bits.
//!
//! # Rounding
//!
//! Nearest, ties to even, matching [`Unit`](crate::math::Unit) and the narrowing
//! steps inside [`FixedPoint`](crate::math::FixedPoint).
//!
//! This is deliberately *not* what [`FixedPoint`](crate::math::FixedPoint)'s
//! arithmetic operators do — they
//! truncate towards zero. The two are different operations. A product is an exact
//! value being cut to the grid; a literal is a real number being named, and naming it
//! should pick the nearest grid point rather than the one below.

use std::fmt;

/// The largest number of significant digits a [`Decimal`] can hold.
///
/// `u128` reaches `3.4e38`, so 38 digits always fit and a 39th sometimes does not.
/// The limit is stated rather than discovered so that the error is predictable.
pub const MAX_SIGNIFICANT_DIGITS: u32 = 38;

/// A decimal number held exactly, as `significand * 10^exponent`.
///
/// `"12.345e-2"` arrives as significand `12345` and exponent `-5`, which is the same
/// number with none of the digits lost. Nothing here is approximate: the conversions
/// below are where a value first meets a grid it may not sit on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Decimal {
    negative: bool,
    significand: u128,
    exponent: i32,
}

/// Why a decimal could not be read, or could not be represented.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecimalError {
    /// There was nothing to read.
    Empty,
    /// The text is not a decimal number.
    Invalid,
    /// More significant digits than [`MAX_SIGNIFICANT_DIGITS`].
    ///
    /// Reported rather than quietly truncated: dropping digits would change the
    /// value, and a literal that cannot be represented should say so.
    TooManyDigits,
    /// The value is outside what the target type can hold.
    OutOfRange,
    /// The value is outside `[0, 1]`, where the target requires a unit interval.
    NotAUnit,
}

impl DecimalError {
    /// A fixed message, so it can be used from a `const` context where formatting
    /// is not available.
    pub const fn message(self) -> &'static str {
        match self {
            Self::Empty => "the decimal literal is empty",
            Self::Invalid => "the decimal literal is malformed",
            Self::TooManyDigits => {
                "the decimal literal has more significant digits than can be held exactly"
            }
            Self::OutOfRange => "the decimal literal is outside the range of the target type",
            Self::NotAUnit => "the decimal literal is outside the unit interval [0, 1]",
        }
    }
}

impl fmt::Display for DecimalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message())
    }
}

impl std::error::Error for DecimalError {}

// ---------------------------------------------------------------------------
// Reading the text
// ---------------------------------------------------------------------------

/// Reads decimal notation exactly.
///
/// # Accepted
///
/// ```text
/// 0        1        1.2       -1.2      +1.2
/// 1_000.25           0.000001
/// 1e6      1E6       1.25e-4   1e+6
/// 1_000_000.000_001
/// ```
///
/// Underscores are ignored wherever they fall. A leading `+` or `-` is accepted, and
/// so is an exponent with its own sign.
///
/// # Rejected
///
/// ```text
/// 1..2     1e      e10     --1     1.2.3     0x10     1f32
/// ```
///
/// Rust's literal suffixes are not accepted. A suffix says how to *store* a number,
/// and the storage here is already decided by the target type, so allowing one would
/// only create two ways to say the same thing and a way to say a contradictory one.
///
/// # Const
///
/// This is a `const fn`, which is what lets the literal macros do their work while
/// the program is being compiled.
pub const fn parse(text: &str) -> Result<Decimal, DecimalError> {
    let bytes: &[u8] = text.as_bytes();

    if bytes.is_empty() {
        return Err(DecimalError::Empty);
    }

    let mut index: usize = 0;
    let mut negative: bool = false;

    if bytes[0] == b'-' {
        negative = true;
        index = 1;
    } else if bytes[0] == b'+' {
        index = 1;
    }

    let mut significand: u128 = 0;
    let mut digits: u32 = 0;
    let mut exponent: i32 = 0;
    let mut seen_point: bool = false;
    let mut seen_digit: bool = false;
    let mut at_exponent: bool = false;

    while index < bytes.len() {
        let byte: u8 = bytes[index];

        if byte == b'_' {
            index += 1;
            continue;
        }

        if byte == b'.' {
            if seen_point {
                return Err(DecimalError::Invalid);
            }

            seen_point = true;
            index += 1;
            continue;
        }

        if byte == b'e' || byte == b'E' {
            at_exponent = true;
            index += 1;
            break;
        }

        if byte < b'0' || byte > b'9' {
            return Err(DecimalError::Invalid);
        }

        seen_digit = true;
        let digit: u128 = (byte - b'0') as u128;

        // A leading zero carries no information, so it does not count against the
        // digit budget: `0.000001` is one significant digit, not seven.
        if significand == 0 && digit == 0 {
            if seen_point {
                exponent -= 1;
            }

            index += 1;
            continue;
        }

        digits += 1;

        if digits > MAX_SIGNIFICANT_DIGITS {
            return Err(DecimalError::TooManyDigits);
        }

        significand = significand * 10 + digit;

        if seen_point {
            exponent -= 1;
        }

        index += 1;
    }

    if !seen_digit {
        return Err(DecimalError::Invalid);
    }

    if at_exponent {
        let mut exponent_negative: bool = false;

        if index < bytes.len() {
            if bytes[index] == b'-' {
                exponent_negative = true;
                index += 1;
            } else if bytes[index] == b'+' {
                index += 1;
            }
        }

        let mut written: i32 = 0;
        let mut exponent_digits: u32 = 0;

        while index < bytes.len() {
            let byte: u8 = bytes[index];

            if byte == b'_' {
                index += 1;
                continue;
            }

            if byte < b'0' || byte > b'9' {
                return Err(DecimalError::Invalid);
            }

            // Far beyond any representable value, and the cap is what keeps this
            // from overflowing on a literal like `1e999999999`.
            if written > 1_000_000 {
                return Err(DecimalError::OutOfRange);
            }

            written = written * 10 + (byte - b'0') as i32;
            exponent_digits += 1;
            index += 1;
        }

        if exponent_digits == 0 {
            return Err(DecimalError::Invalid);
        }

        exponent += if exponent_negative { -written } else { written };
    }

    Ok(Decimal {
        // Zero has no sign here: `-0.0` is the same value as `0.0`, and the types
        // this feeds have no negative zero to carry it into.
        negative: negative && significand != 0,
        significand,
        exponent,
    })
}

// ---------------------------------------------------------------------------
// Wide integer arithmetic, in const form
//
// `fixed::wide` does this already, but none of it is `const`, and these have to run
// while the program is being compiled. They are deliberately plain: a compile-time
// division runs once, so clarity is worth more here than speed.
// ---------------------------------------------------------------------------

/// A 128-bit value scaled up by `places`, as a 256-bit `(high, low)`.
const fn widening_shift(value: u128, places: u32) -> (u128, u128) {
    if places == 0 {
        (0, value)
    } else if places < 128 {
        (value >> (128 - places), value << places)
    } else {
        (value << (places - 128), 0)
    }
}

/// A 256-bit value divided by a 128-bit one: `((quotient_high, quotient_low), remainder)`.
///
/// Shift-and-subtract, one bit at a time. 256 iterations is nothing at compile time,
/// and the alternative — reaching for the crate's fast division — is not available
/// here because none of it is `const`.
const fn divide_wide(high: u128, low: u128, divisor: u128) -> ((u128, u128), u128) {
    let mut remainder: u128 = 0;
    let mut quotient_high: u128 = 0;
    let mut quotient_low: u128 = 0;
    let mut step: u32 = 0;

    while step < 256 {
        let bit: u128 = if step < 128 {
            (high >> (127 - step)) & 1
        } else {
            (low >> (255 - step)) & 1
        };

        // Whether the remainder is about to lose its top bit, which decides the
        // comparison below on its own: a 129-bit remainder always exceeds a
        // 128-bit divisor.
        let overflowing: bool = remainder >> 127 == 1;

        remainder = (remainder << 1) | bit;
        quotient_high = (quotient_high << 1) | (quotient_low >> 127);
        quotient_low <<= 1;

        if overflowing || remainder >= divisor {
            remainder = remainder.wrapping_sub(divisor);
            quotient_low |= 1;
        }

        step += 1;
    }

    ((quotient_high, quotient_low), remainder)
}

/// Ten raised to `power`, for `power` up to 38.
const fn power_of_ten(power: u32) -> u128 {
    let mut total: u128 = 1;
    let mut step: u32 = 0;

    while step < power {
        total *= 10;
        step += 1;
    }

    total
}

/// Rounds `quotient` up or not, from the remainder it left behind.
///
/// Nearest, ties to even. `sticky` says whether anything was discarded below the
/// remainder in an earlier division, which turns an exact half into more than half.
const fn round_quotient(quotient: u128, remainder: u128, divisor: u128, sticky: bool) -> u128 {
    // `divisor` is a power of ten and so even, except for `10^0 = 1`, where the
    // remainder is always zero and nothing below is reached.
    let half: u128 = divisor / 2;

    if remainder > half || (remainder == half && sticky) {
        return quotient + 1;
    }

    if remainder == half && divisor != 1 && quotient & 1 == 1 {
        return quotient + 1;
    }

    quotient
}

// ---------------------------------------------------------------------------
// Landing on a grid
// ---------------------------------------------------------------------------

impl Decimal {
    /// Whether the value is negative. Zero is never negative.
    pub const fn is_negative(self) -> bool {
        self.negative
    }

    /// The digits, without the decimal point or the sign.
    pub const fn significand(self) -> u128 {
        self.significand
    }

    /// The power of ten the significand is multiplied by.
    pub const fn exponent(self) -> i32 {
        self.exponent
    }

    /// The raw bits of the nearest [`FixedPoint`](crate::math::FixedPoint) with
    /// `fraction_bits` fractional bits.
    ///
    /// Nearest, ties to even. The whole calculation is
    ///
    /// ```text
    /// raw = round(significand * 10^exponent * 2^fraction_bits)
    /// ```
    ///
    /// done exactly in 256-bit integers, so there is no intermediate to round twice
    /// through — which is the error that converting by way of an `f64` would make.
    pub const fn to_fixed_bits(self, fraction_bits: u32) -> Result<i128, DecimalError> {
        let magnitude: u128 = match self.scaled_magnitude(fraction_bits) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };

        if self.negative {
            // The negative side reaches one further than the positive one.
            if magnitude > i128::MAX as u128 + 1 {
                return Err(DecimalError::OutOfRange);
            }

            Ok((magnitude as i128).wrapping_neg())
        } else {
            if magnitude > i128::MAX as u128 {
                return Err(DecimalError::OutOfRange);
            }

            Ok(magnitude as i128)
        }
    }

    /// The raw count of `2^-63` for the nearest [`Unit`](crate::math::Unit).
    ///
    /// [`DecimalError::NotAUnit`] outside `[0, 1]`: a literal that is not a unit
    /// value is a mistake in the source, and silently folding it to zero or one would
    /// hide that.
    pub const fn to_unit_bits(self) -> Result<u64, DecimalError> {
        if self.negative {
            return Err(DecimalError::NotAUnit);
        }

        // Unit's scale is 2^63, so its raw count and a 63-bit fixed-point layout are
        // the same integer — one conversion serves both.
        let bits: i128 = match self.to_fixed_bits(63) {
            Ok(value) => value,
            Err(DecimalError::OutOfRange) => return Err(DecimalError::NotAUnit),
            Err(error) => return Err(error),
        };

        if bits > 1 << 63 {
            return Err(DecimalError::NotAUnit);
        }

        Ok(bits as u64)
    }

    /// `|value| * 2^fraction_bits`, rounded to nearest with ties to even.
    const fn scaled_magnitude(self, fraction_bits: u32) -> Result<u128, DecimalError> {
        if self.significand == 0 {
            return Ok(0);
        }

        if self.exponent >= 0 {
            // Whole: multiply out, and refuse anything that leaves the range on the
            // way rather than wrapping quietly.
            let mut total: u128 = self.significand;
            let mut remaining: i32 = self.exponent;

            while remaining > 0 {
                let Some(next) = total.checked_mul(10) else {
                    return Err(DecimalError::OutOfRange);
                };

                total = next;
                remaining -= 1;
            }

            let (high, low): (u128, u128) = widening_shift(total, fraction_bits);

            if high != 0 {
                return Err(DecimalError::OutOfRange);
            }

            return Ok(low);
        }

        // Fractional: divide, exactly, keeping the remainder so the rounding below
        // is decided rather than guessed.
        let mut places: u32 = self.exponent.unsigned_abs();
        let (mut high, mut low): (u128, u128) = widening_shift(self.significand, fraction_bits);
        let mut sticky: bool = false;

        // A power of ten above 10^38 does not fit a `u128`, so a long fractional
        // exponent is divided out in whole steps first. The sticky flag carries
        // whatever those steps discarded, which is all the rounding below needs.
        while places > MAX_SIGNIFICANT_DIGITS {
            let step: u128 = power_of_ten(MAX_SIGNIFICANT_DIGITS);
            let ((quotient_high, quotient_low), remainder) = divide_wide(high, low, step);

            if remainder != 0 {
                sticky = true;
            }

            high = quotient_high;
            low = quotient_low;
            places -= MAX_SIGNIFICANT_DIGITS;
        }

        let divisor: u128 = power_of_ten(places);
        let ((quotient_high, quotient_low), remainder) = divide_wide(high, low, divisor);

        if quotient_high != 0 {
            return Err(DecimalError::OutOfRange);
        }

        Ok(round_quotient(quotient_low, remainder, divisor, sticky))
    }
}

// ---------------------------------------------------------------------------
// What the macros call
// ---------------------------------------------------------------------------

/// The raw bits for a fixed-point literal, failing the build on a bad one.
///
/// Called by [`fixed!`](crate::fixed) and [`fixed_point!`](crate::fixed_point). A
/// `panic!` reached while a constant is being evaluated is a compile error, which is
/// exactly the behaviour wanted: a malformed or out-of-range literal should never
/// become a silently clamped value at run time.
#[track_caller]
pub const fn fixed_bits(text: &str, fraction_bits: u32) -> i128 {
    let decimal: Decimal = match parse(text) {
        Ok(value) => value,
        Err(error) => panic!("{}", error.message()),
    };

    match decimal.to_fixed_bits(fraction_bits) {
        Ok(bits) => bits,
        Err(error) => panic!("{}", error.message()),
    }
}

/// The raw bits for a unit-interval literal, failing the build on a bad one.
///
/// Called by [`unit!`](crate::unit).
#[track_caller]
pub const fn unit_bits(text: &str) -> u64 {
    let decimal: Decimal = match parse(text) {
        Ok(value) => value,
        Err(error) => panic!("{}", error.message()),
    };

    match decimal.to_unit_bits() {
        Ok(bits) => bits,
        Err(error) => panic!("{}", error.message()),
    }
}

// ---------------------------------------------------------------------------
// The literal macros
// ---------------------------------------------------------------------------

/// A [`Fixed`](crate::math::Fixed) from decimal notation, read exactly.
///
/// # Question
///
/// "What `Fixed` value most closely represents this decimal number, exactly as
/// written?"
///
/// # Example
///
/// ```
/// use voxel_world::fixed;
/// use voxel_world::math::Fixed;
///
/// // In a constant, which is where most of these belong.
/// const GRAVITY: Fixed = fixed!(9.80665);
///
/// let a = fixed!(1.2);
/// let b = fixed!(-17.625);
/// let c = fixed!(1_000.25);
/// let d = fixed!(1.25e-4);
/// let e = fixed!(3e6);
///
/// // Exactly representable values are exact.
/// assert_eq!(fixed!(0.5), Fixed::ONE / 2);
/// assert_eq!(fixed!(1e3), fixed!(1000));
/// # let _ = (a, b, c, d, e, GRAVITY);
/// ```
///
/// # Never through a float
///
/// The literal's own text is recovered with `stringify!` and read by
/// [`parse`], so `1.2` is treated as the exact rational `12/10` and rounded once onto
/// the fixed-point grid. Writing `Fixed::from_f64(1.2)` instead would round twice:
/// first to the nearest `f64`, then to the nearest `Fixed`, and the two roundings do
/// not always compose to the nearest value.
///
/// [`FixedPoint::from_f64`](crate::math::FixedPoint::from_f64) remains the right
/// call when there genuinely is an `f64` in hand — a sensor reading, something from a
/// float API. It answers a different question.
///
/// # Rounding
///
/// Nearest, ties to even.
///
/// # Failure
///
/// At compile time. A malformed literal, one with more significant digits than can be
/// held, or one outside the range fails the build with a message rather than becoming
/// a clamped value.
///
/// ```compile_fail
/// # use voxel_world::fixed;
/// const TOO_BIG: voxel_world::math::Fixed = fixed!(1e999999999);
/// ```
#[macro_export]
macro_rules! fixed {
    ($literal:literal) => {
        $crate::math::fixed::Fixed::from_bits(const {
            $crate::math::decimal::fixed_bits(
                ::core::stringify!($literal),
                <$crate::math::fixed::Fixed>::FRACTION_BITS,
            )
        })
    };
    (+ $literal:literal) => {
        $crate::fixed!($literal)
    };
}

/// A [`FixedPoint<N>`](crate::math::FixedPoint) of a chosen width, from decimal
/// notation.
///
/// # Question
///
/// "What value of *this* layout most closely represents this decimal number?"
///
/// # Example
///
/// ```
/// use voxel_world::fixed_point;
/// use voxel_world::math::FixedPoint;
///
/// const NARROW: FixedPoint<16> = fixed_point!(16, 1.25);
/// const WIDE: FixedPoint<48> = fixed_point!(48, 1.25);
///
/// // 1.25 is a dyadic rational, so both hold it exactly.
/// assert_eq!(NARROW.to_bits(), 1 << 16 | 1 << 14);
/// assert_eq!(WIDE.to_bits(), 1 << 48 | 1 << 46);
/// ```
///
/// [`fixed!`] is the one to reach for at the default width; this exists for the
/// layouts that have no alias of their own.
#[macro_export]
macro_rules! fixed_point {
    ($bits:literal, $literal:literal) => {
        $crate::math::fixed::FixedPoint::<$bits>::from_bits(const {
            $crate::math::decimal::fixed_bits(::core::stringify!($literal), $bits)
        })
    };
    ($bits:literal, + $literal:literal) => {
        $crate::fixed_point!($bits, $literal)
    };
}

/// A [`Unit`](crate::math::Unit) from decimal notation, read exactly.
///
/// # Question
///
/// "What `Unit` most closely represents this decimal value between zero and one?"
///
/// # Example
///
/// ```
/// use voxel_world::unit;
/// use voxel_world::math::Unit;
///
/// const CHANCE: Unit = unit!(0.025);
///
/// assert_eq!(unit!(0.5), Unit::HALF);
/// assert_eq!(unit!(1.0), Unit::ONE);
/// assert_eq!(unit!(0), Unit::ZERO);
/// assert_eq!(unit!(2.5e-3), unit!(0.0025));
/// # let _ = CHANCE;
/// ```
///
/// # Never through a float
///
/// As [`fixed!`]: the decimal text is read with integers and rounded once onto the
/// `k / 2^63` grid, nearest with ties to even — the same rule
/// [`Unit::try_from_ratio`](crate::math::Unit::try_from_ratio) uses, so a literal and
/// a converted ratio naming the same number agree.
///
/// # Failure
///
/// At compile time, and that includes being out of range. A literal outside `[0, 1]`
/// is a mistake in the source, so it fails the build rather than quietly saturating —
/// if saturation is wanted, it has to be asked for by name.
///
/// ```compile_fail
/// # use voxel_world::unit;
/// const TOO_LARGE: voxel_world::math::Unit = unit!(1.1);
/// ```
///
/// ```compile_fail
/// # use voxel_world::unit;
/// const NEGATIVE: voxel_world::math::Unit = unit!(-0.1);
/// ```
#[macro_export]
macro_rules! unit {
    ($literal:literal) => {
        // The count is proven to be within `[0, 2^63]` before it gets here, so the
        // clamping form cannot clamp; it is used only because it needs no unwrap.
        $crate::math::unit_interval::Unit::from_bits_clamped(const {
            $crate::math::decimal::unit_bits(::core::stringify!($literal))
        })
    };
    (+ $literal:literal) => {
        $crate::unit!($literal)
    };
}

// ---------------------------------------------------------------------------
// Runtime text, including more digits than a `u128` can hold
// ---------------------------------------------------------------------------

/// The raw bits for a fixed-point value from decimal text, exactly, at any length.
///
/// # Why there are two paths
///
/// [`parse`] holds its significand in a `u128`, which runs out at
/// [`MAX_SIGNIFICANT_DIGITS`]. That is the right answer for a literal — a constant
/// with forty significant digits is almost certainly a mistake, and failing the build
/// says so. It is the wrong answer for run-time text, which may come from a file
/// nobody writing this code controls.
///
/// So a long input falls through to the crate's own [`BigUint`](crate::math::BigUint),
/// where the same calculation is done with no width limit at all. The fast path and
/// the exact path round identically, and `testing/tests/decimal_literals.rs` checks
/// that they agree wherever both apply.
pub fn fixed_bits_exact(text: &str, fraction_bits: u32) -> Result<i128, DecimalError> {
    match parse(text) {
        Ok(decimal) => decimal.to_fixed_bits(fraction_bits),
        Err(DecimalError::TooManyDigits) => long_fixed_bits(text, fraction_bits),
        Err(error) => Err(error),
    }
}

/// The raw count of `2^-63` for a [`Unit`](crate::math::Unit) from decimal text.
pub fn unit_bits_exact(text: &str) -> Result<u64, DecimalError> {
    match parse(text) {
        Ok(decimal) => decimal.to_unit_bits(),
        Err(DecimalError::TooManyDigits) => {
            if text.trim_start().starts_with('-') {
                return Err(DecimalError::NotAUnit);
            }

            let bits: i128 = long_fixed_bits(text, 63)?;

            if !(0..=1 << 63).contains(&bits) {
                return Err(DecimalError::NotAUnit);
            }

            Ok(bits as u64)
        }
        Err(error) => Err(error),
    }
}

/// The sign, every digit, and the power of ten, with no limit on how many digits.
fn split(text: &str) -> Result<(bool, String, i32), DecimalError> {
    let bytes: &[u8] = text.as_bytes();

    if bytes.is_empty() {
        return Err(DecimalError::Empty);
    }

    let mut index: usize = 0;
    let mut negative: bool = false;

    if bytes[0] == b'-' {
        negative = true;
        index = 1;
    } else if bytes[0] == b'+' {
        index = 1;
    }

    let mut digits = String::new();
    let mut exponent: i32 = 0;
    let mut seen_point: bool = false;
    let mut seen_digit: bool = false;
    let mut at_exponent: bool = false;

    while index < bytes.len() {
        match bytes[index] {
            b'_' => {}
            b'.' if !seen_point => seen_point = true,
            b'.' => return Err(DecimalError::Invalid),
            b'e' | b'E' => {
                at_exponent = true;
                index += 1;
                break;
            }
            byte if byte.is_ascii_digit() => {
                seen_digit = true;
                digits.push(byte as char);

                if seen_point {
                    exponent -= 1;
                }
            }
            _ => return Err(DecimalError::Invalid),
        }

        index += 1;
    }

    if !seen_digit {
        return Err(DecimalError::Invalid);
    }

    if at_exponent {
        let tail: &str = &text[index..];
        let (tail, tail_negative) = match tail.strip_prefix('-') {
            Some(rest) => (rest, true),
            None => (tail.strip_prefix('+').unwrap_or(tail), false),
        };

        let cleaned: String = tail.chars().filter(|character| *character != '_').collect();

        if cleaned.is_empty() || !cleaned.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(DecimalError::Invalid);
        }

        let written: i32 = cleaned.parse::<i32>().map_err(|_| DecimalError::OutOfRange)?;

        exponent += if tail_negative { -written } else { written };
    }

    Ok((negative, digits, exponent))
}

/// The exact conversion, with the digits held in a [`BigUint`](crate::math::BigUint).
fn long_fixed_bits(text: &str, fraction_bits: u32) -> Result<i128, DecimalError> {
    use crate::math::BigUint;

    let (negative, digits, exponent) = split(text)?;

    let significand: BigUint = BigUint::parse_decimal(&digits).ok_or(DecimalError::Invalid)?;

    if significand.is_zero() {
        return Ok(0);
    }

    let ten_to = |power: u32| -> Result<BigUint, DecimalError> {
        let mut text = String::from("1");
        text.extend(std::iter::repeat_n('0', power as usize));

        BigUint::parse_decimal(&text).ok_or(DecimalError::OutOfRange)
    };

    let magnitude: BigUint = if exponent >= 0 {
        // Whole: nothing to round, the value simply has to fit.
        significand
            .product(&ten_to(exponent.unsigned_abs())?)
            .shifted_up(u64::from(fraction_bits))
    } else {
        let divisor: BigUint = ten_to(exponent.unsigned_abs())?;
        let scaled: BigUint = significand.shifted_up(u64::from(fraction_bits));

        let (quotient, remainder) = scaled.div_rem(&divisor).ok_or(DecimalError::Invalid)?;

        // Nearest, ties to even, exactly as the `u128` path does it: twice the
        // remainder against the divisor is the comparison with a half.
        let doubled: BigUint = remainder.shifted_up(1);

        let round_up: bool = match doubled.cmp(&divisor) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Equal => quotient.bit(0),
            std::cmp::Ordering::Less => false,
        };

        if round_up {
            quotient.sum(&BigUint::one())
        } else {
            quotient
        }
    };

    let bits: u128 = magnitude.to_u128().ok_or(DecimalError::OutOfRange)?;

    if negative {
        if bits > i128::MAX as u128 + 1 {
            return Err(DecimalError::OutOfRange);
        }

        Ok((bits as i128).wrapping_neg())
    } else {
        if bits > i128::MAX as u128 {
            return Err(DecimalError::OutOfRange);
        }

        Ok(bits as i128)
    }
}
