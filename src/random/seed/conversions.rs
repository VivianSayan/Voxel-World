// Conversions
// ---------------------------------------------------------------------------

/// Every integer type seeds by value, through the same path as
/// [`Seed::from_integer`], so the two can never drift apart.
macro_rules! implement_integer_seed {
    ($($integer:ty),+) => {
        $(
            impl From<$integer> for Seed {
                fn from(value: $integer) -> Self {
                    Self::from_integer(value)
                }
            }
        )+
    };
}

implement_integer_seed!(
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, isize
);

impl From<bool> for Seed {
    fn from(value: bool) -> Self {
        Self::from(value as u8)
    }
}

impl From<char> for Seed {
    fn from(value: char) -> Self {
        Self::from(value as u32)
    }
}

impl From<f64> for Seed {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<f32> for Seed {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<&str> for Seed {
    fn from(value: &str) -> Self {
        Self::from_text(value)
    }
}

impl From<&String> for Seed {
    fn from(value: &String) -> Self {
        Self::from_text(value)
    }
}

impl From<String> for Seed {
    fn from(value: String) -> Self {
        Self::from_text(&value)
    }
}

impl From<&[u8]> for Seed {
    fn from(value: &[u8]) -> Self {
        Self::from_bytes(value)
    }
}

impl<const N: usize> From<[i128; N]> for Seed {
    fn from(value: [i128; N]) -> Self {
        Self::from_raw(0).at(value)
    }
}

impl From<Vector2<i128>> for Seed {
    fn from(value: Vector2<i128>) -> Self {
        Self::from(value.to_array())
    }
}

impl From<Vector3<i128>> for Seed {
    fn from(value: Vector3<i128>) -> Self {
        Self::from(value.to_array())
    }
}

impl From<Vector4<i128>> for Seed {
    fn from(value: Vector4<i128>) -> Self {
        Self::from(value.to_array())
    }
}

impl From<Seed> for u128 {
    fn from(value: Seed) -> Self {
        value.0
    }
}

impl From<Seed> for Random {
    fn from(value: Seed) -> Self {
        value.to_random()
    }
}

impl std::fmt::Display for Seed {
    /// The shareable code, which is what a player should see.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_code())
    }
}

impl std::fmt::Debug for Seed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Seed({} / {:#034x})", self.to_code(), self.0)
    }
}

// ---------------------------------------------------------------------------
