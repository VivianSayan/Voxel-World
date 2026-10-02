impl Seed {
    /// The seed as a code a player can write down and type back in.
    ///
    /// Crockford base32 in four groups of six characters and one of two: 26
    /// digits of five bits, which covers the 128 bits with three to spare. The
    /// alphabet has no `I`, `L`, `O` or `U`, so there is no digit a letter can
    /// be confused with and no accidental word.
    ///
    /// The digits are cut from the low end first, so the grouping is the same
    /// whatever the value's magnitude, and written out most significant first,
    /// which is the order [`Seed::from_code`] reads when it walks the string
    /// backwards.
    pub fn to_code(self) -> String {
        let mut characters: [u8; 26] = [b'0'; 26];

        let mut value: u128 = self.0;

        for slot in characters.iter_mut() {
            *slot = CODE_ALPHABET[(value & 0x1F) as usize];
            value >>= 5;
        }

        let mut code: String = String::with_capacity(30);

        for (position, character) in characters.into_iter().rev().enumerate() {
            if position > 0 && position % 6 == 0 {
                code.push('-');
            }

            code.push(character as char);
        }

        code
    }

    /// Reads a code back, or `None` when it is not one.
    ///
    /// Forgiving about how it was written down: case is ignored, dashes,
    /// spaces and underscores may be anywhere or absent, and the letters `I`
    /// and `L` are read as `1` and `O` as `0`, which is how they are most often
    /// mistyped.
    ///
    /// Strict about shape, though. A code is exactly the 26 digits
    /// [`Seed::to_code`] writes, and the leading one carries only the three
    /// bits left over from 128, so it is never above `7`. Without those two
    /// checks this would accept most ordinary words, since the alphabet is
    /// nearly all of the letters: `"Wandering Hill"` would read as a code
    /// rather than as a name, and [`Seed::parse`] would hand back a world
    /// nobody asked for.
    pub fn from_code(code: &str) -> Option<Self> {
        let mut value: u128 = 0;
        let mut digits: u32 = 0;

        for character in code.chars().rev() {
            if character == '-' || character == ' ' || character == '_' {
                continue;
            }

            let upper: char = character.to_ascii_uppercase();
            let digit: u8 = match upper {
                'I' | 'L' => 1,
                'O' => 0,
                _ => CODE_ALPHABET
                    .iter()
                    .position(|&candidate| candidate == upper as u8)? as u8,
            };

            if digits == 25 && digit > 7 {
                return None;
            }

            if digits == 26 {
                return None;
            }

            value |= (digit as u128) << (digits * 5);
            digits += 1;
        }

        if digits != 26 {
            return None;
        }

        Some(Self(value))
    }

    /// The seed as 32 hexadecimal digits, for logs and save files.
    pub fn to_hex(self) -> String {
        format!("{:032x}", self.0)
    }

    /// Reads back what [`Seed::to_hex`] wrote. Underscores and a leading `0x`
    /// are allowed.
    pub fn from_hex(text: &str) -> Option<Self> {
        let cleaned: String = text
            .trim()
            .trim_start_matches("0x")
            .trim_start_matches("0X")
            .chars()
            .filter(|character| *character != '_')
            .collect();

        u128::from_str_radix(&cleaned, 16).ok().map(Self)
    }
}

// ---------------------------------------------------------------------------
