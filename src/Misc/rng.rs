#[derive(Clone, Debug)]
pub struct Random {
    seed: u128,
    state: [u64; 4],
}

impl Random {
    pub fn new(seed: u128) -> Self {
        let mut random = Self {
            seed,
            state: [0; 4],
        };

        random.reseed(seed);
        random
    }

    pub fn reseed(&mut self, seed: u128) {
        self.seed = seed;

        // Collapse the 128-bit seed into a starting 64-bit value.
        // Both halves influence the resulting state.
        let lower: u64 = seed as u64;
        let upper: u64 = (seed >> 64) as u64;

        let mut seed_state: u64 =
            lower ^ upper.rotate_left(32) ^ 0x9E3779B97F4A7C15;

        // SplitMix64 is used here to expand the seed into the
        // four independent state words required by xoshiro256**.
        for index in 0..4 {
            self.state[index] = Self::splitmix64(&mut seed_state);
        }

        // xoshiro must never have an entirely zero state.
        if self.state == [0; 4] {
            self.state[0] = 0x9E3779B97F4A7C15;
        }
    }

    pub fn seed(&self) -> u128 {
        self.seed
    }
    pub fn next_u16(&mut self) -> u16 {
        let result: u32 = self.next_u32();
        ((result & 0xFFFF) ^ (result >> 32)) as u16
    }

    pub fn next_u32(&mut self) -> u32 {
        let result: u64 = self.next_u64();
        ((result & 0xFFFF) ^ (result >> 32)) as u32
    }

    pub fn next_u64(&mut self) -> u64 {
        // xoshiro256** output transformation.
        let result: u64 = self.state[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);

        // State transition.
        let temporary: u64 = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= temporary;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    pub fn next_f32(&mut self) -> f32 {
        let result: u32 = self.next_u32();
        (result as f32) / (u32::MAX as f32)
    }

    pub fn next_f64(&mut self) -> f64 {
        let result: u64 = self.next_u64();
        (result as f64) / (u64::MAX as f64)
    }

    pub fn next_u128(&mut self) -> u128 {
        let upper: u128 = self.next_u64() as u128;
        let lower: u128 = self.next_u64() as u128;

        (upper << 64) | lower
    }

    pub fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);

        let mut value: u64 = *state;

        value = (value ^ (value >> 30))
            .wrapping_mul(0xBF58476D1CE4E5B9);

        value = (value ^ (value >> 27))
            .wrapping_mul(0x94D049BB133111EB);

        value ^ (value >> 31)
    }

    pub fn nex_i128(&mut self) -> i128 {
        self.next_u128() as i128
    }

    pub fn nex_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    pub fn nex_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }

    pub fn nex_i16(&mut self) -> i16 {
        self.next_u16() as i16
    }
}