//! Full-cycle counter + bijective bit permutation for any bit width
//! from 1 to 128 bits.
//!
//! Every value in `0..2^bits` is produced exactly once per cycle, in a
//! scrambled order determined by the seed. The permutation and the counter
//! are both invertible, so an output value can be mapped back to the index
//! at which it was (or will be) produced.
//!
//! All arithmetic is done modulo `2^bits`, which is why the values are
//! `u128`: wrapping unsigned arithmetic is exactly that ring.

use crate::misc::mixing::mix128;

const NR_BITS_BYTES: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitPermuter {
    bits: u32,
    seed: u128,
    step: u128,
    offset: u128,

    mask: u128,
    counter: u128,

    // Cached permutation parameters
    mul1: u128,
    mul2: u128,
    mul3: u128,
    add1: u128,
    add2: u128,
    shift1: u32,
    shift2: u32,

    // Cached inverses
    inv_mul1: u128,
    inv_mul2: u128,
    inv_mul3: u128,
    inv_step: u128,
}

impl BitPermuter {
    pub fn new(bits: u32, seed: u128, step: u128, offset: u128) -> Self {
        assert!((1..=128).contains(&bits), "bits must be in 1..=128");

        let mask = Self::mask_for(bits);

        let mut permuter = Self {
            bits,
            seed: seed & mask,
            // step must be odd for a full cycle over 2^bits
            step: (step | 1) & mask,
            offset: offset & mask,
            mask,
            counter: 0,
            mul1: 0,
            mul2: 0,
            mul3: 0,
            add1: 0,
            add2: 0,
            shift1: 0,
            shift2: 0,
            inv_mul1: 0,
            inv_mul2: 0,
            inv_mul3: 0,
            inv_step: 0,
        };

        permuter.calculate_values();
        permuter
    }

    /// A permuter with step 1 and offset 0.
    pub fn with_seed(bits: u32, seed: u128) -> Self {
        Self::new(bits, seed, 1, 0)
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn seed(&self) -> u128 {
        self.seed
    }

    pub fn step(&self) -> u128 {
        self.step
    }

    pub fn offset(&self) -> u128 {
        self.offset
    }

    /// Number of values already produced by `next_value`, modulo 2^bits.
    pub fn position(&self) -> u128 {
        self.counter.wrapping_mul(self.inv_step) & self.mask
    }

    /// Moves the counter so that the next call to `next_value` returns
    /// the value at `index` (0-based).
    pub fn set_position(&mut self, index: u128) {
        self.counter = index.wrapping_mul(self.step) & self.mask;
    }

    fn mask_for(bits: u32) -> u128 {
        if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        }
    }

    fn calculate_values(&mut self) {
        self.mask = Self::mask_for(self.bits);

        self.mul1 = self.odd_mult(0xA1);
        self.mul2 = self.odd_mult(0xB7);
        self.mul3 = self.odd_mult(0xC3);

        self.add1 = self.add_const(0x11);
        self.add2 = self.add_const(0x29);

        self.shift1 = (self.bits / 2).max(1);
        self.shift2 = (self.bits / 3).max(1);

        // Odd numbers are invertible mod 2^bits
        self.inv_mul1 = self.inv_odd_mod_pow2(self.mul1);
        self.inv_mul2 = self.inv_odd_mod_pow2(self.mul2);
        self.inv_mul3 = self.inv_odd_mod_pow2(self.mul3);
        self.inv_step = self.inv_odd_mod_pow2(self.step);
    }

    pub fn next_value(&mut self) -> u128 {
        self.counter = self.counter.wrapping_add(self.step) & self.mask;
        self.permute_bits(self.counter.wrapping_add(self.offset))
    }

    /// Returns the value that the `index`-th (0-based) call to `next_value`
    /// produces, without changing internal state. Index `2^bits - 1` is the
    /// last value of the cycle; larger indices wrap around.
    pub fn peek_value_at(&self, index: u128) -> u128 {
        let n = index.wrapping_add(1) & self.mask;
        let counter = n.wrapping_mul(self.step) & self.mask;
        self.permute_bits(counter.wrapping_add(self.offset))
    }

    /// Inverse of `peek_value_at`: the 0-based index at which `next_value`
    /// produces `output_value` within a cycle.
    pub fn index_of_output(&self, output_value: u128) -> u128 {
        let permuted = self.invert_permuted(output_value);
        let counter = permuted.wrapping_sub(self.offset) & self.mask;
        let n = counter.wrapping_mul(self.inv_step) & self.mask;

        // next_value() increments before producing output, so n == 0 is the
        // last value of the cycle.
        n.wrapping_sub(1) & self.mask
    }

    pub fn permute_bits(&self, value: u128) -> u128 {
        let mask = self.mask;

        // Keyed xor
        let mut x = (value ^ self.seed) & mask;

        x = x.wrapping_mul(self.mul1).wrapping_add(self.add1) & mask;
        x ^= x >> self.shift1;
        x = x.wrapping_mul(self.mul2).wrapping_add(self.add2) & mask;
        x ^= x >> self.shift2;
        x = x.wrapping_mul(self.mul3) & mask;
        x ^= x >> self.shift1;

        x
    }

    pub fn invert_permuted(&self, value: u128) -> u128 {
        let mask = self.mask;
        let mut x = value & mask;

        // Invert rounds in reverse order
        x = self.unxorshift_right(x, self.shift1);
        x = x.wrapping_mul(self.inv_mul3) & mask;

        x = self.unxorshift_right(x, self.shift2);
        x = x.wrapping_sub(self.add2) & mask;
        x = x.wrapping_mul(self.inv_mul2) & mask;

        x = self.unxorshift_right(x, self.shift1);
        x = x.wrapping_sub(self.add1) & mask;
        x = x.wrapping_mul(self.inv_mul1) & mask;

        // Undo keyed xor
        (x ^ self.seed) & mask
    }

    /// Produces an odd multiplier in [1..mask], derived from the seed.
    fn odd_mult(&self, salt: u128) -> u128 {
        (mix128(self.seed ^ salt) | 1) & self.mask
    }

    fn add_const(&self, salt: u128) -> u128 {
        mix128(self.seed.wrapping_add(salt)) & self.mask
    }

    /// Inverts `x ^= x >> shift`.
    fn unxorshift_right(&self, value: u128, shift: u32) -> u128 {
        let mut x = value & self.mask;
        let mut s = shift;
        while s < self.bits {
            x ^= x >> s;
            s <<= 1;
        }
        x
    }

    /// Newton iteration; each step doubles the number of correct bits.
    /// An odd `a` is its own inverse mod 8, so 3 -> 6 -> ... -> 192 bits
    /// takes six iterations, enough for 128 bits.
    fn inv_odd_mod_pow2(&self, a: u128) -> u128 {
        let mut inv = a;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u128.wrapping_sub(a.wrapping_mul(inv)));
        }
        inv & self.mask
    }

    fn value_bytes(bits: u32) -> usize {
        bits.div_ceil(8) as usize
    }

    /// Layout: 1 byte for `bits`, then seed, step and offset, each
    /// little-endian in `ceil(bits / 8)` bytes. The counter is not stored.
    pub fn to_bytes(&self) -> Vec<u8> {
        let value_bytes = Self::value_bytes(self.bits);
        let mut output = Vec::with_capacity(NR_BITS_BYTES + 3 * value_bytes);

        output.push(self.bits as u8);
        for value in [self.seed, self.step, self.offset] {
            output.extend_from_slice(&value.to_le_bytes()[..value_bytes]);
        }

        output
    }

    /// Returns `None` if the bytes are too short or `bits` is out of range.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let bits = *bytes.first()? as u32;
        if !(1..=128).contains(&bits) {
            return None;
        }

        let value_bytes = Self::value_bytes(bits);
        let mut values = [0u128; 3];
        for (index, value) in values.iter_mut().enumerate() {
            let start = NR_BITS_BYTES + index * value_bytes;
            let slice = bytes.get(start..start + value_bytes)?;

            let mut buffer = [0u8; 16];
            buffer[..value_bytes].copy_from_slice(slice);
            *value = u128::from_le_bytes(buffer);
        }

        let [seed, step, offset] = values;
        Some(Self::new(bits, seed, step, offset))
    }
}

