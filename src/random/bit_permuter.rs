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
//!
//! What it is for: handing out identifiers, or visiting every cell of a
//! region, in an order that looks unrelated to the order they were asked for,
//! without storing a shuffled list. A shuffle of `2^40` values needs `2^40`
//! entries of memory; this needs a few `u128`s, and can start anywhere in the
//! cycle.
//!
//! The permutation is a keyed round function in the style of a small block
//! cipher: odd multiplies, additions and right xor-shifts, each of which is
//! invertible modulo `2^bits`, so the whole chain is too. It scrambles well
//! enough to look unordered and is not meant to withstand an adversary who
//! wants to recover the key.

use crate::random::mixing::mix128;
use crate::random::seed::Seed;

const NR_BITS_BYTES: usize = 1;

/// A counter that walks every value of a bit width exactly once, with the
/// output scrambled by an invertible keyed permutation.
///
/// The counter advances by an odd step, which visits all `2^bits` residues
/// before repeating, and the permutation turns each counter value into the
/// output. Both halves are invertible, so
/// [`index_of_output`](BitPermuter::index_of_output) recovers where a value
/// sits in the cycle and [`peek_value_at`](BitPermuter::peek_value_at) reads
/// any position without advancing.
///
/// The multipliers, addends and shifts are derived from the seed once at
/// construction, along with their modular inverses, so every later call is a
/// handful of multiplies.
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
    /// A permuter over `bits` bits, keyed by `seed`, walking the counter by
    /// `step` and permuting from `offset`.
    ///
    /// `step` is forced odd, since only an odd step visits every value before
    /// repeating, and `seed`, `step` and `offset` are all masked to the width.
    /// Panics unless `bits` is in `1..=128`.
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

    /// A permuter over `bits` bits with step 1 and offset 0, which is the
    /// plain case: the counter goes up by one and the permutation does the
    /// scrambling.
    pub fn with_seed(bits: u32, seed: u128) -> Self {
        Self::new(bits, seed, 1, 0)
    }

    /// A permuter keyed from a derived [`Seed`].
    ///
    /// The key here is not a world seed and is not stored as one: it is masked
    /// to `bits` and xor-ed into values, so most of a 128-bit seed is thrown
    /// away by a narrow permuter. This takes a `Seed` for convenience at the
    /// call site, and what survives is whatever fits.
    pub fn from_seed(bits: u32, seed: Seed) -> Self {
        Self::with_seed(bits, seed.value())
    }

    /// The width, in bits: the cycle covers `0..2^bits`.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// The key, masked to the width.
    pub fn seed(&self) -> u128 {
        self.seed
    }

    /// The counter's stride, always odd.
    pub fn step(&self) -> u128 {
        self.step
    }

    /// What the counter is shifted by before being permuted.
    pub fn offset(&self) -> u128 {
        self.offset
    }

    /// How many values [`next_value`](BitPermuter::next_value) has produced,
    /// modulo `2^bits`.
    ///
    /// Recovered from the counter by multiplying by the step's modular
    /// inverse, since the counter itself has been multiplied by the step.
    pub fn position(&self) -> u128 {
        self.counter.wrapping_mul(self.inv_step) & self.mask
    }

    /// Moves the counter so that the next call to
    /// [`next_value`](BitPermuter::next_value) returns the value at `index`,
    /// counting from zero.
    ///
    /// Any index is valid; it wraps modulo `2^bits`.
    pub fn set_position(&mut self, index: u128) {
        self.counter = index.wrapping_mul(self.step) & self.mask;
    }

    /// The low `bits` bits set. Written out rather than `(1 << bits) - 1`
    /// because that shift overflows at a width of 128.
    fn mask_for(bits: u32) -> u128 {
        if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        }
    }

    /// Derives the round constants from the seed and inverts each of them.
    ///
    /// Three odd multipliers and two addends come from mixing the seed with
    /// different salts; the shifts are fractions of the width, floored at one
    /// so that a narrow permuter still shifts. Odd numbers are invertible
    /// modulo a power of two, which is why the multipliers are forced odd.
    fn calculate_values(&mut self) {
        self.mask = Self::mask_for(self.bits);

        self.mul1 = self.odd_mult(0xA1);
        self.mul2 = self.odd_mult(0xB7);
        self.mul3 = self.odd_mult(0xC3);

        self.add1 = self.add_const(0x11);
        self.add2 = self.add_const(0x29);

        self.shift1 = (self.bits / 2).max(1);
        self.shift2 = (self.bits / 3).max(1);

        self.inv_mul1 = self.inv_odd_mod_pow2(self.mul1);
        self.inv_mul2 = self.inv_odd_mod_pow2(self.mul2);
        self.inv_mul3 = self.inv_odd_mod_pow2(self.mul3);
        self.inv_step = self.inv_odd_mod_pow2(self.step);
    }

    /// The next value of the cycle, advancing the counter by one step.
    ///
    /// Every value in `0..2^bits` comes out exactly once per cycle.
    pub fn next_value(&mut self) -> u128 {
        self.counter = self.counter.wrapping_add(self.step) & self.mask;
        self.permute_bits(self.counter.wrapping_add(self.offset))
    }

    /// The value the call at `index`, counting from zero, would produce,
    /// without touching the counter.
    ///
    /// Index `2^bits - 1` is the last value of the cycle; larger indices wrap
    /// around.
    pub fn peek_value_at(&self, index: u128) -> u128 {
        let n = index.wrapping_add(1) & self.mask;
        let counter = n.wrapping_mul(self.step) & self.mask;
        self.permute_bits(counter.wrapping_add(self.offset))
    }

    /// The inverse of [`peek_value_at`](BitPermuter::peek_value_at): where in
    /// the cycle, counting from zero, this value is produced.
    ///
    /// Undoes the permutation, the offset and the step in turn, then steps back
    /// by one because [`next_value`](BitPermuter::next_value) advances the
    /// counter before producing its output.
    pub fn index_of_output(&self, output_value: u128) -> u128 {
        let permuted = self.invert_permuted(output_value);
        let counter = permuted.wrapping_sub(self.offset) & self.mask;
        let n = counter.wrapping_mul(self.inv_step) & self.mask;

        n.wrapping_sub(1) & self.mask
    }

    /// The permutation itself: a bijection of `0..2^bits` onto itself.
    ///
    /// The value is xor-ed with the key and then put through three rounds of
    /// odd multiply, add and right xor-shift. Each step is invertible modulo
    /// `2^bits`, and the multiply spreads low bits upwards while the xor-shift
    /// spreads high bits down, so a run of neighbouring inputs comes out
    /// scattered.
    ///
    /// Available on its own for permuting a value that did not come from the
    /// counter, such as an index into a region.
    pub fn permute_bits(&self, value: u128) -> u128 {
        let mask = self.mask;

        let mut x = (value ^ self.seed) & mask;

        x = x.wrapping_mul(self.mul1).wrapping_add(self.add1) & mask;
        x ^= x >> self.shift1;
        x = x.wrapping_mul(self.mul2).wrapping_add(self.add2) & mask;
        x ^= x >> self.shift2;
        x = x.wrapping_mul(self.mul3) & mask;
        x ^= x >> self.shift1;

        x
    }

    /// The inverse of [`permute_bits`](BitPermuter::permute_bits): undoes each
    /// round in reverse, using the cached modular inverses.
    pub fn invert_permuted(&self, value: u128) -> u128 {
        let mask = self.mask;
        let mut x = value & mask;

        x = self.unxorshift_right(x, self.shift1);
        x = x.wrapping_mul(self.inv_mul3) & mask;

        x = self.unxorshift_right(x, self.shift2);
        x = x.wrapping_sub(self.add2) & mask;
        x = x.wrapping_mul(self.inv_mul2) & mask;

        x = self.unxorshift_right(x, self.shift1);
        x = x.wrapping_sub(self.add1) & mask;
        x = x.wrapping_mul(self.inv_mul1) & mask;

        (x ^ self.seed) & mask
    }

    /// An odd multiplier derived from the seed and a salt, masked to the
    /// width. Odd, so that it is invertible modulo `2^bits`.
    fn odd_mult(&self, salt: u128) -> u128 {
        (mix128(self.seed ^ salt) | 1) & self.mask
    }

    /// An addend derived from the seed and a salt, masked to the width. Any
    /// value will do, since addition is always invertible.
    fn add_const(&self, salt: u128) -> u128 {
        mix128(self.seed.wrapping_add(salt)) & self.mask
    }

    /// Inverts `x ^= x >> shift`.
    ///
    /// One application leaves the top `shift` bits untouched, so xor-ing the
    /// result by itself shifted right recovers twice as many bits each time;
    /// doubling the shift until it passes the width undoes it entirely.
    fn unxorshift_right(&self, value: u128, shift: u32) -> u128 {
        let mut x = value & self.mask;
        let mut s = shift;
        while s < self.bits {
            x ^= x >> s;
            s <<= 1;
        }
        x
    }

    /// The multiplicative inverse of an odd number modulo `2^bits`, by Newton
    /// iteration: each step doubles the number of correct bits.
    ///
    /// An odd `a` is already its own inverse modulo 8, so three correct bits
    /// double to 6, 12, 24, 48, 96 and 192, and six iterations cover any width
    /// up to 128.
    fn inv_odd_mod_pow2(&self, a: u128) -> u128 {
        let mut inv = a;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u128.wrapping_sub(a.wrapping_mul(inv)));
        }
        inv & self.mask
    }

    /// How many bytes one value of this width occupies when written down.
    fn value_bytes(bits: u32) -> usize {
        bits.div_ceil(8) as usize
    }

    /// The permuter's parameters as bytes, for saving.
    ///
    /// Layout: one byte for `bits`, then seed, step and offset, each
    /// little-endian in `ceil(bits / 8)` bytes. The counter is not stored, so a
    /// restored permuter starts at the beginning of the cycle; save
    /// [`position`](BitPermuter::position) alongside if that matters.
    pub fn to_bytes(&self) -> Vec<u8> {
        let value_bytes = Self::value_bytes(self.bits);
        let mut output = Vec::with_capacity(NR_BITS_BYTES + 3 * value_bytes);

        output.push(self.bits as u8);
        for value in [self.seed, self.step, self.offset] {
            output.extend_from_slice(&value.to_le_bytes()[..value_bytes]);
        }

        output
    }

    /// A permuter read back from [`to_bytes`](BitPermuter::to_bytes), or
    /// `None` if the bytes are too short or the width is out of range.
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
