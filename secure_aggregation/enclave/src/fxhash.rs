use core::hash::Hasher;
use core::ops::BitXor;

// https://github.com/rust-lang/rustc-hash/blob/master/src/lib.rs
pub struct FxHasher {
    hash: usize,
}

const K: usize = 0x517cc1b727220a95;
const R: usize = 0;

impl Default for FxHasher {
    #[inline]
    fn default() -> FxHasher {
        FxHasher { hash: R }
    }
}

impl FxHasher {
    #[inline]
    pub fn add_to_hash(&mut self, i: usize) {
        self.hash = self.hash.rotate_left(5).bitxor(i).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut _bytes: &[u8]) {
        panic!("Not implemented");
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }
}