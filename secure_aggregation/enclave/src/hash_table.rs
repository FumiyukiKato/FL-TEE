use std::collections::HashMap;
use std::vec::Vec;

use crate::parameters::{Index, Value, Weight};

#[derive(Default, Debug)]
pub struct OHashMap {
    pub table: HashMap<Index, Value>
}

impl OHashMap {
    pub fn init(size: usize, max_dummy_size: usize) -> Self {
        // Rust HashMap has randomness, the randomness of memory allocation here depends on it.
        // https://doc.rust-lang.org/std/collections/struct.HashMap.html
        // But, we can replace alternatives using something based on rsgx_read_rand.
        let mut table: HashMap<Index, Value> = HashMap::with_capacity(size + max_dummy_size);
        for i in 0..(size as u32) {
            table.insert(i, 0.0);
        }
        for i in 0..(max_dummy_size as u32) {
            table.insert(u32::MAX - i, 0.0);
        }
        Self { table }
    }

    pub fn get_weights(&self) -> Vec<Weight> {
        self.table.iter().map(|(i, v)| Weight(*i, *v)).collect()
    }
}