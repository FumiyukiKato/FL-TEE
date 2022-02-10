use std::vec::Vec;

use crate::parameters::Weight;
use crate::common::average_params;
use crate::oblivious_primitives::o_mov;

pub fn baseline(
    global_params: &mut [f32],
    uploaded_params: &Vec<Weight>,
    client_size: usize,
) {
    let num_of_parameters = global_params.len() as isize;
    for w in uploaded_params.iter() {
        // cache-line optimized o_write
        o_update(
            &global_params[0],
            num_of_parameters,
            w.0 as isize,
            w.1,
        );
    }
    average_params(global_params, client_size);
}

// Adversary can observe cache line which is 64 bytes.
const CACHE_LINE_NUM_OF_WEIGHT: isize = 16;
#[inline]
fn o_update(dst_base: &f32, num_of_weights: isize, addr_offset: isize, val: f32) {
    // example:
    // number_of_weights = 100, dst_base = 1000, addr = 1080
    //  Memory => (Weight1 (dst_base=1000), Weight2, ..., Weight10, ""Weight11"" (addr=1080),..., Weight100 (1800))
    // ith = (1080 - 1000) % 64 = 16
    // the number of loops is 100 / 8 = 12.
    // rest array size is 100 % 8 = 4
    // cache_line_addr = 1000 + 64*0 = 1000

    unsafe {
        let dst_base = dst_base as *const f32;
        let addr = dst_base.offset(addr_offset) as *mut f32;
        let ith = addr_offset % CACHE_LINE_NUM_OF_WEIGHT;
        let mut last = 0;

        for i in 0..(num_of_weights / CACHE_LINE_NUM_OF_WEIGHT) {
            // Only one of the values to be read needs to be written.
            // Since it is not known from the outside which value was written
            let cache_line_addr = dst_base.offset(CACHE_LINE_NUM_OF_WEIGHT * i + ith) as *mut f32;
            let flag = (cache_line_addr == addr) as isize;
            *cache_line_addr = o_mov(flag, *cache_line_addr, *cache_line_addr + val);
            last += 1;
        }

        // Check all the rest one by one
        let cache_line_addr = dst_base.offset(CACHE_LINE_NUM_OF_WEIGHT * (last));
        for j in 0..num_of_weights % CACHE_LINE_NUM_OF_WEIGHT {
            let read_value_addr = cache_line_addr.offset(j) as *mut f32;
            let flag = (read_value_addr == addr) as isize;
            *read_value_addr = o_mov(flag, *read_value_addr, *read_value_addr + val);
        }
    }
}
