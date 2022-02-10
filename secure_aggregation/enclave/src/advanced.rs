use std::vec::Vec;
use std::mem;
use std::time::Instant;
use std::untrusted::time::InstantEx;

use crate::hash_table::OHashMap;
use crate::parameters::Weight;
use crate::common::{average_params};
use crate::oblivious_primitives::{o_swap, o_mov};


pub fn advanced(
    num_of_sparse_parameters: usize,
    global_params: &mut [f32],
    uploaded_params: &mut Vec<Weight>,
    client_size: usize,
    verbose: bool,
) {
    let k = num_of_sparse_parameters;
    let n = client_size;

    // bottle neck O(nk(log(nk))^2))  n=10e2 k=10e5 nk=10e7 (log(nk))^2 = (7*log(10))^2 ~ 500
    let start = Instant::now();
    if verbose { println!("oblivious_sort_idx"); }
    oblivious_sort_idx(uploaded_params);
    let end = start.elapsed();
    if verbose { println!("oblivious_sort_idx done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }

    // O(nk)
    let start = Instant::now();
    if verbose { println!("oblivious aggregation"); }
    let mut pre_idx = uploaded_params[0].0;
    let mut pre_val = uploaded_params[0].1;
    let mut dummy_idx = u32::MAX;

    for i in 1..n * k {
        unsafe {
            let update = o_mov(
                (pre_idx == uploaded_params[i].0) as isize,
                mem::transmute::<(u32, f32), u64>((pre_idx, pre_val)),
                mem::transmute::<(u32, f32), u64>((dummy_idx, 0.0)),
            );
            let (update_idx, update_val) = mem::transmute::<u64, (u32, f32)>(update);
            uploaded_params[i - 1].0 = update_idx;
            uploaded_params[i - 1].1 = update_val;
        }
        unsafe {
            let pre_update = o_mov(
                (pre_idx == uploaded_params[i].0) as isize,
                mem::transmute::<(u32, f32), u64>((
                    uploaded_params[i].0,
                    uploaded_params[i].1,
                )),
                mem::transmute::<(u32, f32), u64>((pre_idx, pre_val + uploaded_params[i].1)),
            );
            let (pre_update_idx, pre_update_val) =
                mem::transmute::<u64, (u32, f32)>(pre_update);
            pre_idx = pre_update_idx;
            pre_val = pre_update_val;
        }
        dummy_idx -= 1;
    }

    uploaded_params[n * k - 1].0 = pre_idx;
    uploaded_params[n * k - 1].1 = pre_val;

    let end = start.elapsed();
    if verbose { println!("oblivious aggregation done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }

    // bottle neck O(nk(log(nk)^2))
    let start = Instant::now();
    if verbose { println!("oblivious_sort_abs_val_and_idx"); }
    oblivious_sort_abs_val_and_idx(uploaded_params);
    let end = start.elapsed();
    if verbose { println!("oblivious_sort_abs_val_and_idx done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }

    // O(d(log(d)^2))
    let start = Instant::now();
    if verbose { println!("set_valued_memory_access_leak_aggregation"); }
    set_valued_memory_access_leak_aggregation(
        uploaded_params,
        global_params
    );
    let end = start.elapsed();
    if verbose { println!("set_valued_memory_access_leak_aggregation done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }
    average_params(global_params, n);
}


fn set_valued_memory_access_leak_aggregation(
    uploaded_params: &[Weight],
    global_params: &mut [f32],
) {
    let d = global_params.len();
    // (0,0),...,(d-1, 0), with d dummies (M_0, 0), (M_0-1, 0),...
    let mut hash_map = OHashMap::init(d, d);
    let min = if d < uploaded_params.len() { d } else { uploaded_params.len() };
    for i in 0..min {
        hash_map
            .table
            .insert(uploaded_params[i].0, uploaded_params[i].1);
    }
    let mut random_global_params = hash_map.get_weights();
    oblivious_sort_idx(&mut random_global_params);
    for i in 0..global_params.len() {
        global_params[i] = random_global_params[i].1;
    }
}


#[inline]
fn oblivious_sort_idx(source: &mut Vec<Weight>) {
    let number_of_pads = pad_max_idx_weight_to_power_of_two(source);
    o_bitonic_sort_by_idx(source);
    source.truncate(source.len() - number_of_pads);
}


#[inline]
fn oblivious_sort_abs_val_and_idx(source: &mut Vec<Weight>) {
    let number_of_pads = pad_min_idx_weight_to_power_of_two(source);
    o_bitonic_sort_by_abs_val_and_idx(source);
    source.truncate(source.len() - number_of_pads);
}


#[inline]
fn pad_max_idx_weight_to_power_of_two(source: &mut Vec<Weight>) -> usize {
    let source_size = source.len();
    let size = source_size.next_power_of_two();
    // padding (size - source_size) dummies which is (Index=u32::MAX, Value=0.0)
    // padding index is irrelevant to the specific parameters
    let pads = vec![Weight(u32::MAX, 0.0); size - source_size];
    source.extend(pads);
    return size - source_size;
}


#[inline]
fn pad_min_idx_weight_to_power_of_two(source: &mut Vec<Weight>) -> usize {
    let source_size = source.len();
    let size = source_size.next_power_of_two();
    let pads = vec![Weight(u32::MIN, 0.0); size - source_size];
    source.extend(pads);
    return size - source_size;
}


// By bitonic sort with oblivious primtives using inline assembler
#[inline]
fn o_bitonic_sort_by_idx(source: &mut [Weight]) {
    let size = source.len();
    if size != size.next_power_of_two() {
        panic!("source length is invalid")
    }

    let half_size = size >> 1;
    let mut i = 2;
    while i <= size {
        let mut j = i >> 1;
        while j > 0 {
            let ml = j - 1;
            let mh = !ml;

            for k in 0..half_size {
                let l = ((k & mh) << 1) | (k & ml);
                let m = l + j;

                let cond1 = (l & i) == 0;
                let cond2 = source[l].0 < source[m].0;

                // Note: oswap swaps 8byte once.
                // it means that parameters::Weight (= 4byte Index + 4byte Value) is swapped together.
                o_swap((cond1 ^ cond2) as isize, &source[l].0, &source[m].0);
            }
            j >>= 1;
        }
        i <<= 1;
    }
}


#[inline]
fn o_bitonic_sort_by_abs_val_and_idx(source: &mut [Weight]) {
    let size = source.len();
    if size != size.next_power_of_two() {
        panic!("source length is invalid")
    }

    let half_size = size >> 1;
    let mut i = 2;
    while i <= size {
        let mut j = i >> 1;
        while j > 0 {
            let ml = j - 1;
            let mh = !ml;

            for k in 0..half_size {
                let l = ((k & mh) << 1) | (k & ml);
                let m = l + j;

                let cond1 = (l & i) == 0;
                let cond2 = source[l].1.abs() > source[m].1.abs()
                    || (source[l].1 == source[m].1 && source[l].0 > source[m].0);

                // Note: oswap swaps 8byte once.
                // it means that parameters::Weight (= 4byte Index + 4byte Value) is swapped together.
                o_swap((cond1 ^ cond2) as isize, &source[l].0, &source[m].0);
            }
            j >>= 1;
        }
        i <<= 1;
    }
}
