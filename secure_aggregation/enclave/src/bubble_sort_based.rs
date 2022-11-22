use std::vec::Vec;
use std::mem;
use std::time::Instant;
use std::untrusted::time::InstantEx;

use crate::parameters::Weight;
use crate::common::{average_params};
use crate::oblivious_primitives::{o_swap, o_mov};


pub fn bubble_sort_based(
    num_of_sparse_parameters: usize,
    global_params: &mut [f32],
    uploaded_params: &mut Vec<Weight>,
    client_size: usize,
    verbose: bool,
) {
    let d = global_params.len();
    let k = num_of_sparse_parameters;
    let n = client_size;

    let start = Instant::now();
    if verbose { println!("insert_initial_data"); }
    insert_initial_data(uploaded_params, d);
    let end = start.elapsed();
    if verbose { println!("insert_initial_data done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }

    // O((nk+d)^2))
    let start = Instant::now();
    if verbose { println!("oblivious_sort_idx"); }
    oblivious_sort_idx(uploaded_params);
    let end = start.elapsed();
    if verbose { println!("oblivious_sort_idx done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }

    // Oblivious folding O(nk)
    let start = Instant::now();
    if verbose { println!("oblivious folding"); }
    let mut pre_idx = uploaded_params[0].0;
    let mut pre_val = uploaded_params[0].1;
    let mut dummy_idx = u32::MAX;

    let initialized_parameter_length = n * k + d;

    for i in 1..initialized_parameter_length {
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

    uploaded_params[initialized_parameter_length - 1].0 = pre_idx;
    uploaded_params[initialized_parameter_length - 1].1 = pre_val;

    let end = start.elapsed();
    if verbose { println!("oblivious folding done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }

    // O((nk+d)^2))
    let start = Instant::now();
    if verbose { println!("second oblivious_sort_idx"); }
    oblivious_sort_idx(uploaded_params);
    let end = start.elapsed();
    if verbose { println!("second oblivious_sort_idx done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }
    
    for i in 0..d {
        global_params[i] = uploaded_params[i].1;
    }
    
    average_params(global_params, n);
}


fn insert_initial_data(
    uploaded_params: &mut Vec<Weight>,
    d: usize,
) {
    let d = d as u32;
    let initial_data = (0..d).map(|idx| Weight(idx, 0.0));
    uploaded_params.extend(initial_data);
}

#[inline]
fn oblivious_sort_idx(source: &mut Vec<Weight>) {
    o_bubble_sort_by_idx(source);
}


// By bubble sort with oblivious primtives using inline assembler
#[inline]
fn o_bubble_sort_by_idx(source: &mut [Weight]) {
    for i in 0..source.len() {
        for j in 0..source.len() - 1 - i {
            let cond = source[j].0 > source[j + 1].0;
            // Note: oswap swaps 8byte once.
            // it means that parameters::Weight (= 4byte Index + 4byte Value) is swapped together.
            o_swap(cond as isize, &source[j].0, &source[j+1].0);
        }
    }
}
