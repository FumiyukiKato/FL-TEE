use std::vec::Vec;
use core::hash::Hasher;
use sgx_rand::{StdRng};
// use sgx_rand::{SeedableRng};

use crate::fxhash::FxHasher;
use crate::common::{
    pad_weight_to_power_of_two,
    laplace_noise_vec,
    get_safe_random_seed,
    safe_aggregate,
    transform_to_random_int_vec,
    oblivious_pad
};
use crate::parameters::Weight;
use crate::oblivious_primitives::{o_equal, o_setb, o_swap};

pub fn nips19(
    num_of_sparse_parameters: usize,
    global_params: &mut [f32],
    uploaded_params: &mut Vec<Weight>,
    client_size: usize,
) {
    // using very big epsilon, delta
    let epsilon = 100.0;
    let delta = 1.0 / client_size as f32;

    let d = global_params.len();
    let k = num_of_sparse_parameters;

    // Fixed seed
    // let random_seed: &[_] = &[0, 0, 0, 0];
    // let mut rng: StdRng = SeedableRng::from_seed(random_seed);
    // Random 
    let mut rng: StdRng = get_safe_random_seed();

    // using oblivious padding
    let (noise_vec, cut_off_threshold) = laplace_noise_vec(d, k, epsilon, delta, &mut rng);
    let random_int_vec: Vec<u32> = transform_to_random_int_vec(&noise_vec, cut_off_threshold);
    oblivious_pad(uploaded_params, &random_int_vec, d, cut_off_threshold);

    // original NIPS'19 (but it's non-oblivious pading)
    // for x_i in noise_vec.iter() {
    //     if *x_i.abs() > cut_off_threshold {
    //         *x_i = 0.0;
    //     } else {
    //         *x_i = cut_off_threshold + *x_i.ceil();
    //     }
    //     num_of_dummies -= *x_i as usize;
    //     // Creating fake records (This is not olbivious...?)
    //     for i in 0..*x_i as usize {
    //         uploaded_params.push(Weight(i as u32, 0.0));
    //     }
    // }
    // // The purpose of this step is to ensure that the length of the output is exactly T = n + 20k*log(n/ep).
    // for _ in 0..num_of_dummies {
    //     uploaded_params.push(Weight((k+1) as u32, 0.0));
    // }

    pad_weight_to_power_of_two(uploaded_params);
    o_shuffle(uploaded_params);
    safe_aggregate(global_params, uploaded_params, client_size, d as u32);
}


fn o_shuffle(source: &mut [Weight]) {
    let size = source.len();
    if size != size.next_power_of_two() {
        panic!("source length is invalid")
    }
    let seed = 100;
    let mut h = FxHasher::default();
    h.add_to_hash(seed);

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

                let cond1 = o_equal((l & i) as usize, 0 as usize);
                let h1 = unsafe { 
                    h.add_to_hash(std::mem::transmute::<&Weight, usize>(&source[l])); 
                    h.finish()};
                let h2 = unsafe { 
                    h.add_to_hash(std::mem::transmute::<&Weight, usize>(&source[m]));
                    h.finish()};
                let cond2 =
                    o_setb(h1 as usize, h2 as usize);

                // Note: oswap swaps 8byte once.
                // it means that parameters::Weight (= 4byte Index + 4byte Value) is swapped together.
                o_swap((cond1 ^ cond2) as isize, &source[l].0, &source[m].0);
            }
            j >>= 1;
        }
        i <<= 1;
    }
}