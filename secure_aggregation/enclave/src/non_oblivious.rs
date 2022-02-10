use std::vec::Vec;

use crate::parameters::Weight;
use crate::common::average_params;

pub fn non_oblivious(
    global_params: &mut [f32],
    uploaded_params: &Vec<Weight>,
    client_size: usize,
) {
    for w in uploaded_params.iter() {
        global_params[w.0 as usize] += w.1;
    }
    average_params(global_params, client_size);
}
