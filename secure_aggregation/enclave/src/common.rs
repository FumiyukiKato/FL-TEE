use std::vec::Vec;
use sgx_rand::{Rng, StdRng, SeedableRng, sample};
// use sgx_rand::distributions::Exp;
use sgx_rand::distributions::{Normal, IndependentSample};
use sgx_trts::trts::rsgx_read_rand;

use crate::oblivious_primitives::{o_mov, o_setb};


use crate::parameters::Weight;


#[inline]
pub fn average_params(global_params: &mut [f32], client_size: usize) {
    let average_coefficient = 1f32 / client_size as f32;
    for i in 0..global_params.len() {
        global_params[i] *= average_coefficient;
    }
}

// To avoid to access with dummy data index,
// dummy data is randomly scattered and adversary can only see
// the random accesses which is perfectly independent of the secret data
#[inline]
pub fn safe_aggregate(
    global_params: &mut [f32],
    uploaded_params: &mut Vec<Weight>,
    client_size: usize,
    k: u32,
) {
    for w in uploaded_params.iter() {
        if w.0 < k { global_params[w.0 as usize] += w.1; }
    }
    average_params(global_params, client_size);
}

// SGX provides safe random number generator sgx_read_rand()
// c.f.
// https://download.01.org/intel-sgx/linux-1.8/docs/Intel_SGX_SDK_Developer_Reference_Linux_1.8_Open_Source.pdf
// https://en.wikipedia.org/wiki/RDRAND

#[inline]
pub fn get_safe_random_seed() -> StdRng {
    let mut random_u8_arr: [u8; 8] = [0; 8];
    match rsgx_read_rand(&mut random_u8_arr[..]) {
        Ok(()) => {},
        Err(_e) => { panic!("rsgx_read_rand error") }
    }
    let random_seed: &[usize] = &[usize::from_be_bytes(random_u8_arr)];
    let rng: StdRng = SeedableRng::from_seed(random_seed);
    rng
}


#[inline]
pub fn rdp_gaussian_mechanism(global_params: &mut [f32], sigma: f32, clipping: f32, alpha: f32, client_size: usize) {
    // Fixed seed
    // let random_seed: &[_] = &[0, 0, 0, 0];
    // let mut rng: StdRng = SeedableRng::from_seed(random_seed);
    // Random 
    let mut rng: StdRng = get_safe_random_seed();

    let n = client_size as f64;
    
    // We are not sure to use k/d sensitivity in the case of top-k sparsification
    // let normal = Normal::new(0.0, (clipping * alpha * sigma) as f64);
    let normal = Normal::new(0.0, (clipping * sigma) as f64);
    for i in 0..global_params.len() {
        let gauss_noise = normal.ind_sample(&mut rng);
        global_params[i] += (gauss_noise / n) as f32;
    }
}


// c.f. https://docs.rs/probability/0.17.0/src/probability/distribution/laplace.rs.html#7-10
#[inline]
pub fn laplace_noise_vec(d: usize, k: usize, epsilon: f32, delta: f32, rng: &mut StdRng) -> (Vec<f32>, f32) {
    let l1_sensitivity = 2.0 * k as f32;
    println!("l1_sensitivity {}", l1_sensitivity);
    // Note: when acheivable delta = 1/n^2, cut_off = 1/ep * (4*log(n)+2*log(d))
    let cut_off_threshold = l1_sensitivity / epsilon * ((d as f32 / delta).ln());
    println!("(d as f32 / delta).ln() {}", (d as f32 / delta).ln());
    println!("cut_off_threshold {}", cut_off_threshold);
    let mut noise_vec = Vec::with_capacity(d);
    let b = l1_sensitivity / epsilon;

    for _ in 0..d {
        let p: f32 = rng.gen::<f32>();
        if !(0.0 <= p && p <= 1.0) { panic!("RNG is invalid."); }
        let noise: f32 = if p > 0.5 {
            - b * (2.0 - 2.0 * p).ln()
        } else {
            b * (2.0 * p).ln()
        };
        noise_vec.push(noise as f32);
    }
    (noise_vec, cut_off_threshold)
}

#[inline]
pub fn sample_client_ids(client_ids_vec: &Vec<u32>, size: usize) -> Vec<u32> {
    let mut rng: StdRng = get_safe_random_seed();
    let sample = sample(&mut rng, client_ids_vec.iter(), size);
    return sample.into_iter().map(|x| *x).collect()
}


// #[inline]
// pub fn exponential_noise_vec(d: usize, k: usize, epsilon: f32, delta: f32, rng: &mut StdRng) -> (Vec<f32>, f32) {
//     let l1_sensitivity = 2.0 * k as f32;
//     println!("l1_sensitivity {}", l1_sensitivity);
//     let cut_off_threshold = l1_sensitivity / epsilon * ((d as f32 / delta).ln());
//     println!("(d as f32 / delta).ln() {}", (d as f32 / delta).ln());
//     println!("cut_off_threshold {}", cut_off_threshold);
//     let mut noise_vec = Vec::with_capacity(d);
//     let lamb = l1_sensitivity / epsilon;
//     let exp = Exp::new(lamb as f64);
//     for _ in 0..d {
//         noise_vec.push(exp.ind_sample(rng) as f32);
//     }
//     (noise_vec, cut_off_threshold)
// }

// #[inline]
// pub fn geometric_noise_vec(d: usize, k: usize, epsilon: f32, delta: f32, rng: &mut StdRng) -> (Vec<f32>, f32) {
//     let l1_sensitivity = 2.0 * k as f32;
//     let cut_off_threshold = l1_sensitivity / epsilon * ((d as f32 / delta).ln()) - 1.0;
//     let mut noise_vec = Vec::with_capacity(d);
//     let p = 1.0 - (- epsilon / l1_sensitivity).exp();
//     for _ in 0..d {
//         let noise = if p == 1.0 {
//             0.0
//         } else {
//             (1.0 - rng.gen::<f32>()).log(1.0 - p).ceil() - 1.0
//         };
//         noise_vec.push(noise);
//     }
//     (noise_vec, cut_off_threshold)
// }


// // https://github.com/IBM/discrete-gaussian-differential-privacy
// #[inline]
// pub fn discrete_gauss_noise_vec(d: usize, l2_sensitivity: f32, epsilon: f32, delta: f32, rng: &mut StdRng) -> Vec<f32> {
//     let mut noise_vec = Vec::with_capacity(d);
//     noise_vec
//     // TODO;
// }

#[inline]
pub fn transform_to_random_int_vec(noise_vec: &Vec<f32>, cut_off_threshold: f32) -> Vec<u32> {
    let mut random_int_vec = Vec::with_capacity(noise_vec.len());
    for x_i in noise_vec.iter() {
        if x_i.abs() > cut_off_threshold {
            random_int_vec.push(cut_off_threshold.ceil() as u32);
        } else {
            random_int_vec.push((cut_off_threshold + x_i.ceil()) as u32);
        }
    }
    random_int_vec
}

#[inline]
pub fn pad_weight_to_power_of_two(source: &mut Vec<Weight>) -> usize {
    let source_size = source.len();
    let size = source_size.next_power_of_two();
    // padding (size - source_size) dummies which is (Index=u32::MAX, Value=0.0)
    // padding index is irrelevant to the specific parameters
    let pads = vec![Weight(u32::MAX, 0.0); size - source_size];
    source.extend(pads);
    return size - source_size;
}


// idx=k+1のダミーデータの数はヒストグラムが公開される際に明らかになってしまう問題
// idx=k+1 の数だけ明らかになってしまうが，これはノイズの合計値を明らかにすることに対応する．
// kが十分に大きい(>>2)場合は個々のノイズを復元することができないので問題ないと考えて良い？
// その場合，最大値を固定する意味はない．最大値を固定する目的が不明
// CCS '18 (https://eprint.iacr.org/2017/1016.pdf) footprint.6 says
// "Revealing these blank edges before shuffling would reveal 
// how many dummy edges there are of the form (∗, i), which would break privacy. 
// After all the edges are shuffled, revealing the number of blank edges
// only reveals the total number of dummy edges, which is fine."
// つまり，個々のダミーの値は隠す必要があるけど，合計値は隠さないといけないと言っている．
// そのため，paddingのプロセスがobliviousになっていないとそもそも意味がない
// NIPSやCCSではそこについて何も行っていない気がする
// oblivious shuffleの前にダミーデータ削除したい感じはあるけど，そもそもshuffleしないとアドレスが決まっているのでダメか
#[inline]
pub fn oblivious_pad(source: &mut Vec<Weight>, rand_int_vec: &Vec<u32>, d: usize, rand_max: f32) {
    for i in 0..d {
        let r_i = rand_int_vec[i] as usize;
        for j in 0..rand_max as usize {
            let flag = o_setb(r_i, j);
            source.push(Weight(o_mov(flag as isize, u32::MAX, i as u32), 0.0));
        }        
    }
}
