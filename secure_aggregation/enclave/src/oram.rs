use mc_oblivious_ram::{PathORAM};
use mc_oblivious_traits::{rng_maker, HeapORAMStorageCreator, ORAM, ORAMCreator, ORAMStorageCreator, HeapORAMStorage};
use mc_oblivious_ram::U32PositionMapCreator;
use aligned_cmov::typenum::{U4, U64, U256};
use aligned_cmov::{A64Bytes, ArrayLength};
use rand_core::{CryptoRng, RngCore, SeedableRng};
use rand_hc::Hc128Rng;
use core::marker::PhantomData;
use std::vec::Vec;
use sgx_trts::trts::rsgx_read_rand;

use std::time::Instant;
use std::untrusted::time::InstantEx;

use crate::parameters::Weight;
use crate::common::{
    average_params,
};

type RngType = Hc128Rng;

pub struct PathORAM256Z4Creator<R, SC>
where
    R: RngCore + CryptoRng + 'static,
    SC: ORAMStorageCreator<U256, U64>,
{
    _rng: PhantomData<fn() -> R>,
    _sc: PhantomData<fn() -> SC>,
}

impl<R, SC> ORAMCreator<U64, R> for PathORAM256Z4Creator<R, SC>
where
    R: RngCore + CryptoRng + Send + Sync + 'static,
    SC: ORAMStorageCreator<U256, U64>,
 {
    type Output = PathORAM<U64, U4, SC::Output, R>;

    fn create<M: 'static + FnMut() -> R>(
        size: u64,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Self::Output {
        PathORAM::new::<U32PositionMapCreator<U64, R, Self>, SC, M>(size, stash_size, rng_maker)
    }
}

fn to_a64_bytes<N: ArrayLength<u8>>(src: f32) -> A64Bytes<N> {
    let src_bytes = src.to_be_bytes();
    let mut result = A64Bytes::<N>::default();
    for i in 0..4 {
        result[i] = src_bytes[i];
    }
    result
}

fn from_a64_bytes<N: ArrayLength<u8>>(a64_bytes: A64Bytes<N>) -> f32 {
    let mut f32_by_bytes = [0; 4];
    for i in 0..4 {
        f32_by_bytes[i] = a64_bytes[i]
    }
    f32::from_be_bytes(f32_by_bytes)
}

pub fn prepare(num_of_parameters: u64) -> PathORAM<U64, U4, HeapORAMStorage<U256, U64>, RngType> {
    let mut random_u8_arr: [u8; 32] = [0; 32];
    match rsgx_read_rand(&mut random_u8_arr[..]) {
        Ok(()) => {},
        Err(_e) => { panic!("rsgx_read_rand error") }
    }
    // let random_seed: [u8; 8] = random_u8_arr;
    let rng = RngType::from_seed(random_u8_arr);

    let stash_size = 20;
    let mut oram = PathORAM256Z4Creator::<RngType, HeapORAMStorageCreator>::create(
        num_of_parameters.next_power_of_two(),
        stash_size,
        &mut rng_maker(rng),
    );
    let zero_a64_bytes = to_a64_bytes(0.0);
    for i in 0..num_of_parameters {
        oram.write(i, &zero_a64_bytes);
    }
    return oram
}

pub fn path_oram_with_zerotrace(
    global_params: &mut [f32],
    uploaded_params: &mut Vec<Weight>,
    client_size: usize,
    verbose: bool,
) {
    let start = Instant::now();
    if verbose { println!("prepare"); }
    let mut oram = prepare(global_params.len() as u64);
    let end = start.elapsed();
    if verbose { println!("prepare done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }

    let start = Instant::now();
    if verbose { println!("aggregation"); }
    for w in uploaded_params.iter() {
        let idx = w.0 as u64;
        let mut current_val = from_a64_bytes(oram.read(idx));
        current_val += w.1 ;
        oram.write(idx, &to_a64_bytes(current_val));
    }
    let end = start.elapsed();
    if verbose { println!("aggregation done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }

    let start = Instant::now();
    if verbose { println!("take aggregation data"); }
    for i in 0..global_params.len() {
        global_params[i] = from_a64_bytes(oram.read(i as u64));
    }
    let end = start.elapsed();
    if verbose { println!("take aggregation data done:  {}.{:06} seconds", end.as_secs(), end.subsec_nanos() / 1_000); }

    average_params(global_params, client_size);
}
