// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License..

#![crate_name = "secure_agg"]
#![crate_type = "staticlib"]
#![cfg_attr(not(target_env = "sgx"), no_std)]
#![cfg_attr(target_env = "sgx", feature(rustc_private))]
#![feature(llvm_asm)]
#![feature(core_intrinsics)]

extern crate sgx_trts;
extern crate sgx_types;
#[cfg(not(target_env = "sgx"))]
#[macro_use]
extern crate sgx_tstd as std;
extern crate sgx_rand;
extern crate sgx_tcrypto;

use sgx_tcrypto::*;
use sgx_types::*;
use std::slice;
use std::time::Instant;
use std::untrusted::time::InstantEx;
use std::vec::Vec;
// use std::boxed::Box;
// use std::cell::RefCell;
// use std::sync::atomic::{AtomicPtr, Ordering};

mod parameters;
use parameters::{Parameters, WEIGHT_BYTE_SIZE};

mod common;
use common::rdp_gaussian_mechanism;

mod fxhash;
mod oblivious_primitives;

mod nips19;
use nips19::nips19;

mod advanced;
use advanced::advanced;

mod advanced_simple;
use advanced_simple::advanced_simple;

mod baseline;
use baseline::baseline;

mod non_oblivious;
use non_oblivious::non_oblivious;

mod oram;
use oram::{path_oram_with_zerotrace};

mod hash_table;

// for secure channel encryption
pub const COUNTER_BLOCK: [u8; 16] = [0; 16];
pub const SGXSSL_CTR_BITS: u32 = 128;

// pub static O_HT_BUFFER: AtomicPtr<()> = AtomicPtr::new(0 as * mut ());
// pub fn get_ref_o_ht_buffer() -> Option<&'static RefCell<OHashMap>> {
//     let ptr = O_HT_BUFFER.load(Ordering::SeqCst) as * mut RefCell<OHashMap>;
//     if ptr.is_null() {
//         None
//     } else {
//         Some(unsafe { &* ptr })
//     }
// }

// fn init_o_hash(size: usize) {
//     let o_hash_map = OHashMap::init(size);
//     let o_hash_map_box = Box::new(RefCell::<OHashMap>::new(o_hash_map));
//     let o_hash_map_ptr = Box::into_raw(o_hash_map_box);
//     O_HT_BUFFER.store(o_hash_map_ptr as *mut (), Ordering::SeqCst);
// }

/// Secure aggregation
#[no_mangle]
pub extern "C" fn ecall_secure_aggregation(
    encrypted_parameters_data: *const u8,
    encrypted_parameters_size: usize,
    num_of_parameters: usize,
    num_of_sparse_parameters: usize,
    client_ids: *const u32,
    client_size: usize,
    sigma: f32,
    clipping: f32,
    alpha: f32,
    aggregation_alg: u32,
    updated_parameters_data: *mut f32,
    execution_time_results: *mut f32,
    verbose: u8,
    dp: u8,
) -> sgx_status_t {
    let verbose = match verbose { 0 => false, 1 => true, _ => true };
    let dp = match dp { 0 => false, 1 => true, _ => true };
    // initialize parameter buffer
    let start = Instant::now();

    // store global parameters as local variable
    let aggregated_parameters: &mut [f32] =
        unsafe { slice::from_raw_parts_mut(updated_parameters_data, num_of_parameters) };

    // read uploaded parameters
    let encrypted_parameters_vec: Vec<u8> =
        unsafe { slice::from_raw_parts(encrypted_parameters_data, encrypted_parameters_size) }
            .to_vec();
    if encrypted_parameters_vec.len() != encrypted_parameters_size {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    let client_ids_vec: Vec<u32> =
        unsafe { slice::from_raw_parts(client_ids, client_size) }.to_vec();
    if client_ids_vec.len() != client_size {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let end = start.elapsed();
    unsafe { *execution_time_results.offset(0) = end.as_secs_f32() };
    if verbose { println!("[SGX CLOCK] {}:  {}.{:06} seconds", "Loading", end.as_secs(), end.subsec_nanos() / 1_000); }

    // decryption
    let start = Instant::now();
    let byte_size_per_client = encrypted_parameters_vec.len() / client_size;
    let given_num_of_sparse_parameters = byte_size_per_client / WEIGHT_BYTE_SIZE;
    let mut all_uploaded_parameters: Parameters = Parameters::new(given_num_of_sparse_parameters * client_size);
    let mut decrypted_parameters_vec: Vec<u8> = vec![0; byte_size_per_client];
    for (i, client_id) in client_ids_vec.iter().enumerate() {
        let mut counter_block: [u8; 16] = COUNTER_BLOCK;
        let ctr_inc_bits: u32 = SGXSSL_CTR_BITS;

        // Originally shared_key is derived by following Remote Attestation protocol.
        // This is mock of shared key-based encryption.
        // The 128 bit key is [(client_id (64bit))0...0].
        let mut shared_key: [u8; 16] = [0; 16];
        shared_key[4..8].copy_from_slice(&client_id.to_be_bytes());
        let current_cursor = i * (byte_size_per_client); // num_of_parameters * (index: 8 bytes, value: 8bytes)
        let ret = rsgx_aes_ctr_decrypt(
            &shared_key,
            &encrypted_parameters_vec[current_cursor..current_cursor + byte_size_per_client],
            &mut counter_block,
            ctr_inc_bits,
            decrypted_parameters_vec.as_mut_slice(),
        );
        match ret {
            Ok(()) => {}
            Err(_) => {
                return sgx_status_t::SGX_ERROR_UNEXPECTED;
            }
        }
        all_uploaded_parameters
            .weights
            .extend(Parameters::make_weights_from_bytes(
                &decrypted_parameters_vec,
                given_num_of_sparse_parameters,
            ));
    }
    let end = start.elapsed();
    unsafe { *execution_time_results.offset(1) = end.as_secs_f32() };
    if verbose { println!("[SGX CLOCK] {}:  {}.{:06} seconds", "Decryption", end.as_secs(), end.subsec_nanos() / 1_000); }


    /*  Aggregation process under client-level DP */
    let start = Instant::now();

    // Oblivious aggregations and averaging
    match aggregation_alg {
        1 => advanced(
            num_of_sparse_parameters,
            aggregated_parameters,
            &mut all_uploaded_parameters.weights,
            client_size,
            verbose,
        ),
        2 => nips19(
            num_of_sparse_parameters,
            aggregated_parameters,
            &mut all_uploaded_parameters.weights,
            client_size,
        ),
        3 => baseline(
            aggregated_parameters,
            &all_uploaded_parameters.weights,
            client_size,
        ),
        4 => non_oblivious(
            aggregated_parameters,
            &all_uploaded_parameters.weights,
            client_size,
        ),
        5 => path_oram_with_zerotrace(
            aggregated_parameters,
            &mut all_uploaded_parameters.weights,
            client_size,
            verbose,
        ),
        6 => advanced_simple(
            num_of_sparse_parameters,
            aggregated_parameters,
            &mut all_uploaded_parameters.weights,
            client_size,
            verbose,
        ),
        _ => panic!("aggregation algorithm is nothing"),
    }

    // Add Gaussian noise
    if dp { rdp_gaussian_mechanism(aggregated_parameters, sigma, clipping, alpha, client_size); }

    let end = start.elapsed();
    unsafe { *execution_time_results.offset(2) = end.as_secs_f32() };
    if verbose { println!("[SGX CLOCK] {}:  {}.{:06} seconds", "Aggregation", end.as_secs(), end.subsec_nanos() / 1_000); }

    sgx_status_t::SGX_SUCCESS
}
