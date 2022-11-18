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
use core::iter::FromIterator;
use std::collections::HashSet;
use std::slice;
use std::time::Instant;
use std::untrusted::time::InstantEx;
use std::vec::Vec;

use std::boxed::Box;
use std::cell::RefCell;
use std::sync::atomic::{AtomicPtr, Ordering};

mod parameters;
use parameters::{Parameters, WEIGHT_BYTE_SIZE};

mod common;
use common::{average_params, rdp_gaussian_mechanism, sample_client_ids};

mod fxhash;
mod oblivious_primitives;

mod nips19;
use nips19::nips19;

mod advanced;
use advanced::advanced;

mod client_size_optimized;
use client_size_optimized::client_size_optimized;

mod baseline;
use baseline::baseline;

mod non_oblivious;
use non_oblivious::non_oblivious;

mod oram;
use oram::path_oram_with_zerotrace;

mod fl_config;
use fl_config::{FLConfig, FLConfigMap};

mod session_key_store;
use session_key_store::SessionKeyStore;

// for secure channel encryption
pub const COUNTER_BLOCK: [u8; 16] = [0; 16];
pub const SGXSSL_CTR_BITS: u32 = 128;

pub static FL_CONFIG_MAP: AtomicPtr<()> = AtomicPtr::new(0 as *mut ());
pub fn get_ref_fl_config_map() -> Option<&'static RefCell<FLConfigMap>> {
    let ptr = FL_CONFIG_MAP.load(Ordering::SeqCst) as *mut RefCell<FLConfigMap>;
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &*ptr })
    }
}

pub static SESSION_KEYS: AtomicPtr<()> = AtomicPtr::new(0 as *mut ());
pub fn get_ref_session_keys() -> Option<&'static RefCell<SessionKeyStore>> {
    let ptr = SESSION_KEYS.load(Ordering::SeqCst) as *mut RefCell<SessionKeyStore>;
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &*ptr })
    }
}

extern "C" {
    pub fn ocall_load_next_data(
        ret_val: *mut sgx_status_t,
        current_cursor: usize,
        encrypted_parameters_data_ptr: *const u8,
        encrypted_parameters_data: *mut u8,
        encrypted_parameters_size: usize,
    ) -> sgx_status_t;
}

#[no_mangle]
pub extern "C" fn ecall_fl_init(
    fl_id: u32,
    client_ids: *const u32,
    client_size: usize,
    num_of_parameters: usize,
    num_of_sparse_parameters: usize,
    sigma: f32,
    clipping: f32,
    alpha: f32,
    sampling_ratio: f32,
    aggregation_alg: u32,
    verbose: u8,
    dp: u8,
) -> sgx_status_t {
    let fl_config_map = FLConfigMap::new();
    let fl_config_map_box = Box::new(RefCell::<FLConfigMap>::new(fl_config_map));
    let fl_config_map_ptr = Box::into_raw(fl_config_map_box);
    FL_CONFIG_MAP.store(fl_config_map_ptr as *mut (), Ordering::SeqCst);

    let client_ids_vec: Vec<u32> =
        unsafe { slice::from_raw_parts(client_ids, client_size) }.to_vec();
    if client_ids_vec.len() != client_size {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let mut fl_config_map = get_ref_fl_config_map().unwrap().borrow_mut();

    let fl_config = FLConfig {
        client_ids: client_ids_vec.clone(),
        client_size,
        num_of_parameters,
        num_of_sparse_parameters,
        sigma,
        clipping,
        alpha,
        sampling_ratio,
        aggregation_alg,
        verbose,
        dp,
        current_round: 0,
        current_sampled_clients: HashSet::new(),
    };
    fl_config_map.add(fl_id, fl_config);
    println!("[SGX] make fl config id {}", fl_id);

    mock_remote_attestation(client_ids_vec);

    sgx_status_t::SGX_SUCCESS
}

#[no_mangle]
pub extern "C" fn ecall_start_round(
    fl_id: u32,
    round: u32,
    sample_size: usize,
    sampled_client_ids: *mut u32,
) -> sgx_status_t {
    let mut fl_config_map = get_ref_fl_config_map().unwrap().borrow_mut();
    let fl_config = match fl_config_map.configs.get_mut(&fl_id) {
        Some(fl_config) => fl_config,
        None => return sgx_status_t::SGX_ERROR_UNEXPECTED,
    };
    if fl_config.current_round != round {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let sampled_client_ids_slice: &mut [u32] =
        unsafe { slice::from_raw_parts_mut(sampled_client_ids, sample_size) };
    let calc_sampled_size = (fl_config.client_ids.len() as f32 * fl_config.sampling_ratio) as usize;
    if calc_sampled_size != sample_size {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let sampled = sample_client_ids(&fl_config.client_ids, calc_sampled_size);
    for i in 0..sampled.len() {
        sampled_client_ids_slice[i] = sampled[i];
    }
    let new_sampled_clients_set : HashSet<u32> = HashSet::from_iter(sampled.iter().cloned());
    fl_config.current_sampled_clients = new_sampled_clients_set;

    println!(
        "[SGX] sampling for round {} is done and store {}/{} client ids in enclave.",
        round,
        sample_size,
        fl_config.client_ids.len()
    );
    sgx_status_t::SGX_SUCCESS
}

/// Secure aggregation
#[no_mangle]
pub extern "C" fn ecall_secure_aggregation(
    fl_id: u32,
    round: u32,
    client_ids: *const u32,
    client_size: usize,
    encrypted_parameters_data: *const u8,
    encrypted_parameters_size: usize,
    num_of_parameters: usize,
    num_of_sparse_parameters: usize,
    aggregation_alg: u32,
    updated_parameters_data: *mut f32,
    execution_time_results: *mut f32,
) -> sgx_status_t {
    let mut fl_config_map = get_ref_fl_config_map().unwrap().borrow_mut();
    let fl_config = match fl_config_map.configs.get_mut(&fl_id) {
        Some(fl_config) => fl_config,
        None => return sgx_status_t::SGX_ERROR_UNEXPECTED,
    };
    if fl_config.current_round != round {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    if fl_config.aggregation_alg != aggregation_alg {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let verbose = match fl_config.verbose {
        0 => false,
        1 => true,
        _ => true,
    };
    let dp = match fl_config.dp {
        0 => false,
        1 => true,
        _ => true,
    };

    // initialize parameter buffer
    let start = Instant::now();

    let client_ids_vec: Vec<u32> =
        unsafe { slice::from_raw_parts(client_ids, client_size) }.to_vec();
    if client_ids_vec.len() != client_size {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    // check uploaded client ids
    if client_ids_vec.len() != fl_config.current_sampled_clients.len() {
        println!("[VERIFICATION ERROR] Uploaded client id is not matched for secure sampled one.");
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    for uploaded in client_ids_vec.iter() {
        if !fl_config.current_sampled_clients.contains(uploaded) {
            println!("[VERIFICATION ERROR] Uploaded client id is not matched for secure sampled one.");
            return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
        }
    }

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

    let end = start.elapsed();
    unsafe { *execution_time_results.offset(0) = end.as_secs_f32() };
    if verbose {
        println!(
            "[SGX CLOCK] {}:  {}.{:06} seconds",
            "Loading",
            end.as_secs(),
            end.subsec_nanos() / 1_000
        );
    }

    // decryption
    let start = Instant::now();
    let byte_size_per_client = encrypted_parameters_vec.len() / client_size;
    let given_num_of_sparse_parameters = byte_size_per_client / WEIGHT_BYTE_SIZE;
    let mut all_uploaded_parameters: Parameters =
        Parameters::new(given_num_of_sparse_parameters * client_size);
    let mut decrypted_parameters_vec: Vec<u8> = vec![0; byte_size_per_client];
    let session_key_store = get_ref_session_keys().unwrap().borrow();

    for (i, client_id) in client_ids_vec.iter().enumerate() {
        let mut counter_block: [u8; 16] = COUNTER_BLOCK;
        let ctr_inc_bits: u32 = SGXSSL_CTR_BITS;

        // Originally shared_key is derived by following Remote Attestation protocol.
        // This is mock of shared key-based encryption.
        // The 128 bit key is [(client_id (64bit))0...0].
        let session_key = match session_key_store.map.get(client_id) {
            Some(session_key) => session_key,
            None => return sgx_status_t::SGX_ERROR_UNEXPECTED,
        };
        let current_cursor = i * (byte_size_per_client); // num_of_parameters * (index: 8 bytes, value: 8bytes)
        let ret = rsgx_aes_ctr_decrypt(
            &session_key,
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
    if verbose {
        println!(
            "[SGX CLOCK] {}:  {}.{:06} seconds",
            "Decryption",
            end.as_secs(),
            end.subsec_nanos() / 1_000
        );
    }

    /*  Aggregation process under client-level DP */
    let start = Instant::now();

    // Oblivious aggregations and averaging
    match fl_config.aggregation_alg {
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
        _ => panic!("aggregation algorithm is nothing"),
    }

    // Add Gaussian noise
    if dp {
        rdp_gaussian_mechanism(
            aggregated_parameters,
            fl_config.sigma,
            fl_config.clipping,
            fl_config.alpha,
            client_size,
        );
    }

    let end = start.elapsed();
    unsafe { *execution_time_results.offset(2) = end.as_secs_f32() };
    if verbose {
        println!(
            "[SGX CLOCK] {}:  {}.{:06} seconds",
            "Aggregation",
            end.as_secs(),
            end.subsec_nanos() / 1_000
        );
    }

    fl_config.increment_round();
    sgx_status_t::SGX_SUCCESS
}

#[no_mangle]
pub extern "C" fn ecall_client_size_optimized_secure_aggregation(
    fl_id: u32,
    round: u32,
    optimal_num_of_clients: usize,
    client_ids: *const u32,
    client_size: usize,
    encrypted_parameters_data_ptr: *const u8,
    num_of_parameters: usize,
    num_of_sparse_parameters: usize,
    aggregation_alg: u32,
    updated_parameters_data: *mut f32,
    execution_time_results: *mut f32,
) -> sgx_status_t {
    let mut fl_config_map = get_ref_fl_config_map().unwrap().borrow_mut();
    let fl_config = match fl_config_map.configs.get_mut(&fl_id) {
        Some(fl_config) => fl_config,
        None => return sgx_status_t::SGX_ERROR_UNEXPECTED,
    };
    if fl_config.current_round != round {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    if fl_config.aggregation_alg != aggregation_alg {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let verbose = match fl_config.verbose {
        0 => false,
        1 => true,
        _ => true,
    };
    let dp = match fl_config.dp {
        0 => false,
        1 => true,
        _ => true,
    };

    let client_ids_vec: Vec<u32> =
        unsafe { slice::from_raw_parts(client_ids, client_size) }.to_vec();
    if client_ids_vec.len() != client_size {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    // check uploaded client ids
    if client_ids_vec.len() != fl_config.current_sampled_clients.len() {
        println!("[VERIFICATION ERROR] Uploaded client id is not matched for secure sampled one.");
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    for uploaded in client_ids_vec.iter() {
        if !fl_config.current_sampled_clients.contains(uploaded) {
            println!("[VERIFICATION ERROR] Uploaded client id is not matched for secure sampled one.");
            return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
        }
    }

    // initialize parameter buffer
    let start = Instant::now();

    // store global parameters as local variable
    let aggregated_parameters: &mut [f32] =
        unsafe { slice::from_raw_parts_mut(updated_parameters_data, num_of_parameters) };

    let end = start.elapsed();
    unsafe { *execution_time_results.offset(0) = end.as_secs_f32() };
    if verbose {
        println!(
            "[SGX CLOCK] {}:  {}.{:06} seconds",
            "Loading",
            end.as_secs(),
            end.subsec_nanos() / 1_000
        );
    }

    let mut current_cursor = 0;
    let cursor_last = client_size / optimal_num_of_clients;
    let mut rt: sgx_status_t = sgx_status_t::SGX_ERROR_UNEXPECTED;
    let byte_size_per_client = num_of_sparse_parameters * WEIGHT_BYTE_SIZE;
    let mut decrypted_parameters_vec: Vec<u8> = vec![0; byte_size_per_client];

    let session_key_store = get_ref_session_keys().unwrap().borrow();

    while current_cursor <= cursor_last {
        if current_cursor * optimal_num_of_clients >= client_size {
            break;
        }
        let mut loaded_parameters: Parameters =
            Parameters::new(num_of_sparse_parameters * optimal_num_of_clients);
        let loaded_encrypted_parameters_vec: Vec<u8> =
            vec![0; byte_size_per_client * optimal_num_of_clients];
        let to_idx = if (current_cursor + 1) * optimal_num_of_clients < client_size {
            (current_cursor + 1) * optimal_num_of_clients
        } else {
            client_size
        };
        let client_ids_of_this_round =
            &client_ids_vec[current_cursor * optimal_num_of_clients..to_idx];

        // load optimal sized data
        let res = unsafe {
            ocall_load_next_data(
                &mut rt as *mut sgx_status_t,
                current_cursor,
                encrypted_parameters_data_ptr,
                loaded_encrypted_parameters_vec.as_ptr() as *mut u8,
                loaded_encrypted_parameters_vec.len(),
            )
        };
        if res != sgx_status_t::SGX_SUCCESS {
            return res;
        }
        if rt != sgx_status_t::SGX_SUCCESS {
            return rt;
        }

        for (i, client_id) in client_ids_of_this_round.iter().enumerate() {
            let mut counter_block: [u8; 16] = COUNTER_BLOCK;
            let ctr_inc_bits: u32 = SGXSSL_CTR_BITS;

            let session_key = match session_key_store.map.get(client_id) {
                Some(session_key) => session_key,
                None => return sgx_status_t::SGX_ERROR_UNEXPECTED,
            };
            let current_param_cursor = i * (byte_size_per_client); // num_of_parameters * (index: 8 bytes, value: 8bytes)
            let ret = rsgx_aes_ctr_decrypt(
                &session_key,
                &loaded_encrypted_parameters_vec
                    [current_param_cursor..current_param_cursor + byte_size_per_client],
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
            loaded_parameters
                .weights
                .extend(Parameters::make_weights_from_bytes(
                    &decrypted_parameters_vec,
                    num_of_sparse_parameters,
                ));
        }

        client_size_optimized(
            num_of_sparse_parameters,
            aggregated_parameters,
            &mut loaded_parameters.weights,
            client_ids_of_this_round.len(),
            verbose,
        );
        current_cursor += 1;
    }
    average_params(aggregated_parameters, client_size);
    let end = start.elapsed();
    unsafe { *execution_time_results.offset(1) = end.as_secs_f32() };
    if verbose {
        println!(
            "[SGX CLOCK] {}:  {}.{:06} seconds",
            "Decryption",
            end.as_secs(),
            end.subsec_nanos() / 1_000
        );
    }

    // Add Gaussian noise
    if dp {
        rdp_gaussian_mechanism(aggregated_parameters, fl_config.sigma, fl_config.clipping, fl_config.alpha, client_size);
    }

    let end = start.elapsed();
    unsafe { *execution_time_results.offset(2) = end.as_secs_f32() };
    if verbose {
        println!(
            "[SGX CLOCK] {}:  {}.{:06} seconds",
            "Aggregation",
            end.as_secs(),
            end.subsec_nanos() / 1_000
        );
    }

    sgx_status_t::SGX_SUCCESS
}

fn mock_remote_attestation(client_ids_vec: Vec<u32>) -> sgx_status_t {
    println!("[SGX] remote attestation mock");
    let session_key_store = SessionKeyStore::build_mock(client_ids_vec);
    let session_key_store_box = Box::new(RefCell::<SessionKeyStore>::new(session_key_store));
    let session_key_store_ptr = Box::into_raw(session_key_store_box);
    SESSION_KEYS.store(session_key_store_ptr as *mut (), Ordering::SeqCst);
    sgx_status_t::SGX_SUCCESS
}
