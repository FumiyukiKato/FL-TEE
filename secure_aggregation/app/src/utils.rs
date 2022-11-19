use crate::consts::*;
use sgx_types::*;
use std::time::Duration;

type SgxAesCtr128bitKeyT = [uint8_t; 16];
extern "C" {
    fn sgx_aes_ctr_encrypt(
        p_key: *const SgxAesCtr128bitKeyT,
        p_src: *const uint8_t,
        src_len: uint32_t,
        p_ctr: *const uint8_t,
        ctr_inc_bits: uint32_t,
        p_dst: *mut uint8_t,
    ) -> u32;
}

pub fn encrypt_to_flat_vec_u8(parameters: &Vec<(u32, Vec<(u32, f32)>)>) -> (Vec<u8>, Vec<u32>) {
    let mut u8_vec_list: Vec<u8> = Vec::with_capacity(parameters.len() * parameters[0].1.len());
    let mut client_ids = Vec::with_capacity(parameters.len());
    parameters.iter().for_each(|(client_id, parameter)| {
        // encrypt by session key as secure channel to enclave.
        let enc = encypt_by_client_id(*client_id, parameter);
        client_ids.push(*client_id);
        u8_vec_list.extend(enc);
    });
    (u8_vec_list, client_ids)
}

pub fn encypt_by_client_id(client_id: u32, parameter: &Vec<(u32, f32)>) -> Vec<u8> {
    let src_len: usize = parameter.len() * 8;
    let mut u8_vec: Vec<u8> = Vec::with_capacity(src_len);
    parameter.iter().for_each(|(idx, val)| {
        u8_vec.extend(&idx.to_le_bytes());
        u8_vec.extend(&val.to_le_bytes());
    });

    let mut shared_key: [u8; 16] = [0; 16];
    shared_key[4..8].copy_from_slice(&client_id.to_be_bytes());
    let counter_block: [u8; 16] = COUNTER_BLOCK;
    let ctr_inc_bits: u32 = SGXSSL_CTR_BITS;
    let mut encrypted_buf: Vec<u8> = vec![0; src_len];
    unsafe {
        sgx_aes_ctr_encrypt(
            &shared_key,
            u8_vec.as_ptr() as *const u8,
            src_len as u32,
            &counter_block as *const u8,
            ctr_inc_bits,
            encrypted_buf.as_mut_ptr(),
        )
    };
    encrypted_buf
}

pub fn print_total_execution_time(secs: u64, nano_sec: u32) {
    println!(
        "[Total execution time] :  {}.{:06} seconds",
        secs,
        nano_sec / 1_000
    );
}

pub fn bool_to_u8(b: bool) -> u8 {
    return match b {
        false => 0u8,
        true => 1u8,
    };
}

pub fn get_algorithm_name(code: u32) -> String {
    match code {
        1 => "advanced",
        2 => "nips19",
        3 => "baseline",
        4 => "non_oblivious",
        5 => "path_oram",
        6 => "optimized",
        _ => panic!("aggregation algorithm is nothing"),
    }
    .to_string()
}

pub fn print_fl_settings(
    aggregation_alg: String,
    sigma: f32,
    clipping: f32,
    alpha: f32,
    client_size: usize,
    sampling_ratio: f32,
    num_of_parameters: usize,
    num_of_sparse_parameters: usize,
) {
    println!("+++++++++++++++++++++++ FL Params +++++++++++++++++++++++");
    println!("    Aggregation algorithm = {}", aggregation_alg);
    println!(
        "    DP params (sigma, clipping, alpha) = ({}, {}, {})",
        sigma, clipping, alpha
    );
    println!("    Number of Client = {}", client_size);
    println!("    Sampling ratio   = {}", sampling_ratio);
    println!(
        "    Number of Parameter = {}, sparse parameter = {}",
        num_of_parameters, num_of_sparse_parameters
    );
    println!("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
}

pub fn print_fl_settings_for_each_round(
    fl_id: u32,
    round: u32,
    aggregation_alg: String,
) {
    println!("----------------------- FL ID {} ---------------------", fl_id);
    println!("    Round = {}", round);
    println!("    Aggregation algorithm = {}", aggregation_alg);
    println!("--------------------------------------------------------");
}
