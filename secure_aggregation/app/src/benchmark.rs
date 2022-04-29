#[macro_use]
extern crate prettytable;
extern crate sgx_types;
extern crate sgx_urts;
extern crate rand;

use clap::{App, Arg};
use prettytable::{Cell, Row, Table};
use sgx_types::*;
use std::time::Instant;
use rand::{seq::IteratorRandom, SeedableRng};
use rand::rngs::StdRng;
use std::fs::File;
use chrono::Utc;

mod ecalls;
use ecalls::{ecall_secure_aggregation, init_enclave};

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

const COUNTER_BLOCK: [u8; 16] = [0; 16];
const SGXSSL_CTR_BITS: u32 = 128;

const TIME_KIND: usize = 3;

fn encrypt_to_flat_vec_u8(parameters: &Vec<(u32, Vec<(u32, f32)>)>) -> (Vec<u8>, Vec<u32>) {
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

fn encypt_by_client_id(client_id: u32, parameter: &Vec<(u32, f32)>) -> Vec<u8> {
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

fn secure_aggregation(
    aggregation_alg: u32,
    sigma: f32,
    clipping: f32,
    alpha: f32,
    parameters: &Vec<(u32, Vec<(u32, f32)>)>,
    num_of_parameters: usize,
    num_of_sparse_parameters: usize,
    eid: u64,
    verbose: bool,
    dp: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut retval = sgx_status_t::SGX_SUCCESS;

    let (encrypted_parameters_data, client_ids) =
        encrypt_to_flat_vec_u8(parameters);
    let updated_parametes_data: Vec<f32> = vec![0f32; num_of_parameters];
    let mut execution_time_results: Vec<f32> = vec![0f32; TIME_KIND];

    if verbose {
        println!(
            "number of parameters: {}, number of sparse parameters: {}, number of clients: {}",
            num_of_parameters,
            num_of_sparse_parameters,
            client_ids.len()
        );
    }
    let start = Instant::now();
    let result = unsafe {
        ecall_secure_aggregation(
            eid,
            &mut retval,
            encrypted_parameters_data.as_ptr() as *const u8,
            encrypted_parameters_data.len(),
            num_of_parameters,
            num_of_sparse_parameters,
            client_ids.as_ptr() as *const u32,
            client_ids.len(),
            sigma,
            clipping,
            alpha,
            aggregation_alg,
            updated_parametes_data.as_ptr() as *mut f32,
            execution_time_results.as_ptr() as *mut f32,
            match verbose { false => 0u8, true => 1u8},
            match dp { false => 0u8, true => 1u8},
        )
    };
    match result {
        sgx_status_t::SGX_SUCCESS => {
            if verbose {
                println!("[UNTRUSTED] ECALL Succes.");
            }
        }
        _ => {
            println!("[UNTRUSTED] Failed {}!", result.as_str());
        }
    }
    let end = start.elapsed();

    if verbose {
        println!(
            "Total execution_time :  {}.{:06} seconds",
            end.as_secs(),
            end.subsec_nanos() / 1_000
        );
    }
    // if verbose { println!("updated_parametes : {:?}", updated_parametes_data); }
    execution_time_results.push(end.as_secs_f32());
    (updated_parametes_data, execution_time_results)
}

fn create_opts() -> App<'static, 'static> {
    App::new("Benchmark")
        .version("0.1")
        .author("Fumiyuki K. <fumilemon79@gmail.com>")
        .about("Benchmark different oblivious aggregations")
        .arg(Arg::with_name("num_of_clients")
            .short("c")
            .long("num_of_clients")
            .help("Number of clients")
            .default_value("10"))
        .arg(Arg::with_name("num_of_parameters")
            .short("d")
            .long("num_of_parameters")
            .help("Number of parameters per client d")
            .default_value("100000"))
        .arg(Arg::with_name("num_of_sparse_parameters")
            .short("k")
            .long("num_of_sparse_parameters")
            .help("Number of parameters per client k")
            .default_value("1000"))
        .arg(Arg::with_name("aggregation_alg")
            .short("a")
            .long("aggregation_alg")
            .help("Oblivious aggregation algorithm [advanced, nips19, baseline, non_oblivious, path_oram, all] (default: non_oblivious)")
            .default_value("non_oblivious"))
        .arg(Arg::with_name("sigma")
            .long("sigma")
            .help("The scale of Gaussian Distribution for RDP")
            .default_value("1.12"))
        .arg(Arg::with_name("clipping")
            .long("clipping")
            .help("Clipping parameter for client-level DP-SGD")
            .default_value("1.0"))
        .arg(Arg::with_name("alpha")
            .long("alpha")
            .help("Sparse rate")
            .default_value("0.1"))
        .arg(Arg::with_name("trial")
            .help("Number of trials and show average")
            .short("t")
            .long("trial")
            .default_value("1"))
        .arg(Arg::with_name("verbose")
            .help("Turn debugging information on")
            .short("v")
            .long("verbose"))
        .arg(Arg::with_name("dp")
            .help("Adding noise for DP or without noise")
            .long("dp"))
}


fn main() {
    let opts = create_opts().get_matches();
    let aggregation_alg_list = match opts.value_of("aggregation_alg").unwrap() {
        "advanced" => vec![1],
        "nips19" => vec![2],
        "baseline" => vec![3],
        "non_oblivious" => vec![4],
        "path_oram" => vec![5],
        "all" => vec![1, 2, 3, 4, 5],
        _ => panic!("invalid option: aggregation_alg"),
    };
    let aggregation_alg: &str = opts.value_of("aggregation_alg").unwrap();
    let num_of_clients: usize = opts.value_of("num_of_clients").unwrap().parse().unwrap();
    let num_of_parameters: usize = opts.value_of("num_of_parameters").unwrap().parse().unwrap();
    let num_of_sparse_parameters = opts.value_of("num_of_sparse_parameters").unwrap().parse().unwrap();
    let sigma: f32 = opts.value_of("sigma").unwrap().parse().unwrap();
    let clipping: f32 = opts.value_of("clipping").unwrap().parse().unwrap();
    let alpha: f32 = opts.value_of("alpha").unwrap().parse().unwrap();

    let trial: u32 = opts.value_of("trial").unwrap().parse().unwrap();
    let verbose = opts.is_present("verbose") as bool;
    let dp = opts.is_present("dp") as bool;

    println!("");
    println!(" ** Params ** ");
    println!("    Aggregation algorithm = {}", aggregation_alg);
    println!("    DP params (sigma, clipping, alpha) = ({}, {}, {})", sigma, clipping, alpha);
    println!("    Number of Client = {}", num_of_clients);
    println!("    Number of Parameter = {}, sparse parameter = {}", num_of_parameters, num_of_sparse_parameters);

    let seed: [u8; 32] = [13; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut true_sum = 0.0;
    let mut parameters = Vec::<(u32, Vec<(u32, f32)>)>::with_capacity(num_of_clients);
    for i in 0..num_of_clients {
        let mut parameter = Vec::<(u32, f32)>::with_capacity(num_of_sparse_parameters);
        let sample = (0..num_of_parameters).choose_multiple(&mut rng, num_of_sparse_parameters);
        for idx in sample.iter() {
            parameter.push((*idx as u32, (*idx as f32) * 0.001));
            true_sum += (*idx as f32) * 0.001;
        }
        // if verbose { println!("parameter {}: {:?}", i, parameter); }
        parameters.push((i as u32, parameter));
    }

    let mut result_table = Table::new();
    result_table.add_row(row![
        "Algorithm",
        "num_of_parameters",
        "num_of_sparse_parameters",
        "num_of_clients",
        "Load [s]",
        "Decryption [s]",
        "Aggregation [s]",
        "Total [s]"
    ]);

    println!("init_enclave...");
    let enclave = match init_enclave() {
        Ok(r) => {
            println!(" Init Enclave Successful {}!", r.geteid());
            r
        }
        Err(x) => {
            println!(" Init Enclave Failed {}!", x.as_str());
            panic!("")
        }
    };
    let eid = enclave.geteid();

    for aggregation_alg in aggregation_alg_list.iter() {
        let alg_name = match *aggregation_alg {
            1 => "advanced",
            2 => "nips19",
            3 => "baseline",
            4 => "non_oblivious",
            5 => "path_oram",
            _ => panic!("invalid option: aggregation_alg"),
        };
        if verbose {
            println!("aggregation_alg: {}", alg_name);
        }

        let mut averages: Vec<f32> = vec![0.0; TIME_KIND + 1];
        for i in 0..(trial + 1) {
            let (updated_parametes_data, execution_time_results) =
                secure_aggregation(*aggregation_alg, sigma, clipping, alpha, &parameters, num_of_parameters, num_of_sparse_parameters, eid, verbose, dp);
            if verbose {
                let sum = updated_parametes_data.iter().fold(0.0, |sum, x| sum + x);
                println!(" ** Verification ** ");
                println!(
                    "    Sum of updated_parametes : {} == {} : True sum of updated_parametes",
                    sum,
                    true_sum / (num_of_clients as f32)
                );
            }

            if i >= 1 {
                // The result of the first iteration for each test is always discarded 
                // because the instruction and data caches are not yet hot, thus the first iteration is always slower.
                (0..averages.len()).for_each(|j| averages[j] += execution_time_results[j]);
            }

            if verbose {
                let mut row = Row::from(
                    execution_time_results
                        .iter()
                        .map(|x| format!("{:.8}", x))
                        .collect::<Vec<String>>(),
                );
                row.insert_cell(0, Cell::new(format!("{}", num_of_clients).as_str()));
                row.insert_cell(0, Cell::new(format!("{}", num_of_sparse_parameters).as_str()));
                row.insert_cell(0, Cell::new(format!("{}", num_of_parameters).as_str()));
                row.insert_cell(0, Cell::new(format!("[{}]: {}", i, alg_name).as_str()));
                result_table.add_row(row);
            }
        }

        (0..averages.len()).for_each(|j| averages[j] /= trial as f32);
        let mut row = Row::from(
            averages
                .iter()
                .map(|x| format!("{:.8}", x))
                .collect::<Vec<String>>(),
        );
        row.insert_cell(0, Cell::new(format!("{}", num_of_clients).as_str()));
        row.insert_cell(0, Cell::new(format!("{}", num_of_sparse_parameters).as_str()));
        row.insert_cell(0, Cell::new(format!("{}", num_of_parameters).as_str()));
        row.insert_cell(
            0,
            Cell::new(format!("Avg w/o [0] ({} trial): {}", trial, alg_name).as_str()),
        );
        result_table.add_row(row);
    }
    result_table.printstd();


    let text = Utc::now().format("%Y%m%d%H%M%S%Z").to_string();
    let out = File::create(format!("results/{}-{}-{}-{}-{}.txt", 
        aggregation_alg, num_of_parameters, num_of_sparse_parameters, num_of_clients, text)).expect("Faled to create output file");
    result_table.to_csv(out).expect("Faled to output file");

    enclave.destroy();
}
