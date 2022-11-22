#[macro_use]
extern crate prettytable;
extern crate rand;
extern crate sgx_types;
extern crate sgx_urts;

use chrono::Utc;
use clap::{App, Arg};
use prettytable::{Cell, Row, Table};
use rand::rngs::StdRng;
use rand::{seq::IteratorRandom, SeedableRng};
use sgx_types::*;
use std::fs::File;
use std::iter::FromIterator;
use std::time::Instant;
use std::collections::{HashSet, HashMap};

mod consts;
use consts::*;

mod utils;
use utils::*;

mod ecalls;
use ecalls::{
    ecall_client_size_optimized_secure_aggregation, ecall_fl_init, ecall_secure_aggregation,
    ecall_start_round, init_enclave,
};

mod ocalls;
#[allow(unused_imports)]
use ocalls::ocall_load_next_data;


const WEIGHT_BYTE_SIZE: usize =8;

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
        .arg(Arg::with_name("sampling_ratio")
            .long("sampling_ratio")
            .help("Sampling ratio of participants for each round")
            .default_value("0.01"))
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
        .arg(Arg::with_name("optimal_num_of_clients")
            .help("For optimized memory method")
            .long("optimal_num_of_clients")
            .default_value("0"))
}

fn one_shot_secure_aggregation(
    aggregation_alg: u32,
    sigma: f32,
    clipping: f32,
    alpha: f32,
    parameters: &Vec<(u32, Vec<(u32, f32)>)>,
    num_of_parameters: usize,
    num_of_sparse_parameters: usize,
    optimal_num_of_clients: usize,
    sampling_ratio: f32,
    eid: u64,
    verbose: bool,
    dp: bool,
) -> Vec<f32> {
    let fixed_fl_id = 0;
    let fixed_round = 0;

    let (encrypted_parameters_data, client_ids) = encrypt_to_flat_vec_u8(parameters);
    let parameter_byte_size_per_client = encrypted_parameters_data.len() / client_ids.len();
    let mut parameters_of_client_ids: HashMap<u32, Vec<u8>> = HashMap::new();
    for (i, client_id) in client_ids.iter().enumerate() {
        parameters_of_client_ids.insert(
            *client_id, 
            encrypted_parameters_data[(i*parameter_byte_size_per_client)..((i+1)*parameter_byte_size_per_client)].to_vec()
        );
    }

    let updated_parametes_data: Vec<f32> = vec![0f32; num_of_parameters];
    let mut execution_time_results: Vec<f32> = vec![0f32; TIME_KIND];

    let mut retval = sgx_status_t::SGX_SUCCESS;
    let mut result = unsafe {
        ecall_fl_init(
            eid,
            &mut retval,
            fixed_fl_id,
            client_ids.as_ptr() as *const u32,
            client_ids.len(),
            num_of_parameters,
            num_of_sparse_parameters,
            sigma,
            clipping,
            alpha,
            sampling_ratio,
            aggregation_alg,
            bool_to_u8(verbose),
            bool_to_u8(dp),
        )
    };
    if result != sgx_status_t::SGX_SUCCESS || retval != sgx_status_t::SGX_SUCCESS {
        panic!("Error at ecall_fl_init")
    }

    let sample_size = (sampling_ratio * client_ids.len() as f32) as usize;
    if optimal_num_of_clients > sample_size {
        panic!("optimal_num_of_clients is more than client size {}", sample_size);
    }
    let sampled_client_ids: Vec<u32> = vec![0u32; sample_size];
    result = unsafe {
        ecall_start_round(
            eid,
            &mut retval,
            fixed_fl_id,
            fixed_round,
            sample_size,
            sampled_client_ids.as_ptr() as *mut u32,
        )
    };
    if result != sgx_status_t::SGX_SUCCESS || retval != sgx_status_t::SGX_SUCCESS {
        panic!("Error at ecall_start_round")
    }
    // println!("sampled ids {:?}", sampled_client_ids);

    let mut uploaded_encrypted_data: Vec<u8> = vec![0; sample_size*num_of_sparse_parameters*WEIGHT_BYTE_SIZE];
    for i in 0..sample_size {
        uploaded_encrypted_data[i*num_of_sparse_parameters*WEIGHT_BYTE_SIZE..(i+1)*num_of_sparse_parameters*WEIGHT_BYTE_SIZE]
            .copy_from_slice(&parameters_of_client_ids[&sampled_client_ids[i]])
    }

    let start = Instant::now();
    if aggregation_alg == 6 {
        result = unsafe {
            ecall_client_size_optimized_secure_aggregation(
                eid,
                &mut retval,
                fixed_fl_id,
                fixed_round,
                optimal_num_of_clients,
                sampled_client_ids.as_ptr() as *const u32,
                sampled_client_ids.len(),
                uploaded_encrypted_data.as_ptr() as *const u8,
                num_of_parameters,
                num_of_sparse_parameters,
                aggregation_alg,
                updated_parametes_data.as_ptr() as *mut f32,
                execution_time_results.as_ptr() as *mut f32,
            )
        };
        if result != sgx_status_t::SGX_SUCCESS || retval != sgx_status_t::SGX_SUCCESS {
            panic!("Error at ecall_client_size_optimized_secure_aggregation")
        }
    } else {
        result = unsafe {
            ecall_secure_aggregation(
                eid,
                &mut retval,
                fixed_fl_id,
                fixed_round,
                sampled_client_ids.as_ptr() as *const u32,
                sampled_client_ids.len(),
                uploaded_encrypted_data.as_ptr() as *const u8,
                uploaded_encrypted_data.len(),
                num_of_parameters,
                num_of_sparse_parameters,
                aggregation_alg,
                updated_parametes_data.as_ptr() as *mut f32,
                execution_time_results.as_ptr() as *mut f32,
            )
        };
        if result != sgx_status_t::SGX_SUCCESS || retval != sgx_status_t::SGX_SUCCESS {
            panic!("Error at ecall_secure_aggregation")
        }
    }
    let end = start.elapsed();

    if verbose {
        print_total_execution_time(end.as_secs(), end.subsec_nanos() / 1_000)
    }
    // if verbose { println!("updated_parametes : {:?}", updated_parametes_data); }

    let sampled_clients_set: HashSet<u32> = HashSet::from_iter(sampled_client_ids.iter().cloned());
    let mut check_sum = 0.0;
    parameters.iter().for_each(|(client_id, parameter)| {
        if sampled_clients_set.contains(client_id) {
            check_sum += parameter.iter().fold(0.0, |x, y| x + y.1);
        }
    });
    check_sum /= sampled_clients_set.len() as f32;

    let enclave_check_sum = updated_parametes_data.iter().fold(0.0, |sum, x| sum + x);

    if verbose {
        println!("[CheckSum] enclave: {} == raw: {}", enclave_check_sum, check_sum);
    }

    execution_time_results.push(end.as_secs_f32());
    execution_time_results
}

fn main() {
    let opts = create_opts().get_matches();
    let aggregation_alg_list = match opts.value_of("aggregation_alg").unwrap() {
        "advanced" => vec![1],
        "nips19" => vec![2],
        "baseline" => vec![3],
        "non_oblivious" => vec![4],
        "path_oram" => vec![5],
        "optimized" => vec![6],
        "bubble" => vec![7],
        "all" => vec![1, 2, 3, 4, 5, 6],
        _ => panic!("invalid option: aggregation_alg"),
    };
    let aggregation_alg: &str = opts.value_of("aggregation_alg").unwrap();
    let num_of_clients: usize = opts.value_of("num_of_clients").unwrap().parse().unwrap();
    let num_of_parameters: usize = opts.value_of("num_of_parameters").unwrap().parse().unwrap();
    let num_of_sparse_parameters = opts
        .value_of("num_of_sparse_parameters")
        .unwrap()
        .parse()
        .unwrap();
    let sigma: f32 = opts.value_of("sigma").unwrap().parse().unwrap();
    let clipping: f32 = opts.value_of("clipping").unwrap().parse().unwrap();
    let alpha: f32 = opts.value_of("alpha").unwrap().parse().unwrap();
    let sampling_ratio: f32 = opts.value_of("sampling_ratio").unwrap().parse().unwrap();

    let trial: u32 = opts.value_of("trial").unwrap().parse().unwrap();
    let verbose = opts.is_present("verbose") as bool;
    let dp = opts.is_present("dp") as bool;

    let optimal_num_of_clients: usize = opts
        .value_of("optimal_num_of_clients")
        .unwrap()
        .parse()
        .unwrap();

    print_fl_settings(
        aggregation_alg.to_string(), sigma, clipping, alpha,
        num_of_clients, sampling_ratio, num_of_parameters, num_of_sparse_parameters
    );

    let seed: [u8; 32] = [13; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut parameters = Vec::<(u32, Vec<(u32, f32)>)>::with_capacity(num_of_clients);
    for i in 0..num_of_clients {
        let mut parameter = Vec::<(u32, f32)>::with_capacity(num_of_sparse_parameters);
        let sample = (0..num_of_parameters).choose_multiple(&mut rng, num_of_sparse_parameters);
        for idx in sample.iter() {
            parameter.push((*idx as u32, (*idx as f32) * 0.001));
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

    println!("[Server] init_enclave...");
    let enclave = match init_enclave() {
        Ok(r) => {
            println!("[Server] Init Enclave Successful {}!", r.geteid());
            r
        }
        Err(x) => {
            panic!("[Server] Init Enclave Failed {}!", x.as_str());
        }
    };
    let eid = enclave.geteid();

    for aggregation_alg in aggregation_alg_list.iter() {
        let optimized = format!("optimized-{}", optimal_num_of_clients);
        let alg_name = match *aggregation_alg {
            1 => "advanced",
            2 => "nips19",
            3 => "baseline",
            4 => "non_oblivious",
            5 => "path_oram",
            6 => optimized.as_str(),
            7 => "bubble",
            _ => panic!("invalid option: aggregation_alg"),
        };   

        let mut averages: Vec<f32> = vec![0.0; TIME_KIND + 1];
        for i in 0..(trial + 1) {
            println!("------------------- start  {} / {} -------------------", i+1, trial + 1);
            let execution_time_results = one_shot_secure_aggregation(
                *aggregation_alg,
                sigma,
                clipping,
                alpha,
                &parameters,
                num_of_parameters,
                num_of_sparse_parameters,
                optimal_num_of_clients,
                sampling_ratio,
                eid,
                verbose,
                dp,
            );
            println!("---------------------- end -----------------------");

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
                row.insert_cell(
                    0,
                    Cell::new(format!("{}", num_of_sparse_parameters).as_str()),
                );
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
        row.insert_cell(
            0,
            Cell::new(format!("{}", num_of_sparse_parameters).as_str()),
        );
        row.insert_cell(0, Cell::new(format!("{}", num_of_parameters).as_str()));
        row.insert_cell(
            0,
            Cell::new(format!("Avg w/o [0] ({} trial): {}", trial, alg_name).as_str()),
        );
        result_table.add_row(row);
    }
    result_table.printstd();

    let text = Utc::now().format("%Y%m%d%H%M%S%Z").to_string();
    let aggregation_alg_name = if aggregation_alg == "optimized" {
        format!("{}-{}", aggregation_alg, optimal_num_of_clients)
    } else {
        aggregation_alg.to_string()
    };
    let out = File::create(format!(
        "results/{}-{}-{}-{}-{}.txt",
        aggregation_alg_name, num_of_parameters, num_of_sparse_parameters, num_of_clients, text
    ))
    .expect("Faled to create output file");
    result_table.to_csv(out).expect("Faled to output file");

    enclave.destroy();
}
