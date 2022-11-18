extern crate hex;
extern crate sgx_types;
extern crate sgx_urts;

use secure_aggregation::aggregator_server::{Aggregator, AggregatorServer};
use secure_aggregation::{AggregateRequestParameters, AggregateResponseParameters, StartRequestParameters, StartResponseParameters};
use sgx_types::*;
use std::time::Instant;
use tonic::{transport::Server, Request, Response, Status};

pub mod secure_aggregation {
    tonic::include_proto!("secure_aggregation"); // The string specified here must match the proto package name
}

mod utils;
use utils::{get_algorithm_name, print_fl_settings, bool_to_u8, print_total_execution_time, print_fl_settings_for_each_round};

mod consts;

mod ecalls;
use ecalls::{
    ecall_client_size_optimized_secure_aggregation, ecall_fl_init, ecall_secure_aggregation,
    ecall_start_round, init_enclave,
};

mod ocalls;
#[allow(unused_imports)]
use ocalls::ocall_load_next_data;

const TIME_KIND: usize = 3;

#[derive(Debug, Default)]
pub struct CentralServer {
    pub enclave_id: sgx_enclave_id_t,
    pub verbose: bool,
    pub dp: bool,
}

#[tonic::async_trait]
impl Aggregator for CentralServer {
    async fn start(
        &self,
        request: Request<StartRequestParameters>,
    ) -> Result<Response<StartResponseParameters>, Status> {
        println!("Got a start request ...");
        let fl_id = request.get_ref().fl_id as u32;
        let client_ids = &request.get_ref().client_ids;
        let sigma = request.get_ref().sigma as f32;
        let clipping = request.get_ref().clipping as f32;
        let alpha = request.get_ref().alpha as f32;
        let sampling_ratio = request.get_ref().sampling_ratio as f32;
        let aggregation_alg = request.get_ref().aggregation_alg as u32;
        let num_of_parameters = request.get_ref().num_of_parameters as usize;
        let num_of_sparse_parameters = request.get_ref().num_of_sparse_parameters as usize;

        if self.verbose { print_fl_settings(get_algorithm_name(aggregation_alg), sigma, clipping, alpha, client_ids.len(), sampling_ratio, num_of_parameters, num_of_sparse_parameters); }

        let mut retval = sgx_status_t::SGX_SUCCESS;
        let mut result = unsafe {
            ecall_fl_init(
                self.enclave_id,
                &mut retval,
                fl_id,
                client_ids.as_ptr() as *const u32,
                client_ids.len(),
                num_of_parameters,
                num_of_sparse_parameters,
                sigma,
                clipping,
                alpha,
                sampling_ratio,
                aggregation_alg,
                bool_to_u8(self.verbose),
                bool_to_u8(self.dp),
            )
        };
        if result != sgx_status_t::SGX_SUCCESS || retval != sgx_status_t::SGX_SUCCESS {
            panic!("Error at ecall_fl_init")
        }

        let sample_size = (sampling_ratio * client_ids.len() as f32) as usize;
        let sampled_client_ids: Vec<u32> = vec![0u32; sample_size];
        result = unsafe {
            ecall_start_round(
                self.enclave_id,
                &mut retval,
                fl_id,
                0,
                sample_size,
                sampled_client_ids.as_ptr() as *mut u32,
            )
        };
        if result != sgx_status_t::SGX_SUCCESS || retval != sgx_status_t::SGX_SUCCESS {
            panic!("Error at ecall_start_round")
        }

        let reply = StartResponseParameters {
            fl_id: fl_id,
            round: 0,
            client_ids: sampled_client_ids,
        };

        Ok(Response::new(reply))
    }


    async fn aggregate(
        &self,
        request: Request<AggregateRequestParameters>,
    ) -> Result<Response<AggregateResponseParameters>, Status> {
        println!("Got a aggregate request ...");

        // request
        let fl_id = request.get_ref().fl_id as u32;
        let round = request.get_ref().round as u32;
        let aggregation_alg = request.get_ref().aggregation_alg as u32;
        let encrypted_parameters_data = &request.get_ref().encrypted_parameters;
        let num_of_parameters = request.get_ref().num_of_parameters as usize;
        let num_of_sparse_parameters = request.get_ref().num_of_sparse_parameters as usize;
        let client_ids = &request.get_ref().client_ids;
        let optimal_num_of_clients = request.get_ref().fl_id as usize;

        if self.verbose { print_fl_settings_for_each_round(
            fl_id, round, get_algorithm_name(aggregation_alg)) };
        
        // response
        let updated_parametes_data: Vec<f32> = vec![0f32; num_of_parameters];
        let mut execution_time_results: Vec<f32> = vec![0f32; TIME_KIND];

        let mut retval = sgx_status_t::SGX_SUCCESS;
        let mut result = sgx_status_t::SGX_SUCCESS;

        let start = Instant::now();
        if aggregation_alg == 6 {
            result = unsafe {
                ecall_client_size_optimized_secure_aggregation(
                    self.enclave_id,
                    &mut retval,
                    fl_id,
                    round,
                    optimal_num_of_clients,
                    client_ids.as_ptr() as *const u32,
                    client_ids.len(),
                    encrypted_parameters_data.as_ptr() as *const u8,
                    num_of_parameters,
                    num_of_sparse_parameters,
                    aggregation_alg,
                    updated_parametes_data.as_ptr() as *mut f32,
                    execution_time_results.as_ptr() as *mut f32
                )
            };
            if result != sgx_status_t::SGX_SUCCESS || retval != sgx_status_t::SGX_SUCCESS {
                panic!("Error at ecall_client_size_optimized_secure_aggregation")
            }
        } else {
            result = unsafe {
                ecall_secure_aggregation(
                    self.enclave_id,
                    &mut retval,
                    fl_id,
                    round,
                    client_ids.as_ptr() as *const u32,
                    client_ids.len(),
                    encrypted_parameters_data.as_ptr() as *const u8,
                    encrypted_parameters_data.len(),
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
        execution_time_results.push(end.as_secs_f32());
        if self.verbose { print_total_execution_time(end.as_secs(), end.subsec_nanos() / 1_000); }

        // Assuming that the next round is the same number of participants.
        let sample_size = client_ids.len();
        let sampled_client_ids: Vec<u32> = vec![0u32; sample_size];
        let next_round = round + 1;
        result = unsafe {
            ecall_start_round(
                self.enclave_id,
                &mut retval,
                fl_id,
                next_round,
                sample_size,
                sampled_client_ids.as_ptr() as *mut u32,
            )
        };
        if result != sgx_status_t::SGX_SUCCESS || retval != sgx_status_t::SGX_SUCCESS {
            panic!("Error at ecall_start_round")
        }

        let reply = AggregateResponseParameters {
            updated_parameters: updated_parametes_data,
            execution_time: end.as_secs_f32(),
            client_ids: sampled_client_ids,
            round: next_round
        };

        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50051".parse().unwrap();
    let mut central_server = CentralServer::default();

    println!("  init_enclave...");
    let enclave = match init_enclave() {
        Ok(r) => {
            println!("      Init Enclave Successful {}!", r.geteid());
            r
        }
        Err(x) => {
            println!(" Init Enclave Failed {}!", x.as_str());
            panic!("")
        }
    };

    central_server.enclave_id = enclave.geteid();
    central_server.verbose = true;
    central_server.dp = true;

    println!("  Now GRPC Server is binded on {:?}", addr);
    Server::builder()
        .add_service(AggregatorServer::new(central_server))
        .serve(addr)
        .await?;

    enclave.destroy();
    Ok(())
}
