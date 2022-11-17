extern crate hex;
extern crate sgx_types;
extern crate sgx_urts;

use secure_aggregation::aggregator_server::{Aggregator, AggregatorServer};
use secure_aggregation::{ParametersReply, ParametersRequest};
use sgx_types::*;
use std::time::Instant;
use tonic::{transport::Server, Request, Response, Status};

pub mod secure_aggregation {
    tonic::include_proto!("secure_aggregation"); // The string specified here must match the proto package name
}

mod ecalls;
use ecalls::{ecall_secure_aggregation, init_enclave, ecall_client_size_optimized_secure_aggregation};

mod ocalls;
use ocalls::{ocall_load_next_data};

const TIME_KIND: usize = 3;
fn get_algorithm_name(code: u32) -> String {
    match code {
        1 => "advanced",
        2 => "nips19",
        3 => "baseline",
        4 => "non_oblivious",
        5 => "path_oram",
        6 => "optimized",
        _ => panic!("aggregation algorithm is nothing"),
    }.to_string()
}

#[derive(Debug, Default)]
pub struct CentralServer {
    pub enclave_id: sgx_enclave_id_t,
    pub verbose: bool,
    pub dp: bool,
}

#[tonic::async_trait]
impl Aggregator for CentralServer {
    async fn update_parameters(
        &self,
        request: Request<ParametersRequest>,
    ) -> Result<Response<ParametersReply>, Status> {
        println!("Got a request ..."); 

        let mut retval = sgx_status_t::SGX_SUCCESS;
        let encrypted_parameters_data = &request.get_ref().encrypted_parameters;
        let num_of_parameters = request.get_ref().num_of_parameters as usize;
        let num_of_sparse_parameters = request.get_ref().num_of_sparse_parameters as usize;
        let client_ids = &request.get_ref().client_ids;
        let sigma = request.get_ref().sigma as f32;
        let clipping = request.get_ref().clipping as f32;
        let alpha = request.get_ref().alpha as f32;
        let aggregation_alg = request.get_ref().aggregation_alg as u32;
        let optimal_num_of_clients = 100; // TODO
        let updated_parametes_data: Vec<f32> = vec![0f32; num_of_parameters];
        let mut execution_time_results: Vec<f32> = vec![0f32; TIME_KIND];
        
        if self.verbose {
            println!(" ** Request Params ** ");
            println!("    Aggregation algorithm = {}", get_algorithm_name(aggregation_alg));
            println!("    DP params (sigma, clipping, alpha) = ({}, {}, {})", sigma, clipping, alpha);
            println!("    Number of Client = {}", client_ids.len());
            println!("    Number of Parameter = {}, sparse parameter = {}", num_of_parameters, num_of_sparse_parameters);
        }

        let start = Instant::now();
        if aggregation_alg == 6 {
            let result = unsafe {
                ecall_client_size_optimized_secure_aggregation(
                    self.enclave_id,
                    &mut retval,
                    optimal_num_of_clients,
                    encrypted_parameters_data.as_ptr() as *const u8,
                    num_of_parameters,
                    num_of_sparse_parameters,
                    client_ids.as_ptr() as *const u32,
                    client_ids.len(),
                    sigma,
                    clipping,
                    alpha,
                    updated_parametes_data.as_ptr() as *mut f32,
                    execution_time_results.as_ptr() as *mut f32,
                    match self.verbose { false => 0u8, true => 1u8},
                    match self.dp { false => 0u8, true => 1u8},
                )
            };
            match result {
                sgx_status_t::SGX_SUCCESS => {
                    if self.verbose {
                        println!("[UNTRUSTED] ECALL Succes.");
                    }
                }
                _ => {
                    println!("[UNTRUSTED] Failed {}!", result.as_str());
                }
            }
        } else {
            let result = unsafe {
                ecall_secure_aggregation(
                    self.enclave_id,
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
                    match self.verbose { false => 0u8, true => 1u8},
                    match self.dp { false => 0u8, true => 1u8},
                )
            };
            match result {
                sgx_status_t::SGX_SUCCESS => {
                    if self.verbose {
                        println!("[UNTRUSTED] ECALL Succes.");
                    }
                }
                _ => {
                    println!("[UNTRUSTED] Failed {}!", result.as_str());
                }
            }
        }
        let end = start.elapsed();
        execution_time_results.push(end.as_secs_f32());
        if self.verbose {
            println!(
                "Total execution_time :  {}.{:06} seconds",
                end.as_secs(),
                end.subsec_nanos() / 1_000
            );
        }

        let reply = ParametersReply {
            updated_parameters: updated_parametes_data,
            execution_time: end.as_secs_f32(),
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
