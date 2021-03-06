use sgx_types::*;
use sgx_urts::SgxEnclave;

static ENCLAVE_FILE: &'static str = "bin/enclave.signed.so";

extern "C" {
    pub fn ecall_secure_aggregation(
        eid: sgx_enclave_id_t,
        retval: *mut sgx_status_t,
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
    ) -> sgx_status_t;
}

pub fn init_enclave() -> SgxResult<SgxEnclave> {
    let mut launch_token: sgx_launch_token_t = [0; 1024];
    let mut launch_token_updated: i32 = 0;
    // call sgx_create_enclave to initialize an enclave instance
    // Debug Support: set 2nd parameter to 1
    let debug = 1;
    let mut misc_attr = sgx_misc_attribute_t {
        secs_attr: sgx_attributes_t { flags: 0, xfrm: 0 },
        misc_select: 0,
    };
    SgxEnclave::create(
        ENCLAVE_FILE,
        debug,
        &mut launch_token,
        &mut launch_token_updated,
        &mut misc_attr,
    )
}
