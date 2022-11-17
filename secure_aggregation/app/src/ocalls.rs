use std::slice;

use sgx_types::*;


// when handling large datasize, we have to care about external host's stack size limitation
// https://github.com/apache/incubator-teaclave-sgx-sdk/issues/77
#[no_mangle]
pub extern "C" fn ocall_load_next_data(
    current_cursor: usize,
    encrypted_parameters_data_ptr: *const u8,
    encrypted_parameters_data: *mut u8,
    encrypted_parameters_size: usize,
) -> sgx_status_t {
    let encrypted_parameters_to_upload_to_enclave: &mut [u8] =
        unsafe { slice::from_raw_parts_mut(encrypted_parameters_data, encrypted_parameters_size) };

    unsafe {
        let offset = (encrypted_parameters_size * current_cursor) as isize;
        let encrypted_parameters = slice::from_raw_parts(
            encrypted_parameters_data_ptr.offset(offset),
            encrypted_parameters_size
        );
        encrypted_parameters_to_upload_to_enclave.copy_from_slice(encrypted_parameters);
    }
    println!("ocall, finish");
    sgx_status_t::SGX_SUCCESS
}
