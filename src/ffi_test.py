import ctypes


def test():
    lib = ctypes.cdll.LoadLibrary('src/libsgx_enc.so')

    # lib.sgx_aes_ctr_encrypt.argtypes = [
    #     ctypes.c_uint8 * 16,
    #     ctypes.POINTER(ctypes.c_uint8),
    #     ctypes.c_uint32,
    #     ctypes.c_uint8 * 16,
    #     ctypes.c_uint32,
    #     ctypes.POINTER(ctypes.c_uint8)
    # ]
    # lib.sgx_aes_ctr_encrypt.restype = ctypes.c_uint32

    # lib.sgx_aes_ctr_decrypt.argtypes = [
    #     ctypes.c_uint8 * 16,
    #     ctypes.POINTER(ctypes.c_uint8),
    #     ctypes.c_uint32,
    #     ctypes.c_uint8 * 16,
    #     ctypes.c_uint32,
    #     ctypes.POINTER(ctypes.c_uint8)
    # ]
    # lib.sgx_aes_ctr_decrypt.restype = ctypes.c_uint32

    key = [0] * 16
    p_key = (ctypes.c_uint8 * len(key))(*key)
    src = [1] * 100
    src_buffer = (ctypes.c_uint8 * len(src))(*src)
    p_src = ctypes.cast(src_buffer, ctypes.POINTER(ctypes.c_uint8))
    src_len = 100
    ctr = [0] * 16
    p_ctr = (ctypes.c_uint8 * len(ctr))(*ctr)
    ctr_inc_bits = 128
    dst_buffer = (ctypes.c_uint8 * 100)()
    p_dst = ctypes.cast(dst_buffer, ctypes.POINTER(ctypes.c_uint8))
    lib.sgx_aes_ctr_encrypt(
        p_key,
        p_src,
        src_len,
        p_ctr,
        ctr_inc_bits,
        p_dst
    )
    decrypted_dst = ctypes.cast(dst_buffer, ctypes.POINTER(ctypes.c_uint8))
    ctr = [0] * 16
    p_ctr = (ctypes.c_uint8 * len(ctr))(*ctr)
    lib.sgx_aes_ctr_decrypt(
        p_key,
        p_dst,
        src_len,
        p_ctr,
        ctr_inc_bits,
        decrypted_dst
    )
    decrypted = [decrypted_dst[i] for i in range(100)]

    if decrypted == src:
        print('ok')
    else:
        print('error')


test()
