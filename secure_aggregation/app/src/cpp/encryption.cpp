#include <stdio.h>

#include "openssl/evp.h"

#define SGXSSL_CTR_BITS	128
#define SHIFT_BYTE	8

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef uint8_t sgx_aes_ctr_128bit_key_t[16];

/*
* code taken from OpenSSL project.
* increment counter (128-bit int) by 1
*/
static void ctr128_inc(unsigned char *counter)
{
	unsigned int n = 16, c = 1;

	do {
		--n;
		c += counter[n];
		counter[n] = (unsigned char)c;
		c >>= SHIFT_BYTE;
	} while (n);
}

/* AES-CTR 128-bit
 * Parameters:
 *   Return:
 *     int - 0 or -1
 *   Inputs:
 *     sgx_aes_128bit_key_t *p_key - Pointer to the key used in encryption/decryption operation
 *     uint8_t *p_src - Pointer to the input stream to be encrypted/decrypted
 *     uint32_t src_len - Length of the input stream to be encrypted/decrypted
 *     uint8_t *p_ctr - Pointer to the counter block
 *     uint32_t ctr_inc_bits - Number of bits in counter to be incremented
 *   Output:
 *     uint8_t *p_dst - Pointer to the cipher text. Size of buffer should be >= src_len.
 */
extern "C" {
	uint32_t sgx_aes_ctr_encrypt(const sgx_aes_ctr_128bit_key_t *p_key, const uint8_t *p_src,
									const uint32_t src_len, uint8_t *p_ctr, const uint32_t ctr_inc_bits,
									uint8_t *p_dst)
	{

		if ((src_len > INT_MAX) || (p_key == NULL) || (p_src == NULL) || (p_ctr == NULL) || (p_dst == NULL))
		{
			return -2;
		}

		/* SGXSSL based crypto implementation */
		int ret = -1;
		int len = 0;
		EVP_CIPHER_CTX* ptr_ctx = NULL;

		// OpenSSL assumes that the counter is in the x lower bits of the IV(ivec), and that the
		// application has full control over overflow and the rest of the IV. This
		// implementation takes NO responsibility for checking that the counter
		// doesn't overflow into the rest of the IV when incremented.
		//
		if (ctr_inc_bits != SGXSSL_CTR_BITS)
		{
			return -3;
		}


		do {
			// Create and init ctx
			//
			if (!(ptr_ctx = EVP_CIPHER_CTX_new())) {
				ret = -4;
				break;
			}

			// Initialise encrypt, key
			//
			if (1 != EVP_EncryptInit_ex(ptr_ctx, EVP_aes_128_ctr(), NULL, (unsigned char*)p_key, p_ctr)) {
				break;
			}

			// Provide the message to be encrypted, and obtain the encrypted output.
			//
			if (1 != EVP_EncryptUpdate(ptr_ctx, p_dst, &len, p_src, src_len)) {
				break;
			}

			// Finalise the encryption
			//
			if (1 != EVP_EncryptFinal_ex(ptr_ctx, p_dst + len, &len)) {
				break;
			}

			// Encryption success, increment counter
			//
			len = src_len;
			while (len >= 0) {
				ctr128_inc(p_ctr);
				len -= 16;
			}
			ret = 0;
		} while (0);

		//clean up ctx and return
		//
		if (ptr_ctx) {
			EVP_CIPHER_CTX_free(ptr_ctx);
		}
		return ret;
	}

	uint32_t sgx_aes_ctr_decrypt(const sgx_aes_ctr_128bit_key_t *p_key, const uint8_t *p_src,
                                const uint32_t src_len, uint8_t *p_ctr, const uint32_t ctr_inc_bits,
									uint8_t *p_dst)
	{

		if ((src_len > INT_MAX) || (p_key == NULL) || (p_src == NULL) || (p_ctr == NULL) || (p_dst == NULL)) {
			return -2;
		}

		/* SGXSSL based crypto implementation */
		uint32_t ret = -1;
		int len = 0;
		EVP_CIPHER_CTX* ptr_ctx = NULL;

		// OpenSSL assumes that the counter is in the x lower bits of the IV(ivec), and that the
		// application has full control over overflow and the rest of the IV. This
		// implementation takes NO responsibility for checking that the counter
		// doesn't overflow into the rest of the IV when incremented.
		//
		if (ctr_inc_bits != SGXSSL_CTR_BITS) {
			return -2;
		}

		do {
			// Create and initialise the context
			//
			if (!(ptr_ctx = EVP_CIPHER_CTX_new())) {
				ret = -3;
				break;
			}

			// Initialise decrypt, key and CTR
			//
			if (!EVP_DecryptInit_ex(ptr_ctx, EVP_aes_128_ctr(), NULL, (unsigned char*)p_key, p_ctr)) {
				break;
			}

			// Decrypt message, obtain the plaintext output
			//
			if (!EVP_DecryptUpdate(ptr_ctx, p_dst, &len, p_src, src_len)) {
				break;
			}

			// Finalise the decryption. A positive return value indicates success,
			// anything else is a failure - the plaintext is not trustworthy.
			//
			if (EVP_DecryptFinal_ex(ptr_ctx, p_dst + len, &len) <= 0) { // same notes as above - you can't write beyond src_len
				break;
			}
			// Success
			// Increment counter
			//
			len = src_len;
			while (len >= 0) {
				ctr128_inc(p_ctr);
				len -= 16;
			}
			ret = 0;
		} while (0);

		//cleanup ctx, and return
		//
		if (ptr_ctx) {
			EVP_CIPHER_CTX_free(ptr_ctx);
		}
		return ret;
	}

}