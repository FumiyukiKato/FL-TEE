import os
from grpc_tools import protoc
import ffi_test

# print("download rdp_accountant script")
# os.system("wget -nc -P ./src https://raw.githubusercontent.com/tensorflow/privacy/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py")


print("compile c++ encryption library for FFI")
os.system("gcc -shared -fPIC -I/usr/local/opt/openssl/include -L/usr/local/opt/openssl/lib -o src/libsgx_enc.so src/cpp/encryption.cpp -lssl -lcrypto")
ffi_test.test()

print("compile python proto file")
protoc.main(
    (
        '',
        '-I./proto/',
        '--python_out=./src',
        '--grpc_python_out=./src',
        'secure_aggregation.proto',
    )
)
