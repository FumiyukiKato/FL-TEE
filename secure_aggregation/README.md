## Secure-aggregation component
1. `make`
2. `bin/secure-aggregation-server`   
- grpc server at `0.0.0.0:50051`


When using the optimized algorithm, pay attention to the stack size of the host machine. `ulimit -s 100000`

### One shot aggregation test
1. `make`
2. `bin/bench -v -a non_oblivious -c 100 -d 100000 -k 1000 -t 1`   
(-c client size -d original parameter size -k sparsified parameter size)



Note that: Our experiment assumes the enclave and client has already performed a Remote Attestation handshake and established a shared secret key. In our source code, we use fixed keys for each client based on client ID and fixed IV. We think this is sufficient to observe the overhead caused by encryption for research purposes.
