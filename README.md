# FL-TEE

### OLIVE: Oblivious and Differentially Private Federated Learning on TEE
---

We test at
- ubuntu 18.04 
- python 3.9.1 
- docker 20.10.12


Clone including rust-sgx-sdk
```
$ git clone --recursive https://github.com/FumiyukiKato/FL-TEE.git
```

## Setup FL
1. run pip install 
```bash
$ pip install -r requirements.txt
```

2. compile protocol buffer file 
```bash
$ cd src 
$ python compile_proto.py
```

3. If you need to compile client encryption library, run command 
```baash
$ cd src 
$ gcc -shared -fPIC -o src/libsgx_enc.so src/cpp/encryption.cpp -lssl -lcrypto 
$ python src/ffi_test.py
```
require host machine to set up linux-sgx and `export LD_LIBRARY_PATH=/opt/sgxsdk/sdk_libs`

4. If you need Purchase100 Dataset, donwload `purchase100.zip` (at https://drive.google.com/drive/folders/1nDDr8OWRaliIrUZcZ-0I8sEB2WqAXdKZ) to dataset/purchase100/ and unzip (using preprocessed data provided by https://github.com/bargavj/EvaluatingDPML [1][2])


## Setup Server
1. setup [linux-sgx-driver](https://github.com/intel/linux-sgx-driver) and confirm `/dev/isgx` on host machine (we user out-of-tree driver, but the Intel SGX driver is included in Linux kernel 5.11 and later)
```bash
$ cd linux-sgx-driver/
$ sudo make
$ sudo mkdir -p "/lib/modules/"`uname -r`"/kernel/drivers/intel/sgx"    
$ sudo cp isgx.ko "/lib/modules/"`uname -r`"/kernel/drivers/intel/sgx"    
$ sudo sh -c "cat /etc/modules | grep -Fxq isgx || echo isgx >> /etc/modules"    
$ sudo /sbin/depmod
$ sudo /sbin/modprobe isgx
```

2. install Rust sgx sdk of version 1.1.3 
```bash
$ cd FL-TEE
$ git clone https://github.com/apache/incubator-teaclave-sgx-sdk.git secure_aggregation/incubator-teaclave-sgx-sdk -b v1.1.3
```

3. run command (Using Docker image of [ubuntu 18.04 and sgx-rust v1.1.3](https://hub.docker.com/layers/baiduxlab/sgx-rust/1804-1.1.3/images/sha256-fbf4b495a0433ee2ef45ae9780b05d2f181aa6bbbe16dd0cf9ab5b4059ff15a5?context=explore) )
```bash
$ docker run -v /path/to/FL-TEE/secure_aggregation/incubator-teaclave-sgx-sdk:/root/sgx -v /path/to/FL-TEE:/root/FL-TEE -ti -p 50051:50051 -d --device /dev/isgx baiduxlab/sgx-rust:1804-1.1.3 
```

4. Login to the container and run command (Wake up AESM service)
```bash

(docker)$ LD_LIBRARY_PATH=/opt/intel/sgx-aesm-service/aesm /opt/intel/sgx-aesm-service/aesm/aesm_service
```

5. inside the container, run command  (Build the enclave application)
```bash
$ cd secure_aggregation
$ make
```

6. To wake up server, run command (Aggregation server is hosted at `0.0.0.0:50051`)
```bash
$ bin/secure-aggregation-server
```


Note that: Our experiment assumes the enclave and client has already performed a Remote Attestation handshake and established a shared secret key. In our source code, we use fixed keys for each client based on client ID and fixed IV. We think this is sufficient to observe the overhead caused by encryption for research purposes.


## Run
confirm setup server

```
$ python src/fl_main.py --model=mlp --dataset=mnist --epochs=10 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --secure_agg --aggregation_alg=advanced -v
```

```bash
python src/fl_main.py --model=mlp --dataset=mnist --epochs=10 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --secure_agg --aggregation_alg=optimized --optimal_num_of_clients=10
```

#### options

see `src/option.py` in detail

```
usage: fl_main.py [-h] --dataset DATASET --epochs EPOCHS --frac FRAC --num_users NUM_USERS --data_dist DATA_DIST [--num_of_label_k NUM_OF_LABEL_K] [--random_num_label] [--unequal]
                  [--epsilon EPSILON] [--delta DELTA] [--dp] [--sigma SIGMA] [--clipping CLIPPING] [--secure_agg] [--aggregation_alg {advanced,nips19,baseline,non_oblivious,path_oram}]
                  [--alpha ALPHA] --model MODEL [--num_channels NUM_CHANNELS] [--num_classes NUM_CLASSES] [--optimizer OPTIMIZER] [--local_ep LOCAL_EP] [--local_bs LOCAL_BS] [--lr LR]
                  [--momentum MOMENTUM] [--protection {index-privacy,k-anonymization,cacheline,}] [--index_privacy_r INDEX_PRIVACY_R] [--attack {clustering,nn}]
                  [--fixed_inference_number FIXED_INFERENCE_NUMBER] [--single_model] [--attack_from_cache] [--attacker_batch_size ATTACKER_BATCH_SIZE] [--per_round] [--seed SEED] [-v]
                  [--gpu_id GPU_ID] [--prefix PREFIX] [--local_skip]
```



### Experimental scripts
see `exp/` files

run like `$ sh exp/exp1.sh`


---

[1] Evaluating Differentially Private Machine Learning in Practice  
[2] Revisiting Membership Inference Under Realistic Assumptions
