#! /bin/bash
set -ex

# Efficiency of secure aggregation
echo 'cofirm aggregation server is runnning at port 50051'

### alpha=0.1

## mnist
for i in `seq 5`
do
for n_users in 10 100 1000 10000
do
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=$n_users --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=advanced --local_skip
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=$n_users --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=baseline --local_skip
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=$n_users --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --local_skip
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=$n_users --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=path_oram --local_skip
done
done

for i in `seq 5`
do
for r in 0.25 0.5 1.0 2.0 4.0 8.0
do
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --protection=index-privacy --index_privacy_r=$r --local_skip
done
done

## cifar10
for i in `seq 5`
do
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=advanced --local_skip
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=baseline --local_skip
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --local_skip
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=path_oram --local_skip
for r in 0.25 0.5 1.0 2.0 4.0 8.0
do
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --protection=index-privacy --index_privacy_r=$r --local_skip
done
done

## purchase100
for i in `seq 5`
do
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=advanced --local_skip --num_classes=100
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=baseline --local_skip --num_classes=100
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --local_skip --num_classes=100
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=path_oram --local_skip --num_classes=100
for r in 0.25 0.5 1.0 2.0 4.0 8.0
do
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --protection=index-privacy --index_privacy_r=$r --local_skip --num_classes=100
done
done

## cifar100
for i in `seq 5`
do
for alpha in 0.025 0.05 0.1 0.2 0.4
do
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=$alpha  --prefix=exp5 --secure_agg --aggregation_alg=advanced --local_skip  --num_classes=100
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=$alpha  --prefix=exp5 --secure_agg --aggregation_alg=baseline --local_skip  --num_classes=100
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=$alpha  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --local_skip  --num_classes=100
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=$alpha  --prefix=exp5 --secure_agg --aggregation_alg=path_oram --local_skip  --num_classes=100
done
done

for i in `seq 5`
do
for r in 0.25 0.5 1.0 2.0 4.0 8.0
do
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --protection=index-privacy --index_privacy_r=$r --local_skip  --num_classes=100
done
done

### alpha=0.01

## mnist
for i in `seq 5`
do
for n_users in 10 100 1000 10000
do
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=$n_users --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=advanced --local_skip
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=$n_users --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=baseline --local_skip
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=$n_users --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --local_skip
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=$n_users --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=path_oram --local_skip
done
done

for i in `seq 5`
do
for r in 0.25 0.5 1.0 2.0 4.0 8.0
do
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --protection=index-privacy --index_privacy_r=$r --local_skip
done
done

## cifar10
for i in `seq 5`
do
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=advanced --local_skip
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=baseline --local_skip
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --local_skip
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=path_oram --local_skip
for r in 0.25 0.5 1.0 2.0 4.0 8.0
do
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --protection=index-privacy --index_privacy_r=$r --local_skip
done
done

## purchase100
for i in `seq 5`
do
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=advanced --local_skip --num_classes=100
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=baseline --local_skip --num_classes=100
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --local_skip --num_classes=100
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=path_oram --local_skip --num_classes=100
for r in 0.25 0.5 1.0 2.0 4.0 8.0
do
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --protection=index-privacy --index_privacy_r=$r --local_skip --num_classes=100
done
done

## cifar100
for i in `seq 5`
do
for alpha in 0.025 0.05 0.01 0.2 0.4
do
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=$alpha  --prefix=exp5 --secure_agg --aggregation_alg=advanced --local_skip  --num_classes=100
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=$alpha  --prefix=exp5 --secure_agg --aggregation_alg=baseline --local_skip  --num_classes=100
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=$alpha  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --local_skip  --num_classes=100
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=$alpha  --prefix=exp5 --secure_agg --aggregation_alg=path_oram --local_skip  --num_classes=100
done
done

for i in `seq 5`
do
for r in 0.25 0.5 1.0 2.0 4.0 8.0
do
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.3 --num_users=100 --data_dist=IID --optimizer=sgd --alpha=0.01  --prefix=exp5 --secure_agg --aggregation_alg=non_oblivious --protection=index-privacy --index_privacy_r=$r --local_skip  --num_classes=100
done
done