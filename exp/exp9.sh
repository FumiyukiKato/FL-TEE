#! /bin/bash
set -ex


for i in `seq 2`
do
for optimal_num_of_clients in 10 100 1000 10000
do
python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=10000 --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp9 --secure_agg --aggregation_alg=optimized --local_skip --optimal_num_of_clients=$optimal_num_of_clients
done
done


# for i in `seq 5`
# do
# for n_users in 10 100 1000 10000
# do
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=1 --seed=0 --frac=0.3 --num_users=$n_users --data_dist=IID --optimizer=sgd --alpha=0.1  --prefix=exp5 --secure_agg --aggregation_alg=optimized --local_skip
# done
# done
