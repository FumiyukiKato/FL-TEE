#!/bin/bash
set -ex

python src/fl_main.py --model=mlp --dataset=mnist --epochs=5 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering --num_of_label_k=3 --attack_from_cache --random_num_label

python src/fl_main.py --model=mlp --dataset=mnist --epochs=5 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering --num_of_label_k=3 --attack_from_cache --random_num_label --dp --epsilon=5.0 --delta=0.0001

python src/fl_main.py --model=mlp --dataset=mnist --epochs=5 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --num_of_label_k=3 --random_num_label --dp --epsilon=5.0 --delta=0.0001 --secure_agg