#! /bin/bash
set -ex

## Cacheline-Protection

## cifar10
# fixed-number, cnn
# nn
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --protection=cacheline
# nn-batch
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --protection=cacheline
# clustering
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=3 --fixed_inference_number=3  --attack_from_cache --prefix=exp6 --protection=cacheline