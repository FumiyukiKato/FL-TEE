#! /bin/bash
set -ex

# Relation between epoch and vulnerability

## mnist
# variable-number, k=3
# alpha=0.1
# nn
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=10 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --random_num_label --attacker_batch_size=32 --attack_from_cache --prefix=exp4 --per_round
# nn-single
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=10 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --random_num_label --attacker_batch_size=32 --attack_from_cache --prefix=exp4 --per_round --single_model
# clustering (Jac)
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=10 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=3 --random_num_label --attack_from_cache --prefix=exp4 --per_round

# alpha=0.8
# nn
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=10 --seed=0 --frac=0.1 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.8 --attack=nn --num_of_label_k=3 --random_num_label --attacker_batch_size=32 --attack_from_cache --prefix=exp4 --per_round
# nn-single
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=10 --seed=0 --frac=0.1 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.8 --attack=nn --num_of_label_k=3 --random_num_label --attacker_batch_size=32 --attack_from_cache --prefix=exp4 --per_round --single_model
# clustering (Jac)
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=10 --seed=0 --frac=0.1 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.8 --attack=clustering  --num_of_label_k=3 --random_num_label --attack_from_cache --prefix=exp4 --per_round


## cifar10
# variable-number, k=3
# alpha=0.1
# nn
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=10 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=2 --random_num_label --attacker_batch_size=32 --attack_from_cache --prefix=exp4 --per_round
# nn-single
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=10 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=2 --random_num_label --attacker_batch_size=32 --attack_from_cache --prefix=exp4 --per_round --single_model
# clustering (Jac)
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=10 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=2 --random_num_label --attack_from_cache --prefix=exp4 --per_round
