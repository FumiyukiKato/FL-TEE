#! /bin/bash
set -ex

# Relation between attack and noize


## mnist
# variable-number, mlp
# nn
python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
python src/fl_main.py --model=mlp --dp --sigma=0.5 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
python src/fl_main.py --model=mlp --dp --sigma=1.0 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
python src/fl_main.py --model=mlp --dp --sigma=2.0 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
python src/fl_main.py --model=mlp --dp --sigma=4.0 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
python src/fl_main.py --model=mlp --dp --sigma=8.0 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
python src/fl_main.py --model=mlp --dp --sigma=16.0 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8


# ## cifar10
# # variable-number, cnn
# # nn
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=cnn --dp --sigma=0.5 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=cnn --dp --sigma=1.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=cnn --dp --sigma=2.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=cnn --dp --sigma=4.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=cnn --dp --sigma=8.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=cnn --dp --sigma=16.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8

# ## cifar10
# # variable-number, mlp
# # nn
# python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=mlp --dp --sigma=0.5 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=mlp --dp --sigma=1.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=mlp --dp --sigma=2.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=mlp --dp --sigma=4.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=mlp --dp --sigma=8.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
# python src/fl_main.py --model=mlp --dp --sigma=16.0 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --fixed_inference_number=3 --attack_from_cache --attacker_batch_size=32 --prefix=exp8
