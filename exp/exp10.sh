#! /bin/bash
set -ex

# Attack performance for various number of attacker data size

## MNIST
# Fixed number of label
for attacker_data_size in 5000 1000 500 100 50 20 10
do
# mnist
# fixed-number
# nn
python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --attack_from_cache --attacker_data_size=$attacker_data_size --prefix=exp10
# nn-single
python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --single_model --attack_from_cache --attacker_data_size=$attacker_data_size  --prefix=exp10
# clustering (Jac)
python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --attacker_data_size=$attacker_data_size --prefix=exp10
done


## Random number of label
for attacker_data_size in 5000 1000 500 100 50 20 10
do
# nn
python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --attacker_batch_size=32 --attack_from_cache --random_num_label --attacker_data_size=$attacker_data_size --prefix=exp10
# nn-single
python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=3 --attacker_batch_size=32 --single_model --attack_from_cache --random_num_label --attacker_data_size=$attacker_data_size --prefix=exp10
# clustering (Jac)
python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=3  --attack_from_cache --random_num_label --attacker_data_size=$attacker_data_size --prefix=exp10
done


### purchase100
## Fixed number of label
for attacker_data_size in 10000 5000 1000 500 100
do
# nn
python src/fl_main.py --model=mlp --dataset=purchase100  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --attacker_data_size=$attacker_data_size --prefix=exp10
# nn-single
python src/fl_main.py --model=mlp --dataset=purchase100  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --attacker_data_size=$attacker_data_size --prefix=exp10
# clustering (Jac)
python src/fl_main.py --model=mlp --dataset=purchase100  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --attacker_data_size=$attacker_data_size --prefix=exp10
done

## Random number of label
for attacker_data_size in 10000 5000 1000 500 100
do
# nn
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=2 --attack_from_cache --attacker_batch_size=32 --random_num_label --attacker_data_size=$attacker_data_size --prefix=exp10
# nn-single
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=2 --attack_from_cache --attacker_batch_size=32 --single_model --random_num_label --attacker_data_size=$attacker_data_size --prefix=exp10
# clustering
python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=2  --attack_from_cache --random_num_label --attacker_data_size=$attacker_data_size --prefix=exp10
done