#! /bin/bash
set -ex

# Attack performance fixed label number

for attacker_data_size in 100 500 1000 5000 10000
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