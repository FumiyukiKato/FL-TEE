#! /bin/bash
set -ex

# Attack performance fixed label number

# for num_of_label in 1 2 3 4 5 6 7 8 9
# do
# # mnist
# # fixed-number
# # nn
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attacker_batch_size=32 --attack_from_cache --prefix=exp1-no-dp
# # nn-single
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attacker_batch_size=32 --single_model --attack_from_cache  --prefix=exp1-no-dp
# # clustering (Jac)
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label  --attack_from_cache --prefix=exp1-no-dp
# done

for num_of_label in  8 9
do
## cifar10
# fixed-number, mlp
# nn
python src/fl_main.py --model=mlp --dataset=cifar10  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attack_from_cache --attacker_batch_size=32 --prefix=exp1-no-dp
# nn-single
python src/fl_main.py --model=mlp --dataset=cifar10  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp1-no-dp
# clustering (Jac)
python src/fl_main.py --model=mlp --dataset=cifar10  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label  --attack_from_cache --prefix=exp1-no-dp
done

for num_of_label in 1 2 3 4 5 6 7 8 9
do
## cifar10
# fixed-number, cnn
# nn
python src/fl_main.py --model=cnn --dataset=cifar10  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attack_from_cache --attacker_batch_size=32 --prefix=exp1-no-dp
# nn-single
python src/fl_main.py --model=cnn --dataset=cifar10  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp1-no-dp
# clustering (Jac)
python src/fl_main.py --model=cnn --dataset=cifar10  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label  --attack_from_cache --prefix=exp1-no-dp
done

for num_of_label in 1 2 4 8 16
do
## purchase100
# fixed-number
# nn
python src/fl_main.py --model=mlp --dataset=purchase100  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attack_from_cache --attacker_batch_size=32 --prefix=exp1-no-dp
# nn-single
python src/fl_main.py --model=mlp --dataset=purchase100  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp1-no-dp
# clustering (Jac)
python src/fl_main.py --model=mlp --dataset=purchase100  --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label  --attack_from_cache --prefix=exp1-no-dp
done

for num_of_label in 1 2 4 8 16
do
## cifar100
# fixed-number
# nn
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attack_from_cache --attacker_batch_size=32 --prefix=exp1-no-dp
# nn-single
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp1-no-dp
# clustering (Jac)
python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=1 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label --fixed_inference_number=$num_of_label  --attack_from_cache --prefix=exp1-no-dp
done
