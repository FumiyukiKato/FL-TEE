#! /bin/bash
set -ex

# Attack performance variable label number

for num_of_label in 1 2 3 4 5 6 7 8 9
do
## mnist
# variable-number
# nn
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attacker_batch_size=32 --attack_from_cache --random_num_label --prefix=exp2
# nn-single
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attacker_batch_size=32 --single_model --attack_from_cache  --random_num_label --prefix=exp2
# clustering (Jac)
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label  --attack_from_cache --random_num_label --prefix=exp2
done

for num_of_label in 1 2 3 4 5 6 7 8 9
do
## cifar10
# variable-number, mlp
# nn
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attack_from_cache --attacker_batch_size=32 --random_num_label --prefix=exp2
# nn-single
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attack_from_cache --attacker_batch_size=32 --single_model --random_num_label --prefix=exp2
# clustering (Jac)
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label  --attack_from_cache --random_num_label --prefix=exp2
done

for num_of_label in 1 2 3 4 5 6 7 8 9
do
## cifar10
# variable-number, cnn
# nn
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attack_from_cache --attacker_batch_size=32 --random_num_label --prefix=exp2
# nn-single
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attack_from_cache --attacker_batch_size=32 --single_model --random_num_label --prefix=exp2
# clustering (Jac)
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar10 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label  --attack_from_cache --random_num_label --prefix=exp2
done

for num_of_label in 1 2 4 8 16
do
# purchase100
# variable-number
# nn
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=purchase100 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attack_from_cache --attacker_batch_size=32 --random_num_label --prefix=exp2
# nn-single
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=purchase100 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attack_from_cache --attacker_batch_size=32 --single_model --random_num_label --prefix=exp2
# clustering
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=purchase100 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label  --attack_from_cache --random_num_label --prefix=exp2
done

for num_of_label in 1 2 4 8 16
do
## cifar100
# variable-number
# nn
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attack_from_cache --attacker_batch_size=32 --random_num_label --prefix=exp2
# nn-single
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar100 --epochs=1 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=nn --num_of_label_k=$num_of_label --attack_from_cache --attacker_batch_size=32 --single_model --random_num_label --prefix=exp2
# clustering (Jac)
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar100 --epochs=1 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.1 --attack=clustering  --num_of_label_k=$num_of_label  --attack_from_cache --random_num_label --prefix=exp2
done
