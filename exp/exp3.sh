#! /bin/bash
set -ex

# Attack performance for each sparse ratio

for alpha in 0.0125 0.025 0.05 0.1 0.2 0.4 0.6 0.8
do
# mnist
# fixed-number
# nn
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=$alpha --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --attack_from_cache --prefix=exp3
# nn-single
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=$alpha --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --single_model --attack_from_cache  --prefix=exp3
# clustering (Jac)
python src/fl_main.py --model=mlp --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=mnist --epochs=3 --seed=0 --frac=0.1 --num_users=1000 --data_dist=non-IID --optimizer=sgd --alpha=$alpha --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --prefix=exp3
done

for alpha in 0.003125 0.00625 0.0125 0.025 0.05 0.1 0.2 0.4 0.6 0.8
do
## cifar100
# fixed-number
# nn if epochs set 1, nn is simlar to nn-single
# nn-single
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar100 --epochs=1 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=$alpha --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp3
# clustering (Jac)
python src/fl_main.py --model=cnn --dp --sigma=1.12 --epsilon=5.0 --delta=0.1 --dataset=cifar100 --epochs=1 --seed=0 --frac=0.1 --num_users=1000 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=$alpha --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --prefix=exp3
done
