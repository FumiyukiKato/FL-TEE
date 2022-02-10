#! /bin/bash
set -ex

## Protection

# # index-privacy
# for r in 0.25 0.5 1.0 2.0 4.0 8.0
# do
# ## mnist
# # fixed-number
# # nn
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --attack_from_cache --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# # nn-batch
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --single_model --attack_from_cache  --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# # clustering
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# done

# for r in 0.25 0.5 1.0 2.0 4.0 8.0
for r in 16.0 32.0 64.0
do
## cifar10
# fixed-number, mlp
# nn
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# nn-batch
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# clustering
python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
done

# for r in 0.25 0.5 1.0 2.0 4.0 8.0
# do
# ## cifar10
# # fixed-number, cnn
# # nn
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# # nn-batch
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# # clustering
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# done

# for r in 0.25 0.5 1.0 2.0 4.0 8.0
# do
# ## cifar100
# # fixed-number
# # nn
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# # nn-batch
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# # clustering
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# done

# for r in 0.25 0.5 1.0 2.0 4.0 8.0
# do
# ## purchase100
# # fixed-number
# # nn
# python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# # nn-batch
# python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# # clustering
# python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=index-privacy --index_privacy_r=$r
# done


# # cacheline protection
# ## mnist
# # fixed-number
# # nn
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --attack_from_cache --prefix=exp6 --protection=cacheline
# # nn-batch
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --single_model --attack_from_cache  --prefix=exp6 --protection=cacheline
# # clustering
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=cacheline

# ## cifar10
# # fixed-number, mlp
# # nn
# python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --protection=cacheline
# # nn-batch
# python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --protection=cacheline
# # clustering
# python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=cacheline

# ## cifar10
# # fixed-number, cnn
# # nn
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --protection=cacheline
# # nn-batch
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --protection=cacheline
# # clustering
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=cacheline

# ## cifar100
# # fixed-number
# # nn
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --protection=cacheline
# # nn-batch
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --protection=cacheline
# # clustering
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=cacheline

# ## purchase100
# # fixed-number
# # nn
# python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --protection=cacheline
# # nn-batch
# python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --protection=cacheline
# # clustering
# python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --protection=cacheline


# DP
# DP is not related because adversary can access global model
## mnist
# nn
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --attack_from_cache --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # nn-batch
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attacker_batch_size=32 --single_model --attack_from_cache  --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # clustering
# python src/fl_main.py --model=mlp --dataset=mnist --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001

# for r in 0.25 0.5 1.0 2.0 4.0 8.0
# do
# ## cifar10
# # fixed-number, mlp
# # nn
# python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # nn-batch
# python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # clustering
# python src/fl_main.py --model=mlp --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# done

# for r in 0.25 0.5 1.0 2.0 4.0 8.0
# do
# ## cifar10
# # fixed-number, cnn
# # nn
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # nn-batch
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # clustering
# python src/fl_main.py --model=cnn --dataset=cifar10 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# done

# for r in 0.25 0.5 1.0 2.0 4.0 8.0
# do
# ## cifar100
# # fixed-number
# # nn
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # nn-batch
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # clustering
# python src/fl_main.py --model=cnn --dataset=cifar100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# done

# for r in 0.25 0.5 1.0 2.0 4.0 8.0
# do
# ## purchase100
# # fixed-number
# # nn
# python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # nn-batch
# python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=nn --num_of_label_k=2 --fixed_inference_number=2 --attack_from_cache --attacker_batch_size=32 --single_model --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# # clustering
# python src/fl_main.py --model=mlp --dataset=purchase100 --epochs=3 --seed=0 --frac=0.3 --num_users=100 --num_classes=100 --data_dist=non-IID --optimizer=sgd --alpha=0.01 --attack=clustering  --num_of_label_k=2 --fixed_inference_number=2  --attack_from_cache --prefix=exp6 --dp --epsilon=3.0 --delta=0.0001
# done