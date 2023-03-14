#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9.1


import os
import time
import copy

from tensorboardX import SummaryWriter
import torch
import numpy as np

from rdp_accountant import compute_rdp, get_privacy_spent
from utils import save_result, cache_line_protection, get_buffer_names, get_dataset, get_learnable_parameters, zero_except_top_k_weights, recover_flattened, encrypt_parameters, serialize_dense, serialize_sparse, count_parameters, index_privacy, k_anonymization
from update import LocalUpdate, l2clipping, test_inference, update_global_weights, client_level_dp_update_global_weights, diff_weights
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, MLPPurchase100, ResNetCifar
from attack import Attacker, run_attack
from option import args_parser, get_aggregation_alg_code, exp_details
from sampling import client_iid
from proto_client import call_grpc_start, call_grpc_aggregate

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    start_time = time.time()

    path_project = os.path.abspath('.')
    logger = SummaryWriter(os.path.join(path_project, 'log'))

    args = args_parser()
    if args.verbose:
        exp_details(args)

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu_id else 'cpu'
    
    is_secure_agg = bool(args.secure_agg)
    fl_id = args.fl_id

    np.random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)
    rs_for_index_privacy = np.random.RandomState(args.seed + 3)
    rs_for_store_teacher_indices = np.random.RandomState(args.seed + 4)
    rs_for_gaussian_noise = np.random.RandomState(args.seed + 5)
    rs_for_attacker_data = np.random.RandomState(args.seed + 6)

    train_dataset, test_dataset, user_groups, class_labels = get_dataset(
        args, path_project, args.num_of_label_k, args.random_num_label)

    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar10':
            global_model = CNNCifar(args.num_classes)
        elif args.dataset == 'cifar100':
            global_model = ResNetCifar(args.num_classes)
        else:
            exit('Error: no dataset')

    elif args.model == 'mlp':
        if args.dataset == 'purchase100':
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLPPurchase100(
                dim_in=len_in,
                dim_hidden=64,
                dim_out=args.num_classes)
        else:
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLP(
                dim_in=len_in,
                dim_hidden=64,
                dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    if args.verbose:
        print(global_model)

    # copy weights
    global_weights = global_model.state_dict()
    num_of_params = count_parameters(global_model)
    if args.alpha:
        num_of_sparse_params = int(args.alpha * num_of_params)
    else:
        num_of_sparse_params = 0
    print(
        f'Parameter size --- {num_of_params} (alpha={args.alpha})')
    buffer_names = get_buffer_names(global_model)

    # Training
    train_loss, train_accuracy = [], []
    test_loss_list = []
    print_every = 20 # print training accuracy for each {print_every} epochs

    # Target clients
    target_client_ids = list(range(0, args.num_users))
    target_client_labels = {}
    for target_client_id in target_client_ids:
        target_user_label = set()
        for idx in user_groups[target_client_id]:
            img, label = train_dataset[int(idx)]
            target_user_label.add(label)
        target_client_labels[target_client_id] = target_user_label

    # Evaluate attack from pickled training data
    if (not is_secure_agg) and args.attack_from_cache:
        attacker = Attacker.load_from_pickle(path_project, args)
        if attacker:
            attacker.verbose = args.verbose
            run_attack(path_project, args.prefix, attacker, target_client_ids, target_client_labels, args)
            print(
                '\n Total Run Time: {0:0.4f}'.format(
                    time.time() - start_time))
            exit(0)
        else:
            print('train start')

    # Initialize attacker
    if not is_secure_agg and not args.no_attack:
        if args.attacker_data_size:
            if args.dataset in ['mnist']:
                attacker_dataset = Attacker.attacker_data_sample_mnist(test_dataset, args.attacker_data_size, rs_for_attacker_data)
            elif args.dataset in ['purchase100']:
                attacker_dataset = Attacker.attacker_data_sample_purchase100(test_dataset, args.attacker_data_size, rs_for_attacker_data)
            else:
                exit('Error: dataset must be specified')
        else:
            attacker_dataset = test_dataset
        
        attacker = Attacker(
            args.epochs,
            num_of_params,
            num_of_sparse_params,
            target_client_ids,
            class_labels,
            args.local_bs,
            args.optimizer,
            args.lr,
            args.local_ep,
            args.momentum,
            device,
            args.verbose,
            args.attacker_batch_size,
            attacker_dataset,
            buffer_names)

    # Orders of RDP
    if args.dp:
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                  list(range(5, 64)) + [128, 256, 512])
        
    if is_secure_agg:
        fl_id, current_round, secure_sampled_client_ids = call_grpc_start(
            fl_id,
            target_client_ids,
            args.sigma,
            args.clipping,
            args.alpha,
            args.frac,
            get_aggregation_alg_code(args.aggregation_alg),
            num_of_params,
            num_of_sparse_params,
        )

    for epoch in range(args.epochs):
        print(f' | Global Training Round : {epoch + 1} |')

        local_weights_diffs, local_losses = [], []

        global_model.train()

        if is_secure_agg:
            idxs_users = secure_sampled_client_ids
        else:
            # choose client randomly for this round
            idxs_users = client_iid(args.frac, args.num_users)

        for idx in idxs_users:
            if args.local_skip:
                local_weights_diffs.append(global_weights)
                local_losses.append(0.0)
            else:
                local_model = LocalUpdate(
                    dataset=train_dataset,
                    idxs=user_groups[idx],
                    logger=logger,
                    device=device,
                    local_bs=args.local_bs,
                    optimizer=args.optimizer,
                    lr=args.lr,
                    local_ep=args.local_ep,
                    momentum=args.momentum,
                    verbose=args.verbose)

                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch
                )
                local_weights_diffs.append(diff_weights(global_weights, w))
                local_losses.append(copy.deepcopy(loss))

        # Use secure aggregation server using SGX see FL-TEE/secure_aggregation
        if is_secure_agg:
            encrypted_parameters = []
            for client_id, local_weights_diff in zip(idxs_users, local_weights_diffs):
                # to bytes
                if args.alpha:
                    top_k_local_weights_diff, top_k_indices = zero_except_top_k_weights(local_weights_diff, buffer_names, num_of_sparse_params)
                    if args.dp:
                        top_k_local_weights_diff = l2clipping(top_k_local_weights_diff, buffer_names, args.clipping)
                    if args.protection == 'index-privacy':
                        top_k_indices = index_privacy(
                            top_k_indices, num_of_params, rs_for_index_privacy, r=args.index_privacy_r)
                    bytes_local_weight = serialize_sparse(top_k_local_weights_diff, buffer_names, top_k_indices)
                else:
                    if args.dp:
                        local_weights_diff = l2clipping(local_weights_diff, buffer_names, args.clipping)
                    bytes_local_weight = serialize_dense(local_weights_diff, buffer_names, num_of_params)

                encrypted_local_weight = encrypt_parameters(bytes_local_weight, client_id)
                encrypted_parameters.extend(encrypted_local_weight)

            flattend_aggregated_weights, execution_time, secure_sampled_client_ids, _ = call_grpc_aggregate(
                fl_id,
                epoch,
                encrypted_parameters,
                num_of_params,
                num_of_sparse_params,
                idxs_users,
                get_aggregation_alg_code(args.aggregation_alg),
                args.optimal_num_of_clients
            )

            learnable_parameters = get_learnable_parameters(global_weights, buffer_names)
            aggregated_weights = recover_flattened(torch.Tensor(flattend_aggregated_weights), global_weights, learnable_parameters)
            update_global_weights(global_weights, [aggregated_weights])
            # print("Secure Aggregation execution time: ", execution_time)

        # Without secure aggregation
        else:
            if args.alpha:
                top_k_local_weights_diffs = []
                aggregated_top_k_indices = {}

                for client_id, local_weights_diff in zip(
                        idxs_users, local_weights_diffs):
                    top_k_local_weights_diff, top_k_indices = zero_except_top_k_weights(
                        local_weights_diff, buffer_names, num_of_sparse_params)
                    if args.dp:
                        top_k_local_weights_diff = l2clipping(top_k_local_weights_diff, buffer_names, args.clipping)
                    top_k_local_weights_diffs.append(top_k_local_weights_diff)

                    if client_id in target_client_ids:
                        aggregated_top_k_indices[client_id] = top_k_indices

                # Protections
                if args.protection == 'k-anonymization':
                    anonymize_k = 5
                    concatenated = []
                    for v in aggregated_top_k_indices.values():
                        concatenated.extend(v)
                    key, cnt = np.unique(concatenated, return_counts=True)
                    over_k_indices = set(key[cnt >= anonymize_k])

                for client_id, indices in aggregated_top_k_indices.items():
                    if args.protection == 'index-privacy':
                        top_k_indices = index_privacy(
                            indices, num_of_params, rs_for_index_privacy, r=args.index_privacy_r)
                    elif args.protection == 'cacheline':
                        top_k_indices = cache_line_protection(indices)
                    elif args.protection == 'k-anonymization':
                        top_k_indices = k_anonymization(
                            indices, over_k_indices)
                    else:
                        top_k_indices = indices
                    if not args.no_attack:
                        attacker.store_target_indices(client_id, epoch, top_k_indices)

                local_weights_diffs = top_k_local_weights_diffs
            if args.dp:
                client_level_dp_update_global_weights(global_weights, local_weights_diffs, args.sigma, args.clipping, args.alpha, rs_for_gaussian_noise)
            else:
                update_global_weights(global_weights, local_weights_diffs)

        if not is_secure_agg and not args.no_attack:
            # Observe top-k indices from global model of this round using test data
            attacker.store_teacher_indices(
                epoch,
                global_model,
                rs_for_store_teacher_indices,
                args.index_privacy_r,
                args.protection)
            
        if not args.local_skip:
            # Update global model
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            if (epoch + 1) % print_every == 0:
                # Calculate avg training accuracy over all users at every epoch
                list_acc, list_loss = [], []
                global_model.eval()

                for idx in range(args.num_users):
                    local_model = LocalUpdate(
                        dataset=train_dataset,
                        idxs=user_groups[idx],
                        logger=logger,
                        device=device,
                        local_bs=args.local_bs,
                        optimizer=args.optimizer,
                        lr=args.lr,
                        local_ep=args.local_ep,
                        momentum=args.momentum,
                        verbose=args.verbose)
                    acc, loss = local_model.inference(model=global_model)
                    list_acc.append(acc)
                    list_loss.append(loss)
                    train_accuracy.append(sum(list_acc) / len(list_acc))

                # print global training loss after every 'i' rounds
                print(f' Avg Training Stats after {epoch+1} global rounds:')
                print(f'    Avg Training Loss: {np.mean(np.array(train_loss))}')
                print('    Avg Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        if not args.local_skip:
            # Test inference after completion of training
            test_acc, test_loss = test_inference(global_model, test_dataset, device)
            test_loss_list.append(test_loss)

            print(f" \n Results after {epoch+1} ({args.epochs}) global rounds of training:")
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
            print("|---- Test Loss: {:.8f}".format(test_loss))
            if args.dp:
                rdp = compute_rdp(args.frac, args.sigma, epoch + 1, orders)
                eps_spent, delta_spent, opt_order = get_privacy_spent(
                    orders, rdp, target_delta=args.delta
                )
                print(
                    "|---- Central DP : ({:.6f}, {:.6f})-DP".format(eps_spent, delta_spent)
                )
                if eps_spent > args.epsilon or delta_spent > args.delta:
                    print("|----  ######## Excess setted privacy budget ########")

    # Attack inference
    if not is_secure_agg and not args.no_attack:
        run_attack(path_project, args.prefix, attacker, target_client_ids, target_client_labels, args)
        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
        attacker.save_pickle(path_project, args)

    if is_secure_agg:
        if args.prefix in ['exp5']:
            save_result(path_project, args.prefix,
                        [
                            args.dataset,
                            args.epochs,
                            args.frac,
                            args.num_users,
                            args.num_of_label_k,
                            args.random_num_label,
                            args.model,
                            args.alpha,
                            args.seed,
                            args.aggregation_alg,
                            args.protection,
                            args.index_privacy_r,
                            execution_time
                        ],
                        add=True
            )
    if args.no_attack:
        if args.prefix in ['exp9']:
                save_result(path_project, args.prefix,
                    [
                        args.dataset,
                        args.epochs,
                        args.frac,
                        args.num_users,
                        args.num_of_label_k,
                        args.random_num_label,
                        args.model,
                        args.alpha,
                        args.attack,
                        args.fixed_inference_number,
                        args.single_model,
                        args.attacker_batch_size,
                        args.seed,
                        args.protection,
                        args.index_privacy_r,
                        args.dp,
                        args.epsilon,
                        args.delta,
                        args.sigma,
                        test_acc,
                        test_loss_list,
                    ],
                        add=True
            )
