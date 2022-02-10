import os
import pickle
import struct
import ctypes
import csv
import copy
import datetime
from collections import OrderedDict

import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, purchase100_iid, purchase100_noniid

DATA_SET_DIR = 'dataset'
RESULTS_DIR = 'exp/results'
LIB_CRYPTO = ctypes.cdll.LoadLibrary('src/libsgx_enc.so')
COUNTER_LEN = 16


def get_dataset(args, path_project, num_of_label_k, is_random_num_label):
    if args.dataset == 'cifar10':
        data_dir = os.path.join(path_project, DATA_SET_DIR, 'cifar10')
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        train_dataset = datasets.CIFAR10(
            data_dir,
            train=True,
            download=True,
            transform=apply_transform)
        test_dataset = datasets.CIFAR10(
            data_dir,
            train=False,
            download=True,
            transform=apply_transform)

        if args.data_dist == 'IID':
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:  # args.data_dist == 'non-IID':
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups = cifar_noniid(train_dataset, args.num_users, num_of_label_k, is_random_num_label)
        class_labels = set(test_dataset.class_to_idx.values())

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        data_dir = os.path.join(path_project, DATA_SET_DIR, args.dataset)
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=apply_transform)
        test_dataset = datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=apply_transform)

        if args.data_dist == 'IID':
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:  # args.data_dist == 'non-IID':
            if args.unequal:
                user_groups = mnist_noniid_unequal(
                    train_dataset, args.num_users)
            else:
                user_groups = mnist_noniid(train_dataset, args.num_users, num_of_label_k, is_random_num_label)
        class_labels = set(test_dataset.train_labels.numpy())

    elif args.dataset == 'cifar100':
        data_dir = os.path.join(path_project, DATA_SET_DIR, 'cifar100')
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)

        if args.data_dist == 'IID':
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:  # args.data_dist == 'non-IID':
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups = cifar_noniid(train_dataset, args.num_users, num_of_label_k, is_random_num_label)
        class_labels = set(test_dataset.class_to_idx.values())
        
    elif args.dataset == 'purchase100':
        data_dir = os.path.join(path_project, DATA_SET_DIR, 'purchase100')
        save_purchase100(120000, 0.2, data_dir)
        train_dataset, test_dataset = load_purchase100(120000, 0.2, data_dir)

        if args.data_dist == 'IID':
            user_groups = purchase100_iid(train_dataset, args.num_users)
        else:  # args.data_dist == 'non-IID':
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups = purchase100_noniid(train_dataset, args.num_users, num_of_label_k, is_random_num_label)
        class_labels = set([label for _, label in test_dataset])
        
    else:
        exit('Error: unrecognized model')

    return train_dataset, test_dataset, user_groups, class_labels


def save_purchase100(target_size, target_test_train_ratio, data_dir, seed=0, force_update=False):
    if force_update or os.path.exists(data_dir + '/target_data.npz'):
        print("data already prepared")
        return

    print('-' * 10 + 'Saving purchase100 data' + '-' * 10 + '\n')
    gamma = target_test_train_ratio

    x = pickle.load(open(data_dir + '/purchase_100_features.p', 'rb'))
    y = pickle.load(open(data_dir + '/purchase_100_labels.p', 'rb'))
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(x.shape, y.shape)

    # assert if data is enough for sampling target data
    assert(len(x) >= (1 + gamma) * target_size)
    x, train_x, y, train_y = train_test_split(x, y, test_size=target_size, stratify=y, random_state=seed)
    print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
    x, test_x, y, test_y = train_test_split(x, y, test_size=int(gamma*target_size), stratify=y, random_state=seed+1)
    print("Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))

    # save target data
    np.savez(data_dir + '/target_data.npz', train_x, train_y, test_x, test_y)


def load_purchase100(target_size, target_test_train_ratio, data_dir):
    gamma = target_test_train_ratio
    with np.load(data_dir + '/target_data.npz') as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]

    train_x = np.array(train_x, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.int32)
    test_y = np.array(test_y, dtype=np.int32)

    train_dataset = [(feature, np.int64(label)) for feature, label in zip(train_x, train_y)] 
    test_dataset = [(feature, np.int64(label)) for feature, label in zip(test_x[:int(gamma*target_size)], test_y[:int(gamma*target_size)])] 
    return train_dataset, test_dataset


def serialize_dense(state_dict, buffer_names, d):
    """
    Args:
        state_dict: OrderedDict
            ex. model.state_dict()
        d: int
            how many parameters are in original model
    Returns:
        bytes_buffer: bytes
            bytes format is "Index (4bytes unsigned int) Value (4bytes float)"
    """
    learnable_parameters = get_learnable_parameters(state_dict, buffer_names)
    tensor_flat_params = flatten_params(learnable_parameters)
    float_flat_params = tensor_flat_params.tolist()
    unpakced_flat_params = [element for tupl in enumerate(
        float_flat_params) for element in tupl]
    bytes_buffer = struct.pack(
        d * 'If',
        *unpakced_flat_params)
    return bytes_buffer


def serialize_sparse(state_dict, buffer_names, top_k_indices):
    """
    Args:
        state_dict: OrderedDict
            ex. model.state_dict()
        top_k_indices: [int]
            top-k indices
    Returns:
        bytes_buffer: bytes
            bytes format is "Index (4bytes unsigned int) Value (4bytes float)"
        indices: [(int, int)]
    """
    learnable_parameters = get_learnable_parameters(state_dict, buffer_names)
    tensor_flat_params = flatten_params(learnable_parameters)
    unpakced_flat_params = [elm for tupl in zip(top_k_indices, tensor_flat_params[top_k_indices]) for elm in tupl]
    bytes_buffer = struct.pack(len(top_k_indices) * 'If', *unpakced_flat_params)
    return bytes_buffer


def flatten_params(learnable_parameters):
    """
    Args:
        learnable_parameters (OrderedDict): parameters without buffers (such as bn.running_mean)
    Returns:
        flat (torch.Tensor):
            whose dim is one, like [0.1, ..., 0.2]

    """
    ir = [torch.flatten(p) for _, p in learnable_parameters.items()]
    flat = torch.cat(ir).view(-1, 1).flatten()
    return flat


def get_index_ranges(learnable_parameters):
    """
    Args:
        learnable_parameters (OrderedDict): parameters without buffers (such as bn.running_mean)
    Returns:
        indices: [(int, int)]
            [(start, end)]
    """
    index_ranges = []
    s = 0
    for _, p in learnable_parameters.items():
        size = torch.flatten(p).shape[0]
        index_ranges.append((s, s + size))
        s += size
    return index_ranges


def recover_flattened(flat_params, base_state_dict, learnable_parameters):
    """
    Args:
        flat_params (torch.Tensor):
            whose dim is one, like [0.1, ..., 0.2]
        base_state_dict (OrderedDict)
            ex. model.state_dict():
            buffers are inherent
        learnable_parameters (OrderedDict):
            parameters without buffers (such as bn.running_mean)
    Returns:
        new_state: OrderedDict
            ex. model.state_dict()
    """
    index_ranges = get_index_ranges(learnable_parameters)
    ir = [flat_params[s:e] for (s, e) in index_ranges]
    new_state = copy.deepcopy(base_state_dict)
    for flat, (key, value) in zip(ir, learnable_parameters.items()):
        if len(value.shape) == 0:
            new_state[key] = flat[0]
        else:
            new_state[key] = flat.view(*value.shape)
    return new_state


def encrypt_parameters(bytes_local_weight, client_id):
    """
    Args:
        bytes_local_weight: bytes
        client_id: int
    Returns:
        : [uint8]
    """
    key = [0] * COUNTER_LEN
    client_id_bytes = list(int(client_id).to_bytes(2, 'big'))
    key[8 - len(client_id_bytes):8] = client_id_bytes
    p_key = (ctypes.c_uint8 * len(key))(*key)

    src_len = len(bytes_local_weight)
    src_buffer = (ctypes.c_uint8 * src_len)(*bytes_local_weight)
    p_src = ctypes.cast(src_buffer, ctypes.POINTER(ctypes.c_uint8))

    counter_block = [0] * COUNTER_LEN
    p_ctr = (ctypes.c_uint8 * COUNTER_LEN)(*counter_block)

    ctr_inc_bits = 128

    dst_buffer = (ctypes.c_uint8 * src_len)()
    p_dst = ctypes.cast(dst_buffer, ctypes.POINTER(ctypes.c_uint8))

    rc = LIB_CRYPTO.sgx_aes_ctr_encrypt(
        p_key,
        p_src,
        src_len,
        p_ctr,
        ctr_inc_bits,
        p_dst
    )
    if rc != 0:
        raise RuntimeError

    return [p_dst[i] for i in range(src_len)]


def count_parameters(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params


def get_buffer_names(model):
    return [name for name, _ in model.named_buffers()]


def get_learnable_parameters(state_dict, buffer_names):
    learnable_parameters = OrderedDict()
    for key, value in state_dict.items():
        if key not in buffer_names:
            learnable_parameters[key] = value
    return learnable_parameters


def zero_except_top_k_weights(state_dict, buffer_names, k):
    """Given dense weights and set all parameters except top-k to zero.
    
    Args:
        state_dict: OrderedDict
            ex. model.state_dict()

    Returns:
        new_state: OrderedDict
            ex. model.state_dict()
        top_k_indices: [int]
            indices of top-k parameters
    """
    learnable_parameters = get_learnable_parameters(state_dict, buffer_names)
    tensor_flat_params = flatten_params(learnable_parameters)
    float_flat_params = tensor_flat_params.tolist()
    # convert dense weights to sparse form
    float_flat_sparse_params = [(idx, val)
                                for idx, val in enumerate(float_flat_params)]
    float_flat_sparse_params.sort(key=lambda x: abs(x[1]), reverse=True)
    top_k_float_flat_sparse_params = [0.0] * len(float_flat_sparse_params)
    top_k_indices = []
    for i in range(k):
        idx, val = float_flat_sparse_params[i]
        top_k_float_flat_sparse_params[idx] = val
        top_k_indices.append(idx)

    return recover_flattened(torch.Tensor(top_k_float_flat_sparse_params), state_dict, learnable_parameters), top_k_indices


def index_privacy(top_k_indices, num_of_params, random_state, r):
    sampled_candidates = np.array(list(set(range(num_of_params)) - set(top_k_indices)))
    randomized_indices = list(np.concatenate((random_state.choice(sampled_candidates, size=int(len(top_k_indices) * r), replace=False), np.array(top_k_indices))))
    random_state.shuffle(randomized_indices)
    return randomized_indices


def k_anonymization(top_k_indices, over_k_indices):
    return list(over_k_indices.intersection(set(top_k_indices)))


def cache_line_protection(top_k_indices, cache_line_size=64, idx_byte_size=8):
    div = int(cache_line_size / idx_byte_size)
    return [idx // div for idx in top_k_indices]


def save_result(path_project, prefix, result_arr, add=False):
    """
    Args:
        result_arr []
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S') 
    result_arr.append(timestamp)
    results_dir = os.path.join(path_project, RESULTS_DIR)
    if add:
        option = 'a'
    else:
        option = 'w'
    with open(results_dir + '/' + prefix + '.csv', option) as f:
        writer = csv.writer(f)
        writer.writerow(result_arr)
