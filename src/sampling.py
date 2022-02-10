import numpy as np


def mnist_iid(dataset, num_users):
    """
    Sample IID client data from MNIST dataset
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(
                all_idxs,
                num_items,
                replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users, num_of_label_k, is_random_num_label):
    """
    Sample non-I.I.D client data from MNIST dataset
    """
    # 60,000 training imgs
    data_size = len(dataset)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(data_size)
    labels = dataset.train_labels.numpy()
    unique_labels = np.unique(labels)
    if is_random_num_label:
        k_list = np.random.randint(1, num_of_label_k + 1, size=num_users)
        shard_size = int(data_size / k_list.sum())
    else:
        shard_size = int(data_size / (num_of_label_k * num_users))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide idxs by label
    cursor_by_label = {}
    idxs_by_label = {}
    last_idx = 0
    last_label = unique_labels[0]
    for label in unique_labels[1:]:
        first_idx = np.where(idxs_labels[1, :] == label)[0][0]
        idxs_by_label[last_label] = idxs[last_idx:first_idx]
        cursor_by_label[last_label] = 0
        last_label = label
        last_idx = first_idx
    idxs_by_label[last_label] = idxs[last_idx:]
    cursor_by_label[last_label] = 0

    # shuffle
    for value in idxs_by_label.values():
        np.random.shuffle(value)

    # divide and assign k
    for i in range(num_users):
        if is_random_num_label:
            k = k_list[i]
        else:
            k = num_of_label_k
        if len(unique_labels) > k:
            rand_set = set(np.random.choice(unique_labels, k, replace=False))
        else:
            rand_set = set(unique_labels)
        for rand in rand_set:
            selected_data = idxs_by_label[rand][cursor_by_label[rand]:cursor_by_label[rand] + shard_size]
            if len(selected_data) < shard_size:
                if len(selected_data) > shard_size * 0.5:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], selected_data), axis=0)
                    unique_labels = unique_labels[unique_labels != rand]
                else:
                    done = False
                    for label in unique_labels:
                        selected_data_retry = idxs_by_label[label][cursor_by_label[label]:cursor_by_label[label] + shard_size]
                        if len(selected_data_retry) == shard_size:
                            dict_users[i] = np.concatenate(
                                (dict_users[i], selected_data_retry), axis=0)
                            cursor_by_label[label] += shard_size
                            done = True
                            break
                        elif len(selected_data_retry) > shard_size * 0.5:
                            dict_users[i] = np.concatenate(
                                (dict_users[i], selected_data_retry), axis=0)
                            unique_labels = unique_labels[unique_labels != label]
                            done = True
                            break
                    if not done:
                        dict_users[i] = np.concatenate(
                            (dict_users[i], selected_data), axis=0)
                        unique_labels = unique_labels[unique_labels != rand]
            else:
                dict_users[i] = np.concatenate(
                    (dict_users[i], selected_data), axis=0)
                cursor_by_label[rand] += shard_size
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idxs_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idxs_shard, 1, replace=False))
            idxs_shard = list(set(idxs_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idxs_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idxs_shard):
                shard_size = len(idxs_shard)
            rand_set = set(np.random.choice(idxs_shard, shard_size,
                                            replace=False))
            idxs_shard = list(set(idxs_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idxs_shard, shard_size,
                                            replace=False))
            idxs_shard = list(set(idxs_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
        if len(idxs_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idxs_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idxs_shard, shard_size,
                                            replace=False))
            idxs_shard = list(set(idxs_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample IID client data from CIFAR10 dataset
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(
                all_idxs,
                num_items,
                replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users, num_of_label_k, is_random_num_label):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    """
    data_size = len(dataset)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(data_size)
    labels = np.array(dataset.targets)
    unique_labels = np.unique(labels)
    if is_random_num_label:
        k_list = np.random.randint(1, num_of_label_k + 1, size=num_users)
        shard_size = int(data_size / k_list.sum())
    else:
        shard_size = int(data_size / (num_of_label_k * num_users))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide idxs by label
    cursor_by_label = {}
    idxs_by_label = {}
    last_idx = 0
    last_label = unique_labels[0]
    for label in unique_labels[1:]:
        first_idx = np.where(idxs_labels[1, :] == label)[0][0]
        idxs_by_label[last_label] = idxs[last_idx:first_idx]
        cursor_by_label[last_label] = 0
        last_label = label
        last_idx = first_idx
    idxs_by_label[last_label] = idxs[last_idx:]
    cursor_by_label[last_label] = 0

    # shuffle
    for value in idxs_by_label.values():
        np.random.shuffle(value)

    # divide and assign k
    for i in range(num_users):
        if is_random_num_label:
            k = k_list[i]
        else:
            k = num_of_label_k
        if len(unique_labels) > k:
            rand_set = set(np.random.choice(unique_labels, k, replace=False))
        else:
            rand_set = set(unique_labels)
        for rand in rand_set:
            selected_data = idxs_by_label[rand][cursor_by_label[rand]:cursor_by_label[rand] + shard_size]
            if len(selected_data) < shard_size:
                if len(selected_data) > shard_size * 0.5:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], selected_data), axis=0)
                    unique_labels = unique_labels[unique_labels != rand]
                else:
                    done = False
                    for label in unique_labels:
                        selected_data_retry = idxs_by_label[label][cursor_by_label[label]:cursor_by_label[label] + shard_size]
                        if len(selected_data_retry) == shard_size:
                            dict_users[i] = np.concatenate(
                                (dict_users[i], selected_data_retry), axis=0)
                            cursor_by_label[label] += shard_size
                            done = True
                            break
                        elif len(selected_data_retry) > shard_size * 0.5:
                            dict_users[i] = np.concatenate(
                                (dict_users[i], selected_data_retry), axis=0)
                            unique_labels = unique_labels[unique_labels != label]
                            done = True
                            break
                    if not done:
                        dict_users[i] = np.concatenate(
                            (dict_users[i], selected_data), axis=0)
                        unique_labels = unique_labels[unique_labels != rand]
            else:
                dict_users[i] = np.concatenate(
                    (dict_users[i], selected_data), axis=0)
                cursor_by_label[rand] += shard_size
    return dict_users


def purchase100_iid(dataset, num_users):
    """
    Sample IID client data from Purchase100 dataset
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(
                all_idxs,
                num_items,
                replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def purchase100_noniid(dataset, num_users, num_of_label_k, is_random_num_label):
    """
    Sample non-I.I.D client data from Purchase100 dataset
    """
    data_size = len(dataset)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(data_size)
    labels = [label for _, label in dataset]
    unique_labels = np.unique(labels)
    if is_random_num_label:
        k_list = np.random.randint(1, num_of_label_k + 1, size=num_users)
        shard_size = int(data_size / k_list.sum())
    else:
        shard_size = int(data_size / (num_of_label_k * num_users))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide idxs by label
    cursor_by_label = {}
    idxs_by_label = {}
    last_idx = 0
    last_label = unique_labels[0]
    for label in unique_labels[1:]:
        first_idx = np.where(idxs_labels[1, :] == label)[0][0]
        idxs_by_label[last_label] = idxs[last_idx:first_idx]
        cursor_by_label[last_label] = 0
        last_label = label
        last_idx = first_idx
    idxs_by_label[last_label] = idxs[last_idx:]
    cursor_by_label[last_label] = 0

    # shuffle
    for value in idxs_by_label.values():
        np.random.shuffle(value)

    # divide and assign k
    for i in range(num_users):
        if is_random_num_label:
            k = k_list[i]
        else:
            k = num_of_label_k
        if len(unique_labels) > k:
            rand_set = set(np.random.choice(unique_labels, k, replace=False))
        else:
            rand_set = set(unique_labels)
        for rand in rand_set:
            selected_data = idxs_by_label[rand][cursor_by_label[rand]:cursor_by_label[rand] + shard_size]
            if len(selected_data) < shard_size:
                if len(selected_data) > shard_size * 0.5:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], selected_data), axis=0)
                    unique_labels = unique_labels[unique_labels != rand]
                else:
                    done = False
                    for label in unique_labels:
                        selected_data_retry = idxs_by_label[label][cursor_by_label[label]:cursor_by_label[label] + shard_size]
                        if len(selected_data_retry) == shard_size:
                            dict_users[i] = np.concatenate(
                                (dict_users[i], selected_data_retry), axis=0)
                            cursor_by_label[label] += shard_size
                            done = True
                            break
                        elif len(selected_data_retry) > shard_size * 0.5:
                            dict_users[i] = np.concatenate(
                                (dict_users[i], selected_data_retry), axis=0)
                            unique_labels = unique_labels[unique_labels != label]
                            done = True
                            break
                    if not done:
                        dict_users[i] = np.concatenate(
                            (dict_users[i], selected_data), axis=0)
                        unique_labels = unique_labels[unique_labels != rand]
            else:
                dict_users[i] = np.concatenate(
                    (dict_users[i], selected_data), axis=0)
                cursor_by_label[rand] += shard_size
    return dict_users


def client_iid(frac, num_users):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    return idxs_users
