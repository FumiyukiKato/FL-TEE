import copy
import numpy as np
import torch
import pickle
import hashlib
import os

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import kmeans1d

from utils import cache_line_protection, zero_except_top_k_weights, save_result
from update import diff_weights


class Attacker(object):
    def __init__(
        self,
        rounds,
        num_of_params,
        num_of_sparse_params,
        target_client_ids,
        class_labels,
        local_bs,
        optimizer,
        lr,
        local_ep,
        momentum,
        device,
        verbose,
        batch_size,
        test_dataset,
        buffer_names
    ):
        self.rounds = rounds
        self.num_of_params = num_of_params
        self.num_of_sparse_params = num_of_sparse_params
        self.target_client_ids = target_client_ids
        self.class_labels = class_labels
        self.target_indices = {}
        for client_id in self.target_client_ids:
            self.target_indices[client_id] = {}
        self.teacher_indices = {}
        self.teacher_data = {}
        for label in class_labels:
            self.teacher_data[label] = []
        for image, label in test_dataset:
            self.teacher_data[label].append((image, label))
        self.local_bs = local_bs
        self.optimizer = optimizer
        self.lr = lr
        self.local_ep = local_ep
        self.momentum = momentum
        self.device = device
        self.verbose = verbose
        self.batch_size = batch_size
        self.buffer_names = buffer_names

    @staticmethod
    def make_pickle_file_path(path_project, args):
        args_copy = copy.deepcopy(args)
        # fix not influence the attacker instance
        args_copy.attack_from_cache = None
        args_copy.attack = None
        args_copy.single_model = None
        args_copy.fixed_inference_number = None
        args_copy.secure_agg = None
        args_copy.verbose = None
        args_copy.gpu_id = None
        args_copy.per_round = None
        args_copy.prefix = None

        file_name = 'attacker-' + \
            hashlib.md5(str(args_copy).encode()).hexdigest() + '.pickle'
        file_path = os.path.join(path_project, 'save', 'objects', file_name)
        return file_path

    def save_pickle(self, path_project, args):
        file_path = Attacker.make_pickle_file_path(path_project, args)
        print(f'save to {file_path}')
        with open(os.path.join(file_path), mode="wb") as f:
            pickle.dump(self, f)
        print('save done.')

    @staticmethod
    def load_from_pickle(path_project, args):
        file_path = Attacker.make_pickle_file_path(path_project, args)
        print(f'load from {file_path}')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                attacker = pickle.load(f)
            print('load done.')
            return attacker
        else:
            print(f'{file_path} not found')
            return None

    def get_participates_rounds(self, client_id):
        participated_rounds = []
        for round, _ in self.target_indices[client_id].items():
            participated_rounds.append(round)
        return participated_rounds

    def store_target_indices(self, client_id, round, indices):
        """For inference of the attack model
        """
        self.target_indices[client_id][round] = indices

    def store_teacher_indices(
            self,
            round,
            model,
            random_state,
            index_privacy_r=None,
            strategy=''):
        """For training of the attack model
        """
        for label in self.class_labels:
            self.teacher_indices[(label, round)] = []
            if self.batch_size:
                dataset = self.teacher_data[label]
                random_state.shuffle(dataset)
                dataset_size = int(len(dataset) / self.batch_size)
                for i in range(dataset_size):
                    indices = self.train_and_top_k_select(
                        dataset[i * self.batch_size:i * self.batch_size + self.batch_size], model, index_privacy_r)
                    if strategy == 'cacheline':
                        indices = cache_line_protection(indices)
                    self.teacher_indices[(label, round)].append(indices)
                if dataset_size * self.batch_size < len(dataset):
                    indices = self.train_and_top_k_select(
                        dataset[dataset_size * self.batch_size:], model, index_privacy_r)
                    if strategy == 'cacheline':
                        indices = cache_line_protection(indices)
                    self.teacher_indices[(label, round)].append(indices)
            else:
                dataset = self.teacher_data[label]
                indices = self.train_and_top_k_select(dataset, model, index_privacy_r)
                if strategy == 'cacheline':
                    indices = cache_line_protection(indices)
                self.teacher_indices[(label, round)].append(indices)

    def to_multi_hot_tensor(self, indices):
        """Convert sparse indices to multi-hot vector form
        """
        return torch.zeros(1, self.num_of_params).scatter_(
            1, torch.tensor(indices).unsqueeze(0), 1.).view(-1)

    def take_train_dataset(self, is_concate=False, sklearn=False, specified_round=None):
        """Prepare teacher indices for ML model training

        Args:
            is_concate: using single model over rounds or not
            sklearn: return data for scikit-learn model
            round: return given round data

        Return:
            train_data: [(indices, label)]
            train_data_by_rounds: {round: [(indices, label)]}
        """
        if is_concate:
            
            if sklearn:
                train_data = [[], []]
            else:
                train_data = []

            if specified_round:
                rounds = specified_round
            else:
                rounds = self.rounds

            for label in self.class_labels:
                concatenated_indices_list = [[] for _ in range(len(self.teacher_indices[(label, 0)]))]
                for round in range(rounds):
                    for concatenated_indices, indices in zip(
                            concatenated_indices_list, self.teacher_indices[(label, round)]):
                        concatenated_indices.extend(
                            self.to_multi_hot_tensor(indices).tolist())
                for concatenated_indices in concatenated_indices_list:
                    if sklearn:
                        train_data[0].append(np.array(concatenated_indices))
                        train_data[1].append(label)
                    else:
                        train_data.append((torch.tensor(concatenated_indices), label))
            if sklearn:
                train_data[0] = np.array(train_data[0])
                train_data[1] = np.array(train_data[1])

            return train_data
        else:
            train_data_by_rounds = {}
            for round in range(self.rounds):
                
                if sklearn:
                    train_data_by_rounds[round] = [[], []]
                else:
                    train_data_by_rounds[round] = []
                    
                for label in self.class_labels:
                    for indices in self.teacher_indices[(label, round)]:
                        if sklearn:
                            train_data_by_rounds[round][0].append(np.array(self.to_multi_hot_tensor(indices).tolist()))
                            train_data_by_rounds[round][1].append(label)
                        else:
                            train_data_by_rounds[round].append((self.to_multi_hot_tensor(indices), label))
                if sklearn:
                    train_data_by_rounds[round][0] = np.array(train_data_by_rounds[round][0])
                    train_data_by_rounds[round][1] = np.array(train_data_by_rounds[round][1])

            return train_data_by_rounds

    def take_target_dataset(self, client_id, is_concate=False):
        """Prepare target indices for model inference. one target indices is observed for one client

        Returns:
            indices
                single indices if is_concate = True, list of indices otherwise.
        """

        if is_concate:
            all_target_indices = []
            for indices in self.target_indices[client_id].values():
                all_target_indices.extend(indices)
            return all_target_indices.extend(indices)
        else:
            return list(self.target_indices[client_id].values())

    def class_label_inference(self, method, is_concate, fixed_inference_number, per_round):
        """Infer class labels for each client in target_client_ids

        Args:
            method (str, optional): [description]. Defaults to 'clustering'. or 'nn' Neural network based
            is_concate (bool)
            fixed_inference_number ([type], optional): [description]. Defaults to None. the fixed number of labels
            per_round (bool)

        Returns:
            result {int: ({int}, {int})): {client_id: ({all predicted label}, {top-1 label})}
                a tuple of (all inferred labels, top-1 label)
        """
        result = {}

        if method == 'clustering':
            for client_id in self.target_client_ids:
                if self.target_indices[client_id]:
                    result[client_id] = self.nearest_neighbor_inference(
                        client_id, fixed_inference_number, per_round)
                else:
                    result[client_id] = None
        elif method == 'nn':
            models = self.neural_network_train(is_concate=is_concate, per_round=per_round)
            for client_id in self.target_client_ids:
                if self.target_indices[client_id]:
                    result[client_id] = self.neural_network_inference(
                        client_id, models, is_concate, fixed_inference_number, per_round)
                else:
                    result[client_id] = None
        else:
            exit('invalid method')
        return result
    
    def from_df_to_score(self, decision):
        decision = decision.reshape(-1)
        votes = [(i if decision[p] > 0 else j) for p,(i,j) in enumerate((i,j)
                                            for i in range(len(self.class_labels))
                                            for j in range(i+1,len(self.class_labels)))]
        vote_scores = [votes.count(label) for label in self.class_labels]
        return vote_scores


    def neural_network_inference(self, client_id, models, is_concate, fixed_inference_number, per_round):
        """ Inference by given Neural Network

        Args:
            client_id (int): target client id
            models (pytorch model): trained model
            is_concate (bool): using single model over rounds or not
            fixed_inference_number (int): fixed the number of inferred label
            per_round (bool)

        Returns:
            ({all predicted label}, {top-1 label})
                a tuple of (all inferred labels, top-1 label) for given client id
        """
        for round, model in models.items():
            model.eval()
            
        def topk(fixed_inference_number, outputs_sum):
            if fixed_inference_number is None:
                clusters, centroids = kmeans1d.cluster(outputs_sum.view(-1), 2)
                result = set()
                bigger = 0
                if centroids[0] < centroids[1]:
                    bigger = 1
                for idx, label in zip(clusters, self.class_labels):
                    if idx == bigger:
                        result.add(label)
                return result
            else:
                sorted_outputs = torch.sort(
                    outputs_sum.view(-1), dim=0, descending=True)
                return set(sorted_outputs.indices[:fixed_inference_number].tolist())

        if is_concate:
            # concatenation and inference
            if per_round:
                results_per_round = {}
                for current_round in range(self.rounds):
                    is_emerge = False
                    for r in range(current_round+1):
                        if self.target_indices[client_id].get(r):
                            is_emerge = True
                    if not is_emerge:
                        continue
                    target_indices = []
                    for round in range(current_round+1):
                        if self.target_indices[client_id].get(round):
                            target_indices.extend(
                                self.to_multi_hot_tensor(
                                    self.target_indices[client_id][round]).tolist())
                        else:  # zero padding if there is no participation the round
                            target_indices.extend([0.] * self.num_of_params)
                    indices = torch.tensor(target_indices)
                    indices = indices.view(1, indices.shape[0])
                    indices = indices.to(self.device)
                    outputs = models[current_round](indices).detach()
                    # Inference
                    top1 = {torch.argmax(outputs).item()}
                    results_per_round[current_round] = (topk(fixed_inference_number, outputs), top1)
                return results_per_round
            else:
                model = models[0]
                target_indices = []
                for round in range(self.rounds):
                    if self.target_indices[client_id].get(round):
                        target_indices.extend(
                            self.to_multi_hot_tensor(
                                self.target_indices[client_id][round]).tolist())
                    else:  # zero padding if there is no participation the round
                        target_indices.extend([0] * self.num_of_params)
                indices = torch.tensor(target_indices)
                indices = indices.view(1, indices.shape[0])
                indices = indices.to(self.device)
                outputs_sum = model(indices)
        else:
            outputs_list = []
            if per_round:
                results_per_round = {}
            for round, indices in self.target_indices[client_id].items():
                indices = self.to_multi_hot_tensor(indices)
                indices = indices.view(1, indices.shape[0])
                indices = indices.to(self.device)
                outputs = models[round](indices).detach()
                # Inference
                if per_round:
                    top1 = {torch.argmax(outputs).item()}
                    results_per_round[round] = (topk(fixed_inference_number, outputs), top1)
                else:
                    outputs_list.append(outputs)
            if per_round:
                return results_per_round
            outputs_sum = sum(outputs_list)
            
        assert outputs_sum.shape == (
            1, len(self.class_labels)), "prediction score must be this shape"
        top1 = {torch.argmax(outputs_sum).item()}
        return topk(fixed_inference_number, outputs_sum), top1

    def neural_network_train(self, is_concate, per_round):
        """ Train attacker NN with teacher indices

        Args:
            is_concate: using single model over rounds or not
            per_round (bool): predict per round

        Returns:
            {int: model}: key=round, value=trained model, if concate, round=0
        """
        lr = 0.001
        momentum = 0.5
        epoch = 50
        if self.batch_size:
            batch_size = 32
        else:
            batch_size = 1

        if is_concate:
            models = {}
            if per_round:
                times = self.rounds
            else:
                times = 1
            for round in range(times):
                if per_round:
                    dim_in = self.num_of_params * (round+1)
                else:
                    dim_in = self.num_of_params * (self.rounds)
                dim_hidden = 2000
                model = AttackMLP(
                    dim_in=dim_in,
                    dim_hidden=dim_hidden,
                    dim_out=len(
                        self.class_labels))
                model.train()
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=lr, momentum=momentum)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[int(epoch * 0.7), int(epoch * 0.9)], gamma=1)
                criterion = nn.NLLLoss().to(self.device)
                if per_round:
                    train_dataset = self.take_train_dataset(is_concate=True, specified_round=round+1)
                else:
                    train_dataset = self.take_train_dataset(is_concate=True)
                trainloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True)

                for iter in range(epoch):
                    for indices, labels in trainloader:
                        indices, labels = indices.to(
                            self.device), labels.to(
                            self.device)
                        model.zero_grad()
                        probs = model(indices)
                        loss = criterion(probs, labels)
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                    if self.verbose:
                        print(
                            '| Epoch : {} | \tLoss: {:.6f}'.format(
                                iter, loss.item()))
                models[round] = model

            return models
        else:
            models = {}
            train_dataset_by_round = self.take_train_dataset(is_concate=False)
            dim_in = self.num_of_params
            dim_hidden = 1000
            for round in range(self.rounds):
                model = AttackMLP(
                    dim_in=dim_in,
                    dim_hidden=dim_hidden,
                    dim_out=len(
                        self.class_labels))
                model.train()
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=lr, momentum=momentum)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[int(epoch * 0.7), int(epoch * 0.9)], gamma=1)
                criterion = nn.NLLLoss().to(self.device)
                trainloader = DataLoader(
                    train_dataset_by_round[round],
                    batch_size=batch_size,
                    shuffle=True)
                for iter in range(epoch):
                    for indices, labels in trainloader:
                        indices, labels = indices.to(
                            self.device), labels.to(
                            self.device)
                        model.zero_grad()
                        probs = model(indices)
                        loss = criterion(probs, labels)
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                    if self.verbose:
                        print(
                            '| Round of model: {}, Epoch : {} | \tLoss: {:.6f}'.format(
                                round, iter, loss.item()))

                models[round] = model
            return models

    def nearest_neighbor_inference(self, client_id, fixed_inference_number, per_round):
        """ Inference by given Nearest Neighbor of jaccard similarity to teacher indices

        Args:
            client_id (int): target client id
            fixed_inference_number (int): fixed the number of inferred label
            per_round (bool)

        Returns:
            ({all predicted label}, {top-1 label})
                a tuple of (all inferred labels, top-1 label) for given client id
        """
        
        def top_selection(target_indices, teacher_indices, fixed_inference_number):
            score_list = []
            for label in self.class_labels:
                jaccard = calc_similarity('jaccard', target_indices, teacher_indices[label])
                if self.verbose:
                    print('label: ', label, ', jaccard=', jaccard)
                score_list.append(jaccard)
                
            sorted_score = sorted([(label, score) for label, score in zip(
                self.class_labels, score_list)], key=lambda x: x[1], reverse=True)
            top1 = {sorted_score[0][0]}
            
            if fixed_inference_number is None:
                clusters, centroids = kmeans1d.cluster(score_list, 2)
                result = set()
                bigger = 0
                if centroids[0] < centroids[1]:
                    bigger = 1
                for idx, label in zip(clusters, self.class_labels):
                    if idx == bigger:
                        result.add(label)
                return result, top1
            else:
                return {label for label, _ in sorted_score[:fixed_inference_number]}, top1
            
        if per_round:
            results_per_round = {}
            for round, target_indices_per_round in self.target_indices[client_id].items():
                teacher_indices = {}
                for label in self.class_labels:
                    teacher_indices[label] = self.teacher_indices[(label, round)][0] # clusteringの場合は教師データは１つしかない
                results_per_round[round] = top_selection(target_indices_per_round, teacher_indices, fixed_inference_number)
            return results_per_round
        else:
            all_target_indices = []
            participated_rounds = []
            for round, indices in self.target_indices[client_id].items():
                participated_rounds.append(round)
                all_target_indices.extend(indices)

            teacher_indices = {}
            for label in self.class_labels:
                teacher_indices[label] = []
                for round in participated_rounds:
                    assert len(self.teacher_indices[(
                        label, round)]) == 1, "if nearest neighbor method, teacher incides must be one"
                    teacher_indices[label].extend(
                        self.teacher_indices[(label, round)][0])
            return top_selection(all_target_indices, teacher_indices, fixed_inference_number)

    def train_and_top_k_select(self, test_dataset, model, index_privacy_r):
        model_copy = copy.deepcopy(model)
        model_copy.train()

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model_copy.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model_copy.parameters(), lr=self.lr, weight_decay=1e-4)

        testloader = DataLoader(
            test_dataset,
            batch_size=self.local_bs,
            shuffle=False)

        criterion = nn.NLLLoss().to(self.device)

        for _ in range(self.local_ep):
            batch_loss = []
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                model_copy.zero_grad()
                log_probs = model_copy(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

        if index_privacy_r:
            num_of_sparse_params = self.num_of_sparse_params + int(self.num_of_sparse_params * index_privacy_r)
        else:
            num_of_sparse_params = self.num_of_sparse_params
        _, top_k_indices = zero_except_top_k_weights(
            diff_weights(model.state_dict(), model_copy.state_dict()), self.buffer_names, num_of_sparse_params)
        return top_k_indices


def calc_similarity(metric, indices1, indices2):
    if metric == 'jaccard':
        set1 = set(indices1)
        set2 = set(indices2)
        intersected = set1.intersection(set2)
        return float(len(intersected)) / \
            (len(set1) + len(set2) - len(intersected))
    else:
        exit("no metric")


class AttackMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(AttackMLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return F.log_softmax(x, dim=1)


def run_attack(
        path_project,
        prefix,
        attacker: Attacker,
        target_client_ids,
        target_client_labels,
        args):
    # fix attacker's behavior even when loading from file
    np.random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)

    inferred_result = attacker.class_label_inference(
        method=args.attack, is_concate=args.single_model, fixed_inference_number=args.fixed_inference_number, per_round=args.per_round)

    def calc_accuracy(target_client_ids, target_client_labels, inferred_result, args):
        acc_comp = 0
        acc_part = 0
        num = 0
        missing_labels = 0
        for client_id in target_client_ids:
            if inferred_result[client_id]:
                num += 1
                if args.verbose:
                    print(
                        "(top-1, all): ",
                        inferred_result[client_id][1],
                        inferred_result[client_id][0],
                        ", correct: ",
                        target_client_labels[client_id])
                if args.fixed_inference_number and len(inferred_result[client_id][0]) > len(target_client_labels[client_id]):
                    missing_labels += 1
                elif inferred_result[client_id][0] == target_client_labels[client_id]:
                    acc_comp += 1
                if inferred_result[client_id][1].intersection(
                        target_client_labels[client_id]):
                    acc_part += 1
        return acc_comp, acc_part, num, missing_labels

    def show_attack_results(acc_comp, acc_part, num, missing_labels, epoch):
        if num > 0:
            print(
                f' \n Attack results after {epoch} global rounds of training:')
            print("|---- Attack completely success: ", acc_comp, "/", num - missing_labels)
            print("|---- Attack top-1 success: ", acc_part, "/", num)

    if args.per_round:
        result_per_round = {}
        inferred_result_per_round = {round: {} for round in range(args.epochs)}
        for client_id in target_client_ids:
            for round in range(args.epochs):
                if inferred_result.get(client_id) and inferred_result[client_id].get(round):
                    inferred_result_per_round[round][client_id] = inferred_result[client_id][round]
                else:
                    inferred_result_per_round[round][client_id] = None
        for round in range(args.epochs):
            acc_comp, acc_part, num, missing_labels  = calc_accuracy(target_client_ids, target_client_labels, inferred_result_per_round[round], args)
            show_attack_results(acc_comp, acc_part, num, missing_labels, round)
            result_per_round[round] = (acc_comp, acc_part, num, missing_labels)
    else:
        acc_comp, acc_part, num, missing_labels  = calc_accuracy(target_client_ids, target_client_labels, inferred_result, args)
        show_attack_results(acc_comp, acc_part, num, missing_labels, args.epochs)

    if prefix in ['exp1', 'exp2', 'exp3']:
        save_result(path_project, prefix,
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
                        acc_comp / (num - missing_labels),
                        acc_part / num,
                        num,
                        missing_labels
                    ],
                    add=True
        )
    elif prefix in ['exp4']:
        for round, result in result_per_round.items():
            acc_comp, acc_part, num, missing_labels = result
            save_result(path_project, prefix,
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
                            acc_comp / (num - missing_labels),
                            acc_part / num,
                            num,
                            missing_labels,
                            round
                        ],
                        add=True
            )
    elif prefix in ['exp6']:
        save_result(path_project, prefix,
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
                        acc_comp / (num - missing_labels),
                        acc_part / num,
                        missing_labels,
                        num
                    ],
                    add=True
        )
    elif prefix in ['exp8']:
        save_result(path_project, prefix,
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
                        acc_comp / (num - missing_labels),
                        acc_part / num,
                        missing_labels,
                        num
                    ],
                    add=True
        )

