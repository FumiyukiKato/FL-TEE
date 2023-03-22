import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import flatten_params, get_learnable_parameters

import copy

TRAIN_RATIO = 0.9
# VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(
            self,
            dataset,
            idxs,
            logger,
            device,
            local_bs,
            optimizer,
            lr,
            local_ep,
            momentum,
            verbose):
        self.local_bs = local_bs
        self.optimizer = optimizer
        self.lr = lr
        self.local_ep = local_ep
        self.momentum = momentum
        self.verbose = verbose
        self.logger = logger
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(
        #     dataset, list(idxs))
        self.trainloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = device
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(TRAIN_RATIO * len(idxs))]
        idxs_test = idxs[
            int((TRAIN_RATIO) * len(idxs)):]
        trainloader = DataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=self.local_bs,
            shuffle=True)
        testloader = DataLoader(
            DatasetSplit(dataset, idxs_test),
            batch_size=max(1, int(len(idxs_test) * TEST_RATIO)),
            shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=1e-4)

        for iter in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()
                    ))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0., 0., 0.

        for _, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss


def test_inference(model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=32,
                            shuffle=False)

    for _, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


def diff_weights(global_weights, local_weights):
    """Diff = Local - Global
    """
    w_diff = copy.deepcopy(local_weights)
    for key in w_diff.keys():
        if key.endswith('num_batches_tracked'):  # downcast to float
            global_weights[key] = global_weights[key].float()
            w_diff[key] = w_diff[key].float()
        w_diff[key] -= global_weights[key]
    return w_diff


def update_global_weights(global_weights, local_weights_diffs):
    """Global weights will be updated by side effects"""
    w_avg = copy.deepcopy(local_weights_diffs[0])
    len_of_diffs = len(local_weights_diffs)
    for key in w_avg.keys():
        for i in range(1, len_of_diffs):
            w_avg[key] += local_weights_diffs[i][key]
        w_avg[key] = torch.div(w_avg[key], len_of_diffs)
        if key.endswith('num_batches_tracked'):  # downcast to float
            global_weights[key] = global_weights[key].float()
            w_avg[key] = w_avg[key].float()
        global_weights[key] += w_avg[key]


def l2clipping(weights, buffer_names, clipping):
    """L2 Clipping
    
    Args:
        weights ([OrderedDict]): model.state_dict()
        buffer_names: not learnable parameters keys
        clipping (int): clipping threshold
    Return:
        clipped weights
    """
    clipped = copy.deepcopy(weights)
    learnable_parameters = get_learnable_parameters(weights, buffer_names)
    flat = flatten_params(learnable_parameters)
    coefficient = min(1, clipping / torch.norm(flat, 2))
    for key in clipped.keys():
        if key not in buffer_names:
            clipped[key] *= coefficient
    return clipped


def client_level_dp_update_global_weights(global_weights, local_weights_diffs, sigma, clipping, alpha, random_state):
    """Clipping and adding noise to global_weights
        Perform clipping and noise addition for each layer, but note that this is not possible if the layer is sparse.

    Args:
        global_weights ([OrderedDict]): model.state_dict() , updated by side effects
        local_weights_diffs [([OrderedDict])]: list of model.state_dict()
        sigma (float): standard deviation of gaussian noise
        clipping (int): clipping threshold
    """
    len_of_diffs = len(local_weights_diffs)
    w_avg = copy.deepcopy(local_weights_diffs[0])
    for key in w_avg.keys():
        for i in range(1, len_of_diffs):
            w_avg[key] += local_weights_diffs[i][key]
        # noise = random_state.normal(0, float(clipping * alpha * sigma), size=w_avg[key].shape) # cannot sure to use k/d sensitivity in the case of top-k sparsification
        noise = random_state.normal(0, float(clipping * sigma), size=w_avg[key].shape)
        global_weights[key] += torch.div(w_avg[key] + noise, len_of_diffs)
