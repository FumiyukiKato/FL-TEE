"""
USAGE:

LDP-FL and Shuffle FL are based on the paper
Úlfar Erlingsson, Vitaly Feldman, Ilya Mironov, Ananth Raghunathan, Shuang Song, Kunal Talwar, and Abhradeep Thakurta. 2020. Encode, shuffle, analyze privacy revisited: Formalizations and empirical evaluation. arXiv preprint arXiv:2001.03618 (2020).

LDP-FL
as well as including Shuffle FL analysis
$ python src/eval-ldp-sgd.py --dp_kind=ldp --eps_local=2.0 --epochs=1500

CDP-FL (DP-FedSGD)
$ python src/eval-ldp-sgd.py --dp_kind=cdp --sigma=50.0 --global_lr=1.0 --epochs=500

Plain FL (FedSGD)
$ python src/eval-ldp-sgd.py --dp_kind=nodp --global_lr=0.1 --epochs=300 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# from opacus import PrivacyEngine
# from opacus.utils.uniform_sampler import UniformWithReplacementSampler

from scipy.special import loggamma
import matplotlib.pyplot as plt

import argparse
import os
import datetime
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO

from pathlib import Path
import sys

path_project = Path(os.path.abspath("."))

DATA_SET_DIR = "dataset"

import copy
from rdp_accountant import compute_rdp, get_privacy_spent
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, client_iid
from update import (
    LocalUpdate,
    l2clipping,
    test_inference,
    diff_weights,
    TRAIN_RATIO,
)

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from update import DatasetSplit


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = nn.ZeroPad2d(2)(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)

        return output


def get_dataset(path_project, num_users, iid=True, all=False):
    data_dir = os.path.join(path_project, DATA_SET_DIR, "mnist")
    apply_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=apply_transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=apply_transform
    )

    if all:
        rand_ids = np.random.permutation(num_users)
        user_groups = {
            user_id: {record_id} for user_id, record_id in enumerate(rand_ids)
        }
    elif iid:
        user_groups = mnist_iid(train_dataset, num_users)
    else:  # args.data_dist == 'non-IID':
        user_groups = mnist_noniid(train_dataset, num_users, 5, False)
    class_labels = set(test_dataset.train_labels.numpy())

    return train_dataset, test_dataset, user_groups, class_labels


class CDPLocalUpdate(LocalUpdate):
    def __init__(
        self,
        dataset,
        idxs,
        logger,
        device,
        local_bs,
        optimizer,
        local_lr,
        local_ep,
        momentum,
        verbose,
    ):
        super().__init__(
            dataset,
            idxs,
            logger,
            device,
            local_bs,
            optimizer,
            local_lr,
            local_ep,
            momentum,
            verbose,
        )

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(
            DatasetSplit(dataset, idxs), batch_size=len(idxs), shuffle=True
        )
        return trainloader, None


class SVGLocalUpdate(LocalUpdate):
    def __init__(
        self, dataset, idxs, logger, device, optimizer, momentum, verbose, eps, L
    ):
        self.local_bs = 1
        self.optimizer = optimizer
        self.lr = None
        self.local_ep = 1
        self.momentum = momentum
        self.verbose = verbose
        self.logger = logger
        self.eps = eps
        self.L = L
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = device
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(
            DatasetSplit(dataset, idxs), batch_size=len(idxs), shuffle=True
        )
        return trainloader

    def update_weights(self, model, dp_kind):
        model.train()

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            log_probs = model(images)
            loss = self.criterion(log_probs, labels)
            loss.backward(retain_graph=True)
            if dp_kind == "ldp":
                rand_grad = perturb_grad_ldpsgd(model, self.device, self.eps, self.L)
            elif dp_kind in ["nodp", "cdp"]:
                rand_grad = grad_sgd(model, self.device)

        return model, rand_grad


def compute_shuffle_DP(num_users, eps_local, delta, k, delta_global):
    # calc shuffling bound by https://arxiv.org/abs/2012.12803
    import computeamplification as CA

    num_iterations = 10
    step = 100
    numerical_upperbound_eps = CA.numericalanalysis(
        num_users, eps_local, delta, num_iterations, step, True
    )
    return advanced_composition(numerical_upperbound_eps, delta, k, delta_global)

def advanced_composition(e, delta, k, delta_global):
    # advanced composition followed by https://arxiv.org/pdf/2001.03618.pdf
    delta_dash = delta_global - k * delta
    if delta_dash < 0:
        print("##### Delta must be positive")
        return None
    cal_eps = k * (e**2) * 0.5 + np.sqrt(k) * e * np.sqrt(
        2 * np.log(np.sqrt(k * np.pi * 0.5) * e / delta_dash)
    )
    return cal_eps


def update_server_model_by_local_grad_agg(
    base_model,
    mean_grad,
    dp_kind,
    global_lr,
    const_grad,
    param_space_norm,
    scheduler=None,
    optimizer=None,
):
    if dp_kind == "ldp":
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                if optimizer is not None:
                    param.grad = mean_grad[name] * const_grad
                else:
                    param.data -= mean_grad[name] * global_lr * const_grad

        if optimizer is not None:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if param_space_norm > 0:
            base_model = l2projection(base_model, param_space_norm)

        return base_model

    elif dp_kind in ["nodp", "cdp"]:
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                param.grad = mean_grad[name]
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return base_model


def l2projection(model, radius):
    l2norm_updated = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            l2norm_updated += param.data.norm(2).item() ** 2
    l2norm_updated = l2norm_updated ** (1.0 / 2)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data / max(1.0, l2norm_updated) * radius

    return model


def grad_sgd(model, device):
    rand_grad = dict()

    for name, param in model.named_parameters():
        if param.requires_grad:
            rand_grad[name] = param.grad.detach().clone()
    return rand_grad


# https://anonymous.4open.science/r/ldp-hypothesis-testing-28FA/src/dpsgd_local.py
TINYNUM = 1e-10


def perturb_grad_ldpsgd(model, device, eps, L):
    sample_vec = dict()
    rand_grad = dict()

    ## To avoid the l2-norm of grad being zero.
    for tensor_name, tensor in model.named_parameters():
        if tensor.requires_grad:
            tensor.grad += (torch.rand(tensor.grad.shape).to(device) - 0.5) * TINYNUM

    ## Compute the norm of the clipped gradient, and compy the clipped gradient into the rand_grad
    torch.nn.utils.clip_grad_norm_(model.parameters(), L)
    l2norm = 0
    for tensor_name, tensor in model.named_parameters():
        if tensor.requires_grad:
            rand_grad[tensor_name] = tensor.grad.detach().clone()
            l2norm += rand_grad[tensor_name].norm(2).item() ** 2
    model.zero_grad()
    l2norm = l2norm ** (1.0 / 2)

    ## Compute the sign of first flipping.
    p1 = 0.5 + (l2norm / (2 * L))
    r1 = np.random.rand(1)
    if r1 < p1:
        z_sign = 1
    else:
        z_sign = -1

    ## First flipping
    for name, param in model.named_parameters():
        if param.requires_grad:
            rand_grad[name] = rand_grad[name] / l2norm * L * z_sign

    ## uniform random vector sampling from L2-ball (unit sphere)
    sampvec_sqsum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            sample_vec[name] = (
                torch.FloatTensor(rand_grad[name].size()).normal_(0, 1).to(device)
            )
            sampvec_sqsum += torch.sum(sample_vec[name] ** 2)
    sampvec_sq = torch.sqrt(sampvec_sqsum)

    inner_prod = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            sample_vec[name] = sample_vec[name] / sampvec_sq
            inner_prod += torch.sum(sample_vec[name] * rand_grad[name])

    ## Compute the sign of last flipping.
    last_sign = torch.sign(inner_prod)
    p2 = np.exp(eps) / (np.exp(eps) + 1)
    r2 = np.random.rand(1)
    if r2 > p2:
        last_sign *= -1

    ## Last flipping for ensuring eps-LDP
    for name, param in model.named_parameters():
        if param.requires_grad:
            sample_vec[name] = sample_vec[name] * last_sign

    return sample_vec


def client_level_dp_update_global_weights(global_weights, sum_of_local_weights_diffs, sigma, clipping, num_users, random_state):
    """Clipping and adding noise to global_weights
        Perform clipping and noise addition for each layer, but note that this is not possible if the layer is sparse.

    Args:
        global_weights ([OrderedDict]): model.state_dict() , updated by side effects
        local_weights_diffs [([OrderedDict])]: list of model.state_dict()
        sigma (float): standard deviation of gaussian noise
        clipping (int): clipping threshold
    """
    for key in sum_of_local_weights_diffs.keys():
        noise = random_state.normal(0, float(clipping * sigma), size=sum_of_local_weights_diffs[key].shape)
        global_weights[key] += torch.div(sum_of_local_weights_diffs[key] + noise, num_users)


def update_global_weights(global_weights, sum_of_local_weights_diffs, num_users):
    """Global weights will be updated by side effects"""
    for key in sum_of_local_weights_diffs.keys():
        sum_of_local_weights_diffs[key] = torch.div(sum_of_local_weights_diffs[key], num_users)
        if key.endswith('num_batches_tracked'):  # downcast to float
            global_weights[key] = global_weights[key].float()
            sum_of_local_weights_diffs[key] = sum_of_local_weights_diffs[key].float()
        global_weights[key] += sum_of_local_weights_diffs[key]


def eval_fed_sgd(
    seed,
    gpu_id,
    logger,
    num_users,
    frac,
    epochs,
    delta,
    optimizer,
    momentum,
    eps_local,
    verbose,
    dp_kind,
    global_lr,
    clipping,
    eps_global=None,
    sigma=None,
):
    if gpu_id:
        torch.cuda.set_device(gpu_id)
    device = "cuda" if gpu_id else "cpu"
    print("DP: ", dp_kind)
    if dp_kind == "ldp":
        print(f"    {eps_local}-LDP for local randomizer")
    elif dp_kind == "cdp":
        print(f"    noise multiplier {sigma} ")
    global_model = MNIST_CNN()
    global_model.to(device)
    global_model.train()

    train_dataset, test_dataset, user_groups, class_labels = get_dataset(
        path_project, num_users, True, num_users == 60000
    )

    # Training
    global_test_result = []
    L = None
    param_space_norm = None
    const_grad = None

    if dp_kind == "ldp":
        d = 0
        L = 1.0
        for name, param in global_model.named_parameters():
            if param.requires_grad:
                d += param.data.numel()
        const_gamma = np.exp(loggamma((d - 1) * 0.5 + 1) - loggamma(d * 0.5 + 1))
        const_eps = (np.exp(eps_local) + 1) / (np.exp(eps_local) - 1)
        const_grad = L * np.sqrt(np.pi) * 0.5 * const_gamma * const_eps * d
        param_space_norm = L * const_eps * 0.75 * np.sqrt(np.pi) * np.sqrt(d)
        global_lr = (
            param_space_norm * np.sqrt(num_users) / (const_eps * L * np.sqrt(d)) * 0.05
        )
        if param_space_norm > 0:
            global_model = l2projection(global_model, param_space_norm)

    elif dp_kind == "cdp":
        rs_for_gaussian_noise = np.random.RandomState(seed)
        orders = (
            [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
            + list(range(5, 64))
            + [128, 256, 512]
        )

    opt = optim.SGD(global_model.parameters(), lr=global_lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    for epoch in range(epochs):
        global_model.train()

        idxs_users = client_iid(frac, num_users)
        mean_grad = dict()

        for idx in idxs_users:
            local_model = SVGLocalUpdate(
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger,
                device=device,
                optimizer=optimizer,
                momentum=momentum,
                verbose=verbose,
                eps=eps_local,
                L=L,
            )
            model, rand_grad = local_model.update_weights(
                model=copy.deepcopy(global_model), dp_kind=dp_kind
            )
            if dp_kind == "cdp":
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
                for tensor_name, tensor in model.named_parameters():
                    if tensor.requires_grad:
                        rand_grad[tensor_name] = tensor.grad.detach().clone()

            for name in rand_grad.keys():
                if mean_grad.get(name) is None:
                    mean_grad[name] = rand_grad[name]
                else:
                    mean_grad[name] += rand_grad[name]

        for name in mean_grad.keys():
            if dp_kind == "cdp":
                noise = torch.from_numpy(rs_for_gaussian_noise.normal(0, float(clipping * sigma), size=mean_grad[name].shape)).to(device)
                mean_grad[name] += noise
            mean_grad[name] /= len(idxs_users)

        global_model = update_server_model_by_local_grad_agg(
            global_model,
            mean_grad,
            dp_kind,
            global_lr=global_lr,
            optimizer=opt,
            const_grad=const_grad,
            scheduler=scheduler,
            param_space_norm=param_space_norm,
        )

        test_acc, test_loss = test_inference(global_model, test_dataset, device)
        print(f" \n Results after {epoch+1} ({epochs}) global rounds of training:")
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
        global_test_result.append([epoch, test_acc, test_loss])

        if dp_kind == "ldp":
            individual_delta = (delta / 2.0) / (epoch + 1)
            shuffle_dp_eps = compute_shuffle_DP(
                num_users, eps_local, individual_delta, epoch + 1, delta
            )
            print(
                "|---- Shuffle DP : ({:.6f}, {:.6f})-DP".format(shuffle_dp_eps, delta)
            )
            global_test_result[-1].append(shuffle_dp_eps)
        elif dp_kind == "cdp":
            individual_delta = (delta / 2.0) / (epoch + 1)
            gauss_eps = clipping / sigma * np.sqrt(2 * np.log(1.25 / individual_delta))
            cdp_eps = advanced_composition(gauss_eps, individual_delta, epoch + 1, delta)
            print(
                "|---- Central DP (Advanced Comp) : ({:.6f}, {:.6f})-DP".format(cdp_eps, delta)
            )

            rdp = compute_rdp(frac, sigma, epoch + 1, orders)
            eps_spent, delta_spent, opt_order = get_privacy_spent(
                orders, rdp, target_delta=delta
            )
            print(
                "|---- Central DP (RDP) : ({:.6f}, {:.6f})-DP".format(eps_spent, delta_spent)
            )
            global_test_result[-1].append(eps_spent)
            if eps_spent > eps_global or delta_spent > delta:
                print("|----  ######## Excess setted privacy budget ########")

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S') 
    f = open(path_project / "exp" / "results" / f'fedsgd-{dp_kind}-{timestamp}.txt', "w")
    f.write(str(args))
    f.write('\n')
    f.write(str(global_test_result))
    f.close()
    print("done.")


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(
        description="DP-SGD (Local)", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=str, default=None)
    parser.add_argument("--num_users", type=int, default=60000)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=0.00001)
    parser.add_argument("--clipping", type=float, default=1.0)
    parser.add_argument("--eps_central", type=float, default=5.0)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--global_lr", type=float, default=0.01)
    parser.add_argument("--eps_local", type=float, default=1.9)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--local_bs", type=int, default=10)
    parser.add_argument("--local_lr", type=float, default=0.01)
    parser.add_argument("--local_ep", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--dp_kind", type=str, default="ldp", metavar="ldp, nodp, cdp")
    args = parser.parse_args()

    logger = SummaryWriter(os.path.join(path_project, "log"))

    if args.dp_kind in ["ldp", "nodp", "cdp"]:
        eval_fed_sgd(
            seed=args.seed,
            gpu_id=args.gpu_id,
            logger=logger,
            num_users=args.num_users,
            frac=args.frac,
            epochs=args.epochs,
            delta=args.delta,
            optimizer="sgd",
            momentum=args.momentum,
            eps_local=args.eps_local,
            verbose=False,
            dp_kind=args.dp_kind,
            global_lr=args.global_lr,
            clipping=args.clipping,
            eps_global=args.epsilon,
            sigma=args.sigma,
        )
    else:
        exit("Error: dp_kind must be ldp, nodp for fedsvg")

