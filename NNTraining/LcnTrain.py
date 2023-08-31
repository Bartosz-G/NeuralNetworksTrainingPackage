import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def print_args(args):
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


def train(args, model, device, train_loader, optimizer, epoch, anneal, alpha=1):
    model.train()
    dataset_len = 0
    avg_loss = AverageMeter()

    for (data, target) in train_loader:
        dataset_len += len(target)
        data, target = data.to(device), target.to(device)
        if args.task == 'classification':
            target = target.type(torch.cuda.LongTensor)

        optimizer.zero_grad()
        ###############
        data.requires_grad = True
        if model.net_type == 'locally_constant':
            if args.p != -1:
                assert (args.p >= 0. and args.p < 1)
                output, regularization = model(data, alpha=alpha, anneal=anneal, p=args.p, training=True)
            else:
                output, regularization = model(data, alpha=alpha, anneal=anneal, p=1 - alpha, training=True)

        elif model.net_type == 'locally_linear':
            output, regularization = model.normal_forward(data)
        ###############

        optimizer.zero_grad()
        if args.task == 'classification':
            # Added: Bart
            target_one_dim = torch.argmax(target, dim=1)
            loss = F.cross_entropy(output, target_one_dim)
        elif args.task == 'regression':
            output = output.squeeze(-1)
            loss = ((output - target) ** 2).mean()

        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())

    return avg_loss.avg

# Modified from the original paper
def test_metrics(args, model, device, test_loader, metrics_func, test_set_name):
    with torch.no_grad():
        model.eval()

        # ==============================================================
        # ===TODO: Add batched dataloader handling
        # ==============================================================

        data, target = next(iter(test_loader))

        data, target = data.to(device), target.to(device)
        if args.task == 'classification':
            target = target.type(torch.cuda.LongTensor)

        ###############
        data.requires_grad = True
        if model.net_type == 'locally_constant':
            output, relu_masks = model(data, p=0, training=False)
        elif model.net_type == 'locally_linear':
            output, relu_masks = model.normal_forward(data, p=0, training=False)
        ###############

        if args.task == 'classification':
            output = torch.softmax(output, dim=-1)
            metrics = metrics_func(target, output, True)
        elif args.task == 'regression':
            metrics = metrics_func(target, output, False)

        return metrics



def test_loss(args, model, device, test_loader, test_set_name):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0

        score = []
        label = []
        dataset_len = 0

        pattern_to_pred = dict()
        tree_x = []
        tree_pattern = []

        for data, target in test_loader:
            dataset_len += len(target)
            label += list(target)
            data, target = data.to(device), target.to(device)
            if args.task == 'classification':
                target = target.type(torch.cuda.LongTensor)

            ###############
            data.requires_grad = True
            if model.net_type == 'locally_constant':
                output, relu_masks = model(data, p=0, training=False)
            elif model.net_type == 'locally_linear':
                output, relu_masks = model.normal_forward(data, p=0, training=False)
            ###############

            if args.task == 'classification':
                # Modified: Bart
                target_one_dim = torch.argmax(target, dim=1)
                test_loss += F.cross_entropy(output, target_one_dim, reduction='sum').item()
                # Removed: Bart
                # output = torch.softmax(output, dim=-1)
                # ...
                # output = output[:, 1]
            elif args.task == 'regression':
                output = output.squeeze(-1)
                test_loss += ((output - target) ** 2).mean().item() * len(target)

        test_loss /= dataset_len

        # Removed: Bart
        # if args.task == 'classification':
        # ...

        return test_loss
