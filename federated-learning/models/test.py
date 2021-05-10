#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Update import DatasetSplit


def test_img(net_g, dataset_test, test_indices, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    dataset = DatasetSplit(dataset_test, test_indices)
    data_loader = DataLoader(dataset, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        data = data.detach().clone().type(torch.FloatTensor)
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def test_img_total(net_g, dataset_test, idx_list, args):
    net_g.eval()
    subset_idx = [0]
    idx_total = []
    for i in range(len(idx_list)):
        subset_idx.append(subset_idx[-1] + len(idx_list[i]))
        idx_total += idx_list[i]

    test_loss = [0] * len(idx_list)
    correct = [0] * len(idx_list)

    dataset = DatasetSplit(dataset_test, idx_total)
    data_loader = DataLoader(dataset, batch_size=args.bs)

    y_target = []
    y_pred = []
    for idx, (data, target) in enumerate(data_loader):
        data = data.detach().clone().type(torch.FloatTensor)
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        y_target.append(target)
        pred = log_probs.data.max(1, keepdim=True)[1]
        y_pred.append(pred)

    y_target = torch.cat(y_target)
    y_pred = torch.cat(y_pred)
    y_pred = y_pred.squeeze(1)

    for i in range(len(idx_list)):
        correct[i] = sum(y_target[subset_idx[i]:subset_idx[i + 1]] == y_pred[subset_idx[i]:subset_idx[i + 1]])

    return correct
