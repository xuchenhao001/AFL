#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users



def get_indices(labels, user_labels, n_samples):
    indices = []
    for selected_label in user_labels:
        label_samples = np.where(labels[1,:] == selected_label)
        label_indices = labels[0, label_samples]
        selected_indices = list(np.random.choice(label_indices[0], n_samples, replace=False))
        indices += selected_indices
    return indices


def noniid_onepass(dataset_train, dataset_test, num_users, dataset_name='mnist', kept_class=3):
    train_users = {}
    test_users = {}
    skew_users1 = {}
    skew_users2 = {}
    skew_users3 = {}
    skew_users4 = {}

    skew1_pct = 0.05
    skew2_pct = 0.10
    skew3_pct = 0.15
    skew4_pct = 0.20

    train_idxs = np.arange(len(dataset_train))
    train_labels = dataset_train.targets
    train_labels = np.vstack((train_idxs, train_labels))

    test_idxs = np.arange(len(dataset_test))
    test_labels = dataset_test.targets
    test_labels = np.vstack((test_idxs, test_labels))
    if dataset_name == 'mnist':
        labels = list(range(10))
        samples = [150, 50, int(50*skew1_pct), int(50*skew2_pct), int(50*skew3_pct), int(50*skew4_pct)]
    elif dataset_name == 'cifar':
        labels = list(range(10))
        samples = [150, 50, int(50*skew1_pct), int(50*skew2_pct), int(50*skew3_pct), int(50*skew4_pct)]
    elif dataset_name == 'uci':
        labels = list(range(6))
        samples = [500, 200, int(200*skew1_pct), int(200*skew2_pct), int(200*skew3_pct), int(200*skew4_pct)]
    elif dataset_name == 'realworld':
        labels = list(range(8))
        samples = [500, 100, int(100*skew1_pct), int(100*skew2_pct), int(100*skew3_pct), int(100*skew4_pct)]
    for i in range(num_users):
        user_labels = np.random.choice(labels, size=kept_class, replace=False)
        skew_labels = [i for i in labels if i not in user_labels]
        train_indices = get_indices(train_labels, user_labels, n_samples=samples[0])
        test_indices = get_indices(test_labels, user_labels, n_samples=samples[1])

        skew1_indices = get_indices(test_labels, skew_labels, n_samples=samples[2])
        skew2_indices = get_indices(test_labels, skew_labels, n_samples=samples[3])
        skew3_indices = get_indices(test_labels, skew_labels, n_samples=samples[4])
        skew4_indices = get_indices(test_labels, skew_labels, n_samples=samples[5])

        train_users[i] = train_indices
        test_users[i] = test_indices
        skew_users1[i] = skew1_indices
        skew_users2[i] = skew2_indices
        skew_users3[i] = skew3_indices
        skew_users4[i] = skew4_indices
    return train_users, test_users, (skew_users1, skew_users2, skew_users3, skew_users4)


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)

