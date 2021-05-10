#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import logging
import os

from torchvision import datasets, transforms

from datasets.REALWORLD import REALWORLDDataset
from datasets.UCI import UCIDataset
from models.Nets import CNNCifar, CNNMnist, UCI_CNN, MLP
from utils.sampling import mnist_iid, cifar_iid, noniid_onepass


# format colorful log output
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
# The background is set with 40 plus the number of the color, and the foreground with 30
# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    FORMAT = "[$BOLD%(asctime)-20s$RESET][%(levelname)s] %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    # FORMAT = "%(asctime)s %(message)s"
    COLOR_FORMAT = formatter_message(FORMAT, True)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)

        color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        self.addHandler(console)
        return


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger("util")


def dataset_loader(dataset_name, isIID, num_users):
    logger.info("Load dataset [%s]." % dataset_name)
    dataset_train = None
    dataset_test = None
    dict_users = None
    test_users = None
    skew_users = None
    real_path = os.path.dirname(os.path.realpath(__file__))
    # load dataset and split users
    if dataset_name == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_data_path = os.path.join(real_path, "../../data/mnist/")
        dataset_train = datasets.MNIST(mnist_data_path, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(mnist_data_path, train=False, download=True, transform=trans_mnist)
        # sample users
        if isIID:
            dict_users = mnist_iid(dataset_train, num_users)
        else:
            dict_users, test_users, skew_users = noniid_onepass(dataset_train, dataset_test, num_users,
                                                                dataset_name='mnist')
    elif dataset_name == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        cifar_data_path = os.path.join(real_path, "../../data/cifar/")
        dataset_train = datasets.CIFAR10(cifar_data_path, train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(cifar_data_path, train=False, download=True, transform=trans_cifar)
        if isIID:
            dict_users = cifar_iid(dataset_train, num_users)
        else:
            dict_users, test_users, skew_users = noniid_onepass(dataset_train, dataset_test, num_users,
                                                                dataset_name='cifar')
    elif dataset_name == 'uci':
        uci_data_path = os.path.join(real_path, "../../data/uci/")
        dataset_train = UCIDataset(data_path=uci_data_path, phase='train')
        dataset_test = UCIDataset(data_path=uci_data_path, phase='eval')
        if isIID:
            dict_users = cifar_iid(dataset_train, num_users)
        else:
            dict_users, test_users, skew_users = noniid_onepass(dataset_train, dataset_test, num_users,
                                                                dataset_name='uci')
    elif dataset_name == 'realworld':
        realworld_data_path = os.path.join(real_path, "../../data/realworld_client/")
        dataset_train = REALWORLDDataset(data_path=realworld_data_path, phase='train')
        dataset_test = REALWORLDDataset(data_path=realworld_data_path, phase='eval')
        if isIID:
            dict_users = cifar_iid(dataset_train, num_users)
        else:
            dict_users, test_users, skew_users = noniid_onepass(dataset_train, dataset_test, num_users,
                                                                dataset_name='realworld')
    return dataset_train, dataset_test, dict_users, test_users, skew_users


def model_loader(model_name, dataset_name, device, num_channels, num_classes, img_size):
    logger.info("Load model [%s] for dataset [%s]. Train using device [%s]." % (model_name, dataset_name, device))
    net_glob = None
    # build model, init part
    if model_name == 'cnn' and dataset_name == 'cifar':
        net_glob = CNNCifar(num_classes).to(device)
    elif model_name == 'cnn' and dataset_name == 'mnist':
        net_glob = CNNMnist(num_channels, num_classes).to(device)
    elif model_name == 'cnn' and dataset_name == 'uci':
        net_glob = UCI_CNN(n_class=6).to(device)
    elif model_name == 'cnn' and dataset_name == 'realworld':
        net_glob = UCI_CNN(n_class=8).to(device)
    elif model_name == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=num_classes).to(device)
    return net_glob

