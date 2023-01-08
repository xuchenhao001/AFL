import asyncio
import base64
import copy
import gzip
import hashlib
import json
import logging
import random
import requests
import socket
import subprocess
import threading
import time
import numpy as np
import os
import torch
from torchvision import datasets, transforms

from datasets.LOOP import LOOPDataset
from datasets.REALWORLD import REALWORLDDataset
from datasets.UCI import UCIDataset
from models.Nets import CNNCifar, CNNMnist, CNNFashion, UCI_CNN, MLP, LSTM
from models.test import test_img_total, test_lstm
from utils.sampling import iid_onepass, noniid_onepass

lock = threading.Lock()
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


def dataset_loader(dataset_name, dataset_train_size, isIID, num_users):
    dataset_test_size = int(dataset_train_size * 0.25)  # the dataset size ratio of the training to the test is 8:2
    logger.info("Load dataset [%s] with training data size [%d], test data size [%d]" %
                (dataset_name, dataset_train_size, dataset_test_size))
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
        if isIID:
            dict_users, test_users = iid_onepass(dataset_train, dataset_train_size, dataset_test, dataset_test_size,
                                                 num_users, dataset_name=dataset_name)
        else:
            dict_users, test_users, skew_users = noniid_onepass(dataset_train, dataset_train_size, dataset_test,
                                                                dataset_test_size, num_users, dataset_name=dataset_name)
    elif dataset_name == 'fashion_mnist':
        trans_fashion = transforms.Compose([transforms.ToTensor()])
        mnist_data_path = os.path.join(real_path, "../../data/fashion-mnist/")
        dataset_train = datasets.FashionMNIST(mnist_data_path, train=True, download=True, transform=trans_fashion)
        dataset_test = datasets.FashionMNIST(mnist_data_path, train=False, download=True, transform=trans_fashion)
        if isIID:
            dict_users, test_users = iid_onepass(dataset_train, dataset_train_size, dataset_test, dataset_test_size,
                                                 num_users, dataset_name=dataset_name)
        else:
            dict_users, test_users, skew_users = noniid_onepass(dataset_train, dataset_train_size, dataset_test,
                                                                dataset_test_size, num_users,
                                                                dataset_name=dataset_name)
    elif dataset_name == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        cifar_data_path = os.path.join(real_path, "../../data/cifar/")
        dataset_train = datasets.CIFAR10(cifar_data_path, train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(cifar_data_path, train=False, download=True, transform=trans_cifar)
        if isIID:
            dict_users, test_users = iid_onepass(dataset_train, dataset_train_size, dataset_test, dataset_test_size,
                                                 num_users, dataset_name=dataset_name)
        else:
            dict_users, test_users, skew_users = noniid_onepass(dataset_train, dataset_train_size, dataset_test,
                                                                dataset_test_size, num_users, dataset_name=dataset_name)
    elif dataset_name == 'uci':
        # https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
        uci_data_path = os.path.join(real_path, "../../data/uci/")
        dataset_train = UCIDataset(data_path=uci_data_path, phase='train')
        dataset_test = UCIDataset(data_path=uci_data_path, phase='eval')
        if isIID:
            dict_users, test_users = iid_onepass(dataset_train, dataset_train_size, dataset_test, dataset_test_size,
                                                 num_users, dataset_name=dataset_name)
        else:
            dict_users, test_users, skew_users = noniid_onepass(dataset_train, dataset_train_size, dataset_test,
                                                                dataset_test_size, num_users, dataset_name=dataset_name)
    elif dataset_name == 'realworld':
        # https://sensor.informatik.uni-mannheim.de/#dataset_realworld
        realworld_data_path = os.path.join(real_path, "../../data/realworld_client/")
        dataset_train = REALWORLDDataset(data_path=realworld_data_path, phase='train')
        dataset_test = REALWORLDDataset(data_path=realworld_data_path, phase='eval')
        if isIID:
            dict_users, test_users = iid_onepass(dataset_train, dataset_train_size, dataset_test, dataset_test_size,
                                                 num_users, dataset_name=dataset_name)
        else:
            dict_users, test_users, skew_users = noniid_onepass(dataset_train, dataset_train_size, dataset_test,
                                                                dataset_test_size, num_users, dataset_name=dataset_name)
    elif dataset_name == 'loop':
        # https://github.com/zhiyongc/Seattle-Loop-Data
        loop_data_path = os.path.join(real_path, "../../data/loop/")
        dataset_train = LOOPDataset(data_path=loop_data_path, phase='train')
        dataset_test = LOOPDataset(data_path=loop_data_path, phase='eval')
        dict_users, test_users = iid_onepass(dataset_train, dataset_train_size, dataset_test, dataset_test_size,
                                             num_users, dataset_name=dataset_name)
    return dataset_train, dataset_test, dict_users, test_users, skew_users


def model_loader(model_name, dataset_name, device, num_channels, num_classes, img_size):
    logger.info("Load model [%s] for dataset [%s]. Train using device [%s]." % (model_name, dataset_name, device))
    net_glob = None
    # build model, init part
    if model_name == 'cnn' and dataset_name == 'cifar':
        net_glob = CNNCifar(num_classes).to(device)
    elif model_name == 'cnn' and dataset_name == 'mnist':
        net_glob = CNNMnist(num_channels, num_classes).to(device)
    elif model_name == 'cnn' and dataset_name == 'fashion_mnist':
        net_glob = CNNFashion(num_channels, num_classes).to(device)
    elif model_name == 'cnn' and dataset_name == 'uci':
        net_glob = UCI_CNN(n_class=6).to(device)
    elif model_name == 'cnn' and dataset_name == 'realworld':
        net_glob = UCI_CNN(n_class=8).to(device)
    elif model_name == 'lstm' and dataset_name == 'loop':
        net_glob = LSTM(img_size[1], img_size[1], img_size[1], output_last=True)
    elif model_name == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=num_classes).to(device)
    return net_glob


def test_model(net_glob, dataset_test, args, test_users, skew_users, idx):
    if args.model == "lstm":  # for LSTM
        test_indices = test_users[idx]
        loss_mse, loss_mae = test_lstm(net_glob, dataset_test, args, test_indices)
        return loss_mse, loss_mae, 0.0, 0.0, 0.0
    if args.iid:
        idx_total = [test_users[idx]]
        acc_list, _ = test_img_total(net_glob, dataset_test, args, idx_total)
        acc_local = acc_list[0].item()
        return acc_local, 0.0, 0.0, 0.0, 0.0
    else:
        idx_total = [test_users[idx], skew_users[0][idx], skew_users[1][idx], skew_users[2][idx], skew_users[3][idx]]
        acc_list, _ = test_img_total(net_glob, dataset_test, args, idx_total)
        acc_local = acc_list[0].item()
        acc_local_skew1 = acc_list[1].item()
        acc_local_skew2 = acc_list[2].item()
        acc_local_skew3 = acc_list[3].item()
        acc_local_skew4 = acc_list[4].item()
        return acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4


# returns variable from sourcing a file
def env_from_sourcing(file_to_source_path, variable_name):
    source = 'source %s && export MYVAR=$(echo "${%s[@]}")' % (file_to_source_path, variable_name)
    dump = '/usr/bin/python3 -c "import os, json; print(os.getenv(\'MYVAR\'))"'
    pipe = subprocess.Popen(['/bin/bash', '-c', '%s && %s' % (source, dump)], stdout=subprocess.PIPE)
    # return json.loads(pipe.stdout.read())
    return pipe.stdout.read().decode("utf-8").rstrip()


def http_client_post(url, body_data, accumulate_time=True):
    logger.debug("[HTTP Start] [" + body_data['message'] + "] Start http client post to: " + url)
    request_start_time = time.time()
    response = requests.post(url, json=body_data, timeout=300)
    logger.debug("[HTTP Success] [" + body_data['message'] + "] from " + url)
    request_time = time.time() - request_start_time
    if accumulate_time:
        add_communication_time(request_time)
    return response.json()


accumulate_communication_time = 0


def add_communication_time(request_time):
    global accumulate_communication_time
    lock.acquire()
    accumulate_communication_time += request_time
    lock.release()


def reset_communication_time():
    global accumulate_communication_time
    lock.acquire()
    communication_time = accumulate_communication_time
    accumulate_communication_time = 0
    lock.release()
    return communication_time


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def __conver_numpy_value_to_tensor(numpy_data):
    tensor_data = copy.deepcopy(numpy_data)
    for key, value in tensor_data.items():
        tensor_data[key] = torch.from_numpy(np.array(value))
    return tensor_data


def __convert_tensor_value_to_numpy(tensor_data):
    numpy_data = copy.deepcopy(tensor_data)
    for key, value in numpy_data.items():
        numpy_data[key] = value.cpu().numpy()
    return numpy_data


# compress object to base64 string
def __compress_data(data):
    encoded = json.dumps(data, sort_keys=True, indent=4, ensure_ascii=False, cls=NumpyEncoder).encode(
        'utf8')
    compressed_data = gzip.compress(encoded)
    b64_encoded = base64.b64encode(compressed_data)
    return b64_encoded.decode('ascii')


# based64 decode to byte, and then decompress it
def __decompress_data(data):
    base64_decoded = base64.b64decode(data)
    decompressed = gzip.decompress(base64_decoded)
    return json.loads(decompressed)


# compress the tensor data
def compress_tensor(data):
    compressed_data = __compress_data(__convert_tensor_value_to_numpy(data))
    return compressed_data


# decompress the data into tensor
def decompress_tensor(data):
    tensor_data = __conver_numpy_value_to_tensor(__decompress_data(data))
    return tensor_data


# generate md5 hash for global model. Require a tensor type gradients.
def generate_md5_hash(model_weights):
    np_model_weights = __convert_tensor_value_to_numpy(model_weights)
    data_md5 = hashlib.md5(json.dumps(np_model_weights, sort_keys=True, cls=NumpyEncoder).encode('utf-8')).hexdigest()
    return data_md5


def disturb_w(w):
    disturbed_w = copy.deepcopy(w)
    for name, param in w.items():
        beta = (random.random()-0.5)*2  # disturb value in range [-1, 1)
        transformed_w = param * beta
        disturbed_w[name] = transformed_w
    return disturbed_w


def get_ip(test_ip_addr):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        # s.connect(('10.255.255.255', 1))
        s.connect((test_ip_addr, 1))
        ip = s.getsockname()[0]
        logger.debug("Detected IP address: " + ip)
    except socket.error:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


# here starts built in functions
shutdown_count_num = 0
ipMap = {}


def shutdown_count(uuid, from_ip, fed_listen_port, num_users):
    lock.acquire()
    global shutdown_count_num
    global ipMap
    shutdown_count_num += 1
    ipMap[uuid] = from_ip
    lock.release()
    if shutdown_count_num == num_users:
        # send request to shut down the python
        body_data = {
            'message': 'shutdown',
        }
        logger.debug('Send shutdown python request.')
        for uuid in ipMap.keys():
            client_url = "http://" + ipMap[uuid] + ":" + str(fed_listen_port) + "/trigger"
            http_client_post(client_url, body_data)


# time_list: [total_time, round_time, train_time, test_time, commu_time]
# acc_list: [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4]  (for cnn or mlp)
#       or: [loss_mse, loss_mae]  (for lstm)
# model: cnn, mlp, or lstm
def record_log(user_id, epoch, time_list, acc_list, model, clean=False):
    filename = "result-record_" + str(user_id) + ".txt"

    # first time clean the file
    if clean:
        open(filename, 'w').close()

    if model == "lstm":
        with open(filename, "a") as time_record_file:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            time_record_file.write(current_time + "[" + "{:03d}".format(epoch) + "]"
                                   + " <Total Time> " + str(time_list[0])[:8]
                                   + " <Round Time> " + str(time_list[1])[:8]
                                   + " <Train Time> " + str(time_list[2])[:8]
                                   + " <Test Time> " + str(time_list[3])[:8]
                                   + " <Communication Time> " + str(time_list[4])[:8]
                                   + " <mse_loss> " + str(acc_list[0])[:8]
                                   + " <mae_loss> " + str(acc_list[1])[:8]
                                   + "\n")
    else:
        with open(filename, "a") as time_record_file:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            time_record_file.write(current_time + "[" + "{:03d}".format(epoch) + "]"
                                   + " <Total Time> " + str(time_list[0])[:8]
                                   + " <Round Time> " + str(time_list[1])[:8]
                                   + " <Train Time> " + str(time_list[2])[:8]
                                   + " <Test Time> " + str(time_list[3])[:8]
                                   + " <Communication Time> " + str(time_list[4])[:8]
                                   + " <acc_local> " + str(acc_list[0])[:8]
                                   + " <acc_local_skew1> " + str(acc_list[1])[:8]
                                   + " <acc_local_skew2> " + str(acc_list[2])[:8]
                                   + " <acc_local_skew3> " + str(acc_list[3])[:8]
                                   + " <acc_local_skew4> " + str(acc_list[4])[:8]
                                   + "\n")


def my_exit(exit_sleep):
    time.sleep(exit_sleep)  # sleep for a while before exit
    logger.info("########## PYTHON SHUTTING DOWN! ##########")
    os._exit(0)
