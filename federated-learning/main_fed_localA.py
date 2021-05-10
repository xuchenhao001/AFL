#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import asyncio
import base64
import gzip
import json
import logging
import os
import socket
import sys
import time
import subprocess
import copy
import numpy as np
import threading
import torch
from tornado import httpclient, ioloop, web, httpserver, gen

from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img_total
from utils.util import dataset_loader, model_loader, ColoredLogger

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger("main_fed_localA")

torch.manual_seed(0)
np.random.seed(0)

# TO BE CHANGED
# wait in seconds for other nodes to start
start_wait_time = 15
# federated learning server listen port
fed_listen_port = 8888
# TO BE CHANGED FINISHED

# NOT TO TOUCH VARIABLES BELOW
trigger_url = ""
peer_address_list = []
g_user_id = 0
lock = threading.Lock()
wMap = []
wLocalsMap = {}
wLocalsPerMap = {}
hyperparaMap = {}
ipMap = {}
net_glob = None
args = None
dataset_train = None
dataset_test = None
dict_users = []
test_users = []
skew_users = []
g_start_time = {}
g_train_time = {}

differenc1 = None
differenc2 = None


# returns variable from sourcing a file
def env_from_sourcing(file_to_source_path, variable_name):
    source = 'source %s && export MYVAR=$(echo "${%s[@]}")' % (file_to_source_path, variable_name)
    dump = '/usr/bin/python3 -c "import os, json; print(os.getenv(\'MYVAR\'))"'
    pipe = subprocess.Popen(['/bin/bash', '-c', '%s && %s' % (source, dump)], stdout=subprocess.PIPE)
    # return json.loads(pipe.stdout.read())
    return pipe.stdout.read().decode("utf-8").rstrip()


# init: loads the dataset and global model
def init():
    global net_glob
    global dataset_train
    global dataset_test
    global dict_users
    global test_users
    global skew_users

    dataset_train, dataset_test, dict_users, test_users, skew_users = dataset_loader(args.dataset, args.iid,
                                                                                         args.num_users)
    if dict_users is None:
        logger.error('Error: unrecognized dataset')
        sys.exit()
    img_size = dataset_train[0][0].shape
    net_glob = model_loader(args.model, args.dataset, args.device, args.num_channels, args.num_classes, img_size)
    if net_glob is None:
        logger.error('Error: unrecognized model')
        sys.exit()


async def train(user_id, epochs, w_glob_local, w_locals, w_locals_per, hyperpara, start_time):
    global differenc1
    global differenc2
    if user_id is None:
        user_id = await fetch_user_id()

    if epochs is None:
        # calculate initial model accuracy, record it as the bench mark.
        idx = int(user_id) - 1
        net_glob.eval()
        idx_total = [test_users[idx], skew_users[0][idx], skew_users[1][idx], skew_users[2][idx],
                     skew_users[3][idx]]
        correct = test_img_total(net_glob, dataset_test, idx_total, args)
        acc_local = torch.div(100.0 * correct[0], len(test_users[idx]))
        filename = "result-record_" + str(user_id) + ".txt"
        # first time clean the file
        open(filename, 'w').close()

        with open(filename, "a") as time_record_file:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            time_record_file.write(current_time + "[00]"
                                   + " <Total Time> 0.0"
                                   + " <Train Time> 0.0"
                                   + " <Test Time> 0.0"
                                   + " <Communication Time> 0.0"
                                   + " <Alpha> 0.0"
                                   + " <acc_local> " + str(acc_local.item())[:8]
                                   + " <acc_local_skew1> 0.0"
                                   + " <acc_local_skew2> 0.0"
                                   + " <acc_local_skew3> 0.0"
                                   + " <acc_local_skew4> 0.0"
                                   + "\n")

        epochs = args.epochs
        # initialize weights of model
        net_glob.train()
        w_glob = net_glob.state_dict()  # global model initialization
        w_local = copy.deepcopy(w_glob)
        differenc1 = copy.deepcopy(w_glob)
        differenc2 = copy.deepcopy(w_glob)
        # for the first epoch, init user local parameters, w,v,v_bar,alpha
        w_glob_local = copy.deepcopy(w_glob)
        w_locals = copy.deepcopy(w_local)
        w_locals_per = copy.deepcopy(w_local)
        hyperpara = args.hyper
    else:
        w_glob_local = conver_numpy_value_to_tensor(decompress_data(w_glob_local))
        w_locals = conver_numpy_value_to_tensor(decompress_data(w_locals))
        w_locals_per = conver_numpy_value_to_tensor(decompress_data(w_locals_per))
        w_glob = copy.deepcopy(w_glob_local)

    # training for all epochs
    for iter in reversed(range(epochs)):
        logger.info("Epoch [" + str(iter+1) + "] train for user [" + str(user_id) + "]")
        train_start_time = time.time()
        # compute v_bar
        for j in w_glob.keys():
            w_locals_per[j] = hyperpara * w_locals[j] + (1 - hyperpara) * w_glob_local[j]
            differenc1[j] = w_locals[j] - w_glob_local[j]

        # train local global weight
        net_glob.load_state_dict(w_glob_local)
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id - 1])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        for j in w_glob.keys():
            w_glob_local[j] = copy.deepcopy(w[j])

        # train local model weight
        net_glob.load_state_dict(w_locals_per)
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id - 1])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        # loss_locals.append(copy.deepcopy(loss))

        for j in w_glob.keys():
            differenc2[j] = (w[j] - w_locals[j]) * 100.0
            w_locals[j] = copy.deepcopy(w[j])

            # update adaptive alpha
        d1, d2 = [], []
        correlation = 0.0
        l = 0
        for j in w_glob.keys():
            d = differenc1[j].numpy()
            d1 = np.ndarray.flatten(d)
            d = differenc2[j].numpy()
            d2 = np.ndarray.flatten(d)
            l = l + 1
            correlation = correlation + np.dot(d1, d2)
        correlation = correlation / l
        hyperpara = round((hyperpara - args.lr * correlation), 2)
        if hyperpara > 1.0:
            hyperpara = 1.0

        # update local personalized weight
        for j in w_glob.keys():
            w_locals_per[j] = hyperpara * w_locals[j] + (1 - hyperpara) * w_glob_local[j]
        train_time = time.time() - train_start_time

        # start test
        test_start_time = time.time()
        idx = int(user_id) - 1
        idx_total = [test_users[idx], skew_users[0][idx], skew_users[1][idx], skew_users[2][idx], skew_users[3][idx]]
        correct = test_img_total(net_glob, dataset_test, idx_total, args)
        acc_local = torch.div(100.0 * correct[0], len(test_users[idx]))
        # skew 5%
        acc_local_skew1 = torch.div(100.0 * (correct[0] + correct[1]), (len(test_users[idx]) + len(skew_users[0][idx])))
        # skew 10%
        acc_local_skew2 = torch.div(100.0 * (correct[0] + correct[2]), (len(test_users[idx]) + len(skew_users[1][idx])))
        # skew 15%
        acc_local_skew3 = torch.div(100.0 * (correct[0] + correct[3]), (len(test_users[idx]) + len(skew_users[2][idx])))
        # skew 20%
        acc_local_skew4 = torch.div(100.0 * (correct[0] + correct[4]), (len(test_users[idx]) + len(skew_users[3][idx])))
        test_time = time.time() - test_start_time

        # before start next round, record the time
        filename = "result-record_" + str(user_id) + ".txt"

        with open(filename, "a") as time_record_file:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            total_time = time.time() - start_time
            communication_time = total_time - train_time - test_time
            if communication_time < 0.001:
                communication_time = 0.0
            time_record_file.write(current_time + "[" + f"{iter + 1:0>2}" + "]"
                                   + " <Total Time> " + str(total_time)[:8]
                                   + " <Train Time> " + str(train_time)[:8]
                                   + " <Test Time> " + str(test_time)[:8]
                                   + " <Communication Time> " + str(communication_time)[:8]
                                   + " <Alpha> 0.0"
                                   + " <acc_local> " + str(acc_local.item())[:8]
                                   + " <acc_local_skew1> " + str(acc_local_skew1.item())[:8]
                                   + " <acc_local_skew2> " + str(acc_local_skew2.item())[:8]
                                   + " <acc_local_skew3> " + str(acc_local_skew3.item())[:8]
                                   + " <acc_local_skew4> " + str(acc_local_skew4.item())[:8]
                                   + "\n")
        start_time = time.time()
        if (iter + 1) % 10 == 0:  # update global model
            from_ip = get_ip()
            await upload_local_w(user_id, iter, from_ip, w_glob_local, w_locals, w_locals_per,
                                                 hyperpara, start_time)
            return

    logger.info("########## ALL DONE! ##########")
    await gen.sleep(600)  # sleep 600 seconds before exit
    os._exit(0)


class MultiTrainThread(threading.Thread):
    def __init__(self, user_id, epochs, w_glob_local, w_locals, w_locals_per, hyperpara, start_time):
        threading.Thread.__init__(self)
        self.user_id = user_id
        self.epochs = epochs
        self.w_glob_local = w_glob_local
        self.w_locals = w_locals
        self.w_locals_per = w_locals_per
        self.hyperpara = hyperpara
        self.start_time = start_time

    def run(self):
        time.sleep(start_wait_time)
        logger.debug("start new thread")
        loop = asyncio.new_event_loop()
        if self.start_time is None:
            self.start_time = time.time()
        loop.run_until_complete(train(self.user_id, self.epochs, self.w_glob_local, self.w_locals, self.w_locals_per,
                                      self.hyperpara, self.start_time))
        logger.debug("end thread")


def test(data):
    detail = {"data": data}
    return detail


async def load_user_id():
    lock.acquire()
    global g_user_id
    g_user_id += 1
    detail = {"user_id": g_user_id}
    lock.release()
    return detail


async def release_global_w(epochs):
    lock.acquire()
    global g_user_id
    global wMap
    g_user_id = 0
    lock.release()
    w_glob_local = FedAvg(wMap)
    wMap = []  # release wMap after aggregation
    w_glob_local = compress_data(convert_tensor_value_to_numpy(w_glob_local))
    for user_id in ipMap.keys():
        key = str(user_id) + "-" + str(epochs)
        start_time = g_start_time.get(key)
        w_locals = wLocalsMap.get(user_id)
        w_locals_per = wLocalsPerMap.get(user_id)
        hyperpara = hyperparaMap.get(user_id)
        data = {
            'message': 'release_global_w',
            'user_id': user_id,
            'epochs': epochs,
            'w_glob_local': w_glob_local,
            'w_locals': w_locals,
            'w_locals_per': w_locals_per,
            'hyperpara': hyperpara,
            'start_time': start_time,
        }
        json_body = json.dumps(data, sort_keys=True, indent=4, ensure_ascii=False, cls=NumpyEncoder).encode('utf8')
        my_url = "http://" + ipMap[user_id] + ":" + str(fed_listen_port) + "/trigger"
        asyncio.ensure_future(http_client_post(my_url, json_body, 'release_global_w'))


async def average_local_w(user_id, epochs, from_ip, w_glob_local, w_locals, w_locals_per, hyperpara, start_time):
    logger.debug("received average request from user: " + str(user_id))
    lock.acquire()
    global wMap
    global wLocalsMap
    global wLocalsPerMap
    global hyperparaMap
    global ipMap

    global g_start_time
    global g_train_time
    key = str(user_id) + "-" + str(epochs)
    g_start_time[key] = start_time
    ipMap[user_id] = from_ip

    # update wMap (w_glob_local) to be averaged
    w_glob_local = conver_numpy_value_to_tensor(decompress_data(w_glob_local))
    wMap.append(w_glob_local)
    # update wLocalsMap
    wLocalsMap[user_id] = w_locals
    wLocalsPerMap[user_id] = w_locals_per
    hyperparaMap[user_id] = hyperpara
    lock.release()
    if len(wMap) == args.num_users:
        logger.debug("Gathered enough w, average and release them")
        asyncio.ensure_future(release_global_w(epochs))


async def http_client_post(url, json_body, message="None"):
    logger.debug("Start http client post [" + message + "] to: " + url)
    method = "POST"
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    http_client = httpclient.AsyncHTTPClient()
    try:
        request = httpclient.HTTPRequest(url=url, method=method, headers=headers, body=json_body, connect_timeout=300,
                                         request_timeout=300)
        response = await http_client.fetch(request)
        logger.debug("[HTTP Success] [" + message + "] SERVICE RESPONSE: %s" % response.body)
        return response.body
    except Exception as e:
        logger.error("[HTTP Error] [" + message + "] SERVICE RESPONSE: %s" % e)
        return None


async def fetch_user_id():
    fetch_data = {
        'message': 'fetch_user_id',
    }
    json_body = json.dumps(fetch_data, sort_keys=True, indent=4, ensure_ascii=False).encode('utf8')
    response = await http_client_post(trigger_url, json_body, 'fetch_user_id')
    responseObj = json.loads(response)
    detail = responseObj.get("detail")
    user_id = detail.get("user_id")
    return user_id


async def upload_local_w(user_id, epochs, from_ip, w_glob_local, w_locals, w_locals_per, hyperpara, start_time):
    w_glob_local = compress_data(convert_tensor_value_to_numpy(w_glob_local))
    w_locals = compress_data(convert_tensor_value_to_numpy(w_locals))
    w_locals_per = compress_data(convert_tensor_value_to_numpy(w_locals_per))
    upload_data = {
        'message': 'upload_local_w',
        'user_id': user_id,
        'epochs': epochs,
        'w_glob_local': w_glob_local,
        'w_locals': w_locals,
        'w_locals_per': w_locals_per,
        'hyperpara': hyperpara,
        'from_ip': from_ip,
        'start_time': start_time,
    }
    json_body = json.dumps(upload_data, sort_keys=True, indent=4, ensure_ascii=False, cls=NumpyEncoder).encode('utf8')
    await http_client_post(trigger_url, json_body, 'upload_local_w')
    return


def conver_numpy_value_to_tensor(numpy_data):
    tensor_data = copy.deepcopy(numpy_data)
    for key, value in tensor_data.items():
        tensor_data[key] = torch.from_numpy(np.array(value))
    return tensor_data


def convert_tensor_value_to_numpy(tensor_data):
    numpy_data = copy.deepcopy(tensor_data)
    for key, value in numpy_data.items():
        numpy_data[key] = value.cpu().numpy()
    return numpy_data


# compress object to base64 string
def compress_data(data):
    encoded = json.dumps(data, sort_keys=True, indent=4, ensure_ascii=False, cls=NumpyEncoder).encode(
        'utf8')
    compressed_data = gzip.compress(encoded)
    b64_encoded = base64.b64encode(compressed_data)
    return b64_encoded.decode('ascii')


# based64 decode to byte, and then decompress it
def decompress_data(data):
    base64_decoded = base64.b64decode(data)
    decompressed = gzip.decompress(base64_decoded)
    return json.loads(decompressed)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
        logger.debug("Detected IP address: " + IP)
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


class MainHandler(web.RequestHandler):

    async def get(self):
        response = {"status": "yes", "detail": "test"}
        in_json = json.dumps(response, sort_keys=True, indent=4, ensure_ascii=False).encode('utf8')
        self.set_header("Content-Type", "application/json")
        self.write(in_json)

    async def post(self):
        data = json.loads(self.request.body)
        status = "yes"
        detail = {}
        self.set_header("Content-Type", "application/json")

        message = data.get("message")
        if message == "test":
            detail = test(data.get("weight"))
        elif message == "fetch_user_id":
            detail = await load_user_id()
        elif message == "upload_local_w":
            asyncio.ensure_future(average_local_w(data.get("user_id"), data.get("epochs"), data.get("from_ip"),
                                  data.get("w_glob_local"), data.get("w_locals"), data.get("w_locals_per"),
                                  data.get("hyperpara"), data.get("start_time")))
        elif message == "release_global_w":
            thread_train = MultiTrainThread(data.get("user_id"), data.get("epochs"), data.get("w_glob_local"),
                                            data.get("w_locals"), data.get("w_locals_per"), data.get("hyperpara"),
                                            data.get("start_time"))
            thread_train.start()

        response = {"status": status, "detail": detail}
        in_json = json.dumps(response, sort_keys=True, indent=4, ensure_ascii=False).encode('utf8')
        self.write(in_json)


def main():
    global args
    global peer_address_list
    global trigger_url

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    logger.setLevel(args.log_level)

    # parse network.config and read the peer addresses
    real_path = os.path.dirname(os.path.realpath(__file__))
    peer_address_var = env_from_sourcing(os.path.join(real_path, "../fabric-samples/network.config"), "PeerAddress")
    peer_address_list = peer_address_var.split(' ')
    peer_addrs = [peer_addr.split(":")[0] for peer_addr in peer_address_list]
    peer_header_addr = peer_addrs[0]
    trigger_url = "http://" + peer_header_addr + ":" + str(fed_listen_port) + "/trigger"

    # parse participant number
    args.num_users = len(peer_address_list)

    # init dataset and global model
    init()

    # multi-thread training here
    my_ip = get_ip()
    threads = []
    for addr in peer_addrs:
        if addr == my_ip:
            thread_train = MultiTrainThread(None, None, None, None, None, None, None)
            threads.append(thread_train)

    # Start all threads
    for thread in threads:
        thread.start()

    app = web.Application([
        (r"/trigger", MainHandler),
    ])
    http_server = httpserver.HTTPServer(app, max_buffer_size=10485760000)  # 10GB
    http_server.listen(fed_listen_port)
    logger.info("start serving at " + str(fed_listen_port) + "...")
    ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()

