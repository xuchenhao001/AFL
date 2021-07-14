import asyncio
import json
import logging
import os
import sys
import time
import subprocess
import copy
from abc import ABC

import numpy as np
import threading
import torch
from tornado import ioloop, web, httpserver, gen

import utils
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedAvg
from utils.util import dataset_loader, model_loader, ColoredLogger

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger("fed_localA")

# TO BE CHANGED
# wait in seconds for other nodes to start
start_wait_time = 15
# federated learning server listen port
fed_listen_port = 8888
# TO BE CHANGED FINISHED

# NOT TO TOUCH VARIABLES BELOW
trigger_url = ""
test_ip_addr = ""
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
g_init_time = {}
g_train_global_model = None
g_train_global_model_epoch = None
shutdown_count_num = 0
exit_sleep = 0

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
    global g_train_global_model
    global g_train_global_model_epoch

    dataset_train, dataset_test, dict_users, test_users, skew_users = \
        utils.util.dataset_loader(args.dataset, args.dataset_train_size, args.iid, args.num_users)
    if dict_users is None:
        logger.error('Error: unrecognized dataset')
        sys.exit()
    img_size = dataset_train[0][0].shape
    net_glob = model_loader(args.model, args.dataset, args.device, args.num_channels, args.num_classes, img_size)
    if net_glob is None:
        logger.error('Error: unrecognized model')
        sys.exit()
    w = net_glob.state_dict()
    g_train_global_model = utils.util.compress_tensor(w)
    g_train_global_model_epoch = -1  # -1 means the initial global model


async def train(user_id, epochs, w_glob_local, w_locals, w_locals_per, hyperpara, start_time):
    global differenc1
    global differenc2
    global g_init_time
    if user_id is None:
        user_id = await fetch_user_id()

    if epochs is None:
        # download initial global model
        body_data = {
            'message': 'global_model',
            'epochs': -1,
        }
        logger.debug('fetch initial global model from: %s' % trigger_url)
        result = await utils.util.http_client_post(trigger_url, body_data)
        responseObj = json.loads(result)
        detail = responseObj.get("detail")
        global_model_compressed = detail.get("global_model")
        w_glob = utils.util.decompress_tensor(global_model_compressed)
        logger.debug('Downloaded initial global model hash: ' + utils.util.generate_md5_hash(w_glob))
        net_glob.load_state_dict(w_glob)
        # calculate initial model accuracy, record it as the bench mark.
        g_init_time[str(user_id)] = start_time
        idx = int(user_id) - 1
        net_glob.eval()
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            utils.util.test_img(test_users, skew_users, idx, net_glob, dataset_test, args)
        filename = "result-record_" + str(user_id) + ".txt"
        # first time clean the file
        open(filename, 'w').close()

        with open(filename, "a") as time_record_file:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            time_record_file.write(current_time + "[00]"
                                   + " <Total Time> 0.0"
                                   + " <Round Time> 0.0"
                                   + " <Train Time> 0.0"
                                   + " <Test Time> 0.0"
                                   + " <Communication Time> 0.0"
                                   + " <acc_local> " + str(acc_local)[:8]
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
        w_glob_local = utils.util.decompress_tensor(w_glob_local)
        w_locals = utils.util.decompress_tensor(w_locals)
        w_locals_per = utils.util.decompress_tensor(w_locals_per)
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
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            utils.util.test_img(test_users, skew_users, idx, net_glob, dataset_test, args)
        test_time = time.time() - test_start_time

        # before start next round, record the time
        filename = "result-record_" + str(user_id) + ".txt"

        with open(filename, "a") as time_record_file:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            total_time = time.time() - g_init_time[str(user_id)]
            round_time = time.time() - start_time
            communication_time = round_time - train_time - test_time
            if communication_time < 0.001:
                communication_time = 0.0
            time_record_file.write(current_time + "[" + f"{iter + 1:0>2}" + "]"
                                   + " <Total Time> " + str(total_time)[:8]
                                   + " <Round Time> " + str(round_time)[:8]
                                   + " <Train Time> " + str(train_time)[:8]
                                   + " <Test Time> " + str(test_time)[:8]
                                   + " <Communication Time> " + str(communication_time)[:8]
                                   + " <acc_local> " + str(acc_local)[:8]
                                   + " <acc_local_skew1> " + str(acc_local_skew1)[:8]
                                   + " <acc_local_skew2> " + str(acc_local_skew2)[:8]
                                   + " <acc_local_skew3> " + str(acc_local_skew3)[:8]
                                   + " <acc_local_skew4> " + str(acc_local_skew4)[:8]
                                   + "\n")
        start_time = time.time()
        if (iter + 1) % 10 == 0:  # update global model
            from_ip = utils.util.get_ip(test_ip_addr)
            await upload_local_w(user_id, iter, from_ip, w_glob_local, w_locals, w_locals_per,
                                                 hyperpara, start_time)
            return

    logger.info("########## ALL DONE! ##########")
    from_ip = utils.util.get_ip(test_ip_addr)
    body_data = {
        'message': 'shutdown_python',
        'uuid': user_id,
        'from_ip': from_ip,
    }
    await utils.util.http_client_post(trigger_url, body_data)


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
        # time.sleep(start_wait_time)
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
    w_glob_local = utils.util.compress_tensor(w_glob_local)
    for user_id in ipMap.keys():
        key = str(user_id) + "-" + str(epochs)
        start_time = g_start_time.get(key)
        w_locals = wLocalsMap.get(user_id)
        w_locals_per = wLocalsPerMap.get(user_id)
        hyperpara = hyperparaMap.get(user_id)
        json_body = {
            'message': 'release_global_w',
            'user_id': user_id,
            'epochs': epochs,
            'w_glob_local': w_glob_local,
            'w_locals': w_locals,
            'w_locals_per': w_locals_per,
            'hyperpara': hyperpara,
            'start_time': start_time,
        }
        my_url = "http://" + ipMap[user_id] + ":" + str(fed_listen_port) + "/trigger"
        asyncio.ensure_future(utils.util.http_client_post(my_url, json_body))


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
    w_glob_local = utils.util.decompress_tensor(w_glob_local)
    wMap.append(w_glob_local)
    # update wLocalsMap
    wLocalsMap[user_id] = w_locals
    wLocalsPerMap[user_id] = w_locals_per
    hyperparaMap[user_id] = hyperpara
    lock.release()
    if len(wMap) == args.num_users:
        logger.debug("Gathered enough w, average and release them")
        asyncio.ensure_future(release_global_w(epochs))


async def fetch_user_id():
    fetch_data = {
        'message': 'fetch_user_id',
    }
    response = await utils.util.http_client_post(trigger_url, fetch_data)
    responseObj = json.loads(response)
    detail = responseObj.get("detail")
    user_id = detail.get("user_id")
    return user_id


async def upload_local_w(user_id, epochs, from_ip, w_glob_local, w_locals, w_locals_per, hyperpara, start_time):
    w_glob_local = utils.util.compress_tensor(w_glob_local)
    w_locals = utils.util.compress_tensor(w_locals)
    w_locals_per = utils.util.compress_tensor(w_locals_per)
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
    await utils.util.http_client_post(trigger_url, upload_data)
    return


async def download_global_model(epochs):
    if epochs == g_train_global_model_epoch:
        detail = {
            "global_model": g_train_global_model,
        }
    else:
        detail = {
            "global_model": None,
        }
    return detail


class MainHandler(web.RequestHandler, ABC):

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
        elif message == "global_model":
            detail = await download_global_model(data.get("epochs"))
        elif message == "upload_local_w":
            asyncio.ensure_future(average_local_w(data.get("user_id"), data.get("epochs"), data.get("from_ip"),
                                  data.get("w_glob_local"), data.get("w_locals"), data.get("w_locals_per"),
                                  data.get("hyperpara"), data.get("start_time")))
        elif message == "release_global_w":
            thread_train = MultiTrainThread(data.get("user_id"), data.get("epochs"), data.get("w_glob_local"),
                                            data.get("w_locals"), data.get("w_locals_per"), data.get("hyperpara"),
                                            data.get("start_time"))
            thread_train.start()
        elif message == "shutdown_python":
            detail = await utils.util.shutdown_count(data.get("uuid"), data.get("from_ip"), fed_listen_port, lock,
                                                     args.num_users)
        elif message == "shutdown":
            asyncio.ensure_future(utils.util.my_exit(exit_sleep))

        response = {"status": status, "detail": detail}
        in_json = json.dumps(response, sort_keys=True, indent=4, ensure_ascii=False).encode('utf8')
        self.write(in_json)


def main():
    global args
    global peer_address_list
    global trigger_url
    global test_ip_addr
    global exit_sleep

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    logger.setLevel(args.log_level)

    # parse network.config and read the peer addresses
    real_path = os.path.dirname(os.path.realpath(__file__))
    peer_address_var = env_from_sourcing(os.path.join(real_path, "../fabric-network/network.config"), "PeerAddress")
    peer_address_list = peer_address_var.split(' ')
    peer_addrs = [peer_addr.split(":")[0] for peer_addr in peer_address_list]
    peer_header_addr = peer_addrs[0]
    trigger_url = "http://" + peer_header_addr + ":" + str(fed_listen_port) + "/trigger"

    # parse participant number
    args.num_users = len(peer_address_list)

    # parse test ip addr
    test_ip_addr = args.test_ip_addr
    exit_sleep = args.exit_sleep

    # init dataset and global model
    init()

    # multi-thread training here
    my_ip = utils.util.get_ip(test_ip_addr)
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

