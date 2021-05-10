#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import asyncio
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
from tornado import httpclient, ioloop, web, gen

from utils.options import args_parser
from models.Update import LocalUpdate
from models.test import test_img_total
from utils.util import dataset_loader, model_loader, ColoredLogger

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger("main_nn")

torch.manual_seed(0)
np.random.seed(0)

# TO BE CHANGED
# federated learning server listen port
fed_listen_port = 8888
# TO BE CHANGED FINISHED

# NOT TO TOUCH VARIABLES BELOW
args = None
net_glob = None
trigger_url = ""
peer_address_list = []
dataset_train = None
dataset_test = None
dict_users = []
test_users = []
skew_users = []
g_user_id = 0
lock = threading.Lock()


# returns variable from sourcing a file
def env_from_sourcing(file_to_source_path, variable_name):
    source = 'source %s && export MYVAR=$(echo "${%s[@]}")' % (file_to_source_path, variable_name)
    dump = '/usr/bin/python3 -c "import os, json; print(os.getenv(\'MYVAR\'))"'
    pipe = subprocess.Popen(['/bin/bash', '-c', '%s && %s' % (source, dump)], stdout=subprocess.PIPE)
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
    # initialize weights of model
    net_glob.train()


async def train(user_id):
    global args

    if user_id is None:
        user_id = await fetch_user_id()

    # training for all epochs
    for iter in reversed(range(args.epochs)):
        # calculate initial model accuracy, record it as the bench mark.
        idx = int(user_id) - 1
        if iter+1 == args.epochs:
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

        logger.info("Epoch [" + str(iter+1) + "] train for user [" + str(user_id) + "]")
        train_start_time = time.time()
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id - 1])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
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
        # # first time clean the file
        # if iter + 1 == args.epochs:
        #     with open(filename, 'w') as f:
        #         pass

        with open(filename, "a") as time_record_file:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            total_time = time.time() - train_start_time
            time_record_file.write(current_time + "[" + f"{iter + 1:0>2}" + "]"
                                   + " <Total Time> " + str(total_time)[:8]
                                   + " <Train Time> " + str(train_time)[:8]
                                   + " <Test Time> " + str(test_time)[:8]
                                   + " <Communication Time> 0.0"
                                   + " <Alpha> 0.0"
                                   + " <acc_local> " + str(acc_local.item())[:8]
                                   + " <acc_local_skew1> " + str(acc_local_skew1.item())[:8]
                                   + " <acc_local_skew2> " + str(acc_local_skew2.item())[:8]
                                   + " <acc_local_skew3> " + str(acc_local_skew3.item())[:8]
                                   + " <acc_local_skew4> " + str(acc_local_skew4.item())[:8]
                                   + "\n")

        # update net_glob for next round
        net_glob.load_state_dict(w)

    logger.info("########## ALL DONE! ##########")
    await gen.sleep(600)  # sleep 600 seconds before exit
    os._exit(0)


class MultiTrainThread(threading.Thread):
    def __init__(self, user_id):
        threading.Thread.__init__(self)
        self.user_id = user_id

    def run(self):
        logger.debug("start new thread")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(train(self.user_id))
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

        response = {"status": status, "detail": detail}
        in_json = json.dumps(response, sort_keys=True, indent=4, ensure_ascii=False).encode('utf8')
        self.write(in_json)


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
            thread_train = MultiTrainThread(None)
            threads.append(thread_train)

    # Start all threads
    for thread in threads:
        thread.start()

    app = web.Application([
        (r"/trigger", MainHandler),
    ])
    app.listen(fed_listen_port)
    logger.info("start serving at " + str(fed_listen_port) + "...")
    ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
