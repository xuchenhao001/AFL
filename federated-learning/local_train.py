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

import utils
from utils.options import args_parser
from models.Update import LocalUpdate
from utils.util import dataset_loader, model_loader, ColoredLogger

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger("local_train")

# TO BE CHANGED
# federated learning server listen port
fed_listen_port = 8888
# used for self ip address testing
test_ip_addr = "10.150.187.13"
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
g_init_time = {}


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

    dataset_train, dataset_test, dict_users, test_users, skew_users = \
        utils.util.dataset_loader(args.dataset, args.dataset_train_size, args.dataset_test_size, args.iid,
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
    global g_init_time

    if user_id is None:
        user_id = await fetch_user_id()

    # training for all epochs
    for iter in reversed(range(args.epochs)):
        # calculate initial model accuracy, record it as the bench mark.
        idx = int(user_id) - 1
        if iter+1 == args.epochs:
            # for the first time do the training
            g_init_time[str(user_id)] = time.time()
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

        logger.info("Epoch [" + str(iter+1) + "] train for user [" + str(user_id) + "]")
        train_start_time = time.time()
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id - 1])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
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
            round_time = time.time() - train_start_time
            time_record_file.write(current_time + "[" + f"{iter + 1:0>2}" + "]"
                                   + " <Total Time> " + str(total_time)[:8]
                                   + " <Round Time> " + str(round_time)[:8]
                                   + " <Train Time> " + str(train_time)[:8]
                                   + " <Test Time> " + str(test_time)[:8]
                                   + " <Communication Time> 0.0"
                                   + " <acc_local> " + str(acc_local)[:8]
                                   + " <acc_local_skew1> " + str(acc_local_skew1)[:8]
                                   + " <acc_local_skew2> " + str(acc_local_skew2)[:8]
                                   + " <acc_local_skew3> " + str(acc_local_skew3)[:8]
                                   + " <acc_local_skew4> " + str(acc_local_skew4)[:8]
                                   + "\n")

        # update net_glob for next round
        net_glob.load_state_dict(w)

    logger.info("########## ALL DONE! ##########")
    body_data = {
        'message': 'shutdown_python'
    }
    await utils.util.http_client_post(trigger_url, body_data)


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


async def fetch_user_id():
    fetch_data = {
        'message': 'fetch_user_id',
    }
    response = await utils.util.http_client_post(trigger_url, fetch_data)
    responseObj = json.loads(response)
    detail = responseObj.get("detail")
    user_id = detail.get("user_id")
    return user_id


async def shutdown_count():
    global shutdown_count_num
    lock.acquire()
    shutdown_count_num += 1
    lock.release()
    if shutdown_count_num == args.num_users:
        # send request to blockchain for shutting down the python
        body_data = {
            'message': 'ShutdownPython',
            'data': {},
            'uuid': "",
            'epochs': 0,
            'is_sync': False
        }
        logger.debug('Sent shutdown python request to blockchain.')
        await utils.util.http_client_post(trigger_url, body_data)


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
        elif message == "shutdown_python":
            detail = await shutdown_count()

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
    peer_address_var = env_from_sourcing(os.path.join(real_path, "../fabric-network/network.config"), "PeerAddress")
    peer_address_list = peer_address_var.split(' ')
    peer_addrs = [peer_addr.split(":")[0] for peer_addr in peer_address_list]
    peer_header_addr = peer_addrs[0]
    trigger_url = "http://" + peer_header_addr + ":" + str(fed_listen_port) + "/trigger"

    # parse participant number
    args.num_users = len(peer_address_list)

    # init dataset and global model
    init()

    # multi-thread training here
    my_ip = utils.util.get_ip(test_ip_addr)
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
