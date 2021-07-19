import logging
import os
import sys
import time
import copy
import numpy as np
import threading
import torch
from flask import Flask, request

import utils
from utils.options import args_parser
from models.Update import local_update
from models.Fed import FedAvg
from utils.util import dataset_loader, model_loader, ColoredLogger

logging.setLoggerClass(ColoredLogger)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logger = logging.getLogger("fed_localA")

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
g_init_time = {}
g_train_global_model = None
g_train_global_model_epoch = None

differenc1 = None
differenc2 = None


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
        dataset_loader(args.dataset, args.dataset_train_size, args.iid, args.num_users)
    if dataset_train is None:
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


def train(user_id, epochs, w_glob_local, w_locals, w_locals_per, hyperpara, start_time):
    global differenc1
    global differenc2
    global g_init_time
    if user_id is None:
        user_id = fetch_user_id()

    if epochs is None:
        # download initial global model
        body_data = {
            'message': 'global_model',
            'epochs': -1,
        }
        logger.debug('fetch initial global model from: %s' % trigger_url)
        result = utils.util.http_client_post(trigger_url, body_data)
        detail = result.get("detail")
        global_model_compressed = detail.get("global_model")
        w_glob = utils.util.decompress_tensor(global_model_compressed)
        logger.debug('Downloaded initial global model hash: ' + utils.util.generate_md5_hash(w_glob))
        net_glob.load_state_dict(w_glob)
        # calculate initial model accuracy, record it as the bench mark.
        g_init_time[str(user_id)] = start_time
        idx = int(user_id) - 1
        net_glob.eval()
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
        utils.util.record_log(user_id, 0, [0.0, 0.0, 0.0, 0.0, 0.0], [acc_local, 0.0, 0.0, 0.0, 0.0], args.model,
                              clean=True)

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
        logger.info("Epoch [" + str(iter + 1) + "] train for user [" + str(user_id) + "]")
        train_start_time = time.time()
        # compute v_bar
        for j in w_glob.keys():
            w_locals_per[j] = hyperpara * w_locals[j] + (1 - hyperpara) * w_glob_local[j]
            differenc1[j] = w_locals[j] - w_glob_local[j]

        # train local global weight
        net_glob.load_state_dict(w_glob_local)
        w, _ = local_update(copy.deepcopy(net_glob).to(args.device), dataset_train, dict_users[user_id - 1], args)
        for j in w_glob.keys():
            w_glob_local[j] = copy.deepcopy(w[j])

        # train local model weight
        net_glob.load_state_dict(w_locals_per)
        w, _ = local_update(copy.deepcopy(net_glob).to(args.device), dataset_train, dict_users[user_id - 1], args)
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
            utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
        test_time = time.time() - test_start_time

        # before start next round, record the time
        total_time = time.time() - g_init_time[str(user_id)]
        round_time = time.time() - start_time
        communication_time = utils.util.reset_communication_time()
        if communication_time < 0.001:
            communication_time = 0.0
        utils.util.record_log(user_id, iter+1, [total_time, round_time, train_time, test_time, communication_time],
                              [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4],
                              args.model)
        start_time = time.time()
        if (iter + 1) % 10 == 0:  # update global model
            from_ip = utils.util.get_ip(args.test_ip_addr)
            upload_local_w(user_id, iter, from_ip, w_glob_local, w_locals, w_locals_per,
                                 hyperpara, start_time)
            return

    logger.info("########## ALL DONE! ##########")
    from_ip = utils.util.get_ip(args.test_ip_addr)
    body_data = {
        'message': 'shutdown_python',
        'uuid': user_id,
        'from_ip': from_ip,
    }
    utils.util.http_client_post(trigger_url, body_data)


def start_train():
    time.sleep(args.start_sleep)
    start_time = time.time()
    train(None, None, None, None, None, None, start_time)


def test(data):
    detail = {"data": data}
    return detail


def load_user_id():
    lock.acquire()
    global g_user_id
    g_user_id += 1
    detail = {"user_id": g_user_id}
    lock.release()
    return detail


def release_global_w(epochs):
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
        utils.util.http_client_post(my_url, json_body)


def average_local_w(user_id, epochs, from_ip, w_glob_local, w_locals, w_locals_per, hyperpara, start_time):
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
        release_global_w(epochs)


def fetch_user_id():
    fetch_data = {
        'message': 'fetch_user_id',
    }
    response = utils.util.http_client_post(trigger_url, fetch_data)
    detail = response.get("detail")
    user_id = detail.get("user_id")
    return user_id


def upload_local_w(user_id, epochs, from_ip, w_glob_local, w_locals, w_locals_per, hyperpara, start_time):
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
    utils.util.http_client_post(trigger_url, upload_data)
    return


def download_global_model(epochs):
    if epochs == g_train_global_model_epoch:
        detail = {
            "global_model": g_train_global_model,
        }
    else:
        detail = {
            "global_model": None,
        }
    return detail


def my_route(app):
    @app.route('/trigger', methods=['GET', 'POST'])
    def trigger_handler():
        # For POST
        if request.method == 'POST':
            data = request.get_json()
            status = "yes"
            detail = {}
            message = data.get("message")
            if message == "fetch_user_id":
                detail = load_user_id()
            elif message == "global_model":
                detail = download_global_model(data.get("epochs"))
            elif message == "upload_local_w":
                threading.Thread(target=average_local_w, args=(
                    data.get("user_id"), data.get("epochs"), data.get("from_ip"), data.get("w_glob_local"),
                    data.get("w_locals"), data.get("w_locals_per"), data.get("hyperpara"), data.get("start_time"))
                                 ).start()
            elif message == "release_global_w":
                threading.Thread(target=train, args=(
                    data.get("user_id"), data.get("epochs"), data.get("w_glob_local"), data.get("w_locals"),
                    data.get("w_locals_per"), data.get("hyperpara"), data.get("start_time"))).start()
            elif message == "shutdown_python":
                threading.Thread(target=utils.util.shutdown_count, args=(
                    data.get("uuid"), data.get("from_ip"), fed_listen_port, args.num_users)).start()
            elif message == "shutdown":
                threading.Thread(target=utils.util.my_exit, args=(args.exit_sleep, )).start()
            response = {"status": status, "detail": detail}
            return response


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
    peer_address_var = utils.util.env_from_sourcing(os.path.join(real_path, "../fabric-network/network.config"),
                                                    "PeerAddress")
    peer_address_list = peer_address_var.split(' ')
    peer_addrs = [peer_addr.split(":")[0] for peer_addr in peer_address_list]
    peer_header_addr = peer_addrs[0]
    trigger_url = "http://" + peer_header_addr + ":" + str(fed_listen_port) + "/trigger"

    # parse participant number
    args.num_users = len(peer_address_list)

    # init dataset and global model
    init()

    threading.Thread(target=start_train, args=()).start()

    flask_app = Flask(__name__)
    my_route(flask_app)
    logger.info("start serving at " + str(fed_listen_port) + "...")
    flask_app.run(host='0.0.0.0', port=fed_listen_port)


if __name__ == "__main__":
    main()
