import logging
import os
import sys
import time
import copy
import threading
import torch
from flask import Flask, request

import utils.util
from utils.options import args_parser
from utils.util import dataset_loader, model_loader, ColoredLogger
from models.Update import local_update
from models.Fed import FedAvg

logging.setLoggerClass(ColoredLogger)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logger = logging.getLogger("fed_avg")

# TO BE CHANGED
# attackers' ids, must be string type "1", "2", ...
attackers_id = []
# wait in seconds for other nodes to start, seconds
start_wait_time = 60
# federated learning server listen port
fed_listen_port = 8888

# TO BE CHANGED FINISHED

# NOT TO TOUCH VARIABLES BELOW
trigger_url = ""
args = None
net_glob = None
dataset_train = None
dataset_test = None
dict_users = []
lock = threading.Lock()
wMap = []
ipMap = {}
test_users = []
skew_users = []
peer_address_list = []
g_uuid = 0
g_init_time = {}
g_start_time = {}
g_train_time = {}
g_train_global_model = None
g_train_global_model_epoch = None
shutdown_count_num = 0


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
    # finally trained the initial local model, which will be treated as first global model.
    net_glob.train()
    # generate md5 hash from model, which is treated as global model of previous round.
    w = net_glob.state_dict()
    g_train_global_model = utils.util.compress_tensor(w)
    g_train_global_model_epoch = -1  # -1 means the initial global model


def train(uuid, w_glob, epochs):
    global g_init_time
    start_time = time.time()
    logger.debug('Train local model for user: %s, epoch: %s.' % (uuid, epochs))

    if uuid is None:
        uuid = fetch_uuid()

    if epochs is None:
        epochs = args.epochs
    else:
        # load w_glob as net_glob
        net_glob.load_state_dict(w_glob)
        net_glob.eval()

    # calculate initial model accuracy, record it as the bench mark.
    idx = int(uuid) - 1
    if epochs == args.epochs:
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
        g_init_time[str(uuid)] = start_time
        net_glob.eval()
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
        utils.util.record_log(uuid, 0, [0.0, 0.0, 0.0, 0.0, 0.0],
                              [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4],
                              args.model, clean=True)
    logger.info("#################### Epoch #" + str(epochs) + " start now ####################")

    train_start_time = time.time()
    w_local, _ = local_update(copy.deepcopy(net_glob).to(args.device), dataset_train, dict_users[idx], args)
    # fake attackers
    if str(uuid) in attackers_id:
        w_local = utils.util.disturb_w(w_local)
    train_time = time.time() - train_start_time

    # send local model to the first node
    w_local_compressed = utils.util.compress_tensor(w_local)
    from_ip = utils.util.get_ip(args.test_ip_addr)
    upload_data = {
        'message': 'upload_local_w',
        'uuid': uuid,
        'epochs': epochs,
        'w_compressed': w_local_compressed,
        'from_ip': from_ip,
        'start_time': start_time,
        'train_time': train_time,
    }
    utils.util.http_client_post(trigger_url, upload_data)


def start_train():
    time.sleep(args.start_sleep)
    train(None, None, None)


def gathered_global_w(uuid, epochs, w_glob_compressed, start_time, train_time):
    logger.debug('Received latest global model for user: %s, epoch: %s.' % (uuid, epochs))

    # load hash of new global model, which is downloaded from the leader
    w_glob = utils.util.decompress_tensor(w_glob_compressed)
    global_model_hash = utils.util.generate_md5_hash(w_glob)
    logger.debug("Received new global model with hash: " + global_model_hash)

    # epochs count backwards until 0
    new_epochs = epochs - 1

    # finally, test the acc_local, acc_local_skew1~4
    net_glob.load_state_dict(w_glob)
    net_glob.eval()
    test_start_time = time.time()
    idx = int(uuid) - 1
    acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
        utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
    test_time = time.time() - test_start_time

    # before start next round, record the time
    total_time = time.time() - g_init_time[str(uuid)]
    round_time = time.time() - start_time
    communication_time = utils.util.reset_communication_time()
    utils.util.record_log(uuid, epochs, [total_time, round_time, train_time, test_time, communication_time],
                          [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4], args.model)
    if new_epochs > 0:
        # reset a new time for next round
        train(uuid, w_glob, new_epochs)
    else:
        logger.info("########## ALL DONE! ##########")
        from_ip = utils.util.get_ip(args.test_ip_addr)
        body_data = {
            'message': 'shutdown_python',
            'uuid': uuid,
            'from_ip': from_ip,
        }
        utils.util.http_client_post(trigger_url, body_data)


def release_global_w(epochs):
    lock.acquire()
    global g_uuid
    global wMap
    g_uuid = 0
    lock.release()
    w_glob = FedAvg(wMap)
    wMap = []  # release wMap after aggregation
    for uuid in ipMap.keys():
        key = str(uuid) + "-" + str(epochs)
        start_time = g_start_time.get(key)
        train_time = g_train_time.get(key)
        data = {
            'message': 'release_global_w',
            'uuid': uuid,
            'epochs': epochs,
            'w_glob': utils.util.compress_tensor(w_glob),
            'start_time': start_time,
            'train_time': train_time,
        }
        my_url = "http://" + ipMap[uuid] + ":" + str(fed_listen_port) + "/trigger"
        utils.util.http_client_post(my_url, data)


def average_local_w(uuid, epochs, w_compressed, from_ip, start_time, train_time):
    lock.acquire()
    global wMap
    global ipMap

    global g_start_time
    global g_train_time
    key = str(uuid) + "-" + str(epochs)
    g_start_time[key] = start_time
    g_train_time[key] = train_time

    ipMap[uuid] = from_ip
    wMap.append(utils.util.decompress_tensor(w_compressed))
    lock.release()
    if len(wMap) == args.num_users:
        logger.debug("Gathered enough w, average and release them")
        release_global_w(epochs)


def fetch_uuid():
    fetch_data = {
        'message': 'fetch_uuid',
    }
    response = utils.util.http_client_post(trigger_url, fetch_data)
    detail = response.get("detail")
    uuid = detail.get("uuid")
    return uuid


def load_uuid():
    lock.acquire()
    global g_uuid
    g_uuid += 1
    detail = {"uuid": g_uuid}
    lock.release()
    return detail


def load_global_model(epochs):
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
            if message == "fetch_uuid":
                detail = load_uuid()
            elif message == "global_model":
                detail = load_global_model(data.get("epochs"))
            elif message == "upload_local_w":
                threading.Thread(target=average_local_w, args=(
                    data.get("uuid"), data.get("epochs"), data.get("w_compressed"), data.get("from_ip"),
                    data.get("start_time"), data.get("train_time"))).start()
            elif message == "release_global_w":
                threading.Thread(target=gathered_global_w, args=(
                    data.get("uuid"), data.get("epochs"), data.get("w_glob"), data.get("start_time"),
                    data.get("train_time"))).start()
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
