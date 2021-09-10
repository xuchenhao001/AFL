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
logger = logging.getLogger("fed_sync")

# TO BE CHANGED
# federated learning server listen port
fed_listen_port = 8888
# TO BE CHANGED FINISHED

# NOT TO TOUCH VARIABLES BELOW
blockchain_server_url = ""
trigger_url = ""
args = None
net_glob = None
dataset_train = None
dataset_test = None
dict_users = []
lock = threading.Lock()
test_users = []
skew_users = []
next_round_count_num = 0
peer_address_list = []
global_model_hash = ""
train_count_num = 0
g_init_time = {}
g_start_time = {}
g_train_time = {}
g_train_local_models = []
g_train_global_model = None
g_train_global_model_compressed = None
g_train_global_model_epoch = None
shutdown_count_num = 0


# STEP #1
def init():
    global args
    global net_glob
    global dataset_train
    global dataset_test
    global dict_users
    global test_users
    global skew_users
    global blockchain_server_url
    global trigger_url
    global peer_address_list
    global global_model_hash
    global g_train_global_model
    global g_train_global_model_compressed
    global g_train_global_model_epoch
    # parse network.config and read the peer addresses
    real_path = os.path.dirname(os.path.realpath(__file__))
    peer_address_list = utils.util.env_from_sourcing(os.path.join(real_path, "../fabric-network/network.config"),
                                                     "PeerAddress").split(' ')
    peer_header_addr = peer_address_list[0].split(":")[0]
    # initially the blockchain communicate server is load on the first peer
    blockchain_server_url = "http://" + peer_header_addr + ":3000/invoke/mychannel/fabcar"
    # initially the trigger url is load on the first peer
    trigger_url = "http://" + peer_header_addr + ":" + str(fed_listen_port) + "/trigger"

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    logger.setLevel(args.log_level)
    # parse participant number
    args.num_users = len(peer_address_list)

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
    global_model_hash = utils.util.generate_md5_hash(w)
    g_train_global_model = w
    g_train_global_model_compressed = utils.util.compress_tensor(w)
    g_train_global_model_epoch = -1  # -1 means the initial global model


# STEP #1
def start():
    # upload md5 hash to ledger
    body_data = {
        'message': 'Start',
        'data': {
            'global_model_hash': global_model_hash,
            'user_number': args.num_users,
        },
        'epochs': args.epochs,
        'is_sync': True
    }
    utils.util.http_client_post(blockchain_server_url, body_data)


# STEP #2
def train(uuid, epochs, start_time):
    global g_init_time
    logger.debug('Train local model for user: %s, epoch: %s.' % (uuid, epochs))

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
        w_glob_hash = utils.util.generate_md5_hash(w_glob)
        logger.debug('Downloaded initial global model hash: ' + w_glob_hash)
        net_glob.load_state_dict(w_glob)
        g_init_time[str(uuid)] = start_time
        net_glob.eval()
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
        utils.util.record_log(uuid, 0, [0.0, 0.0, 0.0, 0.0, 0.0],
                              [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4],
                              args.model, clean=True)
    train_start_time = time.time()
    w_local, _ = local_update(copy.deepcopy(net_glob).to(args.device), dataset_train, dict_users[idx], args)
    # fake attackers
    if str(uuid) in args.attackers:
        logger.debug("Detected id in attackers' list: {}, manipulate local gradients!".format(args.attackers))
        w_local = utils.util.disturb_w(w_local)
    train_time = time.time() - train_start_time

    # send local model to the first node
    w_local_compressed = utils.util.compress_tensor(w_local)
    body_data = {
        'message': 'train_ready',
        'uuid': str(uuid),
        'epochs': epochs,
        'w_compressed': w_local_compressed,
        'start_time': start_time,
        'train_time': train_time
    }
    utils.util.http_client_post(trigger_url, body_data)

    # send hash of local model to the ledger
    model_md5 = utils.util.generate_md5_hash(w_local)
    body_data = {
        'message': 'UploadLocalModel',
        'data': {
            'w': model_md5,
        },
        'uuid': uuid,
        'epochs': epochs,
        'is_sync': True
    }
    utils.util.http_client_post(blockchain_server_url, body_data)


# STEP #3
def train_count(epochs, uuid, start_time, train_time, w_compressed):
    global train_count_num
    global g_start_time
    global g_train_time
    global g_train_local_models
    global g_train_global_model
    global g_train_global_model_compressed
    global g_train_global_model_epoch
    global global_model_hash
    lock.acquire()
    train_count_num += 1
    logger.debug("Received a train_ready, now: " + str(train_count_num))
    key = str(uuid) + "-" + str(epochs)
    g_start_time[key] = start_time
    g_train_time[key] = train_time
    lock.release()
    # append newly arrived w_local (decompressed) into g_train_local_models list for further aggregation
    w_decompressed = utils.util.decompress_tensor(w_compressed)
    lock.acquire()
    g_train_local_models.append(w_decompressed)
    lock.release()
    if train_count_num == args.num_users:
        logger.debug("Gathered enough train_ready, aggregate global model and send the download link.")
        # reset counts
        lock.acquire()
        train_count_num = 0
        lock.release()
        # aggregate global model first
        w_glob = FedAvg(g_train_local_models)
        # release g_train_local_models after aggregation
        g_train_local_models = []
        # save global model for further download (compressed)
        g_train_global_model = w_glob
        g_train_global_model_compressed = utils.util.compress_tensor(w_glob)
        g_train_global_model_epoch = epochs
        # generate hash of global model
        global_model_hash = utils.util.generate_md5_hash(w_glob)
        logger.debug("As a committee leader, calculate new global model hash: " + global_model_hash)
        # send the download link and hash of global model to the ledger
        body_data = {
            'message': 'UploadGlobalModel',
            'data': {
                'global_model_hash': global_model_hash,
            },
            'uuid': uuid,
            'epochs': epochs,
            'is_sync': True
        }
        logger.debug('aggregate global model finished, send global_model_hash [%s] to blockchain in epoch [%s].'
                     % (global_model_hash, epochs))
        utils.util.http_client_post(blockchain_server_url, body_data)


# STEP #7
def round_finish(uuid, epochs):
    global global_model_hash
    logger.debug('Received latest global model for user: %s, epoch: %s.' % (uuid, epochs))

    # download global model
    body_data = {
        'message': 'global_model',
        'epochs': epochs,
    }
    logger.debug('fetch global model of epoch [%s] from: %s' % (epochs, trigger_url))
    result = utils.util.http_client_post(trigger_url, body_data)
    detail = result.get("detail")
    global_model_compressed = detail.get("global_model")
    w_glob = utils.util.decompress_tensor(global_model_compressed)
    # load hash of new global model, which is downloaded from the leader
    global_model_hash = utils.util.generate_md5_hash(w_glob)
    logger.debug("Received new global model with hash: " + global_model_hash)

    # epochs count backwards until 0
    new_epochs = epochs - 1
    # fetch time record
    fetch_data = {
        'message': 'fetch_time',
        'uuid': uuid,
        'epochs': epochs,
    }
    response = utils.util.http_client_post(trigger_url, fetch_data)
    detail = response.get("detail")
    start_time = detail.get("start_time")
    train_time = detail.get("train_time")

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
        body_data = {
            'message': 'next_round_count',
            'uuid': uuid,
            'epochs': new_epochs,
        }
        utils.util.http_client_post(trigger_url, body_data)
    else:
        logger.info("########## ALL DONE! ##########")
        body_data = {
            'message': 'shutdown_python'
        }
        utils.util.http_client_post(trigger_url, body_data)


# count for STEP #7 the next round requests gathered
def next_round_count(epochs):
    global next_round_count_num
    lock.acquire()
    next_round_count_num += 1
    lock.release()
    if next_round_count_num == args.num_users:
        # reset counts
        lock.acquire()
        next_round_count_num = 0
        lock.release()
        # START NEXT ROUND
        body_data = {
            'message': 'PrepareNextRound',
            'data': {},
            'epochs': epochs,
            'is_sync': True
        }
        utils.util.http_client_post(blockchain_server_url, body_data)


def shutdown_count():
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
            'is_sync': True
        }
        logger.debug('Sent shutdown python request to blockchain.')
        utils.util.http_client_post(blockchain_server_url, body_data)


def fetch_time(uuid, epochs):
    key = str(uuid) + "-" + str(epochs)
    start_time = g_start_time.get(key)
    train_time = g_train_time.get(key)
    detail = {
        "start_time": start_time,
        "train_time": train_time,
    }
    return detail


def download_global_model(epochs):
    if epochs == g_train_global_model_epoch:
        detail = {
            "global_model": g_train_global_model_compressed,
        }
    else:
        detail = {
            "global_model": None,
        }
    return detail


def my_route(app):
    @app.route('/messages', methods=['GET', 'POST'])
    def main_handler():
        # For GET
        if request.method == 'GET':
            start()
            response = {
                'status': 'yes'
            }
            return response
        # For POST
        else:
            data = request.get_json()
            status = "yes"
            detail = {}
            response = {"status": status, "detail": detail}
            # Then judge message type and process
            message = data.get("message")
            if message == "prepare":
                threading.Thread(target=train, args=(data.get("uuid"), data.get("epochs"), time.time())).start()
            elif message == "global_model_update":
                threading.Thread(target=round_finish, args=(data.get("uuid"), data.get("epochs"))).start()
            elif message == "shutdown":
                threading.Thread(target=utils.util.my_exit, args=(args.exit_sleep, )).start()
            return response

    @app.route('/trigger', methods=['GET', 'POST'])
    def trigger_handler():
        # For POST
        if request.method == 'POST':
            data = request.get_json()
            status = "yes"
            detail = {}
            message = data.get("message")

            if message == "train_ready":
                threading.Thread(target=train_count, args=(
                    data.get("epochs"), data.get("uuid"), data.get("start_time"), data.get("train_time"),
                    data.get("w_compressed"))).start()
            elif message == "global_model":
                detail = download_global_model(data.get("epochs"))
            elif message == "next_round_count":
                threading.Thread(target=next_round_count, args=(data.get("epochs"),)).start()
            elif message == "fetch_time":
                detail = fetch_time(data.get("uuid"), data.get("epochs"))
            elif message == "shutdown_python":
                threading.Thread(target=shutdown_count, args=()).start()
            response = {"status": status, "detail": detail}
            return response

    @app.route('/test', methods=['GET', 'POST'])
    def test():
        # For GET
        if request.method == 'GET':
            test_body = {
                "test": "success"
            }
            return test_body
        # For POST
        else:
            doc = request.get_json()
            return doc


if __name__ == "__main__":
    init()
    flask_app = Flask(__name__)
    my_route(flask_app)
    logger.info("start serving at " + str(fed_listen_port) + "...")
    flask_app.run(host='0.0.0.0', port=fed_listen_port)
