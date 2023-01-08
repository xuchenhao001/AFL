import logging
import os
import random
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
from models.Fed import FadeFedAvg

logging.setLoggerClass(ColoredLogger)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logger = logging.getLogger("fed_async")

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
peer_address_list = []
global_model_hash = ""
g_my_uuid = -1
g_init_time = {}
g_start_time = {}
g_train_time = {}
g_train_global_model = None
g_train_global_model_compressed = None
g_train_global_model_version = 0
shutdown_count_num = 0
current_acc_local = -1


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
        'is_sync': False
    }
    utils.util.http_client_post(blockchain_server_url, body_data)


# STEP #2
def train(uuid, epochs, start_time):
    global g_my_uuid
    global g_init_time
    logger.debug('Train local model for user: %s, epoch: %s.' % (uuid, epochs))
    if g_my_uuid == -1:
        g_my_uuid = uuid  # init my_uuid at the first time

    # calculate initial model accuracy, record it as the bench mark.
    idx = int(uuid) - 1
    if epochs == args.epochs:
        # download initial global model
        body_data = {
            'message': 'global_model',
        }
        logger.debug('fetch initial global model from: %s' % trigger_url)
        # time.sleep(20000)  # test to pause
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
    if str(uuid) in args.poisoning_attackers:
        logger.debug("Detected id in poisoning attackers' list: {}, manipulate local gradients!"
                     .format(args.poisoning_attackers))
        w_local = utils.util.disturb_w(w_local)
    train_time = time.time() - train_start_time

    # send local model to the first node for aggregation
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
        'is_sync': False
    }
    utils.util.http_client_post(blockchain_server_url, body_data)

    # finished aggregate global model, continue next round
    round_finish(uuid, epochs)


# STEP #3
def aggregate(epochs, uuid, start_time, train_time, w_compressed):
    global g_start_time
    global g_train_time
    global g_train_global_model
    global g_train_global_model_compressed
    global g_train_global_model_version
    global global_model_hash

    logger.debug("Received a train_ready.")
    lock.acquire()
    key = str(uuid) + "-" + str(epochs)
    g_start_time[key] = start_time
    g_train_time[key] = train_time
    lock.release()
    # mimic DDoS attacks here
    if args.ddos_duration == 0 or args.ddos_duration > g_train_global_model_version:
        logger.debug("Mimic the aggregator under DDoS attacks!")
        if random.random() < args.ddos_no_response_percent:
            logger.debug("Unfortunately, the aggregator does not response to the local update gradients")
            lock.acquire()
            # ignore the update to the global model
            g_train_global_model_version += 1
            lock.release()
            return

    logger.debug("Aggregate global model after received a new local model.")
    w_local = utils.util.decompress_tensor(w_compressed)
    # aggregate global model
    if g_train_global_model is not None:
        fade_c = calculate_fade_c(uuid, w_local, args.fade, args.model, args.poisoning_detect_threshold)
        w_glob = FadeFedAvg(g_train_global_model, w_local, fade_c)
    # test new global model acc and record onto the log
    intermediate_acc_record(w_glob)
    # save global model for further download
    g_train_global_model_compressed = utils.util.compress_tensor(w_glob)
    lock.acquire()
    g_train_global_model = w_glob
    g_train_global_model_version += 1
    lock.release()
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
        'is_sync': False
    }
    logger.debug('aggregate global model finished, send global_model_hash [%s] to blockchain in epoch [%s].'
                 % (global_model_hash, epochs))
    utils.util.http_client_post(blockchain_server_url, body_data)


def calculate_fade_c(uuid, w_local, fade_target, model, acc_detect_threshold):
    if fade_target == -1:  # -1 means fade dynamic setting
        logger.debug("fade=-1, dynamic fade setting is adopted!")
        # dynamic fade setting, test new acc_local first
        net_glob.load_state_dict(w_local)
        net_glob.eval()
        idx = int(uuid) - 1

        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
        logger.debug("after test, acc_local: {}, current_acc_local: {}".format(acc_local, current_acc_local))
        if model == "lstm":
            # for lstm, acc_local means the mse loss instead of accuracy, the less the better
            if current_acc_local == -1:
                fade_c = 10
            else:
                try:
                    fade_c = current_acc_local / acc_local
                except ZeroDivisionError as err:
                    logger.debug('Divided by zero: {}, set scaling factor to 10 by default.'.format(err))
                    fade_c = 10
        else:
            # for cnn or mlp models, accuracy the higher the better.
            if current_acc_local == -1:
                fade_c = 10
            else:
                try:
                    fade_c = acc_local / current_acc_local
                except ZeroDivisionError as err:
                    logger.debug('Divided by zero: {}, set scaling factor to 10 by default.'.format(err))
                    fade_c = 10
        # filter out poisoning local updated gradients whose test accuracy is less than acc_detect_threshold
        if fade_c < acc_detect_threshold:
            fade_c = 0
    else:
        logger.debug("fade={}, static fade setting is adopted!".format(fade_target))
        # static fade setting
        fade_c = fade_target
    logger.debug("calculated fade_c: %f" % fade_c)
    return fade_c


def intermediate_acc_record(w_glob):
    net_glob.load_state_dict(w_glob)
    net_glob.eval()
    total_time = time.time() - g_init_time[str(g_my_uuid)]
    idx = int(g_my_uuid) - 1
    acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
        utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
    utils.util.record_log(g_my_uuid, 0, [total_time, 0.0, 0.0, 0.0, 0.0],
                          [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4], args.model)


# STEP #7
def round_finish(uuid, epochs):
    global global_model_hash
    global current_acc_local
    logger.debug('Download latest global model for user: %s, epoch: %s.' % (uuid, epochs))

    # download global model
    body_data = {
        'message': 'global_model',
    }
    result = utils.util.http_client_post(trigger_url, body_data)
    detail = result.get("detail")
    global_model_compressed = detail.get("global_model")
    global_model_version = detail.get("version")
    logger.debug('Successfully fetched global model [%s] of epoch [%s] from: %s' % (global_model_version, epochs,
                                                                                    trigger_url))
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
    current_acc_local = acc_local
    test_time = time.time() - test_start_time

    # before start next round, record the time
    total_time = time.time() - g_init_time[str(uuid)]
    round_time = time.time() - start_time
    communication_time = utils.util.reset_communication_time()
    utils.util.record_log(uuid, epochs, [total_time, round_time, train_time, test_time, communication_time],
                          [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4], args.model)
    if new_epochs > 0:
        # start next round of train right now
        train(uuid, new_epochs, time.time())
    else:
        logger.info("########## ALL DONE! ##########")
        body_data = {
            'message': 'shutdown_python'
        }
        utils.util.http_client_post(trigger_url, body_data)


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
            'is_sync': False
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


def download_global_model():
    detail = {
        "global_model": g_train_global_model_compressed,
        "version": g_train_global_model_version,
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
                threading.Thread(target=aggregate, args=(
                    data.get("epochs"), data.get("uuid"), data.get("start_time"), data.get("train_time"),
                    data.get("w_compressed"))).start()
            elif message == "global_model":
                detail = download_global_model()
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
