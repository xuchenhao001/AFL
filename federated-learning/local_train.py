import logging
import os
import sys
import time
import subprocess
import copy
import threading
import torch
from flask import Flask, request

import utils
from utils.options import args_parser
from models.Update import local_update
from utils.util import dataset_loader, model_loader, ColoredLogger

logging.setLoggerClass(ColoredLogger)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logger = logging.getLogger("local_train")

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
g_init_time = {}
shutdown_count_num = 0


# init: loads the dataset and global model
def init():
    global net_glob
    global dataset_train
    global dataset_test
    global dict_users
    global test_users
    global skew_users

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
    # initialize weights of model
    net_glob.train()


def train(user_id):
    global args
    global g_init_time

    if user_id is None:
        user_id = fetch_user_id()

    # training for all epochs
    for iter in reversed(range(args.epochs)):
        # calculate initial model accuracy, record it as the bench mark.
        idx = int(user_id) - 1
        if iter+1 == args.epochs:
            # for the first time do the training
            g_init_time[str(user_id)] = time.time()
            # net_glob.eval()
            acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
                utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
            utils.util.record_log(user_id, 0, [0.0, 0.0, 0.0, 0.0, 0.0], [acc_local, 0.0, 0.0, 0.0, 0.0], args.model,
                                  clean=True)

        logger.info("Epoch [" + str(iter+1) + "] train for user [" + str(user_id) + "]")
        train_start_time = time.time()
        w, _ = local_update(copy.deepcopy(net_glob).to(args.device), dataset_train, dict_users[user_id - 1], args)
        train_time = time.time() - train_start_time

        # start test
        test_start_time = time.time()
        idx = int(user_id) - 1
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)

        test_time = time.time() - test_start_time
        total_time = time.time() - g_init_time[str(user_id)]
        round_time = time.time() - train_start_time
        # before start next round, record the time
        utils.util.record_log(user_id, iter+1, [total_time, round_time, train_time, test_time, 0.0],
                              [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4],
                              args.model)

        # update net_glob for next round
        net_glob.load_state_dict(w)

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
    train(None)


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


def fetch_user_id():
    fetch_data = {
        'message': 'fetch_user_id',
    }
    response = utils.util.http_client_post(trigger_url, fetch_data)
    detail = response.get("detail")
    user_id = detail.get("user_id")
    return user_id


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
