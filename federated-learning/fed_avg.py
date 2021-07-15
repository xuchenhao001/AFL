import asyncio
import json
import logging
import os
import sys
import time
import copy
import threading
from abc import ABC

import torch
from tornado import ioloop, web, httpserver

import utils
from utils.options import args_parser
from utils.util import dataset_loader, model_loader, ColoredLogger
from models.Update import LocalUpdate, LocalUpdateLSTM
from models.Fed import FedAvg

logging.setLoggerClass(ColoredLogger)
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


def test(data):
    detail = {"data": data}
    return "yes", detail


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


async def train(uuid, w_glob, epochs):
    global g_init_time
    start_time = time.time()
    logger.debug('Train local model for user: %s, epoch: %s.' % (uuid, epochs))

    if uuid is None:
        uuid = await fetch_uuid()

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
        result = await utils.util.http_client_post(trigger_url, body_data)
        response_json = json.loads(result)
        detail = response_json.get("detail")
        global_model_compressed = detail.get("global_model")
        w_glob = utils.util.decompress_tensor(global_model_compressed)
        logger.debug('Downloaded initial global model hash: ' + utils.util.generate_md5_hash(w_glob))
        net_glob.load_state_dict(w_glob)
        g_init_time[str(uuid)] = start_time
        net_glob.eval()
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            utils.util.test_model(net_glob, dataset_test, args, test_users, skew_users, idx)
        filename = "result-record_" + str(uuid) + ".txt"
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
                                   + " <acc_local_skew1> " + str(acc_local_skew1)[:8]
                                   + " <acc_local_skew2> " + str(acc_local_skew2)[:8]
                                   + " <acc_local_skew3> " + str(acc_local_skew3)[:8]
                                   + " <acc_local_skew4> " + str(acc_local_skew4)[:8]
                                   + "\n")
    logger.info("#################### Epoch #" + str(epochs) + " start now ####################")

    if dict_users is not None:
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
    else:
        local = LocalUpdateLSTM(args=args, dataset=dataset_train)
    train_start_time = time.time()
    w_local, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
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
    await utils.util.http_client_post(trigger_url, upload_data)


async def start_train():
    await asyncio.sleep(args.start_sleep)
    await train(None, None, None)


async def gathered_global_w(uuid, epochs, w_glob_compressed, start_time, train_time):
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
    filename = "result-record_" + str(uuid) + ".txt"

    with open(filename, "a") as time_record_file:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        total_time = time.time() - g_init_time[str(uuid)]
        round_time = time.time() - start_time
        communication_time = round_time - train_time - test_time
        time_record_file.write(current_time + "[" + f"{epochs:0>2}" + "]"
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
    if new_epochs > 0:
        # reset a new time for next round
        asyncio.ensure_future(train(uuid, w_glob, new_epochs))
    else:
        logger.info("########## ALL DONE! ##########")
        from_ip = utils.util.get_ip(args.test_ip_addr)
        body_data = {
            'message': 'shutdown_python',
            'uuid': uuid,
            'from_ip': from_ip,
        }
        await utils.util.http_client_post(trigger_url, body_data)


async def release_global_w(epochs):
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
        await utils.util.http_client_post(my_url, data)


async def average_local_w(uuid, epochs, w_compressed, from_ip, start_time, train_time):
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
        asyncio.ensure_future(release_global_w(epochs))


async def fetch_uuid():
    fetch_data = {
        'message': 'fetch_uuid',
    }
    response = await utils.util.http_client_post(trigger_url, fetch_data)
    response_json = json.loads(response)
    detail = response_json.get("detail")
    uuid = detail.get("uuid")
    return uuid


async def load_uuid():
    lock.acquire()
    global g_uuid
    g_uuid += 1
    detail = {"uuid": g_uuid}
    lock.release()
    return detail


async def load_global_model(epochs):
    if epochs == g_train_global_model_epoch:
        detail = {
            "global_model": g_train_global_model,
        }
    else:
        detail = {
            "global_model": None,
        }
    return detail


class TriggerHandler(web.RequestHandler, ABC):

    async def post(self):
        data = json.loads(self.request.body)
        status = "yes"
        detail = {}
        self.set_header("Content-Type", "application/json")

        message = data.get("message")
        if message == "fetch_uuid":
            detail = await load_uuid()
        elif message == "global_model":
            detail = await load_global_model(data.get("epochs"))
        elif message == "upload_local_w":
            await average_local_w(data.get("uuid"), data.get("epochs"), data.get("w_compressed"), data.get("from_ip"),
                                  data.get("start_time"), data.get("train_time"))
        elif message == "release_global_w":
            await gathered_global_w(data.get("uuid"), data.get("epochs"), data.get("w_glob"),
                                    data.get("start_time"), data.get("train_time"))
        elif message == "shutdown_python":
            detail = await utils.util.shutdown_count(data.get("uuid"), data.get("from_ip"), fed_listen_port, lock,
                                                     args.num_users)
        elif message == "shutdown":
            asyncio.ensure_future(utils.util.my_exit(args.exit_sleep))

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

    asyncio.ensure_future(start_train())

    app = web.Application([
        (r"/trigger", TriggerHandler),
    ])
    http_server = httpserver.HTTPServer(app, max_buffer_size=10485760000)  # 10GB
    http_server.listen(fed_listen_port)
    logger.info("start serving at " + str(fed_listen_port) + "...")
    ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
