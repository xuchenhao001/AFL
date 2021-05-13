import asyncio
import json
import logging
import os
import random
import sys
import time
import copy
import threading
import torch
from tornado import ioloop, web, httpserver

import utils.util
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedAvg

logging.setLoggerClass(utils.util.ColoredLogger)
logger = logging.getLogger("fed_sync")

# TO BE CHANGED
# attackers' ids, must be string type "1", "2", ...
attackers_id = []
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
g_train_global_model_epoch = None
shutdown_count_num = 0


def test(data):
    detail = {"data": data}
    return "yes", detail


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
        utils.util.dataset_loader(args.dataset, args.dataset_train_size, args.dataset_test_size, args.iid,
                                  args.num_users)
    if dict_users is None:
        logger.error('Error: unrecognized dataset')
        sys.exit()

    img_size = dataset_train[0][0].shape
    net_glob = utils.util.model_loader(args.model, args.dataset, args.device, args.num_channels, args.num_classes,
                                       img_size)
    if net_glob is None:
        logger.error('Error: unrecognized model')
        sys.exit()
    # finally trained the initial local model, which will be treated as first global model.
    net_glob.train()
    # generate md5 hash from model, which is treated as global model of previous round.
    w = net_glob.state_dict()
    global_model_hash = utils.util.generate_md5_hash(w)
    g_train_global_model = utils.util.compress_tensor(w)
    g_train_global_model_epoch = -1  # -1 means the initial global model


# STEP #1
async def start():
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
    await utils.util.http_client_post(blockchain_server_url, body_data)


# STEP #2
async def train(uuid, epochs, start_time):
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
        result = await utils.util.http_client_post(trigger_url, body_data)
        responseObj = json.loads(result)
        detail = responseObj.get("detail")
        global_model_compressed = detail.get("global_model")
        w_glob = utils.util.decompress_tensor(global_model_compressed)
        logger.debug('Downloaded initial global model hash: ' + utils.util.generate_md5_hash(w_glob))
        net_glob.load_state_dict(w_glob)
        g_init_time[str(uuid)] = start_time
        net_glob.eval()
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            utils.util.test_img(test_users, skew_users, idx, net_glob, dataset_test, args)
        filename = "result-record_" + uuid + ".txt"
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

    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
    train_start_time = time.time()
    w_local, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
    # fake attackers
    if str(uuid) in attackers_id:
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
    await utils.util.http_client_post(trigger_url, body_data)

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
    await utils.util.http_client_post(blockchain_server_url, body_data)


# STEP #3
async def train_count(epochs, uuid, start_time, train_time, w_compressed):
    lock.acquire()
    global train_count_num
    global g_start_time
    global g_train_time
    global g_train_local_models
    global g_train_global_model
    global g_train_global_model_epoch
    global global_model_hash
    train_count_num += 1
    logger.debug("Received a train_ready, now: " + str(train_count_num))
    key = str(uuid) + "-" + str(epochs)
    g_start_time[key] = start_time
    g_train_time[key] = train_time
    # append newly arrived w_local (decompressed) into g_train_local_models list for further aggregation
    g_train_local_models.append(utils.util.decompress_tensor(w_compressed))
    lock.release()
    if train_count_num == args.num_users:
        logger.debug("Gathered enough train_ready, aggregate global model and send the download link.")
        # aggregate global model first
        w_glob = FedAvg(g_train_local_models)
        # release g_train_local_models after aggregation
        g_train_local_models = []
        # save global model for further download (compressed)
        w_glob_compressed = utils.util.compress_tensor(w_glob)
        g_train_global_model = w_glob_compressed
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
        await utils.util.http_client_post(blockchain_server_url, body_data)


# STEP #7
async def round_finish(uuid, epochs):
    global global_model_hash
    logger.debug('Received latest global model for user: %s, epoch: %s.' % (uuid, epochs))

    # download global model
    body_data = {
        'message': 'global_model',
        'epochs': epochs,
    }
    logger.debug('fetch global model of epoch [%s] from: %s' % (epochs, trigger_url))
    result = await utils.util.http_client_post(trigger_url, body_data)
    responseObj = json.loads(result)
    detail = responseObj.get("detail")
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
    response = await utils.util.http_client_post(trigger_url, fetch_data)
    responseObj = json.loads(response)
    detail = responseObj.get("detail")
    start_time = detail.get("start_time")
    train_time = detail.get("train_time")

    # finally, test the acc_local, acc_local_skew1~4
    net_glob.load_state_dict(w_glob)
    net_glob.eval()
    test_start_time = time.time()
    idx = int(uuid) - 1
    acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
        utils.util.test_img(test_users, skew_users, idx, net_glob, dataset_test, args)
    test_time = time.time() - test_start_time

    # before start next round, record the time
    filename = "result-record_" + uuid + ".txt"

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
        body_data = {
            'message': 'next_round_count',
            'uuid': uuid,
            'epochs': new_epochs,
        }
        await utils.util.http_client_post(trigger_url, body_data)
    else:
        logger.info("########## ALL DONE! ##########")
        body_data = {
            'message': 'shutdown_python'
        }
        await utils.util.http_client_post(trigger_url, body_data)


# count for STEP #7 the next round requests gathered
async def next_round_count(epochs):
    global train_count_num
    global next_round_count_num
    lock.acquire()
    next_round_count_num += 1
    lock.release()
    if next_round_count_num == args.num_users:
        # reset counts
        lock.acquire()
        train_count_num = 0
        next_round_count_num = 0
        lock.release()
        # sleep 20 seconds before trigger next round
        # logger.info("SLEEP FOR A WHILE...")
        # await gen.sleep(20)
        # START NEXT ROUND
        body_data = {
            'message': 'PrepareNextRound',
            'data': {},
            'epochs': epochs,
            'is_sync': True
        }
        await utils.util.http_client_post(blockchain_server_url, body_data)


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
        await utils.util.http_client_post(blockchain_server_url, body_data)


async def fetch_time(uuid, epochs):
    key = str(uuid) + "-" + str(epochs)
    start_time = g_start_time.get(key)
    train_time = g_train_time.get(key)
    detail = {
        "start_time": start_time,
        "train_time": train_time,
    }
    return detail


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


class TriggerHandler(web.RequestHandler):

    async def post(self):
        data = json.loads(self.request.body)
        status = "yes"
        detail = {}
        self.set_header("Content-Type", "application/json")

        message = data.get("message")
        if message == "train_ready":
            await train_count(data.get("epochs"), data.get("uuid"), data.get("start_time"), data.get("train_time"),
                              data.get("w_compressed"))
        elif message == "global_model":
            detail = await download_global_model(data.get("epochs"))
        elif message == "next_round_count":
            await next_round_count(data.get("epochs"))
        elif message == "fetch_time":
            detail = await fetch_time(data.get("uuid"), data.get("epochs"))
        elif message == "shutdown_python":
            detail = await shutdown_count()

        response = {"status": status, "detail": detail}
        in_json = json.dumps(response, sort_keys=True, indent=4, ensure_ascii=False).encode('utf8')
        self.write(in_json)


class MainHandler(web.RequestHandler):

    async def get(self):
        asyncio.ensure_future(start())
        response = {
            'status': 'yes'
        }
        in_json = json.dumps(response, sort_keys=True, indent=4, ensure_ascii=False).encode('utf8')
        self.set_header("Content-Type", "application/json")
        self.write(in_json)

    async def post(self):
        # reply to smart contract first
        data = json.loads(self.request.body)
        status = "yes"
        detail = {}
        self.set_header("Content-Type", "application/json")
        response = {"status": status, "detail": detail}
        in_json = json.dumps(response, sort_keys=True, indent=4, ensure_ascii=False).encode('utf8')
        self.write(in_json)

        # Then judge message type and process
        message = data.get("message")
        if message == "test":
            test(data.get("data"))
        elif message == "prepare":
            asyncio.ensure_future(train(data.get("uuid"), data.get("epochs"), time.time()))
        elif message == "global_model_update":
            asyncio.ensure_future(round_finish(data.get("uuid"), data.get("epochs")))
        elif message == "shutdown":
            logger.info("########## PYTHON SHUTTING DOWN! ##########")
            sys.exit()
        return


if __name__ == "__main__":
    init()
    app = web.Application([
        (r"/messages", MainHandler),
        (r"/trigger", TriggerHandler),
    ])
    http_server = httpserver.HTTPServer(app, max_buffer_size=10485760000)  # 10GB
    http_server.listen(fed_listen_port)
    logger.info("start serving at " + str(fed_listen_port) + "...")
    ioloop.IOLoop.current().start()
