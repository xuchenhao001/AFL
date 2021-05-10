#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import asyncio
import base64
import gzip
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import subprocess
import copy
import numpy as np
import threading
import torch
from tornado import httpclient, ioloop, web, gen, httpserver

from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img, test_img_total
from utils.util import dataset_loader, model_loader, ColoredLogger

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger("fed_server_alpha")

np.random.seed(0)
torch.random.manual_seed(0)

# TO BE CHANGED
# To change the static alpha, use utils/options: "hyperpara"
# attackers' ids, must be string type "1", "2", ...
attackers_id = []
# rounds to negotiate alpha
negotiate_round = 10
# committee members proportion
committee_proportion = 0.3
# federated learning server listen port
fed_listen_port = 8888
# TO BE CHANGED FINISHED

# NOT TO TOUCH VARIABLES BELOW
blockchain_server_url = ""
trigger_url = ""
total_epochs = 0  # epochs must be an integer
args = None
net_glob = None
dataset_train = None
dataset_test = None
dict_users = []
lock = threading.Lock()
test_users = []
skew_users = []
negotiate_count_num = 0
next_round_count_num = 0
peer_address_list = []
raft_leader_http_addr = ""
my_local_model_tensor = {}
my_global_model_tensor = {}
global_model_hash = ""
raft_subprocess = None
# Global parameters for the committee leader
train_count_num = 0
g_start_time = {}
g_train_time = {}
g_train_local_models = []
g_train_global_model = None
g_train_global_model_epoch = None
g_test_time = {}


def test(data):
    detail = {"data": data}
    return "yes", detail


# returns variable from sourcing a file
def env_from_sourcing(file_to_source_path, variable_name):
    source = 'source %s && export MYVAR=$(echo "${%s[@]}")' % (file_to_source_path, variable_name)
    dump = '/usr/bin/python3 -c "import os, json; print(os.getenv(\'MYVAR\'))"'
    pipe = subprocess.Popen(['/bin/bash', '-c', '%s && %s' % (source, dump)], stdout=subprocess.PIPE)
    # return json.loads(pipe.stdout.read())
    return pipe.stdout.read().decode("utf-8").rstrip()


async def http_client_post(url, body_data):
    json_body = json.dumps(body_data, sort_keys=True, indent=4, ensure_ascii=False, cls=NumpyEncoder).encode('utf8')
    logger.debug("Start http client post [" + body_data['message'] + "] to: " + url)
    method = "POST"
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    http_client = httpclient.AsyncHTTPClient()
    try:
        request = httpclient.HTTPRequest(url=url, method=method, headers=headers, body=json_body, connect_timeout=300,
                                         request_timeout=300)
        response = await http_client.fetch(request)
        logger.debug("[HTTP Success] [" + body_data['message'] + "] from " + url)
        return response.body
    except Exception as e:
        logger.error("[HTTP Error] [" + body_data['message'] + "] from " + url + " ERROR DETAIL: %s" % e)
        return None


# STEP #1
def init():
    global args
    global total_epochs
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
    # parse network.config and read the peer addresses
    real_path = os.path.dirname(os.path.realpath(__file__))
    peerAddressVar = env_from_sourcing(os.path.join(real_path, "../fabric-samples/network.config"), "PeerAddress")
    peer_address_list = peerAddressVar.split(' ')
    peerHeaderAddr = peer_address_list[0].split(":")[0]
    # initially the blockchain communicate server is load on the first peer
    blockchain_server_url = "http://" + peerHeaderAddr + ":3000/invoke/mychannel/fabcar"
    # initially the trigger url is load on the first peer (will change when raft committee do elect)
    trigger_url = "http://" + peerHeaderAddr + ":" + str(fed_listen_port) + "/trigger"

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    total_epochs = args.epochs
    logger.setLevel(args.log_level)
    # parse participant number
    args.num_users = len(peer_address_list)

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
    # finally trained the initial local model, which will be treated as first global model.
    net_glob.train()
    # generate md5 hash from model, which is treated as global model of previous round.
    w = net_glob.state_dict()
    global_model_hash = generate_md5_hash(w)


# STEP #1
# (prepare for the training) BC-node1-python initiate local (global) model, and then send the hash of global model
# to the ledger.
async def start():
    # upload md5 hash to ledger
    body_data = {
        'message': 'Start',
        'data': {
            'global_model_hash': global_model_hash,
            'user_number': args.num_users,
            'do_elect': True
        },
        'epochs': total_epochs
    }
    await http_client_post(blockchain_server_url, body_data)


# STEP #2
# BC-nodes-python choose committee members according to global model hash, pull up hraftd distributed processes,
# send setup request to raftd and start up raft consensus，finally send raft network info to the ledger.
async def prepare_committee(uuid, epochs, do_elect):
    global raft_leader_http_addr
    global trigger_url
    logger.info("###################### Epoch #" + str(epochs) + " start now ######################")

    logger.debug('[RAFT] Received prepare committee request for user: %s, epoch: %s.' % (uuid, epochs))
    # if need, re-elect the committee members
    if do_elect:
        logger.debug('[RAFT] Received elect request! Kill the old raft processes and elect new committee members!')
        kill_local_raft_proc()
        logger.debug('[RAFT] Current global model hash: ' + global_model_hash)
        committee_leader_id = int(global_model_hash, 16) % args.num_users + 1
        committee_proportion_num = math.ceil(args.num_users * committee_proportion)  # committee id delta value
        committee_highest_id = committee_proportion_num + committee_leader_id - 1
        logger.debug('[RAFT] The leader id is: %s' % str(committee_leader_id))
        # update trigger url based on committee leader (to accept the global model or local models)
        trigger_ip = peer_address_list[int(committee_leader_id) - 1].split(":")[0]
        trigger_url = "http://" + trigger_ip + ":" + str(fed_listen_port) + "/trigger"
        # pull up hraftd distributed processes, if the value of uuid is in range of committee leader id and highest id.
        rounded_committee_highest_id = 0
        if committee_highest_id > args.num_users:
            rounded_committee_highest_id = committee_highest_id % args.num_users
        if committee_leader_id <= int(uuid) <= committee_highest_id or \
                1 <= int(uuid) <= rounded_committee_highest_id:
            logger.debug("[RAFT] # BOOT LOCAL RAFT PROCESS! #")
            http_addr, raft_addr = generate_raft_addr_info(uuid)
            boot_local_raft_proc(uuid, http_addr, raft_addr)
        # wait for a while in case raft processes on some nodes are not running.
        await gen.sleep(2)
        # all nodes need to store the committee leader info for later local model upload
        raft_leader_http_addr, raft_leader_raft_addr = generate_raft_addr_info(committee_leader_id)

        # if this node is elected as committee leader, boot the raft network.
        if int(uuid) == committee_leader_id:
            logger.debug("[RAFT] Match the leader ID: " + uuid)
            logger.debug("[RAFT] # BOOT RAFT CONSENSUS NETWORK! #")
            client_addrs = []
            client_raft_addrs = []
            client_ids = []
            # i starts from the leader id and end at committee highest id
            logger.debug("[RAFT] committee_highest_id: " + str(committee_highest_id))
            for i in range(int(uuid) + 1, committee_highest_id + 1):
                if i > args.num_users:
                    index = i % args.num_users
                else:
                    index = i
                client_http_addr, client_raft_addr = generate_raft_addr_info(index)
                client_ids.append(str(index))
                client_addrs.append(client_http_addr)
                client_raft_addrs.append(client_raft_addr)
            body_data = {
                'message': 'raft_start',
                'leaderAddr': raft_leader_http_addr,
                'leaderRaftAddr': raft_leader_raft_addr,
                'leaderId': str(uuid),
                'clientAddrs': client_addrs,
                'clientRaftAddrs': client_raft_addrs,
                'clientIds': client_ids,
            }
            await http_client_post("http://" + raft_leader_http_addr + "/setup", body_data)

            # finally send raft network info to the ledger
            body_data = {
                'message': 'RaftInfo',
                'data': {
                    'leader_addr': raft_leader_http_addr,
                },
                'uuid': uuid,
                'epochs': epochs,
            }
            await http_client_post(blockchain_server_url, body_data)
        else:
            await gen.sleep(5)

    # finally go ahead to STEP #3, train the local model
    await train(uuid, epochs, time.time())


# STEP #3
# Federated Learning: train step
async def train(uuid, epochs, start_time):
    global my_local_model_tensor
    logger.debug('Train local model for user: %s, epoch: %s.' % (uuid, epochs))

    # calculate initial model accuracy, record it as the bench mark.
    idx = int(uuid) - 1
    if epochs == total_epochs:
        net_glob.eval()
        idx_total = [test_users[idx], skew_users[0][idx], skew_users[1][idx], skew_users[2][idx], skew_users[3][idx]]
        correct = test_img_total(net_glob, dataset_test, idx_total, args)
        acc_local = torch.div(100.0 * correct[0], len(test_users[idx]))
        filename = "result-record_" + uuid + ".txt"
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

    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
    train_start_time = time.time()
    w_local, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
    # update local cached local model for this epoch
    my_local_model_tensor = w_local
    # fake attackers
    if uuid in attackers_id:
        w_local = disturb_w(w_local)
    train_time = time.time() - train_start_time

    # send local model to the committee leader
    w_local_compressed = compress_data(convert_tensor_value_to_numpy(w_local))
    body_data = {
        'message': 'train_ready',
        'uuid': str(uuid),
        'epochs': epochs,
        'w_compressed': w_local_compressed,
        'start_time': start_time,
        'train_time': train_time
    }
    await http_client_post(trigger_url, body_data)

    # send hash of local model to the ledger
    model_md5 = generate_md5_hash(w_local)
    body_data = {
        'message': 'Train',
        'data': {
            'w': model_md5,
        },
        'uuid': uuid,
        'epochs': epochs,
    }
    await http_client_post(blockchain_server_url, body_data)


# STEP #4
# committee leader received all local models, aggregate to global model, then send the download link of global model
# and the hash of global model to the ledger.
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
    g_train_local_models.append(conver_numpy_value_to_tensor(decompress_data(w_compressed)))
    lock.release()
    if train_count_num == args.num_users:
        logger.debug("Gathered enough train_ready, aggregate global model and send the download link.")
        # aggregate global model first
        w_glob = FedAvg(g_train_local_models)
        # release g_train_local_models after aggregation
        g_train_local_models = []
        # save global model for further download (compressed)
        w_glob_compressed = compress_data(convert_tensor_value_to_numpy(w_glob))
        g_train_global_model = w_glob_compressed
        g_train_global_model_epoch = epochs
        # generate hash of global model
        global_model_hash = generate_md5_hash(w_glob)
        logger.debug("As a committee leader, calculate new global model hash: " + global_model_hash)
        # send the download link and hash of global model to the ledger
        body_data = {
            'message': 'GlobalModelUpdate',
            'data': {
                'global_model_hash': global_model_hash,
            },
            'uuid': uuid,
            'epochs': epochs,
        }
        logger.debug('aggregate global model finished, send global_model_hash [%s] to blockchain in epoch [%s].'
              % (global_model_hash, epochs))
        await http_client_post(blockchain_server_url, body_data)


# STEP #5
# BC-nodes-python get the download link of global model from the ledger, download the global model, then calculate
# alpha-accuracy map, which will be uploaded to the committee leader.
async def calculate_acc_alpha(uuid, epochs):
    global my_global_model_tensor
    global global_model_hash
    logger.debug('Start calculate accuracy and alpha for user: %s, epoch: %s.' % (uuid, epochs))
    # download global model
    body_data = {
        'message': 'global_model',
        'epochs': epochs,
    }
    logger.debug('fetch global model of epoch [%s] from: %s' % (epochs, trigger_url))
    result = await http_client_post(trigger_url, body_data)
    responseObj = json.loads(result)
    detail = responseObj.get("detail")
    global_model_compressed = detail.get("global_model")
    w_glob = conver_numpy_value_to_tensor(decompress_data(global_model_compressed))
    # load hash of new global model, which is downloaded from the leader
    global_model_hash = generate_md5_hash(w_glob)
    logger.debug("As a follower, received new global model with hash: " + global_model_hash)
    # update local cached global model for this epoch
    my_global_model_tensor = w_glob

    # stable alpha is hyperpara
    alpha = args.hyperpara
    alpha_list = []
    acc_test_list = []
    test_start_time = time.time()
    w_local2 = {}
    for key in w_glob.keys():
        w_local2[key] = alpha * my_local_model_tensor[key] + (1 - alpha) * w_glob[key]

    # change parameters to model
    net_glob.load_state_dict(w_local2)
    net_glob.eval()
    # test the accuracy
    idx = int(uuid) - 1
    acc_test, loss_test = test_img(net_glob, dataset_test, test_users[idx], args)
    alpha_list.append(alpha)
    acc_test_list.append(acc_test.numpy().item(0))
    logger.debug("uuid: " + uuid + ", alpha: " + str(alpha) + ", acc_test result: " + str(acc_test.numpy().item(0)))

    test_time = time.time() - test_start_time

    # upload acc-alpha map to committee leader and the ledger
    body_data = {
        'message': 'AccAlphaMap',
        'data': {
            'acc_test': acc_test_list,
            'alpha': alpha_list,
        },
        'uuid': uuid,
        'epochs': epochs,
    }
    logger.debug('calculate acc-alpha map finished, send accuracy and alpha map to the ledger for uuid: ' + uuid)
    await http_client_post(blockchain_server_url, body_data)
    trigger_data = {
        'message': 'acc_alpha_map_ready',
        'epochs': epochs,
        'uuid': uuid,
        'test_time': test_time,
    }
    await http_client_post(trigger_url, trigger_data)


# count for STEP #5 the acc alpha map gathered
async def acc_alpha_map_count(epochs, uuid, test_time):
    lock.acquire()
    global negotiate_count_num
    global g_test_time
    negotiate_count_num += 1
    key = str(uuid) + "-" + str(epochs)
    g_test_time[key] = test_time
    lock.release()
    if negotiate_count_num == args.num_users:
        # check negotiate read first, since network delay may cause blockchain dirty read.
        while True:
            trigger_data = {
                'message': 'CheckAccAlphaMapRead',
                'epochs': epochs,
            }
            response = await http_client_post(blockchain_server_url, trigger_data)
            if response is not None:
                break
            await gen.sleep(1)  # if dirty read happened, sleep for 1 second then retry.
        # trigger STEP #6
        trigger_data = {
            'message': 'FindBestAlpha',
            'epochs': epochs,
        }
        await http_client_post(blockchain_server_url, trigger_data)


# STEP #7
# BC-nodes-python get the appropriate alpha, merge the local model and the global model with alpha to generate the
# new local model. Test the new local model, then repeat from step 2.
async def round_finish(data, uuid, epochs):
    logger.debug('Received best alpha, train new w_local for user: %s, epoch: %s.' % (uuid, epochs))
    alpha = data.get("alpha")
    # calculate new w_glob (w_local2) according to the alpha
    w_local2 = {}
    for key in my_global_model_tensor.keys():
        w_local2[key] = alpha * my_local_model_tensor[key] + (1 - alpha) * my_global_model_tensor[key]
    # epochs count backwards until 0
    new_epochs = epochs - 1
    # fetch time record
    fetch_data = {
        'message': 'fetch_time',
        'uuid': uuid,
        'epochs': epochs,
    }
    response = await http_client_post(trigger_url, fetch_data)
    responseObj = json.loads(response)
    detail = responseObj.get("detail")
    start_time = detail.get("start_time")
    test_time = detail.get("test_time")
    train_time = detail.get("train_time")

    # finally, test the acc_local, acc_local_skew1~4
    # conver_numpy_value_to_tensor(w_local2)
    net_glob.load_state_dict(w_local2)
    net_glob.eval()
    test_addition_start_time = time.time()
    idx = int(uuid) - 1
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
    test_addition_time = time.time() - test_addition_start_time
    test_time += test_addition_time

    # before start next round, record the time
    filename = "result-record_" + uuid + ".txt"

    with open(filename, "a") as time_record_file:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        total_time = time.time() - start_time
        communication_time = total_time - train_time - test_time
        time_record_file.write(current_time + "[" + f"{epochs:0>2}" + "]"
                               + " <Total Time> " + str(total_time)[:8]
                               + " <Train Time> " + str(train_time)[:8]
                               + " <Test Time> " + str(test_time)[:8]
                               + " <Communication Time> " + str(communication_time)[:8]
                               + " <Alpha> " + str(alpha)[:8]
                               + " <acc_local> " + str(acc_local.item())[:8]
                               + " <acc_local_skew1> " + str(acc_local_skew1.item())[:8]
                               + " <acc_local_skew2> " + str(acc_local_skew2.item())[:8]
                               + " <acc_local_skew3> " + str(acc_local_skew3.item())[:8]
                               + " <acc_local_skew4> " + str(acc_local_skew4.item())[:8]
                               + "\n")
    if new_epochs > 0:
        body_data = {
            'message': 'next_round_count',
            'uuid': uuid,
            'epochs': new_epochs,
        }
        await http_client_post(trigger_url, body_data)
    else:
        logger.info("########## ALL DONE! ##########")
        await gen.sleep(600)  # sleep 600 seconds before exit
        sys.exit()


# count for STEP #7 the next round requests gathered
async def next_round_count(epochs):
    global train_count_num
    global negotiate_count_num
    global next_round_count_num
    lock.acquire()
    next_round_count_num += 1
    lock.release()
    if next_round_count_num == args.num_users:
        # reset counts
        lock.acquire()
        train_count_num = 0
        negotiate_count_num = 0
        next_round_count_num = 0
        lock.release()
        # trigger next round's committee election
        do_elect = True
        # sleep 20 seconds before trigger next round
        logger.info("SLEEP FOR A WHILE...")
        await gen.sleep(20)
        # START NEXT ROUND
        body_data = {
            'message': 'PrepareNextRoundCommittee',
            'data': {
                'do_elect': do_elect
            },
            'epochs': epochs,
        }
        await http_client_post(blockchain_server_url, body_data)


async def fetch_time(uuid, epochs):
    key = str(uuid) + "-" + str(epochs)
    start_time = g_start_time.get(key)
    train_time = g_train_time.get(key)
    test_time = g_test_time.get(key)
    detail = {
        "start_time": start_time,
        "train_time": train_time,
        "test_time": test_time,
    }
    return detail


# boot local raft process, preparing for the raft consensus algorithm
def boot_local_raft_proc(uuid, http_addr, raft_addr):
    global raft_subprocess
    node_id = "node" + str(uuid)
    real_path = os.path.dirname(os.path.realpath(__file__))
    hraftd_path = os.path.join(real_path, "../raft/hraftd")
    snapshot_path = os.path.join(real_path, "../raft/" + node_id)
    raft_subprocess = subprocess.Popen([hraftd_path, "-id", node_id, "-haddr", http_addr, "-raddr", raft_addr,
                                        snapshot_path])
    return


# kill local raft process
def kill_local_raft_proc():
    global raft_subprocess
    if raft_subprocess is not None:
        logger.debug("[RAFT] Found raft subprocess. Kill it now.")
        raft_subprocess.kill()
        raft_subprocess = None
    else:
        logger.debug("[RAFT] Didn't find raft subprocess. Do not kill.")


# generate raft address info from uuid
def generate_raft_addr_info(uuid):
    raft_ip = peer_address_list[int(uuid) - 1].split(":")[0]
    raft_port = peer_address_list[int(uuid) - 1].split(":")[1]
    http_port = int(raft_port) + 99
    raft_port = int(raft_port) + 100
    http_addr = raft_ip + ":" + str(http_port)
    raft_addr = raft_ip + ":" + str(raft_port)
    return http_addr, raft_addr


# generate raft address from peer address
def generate_raft_addr(peer_addr):
    logger.debug('generate peer addr for: ' + peer_addr)
    ip = peer_addr.split(":")[0]
    port = peer_addr.split(":")[1]
    http_port = str(int(port) + 100)  # raft http port is (100 + peer port)
    raft_port = str(int(port) + 101)  # raft port is (101 + peer port)
    return {'httpAddr': ip + ':' + http_port, 'raftAddr': ip + ':' + raft_port}


def disturb_w(w):
    disturbed_w = copy.deepcopy(w)
    for name, param in w.items():
        beta = random.random()
        transformed_w = param * beta
        disturbed_w[name] = transformed_w
    return disturbed_w


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def conver_numpy_value_to_tensor(numpy_data):
    tensor_data = copy.deepcopy(numpy_data)
    for key, value in tensor_data.items():
        tensor_data[key] = torch.from_numpy(np.array(value))
    return tensor_data


def convert_tensor_value_to_numpy(tensor_data):
    numpy_data = copy.deepcopy(tensor_data)
    for key, value in numpy_data.items():
        numpy_data[key] = value.cpu().numpy()
    return numpy_data


# compress object to base64 string
def compress_data(data):
    encoded = json.dumps(data, sort_keys=True, indent=4, ensure_ascii=False, cls=NumpyEncoder).encode(
        'utf8')
    compressed_data = gzip.compress(encoded)
    b64_encoded = base64.b64encode(compressed_data)
    return b64_encoded.decode('ascii')


# based64 decode to byte, and then decompress it
def decompress_data(data):
    base64_decoded = base64.b64decode(data)
    decompressed = gzip.decompress(base64_decoded)
    return json.loads(decompressed)


# generate md5 hash for global model. Require a tensor type gradients.
def generate_md5_hash(model_weights):
    np_model_weights = convert_tensor_value_to_numpy(model_weights)
    data_md5 = hashlib.md5(json.dumps(np_model_weights, sort_keys=True, cls=NumpyEncoder).encode('utf-8')).hexdigest()
    return data_md5


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
        elif message == "acc_alpha_map_ready":
            await acc_alpha_map_count(data.get("epochs"), data.get("uuid"), data.get("test_time"))
        elif message == "next_round_count":
            await next_round_count(data.get("epochs"))
        elif message == "fetch_time":
            detail = await fetch_time(data.get("uuid"), data.get("epochs"))

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
            asyncio.ensure_future(prepare_committee(data.get("uuid"), data.get("epochs"),
                                                    data.get("data").get("do_elect")))
        elif message == "global_model_update":
            asyncio.ensure_future(calculate_acc_alpha(data.get("uuid"), data.get("epochs")))
        elif message == "best_alpha":
            asyncio.ensure_future(round_finish(data.get("data"), data.get("uuid"), data.get("epochs")))
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
