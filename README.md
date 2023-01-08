# Blockchain-Based Asynchronous Federated Learning

Blockchain-Based Asynchronous Federated Learning (BAFL) code. Based on Hyperledger Fabric v2.3.

If you think my code is useful, please cite my paper:

```latex
@article{xu2022efficient,
  author={Xu, Chenhao and Qu, Youyang and Luan, Tom H. and Eklund, Peter W. and Xiang, Yong and Gao, Longxiang},
  journal={IEEE Transactions on Vehicular Technology},
  title={An Efficient and Reliable Asynchronous Federated Learning Scheme for Smart Public Transportation},
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TVT.2022.3232603}
}
```

or:

```latex
@inproceedings{xu2021bafl,
  author={Xu, Chenhao and Qu, Youyang and Eklund, Peter W. and Xiang, Yong and Gao, Longxiang},
  booktitle={2021 IEEE Symposium on Computers and Communications (ISCC)}, 
  title={BAFL: An Efficient Blockchain-Based Asynchronous Federated Learning Framework}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ISCC53001.2021.9631405}
}

```

## Install

How to install this project on your operating system.

### Prerequisite

* Ubuntu 20.04

* Python 3.8.5 (pip 20.0.2)

* Docker 20.10.6 (docker-compose 1.29.1)

* Node.js v14.16.1 (npm 6.14.12)

* Golang 1.16.4

* The AFL project should be cloned into the home directory, like `~/AFL`.

* A password-free login from a host (host A) to other hosts (host B, C, ...) in the cluster:

```bash
# First generate an SSH key for each node (on host A,B,C, ...)
ssh-keygen
# Then install the SSH key (from host A) to other hosts (to host A, B, C, ...) as an authorized key
ssh-copy-id <hostA-user>@<hostA-ip>
ssh-copy-id <hostB-user>@<hostB-ip>
ssh-copy-id <hostC-user>@<hostC-ip>
...
```

### Blockchain

All blockchain scripts are under `fabric-network` directory.

```bash
cd fabric-network/
./downloadEnv.sh
```

Then copy the binaries under `fabric-network/bin/` into your PATH.

### Blockchain rest server

```bash
cd blockchain-server/
npm install
```

### Federated Learning

Require matplotlib (>=3.3.1), numpy (>=1.18.5), torch (>=1.7.1) torchvision (>=0.8.2) flask (>=2.0.1) and sklearn.

```bash
pip3 install matplotlib numpy torch torchvision flask sklearn hickle pandas
```

For Raspberry PI, download wheels from [here](https://github.com/Qengineering/PyTorch-Raspberry-Pi-64-OS), then:

```
sudo apt install -y python3-h5py libopenblas-dev
# Download the torch wheels from the website, then install the wheels. Finally:
pip3 install matplotlib numpy flask sklearn hickle pandas
```

### GPU

It's better to have a gpu cuda, which could accelerate the training process. To check if you have any gpu(cuda):

```bash
nvidia-smi
# or
sudo lshw -C display
```

## Run

How to start & stop this project.

### Blockchain

Before start blockchain network, you need to determine the number of blockchain nodes, the user name (should be the same) of remote hosts, and their location in the network. The configure file is located at `fabric-network/network.config`.

For example, you have two nodes running on the same node `10.0.2.15`, the user name of the host is `xueri`, then you can do it like:

```bash
#!/bin/bash
HostUser="ubuntu"
PeerAddress=(
  "10.0.2.15:7051"
  "10.0.2.15:8051"
)
```

> Notice that only one node is allowed to be allocated on the one node.

Another example is you have three nodes running the the different hosts (`10.0.2.15` and `10.0.2.16`) and the user name for all the hosts is `ubuntu`, then your configuration could be like this:

```bash
#!/bin/bash
HostUser="ubuntu"
PeerAddress=(
  "10.0.2.15:7051"
  "10.0.2.15:8051"
  "10.0.2.16:7051"
)
```

After modified the configuration file, now start your blockchain network:

```bash
cd fabric-network/
./network.sh up
```

>  When finished experiment, stop your blockchain network with `./network.sh down`

### Blockchain rest server

After you started a blockchain network, start a blockchain rest server for the communicate between python federated learning processes with blockchain smart contract.

```bash
cd blockchain-server/
# Start in background:
nohup npm start > server.log 2>&1 &
```

or:

```bash
cd blockchain-server/cluster-scripts/
./restart_blockchain_server.sh
```

### Federated Learning

The parameters for the training are at `./AFL/federated-learning/utils/options.py`

```bash
cd federated-learning/
rm -f result-*
python3 fed_server.py
# Or start in background
nohup python3 -u fed_server.py > fed_server.log 2>&1 &
```

Trigger training to start:

```bash
curl -i -X GET 'http://localhost:8888/messages'
```

# Comparative Experiments

The comparative experiments include (under `AFL/federated-learning/` directory):

```bash
fed_async.py  # our proposed asynchronous federated learning schema (need blockchain)
fed_sync.py  # synchronous federated learning schema (need blockchain)
fed_avg.py  # synchronous federated learning (FedAvg) algorithm (no need blockchain)
fed_localA.py  # adaptive personalized federated learning (APFL) (no need blockchain)
local_train.py  # local deep learning algorithm (Local Training) (no need blockchain)
```

Before running tests automatically, adjust parameters at `cluster-scripts/test.config`:

```bash
#!/bin/bash

# all schemes to test
TestSchema=(
        "cnn-fashion_mnist"
        "cnn-cifar"
        "mlp-fashion_mnist"
        "lstm-loop"
)

# test iid or non-iid datasets
IS_IID=true

# the default scaling factor setting, -1 means dynamic scaling factor
FADE=-1

# the ID of the poisoning attacker
ATTACKER=5

# the training dataset size on each node
TrainDataSize=(
        "1500"
        "1500"
        "1500"
        "1500"
        "1500"
)
```

To run all tests automatically, go to `cluster-scripts/`, and run:

```bash
./all_nodes_test_bg.sh  # test BAFL, BSFL, FedAVG, APFL, and Local Training
./all_nodes_test_async_bg.sh  # test BAFL
./all_nodes_test_ddos_attack_bg.sh  # test BAFL and AFL under DDoS attacks
./all_nodes_test_poisoning_attack_bg.sh  # test BAFL and AFL under poisoning attacks
./all_nodes_test_static_bg.sh  # test BAFL under dynamic or static settings
```

There are additional scripts that ease cluster operations under `cluster-scripts/`:

```bash
./all_nodes_update.sh  # updates codes on all nodes, following the IPs at `fabric-network/network.config`
./clean-output.sh  # clean logs on all nodes
./gather-output.sh  # gather logs on all nodes
./replace_network_config.sh  # replace `fabric-network/network.config` on all nodes by that on this node
./restart_blockchain_server.sh  # restart the blockchain server on this node
```
