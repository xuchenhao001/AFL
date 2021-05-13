# Asynchronous Federated Learning

Asynchronous Federated Learning code. Based on Hyperledger Fabric v2.3.

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

Require matplotlib (>=3.3.1), numpy (>=1.18.5), torch (>=1.7.1) torchvision (>=0.8.2) tornado (>=6.1) and sklearn.

```bash
pip3 install matplotlib numpy torch torchvision tornado sklearn hickle pandas
```

For Raspberry PI, download wheels from [here](https://github.com/Qengineering/PyTorch-Raspberry-Pi-64-OS), then:
```
sudo apt install -y python3-h5py libopenblas-dev
# Download the torch wheels from the website, then install the wheels. Finally:
pip3 install matplotlib numpy tornado sklearn hickle pandas
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

Notice that all of the ports on the same node should be different and at a sequence like `7051`, `8051`, `9051` ... `30051`. (must be ended with `*051`)

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
./network.sh up createChannel && ./network.sh deployCC
```

>  After finished experiment, stop your blockchain network with `./network.sh down`

### Blockchain rest server

After you started a blockchain network, start a blockchain rest server for the communicate between python federated learning processes with blockchain smart contract.

```bash
cd blockchain-server/
# Start in background:
nohup npm start > server.log 2>&1 &
```

### Federated Learning

```bash
cd federated-learning/
rm -f result-*
python3 fed_server.py
# Or start in background
nohup python3 -u fed_server.py > fed_server.log 2>&1 &
```

Trigger training start:

```bash
curl -i -X GET 'http://localhost:8888/messages'
```

# Comparative Experiments

The comparative experiments include (under `AFL/federated-learning/` directory):

```bash
fed_server.py  # our proposed schema (need for blockchain)
fed_server_alpha.py  # our proposed schema with fixed alpha (need for blockchain)
main_fed_localA.py  # Adaptive personalized federated learning (APFL) (no need for blockchain)
main_nn.py  # local deep learning algorithm (Local Training) (no need for blockchain)
main_fed.py  # FedAvg algorithm (no need for blockchain)
```

