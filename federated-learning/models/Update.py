import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images = images.detach().clone().type(torch.FloatTensor)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateLSTM(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = torch.nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(dataset, batch_size=40, shuffle=True, drop_last=True)

    def train(self, net):
        optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-5)
        losses_train = []

        # use_gpu = torch.cuda.is_available()
        losses_epochs_train = []
        pre_time = time.time()
        is_best_model = 0
        for epoch in range(self.args.local_ep):
            print("train epoch: {}".format(epoch))
            losses_epoch_train = []
            print("self.ldr_train len: {}".format(len(self.ldr_train)))
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                data = data.detach().clone().type(torch.FloatTensor)
                data, labels = data.to(self.args.device), labels.to(self.args.device)
            # for data in self.ldr_train:
            #     inputs, labels = data
            #     if use_gpu:
            #         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            #     else:
            #         inputs, labels = Variable(inputs), Variable(labels)
                # test above
                net.zero_grad()
                outputs = net(data)
                loss_train = self.loss_func(outputs, torch.squeeze(labels))
                losses_train.append(loss_train.data)
                losses_epoch_train.append(loss_train.data)
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
            losses_epochs_train.append(avg_losses_epoch_train)
            cur_time = time.time()
            print('Epoch: {}, train_loss: {}, time: {}, best model: {}'.format(
                epoch,
                np.around(avg_losses_epoch_train, decimals=8),
                np.around([cur_time - pre_time], decimals=2),
                is_best_model))
            pre_time = cur_time

        return net.state_dict(), losses_train
