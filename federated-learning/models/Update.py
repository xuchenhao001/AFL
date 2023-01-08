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


def train_cnn_mlp(net, dataset, idxs, local_ep, device, lr, local_bs):
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)
    ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=local_bs, shuffle=True)
    loss_func = nn.CrossEntropyLoss()

    epoch_loss = []
    for iter in range(local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images = images.detach().clone().type(torch.FloatTensor)
            images, labels = images.to(device), labels.to(device)
            net.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def train_lstm(net, dataset, idxs, local_ep, device):
    optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-5)
    ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=40, shuffle=True, drop_last=True)
    loss_func = torch.nn.MSELoss()
    losses_train = []

    losses_epochs_train = []
    for epoch in range(local_ep):
        losses_epoch_train = []
        for batch_idx, (data, labels) in enumerate(ldr_train):
            data = data.detach().clone().type(torch.FloatTensor)
            data, labels = data.to(device), labels.to(device)
            net.zero_grad()
            outputs = net(data)
            loss_train = loss_func(outputs, torch.squeeze(labels))
            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
        losses_epochs_train.append(avg_losses_epoch_train)

    return net.state_dict(), losses_train


def local_update(net, dataset, idxs, args):
    if args.model == "lstm":
        return train_lstm(net, dataset, idxs, args.local_ep, args.device)
    else:
        return train_cnn_mlp(net, dataset, idxs, args.local_ep, args.device, args.lr, args.local_bs)
