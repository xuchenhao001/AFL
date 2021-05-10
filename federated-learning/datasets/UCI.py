from torch import optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
from models.Nets import UCI_CNN
from torch.utils.data import DataLoader
import torch.nn as nn


class UCIDataset(Dataset):
    def __init__(self, data_path, phase="train"):
        self.data_path = data_path
        self.phase = phase
        self.data, self.targets = self.get_data()

    def get_data(self):
        data_acc_x = pd.read_csv(os.path.join(self.data_path, self.phase, 'AccXUCI.csv'), header=None).values
        data_acc_y = pd.read_csv(os.path.join(self.data_path, self.phase, 'AccYUCI.csv'), header=None).values
        data_acc_z = pd.read_csv(os.path.join(self.data_path, self.phase, 'AccZUCI.csv'), header=None).values
        data_gyro_x = pd.read_csv(os.path.join(self.data_path, self.phase, 'GyroXUCI.csv'), header=None).values
        data_gyro_y = pd.read_csv(os.path.join(self.data_path, self.phase, 'GyroYUCI.csv'), header=None).values
        data_gyro_z = pd.read_csv(os.path.join(self.data_path, self.phase, 'GyroZUCI.csv'), header=None).values
        data = np.dstack((data_acc_x, data_acc_y, data_acc_z, data_gyro_x, data_gyro_y, data_gyro_z)).transpose(0, 2, 1)
        label = pd.read_csv(os.path.join(self.data_path, self.phase, 'LabelUCI.csv'), header=None).values.reshape(-1)
        label = label.astype(np.int)
        return data, label

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    real_path = os.path.dirname(os.path.realpath(__file__))
    uci_data_path = os.path.join(real_path, "../../data/uci/")
    device = torch.device('cpu')
    dataset = UCIDataset(uci_data_path)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    net = UCI_CNN().to(device)
    loss_fun = nn.CrossEntropyLoss()
    params_to_update = []
    for name, param in net.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer = optim.SGD(params_to_update, lr=0.01)
    for i in range(200):
        for step, (image, label) in enumerate(dataloader):
            image = torch.tensor(image).type(torch.FloatTensor)
            image = image.to(device)
            label = label.to(device)
            pred = net(image)
            loss = loss_fun(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("Epoch %d, step %d, loss %f" % (i, step, loss))

    test_dataset = UCIDataset(uci_data_path, phase='eval')
    test_dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    total = 0
    error = 0
    net.eval()
    with torch.no_grad():
        for image, label in test_dataloader:
            image = torch.tensor(image).type(torch.FloatTensor)
            image = image.to(device)
            label = label.to(device)
            pred = net(image)
            total += pred.shape[0]
            error += sum(torch.argmax(pred, dim=1) != label)

        print(error, total, (total-error)/total)
