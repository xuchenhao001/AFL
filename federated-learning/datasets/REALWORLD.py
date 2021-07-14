import os
from collections import Counter

from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import numpy as np
import hickle as hkl


class REALWORLDDataset(Dataset):
    def __init__(self, data_path, phase="train"):

        self.data_path = data_path
        self.phase = phase
        self.data, self.targets = self.get_data()

    def get_data(self):
        clientLabel = []
        clientData = []
        for i in range(0, 15):
            accX = hkl.load(self.data_path + str(i) + '/AccX' + "REALWORLD_CLIENT" + '.hkl')
            accY = hkl.load(self.data_path + str(i) + '/AccY' + "REALWORLD_CLIENT" + '.hkl')
            accZ = hkl.load(self.data_path + str(i) + '/AccZ' + "REALWORLD_CLIENT" + '.hkl')
            gyroX = hkl.load(self.data_path + str(i) + '/GyroX' + "REALWORLD_CLIENT" + '.hkl')
            gyroY = hkl.load(self.data_path + str(i) + '/GyroY' + "REALWORLD_CLIENT" + '.hkl')
            gyroZ = hkl.load(self.data_path + str(i) + '/GyroZ' + "REALWORLD_CLIENT" + '.hkl')
            label = hkl.load(self.data_path + str(i) + '/Label' + "REALWORLD_CLIENT" + '.hkl')
            clientData.append(np.dstack((accX, accY, accZ, gyroX, gyroY, gyroZ)).transpose(0, 2, 1))
            clientLabel.append(label)

        data = []
        label = []
        for i in range(0, 15):
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            kf.get_n_splits(clientData[i])
            partitionedData = list()
            partitionedLabel = list()
            for train_index, test_index in kf.split(clientData[i]):
                partitionedData.append(clientData[i][test_index])
                partitionedLabel.append(clientLabel[i][test_index])

            if self.phase == "train":
                data.append((np.vstack((partitionedData[:4]))))
                label.append((np.hstack((partitionedLabel[:4]))))
            else:
                data.append((partitionedData[4]))
                label.append((partitionedLabel[4]))
        data = np.vstack(data)
        label = np.hstack(label)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == '__main__':
    real_path = os.path.dirname(os.path.realpath(__file__))
    realworld_client_data_path = os.path.join(real_path, "../../data/realworld_client/")
    dataset = REALWORLDDataset(data_path=realworld_client_data_path)
    # print(dataset[0][0].shape, dataset[0][1])
    # print(len(dataset))
    # print(Counter(dataset.targets))
