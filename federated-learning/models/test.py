import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Update import DatasetSplit


def test_img(net_g, dataset_test, args, test_indices):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(dataset_test, test_indices), batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        data = data.detach().clone().type(torch.FloatTensor)
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    # test_loss /= len(data_loader.dataset)
    # accuracy = 100.00 * correct / len(data_loader.dataset)
    return correct, test_loss


def test_img_total(net_g, dataset_test, args, idx_list):
    accuracy_list = []
    test_loss_list = []
    correct_test_local = 0
    loss_test_local = 0
    for i in range(len(idx_list)):
        correct_test, loss_test = test_img(net_g, dataset_test, args, idx_list[i])
        if i == 0:
            correct_test_local = correct_test
            accuracy_local = 100.0 * correct_test_local / len(idx_list[0])
            accuracy_list.append(accuracy_local)
            loss_test_local = loss_test
            loss_local = 100.0 * loss_test_local / len(idx_list[0])
            test_loss_list.append(loss_local)
        else:
            accuracy_skew = 100.0 * (correct_test_local + correct_test) / (len(idx_list[0]) + len(idx_list[i]))
            accuracy_list.append(accuracy_skew)
            loss_skew = 100.0 * (loss_test_local + loss_test) / (len(idx_list[0]) + len(idx_list[i]))
            test_loss_list.append(loss_skew)

    return accuracy_list, test_loss_list


def test_lstm(net_g, dataset_test, args, test_indices):
    data_loader = DataLoader(DatasetSplit(dataset_test, test_indices), batch_size=40, shuffle=True, drop_last=True)
    losses_mse = []
    losses_mae = []
    for idx, (data, target) in enumerate(data_loader):
        data = data.detach().clone().type(torch.FloatTensor)
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        outputs = net_g(data)
        loss_MSE = torch.nn.MSELoss()
        loss_MAE = torch.nn.L1Loss()
        loss_mse = loss_MSE(outputs, torch.squeeze(target))
        loss_mae = loss_MAE(outputs, torch.squeeze(target))
        losses_mse.append(loss_mse.cpu().data.numpy())
        losses_mae.append(loss_mae.cpu().data.numpy())

    # losses_l1 = np.array(losses_mae)
    loss_mae_mean = np.mean(losses_mae)
    # losses_mse = np.array(losses_mse)
    loss_mse_mean = np.mean(losses_mse)
    # mean_l1 = np.mean(losses_l1) * max_speed
    # std_l1 = np.std(losses_l1) * max_speed

    # print('Tested: MSE_loss: {}, MAE_mean: {}'.format(loss_mse_mean, loss_mae_mean))
    return loss_mse_mean, loss_mae_mean
