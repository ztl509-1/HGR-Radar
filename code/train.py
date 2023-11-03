import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from torchstat import stat
from torchsummary import summary
from scipy.fftpack import fftshift
from sklearn import metrics
from torch import Tensor
from collections import Counter
from model import Model_groups
import datetime


class MyDataset(Dataset):
    def __init__(self, folder_path):
        self.dataset = []
        self.labels = []
        self.label_to_int = {}
        label_counter = 0
        for folder_name in os.listdir(folder_path):
            #             print(folder_name)
            folder_name = folder_name
            if folder_name not in self.label_to_int:
                self.label_to_int[folder_name] = label_counter
                label_counter += 1

            folder_dir = os.path.join(folder_path, folder_name)

            for npy_file in os.listdir(folder_dir):
                if npy_file.endswith('.npy'):
                    npy_path = os.path.join(folder_dir, npy_file)
                    data = np.load(npy_path, allow_pickle=True)
                    for i in data:
                        self.dataset.append(i)
                        self.labels.append(self.label_to_int[folder_name])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data = torch.tensor(data).float()
        label = self.labels[idx]

        return data, label


def train(net, epoch, optimizer, criterion, batch_size, trainloader, trainset, device):

    global loss_x
    global loss_y

    loss_ = 0
    print("Epoch: %d, Learning rate: %f" % (epoch + 1, optimizer.param_groups[0]['lr']))
    net.train()
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        predict = torch.max(outputs, 1)[1].data.squeeze()
        accuracy = (predict == labels).sum().item() / labels.size(0)

        loss_x.append(epoch * len(trainloader) + i)
        loss_y.append(loss.item())
        loss_ += loss.item()
        if i % 200 == 0:
            print('Epoch: %d, [%6d, %6d], Train loss: %.4f, Train accuracy: %.4f ' % (
                epoch + 1, (i + 1) * batch_size, len(trainset), loss.item(), accuracy))
    #         scheduler.step()
    print('Epoch: %d, train loss is %f' % (epoch + 1, loss_))

def test(net, epoch, testloader, testset, device):
    global acc_x
    global acc_y

    net.eval()
    n = 0
    sum_acc = 0

    classes = ['pinch', 'swipe_downward', 'circle', 'swipe_from_left_to_right', 'push_in']

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        predict = torch.max(outputs, 1)[1].data.squeeze()
        sum_acc += (predict == labels).sum().item() / labels.size(0)
        n += 1

        for label, pred in zip(labels, predict):
            if label == pred:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    avg_acc = 0
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            accur = 0
        else:
            accur = 100 * float(correct_count) / total_pred[classname]
        avg_acc += accur
        print("Accuracy for class {:5s} is: {:.1f}%({}/{})".format(classname, accur, correct_count,
                                                                   total_pred[classname]))
    print("Average accuracy is: {:.2f}%".format(avg_acc / len(classes)))

    test_acc = sum_acc / n
    acc_x.append(epoch + 1)
    acc_y.append(test_acc)
    print('Epoch: %d, Test accuracy: %.4f ' % (epoch + 1, test_acc))

    return test_acc

if __name__ == "__main__":
    acc_best = 0
    lr = 0.001

    batch_size = 32
    train_dataset = MyDataset('./data/train')
    test_dataset = MyDataset('./data/test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda"
    learning_rate = 0.001
    epoch_n = 200
    best_test_acc = 0

    net = Model_groups()
    print(net)
    net.to(device)

    loss_x = []
    loss_y = []
    acc_x = []
    acc_y = []

    starttime = datetime.datetime.now()
    for epoch in range(epoch_n):
        criterion = nn.CrossEntropyLoss(weight=None)
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate, eps=1e-05)
        train(net, epoch, optimizer, criterion, batch_size, train_loader, train_dataset, device)
        # scheduler.step()
        test_acc = test(net, epoch, test_loader, test_dataset, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = './model_pth/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(net.state_dict(), save_path+'model_groups.pth')
        print('The best test accuracy is %.4f' % (best_test_acc))
        print("# ---------------------------------------------------- #")
    print("Finished!")
    print('The best test accuracy is %.4f' % (best_test_acc))

    endtime = datetime.datetime.now()
    ti = (endtime - starttime).seconds
    hou = ti / 3600
    ti = ti % 3600
    sec = ti / 60
    ti = ti % 60
    print('Time expended: %dh-%dm-%ds' % (hou, sec, ti))
    print('\n')

    plt.figure()
    plt.plot(loss_x, loss_y)
    plt.title("Train Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    # plt.savefig("Loss.jpg")
    plt.show()

    plt.figure()
    plt.plot(acc_x, acc_y)
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # plt.savefig("Accuracy.jpg")
    plt.show()

