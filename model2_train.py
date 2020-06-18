from __future__ import print_function, division
import torchvision
import torchvision.datasets as dsets
import torch.nn.functional as Func

import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as LRscheduler
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os
import shutil
import re
import math
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

data_root = args.input_folder
train_folder = os.path.join(data_root,'train/')
test_folder = os.path.join(data_root,'test/')


class Dataset_Masks(Dataset):
    def __init__(self, listIDs, labels, transform=None, train=True):
        self.labels = labels
        self.list_IDs = listIDs
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        if self.train:
            x = Image.open(train_folder + self.list_IDs[idx] + '_' + str(self.labels[idx]) + '.jpg')
        else:
            x = Image.open(test_folder + self.list_IDs[idx] + '_' + str(self.labels[idx]) + '.jpg')
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

# Hyper Parameters
num_epochs = 300
batch_size = 256
learning_rate = 0.01


mean_nums = [0.5226, 0.4404, 0.4206]
std_nums = [0.2336, 0.22224, 0.2187]


transform_train = transforms.Compose([#transforms.RandomCrop(32, padding=4),
        transforms.Resize((40,40), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)])

transform_test = transforms.Compose([#transforms.RandomCrop(32, padding=4),
        transforms.Resize((40,40), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)])


files_train = os.listdir(train_folder)
x_train = [file_path.split('_')[0] for file_path in files_train]
y_train = [int(file_path.split('_')[1].split('.')[0]) for file_path in files_train]
train_dataset = Dataset_Masks(x_train, y_train, transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

files_test = os.listdir(test_folder)
x_test = [file_path.split('_')[0] for file_path in files_test]
y_test = [int(file_path.split('_')[1].split('.')[0]) for file_path in files_test]
test_dataset = Dataset_Masks(x_test, y_test, transform_test, False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def save_checkpoint(model, filepath):
    torch.save(model.state_dict(), filepath)

class DenseNet(nn.Module):
    def __init__(self, growthRate, depth):
        super(DenseNet, self).__init__()

        self.loss = nn.NLLLoss()
        dense_blocks = (depth-4) // 3
        dense_blocks //= 2

        in_channels = 2*growthRate
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, padding=1,  bias=False)
        self.dense1 = self._make_dense(in_channels, growthRate, dense_blocks)
        in_channels += dense_blocks*growthRate
        out_channels = int(math.floor(in_channels//2))
        self.trans1 = Transition(in_channels, out_channels)

        in_channels = out_channels
        self.dense2 = self._make_dense(in_channels, growthRate, dense_blocks)
        in_channels += dense_blocks*growthRate
        out_channels = int(math.floor(in_channels//2))
        self.trans2 = Transition(in_channels, out_channels)

        in_channels = out_channels
        self.dense3 = self._make_dense(in_channels, growthRate, dense_blocks)
        in_channels += dense_blocks*growthRate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, 2) #################### 10

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(Func.avg_pool2d(Func.relu(self.bn1(out)), 8))
        out = Func.softmax(self.fc(out))
        return out

    def _make_dense(self, in_channels, growthRate, dense_blocks):
        layers = []
        for i in range(int(dense_blocks)):
            layers.append(Bottleneck(in_channels, growthRate))
            in_channels += growthRate
        return nn.Sequential(*layers)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = Func.avg_pool2d(out, 2)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growthRate):
        super(Bottleneck, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4 * growthRate, kernel_size=1, bias=False)
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(4 * growthRate),
            nn.ReLU(),
            nn.Conv2d(4 * growthRate, growthRate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.cat((x, out), 1)
        return out

def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x

cnn = DenseNet(growthRate=12, depth=20)
cnn = to_gpu(cnn)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
print(f'Num of trainable parameters : {sum(p.numel() for p in cnn.parameters() if p.requires_grad)}')

lr_scheduler = LRscheduler.ReduceLROnPlateau(optimizer, eps=0.001)

epoch_train_error_list = np.zeros(num_epochs)
epoch_train_loss_list = np.zeros(num_epochs)
epoch_train_f1_list = np.zeros(num_epochs)
epoch_train_auc_list = np.zeros(num_epochs)

epoch_test_error_list = np.zeros(num_epochs)
epoch_test_loss_list = np.zeros(num_epochs)
epoch_test_f1_list = np.zeros(num_epochs)
epoch_test_auc_list = np.zeros(num_epochs)
best_epoch = []

for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_train_error = 0.0
    epoch_test_loss = 0.0
    epoch_test_error = 0.0
    epoch_prediction = []
    epoch_prob_class1 = []
    FN_FP = 0
    TP = 0

    cnn.train()
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        cnn.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # loss calculation
        epoch_train_loss += outputs.shape[0] * loss.item()

        # error calculation
        _, predictions = torch.max(outputs.data, 1)
        pos = (predictions == 1)
        True_pred = (predictions == labels)
        TP += (pos == True_pred).sum()
        FN_FP += (predictions != labels).sum()

        epoch_train_error += (predictions != labels).sum()

        for prob in (outputs.data.cpu().detach().numpy())[:, 1]:
            epoch_prob_class1.append(prob)
        for pred in predictions.cpu().detach().numpy():
            epoch_prediction.append(pred)
        print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' % (
        epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

    # avg loss and error
    epoch_loss = epoch_train_loss / len(train_dataset)
    epoch_error = epoch_train_error.type(torch.FloatTensor) / len(train_dataset) * 100
    epoch_train_loss_list[epoch] = epoch_loss
    epoch_train_error_list[epoch] = epoch_error
    epoch_train_f1_list[epoch] = f1_score(train_dataset.labels, epoch_prediction, average='binary')
    epoch_train_auc_list[epoch] = roc_auc_score(train_dataset.labels, epoch_prob_class1)

    # Update the scheduler step
    lr_scheduler.step(epoch_loss)

    cnn.eval()
    correct = 0
    total = 0
    epoch_test_prediction = []
    epoch_test_prob_class1 = []

    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = cnn(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        # loss calculation
        epoch_test_loss += outputs.shape[0] * loss.item()

        # error calculation
        _, predictions = torch.max(outputs.data, 1)

        for prob in (outputs.data.cpu().detach().numpy())[:, 1]:
            epoch_test_prob_class1.append(prob)
        for pred in predictions.cpu().detach().numpy():
            epoch_test_prediction.append(pred)

        epoch_test_error += (predictions != labels).sum()

    # avg loss and error
    epoch_loss = epoch_test_loss / len(test_dataset)
    epoch_error = epoch_test_error.type(torch.FloatTensor) / len(test_dataset) * 100
    epoch_test_loss_list[epoch] = epoch_loss
    epoch_test_error_list[epoch] = epoch_error
    epoch_test_f1_list[epoch] = f1_score(test_dataset.labels, epoch_test_prediction, average='binary')
    epoch_test_auc_list[epoch] = roc_auc_score(test_dataset.labels, epoch_test_prob_class1)

    print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
#

print('Saving model')
correct = 0
total = 0
test_predictions = []
test_prob_class1 = []

for images, labels in test_loader:
    if torch.cuda.is_available():
      images = images.cuda()
      labels = labels.cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    for prob in  (outputs.data.cpu().detach().numpy())[:,1]:
      test_prob_class1.append(prob)
    for pred in  predicted.cpu().detach().numpy():
      test_predictions.append(pred)

result = f1_score(test_dataset.labels, test_predictions, average='binary')
print('F1 of the model on the test images: %.2f' % result)

# best_epoch.append((epoch, result))
file_name = 'model2_result_' + str(result) + '.pkl'
file_path = os.path.join('./', file_name)
torch.save(cnn.state_dict(), file_path)

# plot error and loss
x = range(1, 1 + len(epoch_train_error_list.tolist()))
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,5))
ax.plot(x, epoch_train_loss_list.tolist(), label='Training-set')
ax.plot(x, epoch_test_loss_list.tolist(), label='Test-set')
ax.set_ylabel('Loss')
ax.legend(loc='best')
ax.grid(axis='both', which='both')
ax.set_title('Loss plot')

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,5))
ax.plot(x, epoch_train_error_list.tolist(), label='Training-set')
ax.plot(x, epoch_test_error_list.tolist(), label='Test-set')
ax.set_ylabel('Error [%]')
ax.set_xlabel('Epoch')
ax.legend(loc='best')
ax.grid(axis='both', which='both')
ax.set_title('Error plot')

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,5))
ax.plot(x, epoch_train_f1_list.tolist(), label='Training-set')
ax.plot(x, epoch_test_f1_list.tolist(), label='Test-set')
ax.set_ylabel('F1 Score ')
ax.set_xlabel('Epoch')
ax.legend(loc='best')
ax.grid(axis='both', which='both')
ax.set_title('F1 plot')

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,5))
ax.plot(x, epoch_train_auc_list.tolist(), label='Training-set')
ax.plot(x, epoch_test_auc_list.tolist(), label='Test-set')
ax.set_ylabel('ROC AUC')
ax.legend(loc='best')
ax.grid(axis='both', which='both')
ax.set_title('ROC AUC plot')



