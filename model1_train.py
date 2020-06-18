from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
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

num_epochs = 55
batch_size = 128
learning_rate = 0.001

mean_nums = [0.5226, 0.4404, 0.4206]
std_nums = [0.2336, 0.22224, 0.2187]
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      #transforms.RandomCrop(32, padding=4),
                                      transforms.Resize((40,40), interpolation=Image.BICUBIC),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean_nums, std_nums),
 ])
transform_test = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                     transforms.Resize((40,40), interpolation=Image.BICUBIC),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_nums, std_nums),
])


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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.PReLU())
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.PReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.PReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            # nn.norm to try
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 28, kernel_size=3, padding=1),
            nn.PReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.PReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(28, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(1600, 2)  ######!!!!!!!!!!!!!!!! 10
        self.logsoftmax = nn.LogSoftmax()
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        # self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.dropout(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.dropout(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        # out = self.dropout(out)
        # out= self.layer8(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = self.dropout(out)
        return self.logsoftmax(out)


# Build Model define loss and optimizer
def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


cnn = CNN()
cnn = to_gpu(cnn)

# convert all the weights tensors to cuda()
# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
criterion = to_gpu(criterion)
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
print(f'Num of trainable parameters : {sum(p.numel() for p in cnn.parameters() if p.requires_grad)}')

epoch_train_error_list = []
epoch_train_loss_list = []
epoch_train_f1_list = []
epoch_train_auc_list = []

epoch_test_error_list = []
epoch_test_loss_list = []
epoch_test_f1_list = []
epoch_test_auc_list = []

# Training the Model
for epoch in range(num_epochs):
    print(epoch)
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = to_gpu(images)
        labels = to_gpu(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()  # what this line does
    print('Epoch: [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, (running_loss / len(train_loader))))
    # test
    correct = 0
    total = 0
    running_test_loss = 0
    predictions = []
    prob_class1 = []
    for images, labels in test_loader:
        images = to_gpu(images)
        labels = to_gpu(labels)
        outputs = cnn(images)
        for prob in (outputs.data.cpu().detach().numpy())[:, 1]:
            prob_class1.append(prob)
        _, predicted = torch.max(outputs.data, 1)
        for pred in predicted.cpu().detach().numpy():
            predictions.append(pred)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss = criterion(outputs, labels)
        running_test_loss += loss.item()

    epoch_test_error_list.append(1 - float(correct) / float(total))
    epoch_test_loss_list.append(float(running_test_loss) / float(len(test_loader)))
    epoch_test_f1_list.append(f1_score(test_dataset.labels, predictions, average='binary'))
    epoch_test_auc_list.append(roc_auc_score(test_dataset.labels, prob_class1))

    # train
    correct = 0
    total = 0
    running_train_loss = 0
    predictions = []
    prob_class1 = []
    for images, labels in train_loader:
        images = to_gpu(images)
        labels = to_gpu(labels)
        outputs = cnn(images)
        for prob in (outputs.data.cpu().detach().numpy())[:, 1]:
            prob_class1.append(prob)
        _, predicted = torch.max(outputs.data, 1)
        for pred in predicted.cpu().detach().numpy():
            predictions.append(pred)

        total += labels.size(0)
        correct += (predicted == labels).sum()
    epoch_train_error_list.append(1 - float(correct) / float(total))
    epoch_train_loss_list.append((running_loss / len(train_loader)))
    epoch_train_f1_list.append(f1_score(train_dataset.labels, predictions, average='binary'))
    epoch_train_auc_list.append(roc_auc_score(train_dataset.labels, prob_class1))

# Test the Model
correct = 0
total = 0
test_predictions = []
test_prob_class1 = []

for images, labels in test_loader:
    images = to_gpu(images)
    labels = to_gpu(labels)
    outputs = cnn(images)
    for prob in (outputs.data.cpu().detach().numpy())[:, 1]:
        test_prob_class1.append(prob)
    _, predicted = torch.max(outputs.data, 1)
    for pred in predicted.cpu().detach().numpy():
        test_predictions.append(pred)
    total += labels.size(0)
    correct += (predicted == labels).sum()

result = f1_score(test_dataset.labels, test_predictions, average='binary')
print('F1 of the model on the test images: %.2f' % result)


#Save the Model
file_name = 'model1_result_' + str(result) + '.pkl'
file_path = os.path.join('./', file_name)
torch.save(cnn.state_dict(), file_path)

# plot error and loss

x = range(1, 1 + len(epoch_train_error_list))
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,5))
ax.plot(x, epoch_train_loss_list, label='Training-set')
ax.plot(x, epoch_test_loss_list, label='Test-set')
ax.set_ylabel('Loss')
ax.legend(loc='best')
ax.grid(axis='both', which='both')
ax.set_title('Loss plot')

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,5))
ax.plot(x, epoch_train_error_list, label='Training-set')
ax.plot(x, epoch_test_error_list, label='Test-set')
ax.set_ylabel('Error [%]')
ax.set_xlabel('Epoch')
ax.legend(loc='best')
ax.grid(axis='both', which='both')
ax.set_title('Error plot')

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,5))
ax.plot(x, epoch_train_f1_list, label='Training-set')
ax.plot(x, epoch_test_f1_list, label='Test-set')
ax.set_ylabel('F1 Score ')
ax.set_xlabel('Epoch')
ax.legend(loc='best')
ax.grid(axis='both', which='both')
ax.set_title('F1 plot')


fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,5))
ax.plot(x, epoch_train_auc_list, label='Training-set')
ax.plot(x, epoch_test_auc_list, label='Test-set')
ax.set_ylabel('ROC AUC')
ax.set_xlabel('Epoch')
ax.legend(loc='best')
ax.grid(axis='both', which='both')
ax.set_title('ROC AUC plot')


