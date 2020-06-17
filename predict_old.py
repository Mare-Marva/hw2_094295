from __future__ import print_function, division
import torch.nn.functional as Func
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import math
import os
from PIL import Image
from sklearn.metrics import f1_score
import pandas as pd
import argparse


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

test_folder = args.input_folder
model_path = 'model2.pkl'

class Dataset_Masks(Dataset):
    def __init__(self, listIDs, labels, transform=None, train=True):
        self.labels = labels
        self.list_IDs = listIDs
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        x = Image.open(os.path.join(test_folder, self.list_IDs[idx] + '_' + str(self.labels[idx]) + '.jpg'))
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

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


# Hyper Parameters
num_epochs = 300
batch_size = 256
learning_rate = 0.01

mean_nums = [0.5226, 0.4404, 0.4206]
std_nums = [0.2336, 0.22224, 0.2187]

transform_test = transforms.Compose([transforms.Resize((40,40), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)])

files_test = os.listdir(test_folder)
x_test = [file_path.split('_')[0] for file_path in files_test]
y_test = [int(file_path.split('_')[1].split('.')[0]) for file_path in files_test]
test_dataset = Dataset_Masks(x_test, y_test, transform_test, False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

## Load the model
cnn_load = DenseNet(growthRate=12, depth=20)
cnn_load = to_gpu(cnn_load)
cnn_load.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

## Test the model on the new test set
cnn_load.eval()
correct1 = 0
total1 = 0
predictions = []
for images1, labels1 in test_loader:
    if torch.cuda.is_available():
        images1 = images1.cuda()
        labels1 = labels1.cuda()
    outputs1 = cnn_load(images1)

    _, predicted1 = torch.max(outputs1.data, 1)
    for pred in predicted1.cpu().detach().numpy():
      predictions.append(pred)
    total1 += labels1.size(0)
    correct1 += (predicted1 == labels1).sum()
 
    
result1 = f1_score(test_dataset.labels, predictions, average='binary')
print('F1 of the model on the test images:')
print(result1)


predict_values = {'id': test_dataset.list_IDs , 'label': predictions}
prediction_df = pd.DataFrame(predict_values,columns=['id', 'label'])


## Save the predictions
prediction_df.to_csv("prediction.csv", index=False, header=False)