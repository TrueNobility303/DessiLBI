import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from slbi_toolbox import SLBI_ToolBox
from config import * 
from collections import namedtuple
import matplotlib.pyplot as plt 
from optim.slbi_sgd import SLBI_SGD_ToolBox
from optim.slbi_adam import SLBI_ADAM_ToolBox
import numpy as np 

#使用DessiLBI训练MNIST手写数字识别,与SGD对比

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

BATCH = 512
train_dataset = torchvision.datasets.MNIST(root='/datasets/MNIST', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='/datasets/MNIST', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH, shuffle=False)

model = Net().to(device)
load_pth = torch.load('pth/sgd_lenet.pth')
model.load_state_dict(load_pth)
weight = model.conv3.weight.clone().detach().cpu().numpy()
H = 10
W = 10
sgd_weight = np.zeros((H*5,W*5,1))
for i in range(H):
    for j in range(W):
        sgd_weight[i*5:i*5+5, j*5:j*5+5,0] = weight[i][j]
sgd_weight = np.abs(sgd_weight)

#LeNeT by slbi
model = Net().to(device)
load_pth = torch.load('lenet.pth')
model.load_state_dict(load_pth['model'])
weight = model.conv3.weight.clone().detach().cpu().numpy()
slbi_weight = np.zeros((H*5,W*5,1))
for i in range(H):
    for j in range(W):
        slbi_weight[i*5:i*5+5, j*5:j*5+5,0] = weight[i][j]
slbi_weight = np.abs(slbi_weight)

plt.figure()
plt.clf()
plt.subplot(1,2,1)
plt.imshow(sgd_weight,cmap='gray')
plt.axis('off')
plt.title('sgd')

plt.subplot(1,2,2)
plt.imshow(slbi_weight,cmap='gray')
plt.axis('off')
plt.title('slbi')

plt.savefig('exp/cmp_weight_sparsity.png')