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
from optim.slbi_adam import SLBI_ADAM_ToolBox
import numpy as np 

#使用DessiLBI训练CIFAR10图像分类任务
torch.manual_seed(42)

class ResNet(nn.Module):
    def __init__(self,n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.down1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        
        self.conv2 = nn.Conv2d(64,128,3,1,1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.down2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128,256,3,1,1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU()
        self.down3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(4*4*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, self.n_classes),
        )

    def forward(self,x):
        n_batch = x.shape[0]
        
        #conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.down1(x)   

        #conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.down2(x)

        #conv3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)

        residual = x 
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x += residual
        x = self.relu3_2(x)
        x = self.down3(x)

        x = x.view(n_batch,-1)
        y = self.classifier(x)
        return F.log_softmax(y, dim=1)

def get_accuracy(model,test_loader):
    model.eval()
    correct = 0
    num = 0
    for iter, pack in enumerate(test_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        _, pred = logits.max(1)
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
    acc = correct / num 
    return acc 

def get_slbi(model,lr,kappa=1,mu=20):
    layer_list = []
    name_list = []
    for name, p in model.named_parameters():
        name_list.append(name)
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            layer_list.append(name)
    #定义SLBI优化器
    optimizer = SLBI_ADAM_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
    optimizer.assign_name(name_list)
    optimizer.initialize_slbi(layer_list)
    return optimizer 

def train_one_epoch(model,train_loader):
    model.train()
    num = 0
    loss_val = 0
    for iter, pack in enumerate(train_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = F.nll_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = logits.max(1)
        loss_val += loss.item()
        num += data.shape[0]
    loss_val /= num 
    return loss_val 

BATCH = 512
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH, shuffle=False)


losses_adam = []
losses_adam_slbi = []
accs_adam = []
accs_adam_slbi = []
LR = 3e-4
EPOCH = 20
#使用adam版本的lsbi训练
model = ResNet().to(device)
optimizer = get_slbi(model,lr=LR)

for ep in range(EPOCH):
    loss = train_one_epoch(model,train_loader)
    test_acc = get_accuracy(model,test_loader)
    losses_adam_slbi.append(loss)
    accs_adam_slbi.append(test_acc)
    if ep  % 1 == 0:
        print('epoch', ep , 'loss', loss, 'accuracy', test_acc)
    
    #optimizer.update_prune_order(ep)

adam_weight3_1 = model.conv3_1.weight.clone().detach().cpu().numpy()
adam_weight3_2 = model.conv3_2.weight.clone().detach().cpu().numpy()

#使用torch的adam训练
model = ResNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
for ep in range(EPOCH):
    loss = train_one_epoch(model,train_loader)
    test_acc = get_accuracy(model,test_loader)
    losses_adam.append(loss)
    accs_adam.append(test_acc)
    if ep  % 1 == 0:
        print('epoch', ep , 'loss', loss, 'accuracy', test_acc)
    
    #optimizer.update_prune_order(ep)
slbi_weight3_1 = model.conv3_1.weight.clone().detach().cpu().numpy()
slbi_weight3_2 = model.conv3_2.weight.clone().detach().cpu().numpy()

plt.figure()
plt.clf()
plt.subplot(1,2,1)
plt.plot(losses_adam_slbi,label='slbi')
plt.plot(losses_adam,label='adam')
plt.legend()

plt.subplot(1,2,2)
plt.plot(accs_adam_slbi,label='slbi')
plt.plot(accs_adam,label='adam')
plt.legend()

plt.savefig('exp/cmp_adam.png')

#对比两者的权重

H = 10
W = 10
adam_weight = np.zeros((H*3,W*3,1))
for i in range(H):
    for j in range(W):
        adam_weight[i*3:i*3+3, j*3:j*3+3,0] = adam_weight3_1[i][j]
adam_weight = np.abs(adam_weight)

slbi_weight = np.zeros((H*3,W*3,1))
for i in range(H):
    for j in range(W):
        slbi_weight[i*3:i*3+3, j*3:j*3+3,0] = slbi_weight3_1[i][j]
slbi_weight = np.abs(slbi_weight)

plt.figure()
plt.clf()
plt.subplot(1,2,1)
plt.imshow(adam_weight,cmap='gray')
plt.title('adam')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(slbi_weight,cmap='gray')
plt.axis('off')
plt.title('slbi')
plt.savefig('exp/cmp_adam_slbi_weight.png')
