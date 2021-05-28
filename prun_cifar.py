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

BATCH = 512
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH, shuffle=False)

def get_accuracy(test_loader):
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

EP = 8
model_path = 'pth/train_res.pth' + str(EP)

#定义超参数
lr = 3e-4
kappa = 1
mu = 200
interval = 20

model = ResNet().to(device)
optimizer = get_slbi(model,lr=lr,kappa=kappa,mu=mu)

train_accs  = []
test_accs = []

def cal_sparsity(model):
    model.eval()
    weight = model.conv3_1.weight.clone().detach().cpu().numpy()
    #print('weight', weight.shape)
    H = 10
    W = 10
    before_weight = np.zeros((H*3,W*3,1))
    size = before_weight.shape[0] * before_weight.shape[1]
    for i in range(H):
        for j in range(W):
            before_weight[i*3:i*3+3, j*3:j*3+3,0] = weight[i][j]
    
    zeros_in_weight = np.sum(before_weight==0)
    sparsity = (size - zeros_in_weight) / size
    return sparsity

def train_one_epoch(train_loader):
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

def look_prun(ratio=10):
    original_acc = get_accuracy(test_loader)
    optimizer.prune_layer_by_order_by_name(ratio, 'conv3_1.weight', True)    
    prun_acc = get_accuracy(test_loader)
    optimizer.recover()
    return original_acc, prun_acc

optimizer.update_prune_order(EP)
model.load_state_dict(torch.load(model_path))

#训练后进行剪枝查看

#使用不同的剪枝比例
original_acc_5, prun_acc_5 = look_prun(5)
original_acc_10, prun_acc_10 = look_prun(10)
original_acc_20, prun_acc_20 = look_prun(20)
original_acc_30, prun_acc_30 = look_prun(30)
original_acc_40, prun_acc_40 = look_prun(40)
original_acc_50, prun_acc_50 = look_prun(50)
original_acc_60, prun_acc_60 = look_prun(60)
original_acc_70, prun_acc_70 = look_prun(70)
original_acc_80, prun_acc_80 = look_prun(80)

print('5','original',original_acc_5,'pruned',prun_acc_5)
print('10','original',original_acc_10,'pruned',prun_acc_10)
print('20','original',original_acc_20,'pruned',prun_acc_20)
print('30','original',original_acc_30,'pruned',prun_acc_30)
print('40','original',original_acc_40,'pruned',prun_acc_40)
print('50','original',original_acc_50,'pruned',prun_acc_50)
print('60','original',original_acc_60,'pruned',prun_acc_60)
print('70','original',original_acc_70,'pruned',prun_acc_70)
print('80','original',original_acc_80,'pruned',prun_acc_80)