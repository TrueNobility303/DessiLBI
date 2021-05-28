import torch
from torch.utils.data import dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from slbi_toolbox import SLBI_ToolBox
from config import * 
from collections import namedtuple
import matplotlib.pyplot as plt 
import numpy as np


#加载预训练模型并且使用DessiLBI进行剪枝

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

BATCH = 64
train_dataset = torchvision.datasets.MNIST(root='/datasets/MNIST', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='/datasets/MNIST', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH, shuffle=False)

model = Net().to(device)
#加载预训练模型
load_pth = torch.load('lenet.pth')
model.load_state_dict(load_pth['model'])

layer_list = []
name_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    #print(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)

lr = 0.1
kappa = 1
mu = 20
interval = 20

optimizer = SLBI_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)
optimizer.update_prune_order(0)

def train():
    for ep in range(1,1):
        model.train()

        loss_val = 0
        correct = num = 0
        for iter, pack in enumerate(train_loader):
            data, target = pack[0].to(device), pack[1].to(device)
            logits = model(data)
            loss = F.nll_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = logits.max(1)
            loss_val += loss.item()
            correct += pred.eq(target).sum().item()
            num += data.shape[0]
        if ep  % 1 == 0:
            print('epoch', ep , 'loss', loss_val/100, 'acc', correct/num)
            correct = num = 0
            loss_val = 0
        optimizer.update_prune_order(ep)

def test_acc():
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

#定义全局变量
cnt_conv = 0
cnt_relu = 0
MODE = 'mode_'
def viz_conv(module, inputs):
    global cnt_conv
    cnt_conv += 1
    if cnt_conv > 8:
        return
    x = inputs[0][0]
    x = x.permute(1,2,0).cpu().detach().numpy()
    C = x.shape[2]
    for i in range(C):
        plt.subplot(1, C, i+1)
        plt.imshow(x[:,:,i],cmap='gray')
        plt.axis('off')
    global MODE
    plt.savefig('dump/' + MODE + str(cnt_conv) + 'conv_act.png')

if __name__ == '__main__':
    train()

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz_conv)

    inputs,target = test_dataset[0]
    inputs = inputs.unsqueeze(0).to(device)

    #获取weight
    weight = model.conv3.weight.clone().detach().cpu().numpy()
    #print('weight', weight.shape)
    H = 10
    W = 10
    before_weight = np.zeros((H*5,W*5,1))
    for i in range(H):
        for j in range(W):
            before_weight[i*5:i*5+5, j*5:j*5+5,0] = weight[i][j]
    before_weight = np.abs(before_weight)
    #before_weight = weight[6][0]
    MODE = 'before_prun_'
    model(inputs)
    
    print('origin acc：',test_acc())
    optimizer.prune_layer_by_order_by_name(80, 'conv3.weight', True)
    print('acc after prun conv3：',test_acc())

    weight = model.conv3.weight.clone().detach().cpu().numpy()
    after_weight = np.zeros((H*5,W*5,1))
    for i in range(H):
        for j in range(W):
            after_weight[i*5:i*5+5, j*5:j*5+5,0] = weight[i][j]
    after_weight = np.abs(after_weight)

    cnt_conv = 0
    cnt_relu = 0
    MODE = 'after_prun_'
    model(inputs)

    plt.subplot(1,2,1)
    plt.imshow(before_weight,cmap='gray')
    plt.title('before pruning')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(after_weight,cmap='gray')
    plt.axis('off')
    plt.title('after pruning')
    plt.savefig('dump/conv3_weight.png')
    
    optimizer.recover()
    print('acc after recover：',test_acc())

    print('origin acc：',test_acc())
    optimizer.prune_layer_by_order_by_list(80, ['conv3.weight','fc1.weight'], True)
    print('acc after prun conv3 and fc1：',test_acc())
    optimizer.recover()
    print('acc after recover：',test_acc())

    
   
    


