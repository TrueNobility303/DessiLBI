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
#使用DessiLBI训练MNIST手写数字识别

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
layer_list = []
name_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    #print(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)

lr = 0.01
kappa = 1
mu = 20
interval = 20

optimizer = SLBI_ADAM_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

exp_path = 'exp/train_lenet.png'
model_path = 'pth/train_lenet.pth'

train_accs  = []
test_accs = []

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

#共训练20轮次
for ep in range(20):

    #学习率衰减
    lr = lr * (0.1 ** (ep //interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #训练
    loss_val = 0
    correct = 0
    num = 0
    model.train()
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

    train_acc = get_accuracy(train_loader)
    test_acc = get_accuracy(test_loader)
    train_accs .append(train_acc)
    test_accs.append(test_acc)

    if ep  % 1 == 0:
        print('epoch', ep , 'loss', loss_val, 'acc', test_acc)
        correct = num = 0
        loss_val = 0
    
    torch.save(model.state_dict, model_path)
    #更新剪枝顺序
    optimizer.update_prune_order(ep)

plt.figure()
plt.clf()
plt.plot(train_accs, label='train')
plt.plot(test_accs, label='test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig(exp_path)

#训练后进行剪枝查看
print('origin acc：',get_accuracy(test_loader) )
optimizer.prune_layer_by_order_by_name(80, 'conv3.weight', True)
print('acc after prun conv3：',get_accuracy(test_loader))
optimizer.recover()
print('acc after recover：',get_accuracy(test_loader))

print('origin acc：', get_accuracy(test_loader))
optimizer.prune_layer_by_order_by_list(80, ['conv3.weight','fc1.weight'], True)
print('acc after prun conv3 and fc1：',get_accuracy(test_loader))
optimizer.recover()
print('acc after recover：',get_accuracy(test_loader))
    