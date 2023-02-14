import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

import resnet
import numpy as np
import argparse

import matplotlib.pyplot as plt

learning_rate = 0.1
epsilon = 0.0314
k = 7
alpha = 0.00784

file_name = 'pgd_at'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def parse_args():
#     parser = argparse.ArgumentParser(description='Defense against Deep Leakage.')
#     parser.add_argument('--seed', type=int, default=0,
#                     help='random seed')
        
#     args = parser.parse_args()
#     return args

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        # x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

# Load Model
net = resnet.ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)

net2 = resnet.ResNet18()
net2 = net2.to(device)
net2 = torch.nn.DataParallel(net2)
cudnn.benchmark = True

adversary = LinfPGDAttack(net)
adversary2 = LinfPGDAttack(net2)
adv_prev = None


delta_diff_list = []
acc_current_list = []
acc_previous_on_next_list = []
acc_next_list = []
for epoch in range(1, 200, 10):
    net.load_state_dict(torch.load('./checkpoint/' + file_name+'_epoch'+str(epoch)+'.pt')['net'])
    net2.load_state_dict(torch.load('./checkpoint/' + file_name+'_epoch'+str(epoch+1)+'.pt')['net'])
    net.eval()
    net2.eval()
    
    adv_correct=0
    adv_correct_prev=0
    adv_correct_next = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            print('=====================================================')
            inputs, targets = inputs.to(device), targets.to(device)
            # import ipdb; ipdb.set_trace()
            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()
            print('Epoch'+str(epoch)+' Acc of current adv'+ str(adv_correct/targets.shape[0]))
            acc_current_list.append(adv_correct/targets.shape[0])

            # Check whether previous adversarial examples still adversarial for next epoch model
            adv_prev_outputs = net2(adv)
            _, predicted_prev = adv_prev_outputs.max(1)
            adv_correct_prev += predicted_prev.eq(targets).sum().item()
            print('Epoch'+str(epoch+1)+' Acc of previous adv'+ str(adv_correct_prev/targets.shape[0]))
            acc_previous_on_next_list.append(adv_correct_prev/targets.shape[0])

            # Check the adv acc of adversarial points on next epoch model
            adv_next = adversary2.perturb(inputs, targets)
            adv_next_outputs = net2(adv_next)
            _, predicted_next = adv_next_outputs.max(1)
            adv_correct_next += predicted_next.eq(targets).sum().item()
            print('Epoch'+str(epoch+1)+' Acc of current adv'+ str(adv_correct_next/targets.shape[0]))
            acc_next_list.append(adv_correct_next/targets.shape[0])

            # Check the angle between two adversarial points of epoch and epoch+1
            delta = adv - inputs
            delta_next = adv_next - inputs

            delta_cos_diff = ((delta.view(100,-1) / delta.view(100,-1).norm(dim=1).view(100,-1))@(delta_next.view(100,-1) / delta_next.view(100,-1).norm(dim=1).view(100,-1)).T).diag()
            delta_diff_list.append(delta_cos_diff.cpu().numpy())
            # import ipdb; ipdb.set_trace()

            print('=====================================================')

            break

fig = plt.figure(figsize=(10,6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_title('Adv Acc vs Epoch')
ax.set_ylim([0.2,0.6])
ax.set_yticks([0.2,0.3,0.4,0.5,0.6])
ax.plot(acc_current_list, label="current")
ax.plot(acc_previous_on_next_list, label='cu on next')
ax.plot(acc_next_list, label='next')
plt.legend()
plt.savefig('./figures/test_adv_align_acc_no_rand.png')

fig = plt.figure(figsize =(20, 7))
# Creating axes instance
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_title("delta cos diff vs Epoch")
ax.set_ylim([0.2,1.0])
ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
# Creating plot
bp = ax.boxplot(delta_diff_list)
plt.savefig('./figures/test_adv_align_delta_cos_diff_no_rand.png')