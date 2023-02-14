import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.functional as F

import resnet

from autoattack import AutoAttack
import numpy as np
import copy

import seaborn as sns
import matplotlib.pyplot as plt

def js_divergence(prob_1, prob_2):    
    total_m = 0.5 * (prob_1 + prob_2)
    
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
    return (0.5 * loss)

class_name = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
# "airplane" with "frog", "auto" with "horse" (0 with 6, 1 with 7)
visual_name = ['frog', 'horse', 'bird', 'cat', 'deer', 'dog', 'plane', 'auto','ship','truck']

def swap_indices(A, i,j):
    A_new = copy.deepcopy(A)
    spare_row = copy.deepcopy(A_new[i])
    A_new[i] = A_new[j]
    A_new[j] = spare_row

    spare_col = copy.deepcopy(A_new[:,i])
    A_new[:, i] = A_new[:, j]
    A_new[:, j] = spare_col
    return A_new

def cifar2visual(A):
    A = swap_indices(A, 0, 6)
    A = swap_indices(A, 1, 7)
    return A

# import ipdb; ipdb.set_trace()

model_dir = './checkpoint'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet.ResNet18()
model = torch.nn.DataParallel(model)
model = model.to(device)
# standard trained model
# adv='clean'
adv='adv_linf'
temperature=0.05
if adv == 'clean':
    modelpath = model_dir + '/CIFAR10_VAL_ResNet18_epoch_200.pt'
# adversarial trained model for linf
elif adv == 'adv_linf':
    modelpath = model_dir + '/pgd_remargin_tem'+str(temperature)+'_epoch100.pt'
checkpoint = torch.load(modelpath, map_location=device)
model.load_state_dict(checkpoint['net'])

model.eval()

# Load Test Data
batch_size=128
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])
cifar_test = datasets.CIFAR10("~/dataset", train=False, download=True, transform=transform_test)
test_loader = data.DataLoader(cifar_test, batch_size = batch_size, shuffle=True)


# Set the Adversary
epsilon=8/255
norm='Linf'
adversary = AutoAttack(model, norm=norm, eps=epsilon, version="custom", attacks_to_run=['apgd-ce'])

pred_all = []
y_all = []
with torch.no_grad():
    for idx, (x,y) in enumerate(test_loader):
        x,y = x.to(device), y.to(device)
        adv_complete = adversary.run_standard_evaluation(x, y,bs=batch_size)

        output = model(adv_complete)
        pred = output.argmax(dim=1, keepdim=True)

        pred_all.append(pred)
        y_all.append(y)

# import ipdb; ipdb.set_trace()
pred_all = torch.cat(pred_all)
y_all = torch.cat(y_all)
import ipdb; ipdb.set_trace()
print("Overall Accuracy: ", (pred_all!=y_all).sum()/pred_all.shape[0])

# Each row correspond to one ground truth class
pred_dist = torch.zeros([10, 10])
for i in range(pred_all.shape[0]):
    pred_dist[y_all[i]][pred_all[i]] += 1
pred_correct = torch.diag(pred_dist)
# import ipdb; ipdb.set_trace()
pred_acc = torch.div(pred_correct, torch.sum(pred_dist, dim=1))
print("Per Class Accuracy:", pred_acc.cpu().numpy())
wrong_freq = pred_dist - torch.diag(pred_correct)
wrong_dist = (wrong_freq.T/torch.sum(wrong_freq, dim=1)).T
print("Wrong Distribution: \n",np.around(wrong_dist.cpu().numpy(),3))

# for name in visual_name:
#     print(name)
ax = sns.heatmap(cifar2visual(np.around(wrong_dist.cpu().numpy(),3)), annot=True, xticklabels=visual_name, yticklabels=visual_name)
ax.set_title('Remargin T '+str(temperature))
ax.xaxis.tick_top()
plt.savefig('figures/heatmap_linf'+adv+'remargin'+str(temperature)+'model.png')
# similarity = torch.zeros([10, 10])
# for i in range(10):
#     for j in range(i+1, 10):
#         similarity[i,j] = wasserstein_distance(wrong_dist[i].cpu().numpy(), wrong_dist[j].cpu().numpy())
# print("Similarity Matrix: \n", np.around(similarity.cpu().numpy(), 3))