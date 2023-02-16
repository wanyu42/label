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

def parse_args():
    parser = argparse.ArgumentParser(description='Remargin Training.')
    parser.add_argument('--temperature', type=float, default=0.05,
                    help='temperature for remargin')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10 or CIFAR100')
    parser.add_argument('--train_steps', type=int, default=7, help='attack steps for training')
    parser.add_argument('--test_steps', type=int, default=20, help='attack steps for testing')
        
    args = parser.parse_args()
    return args

args = parse_args()

learning_rate = 0.1
epsilon = 0.0314
# k = 7
alpha = 0.00784

temperature = args.temperature
file_name = args.dataset+'pgd_remargin_tem'+str(temperature)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the embedding
def loadEmbeddings(filename):
    print("Loading embeddings from file", filename)
    f = open(filename,'r', encoding='utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

# Get the similarity score between labels
gembs = loadEmbeddings(os.path.expanduser('~')+"/dataset/glove.840B.300d.top25k.txt")
if args.dataset == "CIFAR10":
    CIFAR_label = ['plane','auto','bird','cat', 'deer','dog', 'frog','horse','ship','truck']
elif args.dataset == "CIFAR100":
    CIFAR_label = torch.load("cifar100_label.pt")
import ipdb; ipdb.set_trace()
label_emb = torch.cat([torch.tensor(gembs[label]).view(1,-1) for label in CIFAR_label]).to(device)
label_sim = label_emb @ label_emb.T
label_sim = label_sim - torch.diag(torch.diag(label_sim))
# import ipdb; ipdb.set_trace()

# Load the dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "CIFAR10":
    num_classes = 10
    train_dataset = torchvision.datasets.CIFAR10(root='~/dataset', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='~/dataset', train=False, download=True, transform=transform_test)
elif args.dataset == "CIFAR100":
    num_classes = 100
    train_dataset = torchvision.datasets.CIFAR100(root='~/dataset', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='~/dataset', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

# Linf Attack
class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y, num_steps):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary, num_steps):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y, num_steps)
    return adv

# Load Model
net = resnet.ResNet18(num_classes)
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

adversary = LinfPGDAttack(net)
# criterion = nn.CrossEntropyLoss()

def RemarginLoss(logits, targets, temp=temperature):
    remargin = label_sim[targets]
    # import ipdb; ipdb.set_trace()
    return F.cross_entropy(logits + temp * remargin / remargin.std() * logits.std().detach(), targets)

criterion = RemarginLoss
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

# best_state = None
train_state = {'train_acc':[], 'test_robust_acc':[], 'test_benign_acc':[], 'train_loss':[], 'test_adv_loss':[], 
                'best_epoch':0, 'best_benign_test':0.0, 'best_adv_test':0.0, 'best_state':None}
def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Still use CE to generate adversarial examples
        adv = adversary.perturb(inputs, targets)
        adv_outputs = net(adv)
        # Use RemarginLoss to train the network
        loss = criterion(adv_outputs, targets)
        loss.backward()

        optimizer.step()
        # RemarginLoss
        train_loss += loss.item() * targets.size(0)
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial train loss:', loss.item())

    print('\nTotal adversarial train accuarcy:', 100. * correct / total)
    print('Total adversarial train loss:', train_loss/total)
    train_state['train_acc'].append(100. * correct / total)
    train_state['train_loss'].append(train_loss/total)

def test(epoch):
    global train_state
    
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            # Remargin Loss
            loss = criterion(outputs, targets)
            benign_loss += loss.item() * targets.size(0)

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            # if batch_idx % 10 == 0:
            #     print('\nCurrent batch:', str(batch_idx))
            #     print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            #     print('Current benign test loss:', loss.item())

            # Still Use CrossEntropy loss to generate adversarial
            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            # Remargin Loss
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item() * targets.size(0)

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

            # if batch_idx % 10 == 0:
            #     print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            #     print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss/total)
    print('Total adversarial test loss:', adv_loss/total)
    train_state["test_benign_acc"].append(100. * benign_correct / total)
    train_state["test_robust_acc"].append(100. * adv_correct / total)
    train_state["test_adv_loss"].append(adv_loss/total)

    if (100. * adv_correct / total) > best_adv_test:
        train_state["best_epoch"] = epoch
        train_state["best_state"] = {
            'net': net.state_dict()
        }
        train_state["best_adv_test"] = 100. * adv_correct / total
        train_state["best_benign_test"] = 100. * benign_correct / total

    if epoch % 10 == 0:
        state = {
            'net': net.state_dict()
        }
        if not os.path.isdir('../checkpoint'):
            os.mkdir('../checkpoint')
        torch.save(state, '../checkpoint/' + file_name+'_epoch'+str(epoch)+'.pt')
        print('Model Saved!')

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

print('Remargin Loss with temperature '+str(temperature))
for epoch in range(1, 201):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
print("Save Best Model at Epoch "+str(best_epoch)+"\tBenign Acc: "+str(best_benign_test)+"\tAdv Acc: "+str(best_adv_test))
print('Remargin Loss with temperature '+str(temperature))
print("Save Best Model")
if not os.path.isdir('../checkpoint'):
    os.mkdir('../checkpoint')
torch.save(train_state, '../checkpoint/' + file_name+'best_epoch'+str(best_epoch)+'.pt')

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.title(args.dataset+'\tRemargin t='+str(temperature))
# plt.subplot(121)
# plt.gca().set_title('Acc vs Epoch')
# plt.plot(train_acc_list, label="train_acc")
# plt.plot(test_benign_acc_list, label='test_benign_acc')
# plt.plot(test_robust_acc_list, label='test_adv_acc')
# plt.legend()
# plt.subplot(122)
# plt.gca().set_title('Loss vs Epoch')
# plt.plot(train_loss_list, label='train_loss')
# plt.plot(test_adv_loss_list, label='test_loss')
# plt.legend()
# plt.savefig('./figures/'+file_name+'.png')
