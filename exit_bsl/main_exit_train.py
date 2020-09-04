'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse


from models import *
from utils import *

import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lrad', type=float, help='adjust learning rate each epoch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--percent', '-p',default = 0, type=float, help='percent of prune')
#parser.add_argument('--resume', '-r', default='checkpoint/ckpt.pth', action='store_true', help='resume from checkpoint')
parser.add_argument('--train', default = True )


args = parser.parse_args()
args.lr = 0.01
args.resume = False
args.lrad = .9
args.train = True
args.percent = 0.35

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=0)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    #net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    #net = MobileNetV3(n_class=10, input_size=224)
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    #net = EfficientNetB0()
    #net = resnet20()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_res20_pruned35.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    #best_acc = 0
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    '''
    print('paras:',net.state_dict())
    for j,i in enumerate(net.modules()):
        print(j ,':',i)
    '''
    
    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            
            outputs1=outputs[0]
            outputs2=outputs[1]
            #loss = criterion(outputs1, targets) + criterion(outputs2, targets)
            loss = criterion(outputs2, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs1.max(1)
            total1 += targets.size(0)
            correct1 += predicted.eq(targets).sum().item()
            
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (train_loss/(batch_idx+1), 100.*correct1/total1, correct1, total1))
            
            _, predicted = outputs2.max(1)
            total2 += targets.size(0)
            correct2 += predicted.eq(targets).sum().item()
    
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
             #   % (train_loss/(batch_idx+1), 100.*correct2/total2, correct2, total2))
    
    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        total1 = 0
        total2 = 0
        
        corec1 = np.array([])
        cross = np.array([])
        corec2 = np.array([])
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                outputs1=outputs[0]
                outputs2=outputs[1]
                #loss = criterion(outputs1, targets) + criterion(outputs2, targets)
                loss = criterion(outputs2, targets)
    
                test_loss += loss.item()
                _, predicted = outputs1.max(1)
                total1 += targets.size(0)
                correct1 += predicted.eq(targets).sum().item()
                
                corec1 = np.append(corec1 , (predicted == targets).cpu().numpy())
                target1 = torch.max(outputs1,1)[1]
                criterions = nn.CrossEntropyLoss(reduce=False).to(device)
                cross = np.append(cross, criterions(outputs1, target1).cpu().numpy())
    
                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (test_loss/(batch_idx+1), 100.*correct1/total1, correct1, total1))
                
                _, predicted = outputs2.max(1)
                total2 += targets.size(0)
                correct2 += predicted.eq(targets).sum().item()
    
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  #  % (test_loss/(batch_idx+1), 100.*correct2/total2, correct2, total2))
                
                
                corec2 = np.append(corec2 , (predicted == targets).cpu().numpy())
                
        
    
        # Save checkpoint.
        acc = 100.*correct2/total2
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_res20_pruned35.pth')
            best_acc = acc
        return corec1, cross , corec2
    
    
    for epoch in range(start_epoch, start_epoch+200):
        if args.lrad :
            for par in optimizer.param_groups:
                par['lr'] = args.lrad * par['lr']
        if args.train:  
            train(epoch)
        '''
        # -------------------------------------------------------------
        #pruning 
        total = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                total += m.weight.data.numel()
        conv_weights = torch.zeros(total)
        index = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                size = m.weight.data.numel()
                conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
                index += size
    
        y, i = torch.sort(conv_weights)
        thre_index = int(total * args.percent)
        thre = y[thre_index].cuda()
        pruned = 0
        print('Pruning threshold: {}'.format(thre))
        zero_flag = False
        for k, m in enumerate(net.modules()):
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                
                mask = weight_copy.gt(thre).float().cuda()
                pruned = pruned + mask.numel() - torch.sum(mask)
                m.weight.data.mul_(mask)
                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                #print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                    #format(k, mask.numel(), int(torch.sum(mask))))
        print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    # -------------------------------------------------------------
        '''
        corec1, cross , corec2 = test(epoch)
        i = np.argsort(cross)
        #corec1=corec1[i].cumsum()
        #corec2=corec2[i].cumsum()
        #print(corec1)
        for inde in range(1):
            index = round(len(i)*inde/10)
            print((np.sum(corec1[i[:index]])-np.sum(corec2[i[:index]])+np.sum(corec2))/len(i)*100,'%  ',index/len(i),cross[i[index-1]])
    
