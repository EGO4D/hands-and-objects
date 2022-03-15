import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import types
import os.path as osp
from tqdm import tqdm

def accuracy(output, target):
    maxk = 1
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    k = 1
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    res = correct_k.mul_(100.0 / batch_size)
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train(model, loader, optimizer, criterion, epoch, state):
    losses = AverageMeter('bceloss')
    model.train()
    for i, sample in tqdm(enumerate(loader), total=len(loader)):
        x = sample[0].cuda()
        if state:
            y = sample[2].cuda().long()
        else:
            y = sample[1].cuda().float()
        output = model(x)
        if state:
            output = output[-1]
        loss = criterion(output, y)
        losses.update(loss.item(), x.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 500 == 0:
            print('Training Epoch: %d, %d/%d, loss: %.4f'%(epoch, i, len(loader), losses.avg))
    print('Finished Training Epoch: %d, loss: %.4f'%(epoch, losses.avg))

def val(model, loader, criterion, epoch, state):
    losses = AverageMeter('loss')
    accs = AverageMeter('acc')
    model.eval()
    for i, sample in tqdm(enumerate(loader), total=len(loader)):
        x = sample[0].cuda()
        if state:
            y = sample[2].cuda().long()
        else:
            y = sample[1].cuda().float()
        with torch.no_grad():
            output = model(x)
            if state:
                output = output[-1]
            loss = criterion(output, y)
            losses.update(loss.item(), x.shape[0])
            if state:
                acc = accuracy(output, y)
                accs.update(acc.item(), x.shape[0])

        if (i+1) % 500 == 0:
            print('Validation Epoch: %d, %d/%d, loss: %.4f'%(epoch, i, len(loader), losses.avg))
            

    print('Finished Validation Epoch: %d, loss: %.4f'%(epoch, losses.avg))
    return losses.avg, None, None