import torch
import numpy as np
from trainval import accuracy
from tqdm import tqdm

def keyframe_distance(preds, labels, fps_list):
    distance_list = list()
    fps_list_return = list()
    for pred, label, fps in zip(preds, labels, fps_list):
        if np.amax(label) > 0:
            # Selecting the row with 
            keyframe_loc_pred = np.argmax(pred)
            keyframe_loc_gt = np.argmax(label)
            distance = np.abs(keyframe_loc_gt - keyframe_loc_pred)
            distance_list.append(distance.item())
            fps_list_return.append(fps.item())
    # When there is no false positive
    if len(distance_list) == 0:
        # Should we return something else here?
        return 0
    return np.array(distance_list), np.array(fps_list_return)

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

def evaluate(model, loader, criterion, epoch, state):
    losses = AverageMeter('loss')
    accs = AverageMeter('acc')
    model.eval()
    ys = []
    os = []
    fps_list = []
    calc = [0,0]
    for i, sample in tqdm(enumerate(loader), total=len(loader)):
        # 16 frames
        x = sample[0].cuda()
        if state:
            y = sample[2].cuda().long()
            calc[y] += 1
        else:
            y = sample[1].cuda().float()
        effective_fps = sample[-1]
        with torch.no_grad():
            output = model(x)
            if state:
                output = output[-1]
            loss = criterion(output, y)
            losses.update(loss.item(), x.shape[0])
            if state:
                acc = accuracy(output, y)
                accs.update(acc.item(), x.shape[0])
        
        ys.append(y.cpu().numpy())
        os.append(output.cpu().numpy())
        fps_list.append(effective_fps.cpu().numpy())

        if (i+1) % 500 == 0:
            print('Validation Epoch: %d, %d/%d, loss: %.4f'%(epoch, i, len(loader), losses.avg))
            if not state:
                dist, fps = keyframe_distance(os, ys, fps_list)
                print('frame dist:', np.mean(dist), 'seconds dist:', np.mean(dist/fps))
    if state:
        print('Finished Validation, acc: %.4f'%(accs.avg))
    if not state:
        dist, fps = keyframe_distance(os, ys, fps_list)
        print('Finished Validation, frame dist:', np.mean(dist), 'seconds dist:', np.mean(dist/fps))
    return accs.avg, ys, os, fps_list