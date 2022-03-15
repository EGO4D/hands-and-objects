import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from trainval import accuracy
from tqdm import tqdm
import pickle

# state = True

def keyframe_distance(preds, uid_list):
    distance_list = list()
    sec_list = list()
    for pred, gt in zip(preds, uid_list):
        clip_length = gt['json_parent_end_sec'].item() - gt['json_parent_start_sec'].item()
        clip_frames = gt['json_parent_end_frame'].item() - gt['json_parent_start_frame'].item() + 1
        fps = clip_frames / clip_length
        keyframe_loc_pred = np.argmax(pred)
        keyframe_loc_pred = np.argmax(pred)
        keyframe_loc_pred_mapped = (gt['json_parent_end_frame'].item() - gt['json_parent_start_frame'].item()) / 16 * keyframe_loc_pred
        keyframe_loc_gt = gt['pnr_frame'].item() - gt['json_parent_start_frame'].item()
        err_frame = abs(keyframe_loc_pred_mapped - keyframe_loc_gt)
        err_sec = err_frame / fps
        distance_list.append(err_frame.item())
        sec_list.append(err_sec.item())
    # When there is no false positive
    if len(distance_list) == 0:
        # Should we return something else here?
        return 0, 0
    return np.array(distance_list), np.array(sec_list)

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

def evaluate(model, loader, criterion, epoch, state, is_validation=False):
    losses = AverageMeter('loss')
    accs = AverageMeter('acc')
    model.eval()
    ys = []
    os = []
    # fps_list = []
    uid_list = []
    for i, sample in tqdm(enumerate(loader), total=len(loader)):
        # 16 frames
        x = sample[0].cuda()
        if is_validation:
            if state:
                y = sample[2].cuda().long()
            else:
                y = sample[1].cuda().float()
        effective_fps = sample[-2]
        with torch.no_grad():
            output = model(x)
            if state:
                output = output[-1]
            if is_validation:
                loss = criterion(output, y)
                losses.update(loss.item(), x.shape[0])
                if state:
                    acc = accuracy(output, y)
                    accs.update(acc.item(), x.shape[0])
        if not state:
            if is_validation:
                ys.append(y.cpu().numpy())
        os.append(output.cpu().numpy())
        uid_list.append(sample[-1])

        if (i+1) % 500 == 0:
            if not state and is_validation:
                dist, sec = keyframe_distance(os, uid_list)
                print('frame dist:', np.mean(dist), 'seconds dist:', np.mean(sec))

    if state and is_validation:
        print('Finished Evaluation, accuracy is: %.4f'%(accs.avg))
    if not state and is_validation:
        dist, sec = keyframe_distance(os, uid_list)
        print('Finished Evaluation, frame dist:', np.mean(dist), 'seconds dist:', np.mean(sec))
    return accs.avg, os, uid_list

def generate_submission_file(output_list, uid_list):
    res = []
    for output, info in zip(output_list, uid_list):
        pred = np.argmax(output)
        pred_mapped = (info['json_parent_end_frame'].item() - info['json_parent_start_frame'].item()) / 16 * pred
        res.append({'unique_id': info['unique_id'][0], 'pnr_frame': pred_mapped})
    
    return res

def generate_submission_file_cls(output_list, uid_list):
    res = []
    for output, info in zip(output_list, uid_list):
        pred = np.argmax(output)
        pred = pred == 1
        res.append({'unique_id': info['unique_id'][0], 'state_change': bool(pred)})
    
    return res