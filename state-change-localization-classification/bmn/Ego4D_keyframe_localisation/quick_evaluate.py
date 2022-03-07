"""
This file is used to quickly evaluate the trained model for state change
accuracy and keyframe localisation metric
"""
import pdb
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from configs.defaults import get_cfg_defaults
from datasets.Ego4DKeyframeLocalisation import Ego4DKeyframeLocalisation
from torch.utils.data import DataLoader
from utils.parser import parse_args, load_config
from models.build import build_model


def state_change_accuracy(preds, labels):
    correct = 0
    total = 0
    for pred, label in zip(preds, labels):
        pred_ = torch.argmax(pred)
        if pred_.item() == label.item():
            correct += 1
        total += 1
    accuracy = correct/total
    return accuracy


def keyframe_distance(preds, labels):
    distance_list = list()
    for pred, label in zip(preds, labels):
        keyframe_loc_pred = torch.argmax(pred[0][1])
        keyframe_loc_gt = torch.argmax(label)
        distance = torch.abs(keyframe_loc_gt - keyframe_loc_pred)
        distance_list.append(distance)
    return np.mean(distance_list)


def evaluate(cfg):
    val_dataset = Ego4DKeyframeLocalisation(cfg, 'val')
    val_dataset_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )

    model = build_model(cfg)
    checkpoint = torch.load(cfg.MISC.CHECKPOINT_FILE_PATH)
    state_dict = checkpoint['state_dict']

    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[6:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model = model.to('cuda')

    sc_preds = list()
    sc_gt = list()
    kf_preds = list()
    kf_gt = list()

    for count, data in enumerate(tqdm(val_dataset_loader, desc='Evaluating')):
        frames, keyframe_labels, state_change_labels = data
        frames[0] = frames[0].to('cuda')
        keyframe_preds, state_change_preds = model(frames)
        sc_preds.append(state_change_preds)
        sc_gt.append(state_change_labels)
        # Processing keyframe prediction only when we predict a state change and there is a state change
        if torch.argmax(state_change_preds).item() == 1 and state_change_labels.item() == 1:
            kf_preds.append(keyframe_preds)
            kf_gt.append(keyframe_labels)

    accuracy = state_change_accuracy(sc_preds, sc_gt)
    keyframe_avg_distance = keyframe_distance(kf_preds, kf_gt)
    print('Accuracy is: {}'.format(accuracy))
    print('Keyframe location average distance is: {}'.format(keyframe_avg_distance))
    pdb.set_trace()

if __name__ == '__main__':
    evaluate(load_config(parse_args()))
