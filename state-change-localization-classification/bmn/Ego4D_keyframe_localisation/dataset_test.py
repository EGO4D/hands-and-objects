"""
This file is used to test various functions created for the keyframe
localisation benchmark
"""
import pdb
import numpy as np
from tqdm import tqdm

from configs.defaults import get_cfg_defaults
from datasets.Ego4DKeyframeLocalisation import Ego4DKeyframeLocalisation
from torch.utils.data import DataLoader
from utils.parser import parse_args, load_config

def test_dataloader(cfg):
    # Loading and updating the configuration
    # cfg = get_cfg_defaults()
    # cfg.merge_from_file(config_file)
    # cfg.freeze()
    train_dataset = Ego4DKeyframeLocalisation(cfg, 'train')
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.DATA_LOADER.SHUFFLE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )
    val_dataset = Ego4DKeyframeLocalisation(cfg, 'val')
    val_dataset_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )
    test_dataset = Ego4DKeyframeLocalisation(cfg, 'test')
    test_dataset_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    for count, data in enumerate(tqdm(train_dataset_loader, disable=True)):
        frames, labels, state_change = data
        print('[{} TRAIN] Frames: {} labels: {} state change: {}'.format(
            count,
            frames[0].shape,
            labels.shape,
            state_change.shape
        ))
    for count, data in enumerate(tqdm(test_dataset_loader, disable=True)):
        frames, labels, state_change = data
        print('[{} TEST] Frames: {} labels: {} state change: {}'.format(
            count,
            frames[0].shape,
            labels.shape,
            state_change.shape
        ))
    for count, data in enumerate(tqdm(val_dataset_loader, disable=True)):
        frames, labels, state_change = data
        print('[{} VAL] Frames: {} labels: {} state change: {}'.format(
            count,
            frames[0].shape,
            labels.shape,
            state_change.shape
        ))

if __name__ == '__main__':
    args = parse_args()
    test_dataloader(load_config(args))
