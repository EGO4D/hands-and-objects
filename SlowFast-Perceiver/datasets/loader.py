"""
Data Loader
"""

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from .build_dataset import build_dataset

def frame_selection_collate(batch):
    frames, labels, state, fps, lengths = zip(*batch)
    frames = pad_sequence(frames, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    return frames, labels, state, fps, lengths

def construct_loader(cfg, split):
    """
    Construct the data loader for the given dataset
    """
    assert split in [
        'train',
        'val',
        'test'
    ], "Split `{}` not supported".format(split)

    if split == 'train':
        dataset_name = cfg.TRAIN.DATASET
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = cfg.DATA_LOADER.SHUFFLE
        drop_last = cfg.DATA_LOADER.DROP_LAST
    elif split == 'val':
        dataset_name = cfg.TRAIN.DATASET
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = False
        drop_last = False
    elif split == 'test':
        dataset_name = cfg.TEST.DATASET
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    if cfg.SOLVER.ACCELERATOR == 'dp':
        sampler = None  # As we are using 'dp' as our accelerator
    else:
        raise NotImplementedError("{} not implemented".format(
            cfg.SOLVER.ACCELERATOR
        ))
    if cfg.TRAIN.DATASET == "FrameSelectionDataset":

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn = frame_selection_collate,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
        )

    return loader
