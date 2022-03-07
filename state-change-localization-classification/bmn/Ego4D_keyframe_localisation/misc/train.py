"""
This file is used to test various functions created for the keyframe
localisation benchmark
"""

import os
import pdb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from configs.defaults import get_cfg_defaults
from datasets.Ego4DKeyframeLocalisation import Ego4DKeyframeLocalisation
from torch.utils.data import DataLoader
from models.video_model_builder import ResNet

def train(config_file):
    # Loading and updating the configuration
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()
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
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )
    # Checking if GPU is available
    if torch.cuda.is_available():
        if cfg.MISC.VERBOSE:
            print('[INFO] Using GPU...', flush=True)
        device = 'cuda'
    else:
        if cfg.MISC.VERBOSE:
            print('[INFO] Using CPU...', flush=True)
        device = 'cpu'
    # Loading the model
    model = ResNet(cfg).to(device)
    model.train()
    if os.path.isfile(cfg.MISC.CHECKPOINT_FILE_PATH):
        if cfg.MISC.VERBOSE:
            print('[INFO] Loading from previous checkpoint {}...'.format(
                cfg.MISC.CHECKPOINT_FILE_PATH
            ), flush=True)
        model.load_state_dict(torch.load(cfg.MISC.CHECKPOINT_FILE_PATH))
    else:
        if cfg.MISC.VERBOSE:
            print('[INFO] Training the model from scratch...', flush=True)
    # Creating the optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    # Creating the loss function
    if cfg.DATA.TASK == 'frame_detection_classification':
        criterion = nn.CrossEntropyLoss()
    elif cfg.DATA.TASK == 'frame_detection_regression':
        # criterion = nn.MSELoss()
        criterion = nn.HingeEmbeddingLoss()
    elif cfg.DATA.TASK == 'frame_detection_multiclass_classification':
        criterion = nn.CrossEntropyLoss()

    loss_list = list()
    save_loss = np.inf
    # Starting the train loop
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        for data in tqdm(train_dataset_loader, disable=True):
            frames, labels = data
            for pathway in range(len(frames)):
                frames[pathway] = frames[pathway].to(device).float()
            labels = labels.to(device).float()
            # Zeroing the parameter gradients
            optimizer.zero_grad()
            if cfg.DATA.TASK == 'frame_detection_classification':
                labels = labels.to(torch.int64)
                output = model(frames.copy())
                loss = criterion(output, labels.squeeze())
                if 1 not in torch.argmax(output, dim=1):
                    loss += 100
            elif cfg.DATA.TASK == 'frame_detection_regression':
                # Getting the output from the network
                output = model(frames.copy()).T
                # Calculating the loss
                loss = criterion(output, labels)
                # Penalizing more as the model tends to predict 1 as output everytime
                if 1 in output:
                    loss += 100
            elif cfg.DATA.TASK == 'frame_detection_multiclass_classification':
                output = model(frames.copy()).T
                labels = torch.argmax(labels)
                labels = labels.unsqueeze(dim=0)
                loss = criterion(output, labels)
                # pdb.set_trace()
            # Backpropagating
            loss.backward()
            # Updating the weights
            optimizer.step()
            loss_list.append(loss.item())
            if cfg.DATA.TASK == 'frame_detection_regression':
                print("Loss: {}\nLabels: {}\nOutput: {}".format(
                    loss.item(),
                    torch.argmax(labels).item(),
                    torch.argmax(output).item()
                ), flush=True)
            elif cfg.DATA.TASK == 'frame_detection_classification':
                print("Loss: {}\nLabels: {}\nOutput: {}\nOutput: {}".format(
                    loss.item(),
                    labels,
                    torch.argmax(output, dim=1),
                    output.T
                ), flush=True)
            elif cfg.DATA.TASK == 'frame_detection_multiclass_classification':
                print("Loss: {}\nLabels: {}\nOutput: {}".format(
                    loss.item(),
                    labels.item(),
                    torch.argmax(output).item()
                ), flush=True)
    # Starting validation loop
    if cfg.MISC.VERBOSE:
        print('[INFO] Starting the validation loop...', flush=True)
    model.eval()
    for data in val_dataset_loader:
        frames, labels = data
        for pathway in range(len(frames)):
            frames[pathway] = frames[pathway].to(device).float()
        labels = labels.to(device).float()
        output = model(frames.copy()).T
        print("Labels: {}\nOutput: {}".format(
            torch.argmax(labels).item(),
            torch.argmax(output).item()
        ), flush=True)
    print(loss_list, flush=True)


if __name__ == '__main__':
    train('configs/ego4d_kf_loc_test.yaml')
