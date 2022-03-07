
import torch
import numpy as np
import pandas as pd
from datasets.Ego4DKeyframeLocalisation import Ego4DKeyframeLocalisation
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.optim as optim
from utils.parser import parse_args, load_config
import pdb
from utils.BMN_utils import ioa_with_anchors, iou_with_anchors
from bmn_models import BMN
from loss_function import bmn_loss_func, get_mask
from models.video_model_builder import DualHeadResNet, ActionProposal
import os
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
writer = SummaryWriter(log_dir='../TB_logs_BMN/jun5v2')
cfg = load_config(parse_args())

def train_BMN(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    device = torch.device('cuda:2')
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0

    for n_iter, data in tqdm.tqdm(enumerate(data_loader)):
        clip_uid, frames, labels, prec_labels, state_change_label, label_confidence_score, label_match_score_start, label_match_score_end = data
        input_data = [frames[0].to(device)]
        label_start = label_match_score_start.to(device)
        label_end = label_match_score_end.to(device)
        label_confidence = label_confidence_score.to(device)
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.to(device))
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
        writer.add_scalar("TEM_Loss/train", epoch_tem_loss / (n_iter + 1), epoch)
        writer.add_scalar("PEM_CLR_Loss/train", epoch_pemclr_loss / (n_iter + 1), epoch)
        writer.add_scalar("PEM_REG_Loss/train", epoch_pemreg_loss / (n_iter + 1), epoch)
        writer.add_scalar("Loss/train", epoch_loss / (n_iter + 1), epoch)
    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))

def test_BMN(data_loader, model, epoch, bm_mask):
    device = torch.device('cuda:2')
    model.eval()
    best_loss = 1e10
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, data in tqdm.tqdm(enumerate(data_loader)):
        clip_uid, frames, labels, prec_labels, state_change_label, label_confidence_score, label_match_score_start, label_match_score_end = data
        input_data = [frames[0].to(device)]
        label_start = label_match_score_start.to(device)
        label_end = label_match_score_end.to(device)
        label_confidence = label_confidence_score.to(device)
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.to(device))

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
        writer.add_scalar("TEM_Loss/val", epoch_tem_loss / (n_iter + 1), epoch)
        writer.add_scalar("PEM_CLR_Loss/val", epoch_pemclr_loss / (n_iter + 1), epoch)
        writer.add_scalar("PEM_REG_Loss/val", epoch_pemreg_loss / (n_iter + 1), epoch)
        writer.add_scalar("Loss/val", epoch_loss / (n_iter + 1), epoch)
    print(
        "BMN test loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))
    state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict()}
    torch.save(state, cfg.TRAIN.CHECKPOINT + "/BMN_checkpoint_{}.pth.tar".format(epoch))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(state, cfg.TRAIN.CHECKPOINT + "/BMN_best.pth.tar")

def BMN_Train(cfg):
    device = torch.device('cuda:2')
    model = ActionProposal(cfg)
    if cfg.TRAIN.INIT_PRETRAIN_FEATURE == True:
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT + "/BMN_best.pth.tar", map_location = torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, #opt["training_lr"],
                           weight_decay=1e-4 #opt["weight_decay"]
                           )
    train_dataset = Ego4DKeyframeLocalisation(cfg, 'train')
    val_dataset = Ego4DKeyframeLocalisation(cfg, 'val')
    val_dataset_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    bm_mask = get_mask(cfg.BMN.TEMPORAL_SCALE)
    for epoch in tqdm.tqdm(range(0, cfg.TRAIN.EPOCH)):
        train_BMN(train_dataset_loader, model, optimizer, epoch, bm_mask)
        writer.flush()
        test_BMN(val_dataset_loader, model, epoch, bm_mask)
        writer.flush()
        scheduler.step()

def BMN_inference(cfg):
    device = torch.device('cuda:2') 
    model = ActionProposal(cfg).to(device)
    checkpoint = torch.load(cfg.TRAIN.CHECKPOINT + "/BMN_checkpoint_14.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_dataset = Ego4DKeyframeLocalisation(cfg, 'train')
    test_dataset_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )
    tscale = cfg.BMN.TEMPORAL_SCALE
    with torch.no_grad():
        new_props = []
        only_keyF = []
        for lll, data in tqdm.tqdm(enumerate(test_dataset_loader)):
            clip_uid, frames, labels, prec_labels, state_change_label, label_confidence_score, label_match_score_start, label_match_score_end, keyframe_time, prec_frame_time = data
            input_data = [frames[0].to(device)]
            label_start = label_match_score_start.to(device)
            label_end = label_match_score_end.to(device)
            label_confidence = label_confidence_score.to(device)
            confidence_map, start, end = model(input_data)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            for idx  in range(tscale):
                end_index = idx+1
                if end_index<tscale:
                    xmax = end_index / tscale
                    xmax_score = end_scores[end_index]
                    only_keyF.append([clip_uid[0], xmax, xmax_score, keyframe_time.item()])
            # 遍历起始分界点与结束分界点的组合
            #new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and  end_index<tscale :
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([clip_uid[0], xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score, keyframe_time.item(), prec_frame_time.item()])
        new_props = np.stack(new_props)
        only_keyF = np.stack(only_keyF)
            #########################################################################

        col_name = ["clip_uid","xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_score", "score", "Keyframe_time", "PreC_time"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df.to_csv(os.path.join(cfg.DATA.OUTPUT_DIR, "{}.csv".format("Train_results")), index=False)

        only_keyF_cols = ["clip_uid", "xmax", "xmax_score", "Keyframe_time"]
        new_df = pd.DataFrame(only_keyF, columns=only_keyF_cols)
        new_df.to_csv(os.path.join(cfg.DATA.OUTPUT_DIR, "{}.csv".format("Train_results_only_keyF")), index=False)

def main(cfg):
    #if cfg.MODEL.MODE == "Train":
    #    BMN_Train(cfg)
    #elif cfg.MODEL.MODE == "inference":
    #    if not os.path.exists(cfg.DATA.OUTPUT_DIR):
    #        os.makedirs(cfg.DATA.OUTPUT_DIR)
    BMN_inference(cfg)
pdb.set_trace()
if __name__ == '__main__':
    main(cfg)
    writer.close()
