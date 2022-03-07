import argparse
from trainval import train, val
from evaluate import evaluate
import torch
from simple_cnn_lstm import cnnlstm
from dataset import CanonicalKeyframeLocalisation_v2
import torch.nn as nn
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser()

# general 
parser.add_argument('--task', type=str, default='PNR', choices=['PNR', 'State_change'])
parser.add_argument('--save_name', type=str, default='cnn_lstm_PNR.pth')
parser.add_argument('--mode', type=str, default='trainval', choices=['trainval', 'test'])

# dataset 
parser.add_argument('--ann_path', type=str, default='/home/sid/canonical_dataset/fho_pre_period_draft_updated_schema_tested.json')
parser.add_argument('--videos_dir', type=str, default='/mnt/nas/datasets/ego4d-release1/fho_canonical_videos_24-08-21')
parser.add_argument('--split_path', type=str, default='/home/sid/canonical_dataset/2021-08-09_provided_splits')
parser.add_argument('--clips_save_path', type=str, default='/mnt/hdd/datasets/ego4d-release1/fho_canonical_extracted_frames_27-08-21')
parser.add_argument('--no_sc_clips_dir', type=str, default='/mnt/hdd/datasets/ego4d-release1/fho_canonical_extracted_frames_negative_clips_27-08-21')
parser.add_argument('--no_sc_split_path', type=str, default='/home/sid/canonical_dataset/negative_clips_splits_json_2021-09-17.json')
parser.add_argument('--val_json', type=str, default='/home/sid/canonical_dataset/fixed_val_set_canonical_17-09-21.json')
parser.add_argument('--test_json', type=str, default='/home/sid/canonical_dataset/fixed_test_set_canonical_17-09-21.json')
parser.add_argument('--no_sc_info_file', type=str, default='/home/sid/canonical_dataset/negative-for-loop-faster_mode-%s_2021-09-20.json')

# model & training
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=1)

args = parser.parse_args()
if args.task == 'PNR':
    args.no_state_chng = False
    criterion = nn.BCELoss().cuda()
else:
    args.no_state_chng = True
    criterion = nn.CrossEntropyLoss().cuda()

model = cnnlstm(hidden_size=args.hidden_size, num_layers=args.num_layers, state=args.no_state_chng)
if args.mode == 'test':
    state_dict = torch.load(args.save_name, map_location='cpu')
    model.load_state_dict(state_dict)
    test_dataset = CanonicalKeyframeLocalisation_v2('test', no_state_chng=args.no_state_chng, 
                ann_path=args.ann_path, videos_dir=args.videos_dir, split_path=args.split_path,
                clips_save_path=args.clips_save_path, no_sc_clips_dir=args.no_sc_clips_dir, 
                no_sc_split_path=args.no_sc_split_path, val_json=args.val_json, test_json=args.test_json, no_sc_info_file=args.no_sc_info_file)
    test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, num_workers=8, shuffle=False)
else:
    train_dataset = CanonicalKeyframeLocalisation_v2('train', no_state_chng=args.no_state_chng, 
                ann_path=args.ann_path, videos_dir=args.videos_dir, split_path=args.split_path,
                clips_save_path=args.clips_save_path, no_sc_clips_dir=args.no_sc_clips_dir, 
                no_sc_split_path=args.no_sc_split_path, val_json=args.val_json, test_json=args.test_json, no_sc_info_file=args.no_sc_info_file)
    val_dataset = CanonicalKeyframeLocalisation_v2('val', no_state_chng=args.no_state_chng, 
                ann_path=args.ann_path, videos_dir=args.videos_dir, split_path=args.split_path,
                clips_save_path=args.clips_save_path, no_sc_clips_dir=args.no_sc_clips_dir, 
                no_sc_split_path=args.no_sc_split_path, val_json=args.val_json, test_json=args.test_json, no_sc_info_file=args.no_sc_info_file)
    train_loader = DataLoader(train_dataset, batch_size=4, pin_memory=True, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=8)
model.cuda()

if args.mode == 'trainval':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = 99999
    best_epoch = 0
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, epoch, state=(args.no_state_chng))
        
        loss, _, _ = val(model, val_loader, criterion, epoch, state=(args.no_state_chng))
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_name)
            print('best model at epoch %d' % best_epoch)

else:
    loss, ys, outs, fps_list = evaluate(model, test_loader, criterion, 0, state=(args.no_state_chng))