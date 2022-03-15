import argparse
from trainval import train, val
from evaluate import evaluate, generate_submission_file, generate_submission_file_cls
import torch
from simple_cnn_lstm import cnnlstm
from dataset import StateChangeDetectionAndKeyframeLocalisation_FB_annotations
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
import json
parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='PNR', choices=['PNR', 'State_change'])
parser.add_argument('--save_name', type=str, default='cnn_lstm.pth')
parser.add_argument('--mode', type=str, default='trainval', choices=['trainval', 'test'])
parser.add_argument('--pretrained_model', type=str, default=None)

# dataset 
parser.add_argument('--ann_path', type=str, default='/mnt/nas/datasets/ego4d-launch/fho_220208.json')
parser.add_argument('--videos_dir', type=str, default='/mnt/nas/datasets/ego4d_data/v1/full_scale')
parser.add_argument('--split_path', type=str, default='/home/sid/canonical_dataset/2021-08-09_provided_splits')
parser.add_argument('--clips_save_path', type=str, default='/media/hdd2/datasets/ego4d/fho_canonical_extracted_frames_final_release_12-02-22')
parser.add_argument('--no_sc_clips_dir', type=str, default='/media/hdd2/datasets/ego4d/fho_canonical_extracted_frames_final_release_negative_{}_clips_12-02-22')
parser.add_argument('--no_sc_split_path', type=str, default='/home/sid/canonical_dataset/negative_clips_splits_json_2021-09-17.json')
parser.add_argument('--val_json', type=str, default='/home/sid/canonical_dataset/fixed_val_set_canonical_final-release_18-02-22.json')
parser.add_argument('--test_json', type=str, default='/home/sid/canonical_dataset/fixed_test_set_canonical_final-release_18-02-22.json')
parser.add_argument('--no_sc_info_file', type=str, default='/home/sid/canonical_dataset/negative-for-loop-faster_mode-%s_2021-09-20.json')

args = parser.parse_args()
if args.task == 'PNR':
    args.no_state_chng = False
    criterion = nn.BCELoss().cuda()
    args.save_name = 'PNR_' + args.save_name
else:
    args.no_state_chng = True
    criterion = nn.CrossEntropyLoss().cuda()
    args.save_name = 'State_change_' + args.save_name

model = cnnlstm(state=(args.no_state_chng))
if args.mode == 'test':
    if args.pretrained_model is not None:
        pretrained = torch.load(args.pretrained_model, map_location='cpu')
        model.load_state_dict(pretrained)
        model.cuda()
    else:
        state_dict = torch.load(args.save_name, map_location='cpu')
        model.load_state_dict(state_dict)
    test_dataset = StateChangeDetectionAndKeyframeLocalisation_FB_annotations('test', no_state_chng=args.no_state_chng, 
                ann_path=args.ann_path, videos_dir=args.videos_dir, split_path=args.split_path,
                clips_save_path=args.clips_save_path, no_sc_clips_dir=args.no_sc_clips_dir, 
                no_sc_split_path=args.no_sc_split_path, val_json=args.val_json, test_json=args.test_json, no_sc_info_file=args.no_sc_info_file)
    test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False)
    val_dataset = StateChangeDetectionAndKeyframeLocalisation_FB_annotations('val', no_state_chng=args.no_state_chng, 
                ann_path=args.ann_path, videos_dir=args.videos_dir, split_path=args.split_path,
                clips_save_path=args.clips_save_path, no_sc_clips_dir=args.no_sc_clips_dir, 
                no_sc_split_path=args.no_sc_split_path, val_json=args.val_json, test_json=args.test_json, no_sc_info_file=args.no_sc_info_file)
    val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=4)
else:
    train_dataset = StateChangeDetectionAndKeyframeLocalisation_FB_annotations('train', no_state_chng=args.no_state_chng, 
                ann_path=args.ann_path, videos_dir=args.videos_dir, split_path=args.split_path,
                clips_save_path=args.clips_save_path, no_sc_clips_dir=args.no_sc_clips_dir, 
                no_sc_split_path=args.no_sc_split_path, val_json=args.val_json, test_json=args.test_json, no_sc_info_file=args.no_sc_info_file)
    val_dataset = StateChangeDetectionAndKeyframeLocalisation_FB_annotations('val', no_state_chng=args.no_state_chng, 
                ann_path=args.ann_path, videos_dir=args.videos_dir, split_path=args.split_path,
                clips_save_path=args.clips_save_path, no_sc_clips_dir=args.no_sc_clips_dir, 
                no_sc_split_path=args.no_sc_split_path, val_json=args.val_json, test_json=args.test_json, no_sc_info_file=args.no_sc_info_file)
    train_loader = DataLoader(train_dataset, batch_size=4, pin_memory=True, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=4)
    print('train loader len:', len(train_loader))
    print('test loader len:', len(val_loader))
    if args.pretrained_model is not None:
        pretrained = torch.load(args.pretrained_model, map_location='cpu')
        model.load_state_dict(pretrained)
        model.cuda()
model.cuda()




if args.mode == 'trainval':
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best_loss = 99999
    best_epoch = 0
    for epoch in range(10):
        train(model, train_loader, optimizer, criterion, epoch, state=(args.no_state_chng))
        # torch.save(model.state_dict(), 'epoch_%d_'%epoch+args.save_name)
        
        loss, _, _ = val(model, val_loader, criterion, epoch, state=(args.no_state_chng))
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_name)
            print('best model at epoch %d' % best_epoch)

else:
    # uncomment for getting results on the validation set.
    acc, outs, uid_list = evaluate(model, val_loader, criterion, 0, state=(args.no_state_chng), is_validation=True)
    print('finished evaluation on the validation set!')
    acc, outs, uid_list = evaluate(model, test_loader, criterion, 0, state=(args.no_state_chng))
    generate_submission = generate_submission_file if args.task == 'PNR' else generate_submission_file_cls
    submission_data = generate_submission(outs, uid_list)
    with open(args.task+'submission.json', 'w') as f:
        json.dump(submission_data, f)
    print('finished generating the sumbission file!')
    