"""
This file generate negative clips apart from the negative clips provided by FB
"""

import os
import json
import time
import argparse

import av
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from utils.trim import _get_frames
from misc.no_state_change_data_prep_v2 import possible_locations
from misc.no_state_change_data_prep_canonical import create_batches


parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    default=('/home/sid/canonical_dataset/fho_pre_period_draft_updated_schema_tested.json'),
    help='Path to the json containing annotations'
)
parser.add_argument(
    '-d',
    default='/mnt/hdd/datasets/ego4d-release1/fho_canonical_extracted_frames_negative_clips_15-09-21',
    help='Path to the directory for storing the negative clips'
)
parser.add_argument(
    '-s',
    default='/mnt/nas/datasets/ego4d-release1/fho_canonical_videos_24-08-21',
    help='Path to the directory containing canonical videos'
)
parser.add_argument(
    '--split_path',
    default='/home/sid/canonical_dataset/2021-08-09_provided_splits',
    help='Path to the folder containing split files'
)
parser.add_argument(
    '--mode',
    required=True,
    help='Which -ive clips to generate. Options are: train, test, val'
)
parser.add_argument(
    '--manifest',
    default='/home/sid/canonical_dataset/manifest.csv',
    help='Path to the CSV containing canonical videos information'
)

def get_video_duration(manifest_info, video_id):
    occurances = manifest_info.query(f'video_uid=="{video_id}"').values
    return occurances[:, 3].item()


def get_frames_for(video_path, frames_list):
    with av.open(video_path) as container:
        for frame in _get_frames(frames_list, container, include_audio=False, audio_buffer_frames=0):
            frame = frame.to_rgb().to_ndarray()
            yield frame


def get_keyframe_info(args):
    if args.mode == 'train':
        split_file = os.path.join(args.split_path, 'clips_train.json')
    elif args.mode == 'test':
        split_file = os.path.join(args.split_path, 'clips_test.json')
    else:
        split_file = os.path.join(args.split_path, 'clips_val.json')
    split_data = json.load(open(split_file, 'r'))
    selected_clip_ids = list()
    for data in tqdm(split_data, desc='Processing Splits'):
        selected_clip_ids.append(data['clip_id'])

    manifest_info = pd.read_csv(args.manifest)

    ann = json.load(open(args.p, 'r'))
    keyframe_track_dict = dict()
    for video_id in tqdm(ann.keys(), desc='Video ID'):
        intervals = ann[video_id]['annotated_intervals']
        video_duration = get_video_duration(manifest_info, video_id)
        for interval in intervals:
            interval_narrated_actions = interval['narrated_actions']
            for action in interval_narrated_actions:
                if action['is_rejected'] is True or action['error'] is \
                        not None or action['is_invalid_annotation'] is True:
                    continue
                clip_id = action['clip_id']
                if clip_id in selected_clip_ids:
                    pnr_frame = action['critical_frames']['pnr_frame']
                    pnr_sec = np.float32(pnr_frame/30)
                    if clip_id not in keyframe_track_dict.keys():
                        keyframe_track_dict[clip_id] = {
                            "keyframes_list": [pnr_sec],
                            "video_duration": video_duration,
                            "orig_video_id": video_id,
                        }
                    else:
                        assert 'video_duration' in keyframe_track_dict[clip_id].keys()
                        keyframe_track_dict[clip_id]['keyframes_list'].append(pnr_sec)
    return keyframe_track_dict


def get_trim_info(args, keyframe_track_dict):
    # Get information according to the mode in which we want to work
    # (train, test, and val)
    info_dict = dict()
    for video_id in tqdm(keyframe_track_dict.keys(), desc='Info. Dict'):
        keyframe_locations = keyframe_track_dict[video_id]['keyframes_list']
        video_duration = keyframe_track_dict[video_id]['video_duration']
        orig_video_id = keyframe_track_dict[video_id]['orig_video_id']
        trim_locations_ = possible_locations(
            keyframe_locations=keyframe_locations,
            duration=video_duration,
            clip_duration=8,
            slack=1,
            verbose=False
        )
        # Maximum number of negative clips to select from a single video.
        # Decided based on the number of extra clips required.
        if args.mode == 'train':
            thresh = 30
        elif args.mode == 'test':
            thresh = 29
        else:
            thresh = 29
        if len(trim_locations_) > thresh:
            trim_locations = trim_locations_[:thresh]
        else:
            trim_locations = trim_locations_
        for location in trim_locations:
            start_sec_str = str(location[0]).replace('.', '_')
            end_sec_str = str(location[1]).replace('.', '_')
            unique_id = f'{video_id}-{start_sec_str}-{end_sec_str}'
            assert unique_id not in info_dict.keys()
            info_dict[unique_id] = {
                "trim_location": location,
                "unique_id": unique_id,
                "video_id": orig_video_id
            }
    print(f'Extracting {len(info_dict)} clips for mode {args.mode}!')
    return info_dict


def save_frames(info, args):
    # We can simply multiply by 30 as the data is in canonical form
    clip_start_frame = int(info['trim_location'][0]*30)
    clip_end_frame = int(info['trim_location'][1]*30)
    clip_save_path = os.path.join(args.d, info['unique_id'])
    assert 240 <= (clip_end_frame - clip_start_frame) <= 241
    if os.path.isdir(clip_save_path):
        print(f'{clip_save_path} exists...')
        num_frames = len(os.listdir(clip_save_path))    
        if num_frames < 16:
            print(
                f'Deleting {clip_save_path} as it has {num_frames} frames'
            )
            os.system(f'rm -r {clip_save_path}')
        else:
            return None
    print(f'Saving frames for {clip_save_path}')
    video_path = os.path.join(args.s, info['video_id'])
    os.makedirs(clip_save_path)
    start = time.time()
    frames_list = [
        i for i in range(clip_start_frame, clip_end_frame + 1, 1)
    ]
    try:
        assert np.isclose(len(frames_list), 241, 1)
    except AssertionError:
        assert np.isclose(
            len(frames_list),
            (clip_end_frame - clip_start_frame) * 30,
            1
        )
    frames_iterator = get_frames_for(video_path, frames_list)
    desired_shorter_side = 384
    for frame, frame_count in zip(frames_iterator, frames_list):
        original_height, original_width, _ = frame.shape
        if original_height < original_width:
            # Height is the shorter side
            new_height = desired_shorter_side
            new_width = np.round(
                original_width*(desired_shorter_side/original_height)
            ).astype(np.int32)
        elif original_height > original_width:
            # Width is the shorter side
            new_width = desired_shorter_side
            new_height = np.round(
                original_height*(desired_shorter_side/original_width)
            ).astype(np.int32)
        else:
            # Both are the same
            new_height = desired_shorter_side
            new_width = desired_shorter_side
        assert np.isclose(
            new_width/new_height,
            original_width/original_height,
            0.01
        )
        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(
            os.path.join(
                clip_save_path,
                f'{frame_count}.jpeg'
            ),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )
    print(f'Time taken: {time.time() - start}')
    return None


def process_batch(info_dict, batch, args):
    for video_id in batch:
        info = info_dict[video_id]
        save_frames(info, args)


def main(args):
    keyframe_track_dict = get_keyframe_info(args)
    info_dict = get_trim_info(args, keyframe_track_dict)
    batches = create_batches(info_dict)
    # Using multi-processing
    pool = mp.Pool(processes=36)
    multi_result = [pool.apply_async(
        process_batch,(info_dict, batch, args)
    ) for batch in batches]
    for proc in multi_result:
        _ = proc.get()
    # Without multi-processing
    # for batch in batches:
    #     process_batch(info_dict, batch, args)

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'val'], "Options are: train, test, and val"
    main(args)
