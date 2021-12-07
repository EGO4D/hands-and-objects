
import os
import json
import time
import argparse

import av
import cv2
import numpy as np
from utils.trim import _get_frames
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    default=('/home/sid/canonical_dataset/fho_crtical_frame_draft_export.json'),
    help='Path to the json containing annotations'
)
parser.add_argument(
    '-d',
    default='/mnt/hdd/datasets/ego4d-release1/fho_canonical_extraced_frames_negative_clips_27-08-21',
    help='Path to the directory for storing the negative clips'
)
parser.add_argument(
    '-s',
    default='/mnt/nas/datasets/ego4d-release1/fho_canonical_videos_24-08-21',
    help='Path to the directory containing canonical videos'
)


def get_trim_info(ann):
    info_dict = dict()
    for video_id in ann.keys():
        intervals = ann[video_id]['annotated_intervals']
        for interval in intervals:
            interval_narrated_actions = interval['narrated_actions']
            for action in interval_narrated_actions:
                if action['is_rejected'] is True:
                # if 'is_rejected' in action.keys():
                    # reason = action['reject_reason']
                    # if reason == '"no_human-object_interaction_from_camera_wearer_(e.g._c_looks_around,_c_pauses_work)"':
                        # As fps is 30 for the canonical dataset.
                    start_sec = action['start_sec']
                    end_sec = action['end_sec']
                    start_frame = np.floor(start_sec * 30)
                    end_frame = np.floor(end_sec * 30)
                    uid = f"{video_id}-{str(start_sec).replace('.', '_')}-{str(end_sec).replace('.', '_')}"
                    assert uid not in info_dict.keys()
                    info_dict[uid] = {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "video_id": video_id,
                        "unique_id": uid
                    }
    return info_dict


def save_frames(info, args):
    clip_start_frame = info['start_frame'].astype(np.int32)
    clip_end_frame = info['end_frame'].astype(np.int32)
    clip_save_path = os.path.join(args.d, info['unique_id'])
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


def get_frames_for(video_path, frames_list):
    with av.open(video_path) as container:
        for frame in _get_frames(frames_list, container, include_audio=False, audio_buffer_frames=0):
            frame = frame.to_rgb().to_ndarray()
            yield frame


def create_batches(info_dict):
    num_clips = len(info_dict)
    batch_size = num_clips // 36
    last_batch = num_clips % 36
    video_ids = list(info_dict.keys())
    batches = list()
    for count, i in enumerate(range(0, num_clips, batch_size)):
        if count == 36:
            break
        if count == 35:
            batches.append(video_ids[i:i+last_batch+batch_size])
        else:
            batches.append(video_ids[i:i+batch_size])
    return batches


def process_batch(info_dict, batch, args):
    for video_id in batch:
        info = info_dict[video_id]
        save_frames(info, args)


def main(args):
    ann = json.load(open(args.p, 'r'))
    info_dict = get_trim_info(ann)
    batches = create_batches(info_dict)
    # Using multi-processing
    pool = mp.Pool(processes=10)
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
    main(args)
