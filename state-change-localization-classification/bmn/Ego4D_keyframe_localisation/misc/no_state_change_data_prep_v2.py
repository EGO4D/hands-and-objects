"""
This file is used to extract no-state change clips from the annotated videos
itself
"""

import os
import cv2
import pdb
import json
import sys
import time
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from utils.parser import parse_args, load_config

def get_video_names(cfg, id):
    videos = os.listdir(cfg.DATA.VIDEO_DIR_PATH)
    selected_names = list()
    for video in videos:
        if id in video:
            selected_names.append(video)
    selected_names = sorted(
        selected_names,
        key=lambda a: a.split('_')[-1].split('.')[0]
    )
    return [
        os.path.join(cfg.DATA.VIDEO_DIR_PATH, name) for name in selected_names
    ]

def get_locations(annotation, frame='key'):
    """
    This method returns location of specified frame
    """
    assert frame in ['key', 'start', 'end']
    key = {
        'key': 'pnr_frame_sec',
        'start': 'parent_start_sec',
        'end': 'parent_end_sec'
    }
    locations = list()
    for ann in annotation:
        locations.append(ann[key[frame]])
    return sorted(locations)

def possible_locations(
    keyframe_locations,
    duration,
    clip_duration=8,
    slack=1,
    verbose=False
):
    locations = list()
    for count, kf_location in enumerate(keyframe_locations):
        if verbose:
            print('Processing keyframe: {} with count {}'.format(
                kf_location,
                count
            ))
        start_location = None
        end_location = None
        if kf_location < 8:
            if verbose:
                print('Skipping as keyframe is smaller than 8 seconds')
            pass
        elif kf_location >= 8:
            if verbose:
                print('Keyframe is larger than 8 seconds')
            if (count - 1 >= 0) and (count + 1 < len(keyframe_locations)):
                if verbose:
                    print("There are keyframes on left and right side")
                previous_keyframe = keyframe_locations[count - 1]
                next_keyframe = keyframe_locations[count + 1]
                if kf_location - clip_duration - slack > previous_keyframe:
                    if verbose:
                        print("Previous keyframe is not in the 8 second range")
                    if kf_location - clip_duration > 0:
                        if verbose:
                            print('Clip is within 0 seconds')
                        start_location = kf_location - clip_duration - slack
                        end_location = start_location + clip_duration
                    else:
                        if verbose:
                            print('Clip is going less than 0 seconds')
                else:
                    if verbose:
                        print("previous keyframe is in the 8 seconds range")
            else:
                if verbose:
                    print("Keyframe is not there on left or right side")
                if count + 1 >= len(keyframe_locations):
                    if verbose:
                        print("Keyframe is not there on right side. Using "
                                "duration...")
                    # Use duration
                    start_location = kf_location + slack
                    if duration > start_location + clip_duration:
                        if verbose:
                            print('Enough space in right side for generating '
                                    'clip')
                        end_location = start_location + clip_duration
                    else:
                        if verbose:
                            print('Not enough space in right side to generate'
                                    ' clip')
                elif count - 1 < 0:
                    if verbose:
                        print("Keyframe is not there on the left side")
                    pass
                else:
                    raise Exception('TRAHIMAAM...')
        if start_location is not None and end_location is not None:
            locations.append((start_location, end_location))
    return locations

def crop(cfg, videos, location, count):
    videocap = cv2.VideoCapture(videos[0])
    fps = int(videocap.get(cv2.CAP_PROP_FPS))
    videocap.release()
    start_location = location[0]
    end_location = location[1]
    start_frame_count = np.round(start_location *  fps)
    end_frame_count = np.round(end_location *  fps)
    parent_frame_count = 0
    check = 0
    for video in videos:
        clip_folder = '{}/{}_{}/'.format(
                        cfg.DATA.NO_SC_PATH,
                        video.split('/')[-1].split('.')[0],
                        count
                    )
        if os.path.isdir(clip_folder):
            if len(os.listdir(clip_folder)) > 0:
                print('[INFO] Folder exists, skipping 1...')
                return None
        videocap = cv2.VideoCapture(video)
        while videocap.isOpened():
            success, frame = videocap.read()
            if not success:
                break
            else:
                parent_frame_count += 1
            condition = parent_frame_count >= start_frame_count and \
                parent_frame_count <= end_frame_count
            if condition:
                if check == 0:
                    # We we are processing this video for the first time
                    clip_folder = '{}/{}_{}/'.format(
                        cfg.DATA.NO_SC_PATH,
                        video.split('/')[-1].split('.')[0],
                        count
                    )
                    if os.path.isdir(clip_folder):
                        if len(os.listdir(clip_folder)) > 0:
                            print('[INFO] Folder exists, skipping 2...')
                            return None
                        else:
                            pass
                    else:
                        os.mkdir(clip_folder)
                clip_path = clip_folder + '/{}_{}.jpg'.format(
                    parent_frame_count,
                    str(fps)
                )
                cv2.imwrite(
                    clip_path.format(
                        cfg.DATA.NO_SC_PATH,
                        video.split('/')[-1].split('.')[0],
                        count,
                        parent_frame_count,
                        str(fps)
                    ),
                    frame
                )
                check += 1
    assert np.isclose(check, 8*fps, 1)
    print('[INFO] Cropped...')

def process(cfg, id, complete_annotations):
    start = time.time()
    print('[INFO] Processing ID: {}'.format(id))
    data = complete_annotations[id]
    duration = complete_annotations[id]['duration_sec']
    annotation = data['annotations'][
        'forecast_hand_object_frame_selection_4'
    ]
    videos = get_video_names(cfg, id)
    # Get possible places for getting clips
    keyframe_locations = get_locations(annotation, frame='key')
    # Search for those places in the videos
    locations = possible_locations(keyframe_locations, duration, slack=0.3)
    # Clip them
    for count, location in enumerate(locations):
        crop(cfg, videos, location, count)
    end = time.time() - start
    print('[INFO] Done processing ID: {} time taken: {}'.format(
        id,
        np.round(end, 3)
    ))

def main(cfg):
    complete_annotations = json.load(open(cfg.DATA.ANN_PATH, 'r'))
    video_ids = list(complete_annotations.keys())
    print('[INFO] Number of CPU cores available: {}'.format(mp.cpu_count()))
    pool = mp.Pool(processes=mp.cpu_count())
    pool.starmap(
        process,
        [(cfg, id, complete_annotations) for id in video_ids[-300:]]
    )


if __name__ == '__main__':
    main(load_config(parse_args()))
