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
        key=lambda a: int(a.split('_')[-1].split('.')[0])
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

def more_locations_rhs(
    duration,
    orig_end_location,
    next_keyframe_loc=None,
    stride=1,
    clip_duration=8
):
    rhs_locations = list()
    if next_keyframe_loc is None:
        next_keyframe_loc = duration
    for i in range(
        int(np.floor(orig_end_location)),
        int(np.floor(next_keyframe_loc)),
        stride
    ):
        start_location = i
        end_location = None
        if start_location + clip_duration < next_keyframe_loc:
            end_location = start_location + clip_duration
        if start_location is not None and end_location is not None:
            rhs_locations.append((start_location, end_location))
    return rhs_locations

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
                        # We can extract more frames towards LHS of the start
                        # and end location
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
                        # Extracting more clips towards RHS of the
                        # start and end location
                        if len(keyframe_locations) - 1 == count:
                            next_keyframe_loc = None
                        else:
                            next_keyframe_loc = keyframe_locations[count]
                        locations.extend(more_locations_rhs(
                            duration,
                            end_location,
                            next_keyframe_loc,
                            clip_duration=clip_duration,
                            stride=3
                        ))
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
    start = time.time()
    ind_frame_count = dict()

    previous_video_frame_count = 0
    for video in videos:
        tempcap = cv2.VideoCapture(video)
        ind_frame_count[video] = int(tempcap.get(cv2.CAP_PROP_FRAME_COUNT)) + previous_video_frame_count
        previous_video_frame_count = ind_frame_count[video]
        fps = int(tempcap.get(cv2.CAP_PROP_FPS))
        tempcap.release()

    previous_video_frame_count_ = 0
    for video in videos:
        print(f'processing {video}')
        clip_folder = '{}/{}_{}/'.format(
                        cfg.DATA.NO_SC_PATH,
                        video.split('/')[-1].split('.')[0],
                        count
                    )
        if os.path.isdir(clip_folder):
            if len(os.listdir(clip_folder)) > 0:
                print(f'[INFO] {clip_folder} exists, skipping 1 time taken '
                        f'{np.round(time.time() - start, 3)}...')
                return None

        start_location = location[0]
        end_location = location[1]
        start_frame_count = np.round(start_location *  fps)
        end_frame_count = np.round(end_location *  fps)

        video_end_frame = ind_frame_count[video]
        # If the following condition is true than the video clip is not in this
        # video, it is in a later video
        if start_frame_count > video_end_frame:
            print(f"{video} length {video_end_frame} smaller than "
                    f"{start_frame_count}, time taken "
                    f"{np.round(time.time() - start, 3)}...")
            previous_video_frame_count_ = video_end_frame
            continue
        else:
            print(f"{video} length {video_end_frame} greater than "
                    f"{start_frame_count} looping over it...")

        parent_frame_count = previous_video_frame_count_
        check = 0

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
                            print(f'[INFO] {clip_folder} exists, skipping 2...')
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
                if np.isclose(check, 8*fps+1, atol=1):
                    break
        previous_video_frame_count_ = video_end_frame
        videocap.release()
    time_taken = np.round(time.time() - start, 3)
    try:
        assert np.isclose(check, 8*fps, 1)
        print(f'[INFO] Cropped {videos} to generate {check} frames in {time_taken} secs...')
    except Exception as e:
        with open('no_sc_error_log.txt', 'a') as file:
            file.write("\nFrames found in non of the videos!\n")
            file.write(f"Error:\n{e}\nVideos:\n{videos}\nLocation: {location}"
                        f"\nCount: {count}")
            file.close()
        print('Error. Added to the log file no_sc_error_log.txt ...')


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
    locations = possible_locations(keyframe_locations, duration, slack=0)
    # Checking the extractions
    # main_video_name = videos[0].split('/')[-1].split('_')[0]
    # result, actual_count = check(main_video_name, len(locations), cfg)
    # if result is True:
    #     return actual_count
    # Clip them
    for count, location in enumerate(locations):
        crop(cfg, videos, location, count)
    end = time.time() - start
    print('[INFO] Done processing ID: {} time taken: {}'.format(
        id,
        np.round(end, 3)
    ))
    # return len(locations)


def check(main_video_name, location_count, cfg):
    videos_dir = cfg.DATA.NO_SC_PATH
    check_command = 'ls -1 {} | grep {} | wc -l'
    actual = int(
        os.popen(check_command.format(videos_dir, main_video_name)).read()
    )
    if location_count > actual:
        return False, actual
    return True, actual


def main(cfg):
    complete_annotations = json.load(open(cfg.DATA.ANN_PATH, 'r'))
    video_ids = list(complete_annotations.keys())
    print('[INFO] Number of CPU cores available: {}'.format(mp.cpu_count()))
    pool = mp.Pool(processes=25)
    # location_count = 0
    # for id in video_ids:
    #     location_count += process(cfg, id, complete_annotations)
    # breakpoint()
    multi_result = [pool.apply_async(
        process, (cfg, id, complete_annotations)
    ) for id in video_ids]
    for proc in multi_result:
        _ = proc.get()


if __name__ == '__main__':
    main(load_config(parse_args()))
