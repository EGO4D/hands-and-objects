"""
This file is used for downloading data from FAIR's s3 bucket which does not
include a state change
"""

import os
import cv2
import pdb
import json
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--manifest_path',
    default='annotations/video_component_locations_university_video_access.csv',
    help='Path to the manifest file'
)
parser.add_argument(
    '--fho_ann',
    default='annotations/fho_miniset_v2.json',
    help='Path to the FHO annotation file'
)
parser.add_argument(
    '--num_videos',
    default=1000,
    type=int,
    help='Number of 8 seconds clips to download and save'
)
parser.add_argument(
    '--to_save',
    default='/home/ego4d/ego4d_benchmark/data/no_state_change_clips/',
    help='Path to save clipped videos\'s frames'
)
args = parser.parse_args()
print(args)

fho_annotation = json.load(open(args.fho_ann, 'r'))
manifest = pd.read_csv(open(args.manifest_path, 'r'))
fho_unq_video_ids = list(fho_annotation.keys())
manifest_unq_ids = list(manifest['video_uid'])

np.random.seed(0)

# List of ids not used in FHO annotations
usable_ids = list()
for id in manifest_unq_ids:
    if id not in fho_unq_video_ids:
        usable_ids.append(id)
print('[INFO] Number of usable ids: {}'.format(len(usable_ids)))

def download(video_location):
    """
    This method is used to download the video from FAIR's S3 bucket
    """
    download_command = 'aws s3 cp "{}" {}'
    original_video_path = os.path.join(
        args.to_save,
        video_location.split('/')[-1]
    )
    if not os.path.isfile(original_video_path):
        os.system(download_command.format(
            video_location,
            args.to_save
        ))
    return original_video_path

def save_clip(selected_id, original_video_path, downloaded):
    """
    This method is used to save frames from 8 second clips from random videos
    """
    clip_save_path = os.path.join(
        args.to_save,
        selected_id
    )
    videocap = cv2.VideoCapture(original_video_path)
    fps = videocap.get(cv2.CAP_PROP_FPS)
    if fps != 30:
        print('[INFO] Skipping video as its frame rate is {} and '
        'not equal to 30'.format(fps))
        return False
    frame_count = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    # Making sure we do not process a clip with duration less than 8 seconds
    if duration <= 8:
        print('[INFO] Skipping video as its duration is less than 8 seconds')
        return False
    os.makedirs(clip_save_path)
    clip_start_sec = np.random.randint(0, duration - 9)
    clip_end_sec = clip_start_sec + 8
    start_frame_count = np.round(clip_start_sec * fps)
    end_frame_count = np.round(clip_end_sec * fps)
    save_path = os.path.join(clip_save_path, '{}_{}.jpg')
    parent_frame_count = 0
    saved_frames_count = 0
    while videocap.isOpened():
        success, frame = videocap.read()
        if not success:
            break
        else:
            parent_frame_count += 1
        condition = parent_frame_count >= start_frame_count and \
            parent_frame_count <= end_frame_count
        if condition:
            cv2.imwrite(
                save_path.format(
                    parent_frame_count, str(fps)
                ),
                frame
            )
            saved_frames_count += 1
        elif parent_frame_count > end_frame_count:
            break
    videocap.release()
    print('[INFO] {} frames saved at {} fps!'.format(
        saved_frames_count,
        fps
    ))
    return True

downloaded = 0
while downloaded < args.num_videos:
    random_id_index = np.random.randint(0, len(usable_ids))
    selected_id = usable_ids[random_id_index]
    print('[INF0] Selected id: {}'.format(selected_id))
    manifest_data = manifest.query('video_uid=="{}"'.format(selected_id)).values
    for data in manifest_data:
        start_sec = data[3]
        end_sec = data[4]
        location = data[-1]
        if '.mp4' not in location.lower():
            continue
        clip_save_path = os.path.join(
            args.to_save,
            selected_id
        )
        if os.path.isdir(clip_save_path):
            # Frames with video id already saved
            file_count = os.listdir(clip_save_path)
            if file_count != 0:
                print('[INFO] Video with {} id already exists! Skipping...'.format(
                    clip_save_path
                ))
                break
            else:
                print('[INFO] Empty video folder with id {} exists...'.format(
                    clip_save_path
                ))
                pass
        print('[INFO] Downloading {}...'.format(location))
        original_video_path = download(location)
        print('[INFO] Extracting frames...')
        saved = save_clip(selected_id, original_video_path, downloaded)
        if saved:
            downloaded += 1
            # Deleating the saved video to save space
            os.system('rm {}'.format(original_video_path))
            # Saving only one clip from a single id
            break
