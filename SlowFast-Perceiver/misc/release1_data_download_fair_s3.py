"""
This fils contains the code for downloading the first release data from FAIR's 
S3 bucket to CMU's servers.
"""


import os
import json
import argparse

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    '--download_folder',
    default='/mnt/nas/datasets/ego4d-release1/fho_raw_videos/',
    help='Path to the folder where the videos are to be saved'
)
parser.add_argument(
    '--data_info_json',
    default='/mnt/nas/datasets/ego4d-miniset/fho_release1_target_data.json',
    help='Path to the JSON file containing information about the videos which'
    'are included in the first release'
)
parser.add_argument(
    '--cvdf_loc_file',
    default=('/mnt/nas/datasets/ego4d-miniset/consortium-sharing/dataset_manif'
    'ests/video_component_locations_university_video_access_release1.csv'),
    help='Path to the CSV file containing location of the videos on S3'
)
args = parser.parse_args()
print(args)

download_command = 'aws s3 cp "{}" {}'
rename_command = 'mv {} {}'

data_info = json.load(open(args.data_info_json, 'r'))
cvdf_videos_data = pd.read_csv(open(args.cvdf_loc_file, 'r'))

for data in data_info:
    video_id = data['video_uid']
    clips_duration = data['clips']
    src = data['video_source']
    cvdf_data = cvdf_videos_data.query('video_uid=="{}"'.format(video_id))
    multiple_videos_list = dict()
    for item in cvdf_data.values:
        component_index = item[2]
        parent_start_sec = item[3]
        parent_end_sec = item[4]
        video_location = item[5]
        component_type = item[6]
        if "REDACT" in video_location:
            print(f'[INFO] {video_id} with index {component_index} redacted')
        elif component_type == 'mp4_video':
            new_video_name = os.path.join(
                args.download_folder,
                video_id + f"_{component_index}.{video_location.split('.')[-1]}"
            )
            if os.path.isfile(new_video_name):
                print(f'[INFO] {new_video_name} exists...')
                continue
            download = download_command.format(
                video_location,
                args.download_folder
            )
            video_name = os.path.join(
                args.download_folder,
                download.split('"')[1].split('/')[-1]
            )
            rename_ = rename_command.format(video_name, new_video_name)
            os.system(download)
            os.system(rename_)
