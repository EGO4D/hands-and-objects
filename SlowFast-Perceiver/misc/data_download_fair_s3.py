"""
This file is used to download the data from fair's S3 bucket to IIIT's server
I will download the videos for which we have the annotations. I won't be
downloading the entire dataset. As it is approximately 4.5-5 TB
Currently using the code on Magnetar
"""

import os
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--download_folder',
    default='/home/ego4d/Remote/DataStorage/ego4d_ho_benchmark_data/',
    help='Path to the folder where the videos are to be saved'
)
parser.add_argument(
    '--cvdf_loc_file',
    default='video_component_locations_university_video_access.csv',
    help='Path to the file containing location of a video on CVDF'
)
parser.add_argument(
    '--fho_ann',
    default='fho_miniset_v2.json',
    help='Path to the file containing annotations for the FHO benchmark'
)
args = parser.parse_args()
print(args)

# For Magnetar
# download_command = 'aws s3 cp "{}" {} --profile ego4d_benchmark'
# For Drishti
download_command = 'aws s3 cp "{}" {}'
rename_command = 'mv {} {}'

annotations = json.load(open(args.fho_ann, 'r'))
video_ids = list(annotations.keys())
cvdf_videos_data = pd.read_csv(open(args.cvdf_loc_file, 'r'))

for video in video_ids:
    cvdf_data = cvdf_videos_data.query('video_uid=="{}"'.format(video))
    multiple_videos_list = dict()
    for item in cvdf_data.values:
        component_index = item[2]
        if 's3' in item[-1]:
            multiple_videos_list[component_index] = item[-1]
    for index, data in multiple_videos_list.items():
        if data == 'REDACTED':
            print('[INFO] Video {} redacted'.format(video))
        else:
            assert 's3' in data, 'Wrong path parsed. {}'.format(data)
            command = download_command.format(data, args.download_folder)
            video_name = os.path.join(
                args.download_folder,
                command.split('"')[1].split('/')[-1]
            )
            new_video_name = os.path.join(
                args.download_folder,
                video + '_{}'.format(index) + '.MP4'
            )
            if os.path.isfile(new_video_name):
                print('[INFO] {} exists...'.format(new_video_name))
                continue
            assert "mp4" in video_name.lower(), 'Wrong name parsed'
            rename_ = rename_command.format(
                video_name,
                new_video_name
            )
            os.system(command)
            os.system(rename_)
