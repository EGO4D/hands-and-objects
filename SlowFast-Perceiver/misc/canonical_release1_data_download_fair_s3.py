"""
This fils contains the code for downloading the first release canonical videos 
data from FAIR's S3 bucket to CMU's servers.
"""


import os
import json
import argparse

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    '--download_folder',
    default='/mnt/nas/datasets/ego4d-release1/fho_canonical_videos_24-08-21/',
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
    default=('/home/sid/canonical_dataset/manifest.csv'),
    help='Path to the CSV file containing location of the videos on S3'
)
args = parser.parse_args()
print(args)

download_command = 'aws s3 cp "{}" {}'
rename_command = 'mv {} {}'

data_info = json.load(open(args.data_info_json, 'r'))
cvdf_videos_data = pd.read_csv(open(args.cvdf_loc_file, 'r'))
len_list = list()

for count, data in enumerate(data_info):
    video_id = data['video_uid']
    clips_duration = data['clips']
    src = data['video_source']
    cvdf_data = cvdf_videos_data.query('video_uid=="{}"'.format(video_id))
    multiple_videos_list = dict()
    len_list.append(len(cvdf_data.values))
    for item in cvdf_data.values:
        video_location = item[15]
        assert 'canonical' in video_location
        download_command_ = download_command.format(video_location, args.download_folder)
        os.system(download_command_)
