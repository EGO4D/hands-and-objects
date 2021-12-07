"""
This file is used to understand the annotations provided for the FHO benchmark
"""
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-f',
    '--file',
    default='/Volumes/Storage/Egocentric/EGO_4D/Data/Annotations/'
    'fho_miniset_v1.json',
    help='Path to the file containing annotations'
)
args = parser.parse_args()

file = open(args.file, 'r')
data = json.load(file)
print('[INFO] Total number of keys: {}'.format(len(data.keys())))
video_unique_ids = data.keys()

count = 0
main_count = 0
iiith_id = list()
for video_id in video_unique_ids:
    duration = data[video_id]['duration_sec']
    source = data[video_id]['video_source']
    original_id = data[video_id]['origin_video_id']
    annotations = data[video_id]['annotations']
    scenario = data[video_id]['scenarios']
    if source == 'iiith':
        iiith_id.append(original_id)
        print(original_id)
    # for annotation in annotations.values():
    #     for ann in annotation:
    #         main_count += 1
    #         try:
    #             start_sec = ann['parent_start_sec']
    #             end_sec = ann['parent_end_sec']
    #             contact_frame_sec = ann['contact_frame_sec']
    #             pre_frame_sec = ann['pre_condition_frame_sec']
    #             pnr_frame_sec = ann['pnr_frame_sec']
    #             post_frame_sec = ann['post_condition_frame_sec']
    #             narration = ann['narration']
    #         except Exception as e:
    #             count += 1
    #             print(e, source)
print(iiith_id, len(iiith_id))
