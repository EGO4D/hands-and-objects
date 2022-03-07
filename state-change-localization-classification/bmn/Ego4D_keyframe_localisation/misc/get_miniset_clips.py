"""
This file is used to copy data from FAIR folder to a different folder
"""
import os

source = '/home/ego4d/FAIR/videos/'
dest = '/home/ego4d/benchmark_miniset_videos/'

video_ids = ['0199_007_086_002_006_025_001_4', '0415_007_016_001_006_072_002',
             '0202_007_086_002_006_025_001', '0449_004_008_001_006_053_001',
             '0183_001_018_001_006_000_001', '0198_007_086_002_006_025_001_5']

for video_folder in video_ids:
    print(video_folder)
    copy_path = os.path.join(source, video_folder)
    dest_path = os.path.join(dest, video_folder)
    command = 'cp -r {} {}'.format(copy_path, dest_path)
    # command = 'rsync -zP {} siddhant.b@gnode04:{}'
    os.system(command)
    # pdb.set_trace()
