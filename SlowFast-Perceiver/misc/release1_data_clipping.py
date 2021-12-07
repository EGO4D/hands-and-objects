
import os
import json
import argparse
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm
from numpy import isclose
from moviepy.editor import VideoFileClip


parser = argparse.ArgumentParser()
parser.add_argument(
    '--raw_videos',
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
parser.add_argument(
    '--clips_location',
    default='/mnt/nas/datasets/ego4d-release1/fho_extracted_clips_26-07-21',
    help='Path to the folder where the clips are to be saved'
)
parser.add_argument(
    '--n_procs',
    default=10,
    type=int,
    help='Number of parallel processes to create for extracting the clips'
)
args = parser.parse_args()
print(args)


def get_video_duration(clip_path):
    clip = VideoFileClip(clip_path)
    total_seconds = clip.duration
    print(f'Video duration: {total_seconds}')
    return total_seconds


def check_duration(args, delete=False):
    video_files = os.listdir(args.clips_location)
    print(f'{len(video_files)} videos present...')
    duration_list = list()
    offending_videos = list()
    outstanding_videos = list()
    for video in video_files:
        video_path = os.path.join(args.clips_location, video)
        try:
            video_duration = get_video_duration(video_path)
        except:
            # Case when the extraced video is corrupted
            if delete:
                print(f'Deleting {video_path}...')
                os.system(f'rm {video_path}')
            else:
                print(f'Issue with video {video_path}...')
        if video_duration < 150:
            offending_videos.append(video)
        if video_duration > 300:
            outstanding_videos.append(video)
        duration_list.append(video_duration)
    return None


def generate_parallel_data(args):
    """
    This method is used to create list the list containing all the information
    required to clip the videos parallely.
    """
    data_info = json.load(open(args.data_info_json, 'r'))
    cvdf_videos_data = pd.read_csv(open(args.cvdf_loc_file, 'r'))
    clips_count = 0
    split_1_count = 0
    split_2_count = 0
    normal_count = 0
    parallel_info = list()

    for data in tqdm(data_info, desc='Generating parallel_info'):
        video_id = data['video_uid']
        clips_duration = data['clips']
        cvdf_data = cvdf_videos_data.query('video_uid=="{}"'.format(video_id))
        clips_count += len(clips_duration)
        for clip_count, clip_start_end in enumerate(clips_duration):
            clip_start_sec = round(clip_start_end[0], 2)
            clip_end_sec = round(clip_start_end[1], 2)
            for item in cvdf_data.values:
                component_index = item[2]
                parent_start_sec = round(item[3], 2)
                parent_end_sec = round(item[4], 2)
                component_type = item[6]
                video_location = item[5]
                if 'REDACT' in video_location:
                    continue
                elif component_type == 'mp4_video':
                    video_extension = video_location.split('.')[-1]
                    raw_video_name = os.path.join(args.raw_videos, video_id + \
                        f"_{component_index}.{video_extension}")
                    clip_condition = (clip_start_sec >= parent_start_sec) and \
                        (clip_end_sec <= parent_end_sec)
                    if clip_condition:
                        clip_name = os.path.join(
                            args.clips_location,
                            video_id + f"_component_index_{component_index}_cl"
                            f"ip_num_{clip_count}.{video_extension}"
                        )
                        normal_count += 1
                        # Clip available
                    elif clip_start_sec >= parent_start_sec and clip_start_sec\
                        <= parent_end_sec and clip_end_sec > parent_end_sec:
                        # Starting part for the clip is in this video,
                        # but the end is in the next video
                        clip_name = os.path.join(
                            args.clips_location,
                            video_id + f"_component_index_{component_index}_cl"
                            f"ip_num_{clip_count}_part_one_0.{video_extension}"
                        )
                        split_1_count += 1
                    elif clip_end_sec >= parent_start_sec and clip_end_sec <= \
                        parent_end_sec and clip_start_sec < parent_start_sec:
                        # End part of this clip is in this video,
                        # but the start part was in the previous video
                        clip_name = os.path.join(
                            args.clips_location,
                            video_id + f"_component_index_{component_index}_cl"
                            f"ip_num_{clip_count}_part_two_1.{video_extension}"
                        )
                        split_2_count += 1
                    else:
                        # Clip unavailable in current video
                        clip_name = None
                    parallel_info.append({
                        'clip_name': clip_name,
                        'clip_end_sec': clip_end_sec,
                        'clip_start_sec': clip_start_sec,
                        'parent_start_sec': parent_start_sec,
                        'parent_end_sec': parent_end_sec,
                        'raw_video_name': raw_video_name,
                    })
    print(
        f'Total no. of clips: {normal_count + split_1_count + split_2_count} '
        f'Normal clips count: {normal_count} split 1 clips count: '
        f'{split_1_count} split 2 clips count {split_2_count}'
    )
    return parallel_info


def create_batches(parallel_info, num_batches=10):
    """
    This method is used to create batches for parallel processing

    Args:
        parallel_info (list): list containing information required to extract
        the clips

    Returns:
        parallel_info_batches (list): list containing batches with length equal
        to num_batches
    """
    print('Creating batches...')
    total_num_videos = len(parallel_info)
    parallel_info_batches = list()
    num_videos_per_batch = total_num_videos // num_batches
    # Remaining video after uniformly dividing the videos in batches
    # These many videos will be added to the last batch
    remaining_video_num = total_num_videos % num_batches
    total_videos_check = num_videos_per_batch * num_batches +\
        remaining_video_num
    assert total_num_videos == total_videos_check
    for count in range(num_batches):
        if count == num_batches - 1:
            parallel_info_batch = parallel_info[count * num_videos_per_batch :\
                ((count + 1) * num_videos_per_batch) + remaining_video_num]
        else:
            parallel_info_batch = parallel_info[count * num_videos_per_batch :\
                (count + 1) * num_videos_per_batch]
        parallel_info_batches.append(parallel_info_batch)
    check_count = 0
    for parallel_info_batch in parallel_info_batches:
        check_count += len(parallel_info_batch)
    error_message = (f'check_count: {check_count};'
                        f' total_videos: {total_num_videos}')
    assert check_count == total_num_videos, error_message
    return parallel_info_batches


def clip_videos(parallel_info_batch):
    """
    This method is used to clip the videos. This method can be used as a single
    parallel process.

    Args:
        parallel_info_batch (list): list containing information required to
        extract the clips

    Returns:
        None
    """
    for info in parallel_info_batch:
        clip_name = info['clip_name']
        clip_end_sec = info['clip_end_sec']
        clip_start_sec = info['clip_start_sec']
        parent_start_sec = info['parent_start_sec']
        parent_end_sec = info['parent_end_sec']
        raw_video_name = info['raw_video_name']
        if clip_name is not None:
            if os.path.isfile(clip_name):
                print(f'{clip_name} exists...')
            else:
                clip_len = clip_end_sec - clip_start_sec
                modified_start_sec = clip_start_sec - parent_start_sec
                modified_end_sec = clip_len + modified_start_sec
                if 'two_1' in clip_name:
                    modified_start_sec = 0
                if 'one_0' in clip_name:
                    modified_end_sec = parent_end_sec - clip_start_sec +\
                        modified_start_sec
                if modified_start_sec < 0:
                    breakpoint()
                clip = VideoFileClip(raw_video_name)
                clip_ = clip.subclip(modified_start_sec, modified_end_sec)
                clip_.write_videofile(clip_name)
                assert os.path.isfile(clip_name)
                extracted_duration = get_video_duration(clip_name)
                assert isclose(
                    extracted_duration, modified_end_sec - modified_start_sec,
                    1
                )
    return None


def parallel_clipping(args):
    """
    This method is used to parallely clip the data for extracting 5 min clips
    """
    pool = mp.Pool(processes=args.n_procs)
    # Getting the list of videos to process
    parallel_info = generate_parallel_data(args)
    # Get the batches
    parallel_info_batches = create_batches(parallel_info)
    # Submit jobs
    results = [
        pool.apply_async(clip_videos, [parallel_info_batch])\
            for parallel_info_batch in parallel_info_batches
    ]
    for proc in results:
        _ = proc.get()
    return None


if __name__ == "__main__":
    parallel_clipping(args)
