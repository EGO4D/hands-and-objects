import os
import json
import time

import av
import cv2
from scipy.fftpack import cc_diff
import torch
import numpy as np
from tqdm import tqdm

from trim import _get_frames
from spatial_transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop, CenterCrop

class StateChangeDetectionAndKeyframeLocalisation_FB_annotations(torch.utils.data.Dataset):
    """
    Data loader for keyframe localisation using the canonical form of the
    videos
    """
    def __init__(self, mode, no_state_chng=True, transforms=Compose([Normalize([0.45],[0.225])]), debug_only=False, 
                    ann_path = '/home/sid/canonical_dataset/fho_pre_period_draft_updated_schema_tested.json',
                    videos_dir = '/mnt/nas/datasets/ego4d-release1/fho_canonical_videos_24-08-21',
                    split_path = '/home/sid/canonical_dataset/2021-08-09_provided_splits',
                    clips_save_path = '/mnt/hdd/datasets/ego4d-release1/fho_canonical_extracted_frames_27-08-21',
                    no_sc_clips_dir = '/mnt/hdd/datasets/ego4d-release1/fho_canonical_extracted_frames_negative_clips_27-08-21',
                    no_sc_split_path = '/home/sid/canonical_dataset/negative_clips_splits_json_2021-09-17.json',
                    test_json = '/home/sid/canonical_dataset/fixed_test_set_canonical_17-09-21.json',
                    val_json = '/home/sid/canonical_dataset/fixed_val_set_canonical_17-09-21.json',
                    no_sc_info_file = '/home/sid/canonical_dataset/negative-for-loop-faster_mode-test_2021-09-20.json',
                    ):
        assert mode in [
            'train',
            'val',
            'test'
        ], "Split `{}` not supported for Keyframe detection.".format(mode)
        self.mode = mode
        self.ann_path = ann_path
        assert os.path.exists(self.ann_path), "Wrong annotation path provided"
        self.videos_dir = videos_dir
        # assert os.path.exists(self.videos_dir), "Wrong video path provided"
        self.split_path = split_path
        assert os.path.exists(self.split_path), "Wrong split path provided"
        self.clips_save_path = clips_save_path
        self.SOFT_LABELS = True if not no_state_chng else False
        self.no_sc_info_file = no_sc_info_file
        self.transforms = transforms

        self.no_state_chng = no_state_chng
        self.debug_only = debug_only
        if self.no_state_chng:
            # assert os.path.exists(
            #     no_sc_clips_dir
            # ), "Wrong -ive clips path provided"
            self.no_sc_clips_dir = no_sc_clips_dir
            self.no_sc_split_path = no_sc_split_path
            # assert os.path.isfile(
            #     self.no_sc_split_path
            # ), "Wrong split path for -ive clips provided"
        # Loading the JSON file for a deterministic test set
        self.test_json = test_json
        if self.mode == 'test':
            if os.path.isfile(self.test_json):
                print('[INFO] Test JSON file exists...')
                self.test_det_data = json.load(
                    open(self.test_json, 'r')
                )
            else:
                print('[INFO] Creating the test JSON file...')
                self.test_det_data = dict()
        self.val_json = val_json
        if self.mode == 'val':
            if os.path.isfile(self.val_json):
                print('[INFO] Validation JSON file exists...')
                self.val_det_data = json.load(
                    open(self.val_json, 'r')
                )
            else:
                print('[INFO] Creating the validation JSON file...')
                self.val_det_data = dict()
        
        self._construct_loader()

    def _construct_loader(self):
        self.package = dict()
        uid_check = list()
        if self.mode == 'train':
            split_file = os.path.join(self.split_path, 'clips_train.json')
        elif self.mode == 'test':
            split_file = os.path.join(self.split_path, 'clips_test.json')
        else:
            split_file = os.path.join(self.split_path, 'clips_val.json')
        split_data = json.load(open(split_file, 'r'))
        selected_clip_ids = list()
        for data in split_data:
            selected_clip_ids.append(data['clip_id'])
        self.annotations = json.load(open(self.ann_path, 'r'))
        clip_count = 0
        dup_count = 0
        positive_count = 0
        miss_clip_count = 0
        self.ann_data = self.annotations['video_data']
        for video_id in tqdm(self.ann_data.keys(), desc='+ive clips'):
            if video_id in [
                'f8e5effa-02da-4d28-aa37-7d11e47882b2',
                '3770c680-82ce-481b-9637-28e7e15baabf'
            ]:
                continue
            intervals = self.ann_data[video_id]['annotated_intervals']
            for interval in intervals:
                interval_narrated_actions = interval['narrated_actions']
                if int(interval['clip_id']) not in selected_clip_ids:
                        continue
                for action in interval_narrated_actions:
                    # Selecting the clips based on the mode
                    if action['is_rejected'] is True or \
                        len(action['warnings']) != 0 or \
                            action['is_invalid_annotation'] is True or \
                                action['critical_frames'] is None:
                        continue
                    narration_uid = action['narration_annotation_uid']
                    clip_start_sec_str = str(
                        action['start_sec']
                    ).replace('.', '_')
                    clip_end_sec_str = str(
                        action['end_sec']
                    ).replace('.', '_')
                    new_uid = (f'{narration_uid}-{clip_start_sec_str}-'
                                        f'{clip_end_sec_str}')
                    # The following uids have video length less than
                    # the time provided in the annotations, due to
                    # which we are getting error when sampling the data
                    if new_uid not in [
                        'a4b0cdeb-7191-400f-9169-abc13ccfd026-3520_366666666667-3528_366666666667',
                        'a4b0cdeb-7191-400f-9169-abc13ccfd026-3523_4333333333334-3531_4333333333334',
                        '757bd945-7b78-4636-b9b1-45a44822e829-720_7333333333333-728_7333333333333',
                        'a4b0cdeb-7191-400f-9169-abc13ccfd026-3521_4-3529_4',
                        'a4b0cdeb-7191-400f-9169-abc13ccfd026-3519_5-3527_5',
                        '856e679d-1f17-4d6f-ab05-2826fb1b6f4c-400_43333333333334-408_43333333333334'
                    ]:
                        # As the canonical data has videos normalised to
                        # 30 fps, we can do this.
                        clip_start_sec = np.float32(
                            action['start_sec']
                        )
                        clip_end_sec = np.float32(
                            action['end_sec']
                        )
                        if clip_end_sec - clip_start_sec < 8:
                            # Ignoring such cases; There are 539 such cases
                            # as of 08/02/22
                            continue
                        clip_start_frame = action['start_frame']
                        clip_end_frame = action['end_frame']
                        pnr_frame = action['critical_frames']['pnr_frame']
                        assert clip_start_frame < pnr_frame
                        f_verb = action['freeform_verb']
                        s_verb = action['structured_verb']
                        # Check to make sure UID is not repeated
                        try:
                            # There are 323 such cases as of 05-02-22
                            # There are 330 such cases as of 08-02-22
                            assert new_uid not in uid_check
                        except:
                            dup_count += 1
                            continue
                        uid_check.append(new_uid)
                        assert clip_count not in self.package.keys()
                        assert np.isclose(
                            clip_end_frame - clip_start_frame,
                            30*8,
                            1
                        )
                        assert clip_end_frame > clip_start_frame
                        assert clip_end_frame >= pnr_frame >= \
                            clip_start_frame
                        self.package[clip_count] = {
                            "unique_id": new_uid,
                            "pnr_frame": pnr_frame,
                            "state": 1,
                            # "freeform_verb": f_verb,
                            # "structured_verb": s_verb,
                            "clip_start_sec": clip_start_sec,
                            "clip_end_sec": clip_end_sec,
                            "clip_start_frame": clip_start_frame,
                            "clip_end_frame": clip_end_frame,
                            "video_id": video_id,
                            "clip_id": interval['clip_id'],
                            "json_parent_start_sec": action['start_sec'],
                            "json_parent_end_sec": action['end_sec'],
                            "json_parent_start_frame": action['start_frame'],
                            "json_parent_end_frame": action['end_frame'],
                            "json_clip_start_sec": action['clip_start_sec'],
                            "json_clip_end_sec": action['clip_end_sec'],
                            "json_clip_start_frame": action['clip_start_frame'],
                            "json_clip_end_frame": action['clip_end_frame'],
                            "json_parent_pnr_frame": action['critical_frames']['pnr_frame'],
                            "json_clip_pnr_frame": action['clip_critical_frames']['pnr_frame'],
                            "json_clip_uid": interval['clip_uid'],
                        }
                        positive_count += 1
                        clip_count += 1
                        clip_path = os.path.join(
                            self.videos_dir,
                            f'{video_id}.mp4',
                        )
                        if os.path.isfile(clip_path):
                            pass
                        else:
                            miss_clip_count += 1
        negative_count = 0
        if self.no_state_chng:
            no_sc_clips_path = self.no_sc_clips_dir.format(self.mode)
            negative_clip_list = os.listdir(no_sc_clips_path)
            # no_sc_splits = json.load(open(self.no_sc_split_path, 'r'))
            # no_sc_video_ids = no_sc_splits[self.mode]
            # negative_clip_list = os.listdir(self.no_sc_clips_dir)
            # JSON file contining information to make the following for
            # loop faster
            no_sc_info_file = (f'/home/sid/canonical_dataset/negative-for-loop'
                f'-faster_mode-{self.mode}_2022-02-18.json')
            if os.path.isfile(no_sc_info_file):
                print(f'Using exisiting JSON from {no_sc_info_file} '
                    f'for faster loading...')
                faster = True
                no_sc_info_dict = json.load(open(no_sc_info_file, 'r'))
            else:
                print(f'Creating JSON and saving it to {no_sc_info_file} for '
                    f'faster loading...')
                faster = False
                no_sc_info_dict = dict()
            dropped_count = 0
            for video_id in tqdm(negative_clip_list, desc='-ive clips'):
                # dropped = [
                #     'e766ca0b-d6c6-46c3-8c08-3131e248725f',
                #     '92cf4ff7-4e76-4a50-ab82-e9ab6d241421',
                #     '8940b276-2e3c-4530-a8a6-576272b2be04',
                #     '1ccbd4d9-0af2-4717-bd2c-8bae1142fb7f',
                #     '6fb723ca-a3be-445b-b090-5a9d0e7959b7',
                #     '8a134ccf-d38c-4464-9b19-f05dba364cd6',
                #     'd66f42bb-822b-444a-bce0-ddd15b29bd1b',
                #     '9a7693ab-d9ed-4a6d-b653-766d4793ed7b',
                #     '0e404a7a-4714-40a3-82fc-401d91edf58d',
                #     '2bbcfb41-bb27-4d51-baf0-9a503d74d675',
                #     '2883c199-5d82-4ee7-a80d-1e9888c198fa',
                #     'c4024a00-3c1c-4383-9a0e-f870a388eabb',
                #     '367e82ae-8417-4ae9-8a23-f5e56ff9b41a',
                # ]
                # id = '-'.join(video_id.split('-')[:5])
                # if id in dropped:
                #     # Due to the issues with annotations some of the clips were
                #     # dropped. There are 249 such cases as of 05-02-22.
                #     dropped_count += 1
                #     continue
                if True:
                    assert video_id not in uid_check
                    if video_id.split('-')[-1] == '1884':
                        # Due to two videos with error: 'f8e5effa-02da-4d28-aa37-7d11e47882b2', '3770c680-82ce-481b-9637-28e7e15baabf'
                        continue
                    if faster:
                        clip_start_frame = no_sc_info_dict[video_id][
                            'start_frame'
                        ]
                        clip_end_frame = no_sc_info_dict[video_id]['end_frame']
                    else:
                        clip_path = os.path.join(
                            self.no_sc_clips_dir,
                            video_id
                        )
                        frames_itr = os.walk(clip_path)
                        frames = next(frames_itr)[-1]
                        frames_sorted = [
                            int(item.split('.')[0]) for item in frames
                        ]
                        clip_start_frame = np.min(frames_sorted)
                        clip_end_frame = np.max(frames_sorted)
                        no_sc_info_dict[video_id] = {
                            "start_frame": int(clip_start_frame),
                            "end_frame": int(clip_end_frame)
                        }
                    clip_start_sec = np.float32(clip_start_frame/30)
                    clip_end_sec = np.float32(clip_end_frame/30)
                    uid_check.append(video_id)
                    self.package[clip_count] = {
                        "unique_id": video_id,
                        "pnr_frame": "",
                        "state": 0,
                        "freeform_verb": "",
                        "structured_verb": "",
                        "clip_start_sec": clip_start_sec,
                        "clip_end_sec": clip_end_sec,
                        "clip_start_frame": clip_start_frame,
                        "clip_end_frame": clip_end_frame,
                        "video_id": video_id,
                        "clip_id": "",
                        "json_parent_start_sec": clip_start_sec,
                        "json_parent_end_sec": clip_end_sec,
                        "json_parent_start_frame": clip_start_frame,
                        "json_parent_end_frame": clip_end_frame,
                        "json_parent_pnr_frame": "",
                        "json_clip_uid": "",
                    }
                    clip_count += 1
                    negative_count += 1
            if not faster:
                json.dump(no_sc_info_dict, open(no_sc_info_file, 'w'))

        print(f'Number of clips for {self.mode}: {len(self.package)}')
        print(f'{positive_count} positive count; {negative_count} negative count')

    def __len__(self):
        return len(self.package)

    def __getitem__(self, index):
        info = self.package[index]
        if info['pnr_frame'] != "":
            self._extract_clip_frames(info)
            state = 1
        else:
            assert self.no_state_chng
            state = info['state']

        frames, labels, fps, candidate_frame_nums = self._sample_frames_gen_labels(info)
        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)#.div_(255.) not dividing 255 appears to get better performance, why?
        frames = self.transforms(frames)

        if self.mode == 'test':
            # Saving the test json file if it doesn't exists
            if index == len(self.package) - 1 and not os.path.isfile(
                self.test_json
            ):
                print('Saving the test JSON file...')
                json.dump(self.test_det_data, open(self.test_json, 'w'))

        if self.mode == 'val':
            # Saving the validation json file if it doesn't exists
            if index == len(self.package) - 1 and not os.path.isfile(
                self.val_json
            ):
                print('Saving the validation JSON file...')
                json.dump(self.val_det_data, open(self.val_json, 'w'))
        info["candidate_frame_nums"] = candidate_frame_nums
        return frames, labels, state, fps, info

    def _extract_clip_frames(self, info):
        """
        This method is used to extract and save frames for all the 8 seconds
        clips. If the frames are already saved, it does nothing.
        """
        clip_start_frame = info['clip_start_frame']
        clip_end_frame = info['clip_end_frame']
        unique_id = info['unique_id']
        video_path = os.path.join(
            self.videos_dir,
            info['video_id']
        )
        clip_save_path = os.path.join(self.clips_save_path, unique_id)
        # We can do do this fps for canonical data is 30.
        num_frames_per_video = 2 * 8
        if os.path.isdir(clip_save_path):
            # The frames for this clip are already saved.
            num_frames = len(os.listdir(clip_save_path))
            if num_frames < 240:
                print(
                    f'Deleting {clip_save_path} as it has {num_frames} frames'
                )
                os.system(f'rm -r {clip_save_path}')
            else:
                return None
        print(f'Saving frames for {clip_save_path}...')
        os.makedirs(clip_save_path)
        start = time.time()
        # We need to save the frames for this clip.
        frames_list = [
            i for i in range(clip_start_frame, clip_end_frame + 1, 1)
        ]
        try:
            assert np.isclose(len(frames_list), num_frames_per_video, 1)
        except AssertionError:
            # Cases when clip length is less than 8 seconds
            assert np.isclose(
                len(frames_list),
                (clip_end_frame - clip_start_frame) * 30,
                1
            )
        frames = self.get_frames_for(
            video_path,
            frames_list,
        )
        desired_shorter_side = 384
        num_saved_frames = 0
        for frame, frame_count in zip(frames, frames_list):
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
            num_saved_frames += 1
        print(f'Time taken: {time.time() - start}; {num_saved_frames} '
            f'frames saved; {clip_save_path}')
        return None

    def _sample_frames(
        self,
        unique_id,
        clip_start_frame,
        clip_end_frame,
        num_frames_required,
        pnr_frame
    ):
        num_frames = clip_end_frame - clip_start_frame
        if num_frames < num_frames_required:
            print(f'Issue: {unique_id}; {num_frames}; {num_frames_required}')
        error_message = "Can\'t sample more frames than there are in the video"
        assert num_frames >= num_frames_required, error_message
        lower_lim = np.floor(num_frames/num_frames_required)
        upper_lim = np.ceil(num_frames/num_frames_required)
        lower_frames = list()
        upper_frames = list()
        lower_keyframe_candidates_list = list()
        upper_keyframe_candidates_list = list()
        for frame_count in range(clip_start_frame, clip_end_frame, 1):
            if frame_count % lower_lim == 0:
                lower_frames.append(frame_count)
                if pnr_frame is not None:
                    lower_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    lower_keyframe_candidates_list.append(0.0)
            if frame_count % upper_lim == 0:
                upper_frames.append(frame_count)
                if pnr_frame is not None:
                    upper_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    upper_keyframe_candidates_list.append(0.0)
        if len(upper_frames) < num_frames_required:
            return (
                lower_frames[:num_frames_required],
                lower_keyframe_candidates_list[:num_frames_required]
            )
        return (
            upper_frames[:num_frames_required],
            upper_keyframe_candidates_list[:num_frames_required]
        )

    def _load_frame(self, frame_path):
        """
        This method is used to read a frame and do some pre-processing.

        Args:
            frame_path (str): Path to the frame
        
        Returns:
            frames (ndarray): Image as a numpy array
        """
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(
            224,
            224
        ))
        frame = np.expand_dims(frame, axis=0).astype(np.float32)
        return frame

    def _sample_frames_gen_labels(self, info):
        if info['pnr_frame'] is not None:
            clip_path = os.path.join(
                self.clips_save_path,
                info['unique_id']
            )
        else:
            # Clip path for clips with no state change
            clip_path = os.path.join(
                self.no_sc_clips_dir.format(self.mode),
                info['unique_id']
            )
            # clip_path = os.path.join(
            #     self.no_sc_clips_dir,
            #     info['unique_id']
            # )
        message = f'Clip path {clip_path} does not exists...'
        assert os.path.isdir(clip_path), message
        num_frames_per_video = (
            2 * 8
        )
        # Random clipping
        # Randomly choosing the duration of clip (between 5-8 seconds)
        random_length_seconds = np.random.uniform(5, 8)
        random_start_seconds = info['clip_start_sec'] + np.random.uniform(
            8 - random_length_seconds
        )
        random_start_frame = np.floor(
            random_start_seconds * 30
        ).astype(np.int32)
        random_end_seconds = random_start_seconds + random_length_seconds
        if random_end_seconds > info['clip_end_sec']:
            random_end_seconds = info['clip_end_sec']
        random_end_frame = np.floor(random_end_seconds * 30).astype(np.int32)
        pnr_frame = info['pnr_frame']
        if pnr_frame is not None:
            keyframe_after_end = pnr_frame > random_end_frame
            keyframe_before_start = pnr_frame < random_start_frame
            if keyframe_after_end:
                random_end_frame = info['clip_end_frame']
            if keyframe_before_start:
                random_start_frame = info['clip_start_frame']

        if self.mode == 'test':
            # If the JSON does not exsist, we will need to create the data
            if not os.path.isfile(self.test_json):
                print(f'{self.mode} file length: {len(self.test_det_data)}')
                # Saving the id to the frames after random clippsing
                assert info['unique_id'] not in self.test_det_data.keys()
                self.test_det_data[info['unique_id']] = {
                    'clip_end_frame': int(random_end_frame),
                    'clip_start_frame': int(random_start_frame)
                }
            # If it exists, load the data from it
            else:
                random_start_frame = self.test_det_data[info['unique_id']][
                    'clip_start_frame'
                ]
                random_end_frame = self.test_det_data[info['unique_id']][
                    'clip_end_frame'
                ]
        if self.mode == 'val':
            # if the JSON does not exits, we will need to create the data
            if not os.path.isfile(self.val_json):
                print(f'{self.mode} file length: {len(self.val_det_data)}')
                assert info['unique_id'] not in self.val_det_data.keys()
                self.val_det_data[info['unique_id']] = {
                    'clip_end_frame': int(random_end_frame),
                    'clip_start_frame': int(random_start_frame)
                }
            # If it exists, load the data from it
            else:
                random_start_frame = self.val_det_data[info['unique_id']][
                    'clip_start_frame'
                ]
                random_end_frame = self.val_det_data[info['unique_id']][
                    'clip_end_frame'
                ]

        if pnr_frame is not None:
            message = (f'Random start frame {random_start_frame} Random end '
                f'frame {random_end_frame} info {info} clip path {clip_path}')
            assert random_start_frame <= pnr_frame <= random_end_frame, message
        else:
            message = (f'Random start frame {random_start_frame} Random end '
                f'frame {random_end_frame} info {info} clip path {clip_path}')
            assert random_start_frame < random_end_frame, message

        candidate_frame_nums, keyframe_candidates_list = self._sample_frames(
            info['unique_id'],
            random_start_frame,
            random_end_frame,
            num_frames_per_video,
            pnr_frame
        )
        frames = list()
        for frame_num in candidate_frame_nums:
            frame_path = os.path.join(clip_path, f'{frame_num}.jpeg')
            message = f'{frame_path}; {candidate_frame_nums}'
            assert os.path.isfile(frame_path), message
            frames.append(self._load_frame(frame_path))
        if pnr_frame is not None:
            keyframe_location = np.argmin(keyframe_candidates_list)
            hard_labels = np.zeros(len(candidate_frame_nums))
            hard_labels[keyframe_location] = 1
            labels = hard_labels
        else:
            labels = keyframe_candidates_list
        # Calculating the effective fps. In other words, the fps after sampling
        # changes when we are randomly clipping and varying the duration of the
        # clip
        final_clip_length = (random_end_frame/30) - (random_start_frame/30)
        effective_fps = num_frames_per_video / final_clip_length
        return np.concatenate(frames), np.array(labels), effective_fps, candidate_frame_nums

    def get_frames_for(self, video_path, frames_list):
        """
        Code for decoding the video
        """
        frames = list()
        with av.open(video_path) as container:
            for frame in _get_frames(
                frames_list,
                container,
                include_audio=False,
                audio_buffer_frames=0
            ):
                frame = frame.to_rgb().to_ndarray()
                frames.append(frame)
        return frames
