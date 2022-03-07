import os
import json
import time

import av
import cv2
import torch
import numpy as np
from tqdm import tqdm

from trim import _get_frames
from spatial_transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop, CenterCrop


# @DATASET_REGISTRY.register()
class CanonicalKeyframeLocalisation_v2(torch.utils.data.Dataset):
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
        self.CLIPS_SAVE_PATH = clips_save_path
        self.SOFT_LABELS = True if not no_state_chng else False
        self.no_sc_info_file = no_sc_info_file
        self.transforms = transforms
        # self.state_mapping = {
        #     'activate_-_[object_is_transformed_to_enable_access_to_one_or_more_objects_inside_(e.g.,_open),_or_enables_an_object’s_function_(e.g.,_turn_power_on)]': 1,
        #     'construct_-_[two_or_more_objects_to_become_one_(e.g._attach_drywall_to_studs)._usually_reversible.]': 2,
        #     'deactivate_-_[object_is_transformed_prevent_access_to_one_or_more_objects_inside_(e.g._close)_or_to_negative_some_functionality_(e.g._turn_power_off).]': 3,
        #     'deconstruct_-_[one_object_becomes_two_or_more_independent_objects(e.g.,_cut_wood_with_saw).]': 4,
        #     'deform_-_[a_flexible_or_amorphous_object_like_soft_metal,_dough,_dirt_or_cloth_takes_a_new_form_(e.g._fold_clothes,_roll_dough_into_a_ball,_bend_a_rod_in_half).]': 5,
        #     'deposit_-_[a_liquid,_flexible_or_amorphous_object_is_deposited_on_or_into_on_onto_another_object_(e.g.,_spread_joint_compound_on_drywall,_apply_paint_to_a_wall,_pour_liquid_into_a_cup,_pour_milk_in_a_bowl_of_cereal)]': 6,
        #     'other_-_[chemical_change_(e.g.,_burn_paper),_temperature_change_(e.g.,_boiling_water)]': 7,
        #     'remove_-_[a_liquid,_flexible_or_amorphous_object_is_removed_from_another_object_(e.g._wipe_up_split_milk,_remove_wall_paper,_pour_liquid_out_of_a_container)]': 8,
        # }
        self.no_state_chng = no_state_chng
        self.debug_only = debug_only
        if self.no_state_chng:
            assert os.path.exists(
                no_sc_clips_dir
            ), "Wrong -ive clips path provided"
            self.no_sc_clips_dir = no_sc_clips_dir
            self.no_sc_split_path = no_sc_split_path
            assert os.path.isfile(
                self.no_sc_split_path
            ), "Wrong split path for -ive clips provided"
        # Loading the JSON file for a deterministic test set
        self.TEST_JSON = test_json
        if self.mode == 'test':
            if os.path.isfile(self.TEST_JSON):
                print('[INFO] Test JSON file exists...')
                self.test_det_data = json.load(
                    open(self.TEST_JSON, 'r')
                )
            else:
                print('[INFO] Creating the test JSON file...')
                self.test_det_data = dict()
        self.VAL_JSON = val_json
        if self.mode == 'val':
            if os.path.isfile(self.VAL_JSON):
                print('[INFO] Validation JSON file exists...')
                self.val_det_data = json.load(
                    open(self.VAL_JSON, 'r')
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
        self.ann_data = json.load(open(self.ann_path, 'r'))
        clip_count = 0
        for video_id in tqdm(self.ann_data.keys(), desc='+ive clips'):
            intervals = self.ann_data[video_id]['annotated_intervals']
            for interval in intervals:
                interval_narrated_actions = interval['narrated_actions']
                for action in interval_narrated_actions:
                    # Selecting the clips based on the modes
                    if action['clip_id'] not in selected_clip_ids:
                        continue
                    if action['is_rejected'] is True or action['error'] is \
                        not None or action['is_invalid_annotation'] is True:
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
                            # Ignoring such cases; There are 1529 such cases
                            # as of 08/09/21
                            continue
                        clip_start_frame = np.floor(
                            clip_start_sec * 30
                        )
                        clip_end_frame = np.floor(
                            clip_end_sec * 30
                        )
                        pnr_frame = action['critical_frames']['pnr_frame']
                        if clip_start_frame > pnr_frame:
                            # Ignoring such cases; there are 8 cases as
                            # of 08-09-21
                            continue
                        state = action['state_transition']
                        if state is None:
                            # Ignoring such cases; there are 3474 such
                            # cases as of 08-09-21
                            continue
                        f_verb = action['freeform_verb']
                        s_verb = action['structured_verb']
                        # Check to make sure UID is not repeated
                        assert new_uid not in uid_check
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
                            "state": state,
                            "freeform_verb": f_verb,
                            "structured_verb": s_verb,
                            "clip_start_sec": clip_start_sec,
                            "clip_end_sec": clip_end_sec,
                            "clip_start_frame": clip_start_frame,
                            "clip_end_frame": clip_end_frame,
                            "video_id": video_id,
                        }
                        clip_count += 1
        # json.dump(temp_split_json, open(json_file, 'w'))
        # Using 80 positive samples for toy experiments
        if self.debug_only:
            temp_package = dict()
            for i in range(50):
                info = self.package[i]
                temp_package[i] = info

        if self.no_state_chng:
            no_sc_splits = json.load(open(self.no_sc_split_path, 'r'))
            no_sc_video_ids = no_sc_splits[self.mode]
            negative_clip_list = os.listdir(self.no_sc_clips_dir)
            # JSON file contining information to make the following for
            # loop faster
            no_sc_info_file = self.no_sc_info_file % self.mode
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
            for video_id in tqdm(negative_clip_list, desc='-ive clips'):
                if video_id in no_sc_video_ids:
                    assert video_id not in uid_check
                    if faster:
                        clip_start_frame = no_sc_info_dict[video_id]['start_frame']
                        clip_end_frame = no_sc_info_dict[video_id]['end_frame']
                    else:
                        clip_path = os.path.join(self.no_sc_clips_dir, video_id)
                        frames_itr = os.walk(clip_path)
                        try:
                            frames = next(frames_itr)[-1]
                        except:
                            breakpoint()
                        frames_sorted = [int(item.split('.')[0]) for item in frames]
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
                        "pnr_frame": None,
                        "state": 0,
                        "freeform_verb": None,
                        "structured_verb": None,
                        "clip_start_sec": clip_start_sec,
                        "clip_end_sec": clip_end_sec,
                        "clip_start_frame": clip_start_frame,
                        "clip_end_frame": clip_end_frame,
                        "video_id": video_id,
                    }
                    clip_count += 1
                    if clip_count > 25000 and self.debug_only:
                        break
            if not faster:
                json.dump(no_sc_info_dict, open(no_sc_info_file, 'w'))
        # Using 20 negative samples for toy experiments
        if self.debug_only:
            for i in range(50, 100, 1):
                info = self.package[clip_count-i]
                temp_package[i] = info

        if not self.debug_only:
            print(f'Number of clips for {self.mode}: {len(self.package)}')

        if self.debug_only:
            print(f'Using {len(temp_package)} samples for toy experiments...')
            self.package = temp_package

    def __len__(self):
        return len(self.package)

    def __getitem__(self, index):
        info = self.package[index]
        if info['pnr_frame'] is not None:
            self._extract_clip_frames(info)
            state = 1
        else:
            assert self.no_state_chng
            state = info['state']

        frames, labels, fps = self._sample_frames_gen_labels(info)
        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)
        frames = self.transforms(frames)

        if self.mode == 'test':
            # Saving the test json file if it doesn't exists
            if index == len(self.package) - 1 and not os.path.isfile(self.TEST_JSON):
                print('Saving the test JSON file...')
                json.dump(self.test_det_data, open(self.TEST_JSON, 'w'))

        if self.mode == 'val':
            # Saving the validation json file if it doesn't exists
            if index == len(self.package) - 1 and not os.path.isfile(self.VAL_JSON):
                print('Saving the validation JSON file...')
                json.dump(self.val_det_data, open(self.VAL_JSON, 'w'))

        return frames, labels, state, fps

    def _extract_clip_frames(self, info):
        """
        This method is used to extract and save frames for all the 8 seconds
        clips. If the frames are already saved, it does nothing.
        """
        clip_start_frame = info['clip_start_frame'].astype(np.int32)
        clip_end_frame = info['clip_end_frame'].astype(np.int32)
        unique_id = info['unique_id']
        video_path = os.path.join(
            self.videos_dir,
            info['video_id']
        )
        clip_save_path = os.path.join(self.CLIPS_SAVE_PATH, unique_id)
        # We can do do this fps for canonical data is 30.
        num_frames_per_video = 30 * 8
        if os.path.isdir(clip_save_path):
            # The frames for this clip are already saved.
            num_frames = len(os.listdir(clip_save_path))
            if num_frames < 240:
                return None
                print(
                    f'Deleting {clip_save_path} as it has {num_frames} frames'
                )
                # os.system(f'rm -r {clip_save_path}')
            else:
                return None
        print(f'Saving frames for {clip_save_path}...')
        if not os.path.exists(clip_save_path):
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
        print(f'Time taken: {time.time() - start}; {num_saved_frames} frames saved; {clip_save_path}')
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
                self.CLIPS_SAVE_PATH,
                info['unique_id']
            )
        else:
            # Clip path for clips with no state change
            clip_path = os.path.join(self.no_sc_clips_dir, info['unique_id'])
        message = f'Clip path {clip_path} does not exists...'
        assert os.path.isdir(clip_path), message
        num_frames_per_video = 2 * 8
        # Random clipping
        # Randomly choosing the duration of clip (between 5-8 seconds)
        random_length_seconds = np.random.uniform(5, 8)
        random_start_seconds = info['clip_start_sec'] + np.random.uniform(
            8 - random_length_seconds
        )
        random_start_frame = np.floor(random_start_seconds * 30).astype(np.int32)
        random_end_seconds = random_start_seconds + random_length_seconds
        if random_end_seconds > info['clip_end_sec']:
            random_end_seconds = info['clip_end_sec']
        random_end_frame = np.floor(random_end_seconds * 30).astype(np.int32)
        pnr_frame = info['pnr_frame']
        if pnr_frame is not None:
            keyframe_after_end = pnr_frame > random_end_frame
            keyframe_before_start = pnr_frame < random_start_frame
            if keyframe_after_end:
                random_end_frame = info['clip_end_frame'].astype(np.int32)
            if keyframe_before_start:
                random_start_frame = info['clip_start_frame'].astype(np.int32)

        if self.mode == 'test':
            # If the JSON does not exsist, we will need to create the data
            if not os.path.isfile(self.TEST_JSON):
                # Saving the id to the frames after random clippsing
                assert info['unique_id'] not in self.test_det_data.keys()
                self.test_det_data[info['unique_id']] = {
                    'clip_end_frame': int(random_end_frame),
                    'clip_start_frame': int(random_start_frame)
                }
            # If it exists, load the data from it
            else:
                random_start_frame = self.test_det_data[info['unique_id']]['clip_start_frame']
                random_end_frame = self.test_det_data[info['unique_id']]['clip_end_frame']
        if self.mode == 'val':
            # if the JSON does not exits, we will need to create the data
            if not os.path.isfile(self.VAL_JSON):
                assert info['unique_id'] not in self.val_det_data.keys()
                self.val_det_data[info['unique_id']] = {
                    'clip_end_frame': int(random_end_frame),
                    'clip_start_frame': int(random_start_frame)
                }
            # If it exists, load the data from it
            else:
                random_start_frame = self.val_det_data[info['unique_id']]['clip_start_frame']
                random_end_frame = self.val_det_data[info['unique_id']]['clip_end_frame']

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
            # Creating soft labels
            soft_labels = max(keyframe_candidates_list) - keyframe_candidates_list
            soft_labels = soft_labels/sum(soft_labels)
            if self.SOFT_LABELS:
                labels = soft_labels
            else:
                labels = hard_labels
        else:
            labels = keyframe_candidates_list
        # Calculating the effective fps. In other words, the fps after sampling
        # changes when we are randomly clipping and varying the duration of the
        # clip
        final_clip_length = (random_end_frame/30) - (random_start_frame/30)
        effective_fps = num_frames_per_video / final_clip_length
        return np.concatenate(frames), np.array(labels), effective_fps

    def get_frames_for(self, video_path, frames_list):
        """
        Code for decoding the video shared by FB
        """
        frames = list()
        with av.open(video_path) as container:
            for frame in _get_frames(frames_list, container, include_audio=False, audio_buffer_frames=0):
                frame = frame.to_rgb().to_ndarray()
                frames.append(frame)
        try:
            assert len(frames) == len(frames_list), f'{video_path}'
        except AssertionError:
            with open('data_prep_errors_27-08-21.txt', 'a') as file:
                file.write(f'Video: {video_path}; frames list: {frames_list}\n')
            return frames
        return frames