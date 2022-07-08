"""
This file contains the code for data loader using the canonical version of the
data for the Keyframe localisation task.
"""

import os
import json
import time

import av
import cv2
import torch
import numpy as np
from tqdm import tqdm

from utils.trim import _get_frames
from .build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class StateChangeDetectionAndKeyframeLocalisation(torch.utils.data.Dataset):
    """
    Data loader for state change detection and key-frame localization.
    This data loader assumes that the user has alredy extracted the frames from
    all the videos using the `train.json`, `test_unnotated.json`, and
    'val.json' provided.
    """
    def __init__(self, cfg, mode):
        assert mode in [
            'train',
            'val',
            'test'
        ], "Split `{}` not supported for Keyframe detection.".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.ann_path = os.path.join(cfg.DATA.ANN_DIR, f'{self.mode}.json')
        ann_err_msg = f"Wrong annotation path provided {self.ann_path}"
        assert os.path.exists(self.ann_path), ann_err_msg
        self.video_dir = self.cfg.DATA.VIDEO_DIR_PATH
        assert os.path.exists(self.video_dir), "Wrong videos path provided"
        self.positive_vid_dir = self.cfg.DATA.CLIPS_SAVE_PATH
        positive_vid_err_msg = "Wrong positive clips' frame path provided"
        assert os.path.exists(self.positive_vid_dir), positive_vid_err_msg
        self.negative_vid_dir = self.cfg.DATA.NO_SC_PATH
        negative_vid_err_msg = "Wrong negative clips' frame path provided"
        assert os.path.exists(self.negative_vid_dir), negative_vid_err_msg
        self._construct_loader()

    def _construct_loader(self):
        self.package = dict()
        self.ann_data = json.load(open(self.ann_path, 'r'))
        for count, value in enumerate(
            tqdm(self.ann_data, desc='Preparing data')
        ):
            clip_start_sec = value['parent_start_sec']
            clip_end_sec = value['parent_end_sec']
            clip_start_frame = value['parent_start_frame']
            clip_end_frame = value['parent_end_frame']
            video_id = value['video_uid']
            unique_id = value['unique_id']
            assert count not in self.package.keys()
            if self.mode in ['train', 'val']:
                state_change = value['state_change']
                # Fix for issue #8
                if 'parent_pnr_frame' in value.keys():
                    pnr_frame = value['parent_pnr_frame']
                else:
                    pnr_frame = value['pnr_frame']
            else:
                state_change = None
                pnr_frame = None
            self.package[count] = {
                'unique_id': unique_id,
                'pnr_frame': pnr_frame,
                'state': 0 if state_change is False else 1,
                'clip_start_sec': clip_start_sec,
                'clip_end_sec': clip_end_sec,
                'clip_start_frame': int(clip_start_frame),
                'clip_end_frame': int(clip_end_frame),
                'video_id': video_id,
            }
        print(f'Number of clips for {self.mode}: {len(self.package)}')

    def __len__(self):
        return len(self.package)

    def __getitem__(self, index):
        info = self.package[index]
        state = info['state']
        self._extract_clip_frames(info)
        frames, labels, _ = self._sample_frames_gen_labels(info)
        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)
        clip_len = info['clip_end_sec'] - info['clip_start_sec']
        clip_frame = info['clip_end_frame'] - info['clip_start_frame'] + 1
        fps = clip_frame / clip_len
        return [frames], labels, state, fps, info

    def _extract_clip_frames(self, info):
        """
        This method is used to extract and save frames for all the 8 seconds
        clips. If the frames are already saved, it does nothing.
        """
        clip_start_frame = info['clip_start_frame']
        clip_end_frame = info['clip_end_frame']
        unique_id = info['unique_id']
        video_path = os.path.join(
            self.video_dir,
            info['video_id']
        )
        if info['pnr_frame'] is not None:
            clip_save_path = os.path.join(self.positive_vid_dir, unique_id)
        else:
            clip_save_path = os.path.join(self.negative_vid_dir, unique_id)
        # We can do do this fps for canonical data is 30.
        num_frames_per_video = 30 * self.cfg.DATA.CLIP_LEN_SEC
        if os.path.isdir(clip_save_path):
            # The frames for this clip are already saved.
            num_frames = len(os.listdir(clip_save_path))
            if num_frames < (clip_end_frame - clip_start_frame):
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
            self.cfg.DATA.CROP_SIZE,
            self.cfg.DATA.CROP_SIZE
        ))
        frame = np.expand_dims(frame, axis=0).astype(np.float32)
        return frame

    def _sample_frames_gen_labels(self, info):
        if info['pnr_frame'] is not None:
            clip_path = os.path.join(
                self.positive_vid_dir,
                info['unique_id']
            )
        else:
            # Clip path for clips with no state change
            clip_path = os.path.join(
                self.negative_vid_dir,
                info['unique_id']
            )
        message = f'Clip path {clip_path} does not exists...'
        assert os.path.isdir(clip_path), message
        num_frames_per_video = (
            self.cfg.DATA.SAMPLING_FPS * self.cfg.DATA.CLIP_LEN_SEC
        )
        pnr_frame = info['pnr_frame']
        if self.mode == 'train':
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
            random_end_frame = np.floor(
                random_end_seconds * 30
            ).astype(np.int32)
            if pnr_frame is not None:
                keyframe_after_end = pnr_frame > random_end_frame
                keyframe_before_start = pnr_frame < random_start_frame
                if keyframe_after_end:
                    random_end_frame = info['clip_end_frame']
                if keyframe_before_start:
                    random_start_frame = info['clip_start_frame']
        elif self.mode in ['test', 'val']:
            random_start_frame = info['clip_start_frame']
            random_end_frame = info['clip_end_frame']

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
        return np.concatenate(frames), np.array(labels), effective_fps

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
