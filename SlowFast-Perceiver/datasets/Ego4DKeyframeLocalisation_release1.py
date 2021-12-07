"""
This file contains data loader for Ego4D Keyframe Localisation benchmark
"""

import os
import json

from .build_dataset import DATASET_REGISTRY

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


@DATASET_REGISTRY.register()
class Ego4DKeyframeLocalisation_release1(torch.utils.data.Dataset):
    """
    Ego4D Keyframe detection video loader.
    """
    def __init__(self, cfg, mode):
        """
        Construct the Ego4D keyframe localization loader.
        """
        assert mode in [
            'train',
            'val',
            'test'
        ], "Split `{}` not supported for Keyframe detection.".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.verbose = self.cfg.MISC.VERBOSE
        self.no_state_chng = self.cfg.DATA_LOADER.IS_NO_STATE_CHANGE
        # Getting path to the annotations
        self.ann_path = self.cfg.DATA.ANN_PATH
        assert os.path.exists(self.ann_path), "Wrong annotation path provided"
        if self.verbose:
            print('[INFO] Using annotations from: {}'.format(self.ann_path))
        # Getting path to the videos
        self.videos_dir = self.cfg.DATA.VIDEO_DIR_PATH
        assert os.path.exists(self.videos_dir), "Wrong video path provided"
        if self.verbose:
            print('[INFO] Using videos from: {}'.format(self.videos_dir))
        if self.no_state_chng:
            assert os.path.exists(
                self.cfg.DATA.NO_SC_PATH
            ), "Wrong no state change videos path provided"
            self.no_sc_clips_dir = self.cfg.DATA.NO_SC_PATH
            self.no_sc_split_path = self.cfg.DATA.NO_SC_SPLIT_PATH
            assert os.path.isfile(self.no_sc_split_path), ("Wrong no state "
                "change split path provided")
            if self.verbose:
                print('[INFO] Using no state change clips from: {}'.format(
                    self.no_sc_clips_dir
                ))
        # Getting path to the dataset split
        self.split_path = self.cfg.DATA.SPLIT_PATH
        assert os.path.exists(self.split_path), "Wrong split path provided"
        if self.verbose:
            print('[INFO] Using data splits from: {}'.format(self.split_path))
        # Constructing the dataset
        print('[INFO] Constructing keyframe localisation in mode `{}`'.format(
            self.mode
        ))
        # Loading the JSON file for a deterministic test set
        if self.cfg.TEST.ENABLE and self.mode == 'test':
            if os.path.isfile(self.cfg.TEST.JSON):
                print('[INFO] Test JSON file exists...')
                self.test_det_data = json.load(
                    open(self.cfg.TEST.JSON, 'r')
                )
            else:
                print('[INFO] Creating the test JSON file...')
                self.test_det_data = dict()
        if self.mode == 'val':
            if os.path.isfile(self.cfg.TEST.VAL_JSON):
                print('[INFO] Validation JSON file exists...')
                self.val_det_data = json.load(
                    open(self.cfg.TEST.VAL_JSON, 'r')
                )
            else:
                print('[INFO] Creating the validation JSON file...')
                self.val_det_data = dict()
        self._construct_loader()

    def _construct_loader(self):
        """
        This method constructs the video and annotation loader.

        Returns:
            None
        """
        self._path_to_videos = dict()
        self.package = list()
        self.video_names = list()
        self.video_ids = dict()
        splits = json.load(open(self.split_path, 'r'))
        if self.mode == 'train':
            self.video_uids = splits['train']
        elif self.mode == 'test':
            self.video_uids = splits['test']
        else:
            self.video_uids = splits['val']

        videos = os.listdir(self.videos_dir)
        for video in videos:
            video_uid = video.split('_')[0]
            if video_uid in self.video_uids:
                self.video_names.append(video)

        for name in self.video_names:
            video_uid = name.split('_')[0]
            video_index = name.split('_')[-1].split('.')[0]
            if video_uid not in self.video_ids:
                self.video_ids[video_uid] = [video_index]
                self._path_to_videos[video_uid] = [
                    os.path.join(self.videos_dir, name)
                ]
            else:
                self.video_ids[video_uid].append(video_index)
                self._path_to_videos[video_uid].append(
                    os.path.join(self.videos_dir, name)
                )

        self.ann_data = json.load(open(self.ann_path, 'r'))
        # Handling all the duplicate clip ID cases
        unique_id_list = list()
        for video_id in tqdm(self.ann_data.keys()):
            if video_id in self.video_ids.keys():
                if video_id != '1b9f06a7-b26d-4c74-863e-1d4fa22bbc37':
                    annotations = sorted(
                        self.ann_data[video_id]['benchmarks']['forecasting_hands_objects'],
                        key=lambda a: a["critical_frame_selection_parent_start_sec"]
                    )
                    video_path = self._path_to_videos[video_id]
                    for annotation in annotations:
                        annotation_uid = annotation['annotation_uids']['critical_frame_selection']
                        if annotation_uid not in unique_id_list:
                            unique_id_list.append(annotation_uid)
                            self.package.append(
                                (
                                    video_id,
                                    annotation,
                                    video_path
                                )
                            )
        # Code for adding videos with no state change
        if self.no_state_chng:
            no_sc_splits = json.load(open(self.no_sc_split_path, 'r'))
            no_sc_video_ids = no_sc_splits[self.mode]
            for video_id in tqdm(no_sc_video_ids):
                video_path = os.path.join(self.no_sc_clips_dir, video_id)
                # assert os.path.isdir(video_path)
                annotation = None
                self.package.append(
                    (
                        video_id,
                        annotation,
                        video_path
                    )
                )
        # Making sure that all the no-state change clips are not at the end
        # of the list
        if self.no_state_chng:
            np.random.shuffle(self.package)

        if self.cfg.MISC.TEST_TRAIN_CODE:
            print('[INFO] Using just 50 samples...')
            self.package = self.package[:50]


    def __len__(self):
        """
        Returns:
            (int): the number of clips in the dataset.
        """
        return len(self.package)

    def __getitem__(self, index):
        video_id, annotation, video_path = self.package[index]
        if annotation is None:
            if not self.no_state_chng:
                raise Exception("Clips with no state change loaded.")
            else:
                state_change_label = np.zeros(1)
                frames, labels, effective_fps = self._sample_frames_gen_labels(
                    annotation,
                    clip_frames_path=video_path
                )
        else:
            state_change_label = np.ones(1)
            self._extract_clip_frames(video_path, video_id, annotation)
            frames, labels, effective_fps = self._sample_frames_gen_labels(
                annotation,
                clip_frames_path=None
            )

        if self.cfg.TEST.ENABLE and self.mode == 'test':
            # Saving the test json if it does not exists
            if not os.path.isfile(self.cfg.TEST.JSON):
                if index == len(self.package) - 1:
                    print('[INFO] Saving the test json file...!')
                    json.dump(
                        self.test_det_data,
                        open(self.cfg.TEST.JSON, 'w')
                    )

        if self.mode == 'val':
            # Saving the validation json file if it does not exists
            if not os.path.isfile(self.cfg.TEST.VAL_JSON):
                if index == len(self.package) - 1:
                    print('[INFO] Saving the validation json file...!')
                    json.dump(
                        self.val_det_data,
                        open(self.cfg.TEST.VAL_JSON, 'w')
                    )

        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)
        return [frames], labels, state_change_label, effective_fps

    def _extract_clip_frames(self, video_paths, video_id, annotation):
        """
        This method extract clips from a given set of videos and saves them
        to a directory.

        Args:
            video_paths (list): list of videos linked to a video id
            video_id (str): video unique id
            annotation (dict): annotation for the clip for which the frames
            are to be extracted

        Returns:
            None
        """
        # calculating video's fps
        videocap = cv2.VideoCapture(video_paths[0])
        fps = int(videocap.get(cv2.CAP_PROP_FPS))
        videocap.release()

        clip_uid = annotation['annotation_uids']['critical_frame_selection']
        start_sec = annotation['critical_frame_selection_parent_start_sec']
        end_sec = annotation['critical_frame_selection_parent_end_sec']
        clip_folder = os.path.join(self.cfg.DATA.CLIPS_SAVE_PATH, clip_uid)
        if os.path.isdir(clip_folder):
            # Frames from clip already saved
            num_frames = len(os.listdir(clip_folder))
            if num_frames <= 16:
                print('Deleting folder {} as it has {} frames...'.format(clip_folder, num_frames))
                os.system("rm -r {}".format(clip_folder))
            else:
                return None
        # else:
        try:
            os.makedirs(clip_folder)
        except Exception as e:
            # If the folder is empty it will be caught at a later stage
            pass
        start_frame_count = np.round(start_sec*fps)
        end_frame_count = np.round(end_sec*fps)
        save_path = os.path.join(clip_folder, '{}_{}.jpg')
        video_paths_sorted = sorted(
            video_paths,
            key=lambda a: int(a.split('_')[-1].split('.')[0])
        )
        current_ann_done = False
        chunk_info = self.get_chunk_num(
            video_paths_sorted,
            video_id,
            fps,
            start_frame_count,
            end_frame_count
        )
        saved = 0
        for video, parent_frame_count in chunk_info:
            videocap = cv2.VideoCapture(video)
            # https://stackoverflow.com/a/37408638/9247183
            num_frames = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
            while num_frames > 0 and not current_ann_done:
                success, frame = videocap.read()
                parent_frame_count += 1
                condition = parent_frame_count >= start_frame_count and \
                    parent_frame_count <= end_frame_count
                if condition and success:
                    cv2.imwrite(
                        save_path.format(
                            parent_frame_count, str(fps)
                        ),
                        frame
                    )
                    saved += 1
                elif parent_frame_count > end_frame_count:
                    current_ann_done = True
                num_frames -= 1
            videocap.release()
        print('{} frames saved for {}'.format(saved, clip_uid))
        return None

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

    def _sample_clip(
        self,
        frames,
        no_frames_required,
        clip_frames_path,
        keyframe_frame_count=None
    ):
        """
        This method is used to sample the frames in a way that we always have
        same number of output frames for videos with different lengths and
        different sampling rates.

        Args:
            frames (list): list of names of frames for the clip being processed
            no_frames_required (int): number of frames required to be extracted
                from the clip
            clip_frames_path (str): path to the folder where all the frame 
                from the clip are saved
            keyframe_frame_count (int): keyframe's frame number
        
        Returns:
            frames (list): list of loaded frames
            keyframe_candidates_list (list): list of distance between keyframe
                and other frames in terms of location
        """
        num_frames = len(frames)
        if num_frames < no_frames_required:
            print("Possible issue: {}".format(clip_frames_path))
        error_message = 'Can\'t sample more frames than there are in the video'
        assert num_frames >= no_frames_required, error_message
        lower_lim = np.floor(num_frames/no_frames_required)
        upper_lim = np.ceil(num_frames/no_frames_required)
        lower_frames = list()
        upper_frames = list()
        lower_keyframe_candidates_list = list()
        upper_keyframe_candidates_list = list()
        for count, frame in enumerate(frames):
            if (count + 1) % lower_lim == 0:
                frame_path = os.path.join(clip_frames_path, frame)
                lower_frames.append(self._load_frame(frame_path))
                if keyframe_frame_count is not None:
                    lower_keyframe_candidates_list.append(
                        np.abs((count + 1) - keyframe_frame_count)
                    )
                else:
                    lower_keyframe_candidates_list.append(0.0)
            if (count + 1) % upper_lim == 0:
                frame_path = os.path.join(clip_frames_path, frame)
                upper_frames.append(self._load_frame(frame_path))
                if keyframe_frame_count is not None:
                    upper_keyframe_candidates_list.append(
                        np.abs((count + 1) - keyframe_frame_count)
                    )
                else:
                    upper_keyframe_candidates_list.append(0.0)
        if len(upper_frames) < no_frames_required:
            return (
                lower_frames[:no_frames_required],
                lower_keyframe_candidates_list[:no_frames_required]
            )
        else:
            return (
                upper_frames[:no_frames_required],
                upper_keyframe_candidates_list[:no_frames_required]
            )

    def _sample_frames_gen_labels(self, annotation, clip_frames_path=None):
        """
        This method is used for sampling frames from saved directory and
        generate corresponding hard or soft labels.

        Args:
            annotation (dict): annotation for the clip for which the frames
            are to be sampled

        Returns:
            frames (ndarray): extracted frames
            labels (ndarray): generated labels
        """
        if annotation is not None:
            clip_uid = annotation['annotation_uids']['critical_frame_selection']
            clip_frames_path = os.path.join(
                self.cfg.DATA.CLIPS_SAVE_PATH,
                clip_uid
            )
            assert os.path.isdir(clip_frames_path), ("Frames for clip {} not"
            " extracted".format(clip_uid))
        else:
            clip_uid = clip_frames_path.split('/')[-1]

        frames = os.listdir(clip_frames_path)
        # Sorting the frames to preserve the temporal information
        frames = sorted(frames, key=lambda a: int(a.split('_')[0]))
        try:
            fps = int(frames[0].split('_')[-1].split('.')[0])
        except:
            print('Error in fps')
            print(clip_frames_path)
            import pdb
            pdb.set_trace()
        if annotation is not None:
            # Additional in v5 annotations
            for frame_item in annotation['frames']:
                if frame_item['frame_type'] == 'pnr':
                    pnr_frame_sec = frame_item['timestamp']
            keyframe_sec = (pnr_frame_sec - \
                annotation['critical_frame_selection_parent_start_sec'])
            keyframe_frame_count = np.round(keyframe_sec * fps)
        else:
            keyframe_frame_count = None
        sampling_fps = self.cfg.DATA.SAMPLING_FPS
        # Number of frame we want from every video
        num_frames_per_video = sampling_fps * self.cfg.DATA.CLIP_LEN_SEC
        # Random clipping
        # Randomly choosing the duration of clip (between 5-8)
        random_length_seconds = np.random.uniform(5, 8)
        random_start_seconds = np.random.uniform(8 - random_length_seconds)
        random_end_seconds = random_start_seconds + random_length_seconds
        if annotation is not None:
            keyframe_after_end = keyframe_sec > random_end_seconds
            keyframe_before_start = keyframe_sec < random_start_seconds
            if keyframe_after_end:
                random_end_seconds = 8.
            if keyframe_before_start:
                random_start_seconds = 0.
        clip_start_frame = int(random_start_seconds * fps)
        clip_end_frame = int(random_end_seconds * fps)
        if self.cfg.TEST.ENABLE and self.mode == 'test':
            # If the json does not exists, we will need to create the data
            if not os.path.isfile(self.cfg.TEST.JSON):
                # Saving the id of the last frame after random clipping
                assert clip_uid not in self.test_det_data.keys()
                self.test_det_data[clip_uid] = {
                    'clip_end_frame': clip_end_frame,
                    'clip_start_frame': clip_start_frame
                }
            else:
                # Loading the id of the last frame from a previous run
                clip_end_frame = self.test_det_data[clip_uid]['clip_end_frame']
                clip_start_frame = self.test_det_data[clip_uid]['clip_start_frame']
        if self.mode == 'val':
            # If the json does not exists, we will need to create it
            if not os.path.isfile(self.cfg.TEST.VAL_JSON):
                assert clip_uid not in self.val_det_data.keys()
                self.val_det_data[clip_uid] = {
                    'clip_end_frame': clip_end_frame,
                    'clip_start_frame': clip_start_frame
                }
            else:
                # Loading the id of the last frame from a previous run
                clip_end_frame = self.val_det_data[clip_uid]['clip_end_frame']
                clip_start_frame = self.val_det_data[clip_uid]['clip_start_frame']
        frames = frames[clip_start_frame:clip_end_frame]
        if keyframe_frame_count is not None:
            # The keyframe needs to be adjusted as we might have clipped a
            # portion of the clip from the beginning
            keyframe_frame_count -= clip_start_frame
        candidate_frames, keyframe_candidates_list = self._sample_clip(
            frames,
            num_frames_per_video,
            clip_frames_path,
            keyframe_frame_count=keyframe_frame_count
        )
        if annotation is not None:
            keyframe_location = np.argmin(keyframe_candidates_list)
            hard_labels = np.zeros((len(candidate_frames)))
            hard_labels[keyframe_location] = 1
            # Creating soft labels
            soft_labels = max(keyframe_candidates_list) - keyframe_candidates_list
            soft_labels = soft_labels/sum(soft_labels)
            if self.cfg.DATA_LOADER.SOFT_LABELS:
                labels = soft_labels
            else:
                labels = hard_labels
        else:
            labels = keyframe_candidates_list
        # Calculating the effective fps (the fps after sampling changes when we
        # are randomly clipping and varying the duration of the clip)
        final_clip_length = random_end_seconds - random_start_seconds
        effective_fps = num_frames_per_video / final_clip_length
        return np.concatenate(candidate_frames), np.array(labels), effective_fps


    def get_chunk_num(
        self,
        video_paths_sorted,
        video_id,
        fps,
        start_frame_count,
        end_frame_count
    ):
        """
        This method uses the video_component_location_access file
        and uses that information to tell which chuck of the video has the
        desired frames in it. This helps us with the case when some of
        the chunks are redacted.
        """
        video_locations = pd.read_csv(self.cfg.DATA.VIDEO_LOCATIONS_CSV)
        video_uid_occurances = video_locations.query(
            'video_uid=="{}"'.format(video_id)
        ).values
        video_paths = list()
        for occurance in video_uid_occurances:
            provided_component_index = occurance[2]
            occurance_start_frame_count = np.round(occurance[3] * fps)
            occurance_end_frame_count = np.round(occurance[4] * fps)
            condition = ((occurance_start_frame_count <= start_frame_count) \
                or (occurance_end_frame_count >= end_frame_count))
            if condition:
                for video in video_paths_sorted:
                    video_component_index = int(video.split('_')[-1].split('.')[0])
                    if video_component_index == provided_component_index:
                        video_paths.append(
                            (
                                video,
                                int(occurance_start_frame_count)
                            )
                        )
        return video_paths
