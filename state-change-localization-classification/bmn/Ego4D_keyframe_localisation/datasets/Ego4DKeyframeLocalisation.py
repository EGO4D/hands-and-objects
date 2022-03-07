"""
This file contains data loader for Ego4D Keyframe Localisation benchmark
"""

import pdb
import os
import cv2
import json
import datetime
import numpy as np
import pandas as pd

from .build_dataset import DATASET_REGISTRY

import torch

from utils.BMN_utils import ioa_with_anchors, iou_with_anchors

@DATASET_REGISTRY.register()
class Ego4DKeyframeLocalisation(torch.utils.data.Dataset):
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
        #######
        #BMN
        #######
        self.temporal_scale = self.cfg.BMN.TEMPORAL_SCALE
        self.temporal_gap = 1. / self.temporal_scale
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]

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
        for video_id in self.ann_data.keys():
            if video_id in self.video_ids.keys():
                if video_id != '1b9f06a7-b26d-4c74-863e-1d4fa22bbc37':
                    annotations = sorted(self.ann_data[video_id]['annotations'][
                        'forecast_hand_object_frame_selection_4'
                    ], key=lambda a: a["parent_start_sec"])
                    num_videos = self.video_ids[video_id]
                    video_path = self._path_to_videos[video_id]
                    for annotation in annotations:
                        if annotation['clip_uid'] not in unique_id_list:
                            unique_id_list.append(annotation['clip_uid'])
                            self.package.append(
                                (
                                    video_id,
                                    annotation,
                                    video_path
                                )
                            )
        # Code for adding videos with no state change
        if self.no_state_chng:
            no_sc_video_ids = os.listdir(self.no_sc_clips_dir)
            for video_id in no_sc_video_ids:
                video_path = os.path.join(self.no_sc_clips_dir, video_id)
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
                frames, labels, prec_labels = self._sample_frames_gen_labels(
                    annotation,
                    clip_frames_path=video_path
                )
                keyframe_time, prec_frame_time = -99, -99
        else:
            state_change_label = np.ones(1)
            self._extract_clip_frames(video_path, video_id, annotation)
            frames, labels, prec_labels, keyframe_time, prec_frame_time = self._sample_frames_gen_labels(
                annotation,
                clip_frames_path=None
            )
        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)
        match_score_start, match_score_end, confidence_score = self._get_train_label_bmn(self.anchor_xmin, self.anchor_xmax, prec_frame_time, keyframe_time)
        return annotation['clip_uid'], [frames], labels, prec_labels, state_change_label, confidence_score, match_score_start, match_score_end, keyframe_time, prec_frame_time

    def _get_train_label_bmn(self, anchor_xmin, anchor_xmax, prec_frame_time, keyframe_time, duration=8):
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        tmp_start = max(min(1, prec_frame_time / duration), 0)
        tmp_end = max(min(1, keyframe_time / duration), 0)
        gt_bbox.append([tmp_start, tmp_end])

        #generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[i, j] = np.max(
                    iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        gt_iou_map = torch.Tensor(gt_iou_map)

        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)

        return match_score_start, match_score_end, gt_iou_map



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

        clip_uid = annotation['clip_uid']
        start_sec = annotation['parent_start_sec']
        end_sec = annotation['parent_end_sec']
        clip_folder = os.path.join(self.cfg.DATA.CLIPS_SAVE_PATH, clip_uid)
        if os.path.isdir(clip_folder):
            # Frames from clip already saved
            num_frames = len(os.listdir(clip_folder))
            if num_frames <= 16:
                print('Deleting folder {} as it has {} frames...'.format(clip_folder, num_frames))
                os.system("rm -r {}".format(clip_folder))
            # if num_frames < 241:
            #     print('Deleting folder {} as it has {} frames...'.format(clip_folder, num_frames))
            #     os.system("rm -r {}".format(clip_folder))
            # if num_frames > 0:
            #     if num_frames < 241:
            #         print('Clip folder: {} No. of saved frames: {}'.format(
            #             clip_folder,
            #             num_frames)
            #         )
            #     return None
            # elif num_frames == 241:
            #     return None
            else:
                return None
        # else:
        try:
            os.mkdir(clip_folder)
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
        keyframe_frame_count=None,
        prec_frame_count=None
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
        lower_prec_candidates_list = list()
        upper_prec_candidates_list = list()

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

                if prec_frame_count is not None:
                    lower_prec_candidates_list.append(
                        np.abs((count + 1) - prec_frame_count)
                    )
                else:
                    lower_prec_candidates_list.append(0.0)

            if (count + 1) % upper_lim == 0:
                frame_path = os.path.join(clip_frames_path, frame)
                upper_frames.append(self._load_frame(frame_path))
                if keyframe_frame_count is not None:
                    upper_keyframe_candidates_list.append(
                        np.abs((count + 1) - keyframe_frame_count)
                    )
                else:
                    upper_keyframe_candidates_list.append(0.0)
                if prec_frame_count is not None:
                    upper_prec_candidates_list.append(
                        np.abs((count + 1) - prec_frame_count)
                    )
                else:
                    upper_prec_candidates_list.append(0.0)
        if len(upper_frames) < no_frames_required:
            return (
                lower_frames[:no_frames_required],
                lower_keyframe_candidates_list[:no_frames_required],
                lower_prec_candidates_list[:no_frames_required]
            )
        else:
            return (
                upper_frames[:no_frames_required],
                upper_keyframe_candidates_list[:no_frames_required],
                upper_prec_candidates_list[:no_frames_required]
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
            clip_uid = annotation['clip_uid']
            clip_frames_path = os.path.join(
                self.cfg.DATA.CLIPS_SAVE_PATH,
                clip_uid
            )
            assert os.path.isdir(clip_frames_path), ("Frames for clip {} not"
            " extracted".format(clip_uid))
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
            keyframe_sec = (annotation['pnr_frame_sec'] - \
                annotation['parent_start_sec'])
            prec_frame_sec = (annotation['pre_condition_frame_sec'] - \
                annotation['parent_start_sec'])
            keyframe_frame_count = np.round(keyframe_sec * fps)
            prec_frame_count = np.round(prec_frame_sec * fps)
        else:
            #keyframe_sec = -99.9
            #prec_frame_sec = -99.9
            keyframe_frame_count = None
            prec_frame_count = None
        sampling_fps = self.cfg.DATA.SAMPLING_FPS
        # Number of frame we want from every video
        num_frames_per_video = sampling_fps * self.cfg.DATA.CLIP_LEN_SEC
        # Random clipping
        # Randomly choosing the duration of clip (between 5-8)
        random_length_seconds = np.random.randint(5, 9)
        if annotation is not None:
            keyframe_present = keyframe_sec < random_length_seconds
            pc_frame_present = prec_frame_sec < random_length_seconds
            if not (keyframe_present and pc_frame_present):
                # If keyframe does not fall into the selected region
                random_length_seconds = 8
                # controlled_start_time = np.ceil(keyframe_sec)
                # random_length_seconds = np.random.randint(controlled_start_time, 9)
        clip_end_frame = random_length_seconds * fps
        frames = frames[:clip_end_frame]
        candidate_frames, keyframe_candidates_list, prec_candidates_list = self._sample_clip(
            frames,
            num_frames_per_video,
            clip_frames_path,
            keyframe_frame_count=keyframe_frame_count,
            prec_frame_count = prec_frame_count
        )
        if annotation is not None:
            keyframe_location = np.argmin(keyframe_candidates_list)
            prec_location = np.argmin(prec_candidates_list)

            hard_labels = np.zeros((len(candidate_frames)))
            hard_labels[keyframe_location] = 1

            prec_hard_labels = np.zeros((len(candidate_frames)))
            prec_hard_labels[prec_location] = 1
            # Creating soft labels
            soft_labels = max(keyframe_candidates_list) - keyframe_candidates_list
            soft_labels = soft_labels/sum(soft_labels)
            prec_soft_labels = max(prec_candidates_list) - prec_candidates_list
            prec_soft_labels = prec_soft_labels/sum(prec_soft_labels)

            if self.cfg.DATA_LOADER.SOFT_LABELS:
                labels = soft_labels
                prec_labels = prec_soft_labels
            else:
                labels = hard_labels
                prec_labels = prec_hard_labels
            return np.concatenate(candidate_frames), np.array(labels), np.array(prec_labels), keyframe_sec, prec_frame_sec
        else:
            labels, prec_labels = keyframe_candidates_list, prec_candidates_list
            return np.concatenate(candidate_frames), np.array(labels), np.array(prec_labels)
        

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
