# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn

from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class BMN(nn.Module):
    def __init__(self, cfg):
        super(BMN, self).__init__()
        self.observation_window = cfg.BMN.OBSERVATION_WINDOW
        self.max_duration = cfg.BMN.MAX_DURATION
        self.prop_boundary_ratio = cfg.BMN.PROP_BOUNDARY_RATIO
        self.num_sample = cfg.BMN.NUM_SAMPLE
        self.num_sample_perbin = cfg.BMN.NUM_SAMPLE_PERBIN
        self.feat_dim = cfg.BMN.FEAT_DIM

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        # self._get_interp1d_mask()
        self._construct_network()

    def _construct_network(self):
        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_a = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        '''
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        '''

        # Proposal Evaluation Module
        '''
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )
        '''

    def forward(self, x):
        base_feature = self.x_1d_b(x)
        actionness = self.x_1d_a(base_feature).squeeze(1)
        # start = self.x_1d_s(base_feature).squeeze(1)
        # end = self.x_1d_e(base_feature).squeeze(1)
        # confidence_map = self.x_1d_p(base_feature)
        # confidence_map = self._boundary_matching_layer(confidence_map)
        # confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        # confidence_map = self.x_2d_p(confidence_map)
        # return confidence_map, start, end
        return actionness

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        # BS x C x N x D x T
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0], input_size[1], self.num_sample,
                                                        self.max_duration, self.observation_window)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(self.observation_window):
            mask_mat_vector = []
            for duration_index in range(self.max_duration):
                if start_index + duration_index < self.observation_window:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.observation_window, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.observation_window, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.observation_window, -1), requires_grad=False)
