"""
Video model for Ego4D benchmark on Keyframe Localisation
"""

import torch
import numpy as np
import torch.nn as nn

from .video_model_builder import ResNet, _POOL1
from . import head_helper
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class KeyframeLocalisationClassification(ResNet):
    def _construct_network(self, cfg):
        super()._construct_network(cfg, with_head=False)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = head_helper.ResNetKeyframeLocalizationHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES[0],
            pool_size=[
                [
                    1,
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        self.head_name = "head_kf_loc_class"
        self.add_module(self.head_name, head)


@MODEL_REGISTRY.register()
class KeyframeLocalisationRegression(ResNet):
    """
    Not a scalable approach but, adding code here for the sake of completeness
    """
    def _construct_network(self, cfg):
        super()._construct_network(cfg, with_head=False)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = head_helper.ResNetRegressionHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES[0],
            pool_size=[
                [
                    1,
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT
        )
        self.head_name = "head_kf_loc_regress"
        self.add_module(self.head_name, head)
