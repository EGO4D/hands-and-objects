#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .action_det_models import BMN
from .build import MODEL_REGISTRY, build_model  # noqa
from .frameselection_models import CNNRNN
from .multi_models import MultiTaskResNet, MultiTaskSlowFast, MultiTaskSlowFastTx, MultiTaskSlowFastPr, \
    MultiTaskPerceiver  # noqa
from .sta_models import ShortTermAnticipationResNet, ShortTermAnticipationSlowFast  # noqa
from .video_model_builder import ResNet, SlowFast, SlowFastTx, SlowFastPerceiver, Perceiver, PerceiverPyr  # noqa
