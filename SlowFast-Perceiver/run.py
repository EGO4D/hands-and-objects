import os
import pdb
import torch
import pprint
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


import utils.logging as logging
from utils.parser import parse_args, load_config
from tasks.keyframe_detection import KeyframeLocalisation
from tasks.keyframe_detection import StateChangeAndKeyframeLocalisation
from tasks.keyframe_detection import StateChangeDetection, FrameSelection


logger = logging.get_logger(__name__)


def main(cfg):
    cfg.DATA.NUM_FRAMES = cfg.DATA.SAMPLING_FPS * cfg.DATA.CLIP_LEN_SEC
    checkpoint_mode = "min"
    if cfg.DATA.TASK == "frame_detection_classification":
        TaskType = KeyframeLocalisation
    elif cfg.DATA.TASK == "state_change_detection_and_keyframe_localization":
        TaskType = StateChangeAndKeyframeLocalisation
    elif cfg.DATA.TASK == 'state_change_detection':
        TaskType = StateChangeDetection
        checkpoint_mode = "max"
    elif cfg.DATA.TASK == "frame_selection":
        TaskType = FrameSelection
        checkpoint_mode = "max"

    else:
        raise NotImplementedError('Task {} not implemented'.format(
            cfg.DATA.TASK
        ))

    task = TaskType(cfg)

    if cfg.MISC.ENABLE_LOGGING:
        args = {}
        # args = {
        #     'callbacks': [LearningRateMonitor()]
        # }
    else:
        args = {
            'logger': False
        }

    ckpt_callback = ModelCheckpoint(
            dirpath='/home/abrsh/canonical_dataset/benchmark/checkpoints',
            monitor=task.checkpoint_metric,
            mode=checkpoint_mode,
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_val_epochs=1,
            verbose=True,
            save_top_k=1,
            filename=f"{cfg.MODEL.MODEL_NAME}" + "-{epoch}-{val_loss:2f}-{state_change_metric}"
        )

    trainer = Trainer(
        gpus=cfg.MISC.NUM_GPUS,
        num_nodes=cfg.MISC.NUM_SHARDS,
        accelerator=cfg.SOLVER.ACCELERATOR,
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=0,
        benchmark=True,
        check_val_every_n_epoch=5,
        replace_sampler_ddp=False,
        checkpoint_callback=True,
        callbacks=[ckpt_callback, LearningRateMonitor()],
        # checkpoint_callback=True,
        fast_dev_run=cfg.MISC.FAST_DEV_RUN,
        default_root_dir=cfg.MISC.OUTPUT_DIR,
        resume_from_checkpoint=cfg.MISC.CHECKPOINT_FILE_PATH,
        **args
    )

    if cfg.TRAIN.TRAIN_ENABLE and cfg.TEST.ENABLE:
        trainer.fit(task)
        return trainer.test()

    elif cfg.TRAIN.TRAIN_ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        if cfg.MISC.CHECKPOINT_FILE_PATH is not None:
            print("Loading from {}".format(cfg.MISC.CHECKPOINT_FILE_PATH))
            task.load_from_checkpoint(cfg.MISC.CHECKPOINT_FILE_PATH)
        return trainer.test(task)


if __name__ == "__main__":
    args = parse_args()
    main(load_config(args))
