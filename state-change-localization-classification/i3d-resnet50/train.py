
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


import utils.logging as logging
from utils.parser import parse_args, load_config
from tasks.keyframe_detection import StateChangeAndKeyframeLocalisation


logger = logging.get_logger(__name__)


def main(cfg):
    if cfg.DATA.TASK == "state_change_detection_and_keyframe_localization":
        TaskType = StateChangeAndKeyframeLocalisation
    else:
        raise NotImplementedError('Task {} not implemented.'.format(
            cfg.DATA.TASK
        ))

    task = TaskType(cfg)

    if cfg.MISC.ENABLE_LOGGING:
        args = {
            'callbacks': [LearningRateMonitor()]
        }
    else:
        args = {
            'logger': False
        }

    trainer = Trainer(
        gpus=cfg.MISC.NUM_GPUS,
        num_nodes=cfg.MISC.NUM_SHARDS,
        accelerator=cfg.SOLVER.ACCELERATOR,
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=0,
        benchmark=True,
        replace_sampler_ddp=False,
        checkpoint_callback=ModelCheckpoint(
            monitor=task.checkpoint_metric,
            mode="max",
            save_last=True,
            save_top_k=3,
        ),
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
        return trainer.test(task)


if __name__ == "__main__":
    args = parse_args()
    main(load_config(args))
