from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from tqdm import tqdm

import utils.logging as logging
from tasks.keyframe_detection import StateChangeAndKeyframeLocalisation
from utils.parser import parse_args, load_config

logger = logging.get_logger(__name__)


class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=False,
        )
        return bar


# Set a Global seed
seed_everything(42, workers=True)


def main(cfg):
    print("Training on ", cfg.DATA.ANN_PATH)
    cfg.DATA.NUM_FRAMES = cfg.DATA.SAMPLING_FPS * cfg.DATA.CLIP_LEN_SEC

    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=cfg.MODEL.MODEL_NAME)

    checkpoint_mode = "min"
    if cfg.DATA.TASK == "state_change_detection_and_keyframe_localization":
        TaskType = StateChangeAndKeyframeLocalisation
    else:
        raise NotImplementedError('Task {} not implemented'.format(
            cfg.DATA.TASK
        ))

    task = TaskType(cfg)

    if cfg.SOLVER.ACCUMULATE_GRAD_BATCHES is not None:
        args['accumulate_grad_batches'] = dict(zip(*cfg.SOLVER.ACCUMULATE_GRAD_BATCHES))

    ckpt_callback = ModelCheckpoint(
        dirpath='/home/abrsh/canonical_dataset/benchmark/checkpoints',
        monitor=task.checkpoint_metric,
        mode=checkpoint_mode,
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_val_epochs=1,
        verbose=True,
        save_top_k=-1,
        filename=f"{cfg.MODEL.MODEL_NAME}" + "-{epoch}-{val_loss:2f}-{keyframe_loc_metric}-{state_change_metric}"
    )

    plugins = None
    if cfg.SOLVER.ACCELERATOR == 'ddp':
        plugins = DDPPlugin(find_unused_parameters=False)

    trainer = Trainer(
        gpus=cfg.MISC.NUM_GPUS,
        num_nodes=cfg.MISC.NUM_SHARDS,
        accelerator=cfg.SOLVER.ACCELERATOR,
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=0,
        benchmark=True,
        check_val_every_n_epoch=20,
        replace_sampler_ddp=False,
        checkpoint_callback=True,
        callbacks=[ckpt_callback, LearningRateMonitor()],
        # checkpoint_callback=True,
        fast_dev_run=cfg.MISC.FAST_DEV_RUN,
        default_root_dir=cfg.MISC.OUTPUT_DIR,
        resume_from_checkpoint=cfg.MISC.CHECKPOINT_FILE_PATH,
        plugins=plugins,
        logger=[tb_logger, ],
        log_every_n_steps=cfg.MISC.LOG_FREQUENCY,
        # weights_summary='full',
    )

    if cfg.TRAIN.TRAIN_ENABLE and cfg.TEST.ENABLE and not cfg.TRAIN.VAL_ONLY:
        trainer.fit(task)
        if cfg.MISC.FAST_DEV_RUN:
            return trainer.test(ckpt_path="/home/abrsh/canonical_dataset/benchmark/checkpoints/MultiTaskSlowFastPr-epoch=19-val_loss=0.000000-keyframe_loc_metric=1.6130026578903198-state_change_metric=0.ckpt")
        else:
            return trainer.test()
    elif cfg.TRAIN.VAL_ONLY:
        if cfg.MISC.CHECKPOINT_FILE_PATH is not None:
            print("Loading from {}".format(cfg.MISC.CHECKPOINT_FILE_PATH))
            task = task.load_from_checkpoint(cfg.MISC.CHECKPOINT_FILE_PATH)
            task.cfg = cfg
        task.eval()
        return trainer.validate(task)

    elif cfg.TRAIN.TRAIN_ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        if cfg.MISC.CHECKPOINT_FILE_PATH is not None:
            print("Loading from {}".format(cfg.MISC.CHECKPOINT_FILE_PATH))
            task = task.load_from_checkpoint(cfg.MISC.CHECKPOINT_FILE_PATH)
            task.cfg = cfg
        task.eval()
        return trainer.test(task)


if __name__ == "__main__":
    args = parse_args()
    main(load_config(args))
