from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

import models.losses as losses
import optimizers.optimizer as optim
from datasets import loader
from models.build import build_model

from pytorch_lightning.core import LightningModule
from sklearn.metrics import accuracy_score, average_precision_score
from loss.loss_function import bmn_loss_func, get_mask

class VideoTask(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Backwards compatibility.
        if isinstance(cfg.MODEL.NUM_CLASSES, int):
            cfg.MODEL.NUM_CLASSES = [cfg.MODEL.NUM_CLASSES]

        if not hasattr(cfg.TEST, "NO_ACT"):
            cfg.TEST.NO_ACT = False

        self.cfg = cfg
        self.save_hyperparameters()
        self.model = build_model(cfg)
        if self.cfg.MODEL.LOSS_FUNC == "bmn_loss":
            self.loss_fun = bmn_loss_func
            self.bm_mask = get_mask(cfg.BMN.TEMPORAL_SCALE).cuda()
        else:
            self.loss_fun = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(
                reduction=cfg.MODEL.LOSS_REDUCTION
            )
        self.state_change_metric = accuracy_score
        self.keyframe_loc_metric = average_precision_score

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step_end(self, training_step_outputs):
        if self.cfg.SOLVER.ACCELERATOR == 'dp':
            training_step_outputs['loss'] = training_step_outputs['loss'].mean()
        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, inputs):
        return self.model(inputs)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def setup(self, stage):
        # Setup is called immediately after the distributed processes have been
        # registered. We can now setup the distributed process groups for each machine
        # and create the distributed data loaders.
        if self.cfg.SOLVER.ACCELERATOR != 'dp':
            du.init_distributed_groups(self.cfg)

        if self.cfg.TRAIN.TRAIN_ENABLE:
            self.train_loader = loader.construct_loader(self.cfg, "train")
            if self.cfg.MISC.TEST_TRAIN_CODE:
                self.val_loader = self.train_loader
            else:
                self.val_loader = loader.construct_loader(self.cfg, "val")
        if self.cfg.TEST.ENABLE:
            if self.cfg.MISC.TEST_TRAIN_CODE:
                self.test_loader = self.train_loader
            else:
                self.test_loader = loader.construct_loader(self.cfg, "test")

    def configure_optimizers(self):
        optimizer = optim.construct_optimizer(self.model, self.cfg)
        steps_in_epoch = len(self.train_loader)
        if self.cfg.SOLVER.LR_POLICY == "cosine":
            slow_fast_scheduler = CosineAnnealingLR(
                optimizer, self.cfg.SOLVER.MAX_EPOCH * steps_in_epoch, last_epoch=-1
            )
        elif self.cfg.SOLVER.LR_POLICY == "constant":
            slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)
        else:

            def lr_lambda(step):
                return optim.get_epoch_lr(step / steps_in_epoch, self.cfg)

            slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        scheduler = {"scheduler": slow_fast_scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
