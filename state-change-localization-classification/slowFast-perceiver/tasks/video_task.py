from pytorch_lightning.core import LightningModule
from sklearn.metrics import accuracy_score, average_precision_score
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

import models.losses as losses
import optimizers.optimizer as optim
import utils.distributed as du
from datasets import loader
from models.build import build_model


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

        kwargs = dict()
        # if cfg.MODEL.CLASS_WEIGHTS is not None:
        #     kwargs['weight'] = torch.Tensor(cfg.MODEL.CLASS_WEIGHTS)

        self.loss_fun = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(
            reduction=cfg.MODEL.LOSS_REDUCTION, **kwargs
        )
        self.state_change_metric = accuracy_score
        self.keyframe_loc_metric = average_precision_score
        self.test_results = []

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

    def forward(self, inputs, *args):
        return self.model(inputs, *args)

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
            # if self.cfg.MISC.TEST_TRAIN_CODE:
            #     self.val_loader = self.train_loader
            # else:
        self.val_loader = loader.construct_loader(self.cfg, "val")
        if self.cfg.TEST.ENABLE:
                self.test_loader = loader.construct_loader(self.cfg, "test")

    def configure_optimizers(self):
        optimizer = optim.construct_optimizer(self.model, self.cfg)
        for g in optimizer.param_groups:
            print("Learning rate is", g['lr'])

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

            slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=False)

        scheduler = {"scheduler": slow_fast_scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    # def on_load_checkpoint(self, checkpoint):
    #     self.cfg.SOLVER.STEPS = [0, 15, 40, 80]
