"""
This file contains code for the task of keyframe detection
"""

import torch
from torch.nn import MSELoss

from tasks.video_task import VideoTask
from evaluation.metrics import state_change_accuracy, keyframe_distance

mse = MSELoss(reduction='mean')


class StateChangeAndKeyframeLocalisation(VideoTask):
    checkpoint_metric = "state_change_metric_val"

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        keyframe_preds, state_change_preds = self.forward(frames)
        keyframe_preds_ = keyframe_preds.permute(0, 2, 1)

        keyframe_loss = self.loss_fun(
            keyframe_preds_.squeeze(dim=-1),
            torch.argmax(labels.long(), dim=1)
        )
        # We want to calculate the keyframe loss only for the clips with state
        # change
        keyframe_loss = torch.mean(state_change_label.T * keyframe_loss)
        state_change_loss = torch.mean(self.loss_fun(
            state_change_preds.squeeze(),
            state_change_label.long().squeeze()
        ))

        lambda_1 = self.cfg.MODEL.LAMBDA_1
        lambda_2 = self.cfg.MODEL.LAMBDA_2
        loss = (keyframe_loss * lambda_2) + (state_change_loss * lambda_1)
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_preds,
            state_change_label,
            fps
        )

        return {
            "keyframe_loss": keyframe_loss,
            "state_change_loss": state_change_loss,
            "train_loss": loss,
            "loss": loss,
            "state_change_metric_train": accuracy,
            "keyframe_loc_metric_time_dist_train": keyframe_avg_time_dist
        }

    def training_epoch_end(self, training_step_outputs):
        keys = [x for x in training_step_outputs[0].keys()]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].mean() for item in training_step_outputs]
                )
            ).mean()
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        keyframe_preds, state_change_preds = self.forward(frames)

        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_preds,
            state_change_label,
            fps
        )

        return {
            "state_change_metric_val": accuracy,
            "keyframe_loc_metric_time_dist_val": keyframe_avg_time_dist
        }

    def validation_epoch_end(self, validation_step_outputs):
        keys = [x for x in validation_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].item() for item in validation_step_outputs]
                )
            ).mean()
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        keyframe_preds, state_change_preds = self.forward(frames)
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_preds,
            state_change_label,
            fps
        )

        return {
            "labels": labels,
            "preds": keyframe_preds,
            "state_change_metric": accuracy,
            "keyframe_loc_metric_time": keyframe_avg_time_dist
        }

    def test_epoch_end(self, test_step_outputs):
        keys = [x for x in test_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].item() for item in test_step_outputs]
                )
            ).mean()
            self.log(key, metric, prog_bar=True)
