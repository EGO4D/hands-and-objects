"""
This file contains code for the task of keyframe detection
"""

import time

import pandas as pd
import torch
import torch.nn.functional
from torch.nn import MSELoss

from evaluation.metrics import keyframe_distance_seconds
from tasks.video_task import VideoTask

mse = MSELoss(reduction='mean')


class StateChangeAndKeyframeLocalisation(VideoTask):
    checkpoint_metric = "keyframe_loc_metric"
    train_results = []
    test_results = []
    val_results = []

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps = batch
        keyframe_preds, = self.forward(frames)
        keyframe_loss = self.loss_fun(
            keyframe_preds,
            torch.argmax(labels.long(), dim=1)
        )
        pds = keyframe_preds.argmax(dim=1)
        batch = frames[0].shape[0]
        StateChangeAndKeyframeLocalisation.train_results.extend(pds.cpu().numpy().tolist())
        # We want to calculate the keyframe loss only for the clips with state
        # change
        keyframe_loss = torch.mean(state_change_label.T * keyframe_loss)
        lambda_1 = self.cfg.MODEL.LAMBDA_1
        loss = (keyframe_loss * lambda_1)
        self.log('loss', loss, prog_bar=True)

        return {
            "keyframe_loss": keyframe_loss,
            "train_loss": loss,
            "loss": loss
        }

    def training_epoch_end(self, training_step_outputs):
        keys = [x for x in training_step_outputs[0].keys()]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].mean() for item in training_step_outputs]
                )
            ).mean()
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch

        keyframe_preds, = self.forward(frames)
        keyframe_loss = self.loss_fun(
            keyframe_preds,
            torch.argmax(labels.long(), dim=1)
        )

        batch_size = frames[0].shape[0]
        for i in range(batch_size):
            kf = keyframe_preds[i].argmax()
            lbl = labels[i].argmax()
            StateChangeAndKeyframeLocalisation.val_results.append([info['unique_id'][i], lbl.item(), kf.item(),
                                                                   state_change_label[i].item(), ])
        keyframe_sum_distance = keyframe_distance_seconds(
            keyframe_preds,
            info,
        )

        state_change_count = state_change_label.sum()
        count = batch_size
        return {
            "keyframe_loc_metric": torch.Tensor([keyframe_sum_distance]),
            "state_change_count": torch.Tensor([state_change_count]),
            "count": torch.Tensor([count]),
            # "rank": torch.Tensor([self.global_rank])
        }

    def validation_epoch_end(self, validation_step_outputs):
        # Fetch all outputs from all processes
        all_outs = self.all_gather(validation_step_outputs)

        state_change_count = sum([i['state_change_count'].sum() for i in all_outs])
        keyframe_distance_sum = sum([i['keyframe_loc_metric'].sum() for i in all_outs])
        keyframe_loc_metric = keyframe_distance_sum / state_change_count
        val_results = StateChangeAndKeyframeLocalisation.val_results  # self.all_gather(self.test_results)

        # Log/write files on the rank zero process only
        if self.trainer.is_global_zero:
            self.log("state_change_count", state_change_count, rank_zero_only=True)
            self.log("keyframe_loc_metric", keyframe_loc_metric, rank_zero_only=True)
            df = pd.DataFrame(val_results, columns=['id', 'label', 'pred', 'sc', ])
            df.to_csv(f'Val_reults_kf_{time.strftime("%a, %d %b %Y %H:%M:%S")}.csv'.replace(' ', '_'))

    def test_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        keyframe_preds, = self.forward(frames)

        batch_size = frames[0].shape[0]
        for i in range(batch_size):
            kf = keyframe_preds[i].argmax()
            lbl = labels[i].argmax()
            StateChangeAndKeyframeLocalisation.test_results.append([info['unique_id'][i], lbl.item(), kf.item(),
                                                                    state_change_label[i].item(), ])
        keyframe_sum_distance = keyframe_distance_seconds(
            keyframe_preds,
            info,
        )
        state_change_count = state_change_label.sum()
        count = frames[0].shape[0]
        return {
            "labels": labels,
            "preds": keyframe_preds,
            "keyframe_loc_metric": torch.Tensor([keyframe_sum_distance]),
            "state_change_count": torch.Tensor([state_change_count]),
            "count": torch.Tensor([count]),
        }

    def test_epoch_end(self, test_step_outputs):
        all_outs = self.all_gather(test_step_outputs)
        state_change_count = sum([i['state_change_count'].sum() for i in all_outs])
        keyframe_distance_sum = sum([i['keyframe_loc_metric'].sum() for i in all_outs])
        keyframe_loc_metric = keyframe_distance_sum / state_change_count
        test_results = StateChangeAndKeyframeLocalisation.test_results

        # Log/write files on the rank zero process only
        if self.trainer.is_global_zero:
            self.log("keyframe_loc_metric", keyframe_loc_metric, rank_zero_only=True)
            self.log("state_change_count", state_change_count, rank_zero_only=True)
            print("Test finished!")
            df = pd.DataFrame(test_results, columns=['id', 'label', 'pred', 'sc', ])
            df.to_csv(f'Test_reults_kf_{time.strftime("%a, %d %b %Y %H:%M:%S")}.csv'.replace(' ', '_'))
