"""
This file contains code for the task of keyframe detection
"""

import torch
from tasks.video_task import VideoTask
from evaluation.metrics import state_change_accuracy, keyframe_distance
import time
from torch.nn import MSELoss
import pandas as pd

mse = MSELoss(reduction='mean')

class KeyframeLocalisation(VideoTask):
    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label = batch
        preds = self.forward(frames)
        loss = self.loss_fun(preds, labels.long())
        # TODO: Add metric
        return {
            "loss": loss,
            "train_loss": loss.item()
        }

    def validation_step(self, batch, batch_idx):
        frames, labels, state_change_label = batch
        pred = self.forward(frames)
        error = self.loss_fun(pred, labels.long())
        # TODO: Add metric
        return {
            "error": error
        }

    def test_step(self, batch, batch_idx):
        frames, labels, state_change_label = batch
        preds = self.forward(frames)
        #TODO: Add metric
        return {
            "labels": labels,
            "preds": preds
        }


class StateChangeAndKeyframeLocalisation(VideoTask):
    checkpoint_metric = "keyframe_loc_metric"

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps = batch
        keyframe_preds, state_change_preds = self.forward(frames)
        # keyframe_preds_ = keyframe_preds.permute(0, 2, 1)
        keyframe_loss = self.loss_fun(
            keyframe_preds,
            torch.argmax(labels.long(), dim=1)
        )
        # We want to calculate the keyframe loss only for the clips with state
        # change
        keyframe_loss = torch.mean(state_change_label.T * keyframe_loss)
        # state_change_loss = self.loss_fun(
        #     state_change_preds.squeeze(2),
        #     state_change_label.long().squeeze(1)
        # )

        state_change_loss = torch.mean(self.loss_fun(
            state_change_preds,
            state_change_label.long()#.squeeze(1)
        ))
        lambda_1 = self.cfg.MODEL.LAMBDA_1
        lambda_2 = self.cfg.MODEL.LAMBDA_2
        loss = (keyframe_loss * lambda_2) + (state_change_loss * lambda_1)
        return {
            "keyframe_loss": keyframe_loss,
            "state_change_loss": state_change_loss,
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
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps = batch
        keyframe_preds, state_change_preds = self.forward(frames)
        keyframe_loss = self.loss_fun(
            keyframe_preds,
            torch.argmax(labels.long(), dim=1)
        )
        # We want to calculate the keyframe loss only for the clips with state
        # change
        keyframe_loss = torch.mean(state_change_label.T * keyframe_loss)
        # state_change_loss = self.loss_fun(
        #     state_change_preds.squeeze(2),
        #     state_change_label.long().squeeze(1)
        # )

        state_change_loss = torch.mean(self.loss_fun(
            state_change_preds,
            state_change_label.long()#.squeeze(1)
        ))
        lambda_1 = self.cfg.MODEL.LAMBDA_1
        lambda_2 = self.cfg.MODEL.LAMBDA_2
        error = torch.mean((state_change_label.T * keyframe_loss * lambda_2) \
            + (state_change_loss * lambda_1))
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        keyframe_avg_distance = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_preds,
            state_change_label
        )
        return {
            "error": error,
            "state_change_metric": accuracy,
            "keyframe_loc_metric": keyframe_avg_distance
        }

    def validation_epoch_end(self, validation_step_outputs):
        keys = [x for x in validation_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].item() for item in validation_step_outputs]
                )
            ).mean()
            self.log(key, metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, ids = batch
        keyframe_preds, state_change_preds = self.forward(frames)
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        batch_size = frames[0].shape[0]
        for i in range(batch_size):
            kf = keyframe_preds[i].argmax()
            lbl = labels[i].argmax()
            self.test_results.append([ids[i], lbl.item(), kf.item(), state_change_label[i].item()])
        keyframe_avg_distance = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_preds,
            state_change_label
        )
        return {
            "labels": labels,
            "preds": keyframe_preds,
            "state_change_metric": accuracy,
            "keyframe_loc_metric": keyframe_avg_distance
        }

    def test_epoch_end(self, test_step_outputs):
        print("Test finished!")
        df = pd.DataFrame(self.test_results, columns=['id', 'label', 'pred', 'sc'])
        df.to_csv(f'Test_reults_kf_{time.strftime("%a, %d %b %Y %H:%M:%S")}')

        keys = [x for x in test_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].item() for item in test_step_outputs]
                )
            ).mean()
            self.log(key, metric, prog_bar=True)


class StateChangeDetection(VideoTask):
    checkpoint_metric = "state_change_metric"

    def training_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, lengths = batch
        state_change_preds = self.forward(frames)

        state_change_loss = torch.mean(self.loss_fun(
            state_change_preds,
            state_change_label.long()#.squeeze(1)
        ))
        lambda_1 = self.cfg.MODEL.LAMBDA_1
        loss = (state_change_loss * lambda_1)
        return {
            "state_change_loss": state_change_loss,
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
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps = batch
        state_change_preds = self.forward(frames)
        state_change_loss = torch.mean(self.loss_fun(
            state_change_preds,
            state_change_label.long()#.squeeze(1)
        ))
        lambda_1 = self.cfg.MODEL.LAMBDA_1
        error = torch.mean((state_change_loss * lambda_1))
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        return {
            "error": error,
            "state_change_metric": accuracy,
        }

    def validation_epoch_end(self, validation_step_outputs):
        keys = [x for x in validation_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].item() for item in validation_step_outputs]
                )
            ).mean()
            self.log(key, metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        state_change_preds = self.forward(frames)
        info
        for pred, label, inf in zip(state_change_preds, state_change_label, info):
            pred_ = torch.argmax(pred)
            # idx, inf = info
            if pred_.item() != label.item():
                # self.test_set[idx] = frames.numpy()
                with open('test_results.txt', 'a') as f:
                    f.write(f"{inf} {pred_} {label}\n")


        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        return {
            "labels": labels,
            "state_change_metric": accuracy,
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

class FrameSelection(VideoTask):
    checkpoint_metric = "state_change_metric"

    def training_step(self, batch, batch_idx):
        frames, labels, frame_dist_label, fps, lengths = batch
        frame_dist_preds = self.forward(frames, lengths)

        frame_dist_loss = torch.mean(self.loss_fun(
            frame_dist_preds,
            frame_dist_label#.squeeze(1)
        ))
        return {
            # "state_change_loss": state_change_loss,
            # "train_loss": loss,
            "loss": frame_dist_loss
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
        frames, labels, state_change_label, fps = batch
        state_change_preds = self.forward(frames)
        state_change_loss = torch.mean(self.loss_fun(
            state_change_preds,
            state_change_label.long()#.squeeze(1)
        ))
        lambda_1 = self.cfg.MODEL.LAMBDA_1
        error = torch.mean((state_change_loss * lambda_1))
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        return {
            "error": error,
            "state_change_metric": accuracy,
        }

    def validation_epoch_end(self, validation_step_outputs):
        keys = [x for x in validation_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(
                torch.Tensor(
                    [item[key].item() for item in validation_step_outputs]
                )
            ).mean()
            self.log(key, metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        frames, labels, state_change_label, fps, info = batch
        state_change_preds = self.forward(frames)
        info
        for pred, label, inf in zip(state_change_preds, state_change_label, info):
            pred_ = torch.argmax(pred)
            # idx, inf = info
            if pred_.item() != label.item():
                # self.test_set[idx] = frames.numpy()
                with open('test_results.txt', 'a') as f:
                    f.write(f"{inf} {pred_} {label}\n")


        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        return {
            "labels": labels,
            "state_change_metric": accuracy,
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
