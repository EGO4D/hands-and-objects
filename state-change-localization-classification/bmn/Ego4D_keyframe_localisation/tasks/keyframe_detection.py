"""
This file contains code for the task of keyframe detection
"""

import torch
from tasks.video_task import VideoTask
from evaluation.metrics import state_change_accuracy, keyframe_distance

from torch.nn import MSELoss
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
        frames, labels, state_change_label = batch
        keyframe_preds, state_change_preds = self.forward(frames)
        keyframe_preds_ = keyframe_preds.permute(0, 2, 1)
        keyframe_loss = self.loss_fun(
            keyframe_preds_.squeeze(),
            torch.argmax(labels.long(), dim=1)
        )
        state_change_loss = self.loss_fun(
            state_change_preds.squeeze(),
            state_change_label.long().squeeze()
        )
        lambda_1 = self.cfg.MODEL.LAMBDA_1
        lambda_2 = self.cfg.MODEL.LAMBDA_2
        loss = torch.mean((state_change_label * keyframe_loss * lambda_2) + \
            (state_change_loss * lambda_1))
        return {
            "keyframe_loss": keyframe_loss,
            "state_change_loss": state_change_loss,
            "train_loss": loss.item(),
            "loss": loss
        }

    def training_epoch_end(self, training_step_outputs):
        keys = [x for x in training_step_outputs[0].keys()]
        for key in keys:
            metric = torch.Tensor.float(training_step_outputs[0][key]).mean()
            self.log(key, metric, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        frames, labels, state_change_label = batch
        keyframe_preds, state_change_preds = self.forward(frames)
        keyframe_loss = self.loss_fun(keyframe_preds, labels.long())
        state_change_loss = self.loss_fun(
            state_change_preds,
            state_change_label.long()
        )
        lambda_1 = self.cfg.MODEL.LAMBDA_1
        lambda_2 = self.cfg.MODEL.LAMBDA_2
        error = torch.mean((state_change_label * keyframe_loss * lambda_2) + \
            (state_change_loss * lambda_1))
        accuracy = state_change_accuracy(state_change_preds, state_change_label)
        keyframe_avg_distance = keyframe_distance(keyframe_preds, labels, state_change_preds, state_change_label)
        return {
            "error": error,
            "state_change_metric": accuracy,
            "keyframe_loc_metric": keyframe_avg_distance
        }

    def validation_epoch_end(self, validation_step_outputs):
        keys = [x for x in validation_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(validation_step_outputs[0][key]).mean()
            self.log(key, metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        frames, labels, state_change_label = batch
        keyframe_preds, state_change_preds = self.forward(frames)
        accuracy = state_change_accuracy(state_change_preds, state_change_label)
        keyframe_avg_distance = keyframe_distance(keyframe_preds, labels, state_change_preds, state_change_label)
        # import pdb
        # pdb.set_trace()
        return {
            "labels": labels,
            "preds": keyframe_preds,
            "state_change_metric": accuracy,
            "keyframe_loc_metric": keyframe_avg_distance
        }

    def test_epoch_end(self, test_step_outputs):
        keys = [x for x in test_step_outputs[0].keys() if "metric" in x]
        for key in keys:
            metric = torch.Tensor.float(test_step_outputs[0][key]).mean()
            self.log(key, metric, prog_bar=True)

class BMN_Action_Proposal(VideoTask):
    checkpoint_metric = "keyframe_loc_metric"   
    def training_step(self, batch, batch_idx):
        frames, labels, prec_labels, state_change_label, label_confidence_score, label_match_score_start, label_match_score_end = batch
        input_data = [frames[0].cuda()]
        label_start = label_match_score_start.cuda()
        label_end = label_match_score_end.cuda()
        label_confidence = label_confidence_score.cuda()
        confidence_map, start, end = self.forward(input_data)
        loss = self.loss_fun(confidence_map, start, end, label_confidence, label_start, label_end, self.bm_mask)
        return {
        'pemreg_loss': loss[2].cpu().detach().numpy(),
        'pemclr_loss': loss[3].cpu().detach().numpy(),
        'tem_loss': loss[1].cpu().detach().numpy(),
        'loss': loss[0].cpu().detach().numpy()
        }

    def training_epoch_end(self, training_step_outputs):
        n_iter = len(training_step_outputs)
        pemreg_loss = 0
        pemclr_loss = 0
        tem_loss = 0
        loss = 0
        for iteration in training_step_outputs:
            for key in iteration.keys():
                if key == 'pemreg_loss':
                    pemreg_loss+=iteration[key]
                elif key == 'pemclr_loss':
                    pemclr_loss+=iteration[key]
                elif key == 'tem_loss':
                    tem_loss+=iteration[key]
                elif key == 'loss':
                    loss+=iteration[key]
        self.log('pemreg_loss', pemreg_loss/n_iter, prog_bar=True)
        self.log('pemreg_loss', pemclr_loss/n_iter, prog_bar=True)
        self.log('tem_loss', tem_loss/n_iter, prog_bar=True)
        self.log('loss', loss/n_iter, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        frames, labels, prec_labels, state_change_label, label_confidence_score, label_match_score_start, label_match_score_end = batch
        input_data = [frames[0].cuda()]
        label_start = label_match_score_start.cuda()
        label_end = label_match_score_end.cuda()
        label_confidence = label_confidence_score.cuda()
        confidence_map, start, end = self.forward(input_data)
        loss = self.loss_fun(confidence_map, start, end, label_confidence, label_start, label_end, self.bm_mask)
        return {
        'pemreg_loss': loss[2].cpu().detach().numpy(),
        'pemclr_loss': loss[3].cpu().detach().numpy(),
        'tem_loss': loss[1].cpu().detach().numpy(),
        'loss': loss[0].cpu().detach().numpy()
        }

    def validation_epoch_end(self, validation_step_outputs):
        n_iter = len(training_step_outputs)
        pemreg_loss = 0
        pemclr_loss = 0
        tem_loss = 0
        loss = 0
        for iteration in training_step_outputs:
            for key in iteration.keys():
                if key == 'pemreg_loss':
                    pemreg_loss+=iteration[key]
                elif key == 'pemclr_loss':
                    pemclr_loss+=iteration[key]
                elif key == 'tem_loss':
                    tem_loss+=iteration[key]
                elif key == 'loss':
                    loss+=iteration[key]
        self.log('pemreg_loss', pemreg_loss/n_iter, prog_bar=True)
        self.log('pemreg_loss', pemclr_loss/n_iter, prog_bar=True)
        self.log('tem_loss', tem_loss/n_iter, prog_bar=True)
        self.log('loss', loss/n_iter, prog_bar=True)

    def test_step(self, batch, batch_idx):
        frames, labels, prec_labels, state_change_label, label_confidence_score, label_match_score_start, label_match_score_end = batch
        input_data = [frames[0].cuda()]
        label_start = label_matchh_score_start.cuda()
        label_end = label_match_score_end.cuda()
        label_confidence = label_confidence_score.cuda()
        confidence_map, start, end = self.forward(input_data)
        start_scores = start[0].detach().cpu().numpy()
        end_scores = end[0].detach().cpu().numpy()
        clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
        reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()
        new_props = []
        for idx in range(tscale):
            for jdx in range(tscale):
                start_index = idx
                end_index = jdx + 1
                if start_index < end_index and  end_index<tscale :
                    xmin = start_index / tscale
                    xmax = end_index / tscale
                    xmin_score = start_scores[start_index]
                    xmax_score = end_scores[end_index]
                    clr_score = clr_confidence[idx, jdx]
                    reg_score = reg_confidence[idx, jdx]
                    score = xmin_score * xmax_score * clr_score * reg_score
                    new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
        new_props = np.stack(new_props)
        col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)
        return None

    def test_epoch_end(self, test_step_outputs):
        pass
