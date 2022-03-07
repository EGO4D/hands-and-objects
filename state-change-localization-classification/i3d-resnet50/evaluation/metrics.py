import torch
import numpy as np


def state_change_accuracy(preds, labels):
    correct = 0
    total = 0
    for pred, label in zip(preds, labels):
        pred_ = torch.argmax(pred)
        if pred_.item() == label.item():
            correct += 1
        total += 1
    accuracy = correct/total
    return accuracy


def keyframe_distance(
    preds,
    labels,
    sc_preds,
    sc_labels,
    fps,
    info,
    evaluate_trained=False
):
    distance_list = list()
    for pred, label, sc_pred, sc_label, ind_info, ind_fps in zip(
        preds,
        labels,
        sc_preds,
        sc_labels,
        info,
        fps
    ):
        if sc_label.item() == 1:
            keyframe_loc_pred = torch.argmax(pred).item()
            keyframe_loc_pred_mapped = (
                ind_info['clip_end_frame'] - ind_info['clip_start_frame']
            ) / 16 * keyframe_loc_pred
            keyframe_loc_pred_mapped = keyframe_loc_pred_mapped.item()
            gt = ind_info['pnr_frame'].item() - ind_info['clip_start_frame'].item()
            err_frame = abs(keyframe_loc_pred_mapped - gt)
            err_sec = err_frame/fps.item()
            distance_list.append(err_sec)
    if len(distance_list) == 0:
        # If evaluating the trained model, use this
        if evaluate_trained:
            return None
        # Otherwise, Lightning expects us to give a number.
        # Due to this, the Tensorboard graphs' results for keyframe distance
        # will be a little inaccurate.
        return np.mean(0.0)
    return np.mean(distance_list)
