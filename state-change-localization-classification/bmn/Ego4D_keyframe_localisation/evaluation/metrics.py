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


def keyframe_distance(preds, labels, sc_preds, sc_labels):
    distance_list = list()
    for pred, label, sc_pred, sc_label in zip(preds, labels, sc_preds, sc_labels):
        if torch.argmax(sc_pred).item() == 1 and sc_label.item() == 1:
            # Selecting the row with 
            keyframe_loc_pred = torch.argmax(pred)
            keyframe_loc_gt = torch.argmax(label)
            distance = torch.abs(keyframe_loc_gt - keyframe_loc_pred)
            distance_list.append(distance.item())
    # When there is no false positive
    if len(distance_list) == 0:
        # Should we return something else here?
        return 0
    return np.mean(distance_list)
