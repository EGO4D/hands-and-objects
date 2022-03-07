import torch


def keyframe_distance_seconds(scores, info):
    """
    This function takes the scores for clips with PNR label and corresponding info dict returned by the dataloader
    and computes the keyframe distance.
    Args:
        scores: The scores returned by the model
        info: An info dictionary returned by the dataloader containing the ground truth

    Returns: The sum of errors (in seconds) for the given batch

    """
    distance_list = list()
    sec_list = list()
    for i, pred in enumerate(scores):
        clip_length = info['json_parent_end_sec'][i].item() - info['json_parent_start_sec'][i].item()
        clip_frames = info['json_parent_end_frame'][i].item() - info['json_parent_start_frame'][i].item() + 1
        fps = clip_frames / clip_length
        keyframe_loc_pred = torch.argmax(pred)
        keyframe_loc_pred_mapped = (info['json_parent_end_frame'][i].item() - info[
            'json_parent_start_frame'][i].item()) / 16 * keyframe_loc_pred
        keyframe_loc_gt = info['pnr_frame'][i].item() - info['json_parent_start_frame'][i].item()
        err_frame = abs(keyframe_loc_pred_mapped - keyframe_loc_gt)
        err_sec = err_frame / fps
        distance_list.append(err_frame.item())
        sec_list.append(err_sec.item())

    # When there is no false positive
    if len(distance_list) == 0:
        # Should we return something else here?
        return 0

    return sum(sec_list)  # Return the sum for later averaging at the end of the epoch


def keyframe_distance_seconds_alternate(scores, info):
    """
    This function takes the scores for clips with PNR label and corresponding info dict returned by the dataloader
    and computes the keyframe distance.
    Args:
        scores: The scores returned by the model
        info: An info dictionary returned by the dataloader containing the ground truth

    Returns: The sum of errors (in seconds) for the given batch

    """
    clip_length = info['json_parent_end_sec'] - info['json_parent_start_sec']
    clip_frames = info['json_parent_end_frame'] - info['json_parent_start_frame'] + 1
    fps = clip_frames / clip_length
    keyframe_loc_pred = torch.argmax(scores, dim=1)
    keyframe_loc_pred_mapped = (info['json_parent_end_frame'] - info[
        'json_parent_start_frame']) / 16 * keyframe_loc_pred
    keyframe_loc_gt = info['pnr_frame'] - info['json_parent_start_frame']
    err_frame = abs(keyframe_loc_pred_mapped - keyframe_loc_gt)
    err_sec = err_frame / fps
    return err_sec.sum()
