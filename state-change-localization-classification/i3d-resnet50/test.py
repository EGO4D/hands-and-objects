
import torch
from tqdm import tqdm
import torch.nn as nn

from datasets import loader
from models.build import build_model
from utils.parser import parse_args, load_config
from evaluation.metrics import state_change_accuracy, keyframe_distance


def main(cfg):
    model = build_model(cfg)
    model = nn.DataParallel(model)
    model.to('cuda:0')
    print(f'Loading pre-trained model: {cfg.MISC.CHECKPOINT_FILE_PATH}')
    try:
        model.load_state_dict(
            torch.load(cfg.MISC.CHECKPOINT_FILE_PATH)['state_dict']
        )
    except RuntimeError:
        print('Loading the model by modifying the keys...')
        # When the model is trained using data parallel class
        state_dict = torch.load(cfg.MISC.CHECKPOINT_FILE_PATH)['state_dict']
        new_state_dict = dict()
        for key, value in state_dict.items():
            new_state_dict[key.replace('model.', 'module.')] = value
        model.load_state_dict(new_state_dict)

    model.eval()
    dataloader = loader.construct_loader(cfg, "test")
    accuracy_list = list()
    keyframe_dist_list = list()

    for batch in tqdm(dataloader):
        frames, labels, state_change_label, fps, info = batch
        assert len(frames) == 1
        frames = [frames[0].to('cuda:1')]
        keyframe_preds, state_change_preds = model.forward(frames)
        accuracy = state_change_accuracy(
            state_change_preds,
            state_change_label
        )
        keyframe_avg_time_dist = keyframe_distance(
            keyframe_preds,
            labels,
            state_change_preds,
            state_change_label,
            fps,
            evaluate_trained=True,
        )
        accuracy_list.append(accuracy)
        if keyframe_avg_time_dist is not None:
            keyframe_dist_list.append(keyframe_avg_time_dist)
        frames = [frames[0].detach().cpu()]
        del frames
    print(
        f'State change accuracy: {sum(accuracy_list)/len(accuracy_list)}; '
        f'Keyframe distance: {sum(keyframe_dist_list)/len(keyframe_dist_list)}'
    )


if __name__ == '__main__':
    main(load_config(parse_args()))
