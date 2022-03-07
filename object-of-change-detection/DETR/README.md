
## Ego4D Object of Change Detection Basline - DETR

### Environment
- Set up detectron2 environment as in [install.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
- Modify the PATH_TO_DATA_FOLDER in `train_net.py`


### Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py --num-gpus 8 --config detr_256_6_6_torchvision_ego4dv1.yaml
```


### Evaluation
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py --num-gpus 8 --config detr_256_6_6_torchvision_ego4dv1.yaml --eval-only MODEL.WEIGHTS <model_path>
```