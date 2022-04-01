
## Ego4D Object of Change Detection Basline - CenterNet(Zhou's)

### Environment
- Set up CenterNet environment following the official github repository as in [CenterNet](https://github.com/xingyizhou/CenterNet)
- Creat a soft link of your data folder by running `ln -s /PATH_TO_DATA_FOLDER data/ego4dv1`
- We already modified the config and code for you to run on Ego4D dataset



### Train
```
python main.py ctdet --exp_id ego4dv1_dla_1x --batch_size 128 --master_batch 9 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16
```


### Evaluation
```
python test.py ctdet --exp_id ego4dv1_dla_1x --keep_res --resume
```
Or with a custom weight
```
python test.py ctdet --exp_id ego4dv1_dla_eval --keep_res --load_model <model_path>
```