
## Ego4D Object of Change Detection Basline - CenterNet(Duan's)

### Environment
- Set up CenterNet environment following the official github repository as in [CenterNet](https://github.com/Duankaiwen/CenterNet)
- Creat a soft link of your data folder by running `ln -s /PATH_TO_DATA_FOLDER data/ego4dv1`
- We already modified the config and code for you to run on Ego4D dataset



### Train
```
python train.py CenterNet-52
```


### Evaluation
```
python test.py CenterNet-52 --testiter 480000 --split <split>
```