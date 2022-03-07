# Ego4D State Change Detection and Keyframe Localization Baseline

## Setup

Follow the instructions in the Ego4D repository for installing some of the required libraries.
For the rest, use the `requirements.txt` provided. Use the following command:

```
pip install -r requirements.txt
```

### Data Preparation

- Download all the required videos to a directory, for example `/path/to/videos`.
- Create a directory to extract and save the frames for the clips with state change, for example `/path/to/positive_clips`.
- Create a directory to extract and save the frames for the clips without state change, for example `/path/to/negative_clips`.
- Save the json files containing the annotations to a directory, for example `/path/to/annotations`.

## Experiments

- Create and add a configuration file to `configs`. Refer to sample configuration `configs/2021-09-18_keyframe_loc_release1-v2_main-experiment.yaml` provided.
- Refer to `configs/defaults.py` for documentation of all the configuration options.

### Training

- Use `train.py` for training the I3D ResNet-50 model.
- Sample command: 
```
python -m train --cfg configs/2021-09-18_keyframe_loc_release1-v2_main-experiment.yaml DATA.VIDEO_DIR_PATH /path/to/videos DATA.CLIPS_SAVE_PATH /path/to/positive_clips DATA.NO_SC_PATH /path/to/negative/clips DATA.DATA.ANN_DIR /path/to/annotations
```

### Evaluating

- Use `test.py` for testing the trained model (for example `/path/to/trained_model.pt`).
- Sample command: 
```
python -m test --cfg configs/2021-09-18_keyframe_loc_release1-v2_main-experiment.yaml MISC.CHECKPOINT_FILE_PATH /path/to/trained_model.pt DATA.VIDEO_DIR_PATH /path/to/videos DATA.CLIPS_SAVE_PATH /path/to/positive_clips DATA.NO_SC_PATH /path/to/negative/clips DATA.DATA.ANN_DIR /path/to/annotations
```
