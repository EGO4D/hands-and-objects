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

- Create and add a configuration file to `Ego4D_keyframe_localisation/configs`. Refer to sample configuration `Ego4D_keyframe_localisation/configs/ego4d_kf_loc_BMN_can.yaml` provided.
- Refer to `Ego4D_keyframe_localisation/configs/defaults.py` for documentation of all the configuration options.

### Training

- Set attribute MODEL > MODE to Train
- Sample command: 
```
cd Ego4D_keyframe_localisation
python3 bmn_train_test_can.py --cfg /path/to/config/file
```

### Evaluating

- Set attribute MODEL > MODE to inference
- Sample command: 
```
cd Ego4D_keyframe_localisation
python3 bmn_train_test_can.py --cfg /path/to/config/file
```
