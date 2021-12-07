# Slowfast + Perceiver Ego4D Keyframe Localization
Point-of-No-Return (PNR) localization using SlowFast with Perceiver backbone.

To reproduce the results: <br/>
Open the file data_loader_test_canonical_form_25-08-21.yaml
Then, modify the following entries with the appropriate paths: <br/><br/>
Under the DATA category <br/>
Change <br/>
VIDEO_DIR_PATH: to the path to the Ego4D version 1 videos <br/>
ANN_PATH: to the path to your annotation file<br/>
SPLIT_PATH: to the path to the folder with split files<br/>
CLIPS_SAVE_PATH: to the path to the folder that contains frames extracted from Ego4D for videos with state change<br/>
VIDEO_LOCATIONS_CSV: to the path to a CSV file with paths to videos<br/>
NO_SC_PATH: to the path to the folder that contains frames extracted from Ego4D for videos without state change<br/>
NO_SC_SPLIT_PATH: to the path to JSON file containing the splits for videos without state change<br/><br/>

Under the TEST category<br/>
Change<br/>
JSON: to the path to the JSON file that contains the details of videos to be used as test set<br/>
VAL_JSON: to the path to the JSON file that contains the details of videos to be used as test set<br/><br/>

Under MISC<br/>
Change<br/>
NUM_GPUS: to the GPUS that will be used for training<br/>
Then, run the following command<br/>
```
python run.py --cfg configs/data_loader_test_canonical_form_25-08-21.yaml
```
