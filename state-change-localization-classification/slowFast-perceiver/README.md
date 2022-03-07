# Slowfast + Perceiver Ego4D Keyframe Localization
Point-of-No-Return (PNR) localization using SlowFast with Perceiver backbone.

## Setup

Use the following command to install the required packages:

```
pip install -r requirements.txt
```

## Configuration
Change the following depending on your dataset configuration: <br/>
Open the file default.yaml
Then, modify the following entries with the appropriate paths: <br/><br/>
Under the **DATA** category <br/>
Change <br/>
**VIDEO_DIR_PATH**: to the path to the Ego4D videos <br/>
**ANN_PATH**: to the path to your annotation file<br/>
**SPLIT_PATH**: to the path to the folder with split files<br/>
**CLIPS_SAVE_PATH**: to the path to the folder that contains frames extracted from Ego4D videos or the path where you want to
store the extracted frames<br/>
**VIDEO_LOCATIONS_CSV**: to the path to a CSV file with paths to videos<br/>
**NO_SC_PATH**: to the path to the folder that contains frames extracted from Ego4D for videos without state change<br/>
**NO_SC_SPLIT_PATH**: to the path to JSON file containing the splits for videos without state change<br/><br/>


Under MISC<br/>
Change<br/>
**NUM_GPUS**: to the GPUS that will be used for training<br/>
Then, run the following command<br/>

## Train
To train the model run the following command from the project root
```
python run.py --cfg configs/default.yaml
```

## Test
To test the model run the following command from the project root
```
python run.py --cfg configs/default_test.yaml
```