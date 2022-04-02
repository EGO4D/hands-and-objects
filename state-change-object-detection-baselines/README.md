
## Ego4D Object of Change Detection Baslines

### Data Preparation
- Make a `DATA_FOLDER`
- Put PRE + PNR + POST frames inside a folder named `pre_pnr_post_frames` under `DATA_FOLDER`

- Put challenge files under `DATA_FOLDER`
- Downloads [object of change detection annotation files](https://drive.google.com/drive/folders/1ncIJq3nk5t73o7Ur3DHwySzR1iWavl9v) in COCO fomat to a folder named `coco_annotations` under `DATA_FOLDER`
- The final DATA_FOLDER should looks like
    ```
    ${DATA_FOLDER}
    |-- coco_annotations
    |-- |-- trainval.json
    |   |-- train.json
    |   |-- val.json
    |-- pre_pnr_post_frames
        |-- video_id
        |   |-- frame_number.jpg
        |   |-- ...
        |-- ...
    ```


### Run Baselines
Please follow the instructions inside the following baseline folders
- [CenterBox](./CenterBox)
- [CenterNet-Zhou](./CenterNet-Zhou)
- [DETR](./DETR)
- [FasterRCNN](./FasterRCNN)
