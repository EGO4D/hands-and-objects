import os
import math
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from pycocotools.coco import COCO


data_folder = 'PATH_TO_DATA_FOLDER'
for split in ["train", "val", "trainval", "test"]:
    register_coco_instances(
        f"ego4dv1_pre_pnr_post_object_{split}", {},
        os.path.join(data_folder, 'coco_annotations', f"{split}.json"),
        os.path.join(data_folder, "pre_pnr_post_frames"))

test_dataset = 'ego4dv1_pre_pnr_post_object_test'
test_metadata = MetadataCatalog.get(test_dataset)
test_data = DatasetCatalog.get(test_dataset)

test_coco = COCO(test_metadata.json_file)

center_area_pct = 100  # covering 100% area of the frame
r = math.sqrt(center_area_pct / 100)
center_box_predictions = []
for data in test_data:
    w, h = data['width'], data['height']
    center_box_predictions.append({
        "image_id": data['image_id'],
        "bbox": [w/2 - w*r/2, h/2 - h*r/2, r*w, r*h],
        "category_id": 1,  # for the object
        "score": 0.99,
    })

coco_dt = test_coco.loadRes(center_box_predictions)
coco_eval = COCOeval_opt(test_coco, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
