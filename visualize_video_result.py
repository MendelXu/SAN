# %%
from fvcore.common.file_io import PathManager
import json
import argparse
import json
import numpy as np
import os

from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager
import sys

import san
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import GenericMask, random_color
from detectron2.utils.video_visualizer import VideoVisualizer
from tqdm import tqdm


dataset = "ytvis_2019_train"
dicts = list(DatasetCatalog.get(dataset))
metadata = MetadataCatalog.get(dataset)


# %%
output_path = sys.argv[1]
style = sys.argv[2] if len(sys.argv) > 2 else "seg"
file = f"{output_path}/inference/results.json"
output_path = os.path.join(*file.split("/")[:-1], "vis", style)
if not os.path.exists(output_path):
    os.makedirs(output_path)
with PathManager.open(file, "r") as f:
    predictions = json.load(f)

pred_by_video = defaultdict(list)

for p in predictions:
    pred_by_video[p["video_id"]].append(p)

if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

    def dataset_id_map(ds_id):
        return metadata.thing_dataset_id_to_contiguous_id[ds_id]

elif "lvis" in dataset:
    # LVIS results are in the same format as COCO results, but have a different
    # mapping from dataset category id to contiguous category id in [0, #categories - 1]
    def dataset_id_map(ds_id):
        return ds_id - 1

elif "sem_seg" in dataset:

    def dataset_id_map(ds_id):
        return ds_id

else:
    raise ValueError("Unsupported dataset: {}".format(dataset))


# %%
def create_semseg_gt(predictions_per_video, frame_id, image_size, ignore_label=255):
    predictions_per_img = []
    annotations = predictions_per_video[frame_id]
    for annotation in annotations:
        predictions_per_img.append(
            {
                "category_id": annotation["category_id"],
                "segmentation": annotation["segmentation"],
            }
        )
    ret = Instances(image_size)
    labels = np.asarray(
        [predictions_per_img[i]["category_id"] for i in range(len(predictions_per_img))]
    )
    ret.pred_classes = labels
    ret.pred_masks = [
        GenericMask(predictions_per_img[i]["segmentation"], *image_size)
        for i in range(len(predictions_per_img))
    ]
    sem_seg = np.ones(image_size[:2], dtype=np.uint16) * ignore_label
    for mask, label in zip(ret.pred_masks, ret.pred_classes):
        sem_seg[mask.mask == 1] = label
    return sem_seg


def create_semseg(predictions_per_video, frame_id, image_size, ignore_label=255):
    predictions_per_img = []
    for p in predictions_per_video:
        segmentation = p["segmentations"][frame_id]
        # score = p["score"][frame_id]
        category_id = p["category_id"]
        predictions_per_img.append(
            {
                "segmentation": segmentation,
                "score": p["score"],
                "category_id": category_id,
            }
        )
    # only select the top-1 prediction
    if style == "seg":
        predictions_per_img = sorted(predictions_per_img, key=lambda x: x["score"])
    else:
        k = int(style[3:])
        predictions_per_img = sorted(
            predictions_per_img, key=lambda x: x["score"], reverse=True
        )[:k]
    ret = Instances(image_size)
    labels = np.asarray(
        [
            dataset_id_map(predictions_per_img[i]["category_id"])
            for i in range(len(predictions_per_img))
        ]
    )
    ret.scores = np.asarray(
        [predictions_per_img[i]["score"] for i in range(len(predictions_per_img))]
    )
    ret.pred_classes = labels
    ret.pred_masks = [
        GenericMask(predictions_per_img[i]["segmentation"], *image_size)
        for i in range(len(predictions_per_img))
    ]

    sem_seg = np.ones(image_size[:2], dtype=np.uint16) * ignore_label
    for mask, label in zip(ret.pred_masks, ret.pred_classes):
        sem_seg[mask.mask == 1] = label
    return sem_seg


metadata.stuff_classes = metadata.thing_classes
metadata.stuff_colors = {i: random_color(rgb=True, maximum=255) for i in range(256)}
max_seq = 10
for i, dic in tqdm(enumerate(dicts)):
    video_id = dic["video_id"]
    if video_id not in pred_by_video:
        continue
    video_visualizer = VideoVisualizer(metadata=metadata)
    # read video
    file_names = dic["file_names"]
    # load frames
    frames = [cv2.imread(file_name)[:, :, ::-1] for file_name in file_names]
    video_len = len(frames)
    total_input = []
    total_output = []
    for frame_id, frame in enumerate(frames):
        gt = create_semseg_gt(dic["annotations"], frame_id, frame.shape[:2])
        frame_vis = video_visualizer.draw_sem_seg(
            frame.copy(),
            create_semseg(pred_by_video[video_id], frame_id, frame.shape[:2]),
        )
        total_output.append(frame_vis.get_image())
        frame_vis = video_visualizer.draw_sem_seg(frame.copy(), gt)
        total_input.append(frame_vis.get_image())
    if len(total_output) == 0:
        continue
    output = np.concatenate(total_output[::4], axis=1)
    input = np.concatenate(total_input[::4], axis=1)
    output = np.concatenate([input, output], axis=0)
    cv2.imwrite("{}/{}.jpg".format(output_path, video_id), output[:, :, ::-1])
