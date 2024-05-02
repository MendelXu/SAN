#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, GenericMask
import sys

import san


def create_instances(predictions, image_size, ignore_label=255):
    ret = Instances(image_size)

    labels = np.asarray(
        [dataset_id_map(predictions[i]["category_id"]) for i in range(len(predictions))]
    )
    ret.pred_classes = labels
    ret.pred_masks = [
        GenericMask(predictions[i]["segmentation"], *image_size)
        for i in range(len(predictions))
    ]
    # convert instance to sem_seg map
    sem_seg = np.ones(image_size[:2], dtype=np.uint16) * ignore_label
    for mask, label in zip(ret.pred_masks, ret.pred_classes):
        sem_seg[mask.mask == 1] = label
    return sem_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument(
        "--input", required=True, help="JSON file produced by the model"
    )
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument(
        "--dataset", help="name of the dataset", default="coco_2017_val"
    )
    parser.add_argument(
        "--conf-threshold", default=0.5, type=float, help="confidence threshold"
    )
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:

        pred_by_image[p["file_name"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    elif "sem_seg" in args.dataset:

        def dataset_id_map(ds_id):
            return ds_id

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])
        if dic["file_name"] in pred_by_image:
            pred = create_instances(
                pred_by_image[dic["file_name"]],
                img.shape[:2],
                ignore_label=metadata.ignore_label,
            )

            vis = Visualizer(img, metadata)
            vis_pred = vis.draw_sem_seg(pred).get_image()
            # import pdb
            # pdb.set_trace()
            vis = Visualizer(img, metadata)
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                sem_seg = Image.open(f)
                sem_seg = np.asarray(sem_seg, dtype="uint16")
            vis_gt = vis.draw_sem_seg(sem_seg).get_image()
            # reisze pred and gt to the same height
            ratio = vis_gt.shape[0] / 512
            tgt_w = int(vis_pred.shape[1] / ratio)
            vis_pred = cv2.resize(vis_pred, (tgt_w,512))
            vis_gt = cv2.resize(vis_gt, (tgt_w,512))
            img = cv2.resize(img, (tgt_w,512))
            # build grid view
            blank_int = 255 * np.ones((vis_gt.shape[0], 10, 3), dtype=np.uint8)
            concat = np.concatenate(
                (img, blank_int, vis_pred, blank_int, vis_gt), axis=1
            )
            cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])