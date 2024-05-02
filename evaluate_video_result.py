# %%
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import GenericMask, Visualizer
from fvcore.common.file_io import PathManager

import san

dataset = "ytvis_2019_train"
dicts = list(DatasetCatalog.get(dataset))
metadata = MetadataCatalog.get(dataset)


# %%
out_dir = sys.argv[1]
file = f"{out_dir}/inference/results.json"
if len(sys.argv) > 2:
    style = sys.argv[2]
else:
    style = "seg"
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
from math import floor


def db_eval_boundary(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
            foreground_mask (ndarray): binary segmentation image.
            gt_mask         (ndarray): binary annotated image.
    Returns:
            F (float): boundaries F-measure
            P (float): boundaries precision
            R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = (
        bound_th
        if bound_th >= 1
        else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))
    )

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)

    from skimage.morphology import binary_dilation, disk

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
            seg     : Segments labeled from 1..k.
            width	  :	Width of desired bmap  <= seg.shape[1]
            height  :	Height of desired bmap <= seg.shape[0]
    Returns:
            bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + floor((y - 1) + height / h)
                    i = 1 + floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def db_eval_iou(annotation, segmentation):
    """Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
    """

    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / np.sum(
            (annotation | segmentation), dtype=np.float32
        )


class Evaluator:
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.reset()
        print("Evaluation on {} classes".format(self.num_classes))

    def reset(self):
        self.tp_list = [0] * self.num_classes
        self.f_list = [0] * self.num_classes
        self.j_list = [0] * self.num_classes
        self.n_list = [0] * self.num_classes
        self.total_list = [0] * self.num_classes
        self.iou_list = [0] * self.num_classes

        self.f_score = [0] * self.num_classes
        self.j_score = [0] * self.num_classes

    @classmethod
    def batch_eval(cls, pred_seg, gt_seg, num_classes):
        # N H W
        f_list = [0] * num_classes
        j_list = [0] * num_classes
        n_list = [0] * num_classes
        tp_list = [0] * num_classes
        total_list = [0] * num_classes

        label = np.unique(gt_seg)
        # plabel = np.unique(pred_seg)
        # label = np.union1d(label, plabel)
        label = label[label != 255]
        for l in label:
            gt_mask = gt_seg == l  # N H W
            pred_mask = pred_seg == l  # N H W
            # compute tp,fp,fn
            tp = np.logical_and(gt_mask, pred_mask).sum()
            fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
            fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()
            total = tp + fp + fn
            for j in range(gt_seg.shape[0]):
                f_list[l] += db_eval_boundary(pred_mask[j], gt_mask[j])
                j_list[l] += db_eval_iou(pred_mask[j], gt_mask[j])
                n_list[l] += 1
            tp_list[l] += tp
            total_list[l] += total
        return f_list, j_list, n_list, tp_list, total_list

    def update_with_results(self, f_list, j_list, n_list, tp_list, total_list):
        for i in range(self.num_classes):
            self.f_list[i] += f_list[i]
            self.j_list[i] += j_list[i]
            self.n_list[i] += n_list[i]
            self.tp_list[i] += tp_list[i]
            self.total_list[i] += total_list[i]

    def summary(self, per_class=True):
        self.f_score = [
            self.f_list[ic] / float(max(self.n_list[ic], 1))
            for ic in range(self.num_classes)
        ]
        self.j_score = [
            self.j_list[ic] / float(max(self.n_list[ic], 1))
            for ic in range(self.num_classes)
        ]
        self.iou_list = [
            self.tp_list[ic] / float(max(self.total_list[ic], 1))
            for ic in range(self.num_classes)
        ]
        print("\n")
        print("F-score: ", 100.0 * np.mean(self.f_score))
        print("J-score: ", 100.0 * np.mean(self.j_score))
        print("mIoU: ", 100.0 * np.mean(self.iou_list))
        if per_class:
            for name, f, j, iou in zip(
                self.classes, self.f_score, self.j_score, self.iou_list
            ):
                print(
                    "{}: F-score: {:.3f}, J-score: {:.3f} IoU: {:.3f}".format(
                        name, 100.0 * f, 100.0 * j, 100.0 * iou
                    )
                )
        return [
            {f"F-{name}": 100.0 * f, f"J-{name}": 100.0 * j, f"IoU-{name}": 100.0 * iou}
            for name, f, j, iou in zip(
                self.classes, self.f_score, self.j_score, self.iou_list
            )
        ]


# %%
# evaluation
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
        predictions_per_img = sorted(predictions_per_img, key=lambda x: x["score"])[-k:]

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


from multiprocessing import Pool

import imagesize
from tqdm import tqdm
import time


def get_pred_and_gt(dic, pred_dic, num_classes):
    file_names = dic["file_names"]
    # load frames
    video_len = len(file_names)
    frames = [imagesize.get(file_name)[::-1] for file_name in file_names]
    gt_segs = np.stack(
        [
            create_semseg_gt(dic["annotations"], frame_id, frame)
            for frame_id, frame in enumerate(frames)
        ]
    )
    pred_segs = np.stack(
        [
            create_semseg(pred_dic, frame_id, frame)
            for frame_id, frame in enumerate(frames)
        ]
    )
    return Evaluator.batch_eval(pred_segs, gt_segs, num_classes=num_classes)


results = []
with Pool(24) as p:
    evaluator = Evaluator(metadata.thing_classes)
    for i, dic in tqdm(enumerate(dicts)):
        video_id = dic["video_id"]
        pred_dic = pred_by_video[video_id]
        results.append(
            p.apply_async(get_pred_and_gt, args=(dic, pred_dic, evaluator.num_classes))
        )

    pbar = tqdm(enumerate(results), total=len(results))
    # metric_per_image = []
    for i, r in pbar:
        batch_eval_result = r.get()
        evaluator.update_with_results(*batch_eval_result)
        if i % 20 == 0:
            evaluator.summary(per_class=False)
    res = evaluator.summary()
    with open(f"{out_dir}/res.json", "w") as f:
        json.dump(res, f)
