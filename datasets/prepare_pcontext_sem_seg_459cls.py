#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhen Zhu(zzhu@hust.edu.cn)
# Generate train & val data.


import os
import argparse
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

LABEL_DIR = "label"
IMAGE_DIR = "image"


class PascalContextGenerator(object):
    def __init__(self, args, image_dir=IMAGE_DIR, label_dir=LABEL_DIR):
        self.args = args
        self.train_label_dir = os.path.join(self.args.save_dir, "train", label_dir)
        self.val_label_dir = os.path.join(self.args.save_dir, "val", label_dir)
        if not os.path.exists(self.train_label_dir):
            os.makedirs(self.train_label_dir)

        if not os.path.exists(self.val_label_dir):
            os.makedirs(self.val_label_dir)

        self.train_image_dir = os.path.join(self.args.save_dir, "train", image_dir)
        self.val_image_dir = os.path.join(self.args.save_dir, "val", image_dir)
        if not os.path.exists(self.train_image_dir):
            os.makedirs(self.train_image_dir)

        if not os.path.exists(self.val_image_dir):
            os.makedirs(self.val_image_dir)
        self.all_cls = set()

    def _class_to_index(self, mask):
        self.all_cls = self.all_cls.union(set(np.unique(mask).tolist()))
        # import pdb
        # pdb.set_trace()
        mask = mask - 1
        return mask

    def generate_label(self):
        _image_dir = os.path.join(self.args.img_dir, "JPEGImages")
        _anno_dir = self.args.anno_dir
        annFile = os.path.join(self.args.img_dir, "trainval_merged.json")

        from detail import Detail

        train_detail = Detail(annFile, _image_dir, "train")
        train_ids = train_detail.getImgs()

        for img_id in tqdm(train_ids):
            mask = loadmat(
                os.path.join(_anno_dir, img_id["file_name"].replace(".jpg", ".mat"))
            )["LabelMap"]
            mask = Image.fromarray(self._class_to_index(mask))
            filename = img_id["file_name"]
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(_image_dir, filename)
                shutil.copy(imgpath, os.path.join(self.train_image_dir, filename))
                mask_png_name = basename + ".tif"
                mask.save(os.path.join(self.train_label_dir, mask_png_name))

        val_detail = Detail(annFile, _image_dir, "val")
        val_ids = val_detail.getImgs()
        for img_id in tqdm(val_ids):
            mask = loadmat(
                os.path.join(_anno_dir, img_id["file_name"].replace(".jpg", ".mat"))
            )["LabelMap"]
            mask = Image.fromarray(self._class_to_index(mask))
            filename = img_id["file_name"]
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(_image_dir, filename)
                shutil.copy(imgpath, os.path.join(self.val_image_dir, filename))
                mask_png_name = basename + ".tif"
                mask.save(os.path.join(self.val_label_dir, mask_png_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        dest="save_dir",
        help="The directory to save the data.",
    )
    # ori_root_dir: VOCdevkit/VOC2010
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        dest="img_dir",
        help="The directory of the cityscapes data.",
    )
    parser.add_argument(
        "--anno_dir",
        default=None,
        type=str,
        dest="anno_dir",
        help="The directory of the cityscapes data.",
    )
    args = parser.parse_args()

    pascalcontext_seg_generator = PascalContextGenerator(args)
    pascalcontext_seg_generator.generate_label()
    print(pascalcontext_seg_generator.all_cls)
