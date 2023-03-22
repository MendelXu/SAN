from typing import List, Union

import numpy as np

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import os

import huggingface_hub
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer, random_color
from huggingface_hub import hf_hub_download
from PIL import Image

from san import add_san_config
from san.data.datasets.register_coco_stuff_164k import COCO_CATEGORIES

model_cfg = {
    "san_vit_b_16": {
        "config_file": "configs/san_clip_vit_res4_coco.yaml",
        "model_path": "huggingface:san_vit_b_16.pth",
    },
    "san_vit_large_16": {
        "config_file": "configs/san_clip_vit_large_res4_coco.yaml",
        "model_path": "huggingface:san_vit_large_14.pth",
    },
}


def download_model(model_path: str):
    """
    Download the model from huggingface hub.
    Args:
        model_path (str): the model path
    Returns:
        str: the downloaded model path
    """
    if "HF_TOKEN" in os.environ:
        huggingface_hub.login(token=os.environ["HF_TOKEN"])
    model_path = model_path.split(":")[1]
    model_path = hf_hub_download("Mendel192/san", filename=model_path)
    return model_path


def setup(config_file: str, device=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device or "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


class Predictor(object):
    def __init__(self, config_file: str, model_path: str):
        """
        Args:
            config_file (str): the config file path
            model_path (str): the model path
        """
        cfg = setup(config_file)
        self.model = DefaultTrainer.build_model(cfg)
        if model_path.startswith("huggingface:"):
            model_path = download_model(model_path)
        print("Loading model from: ", model_path)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            model_path
        )
        print("Loaded model from: ", model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.cuda()

    def predict(
        self,
        image_data_or_path: Union[Image.Image, str],
        vocabulary: List[str] = [],
        augment_vocabulary: Union[str,bool] = True,
        output_file: str = None,
    ) -> Union[dict, None]:
        """
        Predict the segmentation result.
        Args:
            image_data_or_path (Union[Image.Image, str]): the input image or the image path
            vocabulary (List[str]): the vocabulary used for the segmentation
            augment_vocabulary (bool): whether to augment the vocabulary
            output_file (str): the output file path
        Returns:
            Union[dict, None]: the segmentation result
        """
        if isinstance(image_data_or_path, str):
            image_data = Image.open(image_data_or_path)
        else:
            image_data = image_data_or_path
        w, h = image_data.size
        image_tensor: torch.Tensor = self._preprocess(image_data)
        vocabulary = list(set([v.lower().strip() for v in vocabulary]))
        # remove invalid vocabulary
        vocabulary = [v for v in vocabulary if v != ""]
        print("vocabulary:", vocabulary)
        ori_vocabulary = vocabulary

        if isinstance(augment_vocabulary,str):
            vocabulary = self.augment_vocabulary(vocabulary, augment_vocabulary)
        else:
            vocabulary = self._merge_vocabulary(vocabulary)
        if len(ori_vocabulary) == 0:
            ori_vocabulary = vocabulary
        with torch.no_grad():
            result = self.model(
                [
                    {
                        "image": image_tensor,
                        "height": h,
                        "width": w,
                        "vocabulary": vocabulary,
                    }
                ]
            )[0]["sem_seg"]
        seg_map = self._postprocess(result, ori_vocabulary)
        if output_file:
            self.visualize(image_data, seg_map, ori_vocabulary, output_file)
            return
        return {
            "image": image_data,
            "sem_seg": seg_map,
            "vocabulary": ori_vocabulary,
        }

    def visualize(
        self,
        image: Image.Image,
        sem_seg: np.ndarray,
        vocabulary: List[str],
        output_file: str = None,
        mode: str = "overlay",
    ) -> Union[Image.Image, None]:
        """
        Visualize the segmentation result.
        Args:
            image (Image.Image): the input image
            sem_seg (np.ndarray): the segmentation result
            vocabulary (List[str]): the vocabulary used for the segmentation
            output_file (str): the output file path
            mode (str): the visualization mode, can be "overlay" or "mask"
        Returns:
            Image.Image: the visualization result. If output_file is not None, return None.
        """
        # add temporary metadata
        # set numpy seed to make sure the colors are the same
        np.random.seed(0)
        colors = [random_color(rgb=True, maximum=255) for _ in range(len(vocabulary))]
        MetadataCatalog.get("_temp").set(stuff_classes=vocabulary, stuff_colors=colors)
        metadata = MetadataCatalog.get("_temp")
        if mode == "overlay":
            v = Visualizer(image, metadata)
            v = v.draw_sem_seg(sem_seg, area_threshold=0).get_image()
            v = Image.fromarray(v)
        else:
            v = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
            labels, areas = np.unique(sem_seg, return_counts=True)
            sorted_idxs = np.argsort(-areas).tolist()
            labels = labels[sorted_idxs]
            for label in filter(lambda l: l < len(metadata.stuff_classes), labels):
                v[sem_seg == label] = metadata.stuff_colors[label]
            v = Image.fromarray(v)
        # remove temporary metadata
        MetadataCatalog.remove("_temp")
        if output_file is None:
            return v
        v.save(output_file)
        print(f"saved to {output_file}")

    def _merge_vocabulary(self, vocabulary: List[str]) -> List[str]:
        default_voc = [c["name"] for c in COCO_CATEGORIES]
        return vocabulary + [c for c in default_voc if c not in vocabulary]

    def augment_vocabulary(
        self, vocabulary: List[str], aug_set: str = "COCO-all"
    ) -> List[str]:
        default_voc = [c["name"] for c in COCO_CATEGORIES]
        stuff_voc = [
            c["name"]
            for c in COCO_CATEGORIES
            if "isthing" not in c or c["isthing"] == 0
        ]
        if aug_set == "COCO-all":
            return vocabulary + [c for c in default_voc if c not in vocabulary]
        elif aug_set == "COCO-stuff":
            return vocabulary + [c for c in stuff_voc if c not in vocabulary]
        else:
            return vocabulary

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the input image.
        Args:
            image (Image.Image): the input image
        Returns:
            torch.Tensor: the preprocessed image
        """
        image = image.convert("RGB")
        # resize short side to 640
        w, h = image.size
        if w < h:
            image = image.resize((640, int(h * 640 / w)))
        else:
            image = image.resize((int(w * 640 / h), 640))
        image = torch.from_numpy(np.asarray(image)).float()
        image = image.permute(2, 0, 1)
        return image

    def _postprocess(
        self, result: torch.Tensor, ori_vocabulary: List[str]
    ) -> np.ndarray:
        """
        Postprocess the segmentation result.
        Args:
            result (torch.Tensor): the segmentation result
            ori_vocabulary (List[str]): the original vocabulary used for the segmentation
        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        result = result.argmax(dim=0).cpu().numpy()  # (H, W)
        if len(ori_vocabulary) == 0:
            return result
        result[result >= len(ori_vocabulary)] = len(ori_vocabulary)
        return result


def pre_download():
    """pre downlaod model from huggingface and open_clip to avoid network issue."""
    for model_name, model_info in model_cfg.items():
        download_model(model_info["model_path"])
        cfg = setup(model_info["config_file"])
        DefaultTrainer.build_model(cfg)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, required=True, help="path to config file"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="path to model file"
    )
    parser.add_argument(
        "--img-path", type=str, required=True, help="path to image file."
    )
    parser.add_argument("--aug-vocab", action="store_true", help="augment vocabulary.")
    parser.add_argument(
        "--vocab",
        type=str,
        default="",
        help="list of category name. seperated with ,.",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()
    predictor = Predictor(config_file=args.config_file, model_path=args.model_path)
    predictor.predict(
        args.img_path,
        args.vocab.split(","),
        args.aug_vocab,
        output_file=args.output_file,
    )
