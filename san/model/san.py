from typing import List
import os
import open_clip
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans

from .clip_utils import (
    FeatureExtractor,
    LearnableBgOvClassifier,
    PredefinedOvClassifier,
    RecWithAttnbiasHead,
    get_predefined_templates,
)
from .criterion import SetCriterion
from .matcher import HungarianMatcher
from .side_adapter import build_side_adapter_network


@META_ARCH_REGISTRY.register()
class SAN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        clip_visual_extractor: nn.Module,
        clip_rec_head: nn.Module,
        side_adapter_network: nn.Module,
        ov_classifier: PredefinedOvClassifier,
        criterion: SetCriterion,
        size_divisibility: int,
        asymetric_input: bool = True,
        clip_resolution: float = 0.5,
        video_cluster_num: int = 100,
        pixel_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        pixel_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        sem_seg_postprocess_before_inference: bool = False,
    ):
        super().__init__()
        self.asymetric_input = asymetric_input
        self.clip_resolution = clip_resolution
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility
        self.criterion = criterion

        self.side_adapter_network = side_adapter_network
        self.clip_visual_extractor = clip_visual_extractor
        self.clip_rec_head = clip_rec_head
        self.ov_classifier = ov_classifier
        self.video_cluster_num = video_cluster_num
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        ## copied from maskformer2
        # Loss parameters
        no_object_weight = cfg.MODEL.SAN.NO_OBJECT_WEIGHT
        # loss weights
        class_weight = cfg.MODEL.SAN.CLASS_WEIGHT
        dice_weight = cfg.MODEL.SAN.DICE_WEIGHT
        mask_weight = cfg.MODEL.SAN.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }
        aux_weight_dict = {}
        for i in range(len(cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS) - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks"]

        criterion = SetCriterion(
            num_classes=cfg.MODEL.SAN.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.SAN.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.SAN.IMPORTANCE_SAMPLE_RATIO,
        )
        ## end of copy

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.MODEL.SAN.CLIP_MODEL_NAME,
            pretrained=cfg.MODEL.SAN.CLIP_PRETRAINED_NAME,
        )
        ov_classifier = LearnableBgOvClassifier(
            model, templates=get_predefined_templates(cfg.MODEL.SAN.CLIP_TEMPLATE_SET)
        )

        clip_visual_extractor = FeatureExtractor(
            model.visual,
            last_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,
            frozen_exclude=cfg.MODEL.SAN.CLIP_FROZEN_EXCLUDE,
        )
        clip_rec_head = RecWithAttnbiasHead(
            model.visual,
            first_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,
            frozen_exclude=cfg.MODEL.SAN.CLIP_DEEPER_FROZEN_EXCLUDE,
            cross_attn=cfg.MODEL.SAN.REC_CROSS_ATTN,
            sos_token_format=cfg.MODEL.SAN.SOS_TOKEN_FORMAT,
            sos_token_num=cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES,
            downsample_method=cfg.MODEL.SAN.REC_DOWNSAMPLE_METHOD,
        )

        pixel_mean, pixel_std = (
            preprocess.transforms[-1].mean,
            preprocess.transforms[-1].std,
        )
        pixel_mean = [255.0 * x for x in pixel_mean]
        pixel_std = [255.0 * x for x in pixel_std]

        return {
            "clip_visual_extractor": clip_visual_extractor,
            "clip_rec_head": clip_rec_head,
            "side_adapter_network": build_side_adapter_network(
                cfg, clip_visual_extractor.output_shapes
            ),
            "ov_classifier": ov_classifier,
            "criterion": criterion,
            "size_divisibility": cfg.MODEL.SAN.SIZE_DIVISIBILITY,
            "asymetric_input": cfg.MODEL.SAN.ASYMETRIC_INPUT,
            "clip_resolution": cfg.MODEL.SAN.CLIP_RESOLUTION,
            "sem_seg_postprocess_before_inference": cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,
            "video_cluster_num": cfg.MODEL.SAN.N_CLUSTERS,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
        }

    def forward(self, batched_inputs):
        if isinstance(batched_inputs[0]["image"], list):
            return self.forward_video(batched_inputs)
        else:
            return self.forward_image(batched_inputs)

    def forward_video(self, batched_inputs):
        # get classifier weight for each dataset
        # !! Could be computed once and saved. It will run only once per dataset.
        if "vocabulary" in batched_inputs[0]:
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_vocabulary(
                    batched_inputs[0]["vocabulary"]
                )
            )
        else:
            dataset_names = [x["meta"]["dataset_name"] for x in batched_inputs]
            assert (
                len(list(set(dataset_names))) == 1
            ), "All images in a batch must be from the same dataset."
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_dataset_name(dataset_names[0])
            )  # C+1,ndim

        images = []
        img_metas = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
                img_metas.append(video["meta"])

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images, self.size_divisibility
        )  # (N*L, C, H, W)

        # could extract by chunk to save memory.
        clip_input = images.tensor
        if self.asymetric_input:
            clip_input = F.interpolate(
                clip_input, scale_factor=self.clip_resolution, mode="bilinear"
            )
        clip_image_features = self.clip_visual_extractor(clip_input)
        features = self.side_adapter_network.forward_features(
            images.tensor, clip_image_features
        )
        all_proposal = features[-1]["query"].reshape(
            -1, features[-1]["query"].shape[-1]
        )

        # cluster
        kmeans = KMeans(
            n_clusters=self.video_cluster_num, random_state=0, n_init=1
        ).fit(all_proposal.cpu().numpy())
        all_proposal = torch.from_numpy(kmeans.cluster_centers_).to(all_proposal.device)
        features[-1]["query"] = all_proposal[None, :, :].repeat(
            features[-1]["query"].shape[0], 1, 1
        )
        mask_preds, attn_biases = self.side_adapter_network.decode_masks(features)
        # !! Could be optimized to run in parallel.
        mask_embs = [
            self.clip_rec_head(clip_image_features, attn_bias, normalize=True)
            for attn_bias in attn_biases
        ]  # [B,N,C]
        mask_logits = [
            torch.einsum("bqc,nc->bqn", mask_emb, ov_classifier_weight)
            for mask_emb in mask_embs
        ]
        # torch.cuda.empty_cache()
        if self.training:
            raise NotImplementedError()
        else:
            start = 0
            mask_pred_results = mask_preds[-1]
            mask_cls_results = mask_logits[-1]
            video_outputs = []
            for video in batched_inputs:
                video_length = len(video["image"])
                mask_cls_result = mask_cls_results[start : start + video_length]
                mask_cls_result = mask_cls_result.mean(dim=0, keepdim=True).repeat(
                    video_length, 1, 1
                )
                mask_pred_result = mask_pred_results[start : start + video_length]
                mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                    mask_pred_result,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                image_size = images.image_sizes[start]
                height = video.get(
                    "height", image_size[0]
                )  # raw image size before data augmentation
                width = video.get("width", image_size[1])
                video_output = retry_if_cuda_oom(self.inference_video)(
                    mask_cls_result,
                    mask_pred_result,
                    image_size,
                    height,
                    width,
                )
                video_outputs.append(video_output)
                start += video_length
            del outputs
            del features
            return video_outputs

    def forward_image(self, batched_inputs):
        # get classifier weight for each dataset
        # !! Could be computed once and saved. It will run only once per dataset.
        if "vocabulary" in batched_inputs[0]:
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_vocabulary(
                    batched_inputs[0]["vocabulary"]
                )
            )
        else:
            dataset_names = [x["meta"]["dataset_name"] for x in batched_inputs]
            assert (
                len(list(set(dataset_names))) == 1
            ), "All images in a batch must be from the same dataset."
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_dataset_name(dataset_names[0])
            )  # C+1,ndim
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        clip_input = images.tensor
        if self.asymetric_input:
            clip_input = F.interpolate(
                clip_input, scale_factor=self.clip_resolution, mode="bilinear"
            )
        clip_image_features = self.clip_visual_extractor(clip_input)
        mask_preds, attn_biases = self.side_adapter_network(
            images.tensor, clip_image_features
        )
        # !! Could be optimized to run in parallel.
        mask_embs = [
            self.clip_rec_head(clip_image_features, attn_bias, normalize=True)
            for attn_bias in attn_biases
        ]  # [B,N,C]
        mask_logits = [
            torch.einsum("bqc,nc->bqn", mask_emb, ov_classifier_weight)
            for mask_emb in mask_embs
        ]
        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            outputs = {
                "pred_logits": mask_logits[-1],
                "pred_masks": mask_preds[-1],
                "aux_outputs": [
                    {
                        "pred_logits": aux_pred_logits,
                        "pred_masks": aux_pred_masks,
                    }
                    for aux_pred_logits, aux_pred_masks in zip(
                        mask_logits[:-1], mask_preds[:-1]
                    )
                ],
            }
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_preds = mask_preds[-1]
            mask_logits = mask_logits[-1]
            # torch.cuda.empty_cache()
            # Inference
            mask_preds = F.interpolate(
                mask_preds,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_logits, mask_preds, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r
            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    @property
    def device(self):
        return self.pixel_mean.device

    def inference_video(
        self, pred_cls, pred_masks, img_size, output_height, output_width
    ):
        # if on cpu, convert to fp32
        if not pred_cls.is_cuda:
            pred_cls = pred_cls.float()
            pred_masks = pred_masks.float()
        if len(pred_cls) > 0:
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]  # Q, K, H, W
            scores = F.softmax(pred_cls, dim=-1)[..., :-1]  # seq, query, class
            seg_map = torch.einsum(
                "lqc,lqhw->lchw", scores, pred_masks.sigmoid()
            )  # seq, class, h, w
            seg_map = F.interpolate(
                seg_map,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )
            seg_mask = seg_map.argmax(dim=1)  # seq,h,w
            label = seg_mask.unique()
            out_labels = label[label < 40].tolist()  # remove background
            out_scores = [0 for _ in range(len(out_labels))]
            out_masks = []
            # import pdb; pdb.set_trace()
            for idx, l in enumerate(out_labels):
                mask = seg_mask == l
                for i, m in enumerate(mask):
                    score = seg_map[i, l][m].mean()
                    out_scores[idx] += score.item()
                out_scores[idx] /= len(mask)
                out_masks.append(mask.cpu())
                # all labels exist in all frames.
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
