# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_san_config(cfg):
    # copied from maskformer2
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # optimizer
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP = [
        "absolute_pos_embed",
        "positional_embedding",
        "pos_embed",
        "query_embed",
        "relative_position_bias_table",
    ]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.CLIP_MULTIPLIER = 1.0
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1

    # san
    cfg.MODEL.SAN = CN()
    cfg.MODEL.SAN.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.SAN.CLASS_WEIGHT = 2.0
    cfg.MODEL.SAN.DICE_WEIGHT = 5.0
    cfg.MODEL.SAN.MASK_WEIGHT = 5.0
    cfg.MODEL.SAN.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.SAN.NUM_CLASSES = 171
    cfg.MODEL.SAN.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.SAN.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.SAN.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.SAN.CLIP_PRETRAINED_NAME = "openai"
    cfg.MODEL.SAN.CLIP_TEMPLATE_SET = "vild"
    cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX = 9
    cfg.MODEL.SAN.CLIP_FROZEN_EXCLUDE = ["positional_embedding"]
    cfg.MODEL.SAN.CLIP_DEEPER_FROZEN_EXCLUDE = []
    cfg.MODEL.SAN.REC_CROSS_ATTN = False
    cfg.MODEL.SAN.REC_DOWNSAMPLE_METHOD = "max"
    cfg.MODEL.SAN.SOS_TOKEN_FORMAT = "cls_token"
    cfg.MODEL.SAN.SIZE_DIVISIBILITY = 32
    cfg.MODEL.SAN.ASYMETRIC_INPUT = True
    cfg.MODEL.SAN.CLIP_RESOLUTION = 0.5

    cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE = True
    # side adapter
    cfg.MODEL.SIDE_ADAPTER = CN()
    cfg.MODEL.SIDE_ADAPTER.NAME = "RegionwiseSideAdapterNetwork"
    cfg.MODEL.SIDE_ADAPTER.VIT_NAME = "vit_w240n6d8_patch16"
    cfg.MODEL.SIDE_ADAPTER.PRETRAINED = False
    cfg.MODEL.SIDE_ADAPTER.IMAGE_SIZE = 640
    cfg.MODEL.SIDE_ADAPTER.DROP_PATH_RATE = 0.0
    cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES = 100
    cfg.MODEL.SIDE_ADAPTER.FUSION_TYPE = "add"
    cfg.MODEL.SIDE_ADAPTER.FUSION_MAP = ["0->0", "3->1", "6->2", "9->3"]
    cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS = [7, 8]

    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS = CN()
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_HEADS = 12
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_LAYERS = 1
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.EMBED_CHANNELS = 256
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_CHANNELS = 256
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_NUM_LAYERS = 3
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.RESCALE_ATTN_BIAS = True

    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "san"
    cfg.WANDB.NAME = None
    # use flash attention
    cfg.MODEL.FLASH = False
