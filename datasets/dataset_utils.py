import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as Transforms
from PIL import Image

from configs.config import cfg

def get_ir_pattern(img_ir: np.array, img: np.array, threshold=0.005):
    diff = np.abs(img_ir - img)
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    ir = np.zeros_like(diff)
    ir[diff > threshold] = 1
    return ir


def get_smoothed_ir_pattern(img_ir: np.array, img: np.array, ks=11):
    h, w = img_ir.shape
    hs = int(h // ks)
    ws = int(w // ks)
    diff = np.abs(img_ir - img)
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    diff_avg = cv2.resize(diff, (ws, hs), interpolation=cv2.INTER_AREA)
    diff_avg = cv2.resize(diff_avg, (w, h), interpolation=cv2.INTER_AREA)
    ir = np.zeros_like(diff)
    ir[diff > diff_avg] = 1
    return ir


def get_smoothed_ir_pattern2(
    img_ir: np.array, img: np.array, ks=11, threshold=0.005
):
    h, w = img_ir.shape
    hs = int(h // ks)
    ws = int(w // ks)
    diff = np.abs(img_ir - img)
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    diff_avg = cv2.resize(diff, (ws, hs), interpolation=cv2.INTER_AREA)
    diff_avg = cv2.resize(diff_avg, (w, h), interpolation=cv2.INTER_AREA)
    ir = np.zeros_like(diff)
    diff2 = diff - diff_avg
    ir[diff2 > threshold] = 1
    return ir


def data_augmentation(gaussian_blur=False, color_jitter=False):
    """
    :param gaussian_blur: Whether apply gaussian blur in data augmentation
    :param color_jitter: Whether apply color jitter in data augmentation
    Note:
        If you want to change the parameters of each augmentation, you need to go to config files,
        e.g. configs/remote_train_config.yaml
    """
    transform_list = [Transforms.ToTensor()]
    if gaussian_blur:
        gaussian_sig = random.uniform(
            cfg.DATA_AUG.GAUSSIAN_MIN, cfg.DATA_AUG.GAUSSIAN_MAX
        )
        transform_list += [
            Transforms.GaussianBlur(
                kernel_size=cfg.DATA_AUG.GAUSSIAN_KERNEL, sigma=gaussian_sig
            )
        ]
    if color_jitter:
        bright = random.uniform(cfg.DATA_AUG.BRIGHT_MIN, cfg.DATA_AUG.BRIGHT_MAX)
        contrast = random.uniform(cfg.DATA_AUG.CONTRAST_MIN, cfg.DATA_AUG.CONTRAST_MAX)
        transform_list += [
            Transforms.ColorJitter(
                brightness=[bright, bright], contrast=[contrast, contrast]
            )
        ]
    # Normalization
    transform_list += [
        Transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ]
    custom_augmentation = Transforms.Compose(transform_list)
    return custom_augmentation
