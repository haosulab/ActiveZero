"""
Author: Isabella Liu 8/15/21
Feature:
"""

import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as Transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.config import cfg
from utils.test_util import calc_left_ir_depth_from_rgb
from utils.util import load_pickle

def __data_augmentation__(gaussian_blur=False, color_jitter=False):
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


class MessytableTestDataset(Dataset):
    def __init__(self, split_file, debug=False, sub=100, onReal=False):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        self.onReal = onReal
        (
            self.img_sim_L,
            self.img_sim_R,
            self.img_real_L,
            self.img_real_R,
            self.img_depth_l,
            self.img_depth_r,
            self.img_meta,
            self.img_label,
            self.img_sim_realsense,
            self.img_real_realsense,
            self.mask_scenes,
        ) = self.__get_split_files__(split_file, debug=debug, sub=sub)

    @staticmethod
    def __get_split_files__(split_file, debug=False, sub=100):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        :return: Lists of paths to the entries listed in split file
        """
        sim_dataset = cfg.DIR.DATASET
        real_dataset = cfg.REAL.DATASET
        sim_img_left_name = cfg.SPLIT.LEFT
        sim_img_right_name = cfg.SPLIT.RIGHT
        real_img_left_name = cfg.REAL.LEFT
        real_img_right_name = cfg.REAL.RIGHT
        sim_realsense = cfg.SPLIT.SIM_REALSENSE
        real_realsense = cfg.SPLIT.REAL_REALSENSE

        with open(split_file, "r") as f:
            prefix = [line.strip() for line in f]

            img_L_sim = [
                os.path.join(sim_dataset, p, sim_img_left_name) for p in prefix
            ]
            img_R_sim = [
                os.path.join(sim_dataset, p, sim_img_right_name) for p in prefix
            ]
            img_L_real = [
                os.path.join(real_dataset, p, real_img_left_name) for p in prefix
            ]
            img_R_real = [
                os.path.join(real_dataset, p, real_img_right_name) for p in prefix
            ]
            img_depth_l = [
                os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHL) for p in prefix
            ]
            img_depth_r = [
                os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHR) for p in prefix
            ]
            img_meta = [
                os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.META) for p in prefix
            ]
            img_label = [
                os.path.join(cfg.REAL.DATASET_V9, p, cfg.SPLIT.LABEL) for p in prefix
            ]
            img_sim_realsense = [
                os.path.join(sim_dataset, p, sim_realsense) for p in prefix
            ]
            img_real_realsense = [
                os.path.join(real_dataset, p, real_realsense) for p in prefix
            ]

            if debug is True:
                img_L_sim = img_L_sim[:sub]
                img_R_sim = img_R_sim[:sub]
                img_L_real = img_L_real[:sub]
                img_R_real = img_R_real[:sub]
                img_depth_l = img_depth_l[:sub]
                img_depth_r = img_depth_r[:sub]
                img_meta = img_meta[:sub]
                img_label = img_label[:sub]
                img_sim_realsense = img_sim_realsense[:sub]
                img_real_realsense = img_real_realsense[:sub]

        # Obtain robot arm mask list
        if self.onReal:
            with open(cfg.REAL.MASK_FILE, "r") as f:
                mask_scenes = [line.strip() for line in f]
        else:
            mask_scenes = 0

        return (
            img_L_sim,
            img_R_sim,
            img_L_real,
            img_R_real,
            img_depth_l,
            img_depth_r,
            img_meta,
            img_label,
            img_sim_realsense,
            img_real_realsense,
            mask_scenes,
        )

    def __len__(self):
        return len(self.img_sim_L)

    def __getitem__(self, idx):
        if self.onReal:
            # Adjust brightness of real images
            img_L_rgb = np.array(Image.open(self.img_real_L[idx]).convert(mode="L"))
            img_R_rgb = np.array(Image.open(self.img_real_R[idx]).convert(mode="L"))

            img_L_rgb = img_L_rgb[:, :, None]
            img_R_rgb = img_R_rgb[:, :, None]
            img_L_rgb = np.repeat(img_L_rgb, 3, axis=-1)
            img_R_rgb = np.repeat(img_R_rgb, 3, axis=-1)

            img_L_rgb_sim = (
                np.array(Image.open(self.img_sim_L[idx]).convert(mode="L")) / 255
            )
            img_R_rgb_sim = (
                np.array(Image.open(self.img_sim_R[idx]).convert(mode="L")) / 255
            )
            img_L_rgb_sim = np.repeat(img_L_rgb_sim[:, :, None], 3, axis=-1)
            img_R_rgb_sim = np.repeat(img_R_rgb_sim[:, :, None], 3, axis=-1)
            img_depth_realsense = (
                np.array(Image.open(self.img_real_realsense[idx])) / 1000
            )

        else:
            img_L_rgb = np.array(Image.open(self.img_sim_L[idx]).convert(mode="L")) / 255
            img_R_rgb = np.array(Image.open(self.img_sim_R[idx]).convert(mode="L")) / 255
            img_L_rgb = np.repeat(img_L_rgb[:, :, None], 3, axis=-1)
            img_R_rgb = np.repeat(img_R_rgb[:, :, None], 3, axis=-1)
            img_L_rgb_real = np.array(
                Image.open(self.img_real_L[idx]).convert(mode="L")
            )[:, :, None]
            img_R_rgb_real = np.array(
                Image.open(self.img_real_R[idx]).convert(mode="L")
            )[:, :, None]
            img_L_rgb_real = np.repeat(img_L_rgb_real, 3, axis=-1)
            img_R_rgb_real = np.repeat(img_R_rgb_real, 3, axis=-1)
            img_depth_realsense = (
                np.array(Image.open(self.img_sim_realsense[idx])) / 1000
            )

        img_depth_l = (
            np.array(Image.open(self.img_depth_l[idx])) / 1000
        )  # convert from mm to m
        img_depth_r = (
            np.array(Image.open(self.img_depth_r[idx])) / 1000
        )  # convert from mm to m
        prefix = self.img_sim_L[idx].split("/")[-2]

        img_meta = self.img_meta
        img_label = np.array(Image.open(self.img_label[idx]))

        # Convert depth map to disparity map
        extrinsic = img_meta["extrinsic"]
        extrinsic_l = img_meta["extrinsic_l"]
        extrinsic_r = img_meta["extrinsic_r"]
        intrinsic = img_meta["intrinsic"]
        intrinsic_l = img_meta["intrinsic_l"]
        baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        focal_length = intrinsic_l[0, 0] / 2

        mask = img_depth_l > 0
        img_disp_l = np.zeros_like(img_depth_l)
        img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]
        mask = img_depth_r > 0
        img_disp_r = np.zeros_like(img_depth_r)
        img_disp_r[mask] = focal_length * baseline / img_depth_r[mask]

        # Mask out the robot arm, mask images are stored in 1280 * 720
        if self.onReal:
            scene_id = prefix.split("-")[-1]
            if scene_id in self.mask_scenes:
                robot_mask_file = os.path.join(cfg.REAL.MASK, scene_id + ".png")
                robot_mask = Image.open(robot_mask_file).convert(mode="L")
                h, w = mask.shape
                robot_mask = robot_mask.resize((w, h), resample=Image.BILINEAR)
                robot_mask = np.array(robot_mask) / 255
            else:
                robot_mask = np.zeros_like(img_depth_l)

        # Convert img_depth_realsense to irL camera frame
        img_depth_realsense = calc_left_ir_depth_from_rgb(
            intrinsic, intrinsic_l, extrinsic, extrinsic_l, img_depth_realsense
        )

        # Get data augmentation
        normalization = __data_augmentation__(gaussian_blur=False, color_jitter=False)

        item = {}
        item["img_L"] = normalization(img_L_rgb).type(torch.FloatTensor)
        item["img_R"] = normalization(img_R_rgb).type(torch.FloatTensor)
        item["img_disp_l"] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(
            0
        )  # [bs, 1, H, W]
        item["img_depth_l"] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(
            0
        )  # [bs, 1, H, W]
        item["img_disp_r"] = torch.tensor(img_disp_r, dtype=torch.float32).unsqueeze(
            0
        )  # [bs, 1, H, W]
        item["img_depth_r"] = torch.tensor(img_depth_r, dtype=torch.float32).unsqueeze(
            0
        )  # [bs, 1, H, W]
        item["img_depth_realsense"] = torch.tensor(
            img_depth_realsense, dtype=torch.float32
        ).unsqueeze(
            0
        )  # [bs, 1, H, W]
        item["img_label"] = torch.tensor(img_label, dtype=torch.float32).unsqueeze(
            0
        )  # [bs, 1, H, W]
        item["prefix"] = prefix
        item["focal_length"] = (
            torch.tensor(focal_length, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        item["baseline"] = (
            torch.tensor(baseline, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if self.onReal:
            item["robot_mask"] = torch.tensor(robot_mask, dtype=torch.float32).unsqueeze(0)

        if self.onReal:
            item["img_L_real"] = normalization(img_L_rgb_real).type(torch.FloatTensor)
            item["img_R_real"] = normalization(img_R_rgb_real).type(torch.FloatTensor)
        else:
            item["img_L_sim"] = normalization(img_L_rgb_sim).type(torch.FloatTensor)
            item["img_R_sim"] = normalization(img_R_rgb_sim).type(torch.FloatTensor)

        return item


def get_test_loader(split_file, debug=False, sub=100, onReal=False):
    """
    :param split_file: split file
    :param debug: Whether on debug mode, load less data
    :param sub: If on debug mode, how many items to load into dataset
    :param isTest: Whether on test, if test no random crop on input image
    :param onReal: Whether test on real dataset, folder and file name are different
    :return: dataloader
    """
    messytable_dataset = MessytableTestDataset(split_file, debug, sub, onReal=onReal)
    loader = DataLoader(messytable_dataset, batch_size=1, num_workers=0)
    return loader


if __name__ == "__main__":
    cdataset = MessytableTestDataset(
        "/code/dataset_local_v9/training_lists/all.txt", onReal=True
    )
    item = cdataset.__getitem__(0)
    print(item["img_L"].shape)
    print(item["img_R"].shape)
    # print(item['img_L_real'].shape)
    print(item["img_disp_l"].shape)
    print(item["prefix"])
    print(item["img_depth_realsense"].shape)
    print(item["robot_mask"].shape)
