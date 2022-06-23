import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from configs.config import cfg
from utils.util import load_pickle
from utils.reprojection import *
from datasets.dataset_utils import *
from utils.test_util import *

class MessytableDataset(Dataset):
    def __init__(self, split_sim=cfg.SIM.TRAIN, split_real=cfg.REAL.TRAIN, train=True, debug=False, sub=100, onReal=True, special=[]):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        :param onReal: if model trains on real
        :param special: list of special files that need to be loaded
        """

        self.train = train
        self.debug = debug
        self.sub = sub
        self.onReal = onReal
        self.special = special

        if self.train:
            (self.img_sim_L, self.img_sim_R, self.img_sim_L_no_ir, self.img_sim_R_no_ir,
                self.img_depth_l, self.img_depth_r, self.img_meta, self.prefix) = self.base_files(split_sim)
        else:
            (self.img_sim_L, self.img_sim_R, self.img_depth_l, self.img_depth_r,
                self.img_meta, self.prefix, self.img_sim_realsense) = self.base_files(split_sim)

        if self.onReal:
            if self.train:
                if "temporal" in self.special:
                    (self.img_real_L, self.img_real_R, self.img_real_L_no_ir,
                        self.img_real_R_no_ir, self.img_real_L_temporal_ir,
                        self.img_real_R_temporal_ir) = self.real_files(split_real)
                else:
                    (self.img_real_L, self.img_real_R, self.img_real_L_no_ir,
                        self.img_real_R_no_ir) = self.real_files(split_real)
            else:
                (self.img_label, self.img_real_L, self.img_real_R, self.img_real_realsense,
                    self.mask_scenes) = self.real_files(split_real)
            self.real_len = len(self.img_real_L)

        # make sure the special parameters exist
        for item in special:
            if item=="temporal":
                continue
            elif item=="p1" or item=="p2" or item=="img" or item=="lcn":
                continue
            else:
                print("%s not implemented" % (item))

    def base_files(self, split_file):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        :return: Lists of paths to the entries listed in split file
        """
        # Sim
        with open(split_file, "r") as f:
            prefix = [line.strip() for line in f]
            np.random.shuffle(prefix)

        if self.train:
            dataset = cfg.SIM.DATASET
        else:
            dataset = cfg.SIM.TESTSET
        img_L = [os.path.join(dataset, p, cfg.SIM.LEFT) for p in prefix]
        img_R = [os.path.join(dataset, p, cfg.SIM.RIGHT) for p in prefix]
        if self.train:
            img_L_no_ir = [
                os.path.join(dataset, p, cfg.SIM.LEFT_NO_IR) for p in prefix
            ]
            img_R_no_ir = [
                os.path.join(dataset, p, cfg.SIM.RIGHT_NO_IR) for p in prefix
            ]
        else:
            img_sim_realsense = [os.path.join(dataset, p, cfg.SIM.REALSENSE) for p in prefix]

        img_depth_l = [os.path.join(dataset, p, cfg.SIM.DEPTHL) for p in prefix]
        img_depth_r = [os.path.join(dataset, p, cfg.SIM.DEPTHR) for p in prefix]
        img_meta = [os.path.join(dataset, p, cfg.SIM.META) for p in prefix]

        if self.debug is True:
            img_L = img_L[:self.sub]
            img_R = img_R[:self.sub]
            if self.train:
                img_L_no_ir = img_L_no_ir[:self.sub]
                img_R_no_ir = img_R_no_ir[:self.sub]
            else:
                img_sim_realsense = img_sim_realsense[:self.sub]
            img_depth_l = img_depth_l[:self.sub]
            img_depth_r = img_depth_r[:self.sub]
            img_meta = img_meta[:self.sub]
            prefix = prefix[:self.sub]
        if self.train:
            return img_L, img_R, img_L_no_ir, img_R_no_ir, img_depth_l, img_depth_r, img_meta, prefix
        else:
            img_L.sort()
            img_R.sort()
            img_depth_l.sort()
            img_depth_r.sort()
            img_meta.sort()
            prefix.sort()
            img_sim_realsense.sort()
            return img_L, img_R, img_depth_l, img_depth_r, img_meta, prefix, img_sim_realsense

    def real_files(self, split_file):
        prefix = self.prefix

        if self.train:
            dataset = cfg.REAL.DATASET
        else:
            dataset = cfg.REAL.TESTSET
            img_label = [os.path.join(cfg.REAL.LABELSET, p, cfg.SIM.LABEL) for p in prefix]

        with open(split_file, "r") as f:
            prefix = [line.strip() for line in f]
            np.random.shuffle(prefix)

        img_real_L = [os.path.join(dataset, p, cfg.REAL.LEFT) for p in prefix]
        img_real_R = [os.path.join(dataset, p, cfg.REAL.RIGHT) for p in prefix]
        if self.train:
            img_real_L_no_ir = [
                os.path.join(dataset, p, cfg.REAL.LEFT_NO_IR) for p in prefix
            ]
            img_real_R_no_ir = [
                os.path.join(dataset, p, cfg.REAL.RIGHT_NO_IR) for p in prefix
            ]

            if "temporal" in self.special:
                img_real_L_temporal_ir = [
                    os.path.join(dataset, p, cfg.REAL.LEFT_TEMPORAL_IR) for p in prefix
                ]
                img_real_R_temporal_ir = [
                    os.path.join(dataset, p, cfg.REAL.RIGHT_TEMPORAL_IR) for p in prefix
                ]
        else:
            img_real_realsense = [os.path.join(dataset, p, cfg.REAL.REALSENSE) for p in prefix]
            with open(cfg.REAL.MASK_FILE, 'r') as f:
                mask_scenes = [line.strip() for line in f]

        if self.debug is True:

            img_real_L = img_real_L[:self.sub]
            img_real_R = img_real_R[:self.sub]
            if self.train:
                img_real_L_no_ir = img_real_L_no_ir[:self.sub]
                img_real_R_no_ir = img_real_R_no_ir[:self.sub]
                if "temporal" in self.special:
                    img_real_L_temporal_ir = img_real_L_temporal_ir[:self.sub]
                    img_real_R_temporal_ir = img_real_R_temporal_ir[:self.sub]
            else:
                img_real_realsense = img_real_realsense[:self.sub]
                img_label = img_label[:self.sub]

        if self.train:
            if "temporal" in self.special:
                return [img_real_L, img_real_R, img_real_L_no_ir, img_real_R_no_ir,
                        img_real_L_temporal_ir, img_real_R_temporal_ir]
            else:
                return img_real_L, img_real_R, img_real_L_no_ir, img_real_R_no_ir
        else:
            img_label.sort()
            img_real_L.sort()
            img_real_R.sort()
            img_real_realsense.sort()
            mask_scenes.sort()
            return img_label, img_real_L, img_real_R, img_real_realsense, mask_scenes

    def __len__(self):
        return len(self.img_sim_L)

    def __getitem__(self, idx):
        item = {}
        img_L = np.array(Image.open(self.img_sim_L[idx]).convert(mode="L")) / 255  # [H, W]
        img_R = np.array(Image.open(self.img_sim_R[idx]).convert(mode="L")) / 255
        if self.train:
            img_L_no_ir = np.array(Image.open(self.img_sim_L_no_ir[idx]).convert(mode="L")) / 255
            img_R_no_ir = np.array(Image.open(self.img_sim_R_no_ir[idx]).convert(mode="L")) / 255
        else:
            img_depth_sim_realsense = np.array(Image.open(self.img_sim_realsense[idx])) / 1000

        img_L_rgb = np.repeat(img_L[:, :, None], 3, axis=-1)
        img_R_rgb = np.repeat(img_R[:, :, None], 3, axis=-1)

        img_depth_l = np.array(Image.open(self.img_depth_l[idx])) / 1000 # convert from mm to m
        img_depth_r = np.array(Image.open(self.img_depth_r[idx])) / 1000 # convert from mm to m
        img_meta = load_pickle(self.img_meta[idx])

        # Convert depth map to disparity map (2x resolution)
        extrinsic_l = img_meta["extrinsic_l"]
        extrinsic_r = img_meta["extrinsic_r"]
        intrinsic_l = img_meta["intrinsic_l"]
        baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        focal_length = intrinsic_l[0, 0] / 2

        mask = img_depth_l > 0
        img_disp_l = np.zeros_like(img_depth_l)
        img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]
        mask = img_depth_r > 0
        img_disp_r = np.zeros_like(img_depth_r)
        img_disp_r[mask] = focal_length * baseline / img_depth_r[mask]

        # random crop the image to CROP_HEIGHT * CROP_WIDTH
        h, w = img_L_rgb.shape[:2]
        th, tw = cfg.MODEL.CROP_HEIGHT, cfg.MODEL.CROP_WIDTH
        x = random.randint(0, h - th)
        y = random.randint(0, w - tw)

        for temp in self.special:
            if temp=="temporal" or temp=="p1" or temp=="p2":
                img_sim_L_ir_pattern, img_sim_R_ir_pattern=  self.__getpattern__(
                    idx, temp, (w,h), img_L, img_R, img_L_no_ir, img_R_no_ir)
                img_sim_L_ir_pattern = img_sim_L_ir_pattern[x : (x + th), y : (y + tw)]
                img_sim_R_ir_pattern = img_sim_R_ir_pattern[x : (x + th), y : (y + tw)]
                item["img_sim_L_reproj"] = torch.tensor(
                    img_sim_L_ir_pattern, dtype=torch.float32
                ).unsqueeze(0)
                item["img_sim_R_reproj"] = torch.tensor(
                    img_sim_R_ir_pattern, dtype=torch.float32
                ).unsqueeze(0)
            elif temp=="img" or temp=="lcn":
                img_sim_L_pattern = img_L[x : (x + th), y : (y + tw)]
                img_sim_R_pattern = img_R[x : (x + th), y : (y + tw)]
                item["img_sim_L_reproj"] = torch.tensor(
                    img_sim_L_pattern, dtype=torch.float32
                ).unsqueeze(0)
                item["img_sim_R_reproj"] = torch.tensor(
                    img_sim_R_pattern, dtype=torch.float32
                ).unsqueeze(0)
                if temp=="lcn":
                    item["img_sim_L_reproj"], _ = local_contrast_norm(
                        item["img_sim_L_reproj"].unsqueeze(0),
                        kernel_size=cfg.LOSSES.REPROJECTION.PATCH_SIZE)
                    item["img_sim_L_reproj"] = item["img_sim_L_reproj"].squeeze(0)
                    item["img_sim_R_reproj"], _ = local_contrast_norm(
                        item["img_sim_R_reproj"].unsqueeze(0),
                        kernel_size=cfg.LOSSES.REPROJECTION.PATCH_SIZE)
                    item["img_sim_R_reproj"] = item["img_sim_R_reproj"].squeeze(0)

        if self.train:
            img_L_rgb = img_L_rgb[x : (x + th), y : (y + tw)]
            img_R_rgb = img_R_rgb[x : (x + th), y : (y + tw)]

            img_disp_l = img_disp_l[
                2 * x : 2 * (x + th), 2 * y : 2 * (y + tw)
            ]  # depth original res in 1080*1920
            img_depth_l = img_depth_l[2 * x : 2 * (x + th), 2 * y : 2 * (y + tw)]
            img_disp_r = img_disp_r[2 * x : 2 * (x + th), 2 * y : 2 * (y + tw)]
            img_depth_r = img_depth_r[2 * x : 2 * (x + th), 2 * y : 2 * (y + tw)]

        # Get data augmentation
        custom_augmentation = data_augmentation(
            gaussian_blur=cfg.DATA_AUG.GAUSSIAN_BLUR, color_jitter=cfg.DATA_AUG.COLOR_JITTER
        )

        if self.train:
            item["img_sim_L"] = custom_augmentation(img_L_rgb).type(torch.FloatTensor)
            item["img_sim_R"] = custom_augmentation(img_R_rgb).type(torch.FloatTensor)
        else:
            img_depth_sim_realsense = calc_left_ir_depth_from_rgb(img_meta['intrinsic'],
                                                                    img_meta['intrinsic_l'],
                                                                    img_meta['extrinsic'],
                                                                    img_meta['extrinsic_l'],
                                                                    img_depth_sim_realsense)
            item["img_depth_sim_realsense"] = img_depth_sim_realsense
            normalization = data_augmentation(gaussian_blur=False, color_jitter=False)
            item["img_sim_L"] = normalization(img_L_rgb).type(torch.FloatTensor)
            item["img_sim_R"] = normalization(img_R_rgb).type(torch.FloatTensor)
        item["img_disp_L"] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W] in dataloader
        item["img_depth_L"] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item["img_disp_R"] = torch.tensor(img_disp_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item["img_depth_R"] = torch.tensor(img_depth_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item["prefix"] = self.img_sim_L[idx].split("/")[-2]
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
        item['intrinsic'] = img_meta['intrinsic']
        item['intrinsic_l'] = img_meta['intrinsic_l']
        item['extrinsic'] = img_meta['extrinsic']
        item['extrinsic_l'] = img_meta['extrinsic_l']

        if self.onReal:
            item = self.__getitemreal__(item, (th, tw, x, y), idx)

        return item

    def __getitemreal__(self, item, crop, idx):
        (th, tw, x, y) = crop
        if self.train:
            real_idx = np.random.randint(0, high=self.real_len)
        else:
            real_idx = idx
        img_real_L = Image.open(self.img_real_L[real_idx]).convert(mode="L")
        img_real_R = Image.open(self.img_real_R[real_idx]).convert(mode="L")
        if self.train:
            img_real_L_no_ir = Image.open(self.img_real_L_no_ir[real_idx]).convert(mode="L")
            img_real_R_no_ir = Image.open(self.img_real_R_no_ir[real_idx]).convert(mode="L")
        else:
            img_depth_real_realsense = np.array(Image.open(self.img_real_realsense[real_idx])) / 1000
            img_label = np.array(Image.open(self.img_label[real_idx]))


        if self.train:
            # Resize real images, original 720 * 1280
            w, h = img_real_L.size
            h = int(h * 0.75)
            w = int(w * 0.75)
            img_real_L = img_real_L.resize((w, h), resample=Image.BILINEAR)
            img_real_R = img_real_R.resize((w, h), resample=Image.BILINEAR)
            img_real_L = np.array(img_real_L) / 255
            img_real_R = np.array(img_real_R) / 255
        else:
            img_real_L = np.array(img_real_L)
            img_real_R = np.array(img_real_R)

        if self.train:
            img_real_L_no_ir = img_real_L_no_ir.resize((w, h), resample=Image.BILINEAR)
            img_real_R_no_ir = img_real_R_no_ir.resize((w, h), resample=Image.BILINEAR)
            img_real_L_no_ir = np.array(img_real_L_no_ir) / 255
            img_real_R_no_ir = np.array(img_real_R_no_ir) / 255
        else:
            img_depth_real_realsense = calc_left_ir_depth_from_rgb(item['intrinsic'],
                                                                    item['intrinsic_l'],
                                                                    item['extrinsic'],
                                                                    item['extrinsic_l'],
                                                                    img_depth_real_realsense)
            item["img_depth_real_realsense"] = img_depth_real_realsense
            prefix = self.img_real_L[real_idx].split('/')[-2]
            scene_id = prefix.split('-')[-1]
            if scene_id in self.mask_scenes:
                robot_mask_file = os.path.join(cfg.REAL.MASK, scene_id + '.png')
                robot_mask = Image.open(robot_mask_file).convert(mode='L')
                c, h, w = item["img_depth_L"].shape
                robot_mask = robot_mask.resize((w,h), resample=Image.BILINEAR)
                robot_mask = np.array(robot_mask) / 255.
            else:
                robot_mask = np.zeros_like(img_label) / 1.
            item["robot_mask"] = robot_mask

        for temp in self.special:
            if temp=="temporal" or temp=="p1" or temp=="p2":
                img_real_L_ir_pattern, img_real_R_ir_pattern=  self.__getpattern__(
                    real_idx, temp, (w,h), img_real_L, img_real_R,
                    img_real_L_no_ir, img_real_R_no_ir, onSim=False)
                img_real_L_ir_pattern = img_real_L_ir_pattern[x : (x + th), y : (y + tw)]
                img_real_R_ir_pattern = img_real_R_ir_pattern[x : (x + th), y : (y + tw)]
                item["img_real_L_reproj"] = torch.tensor(
                    img_real_L_ir_pattern, dtype=torch.float32
                ).unsqueeze(0)
                item["img_real_R_reproj"] = torch.tensor(
                    img_real_R_ir_pattern, dtype=torch.float32
                ).unsqueeze(0)
            elif temp=="img" or temp=="lcn":
                img_real_L_pattern = img_real_L[x : (x + th), y : (y + tw)]
                img_real_R_pattern = img_real_R[x : (x + th), y : (y + tw)]
                item["img_real_L_reproj"] = torch.tensor(
                    img_real_L_pattern, dtype=torch.float32
                ).unsqueeze(0)
                item["img_real_R_reproj"] = torch.tensor(
                    img_real_R_pattern, dtype=torch.float32
                ).unsqueeze(0)
                if temp=="lcn":
                    item["img_real_L_reproj"], _ = local_contrast_norm(
                        item["img_real_L_reproj"].unsqueeze(0),
                        kernel_size=cfg.LOSSES.REPROJECTION.PATCH_SIZE)
                    item["img_real_L_reproj"] = item["img_real_L_reproj"].squeeze(0)
                    item["img_real_R_reproj"], _ = local_contrast_norm(
                        item["img_real_R_reproj"].unsqueeze(0),
                        kernel_size=cfg.LOSSES.REPROJECTION.PATCH_SIZE)
                    item["img_real_R_reproj"] = item["img_real_R_reproj"].squeeze(0)

        img_real_L = np.repeat(img_real_L[:, :, None], 3, axis=-1)
        img_real_R = np.repeat(img_real_R[:, :, None], 3, axis=-1)

        if self.train:
            img_real_L = img_real_L[x : (x + th), y : (y + tw)]
            img_real_R = img_real_R[x : (x + th), y : (y + tw)]
        else:
            item['img_label'] = torch.tensor(img_label, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]

        normalization = data_augmentation(gaussian_blur=False, color_jitter=False)
        item["img_real_L"] = normalization(img_real_L).type(torch.FloatTensor)
        item["img_real_R"] = normalization(img_real_R).type(torch.FloatTensor)

        return item

    def __getpattern__(self, idx, item, size, img_L, img_R, img_L_no_ir, img_R_no_ir, onSim=True):
        w, h = size
        if item=="temporal" and not onSim:
            img_L_temporal_ir = Image.open(self.img_real_L_temporal_ir[idx]).convert(mode="L")
            img_R_temporal_ir = Image.open(self.img_real_R_temporal_ir[idx]).convert(mode="L")
            img_L_temporal_ir = img_L_temporal_ir.resize(
                (w, h), resample=Image.BILINEAR
            )
            img_R_temporal_ir = img_R_temporal_ir.resize(
                (w, h), resample=Image.BILINEAR
            )
            img_L_ir_pattern = np.array(img_L_temporal_ir) / 255
            img_R_ir_pattern = np.array(img_R_temporal_ir) / 255
        elif item=="p1":
            img_L_ir_pattern = get_ir_pattern(img_L, img_L_no_ir)  # [H, W]
            img_R_ir_pattern = get_ir_pattern(img_R, img_R_no_ir)  # [H, W]
        elif item=="p2" or (item=="temporal" and onSim):
            img_L_ir_pattern = get_smoothed_ir_pattern2(img_L, img_L_no_ir)  # [H, W]
            img_R_ir_pattern = get_smoothed_ir_pattern2(img_R, img_R_no_ir)  # [H, W]

        return img_L_ir_pattern, img_R_ir_pattern

if __name__ == "__main__":
    cdataset = MessytableDataset(cfg.SIM.TRAIN, onReal=True)
    item = cdataset.__getitem__(0)

    for key, value in item.items():
        print(key)
        if type(value)!=str:
            print(value.shape)
