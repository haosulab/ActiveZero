"""
Author: Isabella Liu 11/19/21
Feature:
"""

import os
import random

import numpy as np
import torch
import torchvision.transforms as Transforms
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from utils.config import cfg
from utils.util import load_pickle

# TODO: is this file still in use?
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


class MessytableSequenceDataset(Dataset):
    def __init__(self, seq_folder, idx_s=0, idx_e=100, debug=False, sub=10):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        (
            self.img_L_real,
            self.img_R_real,
            self.img_real_realsense,
        ) = self.__get_split_files__(
            seq_folder, idx_s=idx_s, idx_e=idx_e, debug=debug, sub=sub
        )

        # Load meta file
        meta_file = os.path.join(seq_folder, "meta.pkl")
        img_meta = load_pickle(meta_file)
        extrinsic_l = img_meta["extrinsic_l"]
        extrinsic_r = img_meta["extrinsic_r"]
        intrinsic_l = img_meta["intrinsic_l"]
        self.baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        self.focal_length = intrinsic_l[0, 0] / 2

    @staticmethod
    def __get_split_files__(seq_folder, idx_s=0, idx_e=100, debug=False, sub=10):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        :return: Lists of paths to the entries listed in split file
        """
        suffix_list = [f"{i+1:04}" for i in range(idx_s, idx_e)]
        img_L_real = [
            os.path.join(seq_folder, "1024_irL_real_" + s + ".png") for s in suffix_list
        ]
        img_R_real = [
            os.path.join(seq_folder, "1024_irR_real_" + s + ".png") for s in suffix_list
        ]
        img_real_realsense = [
            os.path.join(seq_folder, "1024_depth_real_" + s + ".png")
            for s in suffix_list
        ]

        if debug is True:
            img_L_real = img_L_real[:sub]
            img_R_real = img_R_real[:sub]
            img_real_realsense = img_real_realsense[:sub]

        return img_L_real, img_R_real, img_real_realsense

    def __len__(self):
        return len(self.img_L_real)

    def __getitem__(self, idx):
        img_L_rgb = Image.open(self.img_L_real[idx]).convert(mode="L")
        img_R_rgb = Image.open(self.img_R_real[idx]).convert(mode="L")

        # Resize the input image
        w, h = img_L_rgb.size
        h = int(h * 0.75)
        w = int(w * 0.75)
        img_L_rgb = img_L_rgb.resize((w, h), resample=Image.BILINEAR)
        img_R_rgb = img_R_rgb.resize((w, h), resample=Image.BILINEAR)

        img_L_rgb = np.array(img_L_rgb) / 255
        img_R_rgb = np.array(img_R_rgb) / 255
        img_L_rgb = img_L_rgb[:, :, None]
        img_R_rgb = img_R_rgb[:, :, None]
        img_L_rgb = np.repeat(img_L_rgb, 3, axis=-1)
        img_R_rgb = np.repeat(img_R_rgb, 3, axis=-1)

        img_depth_realsense = np.array(Image.open(self.img_real_realsense[idx])) / 1000

        # Get data augmentation
        # custom_augmentation = __data_augmentation__(gaussian_blur=self.gaussian_blur, color_jitter=self.color_jitter)
        normalization = __data_augmentation__(gaussian_blur=False, color_jitter=False)

        item = {}
        item["img_L"] = normalization(img_L_rgb).type(torch.FloatTensor)
        item["img_R"] = normalization(img_R_rgb).type(torch.FloatTensor)
        item["img_depth_realsense"] = torch.tensor(
            img_depth_realsense, dtype=torch.float32
        ).unsqueeze(
            0
        )  # [bs, 1, H, W]
        item["focal_length"] = (
            torch.tensor(self.focal_length, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        item["baseline"] = (
            torch.tensor(self.baseline, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return item


def get_seq_loader(seq_folder, idx_s=0, idx_e=100, debug=False, sub=10):
    """
    :param split_file: split file
    :param debug: Whether on debug mode, load less data
    :param sub: If on debug mode, how many items to load into dataset
    :param isTest: Whether on test, if test no random crop on input image
    :param onReal: Whether test on real dataset, folder and file name are different
    :return: dataloader
    """
    messytable_dataset = MessytableSequenceDataset(
        seq_folder=seq_folder, idx_s=idx_s, idx_e=idx_e, debug=debug, sub=sub
    )
    loader = DataLoader(messytable_dataset, batch_size=1, num_workers=0)
    return loader


if __name__ == "__main__":
    cdataset = MessytableSequenceDataset("/code/supp_video/3/", idx_s=0, idx_e=254)
    item = cdataset.__getitem__(0)
    print(item["img_L"].shape)
    print(item["img_R"].shape)
    print(item["img_depth_realsense"].shape)
    print(item["baseline"].shape)
