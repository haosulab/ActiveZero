"""
Author: Isabella Liu 8/8/21
Feature: Some util functions
"""

import logging
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils

from .reduce import tensor2float, tensor2numpy


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def get_time_string():
    """
    :return: Datetime in '%d_%m_%Y_%H_%M_%S' format
    """
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")
    return dt_string


def setup_logger(name, distributed_rank, save_dir=None):
    """
    Set up logger for the experiment
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def adjust_learning_rate(optimizer, epoch, base_lr, lrepochs):
    splits = lrepochs.split(":")
    assert len(splits) == 2

    # parse the epochs to downscale the learning rate (before :)
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(",")]
    # parse downscale rate (after :)
    downscale_rate = float(splits[1])
    # print("downscale epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rate))

    lr = base_lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    # print("setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_scalars(logger, mode_tag, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for tag, values in scalar_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            scalar_name = "{}/{}".format(mode_tag, tag)
            # if len(values) > 1:
            scalar_name = (
                scalar_name + "_" + str(idx) if len(values) > 1 else scalar_name
            )
            logger.add_scalar(scalar_name, value, global_step)


def save_scalars_graph(logger, mode_tag, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    logger.add_scalars(mode_tag, scalar_dict, global_step)


def save_images(logger, mode_tag, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value)

            image_name = "{}/{}".format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            logger.add_image(
                image_name,
                vutils.make_grid(
                    value, padding=0, nrow=1, normalize=True, scale_each=True
                ),
                global_step,
            )


def save_images_grid(logger, mode_tag, images_dict, global_step, nrow=5):
    for tag, dict_value in images_dict.items():
        img_list = []
        grid_name = f"{mode_tag}/{tag}"
        for subtag, img_value in dict_value.items():
            img_list += [img_value[0]]
        logger.add_image(
            grid_name,
            vutils.make_grid(
                img_list, padding=0, nrow=nrow, normalize=True, scale_each=True
            ),
            global_step,
        )


def gen_error_colormap_depth():
    cols = np.array(
        [
            [0, 0.00001, 0, 0, 0],
            [0.00001, 2000.0 / (2 ** 10), 49, 54, 149],
            [2000.0 / (2 ** 10), 2000.0 / (2 ** 9), 69, 117, 180],
            [2000.0 / (2 ** 9), 2000.0 / (2 ** 8), 116, 173, 209],
            [2000.0 / (2 ** 8), 2000.0 / (2 ** 7), 171, 217, 233],
            [2000.0 / (2 ** 7), 2000.0 / (2 ** 6), 224, 243, 248],
            [2000.0 / (2 ** 6), 2000.0 / (2 ** 5), 254, 224, 144],
            [2000.0 / (2 ** 5), 2000.0 / (2 ** 4), 253, 174, 97],
            [2000.0 / (2 ** 4), 2000.0 / (2 ** 3), 244, 109, 67],
            [2000.0 / (2 ** 3), 2000.0 / (2 ** 2), 215, 48, 39],
            [2000.0 / (2 ** 2), np.inf, 165, 0, 38],
        ],
        dtype=np.float32,
    )
    cols[:, 2:5] /= 255.0
    return cols


def gen_error_colormap_disp():
    cols = np.array(
        [
            [0, 0.00001, 0, 0, 0],
            [0.00001, 0.1875 / 3.0, 49, 54, 149],
            [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
            [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
            [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
            [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
            [3 / 3.0, 6 / 3.0, 254, 224, 144],
            [6 / 3.0, 12 / 3.0, 253, 174, 97],
            [12 / 3.0, 24 / 3.0, 244, 109, 67],
            [24 / 3.0, 48 / 3.0, 215, 48, 39],
            [48 / 3.0, np.inf, 165, 0, 38],
        ],
        dtype=np.float32,
    )
    cols[:, 2:5] /= 255.0
    return cols


def depth_error_img(D_est_tensor, D_gt_tensor, mask, abs_thres=1.0, dilate_radius=1):
    D_gt_np = D_gt_tensor.squeeze(0).detach().cpu().numpy()
    D_est_np = D_est_tensor.squeeze(0).detach().cpu().numpy()
    mask = mask.squeeze(0).detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = (D_gt_np > 0) & (D_gt_np < 1250)
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = error[mask] / abs_thres
    # get colormap
    cols = gen_error_colormap_depth()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[
            i, 2:
        ]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.0
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance : (i + 1) * distance, :] = cols[i, 2:]
    return error_image[0]  # [H, W, 3]


def disp_error_img(
    D_est_tensor, D_gt_tensor, mask, abs_thres=3.0, rel_thres=0.05, dilate_radius=1
):
    D_gt_np = D_gt_tensor.squeeze(0).detach().cpu().numpy()
    D_est_np = D_est_tensor.squeeze(0).detach().cpu().numpy()
    mask = mask.squeeze(0).detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = D_gt_np > 0
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(
        error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres
    )
    # get colormap
    cols = gen_error_colormap_disp()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[
            i, 2:
        ]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.0
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance : (i + 1) * distance, :] = cols[i, 2:]
    return error_image[0]  # [H, W, 3]
