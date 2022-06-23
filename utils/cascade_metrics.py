"""
Author: Isabella Liu 8/14/21
Feature: Compute error metrics for cascades stereo network
"""

import numpy as np
import torch
import torch.nn.functional as F

from utils.reduce import make_nograd_func


# Error metric for messy-table-dataset
# TODO: Ignore instances with small mask? (@compute_metric_for_each_image)
@make_nograd_func
def compute_err_metric(
    disp_gt, depth_gt, disp_pred, focal_length, baseline, mask, depth_pred=None
):
    """
    Compute the error metrics for predicted disparity map
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param mask: Selected pixel
    :return: Error metrics
    """
    epe = F.l1_loss(disp_pred[mask], disp_gt[mask], reduction="mean").item()
    disp_diff = torch.abs(disp_gt[mask] - disp_pred[mask])  # [bs, 1, H, W]
    bad1 = disp_diff[disp_diff > 1].numel() / disp_diff.numel()
    bad2 = disp_diff[disp_diff > 2].numel() / disp_diff.numel()

    # get predicted depth map
    if depth_pred is None:
        depth_pred = focal_length * baseline / disp_pred  # in meters

    depth_abs_err = torch.clip(
        torch.abs(depth_gt[mask] * 1000 - depth_pred[mask] * 1000), min=0, max=100
    )
    depth_abs_err = torch.mean(depth_abs_err).item()
    # depth_pred = torch.clip(depth_pred, max=1.25)
    # depth_abs_err = F.l1_loss(depth_pred[mask] * 1000, depth_gt[mask] * 1000, reduction='mean').item()
    depth_diff = torch.abs(depth_gt[mask] - depth_pred[mask])  # [bs, 1, H, W]
    depth_err2 = depth_diff[depth_diff > 2e-3].numel() / depth_diff.numel()
    depth_err4 = depth_diff[depth_diff > 4e-3].numel() / depth_diff.numel()
    depth_err8 = depth_diff[depth_diff > 8e-3].numel() / depth_diff.numel()

    err = {}
    err["epe"] = epe
    err["bad1"] = bad1
    err["bad2"] = bad2
    err["depth_abs_err"] = depth_abs_err
    err["depth_err2"] = depth_err2
    err["depth_err4"] = depth_err4
    err["depth_err8"] = depth_err8
    return err


# Error metric for messy-table-dataset object error
@make_nograd_func
def compute_obj_err(
    disp_gt, depth_gt, disp_pred, focal_length, baseline, label, mask, obj_total_num=17
):
    """
    Compute error for each object instance in the scene
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param label: Label of the image [bs, 1, H, W]
    :param obj_total_num: Total number of objects in the dataset
    :return: obj_disp_err, obj_depth_err - List of error of each object
             obj_count - List of each object appear count
    """
    depth_pred = focal_length * baseline / disp_pred  # in meters

    obj_list = (
        label.unique()
    )  # TODO this will cause bug if bs > 1, currently only for testing
    obj_num = obj_list.shape[0]

    # Array to store error and count for each object
    total_obj_disp_err = np.zeros(obj_total_num)
    total_obj_depth_err = np.zeros(obj_total_num)
    total_obj_depth_4_err = np.zeros(obj_total_num)
    total_obj_count = np.zeros(obj_total_num)

    for i in range(obj_num):
        obj_id = int(obj_list[i].item())
        obj_mask = label == obj_id
        obj_disp_err = F.l1_loss(
            disp_gt[obj_mask * mask], disp_pred[obj_mask * mask], reduction="mean"
        ).item()
        obj_depth_err = torch.clip(
            torch.abs(
                depth_gt[obj_mask * mask] * 1000 - depth_pred[obj_mask * mask] * 1000
            ),
            min=0,
            max=100,
        )
        obj_depth_err = torch.mean(obj_depth_err).item()
        obj_depth_diff = torch.abs(
            depth_gt[obj_mask * mask] - depth_pred[obj_mask * mask]
        )
        obj_depth_err4 = (
            obj_depth_diff[obj_depth_diff > 4e-3].numel() / obj_depth_diff.numel()
        )

        total_obj_disp_err[obj_id] += obj_disp_err
        total_obj_depth_err[obj_id] += obj_depth_err
        total_obj_depth_4_err[obj_id] += obj_depth_err4
        total_obj_count[obj_id] += 1
    return (
        total_obj_disp_err,
        total_obj_depth_err,
        total_obj_depth_4_err,
        total_obj_count,
    )
