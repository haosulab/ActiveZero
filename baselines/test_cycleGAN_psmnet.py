"""
Author: Isabella Liu 11/12/21
Feature: Test CycleGAN + PSMNet
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from psmnet import PSMNet
from tqdm import tqdm

from datasets.messytable_test import get_test_loader
from nets.cycle_gan import CycleGANModel
from utils.cascade_metrics import compute_err_metric, compute_obj_err
from utils.config import cfg
from utils.test_util import (load_from_dataparallel_model, save_gan_img,
                             save_img, save_obj_err_file)
from utils.util import (depth_error_img, disp_error_img, get_time_string,
                        setup_logger)
from utils.warp_ops import apply_disparity_cu

parser = argparse.ArgumentParser(description="Testing for CycleGAN + PSMNet")
parser.add_argument(
    "--config-file",
    type=str,
    default="./configs/local_test.yaml",
    metavar="FILE",
    help="Config files",
)
parser.add_argument(
    "--model", type=str, default="", metavar="FILE", help="Path to test model"
)
parser.add_argument(
    "--gan-model", type=str, default="", metavar="FILE", help="Path to test gan model"
)
parser.add_argument(
    "--output",
    type=str,
    default="../testing_output_11_12_cycleGAN",
    help="Path to output folder",
)
parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")
parser.add_argument(
    "--annotate", type=str, default="", help="Annotation to the experiment"
)
parser.add_argument(
    "--onreal", action="store_true", default=False, help="Test on real dataset"
)
parser.add_argument(
    "--analyze-objects",
    action="store_true",
    default=True,
    help="Analyze on different objects",
)
parser.add_argument(
    "--exclude-bg",
    action="store_true",
    default=False,
    help="Exclude background when testing",
)
parser.add_argument(
    "--warp-op",
    action="store_true",
    default=True,
    help="Use warp_op function to get disparity",
)
parser.add_argument(
    "--exclude-zeros",
    action="store_true",
    default=False,
    help="Whether exclude zero pixels in realsense",
)
parser.add_argument(
    "--local_rank", type=int, default=0, help="Rank of device in distributed training"
)

args = parser.parse_args()
cfg.merge_from_file(args.config_file)
cuda_device = torch.device("cuda:{}".format(args.local_rank))
# If path to gan model is not specified, use gan model from cascade model
if args.gan_model == "":
    args.gan_model = args.model

# Calculate error for real and 3D printed objects
real_obj_id = [4, 5, 7, 9, 13, 14, 15, 16]

# python test_psmnet_reprojection.py --model /code/models/model_4.pth --onreal --exclude-bg --exclude-zeros
# python test_psmnet_reprojection.py --config-file configs/remote_test.yaml --model ../train_8_14_cascade/train1/models/model_best.pth --onreal --exclude-bg --exclude-zeros --debug --gan-model


def test(gan_model, psmnet_model, val_loader, logger, log_dir):
    gan_model.eval()
    psmnet_model.eval()
    total_err_metrics = {
        "epe": 0,
        "bad1": 0,
        "bad2": 0,
        "depth_abs_err": 0,
        "depth_err2": 0,
        "depth_err4": 0,
        "depth_err8": 0,
    }
    total_obj_disp_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_4_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_count = np.zeros(cfg.SPLIT.OBJ_NUM)
    os.mkdir(os.path.join(log_dir, "pred_disp"))
    os.mkdir(os.path.join(log_dir, "gt_disp"))
    os.mkdir(os.path.join(log_dir, "pred_disp_abs_err_cmap"))
    os.mkdir(os.path.join(log_dir, "pred_depth"))
    os.mkdir(os.path.join(log_dir, "gt_depth"))
    os.mkdir(os.path.join(log_dir, "pred_depth_abs_err_cmap"))
    os.mkdir(os.path.join(log_dir, "gan"))

    for iteration, data in enumerate(tqdm(val_loader)):
        img_L = data["img_L"].cuda()  # [bs, 1, H, W]
        img_R = data["img_R"].cuda()

        img_disp_l = data["img_disp_l"].cuda()
        img_depth_l = data["img_depth_l"].cuda()
        img_depth_realsense = data["img_depth_realsense"].cuda()
        img_label = data["img_label"].cuda()
        img_focal_length = data["focal_length"].cuda()
        img_baseline = data["baseline"].cuda()
        prefix = data["prefix"][0]
        robot_mask = data["robot_mask"].cuda()

        img_disp_l = F.interpolate(
            img_disp_l, (540, 960), mode="nearest", recompute_scale_factor=False
        )
        img_depth_l = F.interpolate(
            img_depth_l, (540, 960), mode="nearest", recompute_scale_factor=False
        )
        img_depth_realsense = F.interpolate(
            img_depth_realsense,
            (540, 960),
            mode="nearest",
            recompute_scale_factor=False,
        )
        img_label = F.interpolate(
            img_label, (540, 960), mode="nearest", recompute_scale_factor=False
        ).type(torch.int)
        img_robot_mask = F.interpolate(
            robot_mask, (540, 960), mode="nearest", recompute_scale_factor=False
        ).type(torch.int)

        # If using warp_op, computing img_disp_l from img_disp_r
        if args.warp_op:
            img_disp_r = data["img_disp_r"].cuda()
            img_depth_r = data["img_depth_r"].cuda()
            img_disp_r = F.interpolate(
                img_disp_r, (540, 960), mode="nearest", recompute_scale_factor=False
            )
            img_depth_r = F.interpolate(
                img_depth_r, (540, 960), mode="nearest", recompute_scale_factor=False
            )
            img_disp_l = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))
            img_depth_l = apply_disparity_cu(
                img_depth_r, img_disp_r.type(torch.int)
            )  # [bs, 1, H, W]

        # If test on real dataset need to crop input image to (540, 960)
        if args.onreal:
            img_L = F.interpolate(
                img_L,
                (540, 960),
                mode="bilinear",
                recompute_scale_factor=False,
                align_corners=False,
            )
            img_R = F.interpolate(
                img_R,
                (540, 960),
                mode="bilinear",
                recompute_scale_factor=False,
                align_corners=False,
            )
        else:
            with torch.no_grad():
                img_L_real = data["img_L_real"].cuda()
                img_R_real = data["img_R_real"].cuda()
                input_sample = {
                    "img_L": img_L,
                    "img_R": img_R,
                    "img_real_L": img_L_real,
                    "img_real_R": img_R_real,
                }
                gan_model.set_input(input_sample)
                gan_model.forward()
                img_L = gan_model.fake_B_L.detach()
                img_R = gan_model.fake_B_R.detach()

                img_outputs = {
                    "img_sim": {
                        "input": gan_model.real_A_L,
                        "fake": gan_model.fake_B_L,
                    },
                    "img_Real": {
                        "input": gan_model.real_B_L,
                        "fake": gan_model.fake_A_L,
                    },
                }
                save_gan_img(img_outputs, os.path.join(log_dir, "gan", f"{prefix}.png"))

        # Pad the imput image and depth disp image to 960 * 544
        right_pad = cfg.REAL.PAD_WIDTH - 960
        top_pad = cfg.REAL.PAD_HEIGHT - 540
        img_L = F.pad(
            img_L, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode="constant", value=0
        )
        img_R = F.pad(
            img_R, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode="constant", value=0
        )

        robot_mask = img_robot_mask == 0
        if args.exclude_bg:
            # Mask ground pixel to False
            img_ground_mask = (img_depth_l > 0) & (img_depth_l < 1.25)
            mask = (
                (img_disp_l < cfg.ARGS.MAX_DISP)
                * (img_disp_l > 0)
                * img_ground_mask
                * robot_mask
            )
        else:
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) * robot_mask

        # Exclude uncertain pixel from realsense_depth_pred
        realsense_zeros_mask = img_depth_realsense > 0
        if args.exclude_zeros:
            mask = mask * realsense_zeros_mask
        mask = mask.type(torch.bool)

        ground_mask = (
            torch.logical_not(mask).squeeze(0).squeeze(0).detach().cpu().numpy()
        )

        with torch.no_grad():
            pred_disp = psmnet_model(img_L, img_R)
        pred_disp = pred_disp[
            :, :, top_pad:, :
        ]  # TODO: if right_pad > 0 it needs to be (:-right_pad)
        pred_depth = img_focal_length * img_baseline / pred_disp  # pred depth in m

        # Get loss metric
        err_metrics = compute_err_metric(
            img_disp_l, img_depth_l, pred_disp, img_focal_length, img_baseline, mask
        )
        for k in total_err_metrics.keys():
            total_err_metrics[k] += err_metrics[k]
        logger.info(f"Test instance {prefix} - {err_metrics}")

        # Get object error
        obj_disp_err, obj_depth_err, obj_depth_4_err, obj_count = compute_obj_err(
            img_disp_l,
            img_depth_l,
            pred_disp,
            img_focal_length,
            img_baseline,
            img_label,
            mask,
            cfg.SPLIT.OBJ_NUM,
        )
        total_obj_disp_err += obj_disp_err
        total_obj_depth_err += obj_depth_err
        total_obj_depth_4_err += obj_depth_4_err
        total_obj_count += obj_count

        # Get disparity image
        pred_disp_np = pred_disp.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
        pred_disp_np[ground_mask] = -1

        # Get disparity ground truth image
        gt_disp_np = img_disp_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_disp_np[ground_mask] = -1

        # Get disparity error image
        pred_disp_err_np = disp_error_img(pred_disp, img_disp_l, mask)

        # Get depth image
        pred_depth_np = (
            pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()
        )  # in m, [H, W]
        # crop depth map to [0.2m, 2m]
        # pred_depth_np[pred_depth_np < 0.2] = -1
        # pred_depth_np[pred_depth_np > 2] = -1
        pred_depth_np[ground_mask] = -1

        # Get depth ground truth image
        gt_depth_np = img_depth_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_depth_np[ground_mask] = -1

        # Get depth error image
        pred_depth_err_np = depth_error_img(pred_depth * 1000, img_depth_l * 1000, mask)

        # Save images
        save_img(
            log_dir,
            prefix,
            pred_disp_np,
            gt_disp_np,
            pred_disp_err_np,
            pred_depth_np,
            gt_depth_np,
            pred_depth_err_np,
        )

    # Get final error metrics
    for k in total_err_metrics.keys():
        total_err_metrics[k] /= len(val_loader)
    logger.info(f"\nTest on {len(val_loader)} instances\n {total_err_metrics}")

    # Save object error to csv file
    total_obj_disp_err /= total_obj_count
    total_obj_depth_err /= total_obj_count
    total_obj_depth_4_err /= total_obj_count
    save_obj_err_file(
        total_obj_disp_err, total_obj_depth_err, total_obj_depth_4_err, log_dir
    )

    logger.info(f"Successfully saved object error to obj_err.txt")

    # Get error on real and 3d printed objects
    real_depth_error = 0
    real_depth_error_4mm = 0
    printed_depth_error = 0
    printed_depth_error_4mm = 0
    for i in range(cfg.SPLIT.OBJ_NUM):
        if i in real_obj_id:
            real_depth_error += total_obj_depth_err[i]
            real_depth_error_4mm += total_obj_depth_4_err[i]
        else:
            printed_depth_error += total_obj_depth_err[i]
            printed_depth_error_4mm += total_obj_depth_4_err[i]
    real_depth_error /= len(real_obj_id)
    real_depth_error_4mm /= len(real_obj_id)
    printed_depth_error /= cfg.SPLIT.OBJ_NUM - len(real_obj_id)
    printed_depth_error_4mm /= cfg.SPLIT.OBJ_NUM - len(real_obj_id)

    logger.info(
        f"Real objects - absolute depth error: {real_depth_error}, depth 4mm: {real_depth_error_4mm} \n"
        f"3D printed objects - absolute depth error {printed_depth_error}, depth 4mm: {printed_depth_error_4mm}"
    )


def main():
    # Obtain the dataloader
    val_loader = get_test_loader(cfg.SPLIT.VAL, args.debug, sub=40, onReal=args.onreal)

    # Tensorboard and logger
    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, f"{get_time_string()}_{args.annotate}")
    os.mkdir(log_dir)
    logger = setup_logger(
        "CycleGAN-PSMNet Testing", distributed_rank=0, save_dir=log_dir
    )
    logger.info(f"Annotation: {args.annotate}")
    logger.info(f"Input args {args}")
    logger.info(f"Loaded config file '{args.config_file}'")
    logger.info(f"Running with configs:\n{cfg}")

    # Get GAN model
    gan_model = CycleGANModel()
    gan_model.set_device(cuda_device)
    gan_model.load_model(args.gan_model)

    # Get PSMNet model
    logger.info(f"Loaded the checkpoint: {args.model}")
    psmnet_model = PSMNet(maxdisp=cfg.ARGS.MAX_DISP).to(cuda_device)
    psmnet_model_dict = load_from_dataparallel_model(args.model, "PSMNet")
    psmnet_model.load_state_dict(psmnet_model_dict)

    test(gan_model, psmnet_model, val_loader, logger, log_dir)


if __name__ == "__main__":
    main()
