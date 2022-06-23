"""
Author: Isabella Liu 1/15/22
Feature: Test the temporal ir reproj model on transparent dataset
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nets.psmnet import PSMNet
from nets.transformer import Transformer
from transparent.dataset_transparent_temporal_ir import \
    MessytableTransparentDataset
from utils.cascade_metrics import compute_err_metric
from utils.config import cfg
from utils.test_util import (load_from_dataparallel_model, save_gan_img,
                             save_img, save_obj_err_file)
from utils.util import (depth_error_img, disp_error_img, get_time_string,
                        setup_logger)
from utils.warp_ops import apply_disparity_cu

parser = argparse.ArgumentParser(description="Testing for Reprojection + PSMNet")
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
    default="../testing_output_11_9_patch_loss",
    help="Path to output folder",
)
parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")
parser.add_argument(
    "--annotate", type=str, default="", help="Annotation to the experiment"
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
    "--local_rank", type=int, default=0, help="Rank of device in distributed training"
)

args = parser.parse_args()
cfg.merge_from_file(args.config_file)
cuda_device = torch.device("cuda:{}".format(args.local_rank))
# If path to gan model is not specified, use gan model from cascade model
if args.gan_model == "":
    args.gan_model = args.model

# python test_psmnet_reprojection.py --model /code/models/model_4.pth --exclude-bg --exclude-zeros
# python transparent/test_temporal_ir_reproj.py --config-file transparent/config_transparent.yaml --model /isabella-slow/FeatureGAN/transparent/train2_debug_1_15/models/model_40000.pth --output /isabella-slow/FeatureGAN/transparent/test_1_15/ --exclude-bg --debug


def test(transformer_model, psmnet_model, val_loader, logger, log_dir):
    transformer_model.eval()
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
    os.mkdir(os.path.join(log_dir, "pred_disp"))
    os.mkdir(os.path.join(log_dir, "gt_disp"))
    os.mkdir(os.path.join(log_dir, "pred_disp_abs_err_cmap"))
    os.mkdir(os.path.join(log_dir, "pred_depth"))
    os.mkdir(os.path.join(log_dir, "gt_depth"))
    os.mkdir(os.path.join(log_dir, "pred_depth_abs_err_cmap"))

    for iteration, data in enumerate(tqdm(val_loader)):
        img_L = data["img_L"].cuda().unsqueeze(0)  # [bs, 1, H, W]
        img_R = data["img_R"].cuda().unsqueeze(0)

        img_disp_l = data["img_disp_l"].cuda().unsqueeze(0)
        img_depth_l = data["img_depth_l"].cuda().unsqueeze(0)
        img_focal_length = data["focal_length"].cuda()
        img_baseline = data["baseline"].cuda()
        prefix = data["prefix"]

        img_disp_l = F.interpolate(
            img_disp_l, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
        )
        img_depth_l = F.interpolate(
            img_depth_l, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
        )

        # If using warp_op, computing img_disp_l from img_disp_r
        if args.warp_op:
            img_disp_r = data["img_disp_r"].cuda().unsqueeze(0)
            img_depth_r = data["img_depth_r"].cuda().unsqueeze(0)
            img_disp_r = F.interpolate(
                img_disp_r,
                scale_factor=0.5,
                mode="nearest",
                recompute_scale_factor=False,
            )
            img_depth_r = F.interpolate(
                img_depth_r,
                scale_factor=0.5,
                mode="nearest",
                recompute_scale_factor=False,
            )
            img_disp_l = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))
            img_depth_l = apply_disparity_cu(
                img_depth_r, img_disp_r.type(torch.int)
            )  # [bs, 1, H, W]

        with torch.no_grad():
            img_L_transformed, img_R_transformed = transformer_model(img_L, img_R)

        # Pad the imput image and depth disp image to 960 * 544
        right_pad = cfg.REAL.PAD_WIDTH - 960
        top_pad = cfg.REAL.PAD_HEIGHT - 540
        img_L = F.pad(
            img_L, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode="constant", value=0
        )
        img_R = F.pad(
            img_R, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode="constant", value=0
        )
        img_L_transformed = F.pad(
            img_L_transformed,
            (0, right_pad, top_pad, 0, 0, 0, 0, 0),
            mode="constant",
            value=0,
        )
        img_R_transformed = F.pad(
            img_R_transformed,
            (0, right_pad, top_pad, 0, 0, 0, 0, 0),
            mode="constant",
            value=0,
        )

        if args.exclude_bg:
            # Mask ground pixel to False
            img_ground_mask = (img_depth_l > 0) & (img_depth_l < 1.25)
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) * img_ground_mask
        else:
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0)

        mask = mask.type(torch.bool)

        ground_mask = (
            torch.logical_not(mask).squeeze(0).squeeze(0).detach().cpu().numpy()
        )

        with torch.no_grad():
            pred_disp = psmnet_model(img_L, img_R, img_L_transformed, img_R_transformed)
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


def main():
    # Obtain the dataloader
    test_loader = MessytableTransparentDataset(
        cfg.SPLIT.TEST,
        gaussian_blur=False,
        color_jitter=False,
        debug=args.debug,
        sub=10,
        test=True,
    )

    # Tensorboard and logger
    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, f"{get_time_string()}_{args.annotate}")
    os.mkdir(log_dir)
    logger = setup_logger(
        "Reprojection-PSMNet Testing", distributed_rank=0, save_dir=log_dir
    )
    logger.info(f"Annotation: {args.annotate}")
    logger.info(f"Input args {args}")
    logger.info(f"Loaded config file '{args.config_file}'")
    logger.info(f"Running with configs:\n{cfg}")

    # Get cascade model
    logger.info(f"Loaded the checkpoint: {args.model}")
    transformer_model = Transformer().to(cuda_device)
    psmnet_model = PSMNet(maxdisp=cfg.ARGS.MAX_DISP).to(cuda_device)
    transformer_model_dict = load_from_dataparallel_model(args.model, "Transformer")
    transformer_model.load_state_dict(transformer_model_dict)
    psmnet_model_dict = load_from_dataparallel_model(args.model, "PSMNet")
    psmnet_model.load_state_dict(psmnet_model_dict)

    test(transformer_model, psmnet_model, test_loader, logger, log_dir)


if __name__ == "__main__":
    main()
