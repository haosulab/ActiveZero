"""
Author: Isabella Liu 11/18/21
Feature: Test PSMNet + re-projection in sequence
"""

# same as temporal_ir.py
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# from nets.psmnet import PSMNet
from nets.psmnet_confidence import PSMNet
from nets.transformer import Transformer
from supplementary.messytable_sequence import get_seq_loader
from utils.cascade_metrics import compute_err_metric, compute_obj_err
from utils.config import cfg
from utils.test_util import (load_from_dataparallel_model, save_gan_img,
                             save_img, save_obj_err_file)
from utils.util import (depth_error_img, disp_error_img, get_time_string,
                        setup_logger)

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
    "--output", type=str, default="../2_4_sequence", help="Path to output folder"
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

# python supplementary/test_sequence.py --model /code/model_40000.pth --onreal --exclude-bg --exclude-zeros

MAX_DEPTH = 2.0


def visualize_depth(depth):
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth


def test(transformer_model, psmnet_model, val_loader, logger, log_dir):
    transformer_model.eval()
    psmnet_model.eval()
    os.mkdir(os.path.join(log_dir, "pred_depth"))
    os.mkdir(os.path.join(log_dir, "realsense_depth"))

    for iteration, data in enumerate(tqdm(val_loader)):
        logger.info(f"Processsing {iteration} / {len(val_loader)}")

        img_L = data["img_L"].cuda()  # [bs, 1, H, W]
        img_R = data["img_R"].cuda()
        img_focal_length = data["focal_length"].cuda()
        img_baseline = data["baseline"].cuda()

        img_depth_realsense = data["img_depth_realsense"].cuda()

        img_depth_realsense = F.interpolate(
            img_depth_realsense,
            (540, 960),
            mode="nearest",
            recompute_scale_factor=False,
        )

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

        with torch.no_grad():
            pred_disp, pred_conf = psmnet_model(
                img_L, img_R, img_L_transformed, img_R_transformed
            )
        pred_disp = pred_disp[
            :, :, top_pad:, :
        ]  # TODO: if right_pad > 0 it needs to be (:-right_pad)
        pred_depth = img_focal_length * img_baseline / pred_disp  # pred depth in m

        # Get depth image

        pred_depth_np = (
            pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()
        )  # in m, [H, W]
        # pred_depth_np = (pred_depth_np * 1000.0).astype(np.uint16)
        # pred_depth_np = pred_depth_np[:, :]
        pred_depth_np = visualize_depth(pred_depth_np)

        # TO BE deleted
        pred_conf_np = (
            pred_conf.squeeze(0).squeeze(0).detach().cpu().numpy()
        )  # in m, [H, W]
        # pred_depth_np = (pred_depth_np * 1000.0).astype(np.uint16)
        # pred_depth_np = pred_depth_np[:, :]
        pred_conf_np = visualize_depth(pred_conf_np)

        # Get depth ground truth image
        realsense_depth_np = (
            img_depth_realsense.squeeze(0).squeeze(0).detach().cpu().numpy()
        )
        # realsense_depth_np = (realsense_depth_np * 1000.0).astype(np.uint16)
        # realsense_depth_np = realsense_depth_np[:, :]
        realsense_depth_np = visualize_depth(realsense_depth_np)

        # Save images
        cv2.imwrite(
            os.path.join(log_dir, "pred_depth", f"{iteration:04}.png"), pred_depth_np
        )
        cv2.imwrite(
            os.path.join(log_dir, "pred_depth", f"conf_{iteration:04}.png"),
            pred_conf_np,
        )
        cv2.imwrite(
            os.path.join(log_dir, "realsense_depth", f"{iteration:04}.png"),
            realsense_depth_np,
        )
        # plt.imsave(os.path.join(log_dir, 'pred_depth', f'{iteration:04}.png'), pred_depth_np, cmap='viridis', vmin=0, vmax=1.25)
        # plt.imsave(os.path.join(log_dir, 'realsense_depth', f'{iteration:04}.png'), realsense_depth_np, cmap='viridis', vmin=0, vmax=1.25)
        # plt.close('all')


def main():
    # Obtain the dataloader
    val_loader = get_seq_loader(
        "/code/supp_video/3/", idx_s=0, idx_e=204, debug=args.debug, sub=10
    )
    # val_loader = get_seq_loader('/code/supp_video/4/', idx_s=0, idx_e=203, debug=args.debug, sub=10)

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

    test(transformer_model, psmnet_model, val_loader, logger, log_dir)


if __name__ == "__main__":
    main()
