import argparse
import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.messytable import MessytableDataset
from utils.cascade_metrics import compute_err_metric, compute_obj_err
from configs.config import cfg
from utils.test_util import (load_from_dataparallel_model, save_img, save_obj_err_file)
from utils.util import (depth_error_img, disp_error_img, get_time_string,
                        setup_logger)
from utils.warp_ops import apply_disparity_cu
from utils.losses import AllLosses # put all new losses here


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-file",
    type=str,
    default="./configs/temp.yaml",
    metavar="FILE",
    help="Config files",
)
parser.add_argument(
    "--local_rank", type=int, default=0, help="Rank of device in distributed training"
)

args = parser.parse_args()
cfg.merge_from_file(args.config_file)

cuda_device = torch.device("cuda:{}".format(args.local_rank))

# Tensorboard and logger
os.makedirs(cfg.SOLVER.LOGDIR, exist_ok=True)
log_dir = os.path.join(cfg.SOLVER.LOGDIR, f"{get_time_string()}")
os.mkdir(log_dir)
logger = setup_logger(
    cfg.NAME, distributed_rank=args.local_rank, save_dir=cfg.SOLVER.LOGDIR
)
logger.info(f"Input args:\n{args}")
logger.info(f"Running with configs:\n{cfg}")

def test(model, adapter, loss_class, test_loader, logger, log_dir):
    if cfg.MODEL.ADAPTER:
        adapter_model = adapter[0]
        adapter_model.eval()

    model.eval()
    total_err_metrics = {
        "epe": 0,
        "bad1": 0,
        "bad2": 0,
        "depth_abs_err": 0,
        "depth_err2": 0,
        "depth_err4": 0,
        "depth_err8": 0,
    }
    total_obj_disp_err = np.zeros(cfg.SIM.OBJ_NUM)
    total_obj_depth_err = np.zeros(cfg.SIM.OBJ_NUM)
    total_obj_depth_4_err = np.zeros(cfg.SIM.OBJ_NUM)
    total_obj_count = np.zeros(cfg.SIM.OBJ_NUM)
    os.mkdir(os.path.join(log_dir, "pred_disp"))
    os.mkdir(os.path.join(log_dir, "gt_disp"))
    os.mkdir(os.path.join(log_dir, "pred_disp_abs_err_cmap"))
    os.mkdir(os.path.join(log_dir, "pred_depth"))
    os.mkdir(os.path.join(log_dir, "gt_depth"))
    os.mkdir(os.path.join(log_dir, "pred_depth_abs_err_cmap"))

    for iteration, data in enumerate(tqdm(test_loader)):

        if cfg.LOSSES.ONREAL:
            img_L = data["img_real_L"].cuda()  # [bs, 1, H, W]
            img_R = data["img_real_R"].cuda()
        else:
            img_L = data["img_sim_L"].cuda()  # [bs, 1, H, W]
            img_R = data["img_sim_R"].cuda()

        img_disp_l = data["img_disp_L"].cuda()
        img_depth_l = data["img_depth_L"].cuda()

        img_label = data["img_label"].cuda()
        img_focal_length = data["focal_length"].cuda()
        img_baseline = data["baseline"].cuda()
        prefix = data["prefix"][0]


        img_disp_l = F.interpolate(
            img_disp_l, (540, 960), mode="nearest", recompute_scale_factor=False
        )
        img_depth_l = F.interpolate(
            img_depth_l, (540, 960), mode="nearest", recompute_scale_factor=False
        )
        img_label = F.interpolate(
            img_label, (540, 960), mode="nearest", recompute_scale_factor=False
        ).type(torch.int)

        img_disp_r = data["img_disp_R"].cuda()
        img_depth_r = data["img_depth_R"].cuda()
        img_disp_r = F.interpolate(
            img_disp_r, (540, 960), mode="nearest", recompute_scale_factor=False
        )
        img_depth_r = F.interpolate(
            img_depth_r, (540, 960), mode="nearest", recompute_scale_factor=False
        )
        img_disp_l = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))
        img_depth_l = apply_disparity_cu(img_depth_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]

        # If test on real dataset need to crop input image to (540, 960)
        if cfg.LOSSES.ONREAL:
            robot_mask = data["robot_mask"].cuda()
            img_robot_mask = F.interpolate(
                robot_mask.unsqueeze(0), (540, 960), mode="nearest", recompute_scale_factor=False
            ).type(torch.int)
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

        if cfg.MODEL.ADAPTER:
            with torch.no_grad():
                img_L_transformed, img_R_transformed = adapter_model(img_L, img_R)

        # Pad the imput image and depth disp image to 960 * 544
        right_pad = cfg.REAL.PAD_WIDTH - 960
        top_pad = cfg.REAL.PAD_HEIGHT - 540

        img_L = F.pad(
            img_L, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode="constant", value=0
        )
        img_R = F.pad(
            img_R, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode="constant", value=0
        )

        if cfg.MODEL.ADAPTER:
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

        if cfg.LOSSES.ONREAL:
            robot_mask = img_robot_mask == 0
        else:
            robot_mask = torch.ones(img_depth_l.shape).cuda()==1

        if cfg.LOSSES.EXCLUDE_BG:
            # Mask ground pixel to False
            img_ground_mask = (img_depth_l > 0) & (img_depth_l < 1.25)
            mask = (
                (img_disp_l < cfg.MODEL.MAX_DISP)
                * (img_disp_l > 0)
                * img_ground_mask
                * robot_mask
            )
        else:
            mask = (img_disp_l < cfg.MODEL.MAX_DISP) * (img_disp_l > 0) * robot_mask

        # Exclude uncertain pixel from realsense_depth_pred
        if cfg.LOSSES.EXCLUDE_ZEROS:
            if cfg.LOSSES.ONREAL:
                img_depth_realsense = data["img_depth_real_realsense"].cuda()
            else:
                img_depth_realsense = data["img_depth_sim_realsense"].cuda()
            img_depth_realsense = F.interpolate(
                img_depth_realsense.unsqueeze(0),
                (540, 960),
                mode="nearest",
                recompute_scale_factor=False,
            )
            realsense_zeros_mask = img_depth_realsense > 0
            mask = mask * realsense_zeros_mask
        mask = mask.type(torch.bool)

        ground_mask = (
            torch.logical_not(mask).squeeze(0).squeeze(0).detach().cpu().numpy()
        )
        values = {
            'img_L': img_L,
            'img_R': img_R,
        }
        if cfg.MODEL.ADAPTER:
            values['img_L_transformed'] = img_L_transformed
            values['img_R_transformed'] = img_R_transformed

        output, pred_disp = loss_class.forward(values, train=False)

        pred_disp = pred_disp[:, :, top_pad:, :] if right_pad == 0 else pred_disp[:, :, top_pad:, :-right_pad]
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
            cfg.SIM.OBJ_NUM,
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
        total_err_metrics[k] /= len(test_loader)
    logger.info(f"\nTest on {len(test_loader)} instances\n {total_err_metrics}")

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

    real_obj_id = cfg.REAL.OBJ
    for i in range(cfg.SIM.OBJ_NUM):
        if i in real_obj_id:
            real_depth_error += total_obj_depth_err[i]
            real_depth_error_4mm += total_obj_depth_4_err[i]
        else:
            printed_depth_error += total_obj_depth_err[i]
            printed_depth_error_4mm += total_obj_depth_4_err[i]
    real_depth_error /= len(real_obj_id)
    real_depth_error_4mm /= len(real_obj_id)
    printed_depth_error /= cfg.SIM.OBJ_NUM - len(real_obj_id)
    printed_depth_error_4mm /= cfg.SIM.OBJ_NUM - len(real_obj_id)

    logger.info(
        f"Real objects - absolute depth error: {real_depth_error}, depth 4mm: {real_depth_error_4mm} \n"
        f"3D printed objects - absolute depth error {printed_depth_error}, depth 4mm: {printed_depth_error_4mm}"
    )

if __name__ == "__main__":
    test_dataset = MessytableDataset(split_sim=cfg.SIM.TEST, split_real = cfg.REAL.TEST,  onReal=True, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

    logger.info(f"Loaded the checkpoint: {cfg.MODEL.CHECKPOINT}")
    if cfg.MODEL.ADAPTER:
        from nets.adapter import Adapter
        adapter_model = Adapter().to(cuda_device)
        adapter_model_dict = load_from_dataparallel_model(cfg.MODEL.CHECKPOINT, "Adapter")
        adapter_model.load_state_dict(adapter_model_dict)

    backbone = cfg.MODEL.BACKBONE
    if backbone=="psmnet" and cfg.MODEL.ADAPTER:
        from nets.psmnet.psmnet import PSMNet
        model = PSMNet(maxdisp=cfg.MODEL.MAX_DISP).to(cuda_device)
    elif backbone=="psmnet":
        from nets.psmnet.psmnet_3 import PSMNet
        model = PSMNet(maxdisp=cfg.MODEL.MAX_DISP).to(cuda_device)
    elif backbone=="dispnet":
        from nets.dispnet.dispnet import DispNet
        model = DispNet().to(cuda_device)
    elif backbone=="raft":
        from nets.raft.raft_stereo import RAFTStereo
        model = RAFTStereo().to(cuda_device)
    else:
        print("Model not implemented!")


    model_dict = load_from_dataparallel_model(cfg.MODEL.CHECKPOINT, "Model")
    model.load_state_dict(model_dict)

    loss_class = AllLosses(model, cfg.MODEL.BACKBONE, cfg.MODEL.ADAPTER)

    if cfg.MODEL.ADAPTER:
        test(model, [adapter_model], loss_class, test_loader, logger, log_dir)
    else:
        test(model, [], loss_class, test_loader, logger, log_dir)
