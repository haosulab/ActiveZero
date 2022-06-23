"""
Author: Isabella Liu 10/13/21
Feature: Train PSMNet IR reprojection
"""
import argparse
import gc
import os
import sys
sys.path.insert(1, os.getcwd())

import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from datasets.messytable import MessytableDataset
from baselines.cycleGAN.cycle_gan import CycleGANModel
from baselines.cycleGAN.psmnet_cycle import PSMNet
from nets.adapter import Adapter
from utils.cascade_metrics import compute_err_metric
from baselines.cycleGAN.config import cfg
from utils.reduce import (AverageMeterDict, make_nograd_func,
                          reduce_scalar_outputs, set_random_seed, synchronize,
                          tensor2float, tensor2numpy)
from utils.reprojection import get_reprojection_error
from utils.util import (adjust_learning_rate, disp_error_img, save_images,
                        save_images_grid, save_scalars, save_scalars_graph,
                        setup_logger, weights_init)
from utils.warp_ops import apply_disparity_cu

cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Reprojection with Pyramid Stereo Network (PSMNet)"
)
parser.add_argument(
    "--config-file",
    type=str,
    default="./configs/local_train_steps.yaml",
    metavar="FILE",
    help="Config files",
)
parser.add_argument(
    "--local_rank", type=int, default=0, help="Rank of device in distributed training"
)

args = parser.parse_args()
cfg.merge_from_file(args.config_file)

# Set random seed to make sure networks in different processes are same
set_random_seed(cfg.SOLVER.SEED)

# Set up distributed training
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
cuda_device = torch.device("cuda:{}".format(args.local_rank))

# Set up tensorboard and logger
os.makedirs(cfg.SOLVER.LOGDIR, exist_ok=True)
os.makedirs(os.path.join(cfg.SOLVER.LOGDIR, "models"), exist_ok=True)
summary_writer = tensorboardX.SummaryWriter(logdir=cfg.SOLVER.LOGDIR)
logger = setup_logger(
    cfg.NAME, distributed_rank=args.local_rank, save_dir=cfg.SOLVER.LOGDIR
)
logger.info(f"Input args:\n{args}")
logger.info(f"Running with configs:\n{cfg}")
logger.info(f"Running with {num_gpus} GPUs")

# python -m torch.distributed.launch train_psmnet_cycle_reprojection.py --summary-freq 1 --save-freq 1 --logdir ../train_10_14_psmnet_reprojection/debug --debug
# python -m torch.distributed.launch train_psmnet_cycle_reprojection.py --config-file configs/remote_train_steps.yaml --summary-freq 10 --save-freq 100 --logdir ../train_10_14_psmnet_reprojection/debug --debug


def train(
    adapter_model,
    psmnet_model,
    adapter_optimizer,
    psmnet_optimizer,
    TrainImgLoader,
):
    cur_err = np.inf  # store best result

    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # One epoch training loop
        avg_train_scalars_psmnet = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = (
                (len(TrainImgLoader) * epoch_idx + batch_idx)
                * cfg.SOLVER.BATCH_SIZE
                * num_gpus
            )
            if global_step > cfg.SOLVER.STEPS:
                break

            # Adjust learning rate
            adjust_learning_rate(
                adapter_optimizer,
                global_step,
                cfg.SOLVER.LR,
                cfg.SOLVER.LR_STEPS,
            )
            adjust_learning_rate(
                psmnet_optimizer,
                global_step,
                cfg.SOLVER.LR,
                cfg.SOLVER.LR_STEPS,
            )

            do_summary = global_step % cfg.SOLVER.SUMMARY_FREQ == 0
            # Train one sample
            (
                scalar_outputs_reproj,
                scalar_outputs_psmnet,
                img_outputs_psmnet,
                img_output_reproj,
            ) = train_sample(
                sample,
                adapter_model,
                psmnet_model,
                adapter_optimizer,
                psmnet_optimizer,
                isTrain=True,
            )
            # Save result to tensorboard
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs_psmnet = tensor2float(scalar_outputs_psmnet)
                avg_train_scalars_psmnet.update(scalar_outputs_psmnet)
                if do_summary:
                    # Update reprojection images
                    save_images_grid(
                        summary_writer,
                        "train_reproj",
                        img_output_reproj,
                        global_step,
                        nrow=3,
                    )
                    save_scalars(
                        summary_writer,
                        "train_reproj",
                        scalar_outputs_reproj,
                        global_step,
                    )
                    # Update PSMNet images
                    save_images(
                        summary_writer, "train_psmnet", img_outputs_psmnet, global_step
                    )
                    # Update PSMNet losses
                    scalar_outputs_psmnet.update(
                        {"lr": psmnet_optimizer.param_groups[0]["lr"]}
                    )
                    save_scalars(
                        summary_writer,
                        "train_psmnet",
                        scalar_outputs_psmnet,
                        global_step,
                    )

                # Save checkpoints
                if (global_step) % cfg.SOLVER.SAVE_FREQ == 0:
                    checkpoint_data = {
                        "epoch": epoch_idx,
                        "Transformer": adapter_model.state_dict(),
                        "PSMNet": psmnet_model.state_dict(),
                        "optimizerTransformer": adapter_optimizer.state_dict(),
                        "optimizerPSMNet": psmnet_optimizer.state_dict(),
                    }
                    save_filename = os.path.join(
                        cfg.SOLVER.LOGDIR, "models", f"model_{global_step}.pth"
                    )
                    torch.save(checkpoint_data, save_filename)

                    # Get average results among all batches
                    total_err_metric_psmnet = avg_train_scalars_psmnet.mean()
                    logger.info(
                        f"Step {global_step} train psmnet: {total_err_metric_psmnet}"
                    )
        gc.collect()

def train_sample(
    sample,
    adapter_model,
    psmnet_model,
    adapter_optimizer,
    psmnet_optimizer,
    isTrain=True,
):
    if isTrain:
        adapter_model.train()
        psmnet_model.train()
    else:
        adapter_model.eval()
        psmnet_model.eval()

    # Load data
    img_sim_L = sample["img_sim_L"].to(cuda_device)  # [bs, 3, H, W]
    img_sim_R = sample["img_sim_R"].to(cuda_device)
    img_real_L = sample["img_real_L"].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_R = sample["img_real_R"].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_L = F.interpolate(
        img_real_L,
        scale_factor=0.5,
        mode="bilinear",
        recompute_scale_factor=False,
        align_corners=False,
    )
    img_real_R = F.interpolate(
        img_real_R,
        scale_factor=0.5,
        mode="bilinear",
        recompute_scale_factor=False,
        align_corners=False,
    )

    # Train on PSMNet
    disp_gt_l = sample["img_disp_L"].to(cuda_device)
    depth_gt_l = sample["img_depth_L"].to(cuda_device)  # [bs, 1, H, W]
    disp_gt_r = sample["img_disp_R"].to(cuda_device)
    depth_gt_r = sample["img_depth_R"].to(cuda_device)  # [bs, 1, H, W]
    img_focal_length = sample["focal_length"].to(cuda_device)
    img_baseline = sample["baseline"].to(cuda_device)

    # Resize the 2x resolution disp and depth back to H * W
    # Note this should go before apply_disparity_cu
    disp_gt_l = F.interpolate(
        disp_gt_l, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )  # [bs, 1, H, W]
    disp_gt_r = F.interpolate(
        disp_gt_r, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )  # [bs, 1, H, W]
    depth_gt_l = F.interpolate(
        depth_gt_l, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )  # [bs, 1, H, W]
    depth_gt_r = F.interpolate(
        depth_gt_r, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )  # [bs, 1, H, W]

    disp_gt_l = apply_disparity_cu(
        disp_gt_r, disp_gt_r.type(torch.int)
    )  # [bs, 1, H, W]
    disp_gt_r = apply_disparity_cu(
        disp_gt_l, -disp_gt_l.type(torch.int)
    )  # [bs, 1, H, W]

    # Get masks
    mask_l = (disp_gt_l < 192) * (
        disp_gt_l > 0
    )  # Note in training we do not exclude bg
    mask_l = mask_l.detach()
    mask_r = (disp_gt_r < 192) * (
        disp_gt_r > 0
    )  # Note in training we do not exclude bg
    mask_r = mask_r.detach()

    # Backward on sim
    (
        img_sim_L_transformed,
        img_sim_R_transformed,
        img_real_L_transformed,
        img_real_R_transformed,
    ) = adapter_model(
        img_sim_L, img_sim_R, img_real_L, img_real_R
    )  # [bs, 3, H, W]
    if isTrain:
        (
            pred_disp1_l,
            pred_disp2_l,
            pred_disp3_l,
            pred_disp1_r,
            pred_disp2_r,
            pred_disp3_r,
        ) = psmnet_model(img_sim_L, img_sim_R, img_sim_L_transformed, img_sim_R_transformed)
        sim_pred_disp_l = pred_disp3_l
        loss_psmnet_l = (
            0.5
            * F.smooth_l1_loss(
                pred_disp1_l[mask_l], disp_gt_l[mask_l], reduction="mean"
            )
            + 0.7
            * F.smooth_l1_loss(
                pred_disp2_l[mask_l], disp_gt_l[mask_l], reduction="mean"
            )
            + F.smooth_l1_loss(
                pred_disp3_l[mask_l], disp_gt_l[mask_l], reduction="mean"
            )
        )
        sim_pred_disp_r = pred_disp3_r
        loss_psmnet_r = (
            0.5
            * F.smooth_l1_loss(
                pred_disp1_r[mask_r], disp_gt_r[mask_r], reduction="mean"
            )
            + 0.7
            * F.smooth_l1_loss(
                pred_disp2_r[mask_r], disp_gt_r[mask_r], reduction="mean"
            )
            + F.smooth_l1_loss(
                pred_disp3_r[mask_r], disp_gt_r[mask_r], reduction="mean"
            )
        )
    else:
        with torch.no_grad():
            sim_pred_disp_l, sim_pred_disp_r = psmnet_model(
                img_sim_L, img_sim_R, img_sim_L_transformed, img_sim_R_transformed
            )

    (
        sim_img_reproj_loss_l,
        sim_img_reproj_loss_r,
        sim_img_warped_l,
        sim_img_warped_r,
        sim_img_reproj_mask_l,
        sim_img_reproj_mask_r,
    ) = get_reprojection_error(
        img_sim_L, img_sim_R, sim_pred_disp_l, sim_pred_disp_r, mask_l, mask_r
    )
    sim_loss = (
        loss_psmnet_l + loss_psmnet_r + sim_img_reproj_loss_l + sim_img_reproj_loss_r
    )
    adapter_optimizer.zero_grad()
    psmnet_optimizer.zero_grad()
    sim_loss.backward()
    psmnet_optimizer.step()
    adapter_optimizer.step()

    # Backward on real
    # Get prediction from left camera frame
    (
        img_sim_L_transformed,
        img_sim_R_transformed,
        img_real_L_transformed,
        img_real_R_transformed,
    ) = adapter_model(
        img_sim_L, img_sim_R, img_real_L, img_real_R
    )  # [bs, 3, H, W]
    if isTrain:
        (
            pred_disp1_l,
            pred_disp2_l,
            pred_disp3_l,
            pred_disp1_r,
            pred_disp2_r,
            pred_disp3_r,
        ) = psmnet_model(
            img_real_L, img_real_R, img_real_L_transformed, img_real_R_transformed
        )
        real_pred_disp_l = pred_disp3_l
        real_pred_disp_r = pred_disp3_r
    else:
        with torch.no_grad():
            real_pred_disp_l, real_pred_disp_r = psmnet_model(
                img_real_L, img_real_R, img_real_L_transformed, img_real_R_transformed
            )
    (
        real_img_reproj_loss_l,
        real_img_reproj_loss_r,
        real_img_warped_l,
        real_img_warped_r,
        real_img_reproj_mask_l,
        real_img_reproj_mask_r,
    ) = get_reprojection_error(
        img_real_L, img_real_R, real_pred_disp_l, real_pred_disp_r
    )
    real_loss = real_img_reproj_loss_l + real_img_reproj_loss_r
    adapter_optimizer.zero_grad()
    psmnet_optimizer.zero_grad()
    real_loss.backward()
    psmnet_optimizer.step()
    adapter_optimizer.step()

    # Save reprojection outputs and images
    img_output_reproj = {
        "sim_reprojection_l": {
            "img": img_sim_L,
            "img_warped": sim_img_warped_l,
            "mask": sim_img_reproj_mask_l,
        },
        "sim_reprojection_r": {
            "img": img_sim_R,
            "img_warped": sim_img_warped_r,
            "mask": sim_img_reproj_mask_r,
        },
        "real_reprojection_l": {
            "img": img_real_L,
            "img_warped": real_img_warped_l,
            "mask": real_img_reproj_mask_l,
        },
        "real_reprojection_r": {
            "img": img_real_R,
            "img_warped": real_img_warped_r,
            "mask": real_img_reproj_mask_r,
        },
    }

    scalar_outputs_reproj = {
        "sim_stereo_loss_l": loss_psmnet_l.item(),
        "sim_stereo_loss_r": loss_psmnet_r.item(),
        "sim_reproj_loss_l": sim_img_reproj_loss_l.item(),
        "sim_reproj_loss_r": sim_img_reproj_loss_r.item(),
        "real_reproj_loss_l": real_img_reproj_loss_l.item(),
        "real_reproj_loss_r": real_img_reproj_loss_r.item(),
    }

    # Compute stereo error metrics on sim
    pred_disp = sim_pred_disp_l
    loss_psmnet = loss_psmnet_l
    disp_gt = disp_gt_l
    depth_gt = depth_gt_l
    mask = mask_l
    scalar_outputs_psmnet = {"loss": loss_psmnet.item()}
    scalar_outputs_psmnet.update(scalar_outputs_reproj)
    err_metrics = compute_err_metric(
        disp_gt, depth_gt, pred_disp, img_focal_length, img_baseline, mask
    )
    scalar_outputs_psmnet.update(err_metrics)
    # Compute error images
    pred_disp_err_np = disp_error_img(pred_disp[[0]], disp_gt[[0]], mask[[0]])
    pred_disp_err_tensor = torch.from_numpy(
        np.ascontiguousarray(pred_disp_err_np[None].transpose([0, 3, 1, 2]))
    )
    img_outputs_psmnet = {
        "disp_gt": disp_gt[[0]].repeat([1, 3, 1, 1]),
        "disp_pred": pred_disp[[0]].repeat([1, 3, 1, 1]),
        "disp_err": pred_disp_err_tensor,
    }

    if is_distributed:
        scalar_outputs_psmnet = reduce_scalar_outputs(
            scalar_outputs_psmnet, cuda_device
        )
    return (
        scalar_outputs_reproj,
        scalar_outputs_psmnet,
        img_outputs_psmnet,
        img_output_reproj,
    )


if __name__ == "__main__":
    # Obtain dataloader
    train_dataset = MessytableDataset(cfg.SIM.TRAIN, onReal=cfg.LOSSES.ONREAL)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )

        TrainImgLoader = torch.utils.data.DataLoader(
            train_dataset,
            cfg.SOLVER.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=cfg.SOLVER.NUM_WORKER,
            drop_last=True,
            pin_memory=True,
        )

    else:
        TrainImgLoader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.SOLVER.NUM_WORKER,
            drop_last=True,
        )


    # Create Transformer model
    adapter_model = Adapter().to(cuda_device)
    adapter_optimizer = torch.optim.Adam(
        adapter_model.parameters(), lr=cfg.SOLVER.LR, betas=cfg.SOLVER.BETAS
    )
    if is_distributed:
        adapter_model = torch.nn.parallel.DistributedDataParallel(
            adapter_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
    else:
        adapter_model = torch.nn.DataParallel(adapter_model)

    # Create PSMNet model
    psmnet_model = PSMNet(maxdisp=cfg.MODEL.MAX_DISP).to(cuda_device)
    psmnet_optimizer = torch.optim.Adam(
        psmnet_model.parameters(), lr=cfg.SOLVER.LR, betas=(0.9, 0.999)
    )
    if is_distributed:
        psmnet_model = torch.nn.parallel.DistributedDataParallel(
            psmnet_model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    else:
        psmnet_model = torch.nn.DataParallel(psmnet_model)

    # Start training
    train(
        adapter_model,
        psmnet_model,
        adapter_optimizer,
        psmnet_optimizer,
        TrainImgLoader,
    )
