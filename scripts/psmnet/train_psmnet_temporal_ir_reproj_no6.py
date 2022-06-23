"""
Author: Isabella Liu 11/10/21
Feature: Train PSMNet IR reprojection wihtout the 6-layer small transform network
"""
import argparse
import gc
import os

import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from datasets.messytable_temporal_ir import MessytableDataset
from layers.psmnet import PSMNet
from utils.cascade_metrics import compute_err_metric
from utils.config import cfg
from utils.reduce import (AverageMeterDict, make_nograd_func,
                          reduce_scalar_outputs, set_random_seed, synchronize,
                          tensor2float, tensor2numpy)
from utils.reprojection import get_reproj_error_patch
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
    "--summary-freq",
    type=int,
    default=500,
    help="Frequency of saving temporary results",
)
parser.add_argument(
    "--save-freq", type=int, default=1000, help="Frequency of saving checkpoint"
)
parser.add_argument(
    "--logdir", required=True, help="Directory to save logs and checkpoints"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="Random seed (default: 1)"
)
parser.add_argument(
    "--local_rank", type=int, default=0, help="Rank of device in distributed training"
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Whether run in debug mode (will load less data)",
)
parser.add_argument(
    "--warp-op",
    action="store_true",
    default=True,
    help="whether use warp_op function to get disparity",
)
parser.add_argument(
    "--loss-ratio-sim",
    type=float,
    default=1,
    help="Ratio between loss_psmnet_sim and loss_reprojection_sim",
)
parser.add_argument(
    "--loss-ratio-real", type=float, default=1, help="Ratio for loss_reprojection_real"
)
parser.add_argument(
    "--gaussian-blur",
    action="store_true",
    default=False,
    help="whether apply gaussian blur",
)
parser.add_argument(
    "--color-jitter",
    action="store_true",
    default=False,
    help="whether apply color jitter",
)
parser.add_argument(
    "--ps", type=int, default=11, help="Patch size of doing patch loss calculation"
)

args = parser.parse_args()
cfg.merge_from_file(args.config_file)

# Set random seed to make sure networks in different processes are same
set_random_seed(args.seed)

# Set up distributed training
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
args.is_distributed = is_distributed
if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
cuda_device = torch.device("cuda:{}".format(args.local_rank))

# Set up tensorboard and logger
os.makedirs(args.logdir, exist_ok=True)
os.makedirs(os.path.join(args.logdir, "models"), exist_ok=True)
summary_writer = tensorboardX.SummaryWriter(logdir=args.logdir)
logger = setup_logger(
    "Reprojection PSMNet", distributed_rank=args.local_rank, save_dir=args.logdir
)
logger.info(f"Loaded config file: '{args.config_file}'")
logger.info(f"Running with configs:\n{cfg}")
logger.info(f"Running with {num_gpus} GPUs")

# python -m torch.distributed.launch layeres/train_psmnet_temporal_ir_reproj.py --summary-freq 1 --save-freq 1 --logdir ../train_10_14_psmnet_ir_reprojection/debug --debug
# python -m torch.distributed.launch layers/train_psmnet_temporal_ir_reproj.py --config-file configs/remote_train_primitive_randscenes.yaml --summary-freq 10 --save-freq 100 --logdir ../train_10_21_psmnet_smooth_ir_reproj/debug --debug


def train(psmnet_model, psmnet_optimizer, TrainImgLoader, ValImgLoader):
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
                psmnet_optimizer,
                global_step,
                cfg.SOLVER.LR_CASCADE,
                cfg.SOLVER.LR_STEPS,
            )

            do_summary = global_step % args.summary_freq == 0
            # Train one sample
            scalar_outputs_psmnet, img_outputs_psmnet, img_output_reproj = train_sample(
                sample, psmnet_model, psmnet_optimizer, isTrain=True
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
                        nrow=4,
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
                if (global_step) % args.save_freq == 0:
                    checkpoint_data = {
                        "epoch": epoch_idx,
                        "PSMNet": psmnet_model.state_dict(),
                        "optimizerPSMNet": psmnet_optimizer.state_dict(),
                    }
                    save_filename = os.path.join(
                        args.logdir, "models", f"model_{global_step}.pth"
                    )
                    torch.save(checkpoint_data, save_filename)

                    # Get average results among all batches
                    total_err_metric_psmnet = avg_train_scalars_psmnet.mean()
                    logger.info(
                        f"Step {global_step} train psmnet: {total_err_metric_psmnet}"
                    )
        gc.collect()


def train_sample(sample, psmnet_model, psmnet_optimizer, isTrain=True):
    if isTrain:
        psmnet_model.train()
    else:
        psmnet_model.eval()

    # Load data
    img_L = sample["img_L"].to(cuda_device)  # [bs, 3, H, W]
    img_R = sample["img_R"].to(cuda_device)
    img_L_ir_pattern = sample["img_L_ir_pattern"].to(cuda_device)  # [bs, 1, H, W]
    img_R_ir_pattern = sample["img_R_ir_pattern"].to(cuda_device)

    # Train on PSMNet
    disp_gt_l = sample["img_disp_l"].to(cuda_device)
    depth_gt = sample["img_depth_l"].to(cuda_device)  # [bs, 1, H, W]
    img_focal_length = sample["focal_length"].to(cuda_device)
    img_baseline = sample["baseline"].to(cuda_device)

    # Resize the 2x resolution disp and depth back to H * W
    # Note this should go before apply_disparity_cu
    disp_gt_l = F.interpolate(
        disp_gt_l, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )  # [bs, 1, H, W]
    depth_gt = F.interpolate(
        depth_gt, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )  # [bs, 1, H, W]

    if args.warp_op:
        img_disp_r = sample["img_disp_r"].to(cuda_device)
        img_disp_r = F.interpolate(
            img_disp_r, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
        )
        disp_gt_l = apply_disparity_cu(
            img_disp_r, img_disp_r.type(torch.int)
        )  # [bs, 1, H, W]
        del img_disp_r

    # Get stereo loss on sim
    mask = (disp_gt_l < cfg.ARGS.MAX_DISP) * (
        disp_gt_l > 0
    )  # Note in training we do not exclude bg
    if isTrain:
        pred_disp1, pred_disp2, pred_disp3 = psmnet_model(img_L, img_R)
        sim_pred_disp = pred_disp3
        loss_psmnet = (
            0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt_l[mask], reduction="mean")
            + 0.7
            * F.smooth_l1_loss(pred_disp2[mask], disp_gt_l[mask], reduction="mean")
            + F.smooth_l1_loss(pred_disp3[mask], disp_gt_l[mask], reduction="mean")
        )
    else:
        with torch.no_grad():
            pred_disp = psmnet_model(img_L, img_R)
            loss_psmnet = F.smooth_l1_loss(
                pred_disp[mask], disp_gt_l[mask], reduction="mean"
            )

    # Get reprojection loss on sim_ir_pattern
    sim_ir_reproj_loss, sim_ir_warped, sim_ir_reproj_mask = get_reproj_error_patch(
        input_L=img_L_ir_pattern,
        input_R=img_R_ir_pattern,
        pred_disp_l=sim_pred_disp,
        mask=mask,
        ps=args.ps,
    )

    # Backward on sim_ir_pattern reprojection
    sim_loss = loss_psmnet * args.loss_ratio_sim + sim_ir_reproj_loss
    if isTrain:
        psmnet_optimizer.zero_grad()
        sim_loss.backward()
        psmnet_optimizer.step()

    # Get reprojection loss on real
    img_real_L = sample["img_real_L"].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_R = sample["img_real_R"].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_L_ir_pattern = sample["img_real_L_ir_pattern"].to(cuda_device)
    img_real_R_ir_pattern = sample["img_real_R_ir_pattern"].to(cuda_device)
    if isTrain:
        pred_disp1, pred_disp2, pred_disp3 = psmnet_model(img_real_L, img_real_R)
        real_pred_disp = pred_disp3
    else:
        with torch.no_grad():
            real_pred_disp = psmnet_model(img_real_L, img_real_R)
    real_ir_reproj_loss, real_ir_warped, real_ir_reproj_mask = get_reproj_error_patch(
        input_L=img_real_L_ir_pattern,
        input_R=img_real_R_ir_pattern,
        pred_disp_l=real_pred_disp,
        ps=args.ps,
    )

    # Backward on real
    real_loss = real_ir_reproj_loss * args.loss_ratio_real
    if isTrain:
        psmnet_optimizer.zero_grad()
        real_loss.backward()
        psmnet_optimizer.step()

    # Save reprojection outputs and images
    img_output_reproj = {
        "sim_reprojection": {
            "target": img_L_ir_pattern,
            "warped": sim_ir_warped,
            "pred_disp": sim_pred_disp,
            "mask": sim_ir_reproj_mask,
        },
        "real_reprojection": {
            "target": img_real_L_ir_pattern,
            "warped": real_ir_warped,
            "pred_disp": real_pred_disp,
            "mask": real_ir_reproj_mask,
        },
    }

    # Compute stereo error metrics on sim
    pred_disp = sim_pred_disp
    scalar_outputs_psmnet = {
        "loss": loss_psmnet.item(),
        "sim_reprojection_loss": sim_ir_reproj_loss.item(),
        "real_reprojection_loss": real_ir_reproj_loss.item(),
    }
    err_metrics = compute_err_metric(
        disp_gt_l, depth_gt, pred_disp, img_focal_length, img_baseline, mask
    )
    scalar_outputs_psmnet.update(err_metrics)
    # Compute error images
    pred_disp_err_np = disp_error_img(pred_disp[[0]], disp_gt_l[[0]], mask[[0]])
    pred_disp_err_tensor = torch.from_numpy(
        np.ascontiguousarray(pred_disp_err_np[None].transpose([0, 3, 1, 2]))
    )
    img_outputs_psmnet = {
        "disp_gt_l": disp_gt_l[[0]].repeat([1, 3, 1, 1]),
        "disp_pred": pred_disp[[0]].repeat([1, 3, 1, 1]),
        "disp_err": pred_disp_err_tensor,
        "input_L": img_L,
        "input_R": img_R,
    }

    if is_distributed:
        scalar_outputs_psmnet = reduce_scalar_outputs(
            scalar_outputs_psmnet, cuda_device
        )
    return scalar_outputs_psmnet, img_outputs_psmnet, img_output_reproj


if __name__ == "__main__":
    # Obtain dataloader
    train_dataset = MessytableDataset(
        cfg.SPLIT.TRAIN,
        gaussian_blur=args.gaussian_blur,
        color_jitter=args.color_jitter,
        debug=args.debug,
        sub=100,
    )
    val_dataset = MessytableDataset(
        cfg.SPLIT.VAL,
        gaussian_blur=args.gaussian_blur,
        color_jitter=args.color_jitter,
        debug=args.debug,
        sub=10,
    )
    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )

        TrainImgLoader = torch.utils.data.DataLoader(
            train_dataset,
            cfg.SOLVER.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=cfg.SOLVER.NUM_WORKER,
            drop_last=True,
            pin_memory=True,
        )
        ValImgLoader = torch.utils.data.DataLoader(
            val_dataset,
            cfg.SOLVER.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=cfg.SOLVER.NUM_WORKER,
            drop_last=False,
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

        ValImgLoader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.SOLVER.NUM_WORKER,
            drop_last=False,
        )

    # Create PSMNet model
    psmnet_model = PSMNet(maxdisp=cfg.ARGS.MAX_DISP).to(cuda_device)
    psmnet_optimizer = torch.optim.Adam(
        psmnet_model.parameters(), lr=cfg.SOLVER.LR_CASCADE, betas=(0.9, 0.999)
    )
    if is_distributed:
        psmnet_model = torch.nn.parallel.DistributedDataParallel(
            psmnet_model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    else:
        psmnet_model = torch.nn.DataParallel(psmnet_model)

    # Start training
    train(psmnet_model, psmnet_optimizer, TrainImgLoader, ValImgLoader)
