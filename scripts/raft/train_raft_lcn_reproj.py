"""
Author: Isabella Liu 1/27/22
Feature: Train ActiveZeroNet with RAFT as backbone, and use LCN patter reprojection
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

from RAFT.core.raft_stereo import RAFTStereo, sequence_loss
from RAFT.messytable_temporal_ir import MessytableDataset
from utils.cascade_metrics import compute_err_metric
from utils.config import cfg
from utils.LCN import local_contrast_norm
from utils.reduce import (AverageMeterDict, make_nograd_func,
                          reduce_scalar_outputs, set_random_seed, synchronize,
                          tensor2float, tensor2numpy)
from utils.reprojection import get_reproj_error_patch
from utils.util import (disp_error_img, save_images, save_images_grid,
                        save_scalars, save_scalars_graph, setup_logger,
                        weights_init)
from utils.warp_ops import apply_disparity_cu

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


cudnn.benchmark = True

parser = argparse.ArgumentParser(description="LCN reprojection with RAFT")
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
    "--sub",
    type=int,
    default=100,
    help="If debug mode is enabled, sub will be the number of data loaded",
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
    default=0.01,
    help="Ratio between loss_psmnet_sim and loss_reprojection_sim",
)
parser.add_argument(
    "--loss-ratio-real", type=float, default=2, help="Ratio for loss_reprojection_real"
)
parser.add_argument(
    "--ps", type=int, default=11, help="Patch size of doing patch loss calculation"
)

# Data augmentation
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

# RAFT parameters
parser.add_argument(
    "--corr_implementation",
    choices=["reg", "alt", "reg_cuda", "alt_cuda"],
    default="reg",
    help="correlation volume implementation",
)
parser.add_argument(
    "--shared_backbone",
    action="store_true",
    help="use a single backbone for the context and feature encoders",
)
parser.add_argument(
    "--corr_levels",
    type=int,
    default=4,
    help="number of levels in the correlation pyramid",
)
parser.add_argument(
    "--corr_radius", type=int, default=4, help="width of the correlation pyramid"
)
parser.add_argument(
    "--n_downsample",
    type=int,
    default=2,
    help="resolution of the disparity field (1/2^K)",
)
parser.add_argument(
    "--slow_fast_gru",
    action="store_true",
    help="iterate the low-res GRUs more frequently",
)
parser.add_argument(
    "--n_gru_layers", type=int, default=3, help="number of hidden GRU levels"
)
parser.add_argument(
    "--hidden_dims",
    nargs="+",
    type=int,
    default=[128] * 3,
    help="hidden state and context dimensions",
)

# Training parameters
parser.add_argument(
    "--mixed_precision", action="store_true", help="use mixed precision"
)
parser.add_argument(
    "--batch_size", type=int, default=6, help="batch size used during training."
)
parser.add_argument(
    "--train_datasets", nargs="+", default=["sceneflow"], help="training datasets."
)
parser.add_argument("--lr", type=float, default=0.0002, help="max learning rate.")
parser.add_argument(
    "--num_steps", type=int, default=40000, help="length of training schedule."
)
parser.add_argument(
    "--image_size",
    type=int,
    nargs="+",
    default=[320, 720],
    help="size of the random image crops used during training.",
)
parser.add_argument(
    "--train_iters",
    type=int,
    default=22,
    help="number of updates to the disparity field in each forward pass.",
)
parser.add_argument(
    "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
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
logger.info(f"Input args:\n{args}")
logger.info(f"Loaded config file: '{args.config_file}'")
logger.info(f"Running with configs:\n{cfg}")
logger.info(f"Running with {num_gpus} GPUs")

# python -m torch.distributed.launch RAFT/train_raft_lcn_reproj.py --summary-freq 1 --save-freq 1 --logdir ../train_10_14_psmnet_ir_reprojection/debug --debug
# python -m torch.distributed.launch RAFT/train_raft_lcn_reproj.py --config-file ./RAFT/raft_config.yaml --summary-freq 10 --save-freq 100 --logdir /isabella-slow/RAFT/train_1_26_ir_reproj/debug --debug --mixed_precision

# Default training parameter from RAFT repository
# python train_stereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 200000 --mixed_precision

l1_loss = torch.nn.L1Loss()


def train(
    raft_model,
    raft_optimizer,
    raft_scheduler,
    raft_scaler,
    TrainImgLoader,
    ValImgLoader,
):
    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # One epoch training loop
        avg_train_scalars_raft = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = (
                (len(TrainImgLoader) * epoch_idx + batch_idx)
                * cfg.SOLVER.BATCH_SIZE
                * num_gpus
            )
            if global_step > cfg.SOLVER.STEPS:
                break

            do_summary = global_step % args.summary_freq == 0
            # Train one sample
            scalar_outputs_raft, img_outputs_raft, img_output_reproj = train_sample(
                sample,
                raft_model,
                raft_optimizer,
                raft_scheduler,
                raft_scaler,
                isTrain=True,
            )

            # Save result to tensorboard
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs_raft = tensor2float(scalar_outputs_raft)
                avg_train_scalars_raft.update(scalar_outputs_raft)
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
                        summary_writer, "train_psmnet", img_outputs_raft, global_step
                    )
                    # Update PSMNet losses
                    scalar_outputs_raft.update(
                        {"lr": raft_optimizer.param_groups[0]["lr"]}
                    )
                    save_scalars(
                        summary_writer, "train_raft", scalar_outputs_raft, global_step
                    )

                # Save checkpoints
                if (global_step) % args.save_freq == 0:
                    checkpoint_data = {
                        "epoch": epoch_idx,
                        "model": raft_model.state_dict(),
                        "optimizer": raft_optimizer.state_dict(),
                    }
                    save_filename = os.path.join(
                        args.logdir, "models", f"model_{global_step}.pth"
                    )
                    torch.save(checkpoint_data, save_filename)

                    # Get average results among all batches
                    total_err_metric_psmnet = avg_train_scalars_raft.mean()
                    logger.info(
                        f"Step {global_step} train psmnet: {total_err_metric_psmnet}"
                    )
        gc.collect()


def train_sample(
    sample, raft_model, raft_optimizer, raft_scheduler, raft_scaler, isTrain=True
):
    if isTrain:
        raft_model.train()
    else:
        raft_model.eval()

    # Load data
    img_L = sample["img_L"].to(cuda_device)  # [bs, 3, H, W]
    img_R = sample["img_R"].to(cuda_device)
    disp_gt_l = sample["img_disp_l"].to(cuda_device)
    depth_gt = sample["img_depth_l"].to(cuda_device)  # [bs, 1, 2H, 2W]
    img_focal_length = sample["focal_length"].to(cuda_device)
    img_baseline = sample["baseline"].to(cuda_device)

    # Resize the 2x resolution disp, depth and flow back to H * W
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

    # Convert gt disp to flow
    img_flow_gt = -disp_gt_l  # [bs, 2, H, W]

    # Get RAFT loss on sim
    mask = (disp_gt_l < cfg.ARGS.MAX_DISP) * (
        disp_gt_l > 0
    )  # Note in training we do not exclude bg [bs, 1, H, W]

    if isTrain:
        flow_pred = raft_model(
            img_L, img_R, iters=args.train_iters
        )  # Note here flow_pred is a list
        loss_raft = sequence_loss(flow_pred, img_flow_gt, mask)
    else:
        with torch.no_grad():
            flow_pred = raft_model(img_L, img_R, iters=args.train_iters)
            loss_raft = sequence_loss(flow_pred, img_flow_gt, mask)

    # Convert pred disp to flow
    sim_pred_disp = -flow_pred[-1]

    # Get reprojection loss on sim_lcn_pattern
    img_L_lcn, _ = local_contrast_norm(img_L, kernel_size=args.ps)
    img_R_lcn, _ = local_contrast_norm(img_R, kernel_size=args.ps)
    sim_lcn_reproj_loss, sim_lcn_warped, sim_lcn_reproj_mask = get_reproj_error_patch(
        input_L=img_L_lcn,
        input_R=img_R_lcn,
        pred_disp_l=sim_pred_disp,
        mask=mask,
        ps=args.ps,
    )

    # Backward on sim_ir_pattern reprojection
    sim_loss = loss_raft * args.loss_ratio_sim + sim_lcn_reproj_loss

    if isTrain:
        raft_optimizer.zero_grad()
        raft_scaler.scale(sim_loss).backward()
        raft_scaler.unscale_(raft_optimizer)
        torch.nn.utils.clip_grad_norm_(raft_model.parameters(), 1.0)

        raft_scaler.step(raft_optimizer)
        raft_scheduler.step()
        raft_scaler.update()

    # Get reprojection loss on real
    img_real_L = sample["img_real_L"].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_R = sample["img_real_R"].to(cuda_device)  # [bs, 3, 2H, 2W]

    if isTrain:
        flow_pred = raft_model(img_real_L, img_real_R, iters=args.train_iters)
    else:
        with torch.no_grad():
            flow_pred = raft_model(img_real_L, img_real_R, iters=args.train_iters)

    # Convert pred disp to flow
    real_pred_disp = -flow_pred[-1]

    img_real_L_lcn, _ = local_contrast_norm(img_real_L, kernel_size=args.ps)
    img_real_R_lcn, _ = local_contrast_norm(img_real_R, kernel_size=args.ps)
    (
        real_lcn_reproj_loss,
        real_lcn_warped,
        real_lcn_reproj_mask,
    ) = get_reproj_error_patch(
        input_L=img_real_L_lcn,
        input_R=img_real_R_lcn,
        pred_disp_l=real_pred_disp,
        ps=args.ps,
    )

    # Backward on real
    real_loss = real_lcn_reproj_loss * args.loss_ratio_real
    if isTrain:
        raft_optimizer.zero_grad()
        raft_scaler.scale(real_loss).backward()
        raft_scaler.unscale_(raft_optimizer)
        torch.nn.utils.clip_grad_norm_(raft_model.parameters(), 1.0)

        raft_scaler.step(raft_optimizer)
        raft_scheduler.step()
        raft_scaler.update()

    # Save reprojection outputs and images
    img_output_reproj = {
        "sim_reprojection": {
            "target": img_L_lcn,
            "warped": sim_lcn_warped,
            "pred_disp": sim_pred_disp,
            "mask": sim_lcn_reproj_mask,
        },
        "real_reprojection": {
            "target": img_real_L_lcn,
            "warped": real_lcn_warped,
            "pred_disp": real_pred_disp,
            "mask": real_lcn_reproj_mask,
        },
    }

    # Compute stereo error metrics on sim
    pred_disp = sim_pred_disp
    scalar_outputs_raft = {
        "loss": loss_raft.item(),
        "sim_reprojection_loss": sim_lcn_reproj_loss.item(),
        "real_reprojection_loss": real_lcn_reproj_loss.item(),
    }
    err_metrics = compute_err_metric(
        disp_gt_l, depth_gt, pred_disp, img_focal_length, img_baseline, mask
    )
    scalar_outputs_raft.update(err_metrics)
    # Compute error images
    pred_disp_err_np = disp_error_img(pred_disp[[0]], disp_gt_l[[0]], mask[[0]])
    pred_disp_err_tensor = torch.from_numpy(
        np.ascontiguousarray(pred_disp_err_np[None].transpose([0, 3, 1, 2]))
    )
    img_outputs_raft = {
        "disp_gt_l": disp_gt_l[[0]].repeat([1, 3, 1, 1]),
        "disp_pred": pred_disp[[0]].repeat([1, 3, 1, 1]),
        "disp_err": pred_disp_err_tensor,
        "input_L": img_L,
        "input_R": img_R,
    }

    if is_distributed:
        scalar_outputs_raft = reduce_scalar_outputs(scalar_outputs_raft, cuda_device)
    return scalar_outputs_raft, img_outputs_raft, img_output_reproj


if __name__ == "__main__":
    # Obtain dataloader
    train_dataset = MessytableDataset(
        cfg.SPLIT.TRAIN,
        gaussian_blur=args.gaussian_blur,
        color_jitter=args.color_jitter,
        debug=args.debug,
        sub=args.sub,
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

    # Create RAFT model
    raft_model = RAFTStereo(args).to(cuda_device)

    # Create optimizer and lr scheduler
    raft_optimizer = torch.optim.AdamW(
        raft_model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    raft_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        raft_optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    raft_scaler = GradScaler(enabled=args.mixed_precision)
    if is_distributed:
        raft_model = torch.nn.parallel.DistributedDataParallel(
            raft_model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    else:
        raft_model = torch.nn.DataParallel(raft_model)

    # Start training
    train(
        raft_model,
        raft_optimizer,
        raft_scheduler,
        raft_scaler,
        TrainImgLoader,
        ValImgLoader,
    )
