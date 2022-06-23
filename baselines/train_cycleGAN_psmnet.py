"""
Author: Isabella Liu 11/11/21
Feature: Train cycle GAN with PSMNet
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

from nets.psmnet.psmnet_3 import PSMNet
from datasets.messytable import MessytableDataset
from baselines.cycleGAN.cycle_gan import CycleGANModel
from utils.cascade_metrics import compute_err_metric
from baselines.cycleGAN.config import cfg
from utils.reduce import (AverageMeterDict, reduce_scalar_outputs,
                          set_random_seed, synchronize,
                          tensor2float, tensor2numpy)
from utils.util import (adjust_learning_rate, disp_error_img, save_images,
                        save_images_grid, save_scalars, save_scalars_graph,
                        setup_logger, weights_init)
from utils.warp_ops import apply_disparity_cu

cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="CycleGAN with Pyramid Stereo Network (PSMNet)"
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

# python -m torch.distributed.launch baselines/train_cycleGAN_psmnet.py --config-file ./baselines/remote_train_GAN.yaml

def train(gan_model, psmnet_model, psmnet_optimizer, TrainImgLoader):
    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # One epoch training loop
        avg_train_scalars_gan = AverageMeterDict()
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
                gan_model.optimizer_G,
                global_step,
                cfg.SOLVER.LR_G,
                cfg.SOLVER.LR_GAN_STEPS,
            )
            adjust_learning_rate(
                gan_model.optimizer_D,
                global_step,
                cfg.SOLVER.LR_D,
                cfg.SOLVER.LR_GAN_STEPS,
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
                scalar_outputs_gan,
                scalar_outputs_psmnet,
                img_outputs_gan,
                img_outputs_psmnet,
            ) = train_sample(
                sample, gan_model, psmnet_model, psmnet_optimizer, isTrain=True
            )

            # Save result to tensorboard
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs_gan = tensor2float(scalar_outputs_gan)
                scalar_outputs_psmnet = tensor2float(scalar_outputs_psmnet)
                avg_train_scalars_gan.update(scalar_outputs_gan)
                avg_train_scalars_psmnet.update(scalar_outputs_psmnet)
                if do_summary:
                    # Update GAN images
                    save_images_grid(
                        summary_writer,
                        "train_gan",
                        img_outputs_gan,
                        global_step,
                        nrow=4,
                    )
                    # Update GAN losses
                    scalar_outputs_gan.update(
                        {"lr_G": gan_model.optimizer_G.param_groups[0]["lr"]}
                    )
                    scalar_outputs_gan.update(
                        {"lr_D": gan_model.optimizer_D.param_groups[0]["lr"]}
                    )
                    save_scalars_graph(
                        summary_writer, "train_gan", scalar_outputs_gan, global_step
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
                        "G_A": gan_model.netG_A.state_dict(),
                        "G_B": gan_model.netG_B.state_dict(),
                        "D_A": gan_model.netD_A.state_dict(),
                        "D_B": gan_model.netD_B.state_dict(),
                        "PSMNet": psmnet_model.state_dict(),
                        "optimizerG": gan_model.optimizer_G.state_dict(),
                        "optimizerD": gan_model.optimizer_D.state_dict(),
                        "optimizerPSMNet": psmnet_optimizer.state_dict(),
                    }
                    save_filename = os.path.join(
                        cfg.SOLVER.LOGDIR, "models", f"model_{global_step}.pth"
                    )
                    torch.save(checkpoint_data, save_filename)

                    # Get average results among all batches
                    total_err_metric_gan = avg_train_scalars_gan.mean()
                    total_err_metric_psmnet = avg_train_scalars_psmnet.mean()
                    logger.info(
                        f"Step {global_step} train gan    : {total_err_metric_gan}"
                    )
                    logger.info(
                        f"Step {global_step} train psmnet: {total_err_metric_psmnet}"
                    )
        gc.collect()


def train_sample(sample, gan_model, psmnet_model, psmnet_optimizer, isTrain=True):
    if isTrain:
        gan_model.train()
        psmnet_model.train()
    else:
        gan_model.eval()
        psmnet_model.eval()

    # Load data
    img_sim_L = sample["img_sim_L"].to(cuda_device)  # [bs, 3, H, W]
    img_sim_R = sample["img_sim_R"].to(cuda_device)  # [bs, 3, H, W]
    img_real_L = sample["img_real_L"].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_R = sample["img_real_R"].to(cuda_device)  # [bs, 3, 2H, 2W]

    # Train on GAN
    input_sample = {
        "img_sim_L": img_sim_L,
        "img_sim_R": img_sim_R,
        "img_real_L": img_real_L,
        "img_real_R": img_real_R,
    }
    gan_model.set_input(input_sample)
    if isTrain:
        gan_model.forward()
    else:
        with torch.no_grad():
            gan_model.forward()

    # Train on PSMNet
    img_sim_L_gan = gan_model.fake_B_L
    img_sim_R_gan = gan_model.fake_B_R
    disp_gt = sample["img_disp_L"].to(cuda_device)
    depth_gt = sample["img_depth_L"].to(cuda_device)  # [bs, 1, H, W]
    img_focal_length = sample["focal_length"].to(cuda_device)
    img_baseline = sample["baseline"].to(cuda_device)

    # Resize the 2x resolution disp and depth back to H * W
    # Note this should go before apply_disparity_cu
    disp_gt = F.interpolate(
        disp_gt, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )  # [bs, 1, H, W]
    depth_gt = F.interpolate(
        depth_gt, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )  # [bs, 1, H, W]

    img_disp_r = sample["img_disp_R"].to(cuda_device)
    img_disp_r = F.interpolate(
        img_disp_r, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )
    disp_gt = apply_disparity_cu(
        img_disp_r, img_disp_r.type(torch.int)
    )  # [bs, 1, H, W]
    del img_disp_r

    # Get stereo loss on sim
    mask = (disp_gt < cfg.MODEL.MAX_DISP) * (
        disp_gt > 0
    )  # Note in training we do not exclude bg
    if isTrain:
        pred_disp1, pred_disp2, pred_disp3 = psmnet_model(img_sim_L_gan, img_sim_R_gan)
        sim_pred_disp = pred_disp3
        loss_psmnet = (
            0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt[mask], reduction="mean")
            + 0.7 * F.smooth_l1_loss(pred_disp2[mask], disp_gt[mask], reduction="mean")
            + F.smooth_l1_loss(pred_disp3[mask], disp_gt[mask], reduction="mean")
        )
    else:
        with torch.no_grad():
            sim_pred_disp = psmnet_model(img_sim_L_gan, img_sim_R_gan)
            loss_psmnet = F.smooth_l1_loss(
                sim_pred_disp[mask], disp_gt[mask], reduction="mean"
            )

    # Backward and optimization
    if isTrain:
        # update Ds
        gan_model.update_D()
        # Update Gs
        total_loss = (
            gan_model.compute_loss_G() + loss_psmnet * cfg.LOSSES.SIMRATIO
        )  # loss_G + loss_psmnet (task loss)
        # Ds require no gradient when optimizing Gs
        gan_model.set_requires_grad([gan_model.netD_A, gan_model.netD_B], False)
        gan_model.optimizer_G.zero_grad()  # set Gs' gradient to zero
        psmnet_optimizer.zero_grad()  # set cascade gradient to zero
        total_loss.backward()  # calculate gradient
        gan_model.optimizer_G.step()  # update Gs weights
        psmnet_optimizer.step()  # update cascade weights
    else:
        gan_model.compute_loss_G()
        gan_model.compute_loss_D_A()
        gan_model.compute_loss_D_B()

    # Save gan scalar outputs and images
    scalar_outputs_gan = {
        "G_A": gan_model.loss_G_A,
        "G_B": gan_model.loss_G_B,
        "cycle_A": gan_model.loss_cycle_A,
        "cycle_B": gan_model.loss_cycle_B,
        "idt_A": gan_model.loss_idt_A,
        "idt_B": gan_model.loss_idt_B,
        "D_A": gan_model.loss_D_A,
        "D_B": gan_model.loss_D_B,
    }
    img_outputs_gan = {
        "img_sim_L": {
            "input": gan_model.real_A_L,
            "fake": gan_model.fake_B_L,
            "rec": gan_model.rec_A_L,
            "idt": gan_model.idt_B_L,
        },
        "img_sim_R": {
            "input": gan_model.real_A_R,
            "fake": gan_model.fake_B_R,
            "rec": gan_model.rec_A_R,
            "idt": gan_model.idt_B_R,
        },
        "img_sim_real_L": {
            "input": gan_model.real_B_L,
            "fake": gan_model.fake_A_L,
            "rec": gan_model.rec_B_L,
            "idt": gan_model.idt_A_L,
        },
        "img_sim_real_R": {
            "input": gan_model.real_B_R,
            "fake": gan_model.fake_A_R,
            "rec": gan_model.rec_B_R,
            "idt": gan_model.idt_A_R,
        },
    }

    # Compute stereo error metrics on sim
    pred_disp = sim_pred_disp
    scalar_outputs_psmnet = {"loss": loss_psmnet.item()}
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
        "input_L": img_sim_L_gan,
        "input_R": img_sim_R_gan,
    }

    if is_distributed:
        scalar_outputs_gan = reduce_scalar_outputs(scalar_outputs_gan, cuda_device)
        scalar_outputs_psmnet = reduce_scalar_outputs(
            scalar_outputs_psmnet, cuda_device
        )
    return (
        scalar_outputs_gan,
        scalar_outputs_psmnet,
        img_outputs_gan,
        img_outputs_psmnet,
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

    # Create GAN model
    gan_model = CycleGANModel()
    gan_model.set_device(cuda_device)
    gan_model.set_distributed(is_distributed=is_distributed, local_rank=args.local_rank)

    # Create PSMNet model
    psmnet_model = PSMNet(maxdisp=cfg.MODEL.MAX_DISP).to(cuda_device)
    psmnet_optimizer = torch.optim.Adam(
        psmnet_model.parameters(), lr=cfg.SOLVER.LR, betas=cfg.SOLVER.BETAS
    )
    if is_distributed:
        psmnet_model = torch.nn.parallel.DistributedDataParallel(
            psmnet_model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    else:
        psmnet_model = torch.nn.DataParallel(psmnet_model)

    # Start training
    train(gan_model, psmnet_model, psmnet_optimizer, TrainImgLoader)
