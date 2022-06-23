import argparse
import gc
import os

import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from datasets.messytable import MessytableDataset
from utils.cascade_metrics import compute_err_metric
from configs.config import cfg
from utils.reduce import (AverageMeterDict, reduce_scalar_outputs, set_random_seed,
                            synchronize, tensor2float, tensor2numpy)
from utils.util import (adjust_learning_rate, disp_error_img, save_images,
                        save_images_grid, save_scalars, setup_logger)
from utils.warp_ops import apply_disparity_cu
from utils.losses import AllLosses # put all new losses here

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

# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file ./configs/train_psmnet.yaml

def train(model, model_optimizer, extra, loss_class, TrainImgLoader, ValImgLoader):
    cur_err = np.inf
    if cfg.MODEL.ADAPTER:
        adapter_model, adapter_optimizer = extra
    elif cfg.MODEL.BACKBONE == "raft":
        model_scheduler, model_scaler = extra

    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # One epoch training loop
        avg_train_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = (
                (len(TrainImgLoader) * epoch_idx + batch_idx)
                * cfg.SOLVER.BATCH_SIZE
                * num_gpus
            )
            if global_step > cfg.SOLVER.STEPS:
                break

            # Adjust learning rate
            if cfg.MODEL.ADAPTER:
                adjust_learning_rate(
                    adapter_optimizer,
                    global_step,
                    cfg.SOLVER.LR,
                    cfg.SOLVER.LR_STEPS,
                )
            if cfg.MODEL.BACKBONE != "raft":
                adjust_learning_rate(
                    model_optimizer,
                    global_step,
                    cfg.SOLVER.LR,
                    cfg.SOLVER.LR_STEPS,
                )

            do_summary = global_step % cfg.SOLVER.SUMMARY_FREQ == 0
            # Train one sample
            # additional output contains all per metric outputs
            scalar_outputs, img_outputs, additional_output = train_sample(
                sample,
                model,
                model_optimizer,
                extra,
                loss_class,
                isTrain=True,
            )
            # Save result to tensorboard
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs = tensor2float(scalar_outputs)
                avg_train_scalars.update(scalar_outputs)
                if do_summary:
                    # Update reprojection images

                    if cfg.LOSSES.REPROJECTION_LOSS:
                        img_output_reproj = additional_output["img_output_reproj"]
                        save_images_grid(
                            summary_writer,
                            "train_reproj",
                            img_output_reproj,
                            global_step,
                            nrow=4,
                        )
                    # Update PSMNet images
                    save_images(summary_writer, "train", img_outputs, global_step)
                    # Update PSMNet losses
                    scalar_outputs.update(
                        {"lr": model_optimizer.param_groups[0]["lr"]}
                    )
                    save_scalars(
                        summary_writer,
                        "train",
                        scalar_outputs,
                        global_step,
                    )

                # Save checkpoints
                if (global_step) % cfg.SOLVER.SAVE_FREQ == 0:
                    checkpoint_data = {
                        "epoch": epoch_idx,
                        "Model": model.state_dict(),
                        "optimizerModel": model_optimizer.state_dict(),
                    }

                    if cfg.MODEL.ADAPTER:
                        checkpoint_data["optimizerAdapter"]= adapter_optimizer.state_dict()
                        checkpoint_data["Adapter"]=adapter_model.state_dict()

                    save_filename = os.path.join(
                        cfg.SOLVER.LOGDIR, "models", f"model_{global_step}.pth"
                    )
                    torch.save(checkpoint_data, save_filename)

                    # Get average results among all batches
                    total_err_metric = avg_train_scalars.mean()
                    logger.info(
                        f"Step {global_step} train model: {total_err_metric}"
                    )
        gc.collect()

        avg_val_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(ValImgLoader):
            global_step = (len(ValImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE

            do_summary = global_step % cfg.SOLVER.SUMMARY_FREQ == 0
            scalar_outputs, img_outputs, additional_output = \
                train_sample(sample, model, model_optimizer, extra, loss_class, isTrain=False)
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs = tensor2float(scalar_outputs)
                avg_val_scalars.update(scalar_outputs)
                if do_summary:
                    # Update PSMNet images
                    save_images(summary_writer, 'val', img_outputs, global_step)
                    # Update Cascade losses
                    scalar_outputs.update({'lr': model_optimizer.param_groups[0]['lr']})
                    save_scalars(summary_writer, 'val', scalar_outputs, global_step)

        if (not is_distributed) or (dist.get_rank() == 0):
            # Get average results among all batches
            total_err_metric = avg_val_scalars.mean()
            logger.info(f'Epoch {epoch_idx} val   model    : {total_err_metric}')

            # Save best checkpoints
            new_err = total_err_metric['depth_abs_err'][0] if num_gpus > 1 \
                else total_err_metric['depth_abs_err']
            if new_err < cur_err:
                cur_err = new_err
                checkpoint_data = {
                    "epoch": epoch_idx,
                    "Model": model.state_dict(),
                    "optimizerModel": model_optimizer.state_dict(),
                }

                if cfg.MODEL.ADAPTER:
                    checkpoint_data["optimizerAdapter"]= adapter_optimizer.state_dict()
                    checkpoint_data["Adapter"]=adapter_model.state_dict()
                save_filename = os.path.join(cfg.SOLVER.LOGDIR, 'models', f'model_best.pth')
                torch.save(checkpoint_data, save_filename)
        gc.collect()


def train_sample(sample, model, model_optimizer, extra, loss_class, isTrain=True):

    if cfg.MODEL.ADAPTER:
        adapter_model, adapter_optimizer = extra
        if isTrain and cfg.LOSSES.ONSIM:
            adapter_model.train()
        else:
            adapter_model.eval()
    elif cfg.MODEL.BACKBONE == "raft":
        model_scheduler, model_scaler = extra

    if isTrain and cfg.LOSSES.ONSIM:
        model.train()
    else:
        model.eval()

    # Load data
    img_L = sample["img_sim_L"].to(cuda_device)  # [bs, 3, H, W]
    img_R = sample["img_sim_R"].to(cuda_device)

    if (cfg.LOSSES.REPROJECTION_LOSS and cfg.LOSSES.REPROJECTION.TRAINSIM):
        img_L_reproj = sample["img_sim_L_reproj"].to(cuda_device)  # [bs, 1, H, W]
        img_R_reproj = sample["img_sim_R_reproj"].to(cuda_device)

    # Train on simple Adapter
    if cfg.MODEL.ADAPTER:
        img_L_transformed, img_R_transformed = adapter_model(img_L, img_R)  # [bs, 3, H, W]

    disp_gt_l = sample["img_disp_L"].to(cuda_device)
    depth_gt = sample["img_depth_L"].to(cuda_device)  # [bs, 1, H, W]
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

    img_disp_r = sample["img_disp_R"].to(cuda_device)
    img_disp_r = F.interpolate(
        img_disp_r, scale_factor=0.5, mode="nearest", recompute_scale_factor=False
    )
    disp_gt_l = apply_disparity_cu(
        img_disp_r, img_disp_r.type(torch.int)
    )  # [bs, 1, H, W]
    del img_disp_r

    # Get stereo loss on sim
    mask = (disp_gt_l < cfg.MODEL.MAX_DISP) * (disp_gt_l > 0)  # Note in training we do not exclude bg
    item = {}
    item['img_sim_L'] = img_L
    item['img_sim_R'] = img_R
    item['mask'] = mask
    item['disp_gt_l'] = disp_gt_l

    if (cfg.LOSSES.REPROJECTION_LOSS and cfg.LOSSES.REPROJECTION.TRAINSIM):
        item['img_L_reproj'] = img_L_reproj
        item['img_R_reproj'] = img_R_reproj
    if cfg.MODEL.ADAPTER:
        item['img_sim_L_transformed'] = img_L_transformed
        item['img_sim_R_transformed'] = img_R_transformed

    if cfg.LOSSES.ONSIM:
        if isTrain and cfg.MODEL.ADAPTER:
            adapter_optimizer.zero_grad()
            model_optimizer.zero_grad()
        elif isTrain:
            model_optimizer.zero_grad()

    sim_loss, item, sim_loss_vals = loss_class.compute_loss(item, onSim=True,
                                        train= (isTrain & cfg.LOSSES.ONSIM))
    sim_loss = cfg.LOSSES.SIMRATIO * sim_loss

    if cfg.LOSSES.ONSIM: # trained on sim
        if isTrain and cfg.MODEL.ADAPTER:
            sim_loss.backward()
            model_optimizer.step()
            adapter_optimizer.step()
        elif isTrain and backbone == "raft":
            model_scaler.scale(sim_loss).backward()
            model_scaler.unscale_(model_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            model_scaler.step(model_optimizer)
            model_scheduler.step()
            model_scaler.update()
        elif isTrain:
            sim_loss.backward()
            model_optimizer.step()

    # on real
    real_loss_vals = {}
    if cfg.LOSSES.ONREAL:
        item, real_loss_vals = train_sample_onreal(sample, item, model, model_optimizer,
                                                    extra, isTrain=isTrain)

    # Save reprojection outputs and images
    additional = {}
    if cfg.LOSSES.REPROJECTION_LOSS:
        img_output_reproj = {}
        if cfg.LOSSES.REPROJECTION.TRAINSIM:
            img_output_reproj["sim_reprojection"]= {
                "target": item["img_L_reproj"],
                "warped": item['sim_ir_warped'],
                "pred_disp": item['sim_pred_disp'],
                "mask": item['sim_ir_reproj_mask'],
            }
        if cfg.LOSSES.REPROJECTION.TRAINREAL:
            img_output_reproj["real_reprojection"]= {
                "target": item["img_real_L_reproj"],
                "warped": item['real_ir_warped'],
                "pred_disp": item['real_pred_disp'],
                "mask": item['real_ir_reproj_mask'],
            }
        additional["img_output_reproj"] = img_output_reproj

    pred_disp = item['sim_pred_disp']

    scalar_outputs = {}
    for name, loss in sim_loss_vals.items():
        scalar_outputs["sim_"+name] = loss
    for name, loss in real_loss_vals.items():
        scalar_outputs["real_"+name] = loss

    err_metrics = compute_err_metric(
        disp_gt_l, depth_gt, pred_disp, img_focal_length, img_baseline, mask
    )
    scalar_outputs.update(err_metrics)
    # Compute error images
    pred_disp_err_np = disp_error_img(pred_disp[[0]], disp_gt_l[[0]], mask[[0]])
    pred_disp_err_tensor = torch.from_numpy(
        np.ascontiguousarray(pred_disp_err_np[None].transpose([0, 3, 1, 2]))
    )
    img_outputs = {
        "disp_gt_l": disp_gt_l[[0]].repeat([1, 3, 1, 1]),
        "disp_pred": pred_disp[[0]].repeat([1, 3, 1, 1]),
        "disp_err": pred_disp_err_tensor,
        "input_L": img_L,
        "input_R": img_R,
    }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs, cuda_device)
    return scalar_outputs, img_outputs, additional

def train_sample_onreal(sample, item, model, model_optimizer, extra, isTrain=True):
    if cfg.MODEL.ADAPTER:
        adapter_model, adapter_optimizer = extra
        if isTrain and cfg.LOSSES.ONREAL:
            adapter_model.train()
        else:
            adapter_model.eval()
    elif cfg.MODEL.BACKBONE == "raft":
        model_scheduler, model_scaler = extra

    if isTrain and cfg.LOSSES.ONREAL:
        model.train()
    else:
        model.eval()

    # Get reprojection loss on real
    img_real_L = sample["img_real_L"].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_R = sample["img_real_R"].to(cuda_device)  # [bs, 3, 2H, 2W]

    if cfg.MODEL.ADAPTER:
        img_real_L_transformed, img_real_R_transformed = adapter_model(
            img_real_L, img_real_R
        )  # [bs, 3, H, W]

    item['img_real_L'] = img_real_L
    item['img_real_R'] = img_real_R

    if (cfg.LOSSES.REPROJECTION_LOSS and cfg.LOSSES.REPROJECTION.TRAINREAL):
        item['img_real_L_reproj'] = sample["img_real_L_reproj"].to(cuda_device)
        item['img_real_R_reproj'] = sample["img_real_R_reproj"].to(cuda_device)

    if cfg.MODEL.ADAPTER:
        item['img_real_L_transformed'] = img_real_L_transformed
        item['img_real_R_transformed'] = img_real_R_transformed

    if cfg.LOSSES.ONREAL:
        if isTrain and cfg.MODEL.ADAPTER:
            adapter_optimizer.zero_grad()
            model_optimizer.zero_grad()
        elif isTrain:
            model_optimizer.zero_grad()

    real_loss, item, real_loss_vals = loss_class.compute_loss(item, onSim=False,
                                        train= (isTrain & cfg.LOSSES.ONREAL))
    real_loss = cfg.LOSSES.REALRATIO * real_loss

    if cfg.LOSSES.ONREAL: # trained on real
        if isTrain and cfg.MODEL.ADAPTER:
            real_loss.backward()
            model_optimizer.step()
            adapter_optimizer.step()
        elif isTrain and backbone == "raft":
            model_scaler.scale(real_loss).backward()
            model_scaler.unscale_(model_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            model_scaler.step(model_optimizer)
            model_scheduler.step()
            model_scaler.update()
        elif isTrain:
            real_loss.backward()
            model_optimizer.step()

    return item, real_loss_vals

if __name__ == "__main__":

    # Obtain dataloader
    train_dataset = MessytableDataset(
        cfg.SIM.TRAIN, onReal=cfg.LOSSES.ONREAL, special=[cfg.LOSSES.REPROJECTION.PATTERN,]
    )
    val_dataset = MessytableDataset(
        cfg.SIM.VAL, onReal=cfg.LOSSES.ONREAL, special=[cfg.LOSSES.REPROJECTION.PATTERN,]
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

    # Create Adapter model
    if cfg.MODEL.ADAPTER:
        from nets.adapter import Adapter
        adapter_model = Adapter().to(cuda_device)
        adapter_optimizer = torch.optim.Adam(
            adapter_model.parameters(), lr=cfg.SOLVER.LR, betas=(0.9, 0.999)
        )
        if is_distributed:
            adapter_model = torch.nn.parallel.DistributedDataParallel(
                adapter_model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )
        else:
            adapter_model = torch.nn.DataParallel(adapter_model)

    # load backbone
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
        model.weight_bias_init()
    elif backbone=="raft":
        from nets.raft.raft_stereo import RAFTStereo
        model = RAFTStereo().to(cuda_device)
    else:
        print("Model not implemented!")

    if backbone == "raft":
        model_optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, eps=1e-8
        )
        model_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model_optimizer,
            cfg.SOLVER.LR,
            cfg.SOLVER.STEPS + 100,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
        model_scaler = GradScaler(enabled=cfg.MODEL.MIXED_PRECISION)
    else:
        model_optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.SOLVER.LR, betas= cfg.SOLVER.BETAS
        )

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    else:
        model = torch.nn.DataParallel(model)

    loss_class = AllLosses(model, cfg.MODEL.BACKBONE, cfg.MODEL.ADAPTER)

    # Start training
    if backbone == "raft":
        train(model, model_optimizer, [model_scheduler, model_scaler], loss_class, TrainImgLoader, ValImgLoader)
    elif cfg.MODEL.ADAPTER:
        train(model, model_optimizer, [adapter_model, adapter_optimizer], loss_class, TrainImgLoader, ValImgLoader)
    else:
        train(model, model_optimizer, [], loss_class, TrainImgLoader, ValImgLoader)
