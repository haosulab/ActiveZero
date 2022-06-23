from configs.config import cfg
from utils.reprojection import get_reproj_error_patch
import torch.nn.functional as F
import torch
import numpy as np

def psmnet_disp(pred_disp, disp_gt_l, mask):
    pred_disp3, pred_disp2, pred_disp1 = pred_disp
    loss_disp = (
        0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt_l[mask], reduction="mean")
        + 0.7
        * F.smooth_l1_loss(pred_disp2[mask], disp_gt_l[mask], reduction="mean")
        + F.smooth_l1_loss(pred_disp3[mask], disp_gt_l[mask], reduction="mean")
    )
    return loss_disp

def dispnet_disp(disp_ests, disp_gt, mask):
    scale = [0, 1, 2, 3, 4, 5, 6]
    weights = [1, 1, 1, 0.8, 0.6, 0.4, 0.2]
    all_losses = []
    for disp_est, weight, s in zip(disp_ests, weights, scale):
        if s != 0:
            dgt = F.interpolate(disp_gt, scale_factor=1 / (2 ** s))
            m = F.interpolate(mask.float(), scale_factor=1 / (2 ** s)).byte()
        else:
            dgt = disp_gt
            m = mask
        all_losses.append(
            weight
            * F.smooth_l1_loss(disp_est[m], dgt[m], size_average=True, reduction="mean")
        )
    return sum(all_losses)

def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):

    """Loss function defined over sequence of flow predictions"""

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    flow_gt = - flow_gt # convert from disp_gt to flow_gt

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = (valid >= 0.5) & (mag < max_flow).unsqueeze(1)  # [bs, 1, H, W]
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert (
            not torch.isnan(flow_preds[i]).any()
            and not torch.isinf(flow_preds[i]).any()
        )
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [
            i_loss.shape,
            valid.shape,
            flow_gt.shape,
            flow_preds[i].shape,
        ]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    return flow_loss

def default_disp(pred_disp, disp_gt_l, mask):
    return F.smooth_l1_loss(pred_disp[mask], disp_gt_l[mask], reduction="mean")

class AllLosses():
    def __init__(self, model, name, adapter=True):
        self.model = model # main backbone model, output format should be main prediction as first index
        self.name = name
        self.adapter = adapter

    # compute total loss
    def compute_loss(self, item, onSim=True, train=True):
        loss = 0
        loss_vals = {}
        loss_disp, item = self.compute_disp_loss(item, onSim, train)
        if (cfg.LOSSES.DISP_LOSS and onSim):
            loss+=loss_disp
            loss_vals['disp'] = loss_disp.item()
        if (cfg.LOSSES.REPROJECTION_LOSS):
            if (not onSim and cfg.LOSSES.REPROJECTION.TRAINREAL):
                loss_reproj, item = self.compute_reprojection_loss(item, onSim)
                loss += cfg.LOSSES.REPROJECTION.REALRATIO * loss_reproj
                loss_vals['reproject'] = loss_reproj.item()
            if (onSim and cfg.LOSSES.REPROJECTION.TRAINSIM):
                loss_reproj, item = self.compute_reprojection_loss(item, onSim)
                loss += cfg.LOSSES.REPROJECTION.SIMRATIO * loss_reproj
                loss_vals['reproject'] = loss_reproj.item()

        return loss, item, loss_vals

    def forward(self, item, train=True):
        type = self.name
        if train:
            if self.adapter and type=="psmnet":
                output = self.model(
                    item['img_L'], item['img_R'], item['img_L_transformed'],
                    item['img_R_transformed'])
                pred_disp = output[0]
            elif type=="psmnet":
                output = self.model(item['img_L'], item['img_R'])
                pred_disp = output[0]
            elif type=="dispnet":
                input = torch.cat((item['img_L'], item['img_R']),dim=1)
                output = self.model(input)
                pred_disp = output[0]
            elif type=="raft":
                output = self.model(item['img_L'], item['img_R'], iters=cfg.MODEL.TRAIN_ITERS)
                pred_disp = -output[-1]
        else:
            with torch.no_grad():
                if self.adapter and type=="psmnet":
                    pred_disp = self.model(
                        item['img_L'], item['img_R'], item['img_L_transformed'],
                        item['img_R_transformed'])
                    output = pred_disp
                elif type=="psmnet":
                    pred_disp = self.model(item['img_L'], item['img_R'])
                    output = pred_disp
                elif type=="dispnet":
                    input = torch.cat((item['img_L'], item['img_R']),dim=1)
                    output = self.model(input)
                    pred_disp = output[0]
                elif type=="raft":
                    output = self.model(item['img_L'], item['img_R'], iters=cfg.MODEL.TRAIN_ITERS)
                    pred_disp = -output[-1]

        return output, pred_disp

    def compute_reprojection_loss(self, item, onSim):
        if onSim:
            sim_ir_reproj_loss, sim_ir_warped, sim_ir_reproj_mask = get_reproj_error_patch(
                input_L=item['img_L_reproj'],
                input_R=item['img_R_reproj'],
                pred_disp_l=item['sim_pred_disp'],
                mask=item['mask'],
                ps=cfg.LOSSES.REPROJECTION.PATCH_SIZE,
            )
            item['sim_ir_warped'] = sim_ir_warped
            item['sim_ir_reproj_mask'] = sim_ir_reproj_mask
            reproj_loss = sim_ir_reproj_loss
        else:
            real_ir_reproj_loss, real_ir_warped, real_ir_reproj_mask = get_reproj_error_patch(
                input_L=item['img_real_L_reproj'],
                input_R=item['img_real_R_reproj'],
                pred_disp_l=item['real_pred_disp'],
                ps=cfg.LOSSES.REPROJECTION.PATCH_SIZE,
            )
            item['real_ir_warped'] = real_ir_warped
            item['real_ir_reproj_mask'] = real_ir_reproj_mask
            reproj_loss = real_ir_reproj_loss
        return reproj_loss, item

    def compute_disp_loss(self, item, onSim, train):
        type = self.name
        mask = item['mask']

        # determines type of disp loss to use
        if type=='psmnet' and train:
            func = psmnet_disp
        elif type=='dispnet':
            func = dispnet_disp
        elif type=='raft':
            func = sequence_loss
        else:
            func = default_disp

        # for disparity loss during training on sim
        loss_disp = 0
        if onSim:
            disp_gt_l = item['disp_gt_l']

            values = {
                'img_L': item['img_sim_L'],
                'img_R': item['img_sim_R'],
            }
            if self.adapter:
                values['img_L_transformed'] = item['img_sim_L_transformed']
                values['img_R_transformed'] = item['img_sim_R_transformed']
            output, pred_disp = self.forward(values, train)
            loss_disp = func(output,disp_gt_l,mask)
            item['sim_pred_disp'] = pred_disp

        # for disparity during training on real
        elif not onSim:
            values = {
                'img_L': item['img_real_L'],
                'img_R': item['img_real_R'],
            }
            if self.adapter:
                values['img_L_transformed'] = item['img_real_L_transformed']
                values['img_R_transformed'] = item['img_real_R_transformed']
            output, pred_disp = self.forward(values, train)
            item['real_pred_disp'] = pred_disp

        return loss_disp, item
