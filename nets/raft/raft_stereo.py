import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.raft.corr import (AlternateCorrBlock, CorrBlock1D, CorrBlockFast1D,
                            PytorchAlternateCorrBlock1D)
from nets.raft.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from nets.raft.update import BasicMultiUpdateBlock
from nets.raft.raft_utils import coords_grid, upflow8

from configs.config import cfg

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFTStereo(nn.Module):
    def __init__(self):
        super().__init__()

        context_dims = cfg.MODEL.HIDDEN_DIMS

        self.cnet = MultiBasicEncoder(
            output_dim=[cfg.MODEL.HIDDEN_DIMS, context_dims],
            norm_fn="batch",
            downsample=cfg.MODEL.N_DOWNSAMPLE,
        )
        self.update_block = BasicMultiUpdateBlock(
            hidden_dims=cfg.MODEL.HIDDEN_DIMS
        )

        self.context_zqr_convs = nn.ModuleList(
            [
                nn.Conv2d(context_dims[i], cfg.MODEL.HIDDEN_DIMS[i] * 3, 3, padding=3 // 2)
                for i in range(cfg.MODEL.N_GRU_LAYERS)
            ]
        )

        if cfg.MODEL.SHARE_BACKBONE:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, "instance", stride=1),
                nn.Conv2d(128, 256, 3, padding=1),
            )
        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn="instance", downsample=cfg.MODEL.N_DOWNSAMPLE
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, D, H, W = flow.shape
        factor = 2 ** cfg.MODEL.N_DOWNSAMPLE
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """Estimate optical flow between pair of frames"""

        # image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        # image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # image1 = (2 * image1 - 1.0).contiguous()
        # image2 = (2 * image2 - 1.0).contiguous()

        # run the context network
        with autocast(enabled=cfg.MODEL.MIXED_PRECISION):
            if cfg.MODEL.SHARE_BACKBONE:
                *cnet_list, x = self.cnet(
                    torch.cat((image1, image2), dim=0),
                    dual_inp=True,
                    num_layers=cfg.MODEL.N_GRU_LAYERS,
                )
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                cnet_list = self.cnet(image1, num_layers=cfg.MODEL.N_GRU_LAYERS)
                fmap1, fmap2 = self.fnet([image1, image2])
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [
                list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                for i, conv in zip(inp_list, self.context_zqr_convs)
            ]

        if cfg.MODEL.CORR_IMPLEMENTATION == "reg":  # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif cfg.MODEL.CORR_IMPLEMENTATION == "alt":  # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif cfg.MODEL.CORR_IMPLEMENTATION == "reg_cuda":  # Faster version of reg
            corr_block = CorrBlockFast1D
        elif cfg.MODEL.CORR_IMPLEMENTATION == "alt_cuda":  # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(
            fmap1, fmap2, radius=cfg.MODEL.CORR_RADIUS, num_levels=cfg.MODEL.CORR_LEVELS
        )

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=cfg.MODEL.MIXED_PRECISION):
                if (
                    cfg.MODEL.N_GRU_LAYERS == 3 and cfg.MODEL.SLOW_FAST_GRU
                ):  # Update low-res GRU
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=True,
                        iter16=False,
                        iter08=False,
                        update=False,
                    )
                if (
                    cfg.MODEL.N_GRU_LAYERS >= 2 and cfg.MODEL.SLOW_FAST_GRU
                ):  # Update low-res GRU and mid-res GRU
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=cfg.MODEL.N_GRU_LAYERS == 3,
                        iter16=True,
                        iter08=False,
                        update=False,
                    )
                net_list, up_mask, delta_flow = self.update_block(
                    net_list,
                    inp_list,
                    corr,
                    flow,
                    iter32=cfg.MODEL.N_GRU_LAYERS == 3,
                    iter16=cfg.MODEL.N_GRU_LAYERS >= 2,
                )

            # in stereo mode, project flow onto epipolar
            delta_flow[:, 1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:, :1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
