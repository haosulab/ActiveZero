"""
Author: Isabella Liu 4/26/21
Feature: CNN (feature extraction) and SPP modules
Reference: https://github.com/JiaRenChang/PSMNet/blob/master/models/submodule.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    """Combination of conv2d and batchnorm2d"""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
    )


def conv(in_planes, out_planes, kernel_size, stride, pad, dilation):
    """Combination of conv2d and batchnorm2d"""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=False,
        )
    )


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    """Combination of conv3d and barchnorm3d"""
    return nn.Sequential(
        nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm3d(out_planes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True)
        )
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.disp = torch.Tensor(
            np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])
        ).cuda()

    def forward(self, x):
        out = torch.sum(x * self.disp, 1, keepdim=True)
        return out


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        # CNN module
        self.inplanes = 32
        # conv0_1, conv0_2, conv0_3
        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)  # conv1_x
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)  # conv2_x
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)  # conv3_x
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)  # conv4_x

        # SPP module
        # self.branch1 = nn.Sequential(
        #     nn.AvgPool2d((64, 64), stride=(64, 64)),
        #     convbn(128, 32, 1, 1, 0, 1),
        #     nn.ReLU(inplace=True)
        # )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
        )
        self.lastconv = nn.Sequential(
            convbn(288, 128, 3, 1, 1, 1),
            # convbn(320, 128, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False),
        )

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        """
        :param block: Block type
        :param planes: Output planes
        :param blocks: Number of blocks
        :param stride: Stride
        :param pad: Pad
        :param dilation: Dilation
        :return: Block network
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   [bs, 3, H, W]
        :return:    [bs, 32, H/4, W/4]
        """
        # CNN module
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)  # conv2_16 [bs, 64, H/2, W/2]
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)  # conv4_3 [bs, 128, H/2, W/2]

        # # SPP module
        [H, W] = output_skip.size()[-2:]
        # output_branch1 = self.branch1(output_skip)
        # output_branch1 = F.interpolate(output_branch1, (H, W), mode='bilinear', align_corners=True)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(
            output_branch2, (H, W), mode="bilinear", align_corners=True
        )

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(
            output_branch3, (H, W), mode="bilinear", align_corners=True
        )

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(
            output_branch4, (H, W), mode="bilinear", align_corners=True
        )

        output_feature = torch.cat(
            (
                output_raw,
                output_skip,
                output_branch4,
                output_branch3,
                output_branch2,
            ),
            1,
        )
        # output_feature = torch.cat((
        #     output_raw,
        #     output_skip,
        #     output_branch4,
        #     output_branch3,
        #     output_branch2,
        #     output_branch1
        # ), 1)
        output_feature = self.lastconv(output_feature)  # [bs, 32, H/4, W/4]
        return output_feature


if __name__ == "__main__":
    # Unit test
    img_test = torch.rand(1, 1, 540, 960).cuda()
    feature_extraction = FeatureExtraction().cuda()
    output_features = feature_extraction(img_test)
    print(output_features.shape)  # torch.Size([1, 32, 135, 240])

    # Test backward
    feature_target = torch.rand(1, 32, 135, 240).cuda()
    mask = (feature_target > 0.1) * (feature_target < 0.8)
    loss = F.smooth_l1_loss(feature_target[mask], output_features[mask])
    loss.backward()
    print(f"Loss {loss.item()}")
