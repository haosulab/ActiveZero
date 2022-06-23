# https://github.com/LettieZ/dispnet

import torch
import torch.nn.functional as F


class DispNet(torch.nn.Module):
    def __init__(self):
        super(DispNet, self).__init__()

        # encoder part start
        self.conv1 = torch.nn.Conv2d(
            in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2
        )
        self.conv3a = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2
        )
        self.conv3b = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv4a = torch.nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        self.conv4b = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv5a = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        self.conv5b = torch.nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv6a = torch.nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1
        )
        self.conv6b = torch.nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1
        )

        # this layer is to produce pr6
        self.conv_predict_flow6 = torch.nn.Conv2d(
            in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        # encoder part end

        # decoder part start
        self.upconv5 = torch.nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
        )

        # this layer is to expand pr6 to larger size
        self.upsample_flow6to5 = torch.nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
        )
        self.iconv5 = torch.nn.Conv2d(
            in_channels=1025, out_channels=512, kernel_size=3, stride=1, padding=1
        )

        # this layer is to produce pr5
        self.conv_predict_flow5 = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        self.upconv4 = torch.nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
        )

        # this layer is to expand pr5 to larger size
        self.upsample_flow5to4 = torch.nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
        )
        self.iconv4 = torch.nn.Conv2d(
            in_channels=769, out_channels=256, kernel_size=3, stride=1, padding=1
        )

        # this layer is to produce pr4
        self.conv_predict_flow4 = torch.nn.Conv2d(
            in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
        )

        # this layer is to expand pr4 to larger size
        self.upsample_flow4to3 = torch.nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
        )
        self.iconv3 = torch.nn.Conv2d(
            in_channels=385, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        # this layer is to produce pr3
        self.conv_predict_flow3 = torch.nn.Conv2d(
            in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )

        # this layer is to expand pr3 to larger size
        self.upsample_flow3to2 = torch.nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
        )
        self.iconv2 = torch.nn.Conv2d(
            in_channels=193, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # this layer is to produce pr2
        self.conv_predict_flow2 = torch.nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )

        # this layer is to expand pr2 to larger size
        self.upsample_flow2to1 = torch.nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
        )
        self.iconv1 = torch.nn.Conv2d(
            in_channels=97, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        # this layer is to produce pr1
        self.conv_predict_flow1 = torch.nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1
        )

        self.upsample_flow1to0 = torch.nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
        )

    # decoder part end

    # weight and bias initialization remain undone

    def forward(self, x):

        conv1 = self.conv1(x)
        conv1 = F.leaky_relu(conv1, negative_slope=0.1)
        # print(conv1.shape)

        conv2 = self.conv2(conv1)
        conv2 = F.leaky_relu(conv2, negative_slope=0.1)
        # print(conv2.shape)

        conv3a = self.conv3a(conv2)
        conv3a = F.leaky_relu(conv3a, negative_slope=0.1)
        # print(conv3a.shape)

        conv3b = self.conv3b(conv3a)
        conv3b = F.leaky_relu(conv3b, negative_slope=0.1)
        # print(conv3b.shape)

        conv4a = self.conv4a(conv3b)
        conv4a = F.leaky_relu(conv4a, negative_slope=0.1)
        # print(conv4a.shape)

        conv4b = self.conv4b(conv4a)
        conv4b = F.leaky_relu(conv4b, negative_slope=0.1)
        # print(conv4b.shape)

        conv5a = self.conv5a(conv4b)
        conv5a = F.leaky_relu(conv5a, negative_slope=0.1)
        # print(conv5a.shape)

        conv5b = self.conv5b(conv5a)
        conv5b = F.leaky_relu(conv5b, negative_slope=0.1)
        # print(conv5b.shape)

        conv6a = self.conv6a(conv5b)
        conv6a = F.leaky_relu(conv6a, negative_slope=0.1)
        # print(conv6a.shape)

        conv6b = self.conv6b(conv6a)
        conv6b = F.leaky_relu(conv6b, negative_slope=0.1)
        # print(conv6b.shape)

        pr6 = self.conv_predict_flow6(conv6b)
        # print(pr6.shape)

        upconv5 = self.upconv5(conv6b)
        upconv5 = F.leaky_relu(upconv5, negative_slope=0.1)
        # print(upconv5.shape)

        larger_pr6 = self.upsample_flow6to5(pr6)
        # print(larger_pr6.shape)
        iconv5 = torch.cat((upconv5, larger_pr6, conv5b), 1)
        # print(iconv5.shape)
        iconv5 = self.iconv5(iconv5)
        # print(iconv5.shape)

        pr5 = self.conv_predict_flow5(iconv5)
        # print(pr5.shape)

        upconv4 = self.upconv4(iconv5)
        upconv4 = F.leaky_relu(upconv4, negative_slope=0.1)
        # print(upconv4.shape)
        larger_pr5 = self.upsample_flow5to4(pr5)
        # print(larger_pr5.shape)

        iconv4 = torch.cat((upconv4, larger_pr5, conv4b), 1)
        # print(iconv4.shape)
        iconv4 = self.iconv4(iconv4)
        # print(iconv4.shape)

        pr4 = self.conv_predict_flow4(iconv4)
        # print(pr4.shape)

        upconv3 = self.upconv3(iconv4)
        upconv3 = F.leaky_relu(upconv3, negative_slope=0.1)
        # print(upconv3.shape)

        larger_pr4 = self.upsample_flow4to3(pr4)
        # print(larger_pr4.shape)
        iconv3 = torch.cat((upconv3, larger_pr4, conv3b), 1)
        # print(iconv3.shape)
        iconv3 = self.iconv3(iconv3)
        # print(iconv3.shape)

        pr3 = self.conv_predict_flow3(iconv3)
        # print(pr3.shape)

        upconv2 = self.upconv2(iconv3)
        upconv2 = F.leaky_relu(upconv2, negative_slope=0.1)
        # print(upconv2.shape)

        larger_pr3 = self.upsample_flow3to2(pr3)
        # print(larger_pr3.shape)
        iconv2 = torch.cat((upconv2, larger_pr3, conv2), 1)
        # print(iconv2.shape)
        iconv2 = self.iconv2(iconv2)
        # print(iconv2.shape)

        pr2 = self.conv_predict_flow2(iconv2)
        # print(pr2.shape)

        upconv1 = self.upconv1(iconv2)
        upconv1 = F.leaky_relu(upconv1, negative_slope=0.1)
        # print(upconv1.shape)

        larger_pr2 = self.upsample_flow2to1(pr2)
        # print(larger_pr2.shape)
        iconv1 = torch.cat((upconv1, larger_pr2, conv1), 1)
        # print(iconv1.shape)
        iconv1 = self.iconv1(iconv1)
        # print(iconv1.shape)

        pr1 = self.conv_predict_flow1(iconv1)
        # print(pr1.shape)

        pr0 = self.upsample_flow1to0(pr1)[:, :, : x.shape[-2], : x.shape[-1]]

        return pr0, pr1, pr2, pr3, pr4, pr5, pr6

    def weight_bias_init(self):
        for i in self.children():
            torch.nn.init.kaiming_normal_(
                i.weight.data, a=0.1, nonlinearity="leaky_relu"
            )
            torch.nn.init.constant_(i.bias.data, 0)
