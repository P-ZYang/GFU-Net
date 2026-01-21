from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
import ml_collections
from thop import profile
import time
from model.VIG.torch_vertex import Grapher
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
   Our complete code will be updated after the paper is accepted.
   我们的完整代码将在论文被接收后更新。
'''


class Ours(nn.Module):
    def __init__(self, config, in_channels=1, n_classes=1, img_size=256, mode='train', deepsuper=True):
        super().__init__()
        self.config = config
        self.mode = mode
        self.deepsuper = deepsuper

        self.n_classes = n_classes
        self.in_channels = config.base_channel  # 32
        block = Res_block
        self.avg_size = config.KV_size // 4

        self.pool = nn.MaxPool2d(2, 2)

        self.inc = self._make_layer(block, in_channels, self.in_channels, 1)
        self.down_encoder1 = self._make_layer(block, self.in_channels, self.in_channels * 2, 1)
        self.down_encoder2 = self._make_layer(block, self.in_channels * 2, self.in_channels * 4, 1)
        self.down_encoder3 = self._make_layer(block, self.in_channels * 4, self.in_channels * 8, 1)
        self.down_encoder4 = self._make_layer(block, self.in_channels * 8, self.in_channels * 8, 1)

        self.GPHer = Grapher(in_channels=config.KV_size, kernel_size=9, dilation=1, conv='edge', act='gelu',
                             norm='batch', bias=True,
                             stochastic=False, epsilon=0.1, r=1, n=img_size // 4 * img_size // 4, drop_path=0.0,
                             relative_pos=True)

        self.up_decoder4 = UpBlock_attention(self.in_channels * 16, self.in_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(self.in_channels * 8, self.in_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(self.in_channels * 4, self.in_channels, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(self.in_channels * 2, self.in_channels, nb_Conv=2)

        self.outc = nn.Conv2d(self.in_channels, self.n_classes, kernel_size=1, stride=1)

        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(self.in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(self.in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(self.in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(self.in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        # print("x2:", x2.shape)
        x3 = self.down_encoder2(self.pool(x2))
        # print("x3:", x3.shape)
        x4 = self.down_encoder3(self.pool(x3))
        # print("x4:", x4.shape)
        d5 = self.down_encoder4(self.pool(x4))
        # print("d5:", d5.shape)

        org1, org2, org3, org4 = x1, x2, x3, x4

        _, c1, h1, w1 = x1.shape
        _, c2, h2, w2 = x2.shape
        _, c3, h3, w3 = x3.shape
        _, c4, h4, w4 = x4.shape

        x1 = F.interpolate( x1 , size=(h3, w3), mode='bilinear', align_corners=True)
        x2 = F.interpolate( x2, size=(h3, w3), mode='bilinear', align_corners=True)
        # x3 = F.interpolate(x3, size=(h2, w2), mode='bilinear', align_corners=True)
        x4 = F.interpolate( x4, size=(h3, w3), mode='bilinear', align_corners=True)

        embx1 = torch.cat([x1, x2, x3, x4], dim=1)

        embx2 = self.GPHer(embx1)

        # embx3 = self.ffn(embx2)

        embx3 = self.fcfn(embx2)

        f1, f2, f3, f4 = torch.split(embx3, [c1, c2, c3, c4], dim=1)

        f1 = F.interpolate(f1, size=(h1, w1), mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, size=(h2, w2), mode='bilinear', align_corners=True)
        # f3 = F.interpolate(f3, size=(h3, w3), mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=(h4, w4), mode='bilinear', align_corners=True)

        r1 = self.rt1(f1) + org1
        r2 = self.rt2(f2) + org2
        r3 = self.rt3(f3) + org3
        r4 = self.rt4(f4) + org4

        d4 = self.up_decoder4(d5, r4)
        d3 = self.up_decoder3(d4, r3)
        d2 = self.up_decoder2(d3, r2)
        d1 = self.up_decoder1(d2, r1)
        out = self.outc(d1)

        if self.deepsuper:
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)
            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out))
            else:
                return torch.sigmoid(out)
        else:
            return torch.sigmoid(out)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)







