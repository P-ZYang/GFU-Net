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

def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32
    config.n_classes = 1

    # ********** useless **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process_mag = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.ReLU(inplace=True),
            eca_layer_2d(expand * nc),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.ReLU(inplace=True),
            eca_layer_2d(expand * nc),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)     
        pha = torch.angle(x_freq)   
        mag = mag + self.process_mag(mag)
        pha = pha + self.process_pha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

class GFA(nn.Module):
    def __init__(self, channels, height, width, reduction=16, expand=2):
        super().__init__()
        self.height, self.width = height, width

        self.fft_width = width // 2 + 1

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.M_spec = nn.Parameter(torch.randn(1, channels, height, self.fft_width) * 0.01)
        self.sigmoid = nn.Sigmoid()

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, expand * channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expand * channels, channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='backward')   
        mag = torch.abs(x_fft)
        pha = torch.angle(x_fft)

        g_c = self.fc(self.global_pool(x).view(B, C)).view(B, C, 1, 1)
        M = self.sigmoid(self.M_spec)
        M = F.interpolate(M, size=(mag.size(2), mag.size(3)), mode='bilinear', align_corners=True)
        mag_enhanced = self.mlp(mag) * g_c * M

        real = mag_enhanced * torch.cos(pha)
        imag = mag_enhanced * torch.sin(pha)
        x_fft_mod = torch.complex(real, imag)  

        x_ifft = torch.fft.irfft2(x_fft_mod, s=(H, W), norm='backward')
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        g_s = self.spatial_gate(torch.cat([max_pool, avg_pool], dim=1))  # [B,1,H,W]

        out = x + x_ifft * g_s
        return out

class FCFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        self.patch_size = 8
        self.project_in = nn.Conv2d(in_channels=in_features, out_channels=hidden_features*2, kernel_size=1, stride=1)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features)
        self.gfa = GFA(hidden_features, 64, 64)
        self.gfa = GFA(hidden_features, 64, 64)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, out_features, kernel_size=1)
        self.eca = eca_layer_2d(out_features)
        self.fft1_3 = nn.Parameter(torch.ones((hidden_features, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.fft1_5 = nn.Parameter(torch.ones((hidden_features, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

    def forward(self, x):
        x_3, x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.dwconv3x3(x_3)
        x1_5 = self.dwconv5x5(x_5)
        x1_3 = self.relu3(self.gfa(x1_3))
        x1_5 = self.relu5(self.gfa(x1_5))
        x = torch.cat([x1_3, x1_5], dim=1)
        x = self.project_out(x)
        return self.eca(x)

class FreCCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.fre1 = FreMLP(F_x)
        self.fre2 = FreMLP(F_g)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.fre1(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.fre2(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = FreCCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


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

        self.fcfn = FCFN(in_features=config.KV_size, hidden_features=config.KV_size, out_features=config.KV_size, act='relu')

        self.rt1 = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1),
                                 nn.BatchNorm2d(self.in_channels),
                                 nn.ReLU(inplace=True))

        self.rt2 = nn.Sequential(nn.Conv2d(self.in_channels * 2, self.in_channels * 2, kernel_size=1, stride=1),
                                 nn.BatchNorm2d(self.in_channels * 2),
                                 nn.ReLU(inplace=True))

        self.rt3 = nn.Sequential(nn.Conv2d(self.in_channels * 4, self.in_channels * 4, kernel_size=1, stride=1),
                                 nn.BatchNorm2d(self.in_channels * 4),
                                 nn.ReLU(inplace=True))

        self.rt4 = nn.Sequential(nn.Conv2d(self.in_channels * 8, self.in_channels * 8, kernel_size=1, stride=1),
                                 nn.BatchNorm2d(self.in_channels * 8),
                                 nn.ReLU(inplace=True))

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


if __name__ == '__main__':
    config_vit = get_CTranS_config()
    model = Ours(config_vit, mode='train', deepsuper=True)
    model = model
    inputs = torch.rand(1, 1, 256, 256)
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')

