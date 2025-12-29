import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .do_conv_pytorch import *


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=norm, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=norm, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class AttentionBranch(nn.Module):
    def __init__(self, nf, k_size=3):
        super(AttentionBranch, self).__init__()
        self.k1 = DOConv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = DOConv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = DOConv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class AAB(nn.Module):
    def __init__(self, nf, reduction=4, K=2, t=30):
        super(AAB, self).__init__()
        self.t = t
        self.K = K
        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.GELU(),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        self.esa = ESA(nf, nn.Conv2d)
        self.attention = AttentionBranch(nf)
        self.non_attention = DOConv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)

    def forward(self, x):
        residual = x
        a, b, c, d = x.shape
        x = self.conv_first(x)
        x = self.lrelu(x)
        y_channel = self.avg_pool(x).view(a, b)
        y_channel = self.ADM(y_channel)
        ax_channel = F.softmax(y_channel / self.t, dim=1)
        scale = self.esa(x)
        attention = self.attention(x)
        non_attention = self.non_attention(x)
        x = attention * ax_channel[:, 0].view(a, 1, 1, 1) + non_attention * ax_channel[:, 1].view(a, 1, 1, 1)
        x = x * scale
        x = self.lrelu(x)
        out = self.conv_last(x)
        out += residual

        return out


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = DOConv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = DOConv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = DOConv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = DOConv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return m