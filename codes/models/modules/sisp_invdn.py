#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
----------------------------
@ Author: JoefZhou         -
@ Home Page: www.zhoef.com -
@ From: tencent, UESTC     -
----------------------------
@ Date: 2021/7/3
@ Project Name InvDN
----------------------------
@ function:

@ Version:

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from scipy import linalg as la
import random

logabs = lambda x: torch.log(torch.abs(x))

class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class Mosaic_Operation(nn.Module):
    def __init__(self):
        super(Mosaic_Operation, self).__init__()
        self.PS_U = nn.PixelShuffle(2)

    def forward(self, input, rev=False):
        if not rev:
            red = input[:, 0, 0::2, 0::2]
            green_red = input[:, 1, 0::2, 1::2]
            green_blue = input[:, 1, 1::2, 0::2]
            blue = input[:, 2, 1::2, 1::2]
            output = torch.stack((red, green_red, green_blue, blue), dim=1)
            return output
        else:
            return self.PS_U(input)

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class InvSHBlock(nn.Module):
    def __init__(self, subnet_constructor, current_channel, channel_out, cal_jacobian=False):
        super(InvSHBlock, self).__init__()
        self.cal_jacobian=cal_jacobian
        self.subInvBlk1 = InvBlockExp(subnet_constructor, current_channel, channel_out)
        self.subInvBlk2 = InvBlockExp(subnet_constructor, current_channel*2, current_channel)

    def forward(self, inps):
        haarfs, packs, rev, jacobian = inps
        if not rev:
            haarfs = self.subInvBlk1(haarfs, rev)
            x = torch.cat([haarfs, packs], dim=1)
            x = self.subInvBlk2(x, rev)
            haarfs, packs = x.chunk(2, dim=1)
            if self.cal_jacobian:
                jacobian += self.subInvBlk1.jacobian(haarfs, rev)
                jacobian += self.subInvBlk2.jacobian(x, rev)
                return haarfs, packs, jacobian
            return haarfs, packs
        else:
            x = torch.cat([haarfs, packs], dim=1)
            x = self.subInvBlk2(x, rev)
            haarfs, packs = x.chunk(2, dim=1)
            haarfs = self.subInvBlk1(haarfs,rev)
            return haarfs, packs

class Noise_Model_Network(nn.Module):
    def __init__(self, channels=3, filters_num = 128, filters_pack = 4):
        super(Noise_Model_Network, self).__init__()

        # Noise Model Network

        self.conv_1 = nn.Conv2d(channels, filters_num, 1, 1, 0, groups=1)

        self.conv_2 = nn.Conv2d(filters_num, filters_num, 1, 1, 0, groups=1)

        self.conv_3 = nn.Conv2d(filters_num, filters_num, 1, 1, 0, groups=1)

        self.conv_4 = nn.Conv2d(filters_num, filters_num, 1, 1, 0, groups=1)

        self.conv_5 = nn.Conv2d(filters_num, filters_pack, 1, 1, 0, groups=1)
        self.rlu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x = self.rlu(self.conv_1(x))
        x = self.rlu(self.conv_2(x))
        x = self.rlu(self.conv_3(x))
        x = self.rlu(self.conv_4(x))
        x = self.rlu(self.conv_5(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class InvNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3,
                 subnet_constructor=None, block_num=[], down_num=2):
        super(InvNet, self).__init__()
        self.down_num = down_num
        self.block_num = block_num
        # operations = []
        self.blk_ops = nn.ModuleList()
        current_channel = channel_in
        self.squeezeF = SqueezeFunction()
        self.haar_downsample = nn.ModuleList()
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            self.haar_downsample.append(b)
            current_channel *= 4
            operations = nn.ModuleList()
            for j in range(block_num[i]):
                b = InvSHBlock(subnet_constructor, current_channel, channel_out)
                operations.append(b)
            # operations = nn.Sequential(*operations)
            self.blk_ops.append(operations)
        self.noise_pred = Noise_Model_Network(channels=current_channel, filters_pack=current_channel-channel_out)

    def forward(self, haarfs=None, x=None, rev=False, cal_jacobian=False, noise_trans=False):
        jacobian = 0
        if not rev:
            haarfs = x
            packs = x
            for d_idx in range(self.down_num):
                haarfs = self.haar_downsample[d_idx](haarfs, rev)
                packs = self.squeezeF(packs, rev)
                for blk_op in self.blk_ops[d_idx]:
                    if cal_jacobian:
                        haarfs, packs, logdet = blk_op((haarfs, packs, rev, 0))
                        jacobian += logdet
                    else:
                        haarfs, packs = blk_op((haarfs, packs, rev, 0))
            nleve = self.noise_pred(packs)
            haarfs[:, 3:, :, :] = haarfs[:, 3:, :, :] * nleve + haarfs[:, 3:, :, :]
        else:
            packs = x
            for d_idx in reversed(range(self.down_num)):
                for blk_op in reversed(self.blk_ops[d_idx]):
                    haarfs, packs = blk_op((haarfs, packs, rev, 0))
                haarfs = self.haar_downsample[d_idx](haarfs, rev=rev)
                packs = self.squeezeF(packs, rev=rev)

        if cal_jacobian:
            return haarfs, packs, jacobian
        else:
            return haarfs, packs


class SqueezeFunction(nn.Module):
    def __init__(self):
        super(SqueezeFunction, self).__init__()
    def forward(self, input, rev=False):
        b_size, n_channel, height, width = input.shape
        if not rev:
            squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
            squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            out = squeezed.contiguous().view(
                b_size, n_channel * 4, height // 2, width // 2)
            return out
        else:
            unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
            unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
            unsqueezed = unsqueezed.contiguous().view(
                b_size, n_channel // 4, height * 2, width * 2)
            return unsqueezed


def define_G(opt):
    from models.modules.Subnet_constructor import subnet
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    netG = InvNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num)

    return netG

if __name__ == '__main__':
    # m = Mosaic_Operation()
    # x = torch.rand(1,3,128,128)
    # y1 = m(x)
    # print(y1.shape)
    # y2 = m(y1, rev=True)
    # print(y2.shape)
    # torch.Size([1, 4, 64, 64])
    # torch.Size([1, 1, 128, 128])

    # f = Flow(12)
    # x = torch.rand(1, 3, 128, 128)
    # y1 = f(x)[0]
    # print(y1.shape)
    # y2 = f(y1, rev=True)
    # print(y2.shape)
    # torch.Size([1, 12, 128, 128])
    # torch.Size([1, 12, 128, 128])

    import argparse
    import options.options as option

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    model = define_G(opt)
    x = torch.rand(1, 3, 128, 128)
    # y1 = model(x)
    # print(y1.shape)
    haarfs, packs = model(x=x)
    cyncl, noisy = model(haarfs=haarfs, x=packs, rev=True, cal_jacobian= False)
    # from thop import profile
    #
    # flops, params = profile(model, inputs=(x,))
    # print(flops / (10 ** 9))
    # print(params / (10 ** 6))