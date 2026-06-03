import torch
import torch.fft
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from utils import *
from .conformer import ConformerBlock
class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=2):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DilatedDenseNet(nn.Module):
    # orgin
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out



class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),      #实例归一化
            nn.PReLU(channels)                              #PReLU激活
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
    def forward(self, x):#
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)

        return x

class TSCB(nn.Module):
    def __init__(self, num_channel=64,drop=0.):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=drop, ff_dropout=drop)
        self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=drop, ff_dropout=drop)

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)
    def forward(self, x):

        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)
class ComplexDecoder(nn.Module):
     #Origin
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)

        return x
class TSCNet(nn.Module):
    def __init__(self,num_channel=64, num_features=201):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)
        self.TSCB_1 = TSCB(num_channel=num_channel,drop=0.2)
        self.TSCB_2 = TSCB(num_channel=num_channel,drop=0.2)
        self.TSCB_3 = TSCB(num_channel=num_channel,drop=0.2)
        self.TSCB_4 = TSCB(num_channel=num_channel,drop=0.2)
        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)
    def forward(self, x,*args):
        mag = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1) #切片操作分别选择了 x 张量中第二个维度（通常用于表示实部和虚部）中的第一个和第二个通道
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1) #(幅度和相位沿第二个维度拼接)
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        mask = self.mask_decoder(out_5)
        out_mag = mask * mag
        complex_out = self.complex_decoder(out_5)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag









import time
import torch
import argparse
import json
import os
from attrdict import AttrDict
from thop import profile
def main():
    model = TSCNet(num_channel=64).cuda()
    noisy_com = torch.randn(1, 2, 321, 201).cuda()
    print('Initializing Inference Process..')
    print('Warming up the model...')
    for i in range(10):
        with torch.no_grad():
            # model(input_data)
            model(noisy_com)
            # audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            # audio_g = audio_g / norm_factor

    print('Measuring inference speed...')
    total_time = 0
    num_iters = 100
    for i in range(num_iters):
        start_time = time.time()
        with torch.no_grad():
            model(noisy_com)

        end_time = time.time()
        total_time += end_time - start_time

    avg_time = total_time / num_iters
    print('Avg. Inference Time: {:.3f} seconds'.format(avg_time))
    noisy_com = torch.randn(1, 2, 321, 201).cuda()
    # Measure FLOPS
    flops, params = profile(model, inputs=(noisy_com,noisy_com))
    print('Number of FLOPs: {:.3f} GFLOPs'.format(flops / 1e9))
    print(f"Total Parameters: {params:,}")
    print(f"Total FLOPs: {flops:,}")
    max_memory = 0
    num_iters = 100
    for i in range(num_iters):
        torch.cuda.reset_max_memory_allocated()
        with torch.no_grad():
            model(noisy_com,noisy_com)
        max_memory = max(max_memory, torch.cuda.max_memory_allocated() / 1024**2)  # 转换为MB
    print('Data Length: {:.3f} seconds'.format(2))
    print('Avg. Inference Time: {:.3f} seconds'.format(avg_time))
    print('Number of FLOPs: {:.3f} GFLOPs'.format(flops / 1e9))
    print('Max Memory Usage: {:.2f} MB'.format(max_memory))
    print(f"Manual calculation of parameters: {params}")
if __name__ == '__main__':
    main()






