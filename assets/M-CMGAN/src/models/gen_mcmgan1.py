import torch
import torch.fft
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from utils import *
# sys.path.append("/home/xyj/Experience/CMG-v1/src/models/")
# from mamba_ssm.modules.mamba_simple import Mamba as Mamba
from modules.mamba_simple import Mamba
from .conformer import (PreNorm, ConformerConvModule, MamformerBlock, ConformerBlock)



class CFB(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(CFB,self).__init__()
        self.conv_gate      = nn.Conv2d(in_channels=in_channels,  out_channels=out_channels, kernel_size=(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=True)
        self.conv_input     = nn.Conv2d(in_channels=in_channels,  out_channels=out_channels, kernel_size=(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=True)
        self.conv           = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,1), stride=1, padding=(1,0), dilation=1, groups=1, bias=True)
        self.ceps_unit  = CepsUnit(ch=out_channels)
        self.LN0     = LayerNorm( in_channels,f=101)
        self.LN1     = LayerNorm(out_channels,f=101)
        self.LN2     = LayerNorm(out_channels,f=101)
    def forward(self, x):
        g = torch.sigmoid(self.conv_gate(self.LN0(x)))
        x = self.conv_input(x)
        y = self.conv(self.LN1(g*x))
        y = y + self.ceps_unit(self.LN2((1-g)*x))
        return y


class CepsUnit(nn.Module):
    def __init__(self, ch):
        super(CepsUnit, self).__init__()
        self.ch = ch
        self.ch_lstm_f  = CH_LSTM_F(ch*2, ch,  ch*2)
        self.LN  = LayerNorm(ch*2,f=51)

    def forward(self, x0):
        x0 = torch.fft.rfft(x0, 101, 2)
        x = torch.cat([x0.real,x0.imag], 1)
        x = self.ch_lstm_f(self.LN(x))
        x = x[:,:self.ch] +1j*x[:,self.ch:]
        x = x*x0
        x = torch.fft.irfft(x, 101, 2)
        return x


class LayerNorm(nn.Module):
    def __init__(self, c, f):
        super(LayerNorm,self).__init__()
        self.w=nn.Parameter(torch.ones(1,c,f,1))
        self.b=nn.Parameter(torch.rand(1,c,f,1)*1e-4)
    def forward(self, x):
        mean = x.mean([1,2],keepdim=True)
        std  = x.std([1,2],keepdim=True)
        x = (x-mean)/(std+1e-8) *self.w +self.b
        return x
class CH_LSTM_F(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=True, num_layers=1):
        super().__init__()
        self.lstm2 = nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.linear= nn.Linear(2*feat_ch,out_ch)
        self.out_ch=out_ch

    def forward(self, x):
        self.lstm2.flatten_parameters()
        b,c,f,t = x.shape
        x = rearrange(x, 'b c f t -> (b t) f c')
        x,_  = self.lstm2(x.float())
        x = self.linear(x)
        x = rearrange(x, '(b t) f c -> b c f t', b=b, f=f, t=t)
        return x

class TimeRecursive_Mamba(nn.Module):
    def __init__(self, time_length , channel_num, dim_state, conv_width,expand):
        super(TimeRecursive_Mamba, self).__init__()
        self.MambaBlock1 = Mamba(
    #uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
    d_model=channel_num, # Model dimension d_model
    d_state=dim_state,  # SSM state expansion factor
    d_conv=conv_width,    # Local convolution width
    expand=expand,    # Block expansion factor
)
        self.MambaBlock2 = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=channel_num,  # Model dimension d_model
            d_state=2*dim_state,  # SSM state expansion factor
            d_conv=2*conv_width,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.act = nn.ReLU()
        self.post_norm = nn.LayerNorm(channel_num)
        self.drop = nn.Dropout(0.)
        self.t_len = time_length
        self.MambaBlock1 = PreNorm(channel_num,self.MambaBlock1)
        self.MambaBlock2 = PreNorm(channel_num, self.MambaBlock2)
        self.conv = ConformerConvModule(dim = channel_num, causal = False, expansion_factor = 2, kernel_size = 31, dropout = 0.)
    def forward(self, x_t0):
        skip1 = x_t0
        out_final = x_t0
        out_f = 0
        for i in range (self.t_len):
            out = self.MambaBlock1(skip1)
            out1= self.post_norm(self.drop(out)) + skip1
            out_ = torch.cat((out_final, out1), dim=1)
            out = self.MambaBlock2(out_)
            skip1 = self.post_norm(self.drop(out))+ out_
            skip1 = (skip1[:, skip1.shape[1] // 2:, :]*skip1[:, :skip1.shape[1] // 2, :])
            # skip1 = skip1[:, skip1.shape[1] // 2:, :]
            # out_final = skip1
            # out_final = self.post_norm(skip1)
            out_final = self.act(self.post_norm(skip1+out1))
            out_final = self.conv(out_final)+ out_final
            out_final = self.act(self.post_norm(out_final))
            out_f += out_final
            # out_f = self.post_norm(out_f)
        return out_f



class FiLM1C(nn.Module):        #沿通道仿射变换
    def __init__(self, input_size, feature_size):
        super(FiLM1C, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)   # 用于生成偏移参数的全连接层
      #  dil=1*2*4*8
        self.conv_1 = nn.Sequential(
            RefConv(3, 64, 3, (1, 1)),
            nn.InstanceNorm2d(64, affine=True),  # 实例归一化
            nn.PReLU(64)  # PReLU激活
        )
    def forward(self, input, context):
        #print(input.shape)
        context=self.conv_1(context).contiguous()
       # print(context.shape)
        context=context.view(-1,context.shape[1])
       # print(context.shape)
        gamma = self.gamma(context)  # 生成调制参数
     #   print(gamma.shape)
        beta = self.beta(context)    # 生成偏移参数
       # print(beta.shape)
        gamma=gamma.view(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
       # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM1F(nn.Module):    ##沿频率仿射变换
    def __init__(self, input_size, feature_size):
        super(FiLM1F, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)  # 用于生成偏移参数的全连接层
        #  dil=1*2*4*8
        self.conv_1 = nn.Sequential(
            RefConv(3, 64, 3, (1, 1)),
            nn.InstanceNorm2d(64, affine=True),  # 实例归一化
            nn.PReLU(64)  # PReLU激活
        )

    def forward(self, input, context):
        # print(input.shape)
        context = self.conv_1(context).contiguous()
       # print(context.shape)
        context = context.view(-1, context.shape[3])

        gamma = self.gamma(context)  # 生成调制参数
        # print(gamma.shape)
        beta = self.beta(context)  # 生成偏移参数
        # print(beta.shape)
        gamma = gamma.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM2C(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FiLM2C, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)   # 用于生成偏移参数的全连接层
      #  dil=1*2*4*8
        self.conv_2 = nn.Sequential(
            RefConv(32, 64, 3, (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
    def forward(self, input, context):
        #print(input.shape)
        context=self.conv_2(context).contiguous()
       # print(context.shape)
        context=context.view(-1,context.shape[1])
        gamma = self.gamma(context)  # 生成调制参数
       # print(gamma.shape)
        beta = self.beta(context)    # 生成偏移参数
       # print(beta.shape)
        gamma=gamma.view(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
       # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input

class FiLM2F(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FiLM2F, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)  # 用于生成偏移参数的全连接层
        #  dil=1*2*4*8
        self.conv_2 = nn.Sequential(
            RefConv(64, 64, 3, (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )

    def forward(self, input, context):
        # print(input.shape)
        context = self.conv_2(context).contiguous()
        # print(context.shape)
        context = context.view(-1, context.shape[3])
        gamma = self.gamma(context)  # 生成调制参数
        # print(gamma.shape)
        beta = self.beta(context)  # 生成偏移参数
        # print(beta.shape)
        gamma = gamma.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM11(nn.Module):        #沿通道仿射变换
    def __init__(self, f,input_size, feature_size):
        super(FiLM11, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)   # 用于生成偏移参数的全连接层
        self.f = f
      #  dil=1*2*4*8
      #   self.conv_1 = nn.Sequential(
      #       nn.Conv2d(3, 64, (1, 1), (1, 1)),
      #       nn.InstanceNorm2d(64, affine=True),  # 实例归一化
      #       nn.PReLU(64)  # PReLU激活
      #   )
    def forward(self, input):

        #print(input.shape)
        context = self.f(input)
        # context = input.clone().contiguous()
        # context=self.conv_1(context).contiguous()
       # print(context.shape)
        context=context.view(-1,context.shape[1])
       # print(context.shape)
        gamma = self.gamma(context)  # 生成调制参数
     #   print(gamma.shape)
        beta = self.beta(context)    # 生成偏移参数
       # print(beta.shape)
        gamma=gamma.view(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
       # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM12(nn.Module):    ##沿频率仿射变换
    def __init__(self, f, input_size, feature_size):
        super(FiLM12, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)  # 用于生成偏移参数的全连接层
        self.f = f

    #  dil=1*2*4*8
    #   self.conv_1 = nn.Sequential(
    #       nn.Conv2d(3, 64, (1, 1), (1, 1)),
    #       nn.InstanceNorm2d(64, affine=True),  # 实例归一化
    #       nn.PReLU(64)  # PReLU激活
    #   )
    def forward(self, input):
        # print(input.shape)
        context = self.f(input)
       # print(context.shape)
        context = context.view(-1, context.shape[3])
        gamma = self.gamma(context)  # 生成调制参数
        # print(gamma.shape)
        beta = self.beta(context)  # 生成偏移参数
        # print(beta.shape)
        gamma = gamma.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM21(nn.Module):
    def __init__(self, f,input_size, feature_size):
        super(FiLM21, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)   # 用于生成偏移参数的全连接层
        self.f = f
    #  dil=1*2*4*8
    #   self.conv_1 = nn.Sequential(
    #       nn.Conv2d(3, 64, (1, 1), (1, 1)),
    #       nn.InstanceNorm2d(64, affine=True),  # 实例归一化
    #       nn.PReLU(64)  # PReLU激活
    #   )


    def forward(self, input):
    # print(input.shape)
        context = self.f(input)
       # print(context.shape)
        context=context.view(-1,context.shape[1])
        gamma = self.gamma(context)  # 生成调制参数
       # print(gamma.shape)
        beta = self.beta(context)    # 生成偏移参数
       # print(beta.shape)
    #     gamma = gamma.expand_as(input)
    # # print(gamma)
    #     beta = beta.expand_as(input)
        gamma=gamma.view(input.shape[0],input.shape[1],input.shape[2],-1)
       # # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], -1)
    #     gamma = gamma.expand_as(input)
    # # print(gamma)
    #     beta = beta.expand_as(input)
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input

class FiLM22(nn.Module):
    def __init__(self,f, input_size, feature_size):
        super(FiLM22, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)  # 用于生成偏移参数的全连接层
        self.f = f

    #  dil=1*2*4*8
    #   self.conv_1 = nn.Sequential(
    #       nn.Conv2d(3, 64, (1, 1), (1, 1)),
    #       nn.InstanceNorm2d(64, affine=True),  # 实例归一化
    #       nn.PReLU(64)  # PReLU激活
    #   )
    def forward(self, input):
        # print(input.shape)
        context = self.f(input)
        # print(context.shape)
        context = context.view(-1, context.shape[3])

        gamma = self.gamma(context)  # 生成调制参数
        # print(gamma.shape)
        beta = self.beta(context)  # 生成偏移参数
        # print(beta.shape)
        gamma = gamma.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class GlobleCModule(nn.Module):
    def __init__(self,channels):
        super(GlobleCModule, self).__init__()
        self.ws=nn.Linear(channels,channels)
        self.wg=nn.Linear(channels,channels)
        self.pool = nn.Sequential(
                                  Rearrange('f t c -> c t f '),
                                  nn.AdaptiveAvgPool1d(1),
                                  Rearrange('c t f -> f c t '),
                                  nn.AdaptiveAvgPool1d(1),
                                  Rearrange('f c t -> f t c'),
                                  )
        self.act= LearnableSigmoid(channels)
    def forward(self, x):
        input_p=self.pool(x)
        ws=self.ws(input_p)     # input_p.shape = 1，1,fea_num
        wg=self.wg(input_p)
        input_ws = ws * input_p
        input_wg = wg * input_p
        # input_wsg = wg * input_ws
        a= x
        b= (self.act(input_ws) + input_wg)
        moduled = torch.mul(a, b.expand_as(a))
        return moduled
class Tmodule(nn.Module):
    def __init__(self,feature):
        super(Tmodule, self).__init__()
        self.fea_num = feature
        self.ws=nn.Linear(self.fea_num,self.fea_num)
        self.wg=nn.Linear(self.fea_num,self.fea_num)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Rearrange('f t c -> c t f '),
                                  nn.AdaptiveAvgPool1d(1),
                                  Rearrange('c t f -> f c t '),
                                  )
        # self.act= LearnableSigmoid(self.fea_num)
    def forward(self, x):
        # if self.fea_num != x.shape[1]:
        #     self.fea_num = x.shape[1]
        #     self.ws = nn.Linear(self.fea_num, self.fea_num).to(x.device)
        #     self.wg = nn.Linear(self.fea_num, self.fea_num).to(x.device)
        #     self.act = LearnableSigmoid(self.fea_num).to(x.device)
        input_p=self.pool(x)
        # print(x.shape)# = 321,101,64
        # print(input_p.shape)    #1,1,101
        ws=self.ws(input_p)     # input_p.shape = 1，1,fea_num
        wg=self.wg(input_p)
        input_ws = ws * input_p
        input_wsg = wg * input_ws
        a= x.permute(0,2,1)
        b= (self.act(input_ws) + input_wsg)
        moduled = torch.mul(a, b.expand_as(a))
        return moduled
class Fmodule(nn.Module):
    def __init__(self, fea_num):
        super(Fmodule, self).__init__()
        self.ws=nn.Linear(fea_num,fea_num)
        self.wg=nn.Linear(fea_num,fea_num)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Rearrange('b f c -> c f b'),
                                  nn.AdaptiveAvgPool1d(1),
                                  Rearrange('c f b -> b c f '),
                                  )
        self.act= LearnableSigmoid(101)
    def forward(self, x):
        input_p=self.pool(x)
        # print(x.shape)# = 321,101,64
        # print(input_p.shape)    #1,1,101
        ws=self.ws(input_p)     # input_p.shape = 1，1,fea_num
        wg=self.wg(input_p)
        input_ws = ws * input_p
        input_wg = wg * input_p
        # input_wsg = wg * input_ws
        a= x.permute(0,2,1)
        b= (self.act(input_ws) + input_wg)
        moduled = torch.mul(a, b.expand_as(a))
        return moduled
class TFmodule(nn.Module):
    def __init__(self, T_dim=321,F_dim=101):
        super(TFmodule, self).__init__()
        self.wst=nn.Linear(T_dim,T_dim)
        self.wgt=nn.Linear(T_dim,T_dim)
        self.tpool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Rearrange('f t c -> c t f '),
                                  nn.AdaptiveAvgPool1d(1),
                                  Rearrange('c t f -> f c t '),
                                  )
        self.T_act= nn.Sigmoid()

        self.wsf=nn.Linear(F_dim,F_dim)
        self.wgf=nn.Linear(F_dim,F_dim)
        self.fpool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Rearrange('b f c -> c f b'),
                                  nn.AdaptiveAvgPool1d(1),
                                  Rearrange('c f b -> b c f '),
                                  )
        self.F_act= nn.Sigmoid()
    def forward(self, x,condition=None):
        if condition == True:
            input_p=self.tpool(x)
        # x.shape = 101,64,321=b*f, t, c
        # print(input_p.shape)#1,1,321
            wst=self.wst(input_p)     # input_p.shape = 1，1,fea_num
            wgt=self.wgt(input_p)
            input_ws = wst * input_p
        # print(input_ws.shape)
            input_wsg = wgt * input_ws
            a= x.permute(0,2,1)
            b= (self.T_act(input_ws) + input_wsg)
            moduled = torch.mul(a, b.expand_as(a))
            return moduled
        else:
            input_p = self.fpool(x)
            # print(x.shape)# = 321,101,64
            # print(input_p.shape)    #1,1,101
            wsf = self.wsf(input_p)  # input_p.shape = 1，1,fea_num
            wgf = self.wgf(input_p)
            input_ws = wsf * input_p
            input_wsg = wgf * input_ws
            a = x.permute(0, 2, 1)
            b = (self.F_act(input_ws) + input_wsg)
            moduled = torch.mul(a, b.expand_as(a))
            return moduled



class DilatedDenseNet(nn.Module):
    #orgin
    # def __init__(self, depth=4, in_channels=64):
    #     super(DilatedDenseNet, self).__init__()
    #     self.depth = depth
    #     self.in_channels = in_channels
    #     self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
    #     self.twidth = 2
    #     self.kernel_size = (self.twidth, 3)
    #     for i in range(self.depth):
    #         dil = 2 ** i
    #         pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
    #         setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
    #         setattr(self, 'conv{}'.format(i + 1),
    #                 nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
    #                           dilation=(dil, 1)))
    #         setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
    #         setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))
    #
    # def forward(self, x):
    #     skip = x
    #     for i in range(self.depth):
    #         out = getattr(self, 'pad{}'.format(i + 1))(skip)
    #         out = getattr(self, 'conv{}'.format(i + 1))(out)
    #         out = getattr(self, 'norm{}'.format(i + 1))(out)
    #         out = getattr(self, 'prelu{}'.format(i + 1))(out)
    #         skip = torch.cat([out, skip], dim=1)
    #    out = out[:, :, :-1, :]         #Casual
    #     return out
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
                              dilation=(dil, 1)),)
            # setattr(self, 'attn{}'.format(i + 1), MultiDilatelocalAttention(in_channels))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            # out = getattr(self, 'attn{}'.format(i + 1))(out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)

            skip = torch.cat([out, skip], dim=1)

        # out = out_a + out
    #    out = out[:, :, :-1, :]         #Casual
        return out

class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),      #实例归一化
            nn.PReLU(channels)                              #PReLU激活
        )
        # self.FiLM11 = FiLM11(self.conv_1,channels,channels)
        # self.FiLM12 = FiLM12(self.conv_1,201,201)
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
        # self.FiLM21 = FiLM21(self.conv_2,channels,channels)
        # self.FiLM22 = FiLM22(self.conv_2,101,101)
    def forward(self, x):#
        x = self.conv_1(x)
        # c1 = x.clone()
        # x = self.FiLM11(x) #+ x
        # x = self.FiLM12(x) #+ x
        # x = x + c1
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        # c2 = x.clone()
        # x = self.FiLM21(x) #+ x
        # x = self.FiLM22(x) #+ x
        # x = x + c2

        return x

# #

class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        # self.T_modulelayer = Tmodule()
        self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.F_modulelayer = Fmodule(101)
        self.norm = nn.LayerNorm(num_channel)

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        #add Tmodule
        # x_tm = self.T_modulelayer(x_t).permute(0,2,1)
        # x_t = x_t + x_tm
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        # add Fmodule

        x_fn = self.norm(x_f)
        x_fm = self.F_modulelayer(x_fn).permute(0,2,1)
        x_f = x_fm +x_f
        x_f = torch.relu(x_f)
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)#+x_fm.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f

class TMFC(nn.Module):
    def __init__(self, num_channel=64):
        super(TMFC, self).__init__()
        # self.CFB1 = CFTSA(dropout=0.2)
        # self.CFB1 = CFB(num_channel,(num_channel//2),201)
        # self.t_recur_mamba = ConMambaBlock(dim=num_channel, dim_head=num_channel // 4, heads=4,
        #                                      conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        # self.t_recur_mamba = Time_Mamba_Inter(1,num_channel,64,4,2)
        self.t_recur_mamba = TimeRecursive_Mamba(1,num_channel,16,4,4)
        # self.freq_conformer = ConMetBlock(dim=num_channel,ffn_expansion_factor=2.66, num_heads=8, bias=False,
        #                                           LayerNorm_type='BiasFree',qk_norm=1)
        self.freq_conformer = MamformerBlock(dim=num_channel, dim_head=num_channel //4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.norm = nn.LayerNorm(num_channel)
        self.norm1 = nn.LayerNorm(num_channel)
        self.act = nn.ReLU()
        self.C_modulelayer = GlobleCModule(num_channel)
        self.F_modulelayer = Fmodule(101)

    def forward(self, x_in):
        # x_in = self.CFB1(x_in) +x_in
        # x_in = self.CFB1(x_in) + x_in
        b, c, t, f = x_in.size()
        # print(x_in)  #2-Step TMamba
        x_t0 = x_in.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        # x_t = self.time_conformer(x_t) + x_t
        x_t = self.t_recur_mamba(x_t0) + x_t0
        x_cn = self.norm(x_t)
        x_cm = self.C_modulelayer(x_cn)
        x_t = x_cm + x_t
        # x_t = self.norm(x_cm)+x_t
        x_t = self.act(x_t)
        # print(x_t.shape)
        # x_tn = self.norm(x_t)
        # x_tm = self.T_modulelayer(x_tn).permute(0,2,1)
        # x_t = x_t +x_tm
        # x_t = self.norm(x_t)
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        # x_f = x_t.view(b, f, t, c).permute(0, 3, 2, 1).contiguous()
        #Freq Block:F2Fmamba or FeqComformer?
        x_f = (self.freq_conformer(x_f) + x_f)
        # x_f = (self.freq_conformer(x_f) + x_f).view(b*t, f, c)
        x_fn = self.norm1(x_f)
        x_fm = self.F_modulelayer(x_fn).permute(0,2,1)
        x_f = x_fm + x_f
        # x_f = self.norm(x_fm) + x_f
        x_f = self.act(x_f)
        # # add Fmodule

        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2) #+x_fm.view(b, t, f, c).permute(0, 3, 1, 2)

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
        # self.prelu_out = nn.PReLU(num_features, init=-0.25)
        self.lsig = LearnableSigmoid_2d(num_features)
    def forward(self, x):

        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.lsig(x).permute(0, 2, 1).unsqueeze(1)
        # return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)
class ComplexDecoder(nn.Module):
     #Origin
    # def __init__(self, num_channel=64):
    #     super(ComplexDecoder, self).__init__()
    #     self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
    #     self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
    #     self.prelu = nn.PReLU(num_channel)
    #     self.norm = nn.InstanceNorm2d(num_channel, affine=True)
    #     self.conv = nn.Conv2d(num_channel, 2, (1, 2))
    #
    # def forward(self, x):
    #     x = self.dense_block(x)
    #     x = self.sub_pixel(x)
    #     x = self.prelu(self.norm(x))
    #     x = self.conv(x)
    #
    #     return x

    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2),)
        # self.Ref_conv = RefConv(2, 2, 3,(1,1))
    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)

        return x

class TSCNet(nn.Module):
    def __init__(self,num_channel=32, num_features=201):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)
        # self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TMFC_1 = TMFC(num_channel=num_channel)
        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)
        self.cfb_e1 = CFB(num_channel, num_channel)
        # self.cfb_e5 = CFB(num_channel, num_channel)
        # self.w = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.ln = LayerNorm(num_channel, 101)
        self.cfb_d5 = CFB(1 * num_channel, num_channel)
    # def forward(self, x, *args, **kwargs):  # 接受任意额外参数
    def forward(self, x):  # 接受任意额外参数
        # print("Model input shape:", x.shape)
        mag = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1) #切片操作分别选择了 x 张量中第二个维度（通常用于表示实部和虚部）中的第一个和第二个通道
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1) #(幅度和相位沿第二个维度拼接)
        # x_in = F.pad(x_in,[0,0,2,0])
        out_1 = self.dense_encoder(x_in).permute(0,1,3,2)
        e1 = self.cfb_e1(out_1)
        # e1 = self.inter0(out_1,e1)
        out_2 = self.TMFC_1(e1.permute(0,1,3,2))
        # out_3=(self.inter(e1.permute(0,1,3,2),out_2))
        out_3 = (self.TSCB_1(out_2)).permute(0,1,3,2)
        # out_3 = (self.inter1(out_3, out_4) ).permute(0,1,3,2)
        d5 = self.cfb_d5(torch.cat([e1*out_3], dim=1)).permute(0,1,3,2)

        # out_4 = self.TSCB_3(out_3)
        # out_5 = self.TSCB_4(out_4)
        # d5 = self.FiLM2(d5)
        mask = self.mask_decoder(d5)
        out_mag = mask * mag
        complex_out = self.complex_decoder(d5)
        # complex_out = complex_out[:,:,2:,:]
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag

import time
import torch

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












