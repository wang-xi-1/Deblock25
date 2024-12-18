import torch
import torch.fft
import torch.nn as nn
import torch_dct as dct
import torch.nn.functional as F
from torchvision import transforms

import os
import math
import random
import numbers
import argparse
import numpy as np
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
from pdb import set_trace as stx
from einops import rearrange, repeat
from typing import Optional, Callable
from skimage.metrics import peak_signal_noise_ratio as psnr
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from collections import Counter
# The Code Implementatio of MambaIR model for Real Image Denoising task
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import warnings  
warnings.filterwarnings("ignore", category=UserWarning)


def feature_save1(tensor, name):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    # tensor = torch.mean(tensor,dim=1).repeat(3,1,1)
    if not os.path.exists(str(name)):
        os.makedirs(str(name))
    for i in range(tensor.shape[1]):
        inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
        inp = np.clip(np.abs(inp),0,1)
        inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
        inp = np.squeeze(inp)
        plt.figure()
        plt.imshow(inp)
        plt.savefig(str(name) + '/' + str(i) + '.png')
        
        
class SS2D_DCT(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),    #########
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,     #########
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=4, D, N)   #########
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=4, D, N)     #########

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def apply_dct_batch(self, x):  
        """  
        Apply DCT to a 4D tensor (B, C, H, W) using batch processing  
        """  
        B, C, H, W = x.shape  
        x = x.reshape(-1, H, W)  

        # Apply DCT to each 2D slice  
        x_dct = dct.dct_2d(x, norm='ortho')  

        # Generate the dynamic frequency basis  
        u = torch.arange(H).reshape(-1, 1).float().to(x.device)  
        v = torch.arange(W).reshape(1, -1).float().to(x.device)  
        omega_u = u / H  
        omega_v = v / W  
        basis_normalized = torch.sqrt(omega_u ** 2 + omega_v ** 2)  
        basis_normalized[0, 0] = 1  # To avoid division by zero, set the DC component to 1.  
        # Normalize the DCT coefficients  
        x_dct = x_dct / basis_normalized  

        x_dct = x_dct.view(B, C, H, W)  # Reshape back to 4D tensor  

        return x_dct.contiguous(), basis_normalized  

    def apply_idct_batch(self, x, basis_normalized):  
        """  
        Apply IDCT (Inverse Discrete Cosine Transform) to a 4D tensor (B, C, H, W)  
        """  
        B, C, H, W = x.shape  
        x = x.reshape(-1, H, W)  # Flatten the tensor to 3D  

        # Denormalize the DCT coefficients  
        x = x * basis_normalized  

        # Apply IDCT to each 2D slice  
        x_idct = dct.idct_2d(x, norm='ortho')  
        x_idct = x_idct.view(B, C, H, W)  

        return x_idct.contiguous()

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 2

        x_dct, dct_norm = self.apply_dct_batch(x) 
        xs = torch.stack([x_dct.view(B, -1, L), torch.transpose(x_dct, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, K, -1, L)
        
        P = 1
        K = K * P
        L = L // P

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        hw_y = self.apply_idct_batch(out_y[:, 0].view(B, -1, H, W), dct_norm).view(B, -1, L)
        wh_y = self.apply_idct_batch(torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous(), dct_norm).view(B, -1, L)

        return hw_y, wh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2                          # B, C, L
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        self.transpose= transpose
        if self.transpose:
            padding = kernel_size // 2 -1
            self.layer = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
        else:
            self.layer = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)

        self.relu= relu
        if self.relu:
            self.act = nn.GELU()

    def forward(self, x):
        if self.relu:
            # if self.transpose:
            #     return self.act(self.layer(x))
            # else:
            return self.act(self.layer(x))
        else:
            # if self.transpose:
            #     return self.layer(x)
            # else:
            return self.layer(x)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = BasicConv(in_channel=Cin, out_channel=G, kernel_size=kSize, stride=1, relu=True)

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDBlock(nn.Module):
    def __init__(self, in_channel, out_channel, nConvLayers=3):
        super(RDBlock, self).__init__()
        G0 = in_channel
        G = in_channel
        C = nConvLayers

        self.conv0 = RDB_Conv(G0 , G)
        self.conv1 = RDB_Conv(G0 + 1 * G , G)
        self.conv2 = RDB_Conv(G0 + 2 * G , G)

        self.LFF = BasicConv(in_channel=G0 + C * G, out_channel=out_channel, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.LFF(out) + x
        return out


class MCBlock(nn.Module):
    def __init__(self, channel, d_state=16):
        super(MCBlock, self).__init__()
        
        self.ss2d_dct = SS2D_DCT(d_model=channel, d_state=d_state, expand=2, dropout=0)
        self.rdblock = nn.Sequential(*[RDBlock(channel, channel) for _ in range(3)])
        self.out = nn.Sequential(*[RDBlock(channel, channel) for _ in range(1)])
        
        self.scale3 = nn.Parameter(torch.ones(1, channel, 1, 1))
        self.scale4 = nn.Parameter(torch.ones(1, channel, 1, 1))
        
    def forward(self, x):
        
        x_ = x.permute(0, 2, 3, 1).contiguous()
        x1 = self.ss2d_dct(x_).permute(0, 3, 1, 2).contiguous()
        x2 = self.rdblock(x)
        
        x1 = x1 * self.scale3
        x2 = x2 * self.scale4
        
        out = x1 + x2
        out = self.out(out)
        
        return out


class EBlock(nn.Module):
    def __init__(self, channel, d_state=16):
        super(EBlock, self).__init__()
        self.layer1 = MCBlock(channel, d_state)
        self.layer2 = MCBlock(channel, d_state)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class DBlock(nn.Module):
    def __init__(self, channel, d_state=16):
        super(DBlock, self).__init__()
        self.layer1 = MCBlock(channel, d_state)
        self.layer2 = MCBlock(channel, d_state)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()

        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class SCM(nn.Module):
    def __init__(self, in_c, out_plane):
        super(SCM, self).__init__()
        self.layer1 = BasicConv(in_c, out_plane//4, kernel_size=3, stride=1, relu=True)
        self.layer2 = BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True)
        self.layer3 = BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True)
        self.layer4 = BasicConv(out_plane // 2, out_plane-in_c, kernel_size=1, stride=1, relu=True)
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        x = torch.cat([x, out], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out
    

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=4, ct_channels = 1):
        super().__init__()
        if num_scales==4:
            scales = (4, 8, 16, 32)
        elif num_scales==3:
            scales = (4, 8, 16)

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        #
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        #
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = []
        # for stage in self.stages:
        #     temp_fea = F.interpolate(input=stage(feats), size=(h, w), mode='nearest')
        #     print('temp_fea.shape:',temp_fea.shape)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class DeblockNet(nn.Module):
    def __init__(self, args, in_c, out_c, base_channel=24, num_res=6):
        super(DeblockNet, self).__init__()

        base_channel = base_channel
        self.args = args

        self.Encoder_feat_extract = nn.ModuleList([
            BasicConv(in_c, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2)
        ])

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, d_state=64),
            EBlock(base_channel * 2, d_state=32),
            EBlock(base_channel * 4, d_state=16)
        ])

        self.FAM2 = FAM(base_channel * 2)
        self.FAM1 = FAM(base_channel * 4)

        self.SCM2 = SCM(in_c, base_channel * 2)
        self.SCM1 = SCM(in_c, base_channel * 4)

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel),
            AFF(base_channel * 7, base_channel * 2),
            AFF(base_channel * 7, base_channel * 4)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, d_state=16),
            DBlock(base_channel * 2, d_state=32),
            DBlock(base_channel, d_state=64)
        ])

        self.Decoder_feat_extract = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, out_c, kernel_size=3, relu=False, stride=1)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)
        ])
        
        self.pyramid = PyramidPooling(base_channel, base_channel)

    def forward(self, x):

        input = x
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        x_ = self.Encoder_feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.Encoder_feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.Encoder_feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z14 = F.interpolate(res1, scale_factor=0.25)
        z21 = F.interpolate(res2, scale_factor=2)
        z24 = F.interpolate(res2, scale_factor=0.5)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        z = self.AFFs[2](z14, z24, z)
        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z = self.Decoder_feat_extract[0](z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.Decoder_feat_extract[1](z)  #

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.pyramid(z)
        z = self.Decoder_feat_extract[2](z)

        if self.args.globel_res == 1:
            return z[..., :h, :w] + input
        else:
            return z[..., :h, :w]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--globel_res', type=int, default=1)
    args = parser.parse_args()

    model =  DeblockNet(args, 1, 1, base_channel=28, num_res=8).cuda()
    input = torch.randn(1, 1, 128, 128).cuda()
    out = model(input)
    print(out.shape)



