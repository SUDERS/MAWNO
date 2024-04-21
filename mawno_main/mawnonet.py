import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import torch.fft
from pytorch_wavelets import DWT1D, IDWT1D
from sa_ode import SelfAttention


class WaveConv1d(nn.Module):
    def __init__(self, config):
        super(WaveConv1d, self).__init__()
        self.config = config
        self.dummy = self.config.x.permute(0, 2, 1)
        self.in_channels = self.config.embed_dim
        self.out_channels = self.config.embed_dim
        self.level = self.config.level
        self.dwt_ = DWT1D(wave=self.config.wave, J=self.level, mode=self.config.mode).to(self.config.device)
        self.mode_data, self.coe_data = self.dwt_(self.dummy)
        self.modes1 = self.mode_data.shape[-1]
        self.dwt1d = DWT1D(wave=self.config.wave, J=self.level, mode=self.config.mode)
        self.idwt1d = IDWT1D(wave=self.config.wave, mode=self.config.mode)

        self.sa_c = SelfAttention(dim=self.config.embed_dim, heads=1)

        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1))


    def mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft, x_coeff = self.dwt1d(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1])
        out_ft = self.mul1d(x_ft, self.weights1)

        x_coeff[-1] = x_coeff[-1].permute(0, 2, 1)
        x_coeff[-1] = self.sa_c(x_coeff[-1])
        x_coeff[-1] = F.gelu(x_coeff[-1])
        x_coeff[-1] = x_coeff[-1].permute(0, 2, 1)

        x_coeff[-1] = self.mul1d(x_coeff[-1], self.weights2)

        x = self.idwt1d((out_ft, x_coeff))
        return x


class Block(nn.Module):
    def __init__(self, config, dim):
        super(Block, self).__init__()
        self.config = config

        self.filter = WaveConv1d(self.config)

        self.conv = nn.Conv1d(dim, dim, 1)

    def forward(self, x):

        x1 = self.filter(x)
        x2 = self.conv(x)
        x = x1 + x2
        x = F.gelu(x)


        return x


class MAWNONet(nn.Module):
    def __init__(self, config, embed_dim=128, depth=4):
        super(MAWNONet, self).__init__()

        self.config = config
        self.setup_seed(self.config.seed)

        self.fc0 = nn.Linear(self.config.prob_dim, embed_dim)

        self.blocks = nn.ModuleList([
            Block(config, dim=embed_dim)
            for _ in range(depth)])

        self.fc1 = nn.Linear(embed_dim, self.config.fc_map_dim)
        self.fc2 = nn.Linear(self.config.fc_map_dim, self.config.prob_dim)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, x):
        B = x.shape[0]
        x = self.fc0(x)

        x = x.permute(0, 2, 1)

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
