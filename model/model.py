"""
CREaTor
Copyright (C) 2023  Microsoft Research

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel
from .attention import TransformerEncoder
from .positional_encoding import RelativePositionBias


class WConv(nn.Module):
    def __init__(self, **kwargs):
        super(WConv, self).__init__()
        self.conv = nn.Conv1d(**kwargs)

    def forward(self, x):
        shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        x = x.transpose(-1, -2)
        x = self.conv(x)
        x = x.transpose(-1, -2)
        x = x.reshape(*shape, *x.shape[-2:])
        return x


class SRBias(nn.Module):
    def __init__(self, odim):
        super(SRBias, self).__init__()
        self.r = [150, 600, 2400, 10000, 40000]
        self.rp_bias = nn.ModuleList(
            [
                RelativePositionBias(num_buckets=64, max_distance=256, n_heads=odim)
                for _ in self.r
            ]
        )

    def forward(self, pos):
        ret = []
        for r, fn in zip(self.r, self.rp_bias):
            bias = fn(pos.div(r, rounding_mode="floor"))
            bias = bias.permute(2, 0, 1)
            ret.append(bias)
        return torch.sum(torch.stack(ret), dim=0)


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.d_model_1 = 256
        self.n_layer_1 = 2
        self.n_head_1 = self.d_model_1 // 64

        self.d_model_2 = 256
        self.n_layer_2 = 4
        self.n_head_2 = self.d_model_1 // 64

        self.embed_genome = nn.Sequential(
            nn.Embedding(6, self.d_model_1, padding_idx=0), nn.Dropout(0.1)
        )
        self.embed_chipseq = nn.Sequential(
            nn.Linear(16, self.d_model_1), nn.Dropout(0.1)
        )
        self.projection = WConv(
            in_channels=self.d_model_1,
            out_channels=self.d_model_1,
            kernel_size=5,
            stride=5,
        )

        self.rp_bias_1 = RelativePositionBias(
            num_buckets=64, max_distance=256, n_heads=self.n_head_1
        )
        self.encoder1 = TransformerEncoder(
            n_layer=self.n_layer_1,
            d_model=self.d_model_1,
            n_head=self.n_head_1,
            dim_feedforward=self.d_model_1 * 4,
            dropout=0.1,
        )
        self.rp_bias_2 = SRBias(self.n_head_2)

        self.encoder2 = TransformerEncoder(
            n_layer=self.n_layer_2,
            d_model=self.d_model_2,
            n_head=self.n_head_2,
            dim_feedforward=self.d_model_2 * 4,
            dropout=0.1,
        )

        self.layer_norm_after = nn.LayerNorm(self.d_model_2)
        self.fc = nn.Sequential(nn.Linear(self.d_model_2, 1))
        self.act = nn.Softplus()

    def forward(self, data):
        E_ge = F.pad(data["E_ge"], (0, 5), mode="constant", value=5)
        G_ge = F.pad(torch.zeros(700).long(), (0, 5), mode="constant", value=5)

        E_chp = F.pad(data["E_chp"], (0, 0, 0, 5), mode="constant", value=0.0)
        G_chp = F.pad(
            torch.zeros((1, 700, 16)), (0, 0, 0, 5), mode="constant", value=0.0
        )

        def stage1(genome, chipseq, pos):
            x_gene = torch.eye(6)[genome].to(genome.device)
            x = torch.matmul(x_gene, self.embed_genome[0].weight)
            # x_gene = self.embed_genome(genome)
            x_chip = self.embed_chipseq(chipseq)
            x = x + x_chip
            x = self.projection(x)
            pos = pos[..., ::5]
            pos = (pos.unsqueeze(-1) - pos.unsqueeze(-2)).div(5, rounding_mode="floor")
            bias = self.rp_bias_1(pos)
            bias = bias.transpose(-1, -3)
            bias = bias.transpose(-1, -2)
            x, _ = self.encoder1(x, bias)
            x = x[:, -1]
            return x, x_gene, chipseq

        e_pos = torch.arange(E_ge.shape[-1], device=E_ge.device)
        xE, x_gene, x_chipseq = stage1(E_ge, E_chp, e_pos)
        gx = data["G_pos"] + torch.arange(700).unsqueeze(0)

        g_pos = torch.cat([data["G_gpos"], data["G_pos"][:, None]], dim=-1)
        xG, _, _ = stage1(G_ge, G_chp, g_pos)

        # stage2
        pos = torch.cat([data["E_pos"], data["G_pos"]], dim=-1)
        pos = pos.unsqueeze(1) - pos.unsqueeze(0)

        bias = self.rp_bias_2(pos)

        x = torch.cat([xE, xG], dim=0)
        ne, ng = xE.shape[0], xG.shape[0]

        msk_bias = torch.cat(
            [torch.zeros((ne + ng, ne)), torch.ones((ne + ng, ng))], dim=1
        )
        msk_bias = msk_bias.to(x.device) * -1000000

        x, attn_weight = self.encoder2(x, bias + msk_bias)

        x = self.layer_norm_after(x)
        x = x[-ng:]
        x = self.fc(x)
        x = self.act(x)
        return {"expression": x}
