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

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=64, max_distance=256, n_heads=2):
        super(RelativePositionBias, self).__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance):
        num_buckets //= 2
        ret = (relative_position < 0).to(relative_position) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(relative_position / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, relative_position):
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bias = self.relative_attention_bias(rp_bucket)
        return rp_bias
