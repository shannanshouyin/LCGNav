from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """
    A lightweight PointNet-style global encoder.

    Input:
        points: (B, N, in_dim)
    Output:
        global_feat: (B, global_dim)
    """

    def __init__(
        self,
        in_dim: int,
        mlp_channels: List[int],
        global_dim: int,
        dropout: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super().__init__()

        assert len(mlp_channels) >= 1, "mlp_channels must have at least 1 layer"
        self.in_dim = int(in_dim)
        self.mlp_channels = [int(x) for x in mlp_channels]
        self.global_dim = int(global_dim)

        layers = []
        last_c = self.in_dim
        for c in self.mlp_channels:
            layers.append(nn.Conv1d(last_c, c, kernel_size=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            last_c = c
        self.shared_mlp = nn.Sequential(*layers)

        if last_c != self.global_dim:
            self.fc = nn.Linear(last_c, self.global_dim)
        else:
            self.fc = nn.Identity()

        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        assert points.ndim == 3, f"points must be (B,N,in_dim), got {points.shape}"
        x = points.transpose(1, 2).contiguous()
        x = self.shared_mlp(x)
        x = torch.max(x, dim=2).values
        x = self.fc(x)
        x = self.dropout(x)
        return x
