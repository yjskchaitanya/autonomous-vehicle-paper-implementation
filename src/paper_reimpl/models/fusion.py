from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth_rate: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth_rate * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.block(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_ch: int, growth_rate: int, layers: int) -> None:
        super().__init__()
        modules = []
        current = in_ch
        for _ in range(layers):
            modules.append(DenseLayer(current, growth_rate))
            current += growth_rate
        self.block = nn.Sequential(*modules)
        self.out_channels = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Transition(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseFusionNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3 + 3 + 3 + 6,
        growth_rate: int = 16,
        block_layers: list[int] | None = None,
        out_channels: int = 96,
    ) -> None:
        super().__init__()
        block_layers = block_layers or [4, 4, 4]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)

        current = 32
        blocks: list[nn.Module] = []
        for idx, layers in enumerate(block_layers):
            dense = DenseBlock(current, growth_rate, layers)
            current = dense.out_channels
            blocks.append(dense)
            if idx != len(block_layers) - 1:
                trans_out = max(current // 2, out_channels)
                blocks.append(Transition(current, trans_out))
                current = trans_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.BatchNorm2d(current),
            nn.ReLU(inplace=True),
            nn.Conv2d(current, out_channels, kernel_size=1),
        )

    def forward(
        self,
        camera: torch.Tensor,
        lidar: torch.Tensor,
        weather: torch.Tensor,
        segmentation_logits: torch.Tensor,
    ) -> torch.Tensor:
        if segmentation_logits.shape[-2:] != camera.shape[-2:]:
            segmentation_logits = F.interpolate(
                segmentation_logits,
                size=camera.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([camera, lidar, weather, segmentation_logits], dim=1)
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)
