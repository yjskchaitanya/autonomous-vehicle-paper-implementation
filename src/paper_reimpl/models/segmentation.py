from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        branch_ch = out_ch // 4
        self.branch1 = ConvBNReLU(in_ch, branch_ch, 1)
        self.branch3 = nn.Sequential(ConvBNReLU(in_ch, branch_ch, 1), ConvBNReLU(branch_ch, branch_ch, 3))
        self.branch5 = nn.Sequential(ConvBNReLU(in_ch, branch_ch, 1), ConvBNReLU(branch_ch, branch_ch, 5))
        self.pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), ConvBNReLU(in_ch, branch_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch3(x), self.branch5(x), self.pool(x)],
            dim=1,
        )


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = InceptionBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.pool(features)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = InceptionBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MultiOrientationSegmenter(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32, num_classes: int = 6) -> None:
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = InceptionBlock(base_channels * 4, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)

    def forward(self, x: torch.Tensor, orientations: list[float] | None = None) -> torch.Tensor:
        if not orientations:
            return self.forward_single(x)

        logits = []
        for angle in orientations:
            rotated = rotate_tensor(x, angle)
            pred = self.forward_single(rotated)
            pred = rotate_tensor(pred, -angle)
            if pred.shape[-2:] != x.shape[-2:]:
                pred = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)
            logits.append(pred)
        return torch.stack(logits, dim=0).mean(dim=0)


def rotate_tensor(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    if abs(angle_deg) < 1e-6:
        return x
    angle = math.radians(angle_deg)
    theta = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
        ],
        dtype=x.dtype,
        device=x.device,
    )
    theta = theta.unsqueeze(0).repeat(x.size(0), 1, 1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
