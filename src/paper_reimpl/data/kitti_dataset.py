from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SamplePaths:
    image: Path
    lidar: Path | None
    weather: Path | None
    seg_label: Path | None
    detection_label: Path | None


def _read_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return image


def _read_optional_image(path: Path | None, size: tuple[int, int]) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)
    return _read_image(path, size)


class KITTIMultiSensorDataset(Dataset):
    def __init__(self, config: dict[str, Any], split: str = "train") -> None:
        root = Path(config["root"])
        self.image_dir = root / config["image_dir"]
        self.lidar_dir = root / config["lidar_dir"]
        self.weather_dir = root / config["weather_dir"]
        self.labels_dir = root / config["labels_dir"]
        self.seg_labels_dir = root / config["seg_labels_dir"]
        self.size = tuple(config["image_size"])
        self.num_classes = int(config["num_segmentation_classes"])

        image_paths = sorted(self.image_dir.glob("*.png"))
        if not image_paths:
            image_paths = sorted(self.image_dir.glob("*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        split_idx = int(len(image_paths) * float(config["train_split"]))
        if split == "train":
            image_paths = image_paths[:split_idx]
        else:
            image_paths = image_paths[split_idx:]

        self.samples = [self._build_paths(path) for path in image_paths]

    def _build_paths(self, image_path: Path) -> SamplePaths:
        stem = image_path.stem
        lidar = self.lidar_dir / f"{stem}.png"
        weather = self.weather_dir / f"{stem}.png"
        seg = self.seg_labels_dir / f"{stem}.png"
        label = self.labels_dir / f"{stem}.txt"
        return SamplePaths(
            image=image_path,
            lidar=lidar if lidar.exists() else None,
            weather=weather if weather.exists() else None,
            seg_label=seg if seg.exists() else None,
            detection_label=label if label.exists() else None,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = _read_image(sample.image, self.size)
        lidar = _read_optional_image(sample.lidar, self.size)
        weather = _read_optional_image(sample.weather, self.size)

        seg_label = np.zeros(self.size, dtype=np.int64)
        if sample.seg_label is not None:
            label = cv2.imread(str(sample.seg_label), cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
            seg_label = label.astype(np.int64)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        lidar_tensor = torch.from_numpy(lidar.transpose(2, 0, 1)).float() / 255.0
        weather_tensor = torch.from_numpy(weather.transpose(2, 0, 1)).float() / 255.0
        seg_tensor = torch.from_numpy(seg_label).long()

        return {
            "id": sample.image.stem,
            "camera": image_tensor,
            "lidar": lidar_tensor,
            "weather": weather_tensor,
            "segmentation_mask": seg_tensor,
            "detection_label_path": str(sample.detection_label) if sample.detection_label else None,
        }
