from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_reimpl.config import Config
from paper_reimpl.pipeline import PaperPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--lidar")
    parser.add_argument("--weather")
    parser.add_argument("--segmenter-weights")
    parser.add_argument("--fusion-weights")
    args = parser.parse_args()

    config = Config.load(args.config).raw
    device = config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    pipeline = PaperPipeline(config, device=device)
    pipeline.load_weights(args.segmenter_weights, args.fusion_weights)

    camera = read_rgb(args.image)
    lidar = read_rgb(args.lidar) if args.lidar else None
    weather = read_rgb(args.weather) if args.weather else None
    output = pipeline.run(camera, lidar, weather)

    print("Detections:", len(output.detections))
    print("Best EVO cell:", output.evo_result["best_cell"])
    print("Best EVO score:", output.evo_result["best_score"])
    print("State vector shape:", output.state_vector.shape)


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
