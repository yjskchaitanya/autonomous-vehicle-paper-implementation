from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .detection.yolov7_wrapper import YOLOv7ONNXDetector
from .models.fusion import DenseFusionNet
from .models.preprocessing import preprocess_image
from .models.segmentation import MultiOrientationSegmenter
from .planning.evo import EnergyValleyOptimizer


@dataclass
class PipelineOutput:
    segmentation_logits: torch.Tensor
    fused_features: torch.Tensor
    detections: list
    evo_result: dict
    state_vector: np.ndarray


class PaperPipeline:
    def __init__(self, config: dict, device: str = "cpu") -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        seg_cfg = config["segmentation"]
        fusion_cfg = config["fusion"]
        self.preprocessing_cfg = config["preprocessing"]
        self.detector = YOLOv7ONNXDetector(config["detection"])
        self.segmenter = MultiOrientationSegmenter(
            in_channels=4,
            base_channels=int(seg_cfg["base_channels"]),
            num_classes=int(seg_cfg["num_classes"]),
        ).to(self.device)
        self.fusion_net = DenseFusionNet(
            in_channels=15,
            growth_rate=int(fusion_cfg["growth_rate"]),
            block_layers=list(fusion_cfg["block_layers"]),
            out_channels=int(fusion_cfg["out_channels"]),
        ).to(self.device)
        planner_cfg = config["planner"]
        self.evo = EnergyValleyOptimizer(
            population=int(planner_cfg["evo_population"]),
            iterations=int(planner_cfg["evo_iterations"]),
            threshold_scale=float(planner_cfg["enrichment_threshold_scale"]),
        )
        self.orientations = list(seg_cfg["orientations"])

    def load_weights(self, segmenter_path: str | Path | None = None, fusion_path: str | Path | None = None) -> None:
        if segmenter_path and Path(segmenter_path).exists():
            self.segmenter.load_state_dict(torch.load(segmenter_path, map_location=self.device))
        if fusion_path and Path(fusion_path).exists():
            self.fusion_net.load_state_dict(torch.load(fusion_path, map_location=self.device))

    def run(self, camera_rgb: np.ndarray, lidar_rgb: np.ndarray | None = None, weather_rgb: np.ndarray | None = None) -> PipelineOutput:
        lidar_rgb = lidar_rgb if lidar_rgb is not None else np.zeros_like(camera_rgb)
        weather_rgb = weather_rgb if weather_rgb is not None else np.zeros_like(camera_rgb)

        pre = preprocess_image(camera_rgb, self.preprocessing_cfg)
        threshold = pre["threshold"][..., None]
        seg_input = np.concatenate([pre["enhanced"], threshold], axis=2)
        seg_input_t = torch.from_numpy(seg_input.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        segmentation_logits = self.segmenter(seg_input_t, self.orientations)

        cam_t = torch.from_numpy(pre["enhanced"].transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        lidar_t = torch.from_numpy(lidar_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        weather_t = torch.from_numpy(weather_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        fused_features = self.fusion_net(cam_t, lidar_t, weather_t, segmentation_logits)

        detections = self.detector.predict(pre["enhanced"]) if self.detector.ready() else []
        quality_map = self._build_quality_map(segmentation_logits, detections)
        evo_result = self.evo.optimize(quality_map)
        state_vector = self._build_state_vector(fused_features, detections, evo_result)

        return PipelineOutput(
            segmentation_logits=segmentation_logits,
            fused_features=fused_features,
            detections=detections,
            evo_result=evo_result,
            state_vector=state_vector,
        )

    def _build_quality_map(self, segmentation_logits: torch.Tensor, detections: list) -> np.ndarray:
        probs = F.softmax(segmentation_logits, dim=1)
        road_prob = probs[:, 0:1].mean(dim=1).squeeze(0).detach().cpu().numpy()
        obstacle_penalty = np.zeros_like(road_prob)
        h, w = obstacle_penalty.shape
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            x1 = np.clip(x1, 0, w - 1)
            x2 = np.clip(x2, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            y2 = np.clip(y2, 0, h - 1)
            obstacle_penalty[y1:y2, x1:x2] += det.confidence
        quality = np.clip(road_prob - obstacle_penalty, 0.0, None)
        return cv2.resize(quality, (24, 24), interpolation=cv2.INTER_AREA)

    def _build_state_vector(self, fused_features: torch.Tensor, detections: list, evo_result: dict) -> np.ndarray:
        pooled = F.adaptive_avg_pool2d(fused_features, output_size=(1, 1)).flatten(1).squeeze(0).detach().cpu().numpy()
        det_stats = np.zeros(6, dtype=np.float32)
        det_stats[0] = len(detections)
        if detections:
            det_stats[1] = max(d.confidence for d in detections)
            det_stats[2] = sum(1 for d in detections if d.label in {"car", "truck", "bus"})
            det_stats[3] = sum(1 for d in detections if d.label == "pedestrian")
            det_stats[4] = float(evo_result["best_cell"][0])
            det_stats[5] = float(evo_result["best_cell"][1])
        else:
            det_stats[4] = float(evo_result["best_cell"][0])
            det_stats[5] = float(evo_result["best_cell"][1])

        state = np.concatenate([pooled[:26], det_stats], axis=0).astype(np.float32)
        return state
