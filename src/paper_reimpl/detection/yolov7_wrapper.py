from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: str


class YOLOv7ONNXDetector:
    def __init__(self, config: dict) -> None:
        self.model_path = Path(config["yolo_onnx_path"])
        self.conf_threshold = float(config["confidence_threshold"])
        self.nms_threshold = float(config["nms_threshold"])
        self.input_size = int(config["input_size"])
        self.class_names = list(config["class_names"])

        if not self.model_path.exists():
            self.net = None
        else:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))

    def ready(self) -> bool:
        return self.net is not None

    def predict(self, image_rgb: np.ndarray) -> list[Detection]:
        if self.net is None:
            raise FileNotFoundError(
                f"YOLOv7 ONNX weights not found at {self.model_path}. Export or place the model there first."
            )

        blob = cv2.dnn.blobFromImage(
            image_rgb,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self._postprocess(outputs, image_rgb.shape[:2])

    def _postprocess(self, outputs: np.ndarray, shape: tuple[int, int]) -> list[Detection]:
        h, w = shape
        boxes = []
        scores = []
        class_ids = []
        preds = outputs[0] if outputs.ndim == 3 else outputs

        for row in preds:
            obj = float(row[4])
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            score = obj * float(class_scores[class_id])
            if score < self.conf_threshold:
                continue

            cx, cy, bw, bh = row[:4]
            x1 = int((cx - bw / 2) * w / self.input_size)
            y1 = int((cy - bh / 2) * h / self.input_size)
            x2 = int((cx + bw / 2) * w / self.input_size)
            y2 = int((cy + bh / 2) * h / self.input_size)
            boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
            scores.append(score)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        detections: list[Detection] = []
        for idx in indices.flatten() if len(indices) else []:
            x, y, bw, bh = boxes[idx]
            class_id = class_ids[idx]
            label = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            detections.append(
                Detection(
                    xyxy=(x, y, x + bw, y + bh),
                    confidence=float(scores[idx]),
                    class_id=class_id,
                    label=label,
                )
            )
        return detections
