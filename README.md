# Multi-Sensor Fusion and Segmentation for Autonomous Vehicle Multi-Object Tracking

This repository is a faithful reimplementation scaffold of the paper:

`Multi-sensor fusion and segmentation for autonomous vehicle multi-object tracking using deep Q networks`

Paper PDF: [s41598-024-82356-0.pdf](D:/Final%20Year%20PRoject/Phase-1/reading%20papers/Autonomus%20vehicle%20paper/s41598-024-82356-0.pdf)

## Important note

The paper does **not** publish enough low-level implementation detail to guarantee bit-for-bit reproduction of the reported numbers. In particular, it omits:

- exact train/validation split policy
- exact KITTI weather augmentation pipeline
- exact DenseNet fusion layout
- exact YOLOv7 training recipe and class mapping
- exact reward shaping used for DQN
- exact segmentation labels and annotation source

So this codebase is designed to be:

- faithful to the paper's pipeline
- runnable and extensible
- explicit about every assumption

If you need the **same reported results**, you will still need to tune hyperparameters and match the authors' unpublished preprocessing and augmentation setup.

## Implemented pipeline

1. `IAEKF`-style denoising approximation
2. `NGT-CLAHE` contrast enhancement
3. `IAWMF` adaptive thresholding
4. multi-orientation segmentation
5. DenseNet-style multi-sensor fusion
6. YOLOv7-compatible detection wrapper
7. grid-map path scoring with EVO
8. DQN-based action selection for path/lane choice

## Project layout

- `configs/paper_reimplementation.yaml`
- `scripts/train_segmentation.py`
- `scripts/train_dqn.py`
- `scripts/run_inference.py`
- `src/paper_reimpl/`

## Dataset layout expected

Create a KITTI-style folder like this:

```text
data/
  kitti/
    image_2/
      000000.png
    lidar_bev/
      000000.png
    weather/
      000000.png
    labels/
      000000.txt
    seg_labels/
      000000.png
```

Notes:

- `lidar_bev/` should contain a projected LiDAR bird's-eye-view or depth image per frame.
- `weather/` is optional and may contain weather-aware auxiliary images or synthetic weather maps.
- `seg_labels/` is optional but required for supervised segmentation training.
- `labels/` uses YOLO text format for detection training or evaluation.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Train segmentation

```bash
python scripts/train_segmentation.py --config configs/paper_reimplementation.yaml
```

## Train DQN

```bash
python scripts/train_dqn.py --config configs/paper_reimplementation.yaml
```

## Run inference

```bash
python scripts/run_inference.py --config configs/paper_reimplementation.yaml --image data/kitti/image_2/000000.png
```

## YOLOv7 weights

The wrapper expects either:

- an ONNX file exported from YOLOv7, or
- your own PyTorch detector integration

Set the path in `configs/paper_reimplementation.yaml`.

## Reproducibility guidance

For your final year project, I recommend:

1. Start with segmentation + fusion only.
2. Use provided KITTI splits and save every experiment.
3. Add YOLOv7 evaluation next.
4. Add DQN path selection last.
5. Report paper-matched metrics plus your own ablation study.

## Assumptions encoded in this repo

- the paper's "IAEKF" is approximated as a practical denoising stage suitable for image pipelines
- multi-orientation segmentation is implemented through rotated-view inference and logit fusion
- EVO is implemented from the paper equations as a grid-quality optimizer
- DQN state representation uses detector boxes, fused-map scores, and vehicle context
