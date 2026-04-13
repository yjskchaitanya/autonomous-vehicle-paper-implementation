from __future__ import annotations

import cv2
import numpy as np


def iaekf_denoise(image: np.ndarray) -> np.ndarray:
    """Practical IAEKF-style approximation for frame-wise denoising."""
    blurred = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
    estimate = cv2.addWeighted(image, 0.35, blurred, 0.65, 0)
    return estimate


def ngt_clahe(
    image: np.ndarray,
    gamma: float = 1.2,
    clip_limit: float = 2.5,
    grid_size: int = 8,
) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    img = np.power(np.clip(img, 1e-6, 1.0), gamma)
    img = (img * 255.0).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def iawmf_threshold(image: np.ndarray, kernel_size: int = 9, c: int = 4) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    weighted_mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    binary = np.where(gray > (weighted_mean - c), 255, 0).astype(np.uint8)
    return binary


def preprocess_image(image: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    denoised = iaekf_denoise(image)
    enhanced = ngt_clahe(
        denoised,
        gamma=float(config["gamma"]),
        clip_limit=float(config["clahe_clip_limit"]),
        grid_size=int(config["clahe_grid_size"]),
    )
    threshold = iawmf_threshold(
        enhanced,
        kernel_size=int(config["weighted_mean_kernel"]),
        c=int(config["weighted_mean_c"]),
    )
    return {
        "denoised": denoised,
        "enhanced": enhanced,
        "threshold": threshold,
    }
