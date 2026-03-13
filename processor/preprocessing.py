"""
Preprocessing pipeline for scanned floor plans.

Real-world scans have: uneven lighting, skew from scanner beds, shadows at
fold lines, faded ink, and scanner artifacts along the edges. This module
handles all of that before anything goes to the detection model.
"""

import cv2
import numpy as np
from typing import Tuple
import config


def preprocess(image_path: str) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline. Returns cleaned image + metadata dict
    with corrections applied (used for downstream coordinate mapping).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    meta = {"original_shape": img.shape[:2]}

    img = _remove_border_artifacts(img)
    img, meta["skew_angle"] = _correct_skew(img)
    img = _normalize_lighting(img)
    img = _denoise(img)
    img, meta["scale"] = _resize(img, config.TARGET_SIZE)
    img = _adaptive_threshold(img)

    return img, meta


def _remove_border_artifacts(img: np.ndarray) -> np.ndarray:
    """
    Scanner beds often leave dark bands along one or more edges.
    Crop them out by finding the largest bright rectangle.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find bounding rect of all non-dark content
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)

    # Add a small margin so we don't clip actual content
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)

    return img[y:y + h, x:x + w]


def _correct_skew(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect and correct document skew using Hough line transform.

    Scans placed slightly askew on the scanner bed produce lines that
    aren't quite horizontal/vertical. We find the dominant angle from
    the strongest lines and rotate to compensate.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return img, 0.0

    angles = []
    for line in lines[:50]:  # only look at strongest lines
        rho, theta = line[0]
        # theta near 0 or pi = horizontal line; near pi/2 = vertical
        angle_deg = np.degrees(theta)
        if angle_deg < 45 or angle_deg > 135:
            angles.append(angle_deg)
        else:
            angles.append(angle_deg - 90)

    if not angles:
        return img, 0.0

    skew_angle = np.median(angles)

    # Only correct if skew is meaningful (avoids over-rotating clean scans)
    if abs(skew_angle) < 0.5:
        return img, skew_angle

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    corrected = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return corrected, skew_angle


def _normalize_lighting(img: np.ndarray) -> np.ndarray:
    """
    Compensate for uneven lighting across the scan surface (common with
    overhead lighting or fold shadows in the middle of a large blueprint).
    Uses CLAHE on the L channel in LAB color space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _denoise(img: np.ndarray) -> np.ndarray:
    """
    Remove scanner noise and minor print artifacts.
    Fast non-local means denoising — preserves edges better than Gaussian.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, h=6, hColor=6,
                                            templateWindowSize=7,
                                            searchWindowSize=21)


def _resize(img: np.ndarray, target: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """
    Resize to target dimensions while preserving aspect ratio.
    Returns image and the scale factor (needed to map detections back
    to original coordinates).
    """
    h, w = img.shape[:2]
    scale = min(target[0] / w, target[1] / h)

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to exact target size
    top = (target[1] - new_h) // 2
    bottom = target[1] - new_h - top
    left = (target[0] - new_w) // 2
    right = target[0] - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded, scale


def _adaptive_threshold(img: np.ndarray) -> np.ndarray:
    """
    Convert to binary using adaptive thresholding. Regular (global) threshold
    fails when a scan has brightness gradients — one side is slightly darker.
    Adaptive threshold computes the threshold locally per tile.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.ADAPTIVE_THRESHOLD_BLOCK,
        config.ADAPTIVE_THRESHOLD_C
    )
    # Return as 3-channel so the rest of the pipeline doesn't have to care
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
