"""
Room segmentation from detected walls.

The challenge: wall lines in scanned floor plans are often faded, broken,
or overlapping. A naive approach (find closed polygons) fails constantly.

This module uses two strategies:
1. Contour analysis on the thresholded image for clean plans
2. Wall clustering + geometric inference for degraded scans

Wall clustering: lines within WALL_CLUSTER_THRESHOLD_PX pixels are grouped
and their midline is computed as the actual wall position. This handles the
common case where a wall is represented by two close parallel lines (showing
wall thickness) that the detector treats as one bounding box.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from .detection import Detection
import config


@dataclass
class Room:
    contour: np.ndarray
    centroid: Tuple[float, float]
    area_px: float
    bounding_rect: Tuple[int, int, int, int]   # x, y, w, h
    label: str = "room"


@dataclass
class SegmentationResult:
    rooms: List[Room] = field(default_factory=list)
    wall_lines: List[Tuple] = field(default_factory=list)   # (x1,y1,x2,y2)
    warnings: List[str] = field(default_factory=list)


def segment(image: np.ndarray, wall_detections: List[Detection]) -> SegmentationResult:
    result = SegmentationResult()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Strategy 1: Try clean contour extraction first
    rooms = _contour_segmentation(gray)

    if len(rooms) < 2:
        # Strategy 2: Fall back to wall-line clustering + flood fill
        result.warnings.append(
            "Contour segmentation found too few rooms — "
            "falling back to wall-line inference"
        )
        wall_lines = _cluster_walls(wall_detections, image.shape)
        result.wall_lines = wall_lines
        rooms = _floodfill_rooms(gray, wall_lines)

    result.rooms = rooms

    if len(rooms) == 0:
        result.warnings.append(
            "Could not identify distinct rooms — "
            "entire floor plan treated as single space"
        )

    return result


def _contour_segmentation(gray: np.ndarray) -> List[Room]:
    """
    Find closed regions by looking for contours in the inverted binary image.
    Works well on clean, digital floor plans.
    """
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Close small gaps in walls
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    rooms = []
    img_area = gray.shape[0] * gray.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter noise (tiny blobs) and the outer boundary of the whole plan
        if area < 500 or area > img_area * 0.6:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(contour)

        rooms.append(Room(
            contour=contour,
            centroid=(cx, cy),
            area_px=area,
            bounding_rect=(x, y, w, h)
        ))

    return rooms


def _cluster_walls(wall_detections: List[Detection], img_shape: tuple) -> List[Tuple]:
    """
    Convert wall bounding boxes into center lines, then cluster nearby
    parallel lines into single walls.

    The key insight: a wall bbox is wider than tall (horizontal wall) or
    taller than wide (vertical wall). We project it onto its dominant axis
    and get a line segment.
    """
    lines = []

    for det in wall_detections:
        x1, y1, x2, y2 = det.bbox
        w = x2 - x1
        h = y2 - y1

        if w >= h:  # horizontal wall
            cy = (y1 + y2) // 2
            lines.append(("H", x1, cy, x2, cy))
        else:       # vertical wall
            cx = (x1 + x2) // 2
            lines.append(("V", cx, y1, cx, y2))

    # Cluster: merge lines of the same orientation that are very close
    merged = _merge_close_lines(lines)
    return [(x1, y1, x2, y2) for (_, x1, y1, x2, y2) in merged]


def _merge_close_lines(lines: list) -> list:
    """
    Group lines within WALL_CLUSTER_THRESHOLD_PX of each other and
    return their average (the inferred wall centerline).
    """
    threshold = config.WALL_CLUSTER_THRESHOLD_PX
    used = [False] * len(lines)
    merged = []

    for i, line_i in enumerate(lines):
        if used[i]:
            continue
        group = [line_i]
        for j, line_j in enumerate(lines[i + 1:], i + 1):
            if used[j]:
                continue
            if line_i[0] == line_j[0]:  # same orientation
                # Distance between the parallel lines
                dist = abs(line_i[2] - line_j[2])  # compare y for H, x for V
                if dist <= threshold:
                    group.append(line_j)
                    used[j] = True

        # Average the group into one line
        avg = (
            group[0][0],
            int(np.mean([l[1] for l in group])),
            int(np.mean([l[2] for l in group])),
            int(np.mean([l[3] for l in group])),
            int(np.mean([l[4] for l in group])),
        )
        merged.append(avg)
        used[i] = True

    return merged


def _floodfill_rooms(gray: np.ndarray, wall_lines: List[Tuple]) -> List[Room]:
    """
    Draw inferred wall lines onto a blank canvas, then use flood fill
    to identify enclosed regions as rooms.
    """
    canvas = np.ones_like(gray) * 255
    for (x1, y1, x2, y2) in wall_lines:
        cv2.line(canvas, (x1, y1), (x2, y2), 0, thickness=3)

    # Close gaps at line endpoints
    kernel = np.ones((5, 5), np.uint8)
    canvas = cv2.dilate(255 - canvas, kernel, iterations=1)
    canvas = 255 - canvas

    _, binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rooms = []
    img_area = gray.shape[0] * gray.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000 or area > img_area * 0.7:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(contour)

        rooms.append(Room(
            contour=contour,
            centroid=(cx, cy),
            area_px=area,
            bounding_rect=(x, y, w, h)
        ))

    return rooms
