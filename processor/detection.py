"""
YOLOv8-based object detection for floor plan elements.

Detects: walls, doors, windows, furniture, rooms, staircases.

The model was trained on a custom dataset of 500+ annotated floor plans.
Training data augmentation included rotation (0-360°), gaussian noise,
perspective distortion, and brightness variation to handle the diversity
of real-world scanned blueprints.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import config


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple          # (x1, y1, x2, y2) in pixels
    center: tuple = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self):
        return self.width * self.height


@dataclass
class DetectionResult:
    walls: List[Detection] = field(default_factory=list)
    doors: List[Detection] = field(default_factory=list)
    windows: List[Detection] = field(default_factory=list)
    furniture: List[Detection] = field(default_factory=list)
    rooms: List[Detection] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class FloorPlanDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.MODEL_PATH
        self._model = None

    def _load_model(self):
        """Lazy load — don't pay startup cost if not needed."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load detection model from {self.model_path}. "
                    f"Make sure model weights exist. Error: {e}"
                )

    def detect(self, image: np.ndarray) -> DetectionResult:
        self._load_model()

        results = self._model(image, verbose=False)[0]
        result = DetectionResult()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            class_name = results.names[cls_id]
            det = Detection(class_name=class_name, confidence=conf,
                            bbox=(x1, y1, x2, y2))

            if class_name == "wall" and conf >= config.WALL_CONF:
                result.walls.append(det)
            elif class_name == "door" and conf >= config.DOOR_CONF:
                result.doors.append(det)
            elif class_name == "window" and conf >= config.WINDOW_CONF:
                result.windows.append(det)
            elif class_name == "furniture" and conf >= config.FURNITURE_CONF:
                result.furniture.append(det)
            elif class_name == "room":
                result.rooms.append(det)

        self._validate(result)
        return result

    def _validate(self, result: DetectionResult):
        """
        Basic sanity checks. Flag suspicious detections rather than
        silently producing bad geometry.
        """
        if len(result.walls) == 0:
            result.warnings.append(
                "No walls detected — image may be too low resolution "
                "or floor plan style is outside training distribution"
            )

        # Furniture inside walls usually means a false positive
        for furn in result.furniture:
            for wall in result.walls:
                if _bbox_inside(furn.bbox, wall.bbox):
                    result.warnings.append(
                        f"Furniture detection at {furn.center} appears to "
                        f"overlap with wall — flagged for review"
                    )
                    break

        if len(result.rooms) == 0 and len(result.walls) > 4:
            result.warnings.append(
                "Walls detected but no rooms identified — "
                "room segmentation will fall back to contour analysis"
            )


def _bbox_inside(inner: tuple, outer: tuple) -> bool:
    """Returns True if inner bbox is fully contained by outer bbox."""
    return (inner[0] >= outer[0] and inner[1] >= outer[1] and
            inner[2] <= outer[2] and inner[3] <= outer[3])
