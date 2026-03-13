import os
import time
from .preprocessing import preprocess
from .detection import FloorPlanDetector
from .segmentation import segment
from .generator import generate_3d
from .result import ProcessingResult
import config


class FloorPlanProcessor:
    def __init__(self):
        self.detector = FloorPlanDetector()
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def process(self, image_path: str, job_id: str) -> ProcessingResult:
        warnings = []
        t_start = time.time()

        # 1. Preprocess
        image, meta = preprocess(image_path)

        # 2. Detect elements
        det_result = self.detector.detect(image)
        warnings.extend(det_result.warnings)

        # 3. Segment rooms
        seg_result = segment(image, det_result.walls)
        warnings.extend(seg_result.warnings)

        # 4. Generate 3D model
        output_path = os.path.join(config.OUTPUT_DIR, f"{job_id}.obj")
        scale = meta.get("scale", 1.0) * 0.005  # px → approx meters
        generate_3d(seg_result, det_result, output_path, scale_factor=scale)

        elapsed = time.time() - t_start

        return ProcessingResult(
            job_id=job_id,
            output_path=output_path,
            warnings=warnings,
            stats={
                "walls_detected": len(det_result.walls),
                "rooms_detected": len(seg_result.rooms),
                "doors_detected": len(det_result.doors),
                "windows_detected": len(det_result.windows),
                "processing_time_s": round(elapsed, 1),
                "skew_corrected": meta.get("skew_angle", 0) != 0,
            }
        )
