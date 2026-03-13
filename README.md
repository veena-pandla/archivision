# ArchiVision

AI-powered pipeline that converts 2D architectural floor plans into 3D models — including scanned, real-world blueprints with all the messiness that comes with them.

---

## Why I built this

Architecture firm clients were waiting weeks for CAD teams to manually trace 2D blueprints into 3D walkthroughs. I looked at existing tools and none of them handled what scanned floor plans actually look like: faded lines, skewed scans, coffee stains, inconsistent dimension labels. So I built one from scratch.

---

## What it does

Upload a floor plan image (scan, photo, or digital). The system:

1. Preprocesses the image (denoising, skew correction, perspective normalization)
2. Detects walls, doors, windows, and furniture using YOLOv8
3. Segments rooms via contour analysis and flood-fill
4. Generates a 3D model in Blender
5. Returns a downloadable `.blend` or `.obj` file

Web interface included — architects don't need to touch any code.

---

## Tech stack

| Component | Tech |
|-----------|------|
| Object detection | YOLOv8 (custom trained) |
| Image preprocessing | OpenCV |
| Deep learning | TensorFlow |
| 3D generation | Blender Python API |
| Web interface | Flask |
| Data processing | NumPy, Pillow |

---

## Architecture

```
Floor Plan Image
       │
       ▼
┌─────────────────────┐
│   Preprocessing     │  ← Skew correction, denoising, adaptive threshold
│   (OpenCV)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Object Detection   │  ← YOLOv8, custom dataset (500+ annotated plans)
│  (YOLOv8)           │  ← Data aug: rotation, noise, perspective distortion
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Room Segmentation  │  ← Contour analysis, flood-fill, wall inference
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3D Generation      │  ← Blender Python API, LOD for furniture
│  (Blender API)      │  ← Batch geometry creation (8h → <30min)
└──────────┬──────────┘
           │
           ▼
     3D Model Output (.blend / .obj)
```

---

## Key technical challenges

**Training data problem**
No existing dataset matched real-world scanned floor plans. I manually annotated 500+ images, then applied heavy augmentation (rotation, noise injection, perspective distortion) to build a training set that actually generalized.

**Wall detection on degraded scans**
Faded or overlapping lines broke naive line detection. I implemented geometric algorithms that infer wall thickness from line clustering patterns — if lines cluster within a threshold distance, they're treated as the same wall.

**Processing time**
Initial implementation: 8 hours per complex floor plan. Root cause: naive Blender Python API calls creating geometry one object at a time. Fixed by batch-processing geometry creation and implementing LOD (level of detail) logic so distant furniture simplified in complexity. Result: under 30 minutes.

**Error handling**
Instead of failing silently on bad inputs (missing dimensions, furniture overlapping walls, ambiguous layouts), the system flags specific regions and returns them for architect review. This was required to make it usable in a real production context.

---

## Setup

```bash
git clone https://github.com/veena-pandla/archivision
cd archivision
pip install -r requirements.txt

# Run web interface
python app.py
```

Requirements: Python 3.9+, Blender 3.x installed and accessible via CLI

---

## Usage

```python
from archivision import FloorPlanProcessor

processor = FloorPlanProcessor()
result = processor.process("floorplan.jpg")
result.export("output.blend")
```

Or via web interface at `localhost:5000` after running `python app.py`.

---

## Project structure

```
archivision/
├── app.py                  # Flask web interface
├── processor/
│   ├── preprocessing.py    # OpenCV preprocessing pipeline
│   ├── detection.py        # YOLOv8 inference + postprocessing
│   ├── segmentation.py     # Room segmentation algorithms
│   └── generator.py        # Blender 3D generation
├── models/
│   └── yolov8_floorplan/   # Trained model weights
├── data/
│   └── annotations/        # Training dataset annotations
├── tests/
└── requirements.txt
```

---

## What I'd do differently

- Use a larger annotated dataset from the start (manual annotation was slow)
- Explore SAM (Segment Anything Model) for wall segmentation
- Add IFC format export for compatibility with professional CAD tools
