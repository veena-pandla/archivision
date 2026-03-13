import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8_floorplan", "best.pt")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Detection confidence thresholds
WALL_CONF = 0.45
DOOR_CONF = 0.50
WINDOW_CONF = 0.50
FURNITURE_CONF = 0.40

# Class IDs from training dataset
CLASS_IDS = {
    "wall": 0,
    "door": 1,
    "window": 2,
    "room": 3,
    "furniture": 4,
    "staircase": 5,
}

# Preprocessing
TARGET_SIZE = (1024, 1024)
SKEW_CORRECTION = True
ADAPTIVE_THRESHOLD_BLOCK = 11
ADAPTIVE_THRESHOLD_C = 2

# Wall detection
WALL_CLUSTER_THRESHOLD_PX = 8   # lines within this distance = same wall
MIN_WALL_LENGTH_PX = 30

# 3D generation
DEFAULT_WALL_HEIGHT = 2.8       # meters
DEFAULT_FLOOR_THICKNESS = 0.2
DOOR_HEIGHT = 2.1
WINDOW_HEIGHT = 1.2
WINDOW_SILL_HEIGHT = 0.9
LOD_DISTANCE_THRESHOLD = 10.0  # meters — simplify furniture beyond this

# Blender
BLENDER_EXEC = os.environ.get("BLENDER_PATH", "blender")
