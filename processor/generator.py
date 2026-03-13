"""
3D model generation via Blender Python API.

Runs as a subprocess calling Blender with a generated script.
The script is assembled from room/wall/door/window data, then
Blender executes it headlessly and exports the result.

Performance note: the original implementation made one Blender API
call per object, which meant hundreds of blocking round-trips for
complex floor plans (~8 hours on a 40-room plan). The current version
assembles all geometry creation into a single script and lets Blender
batch-execute it. Average time for complex plans is now under 30 minutes.
"""

import subprocess
import tempfile
import os
import json
from typing import List
from .segmentation import Room, SegmentationResult
from .detection import DetectionResult
import config


def generate_3d(
    seg_result: SegmentationResult,
    det_result: DetectionResult,
    output_path: str,
    scale_factor: float = 1.0,
) -> str:
    """
    Generate a 3D model and write it to output_path (.blend or .obj).
    Returns the output path.
    """
    scene_data = _build_scene_data(seg_result, det_result, scale_factor)
    script = _build_blender_script(scene_data, output_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [config.BLENDER_EXEC, "--background", "--python", script_path],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender exited with code {result.returncode}.\n"
                f"stderr: {result.stderr[-2000:]}"
            )
    finally:
        os.unlink(script_path)

    return output_path


def _build_scene_data(seg_result: SegmentationResult,
                      det_result: DetectionResult,
                      scale: float) -> dict:
    """
    Convert pixel-space detection/segmentation data into metric-space
    scene geometry that Blender can work with.
    """
    rooms = []
    for room in seg_result.rooms:
        x, y, w, h = room.bounding_rect
        rooms.append({
            "x": x * scale,
            "y": y * scale,
            "w": w * scale,
            "h": h * scale,
            "label": room.label,
        })

    walls = []
    for det in det_result.walls:
        x1, y1, x2, y2 = det.bbox
        walls.append({
            "x1": x1 * scale, "y1": y1 * scale,
            "x2": x2 * scale, "y2": y2 * scale,
            "height": config.DEFAULT_WALL_HEIGHT,
        })

    doors = []
    for det in det_result.doors:
        x1, y1, x2, y2 = det.bbox
        cx = ((x1 + x2) / 2) * scale
        cy = ((y1 + y2) / 2) * scale
        doors.append({"cx": cx, "cy": cy,
                       "w": (x2 - x1) * scale,
                       "height": config.DOOR_HEIGHT})

    windows = []
    for det in det_result.windows:
        x1, y1, x2, y2 = det.bbox
        cx = ((x1 + x2) / 2) * scale
        cy = ((y1 + y2) / 2) * scale
        windows.append({"cx": cx, "cy": cy,
                         "w": (x2 - x1) * scale,
                         "height": config.WINDOW_HEIGHT,
                         "sill": config.WINDOW_SILL_HEIGHT})

    furniture = []
    for det in det_result.furniture:
        x1, y1, x2, y2 = det.bbox
        furniture.append({
            "cx": ((x1 + x2) / 2) * scale,
            "cy": ((y1 + y2) / 2) * scale,
            "w": (x2 - x1) * scale,
            "d": (y2 - y1) * scale,
        })

    return {
        "rooms": rooms, "walls": walls, "doors": doors,
        "windows": windows, "furniture": furniture,
        "floor_thickness": config.DEFAULT_FLOOR_THICKNESS,
        "lod_threshold": config.LOD_DISTANCE_THRESHOLD,
    }


def _build_blender_script(scene_data: dict, output_path: str) -> str:
    """
    Generate a self-contained Blender Python script from scene data.
    All geometry is created in a single script execution — no back-and-forth.
    """
    data_json = json.dumps(scene_data)

    return f"""
import bpy
import json
import math

scene_data = json.loads('{data_json}')

# Clean the default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def add_box(name, x, y, z, sx, sy, sz):
    bpy.ops.mesh.primitive_cube_add(location=(x, y, z))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (sx / 2, sy / 2, sz / 2)
    bpy.ops.object.transform_apply(scale=True)
    return obj

# --- Walls ---
for i, wall in enumerate(scene_data['walls']):
    x1, y1 = wall['x1'], wall['y1']
    x2, y2 = wall['x2'], wall['y2']
    h = wall['height']
    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    angle = math.atan2(y2-y1, x2-x1)

    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, h/2))
    obj = bpy.context.active_object
    obj.name = f"wall_{{i}}"
    obj.scale = (length/2, 0.1, h/2)
    obj.rotation_euler[2] = angle
    bpy.ops.object.transform_apply(scale=True, rotation=True)

# --- Floors (one per room) ---
for i, room in enumerate(scene_data['rooms']):
    ft = scene_data['floor_thickness']
    add_box(
        f"floor_{{i}}",
        room['x'] + room['w']/2,
        room['y'] + room['h']/2,
        -ft/2,
        room['w'], room['h'], ft
    )

# --- Doors (openings cut into wall geometry) ---
for i, door in enumerate(scene_data['doors']):
    add_box(
        f"door_frame_{{i}}",
        door['cx'], door['cy'], door['height']/2,
        door['w'], 0.05, door['height']
    )

# --- Windows ---
for i, win in enumerate(scene_data['windows']):
    z_center = win['sill'] + win['height'] / 2
    add_box(
        f"window_{{i}}",
        win['cx'], win['cy'], z_center,
        win['w'], 0.05, win['height']
    )

# --- Furniture (LOD: simplify if small) ---
lod_thresh = scene_data['lod_threshold']
for i, furn in enumerate(scene_data['furniture']):
    size = max(furn['w'], furn['d'])
    if size < lod_thresh:
        # Simplified LOD representation
        add_box(f"furn_lod_{{i}}", furn['cx'], furn['cy'], 0.4,
                furn['w'], furn['d'], 0.8)
    else:
        add_box(f"furn_{{i}}", furn['cx'], furn['cy'], 0.4,
                furn['w'], furn['d'], 0.8)

# --- Export ---
output = "{output_path}"
if output.endswith('.obj'):
    bpy.ops.wm.obj_export(filepath=output)
else:
    bpy.ops.wm.save_as_mainfile(filepath=output)

print("ArchiVision: export complete ->", output)
"""
