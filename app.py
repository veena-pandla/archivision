import os
import uuid
import random
from flask import Flask, render_template, request, send_file, jsonify
import config

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Try to load the real processor; fall back to demo mode if model weights
# aren't present (e.g. on hosted demo deployments)
DEMO_MODE = False
try:
    from processor import FloorPlanProcessor
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError("Model weights not found")
    processor = FloorPlanProcessor()
except Exception:
    DEMO_MODE = True
    processor = None

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _demo_process(job_id: str, output_path: str) -> dict:
    """
    Return realistic-looking sample output when model weights aren't available.
    Writes a minimal valid .obj file so download works.
    """
    # Write a tiny sample .obj (a cube — stands in for real geometry)
    with open(output_path, "w") as f:
        f.write("# ArchiVision demo output\n")
        f.write("# Real output requires YOLOv8 model weights\n")
        f.write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n")
        f.write("v 0 0 2.8\nv 1 0 2.8\nv 1 1 2.8\nv 0 1 2.8\n")
        f.write("f 1 2 3 4\nf 5 6 7 8\nf 1 2 6 5\nf 2 3 7 6\n")

    return {
        "walls_detected": random.randint(8, 18),
        "rooms_detected": random.randint(3, 7),
        "doors_detected": random.randint(4, 9),
        "windows_detected": random.randint(4, 10),
        "processing_time_s": round(random.uniform(12.0, 28.0), 1),
        "skew_corrected": True,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Accepted: PNG, JPG, TIFF, BMP"}), 400

    job_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_path = os.path.join(config.UPLOAD_DIR, f"{job_id}.{ext}")
    file.save(input_path)
    output_path = os.path.join(config.OUTPUT_DIR, f"{job_id}.obj")

    try:
        if DEMO_MODE:
            stats = _demo_process(job_id, output_path)
            return jsonify({
                "job_id": job_id,
                "status": "success",
                "download_url": f"/download/{job_id}",
                "warnings": [],
                "stats": stats,
                "demo_mode": True,
            })

        result = processor.process(input_path, job_id)
        return jsonify({
            "job_id": job_id,
            "status": "success",
            "download_url": f"/download/{job_id}",
            "warnings": result.warnings,
            "stats": result.stats,
            "demo_mode": False,
        })
    except Exception as e:
        return jsonify({"error": str(e), "job_id": job_id}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.route("/download/<job_id>")
def download(job_id):
    try:
        uuid.UUID(job_id)
    except ValueError:
        return jsonify({"error": "Invalid job ID"}), 400

    output_path = os.path.join(config.OUTPUT_DIR, f"{job_id}.obj")
    if not os.path.exists(output_path):
        return jsonify({"error": "Output not found or expired"}), 404

    return send_file(output_path, as_attachment=True,
                     download_name="floorplan_3d.obj")


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 32MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
