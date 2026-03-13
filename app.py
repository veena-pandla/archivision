import os
import uuid
from flask import Flask, render_template, request, send_file, jsonify
from processor import FloorPlanProcessor
import config

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

processor = FloorPlanProcessor()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp", "pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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

    try:
        result = processor.process(input_path, job_id)
        return jsonify({
            "job_id": job_id,
            "status": "success",
            "download_url": f"/download/{job_id}",
            "warnings": result.warnings,
            "stats": result.stats,
        })
    except Exception as e:
        return jsonify({"error": str(e), "job_id": job_id}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(input_path):
            os.remove(input_path)


@app.route("/download/<job_id>")
def download(job_id):
    # Sanitize job_id — it should be a UUID, nothing else
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
    app.run(debug=False, host="0.0.0.0", port=5000)
