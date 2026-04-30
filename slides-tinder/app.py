import os
import uuid
import random
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template
import fitz  # PyMuPDF

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
SLIDES_DIR = Path("slides")
ACCEPTED_DIR = Path("accepted")

for d in [UPLOAD_DIR, SLIDES_DIR, ACCEPTED_DIR]:
    d.mkdir(exist_ok=True)

# In-memory session state: list of slide paths in random order, index
state = {
    "queue": [],      # list of slide filenames (relative to SLIDES_DIR)
    "current": 0,
    "accepted_count": 0,
}


def pdf_to_pngs(pdf_path: Path, session_id: str) -> list[str]:
    doc = fitz.open(str(pdf_path))
    out_dir = SLIDES_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(2.0, 2.0)  # 2x scale → ~150 DPI → good resolution
        pix = page.get_pixmap(matrix=mat)
        filename = f"{session_id}_p{i:04d}.png"
        pix.save(str(out_dir / filename))
        paths.append(f"{session_id}/{filename}")
    doc.close()
    return paths


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("pdfs")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    all_slides = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        session_id = str(uuid.uuid4())[:8]
        pdf_path = UPLOAD_DIR / f"{session_id}.pdf"
        f.save(str(pdf_path))
        slides = pdf_to_pngs(pdf_path, session_id)
        all_slides.extend(slides)

    if not all_slides:
        return jsonify({"error": "No valid PDFs found"}), 400

    random.shuffle(all_slides)
    state["queue"] = all_slides
    state["current"] = 0
    state["accepted_count"] = 0

    # Clear previous accepted
    shutil.rmtree(str(ACCEPTED_DIR), ignore_errors=True)
    ACCEPTED_DIR.mkdir(exist_ok=True)

    return jsonify({"total": len(all_slides)})


@app.route("/slide/current")
def current_slide():
    q = state["queue"]
    idx = state["current"]
    if idx >= len(q):
        return jsonify({"done": True, "accepted_count": state["accepted_count"]})
    return jsonify({
        "done": False,
        "slide": q[idx],
        "index": idx + 1,
        "total": len(q),
        "accepted_count": state["accepted_count"],
    })


@app.route("/slide/action", methods=["POST"])
def slide_action():
    data = request.get_json()
    action = data.get("action")  # "accept" or "reject"
    q = state["queue"]
    idx = state["current"]

    if idx >= len(q):
        return jsonify({"error": "No more slides"}), 400

    slide_rel = q[idx]
    if action == "accept":
        src = SLIDES_DIR / slide_rel
        dest = ACCEPTED_DIR / Path(slide_rel).name
        shutil.copy2(str(src), str(dest))
        state["accepted_count"] += 1

    state["current"] += 1
    return jsonify({"accepted_count": state["accepted_count"]})


@app.route("/slides/<path:filename>")
def serve_slide(filename):
    return send_from_directory(str(SLIDES_DIR), filename)


@app.route("/accepted_count")
def accepted_count():
    return jsonify({"count": state["accepted_count"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
