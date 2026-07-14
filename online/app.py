from __future__ import annotations
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io
import time
import uuid
import service
from pathlib import Path
import json
from datetime import datetime

app = Flask(
    __name__,
    static_folder=str(service.BASE_DIR / "static"),
    static_url_path="/static",
    template_folder=str(service.BASE_DIR / "templates"),
)
# Project root = online/..  (ciyujiance/)
PROJECT_DIR = service.BASE_DIR.parent
SAVED_DIR = PROJECT_DIR / "resources" / "saved"
INDEX_PATH = SAVED_DIR / "index.json"

SAVED_DIR.mkdir(parents=True, exist_ok=True)

def _load_index() -> dict:
    if not INDEX_PATH.exists():
        return {}
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_index(idx: dict) -> None:
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(INDEX_PATH)

def _make_default_name(idx: dict) -> str:
    # saved_note_01.txt / saved_note_02.txt ...
    n = len(idx) + 1
    return f"saved_note_{n:02d}.txt"

# =========================================================
# Routes: Return to homepage
# =========================================================
@app.route("/")
def index():
    return render_template("index.html")

# =========================================================
# -------- Words / corpus list --------
# =========================================================
# Provide the GUI with a data source for "vocabulary lookup/search".
# input:
# `search` parameter: Keywords input from the front end for word filtering.
# `use_fin` parameter: Whether to enable "Financial Dictionary Extension".
# Returns {"words": [...]}
@app.route("/api/words")
def api_words():
    q = request.args.get("search", "").strip().lower()
    use_fin = request.args.get("use_fin", "false").lower() == "true"

    base = set(service.VOCAB_SET)
    if use_fin and service.FINANCE_EXTRA:
        base |= service.FINANCE_EXTRA

    words = sorted(base)
    if q:
        words = [w for w in words if q in w]

    return jsonify({"words": words})

# =========================================================
# -------- Check spelling --------
# =========================================================
# The core interface for spell checking.
# It is invoked when the front-end clicks "Run check".
# input format:
# {
#   "text": "...",
#   "settings": {
#     "enable_auto_correction": false,
#     "show_candidate_ranking": true,
#     "enable_finance_dictionary": false,
#     "show_confidence": false
#   }
# }
# output format:
# {
#   "errors": [
#     {"word": "...", "start": 10, "end": 15, "type": "non-word"}
#   ],
#   "suggestions": {
#     "wrongWord": [
#       {"candidate": "rightWord", "edit_distance": 1, "score": 0.8}
#     ]
#   }
# }
@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(force=True)
    text = data.get("text", "") or ""
    settings = data.get("settings", {}) or {}
    settings["enable_finance_dictionary"] = True
    print(settings)

    text = text[:500]
    print(text)
    t0 = time.perf_counter()
    result = service.detect_and_suggest(text, settings)
    print(result)
    t1 = time.perf_counter()
    print(f"Total Time: {t1 - t0:.6f} s")
    return jsonify(result)

# =========================================================
# -------- Upload file (text) --------
# =========================================================
# Upload a .txt file, read its text content, and populate it back into the editor.
# Retrieve the file from request.files["file"].
# Preferably use UTF-8 decoding; if that fails, use Latin-1.
# Returns {"filename", "text"}. Frontend trigger: Click "Upload file" and select the file.
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(f.filename)
    content = f.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1", errors="ignore")

    return jsonify({"filename": filename, "text": text})

# =========================================================
# -------- Save text --------
# =========================================================
# Save the current text in the editing area to backend memory (temporary storage).
# This is not written to disk; it's only for your Save modal demo.
# Generate a UUID as the ID.
# Save structure: {name, text, created_at}
# Return the newly created {id, name}
# Frontend trigger: Click "Save text".
@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True) or {}
    text = (data.get("text", "") or "")[:500]

    idx = _load_index()

    name = data.get("name") or _make_default_name(idx)
    name = secure_filename(name) or "saved_note.txt"
    if not name.lower().endswith(".txt"):
        name += ".txt"

    sid = str(uuid.uuid4())
    created_at = int(time.time())

    # 写 txt
    (SAVED_DIR / f"{sid}.txt").write_text(text, encoding="utf-8")

    # 写 index
    idx[sid] = {"name": name, "created_at": created_at}
    _save_index(idx)

    return jsonify({"id": sid, "name": name})

# =========================================================
# -------- List saved --------
# =========================================================
# Retrieves a list of "Saved Texts" for use in the Save Text popup.
# Converts SAVED_TEXTS to a list.
# Sorts by created_at.
# Returns {"items": [...]}. Front-end trigger: Refreshes the list when the Save modal is opened.
@app.route("/api/saved")
def api_saved_list():
    idx = _load_index()

    items = []
    for sid, meta in idx.items():
        fp = SAVED_DIR / f"{sid}.txt"
        if not fp.exists():
            continue
        full_text = fp.read_text(encoding="utf-8", errors="ignore")
        preview = (full_text[:80] + "…") if len(full_text) > 80 else full_text

        items.append({
            "id": sid,
            "name": meta.get("name", f"{sid}.txt"),
            "created_at": int(meta.get("created_at", 0)),
            # 这两个字段是你前端 Save 弹窗需要的
            "preview": preview,
            "full_text": full_text
        })

    items.sort(key=lambda x: x["created_at"])
    return jsonify({"items": items})

# =========================================================
# -------- Open saved --------
# =========================================================
# Open a saved record and populate the text back into the editor.
# If the ID does not exist, return a 404 error.
# If it exists, return {id, name, text}.
# Frontend trigger: Click "Open" in the Save modal.
@app.route("/api/saved/<sid>")
def api_saved_open(sid):
    idx = _load_index()
    meta = idx.get(sid)
    fp = SAVED_DIR / f"{sid}.txt"
    if not meta or not fp.exists():
        return jsonify({"error": "Not found"}), 404

    text = fp.read_text(encoding="utf-8", errors="ignore")
    return jsonify({
        "id": sid,
        "name": meta.get("name", f"{sid}.txt"),
        "text": text
    })

# =========================================================
# -------- Delete saved --------
# =========================================================
# Delete a saved record.
# Remove from SAVED_TEXTS
# Return {"ok": True}
# Frontend trigger: Click Delete in the Save modal.
@app.route("/api/saved/<sid>", methods=["DELETE"])
def api_saved_delete(sid):
    idx = _load_index()

    # 删除正文文件
    fp = SAVED_DIR / f"{sid}.txt"
    if fp.exists():
        try:
            fp.unlink()
        except Exception:
            pass

    if sid in idx:
        idx.pop(sid, None)
        _save_index(idx)

    return jsonify({"ok": True})

# =========================================================
# -------- Export text --------
# =========================================================
# Export the editor content as a downloadable TXT file.
# Receive {text, name}
# Generate an in-memory file using BytesIO
# send_file(... as_attachment=True ...)
# Front-end trigger: Click Export.
@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.get_json(force=True)
    text = (data.get("text", "") or "")[:500]
    name = data.get("name") or "export.txt"

    buf = io.BytesIO(text.encode("utf-8"))
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name=name,
        mimetype="text/plain; charset=utf-8"
    )

@app.route("/api/autocorrect", methods=["POST"])
def api_autocorrect():
    data = request.get_json(force=True) or {}
    text = (data.get("text", "") or "")[:500]
    errors = data.get("errors", []) or []
    suggestions = data.get("suggestions", {}) or {}

    # (start, end, replacement)
    reps = []
    for er in errors:
        try:
            start = int(er.get("start"))
            end = int(er.get("end"))
        except Exception:
            continue

        if start < 0 or end <= start or end > len(text):
            continue

        er_id = er.get("id") or f"{start}:{end}"
        s_list = suggestions.get(er_id) or suggestions.get(er.get("word"))

        if not s_list or not isinstance(s_list, list):
            continue

        top = s_list[0] or {}
        cand = top.get("candidate")
        if not cand or not isinstance(cand, str):
            continue

        wrong = er.get("word")
        if wrong and text[start:end] != wrong:
            continue

        reps.append((start, end, cand))

    reps.sort(key=lambda x: x[0], reverse=True)

    new_text = text
    last_start = len(new_text) + 1
    for start, end, cand in reps:
        if end > last_start:
            continue
        new_text = new_text[:start] + cand + new_text[end:]
        last_start = start

    return jsonify({"text": new_text})


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    # debug=True
    app.run(debug=True)
