import os, uuid, time, logging
import io
from flask import Flask, request, g
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from inference import CatClassifier
from utils.validation import validate_upload, MAX_FILE_MB
from utils.responses import ok, err

# (opsional) CORS
CORS_ENABLED = os.getenv("CORS_ENABLED", "false").lower() == "true"
if CORS_ENABLED:
    from flask_cors import CORS

load_dotenv()

# === Konfigurasi dasar ===
THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))
TOPK      = int(os.getenv("TOPK", "3"))
PORT      = int(os.getenv("PORT", "8000"))

# === Logger sederhana ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("cat-api")

app = Flask(__name__)
if CORS_ENABLED:
    CORS(app)  # default: izinkan semua origin — atur sesuai kebutuhan

# Batasi request body (hard cap) – sedikit di atas MAX_FILE_MB
app.config["MAX_CONTENT_LENGTH"] = int((MAX_FILE_MB + 1) * 1024 * 1024)

# === Load model sekali saat start ===
try:
    app.classifier = CatClassifier(
        onnx_path=os.getenv("ONNX_PATH", "models/mobilenetv3_small.onnx"),
        classes_path=os.getenv("CLASSES_PATH", "models/imagenet_classes.txt"),
        threshold=THRESHOLD,
        topk=TOPK,
    )
    MODEL_READY = True
    log.info("Model loaded: mobilenetv3_small.onnx")
except Exception as e:
    log.exception("MODEL INIT ERROR: %s", e)
    MODEL_READY = False

# ===== Request lifecycle: request_id + timing =====
@app.before_request
def _before():
    g.request_id = str(uuid.uuid4())[:8]
    g.t0 = time.time()

@app.after_request
def _after(resp):
    dur_ms = int((time.time() - getattr(g, "t0", time.time())) * 1000)
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "-")
    resp.headers["X-Response-Time-ms"] = str(dur_ms)
    log.info("%s %s %s %sms rid=%s",
             request.method, request.path, resp.status_code, dur_ms, getattr(g, "request_id", "-"))
    return resp

# ===== Error handlers (JSON) =====
@app.errorhandler(413)
def _too_large(e):
    return err("FILE_TOO_LARGE", f"Max {MAX_FILE_MB} MB", status=413)

@app.errorhandler(Exception)
def _unhandled(e):
    log.exception("UNHANDLED: %s", e)
    return err("INTERNAL_ERROR", "Unexpected error", status=500)

# Main Route
@app.get("/")
def home():
    return "API is Running"

# ===== Health =====
@app.get("/healthz")
def healthz():
    return ok({"model_loaded": MODEL_READY})

# ===== Predict =====
@app.post("/predict/image")
def predict_image():
    if not MODEL_READY:
        return err("MODEL_NOT_READY", "Model not loaded", status=503)

    if "file" not in request.files:
        return err("INVALID_FILE", "Expect form-data field 'file'")

    file = request.files["file"]
    is_valid, code, msg = validate_upload(file)
    if not is_valid:
        status = 415 if code == "UNSUPPORTED_MEDIA_TYPE" else 400
        return err(code, msg, status=status)

    # Baca bytes & verifikasi benar-benar gambar (Pillow)
    try:
        image_bytes = file.read()
        # verifikasi cepat tanpa menyimpan ke disk
        _img = Image.open(io.BytesIO(image_bytes))
        _img.verify()  # akan raise jika file bukan gambar valid
    except UnidentifiedImageError:
        return err("UNSUPPORTED_MEDIA_TYPE", "Not a valid image (Pillow) detected", status=415)
    except Exception as e:
        return err("INVALID_FILE", f"Failed to read image: {e}", status=400)

    try:
        res = app.classifier.predict(image_bytes)
        api_latency_ms = int((time.time() - g.t0) * 1000)
        return ok({
            "request_id": g.request_id,
            "label": res.label,
            "cat_prob": res.cat_prob,
            "threshold": res.threshold,
            "topk": res.topk,
            "meta": {**res.meta, "api_latency_ms": api_latency_ms}
        })
    except Exception as e:
        return err("INFERENCE_ERROR", str(e), status=500)

if __name__ == "__main__":
    app.run(debug=True)
