import os, uuid, time
from flask import Flask, request
from dotenv import load_dotenv

from inference import CatClassifier
from utils.validation import validate_upload, MAX_FILE_MB
from utils.responses import ok, err

load_dotenv()

# === Konfigurasi dasar ===
THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))
TOPK      = int(os.getenv("TOPK", "3"))
PORT      = int(os.getenv("PORT", "8000"))

app = Flask(__name__)
# Batasi request body (hard cap) – sedikit lebih besar dari MAX_FILE_MB
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
except Exception as e:
    # Jangan crash: tetap hidup, tapi /healthz akan bilang false
    print(f"[MODEL INIT ERROR] {e}")
    MODEL_READY = False

@app.get("/healthz")
def healthz():
    return ok({"model_loaded": MODEL_READY})

@app.post("/predict/image")
def predict_image():
    rid = str(uuid.uuid4())[:8]
    t0 = time.time()

    if not MODEL_READY:
        return err("MODEL_NOT_READY", "Model not loaded", status=503)

    if "file" not in request.files:
        return err("INVALID_FILE", "Expect form-data field 'file'")

    file = request.files["file"]
    is_valid, code, msg = validate_upload(file)
    if not is_valid:
        status = 415 if code == "UNSUPPORTED_MEDIA_TYPE" else 400
        return err(code, msg, status=status)

    try:
        image_bytes = file.read()
        res = app.classifier.predict(image_bytes)
        latency_ms = int((time.time() - t0) * 1000)

        return ok({
            "request_id": rid,
            "label": res.label,
            "cat_prob": res.cat_prob,
            "threshold": res.threshold,
            "topk": res.topk,
            "meta": {**res.meta, "api_latency_ms": latency_ms}
        })
    except Exception as e:
        return err("INFERENCE_ERROR", str(e), status=500)

if __name__ == "__main__":
    # Dev run (use gunicorn in prod)
    app.run(host="0.0.0.0", port=PORT, debug=True)
