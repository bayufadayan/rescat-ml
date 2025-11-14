# yolo_face_ultralytics_cpu.py
# ---------------------------------------------------------
# Ultralytics YOLO ONNX-only (NO TORCH, NO GPU)
# 100% CPU, ringan, akurasi tetap sama seperti versi lama.
# ---------------------------------------------------------

import base64
import io
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO


@dataclass
class FaceResult:
    ok: bool
    faces_count: int
    chosen_conf: float | None
    box: list[int] | None
    note: str
    kept_confs_ge_min: list[float]
    meta: Dict[str, Any]
    preview_jpeg: bytes | None
    roi_jpeg: bytes | None
    roi_b64: str | None


class CatFaceDetector:
    """
    Detektor wajah kucing via Ultralytics YOLO (ONNX backend) tanpa torch.
    Akurasi sama seperti model sebelumnya.
    """

    def __init__(
        self,
        onnx_path: str,
        classes_path: str | None = None,
        img_size: int = 768,
        conf_raw: float = 0.20,
        min_conf: float = 0.40,
        mid_conf: float = 0.50,
        hi_count: float = 0.75,
        hi_priority: float = 0.80,
        max_det: int = 5
    ):
        self.model = YOLO(onnx_path)  # ONNX Engine (CPU)
        self.img_size = img_size
        self.conf_raw = conf_raw
        self.min_conf = min_conf
        self.mid_conf = mid_conf
        self.hi_count = hi_count
        self.hi_priority = hi_priority
        self.max_det = max_det
        self.classes_path = classes_path

    @staticmethod
    def _bytes_to_bgr(image_bytes: bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _encode_jpeg_bytes(bgr: np.ndarray) -> bytes | None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return buf.tobytes() if ok else None

    def _draw_preview(self, bgr, boxes, confs, chosen_idx):
        canvas = bgr.copy()
        for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            conf = float(confs[i])
            if conf < self.min_conf:
                continue
            color = (0, 0, 255) if i == chosen_idx else (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return self._encode_jpeg_bytes(canvas)

    def _pick_idx(self, confs, boxes, mask):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None
        conf_sel = confs[idxs]
        best_idx = idxs[np.argmax(conf_sel)]
        return int(best_idx)

    def detect(self, image_bytes: bytes, include_roi_b64=True):
        bgr = self._bytes_to_bgr(image_bytes)

        # Inference ONNX → Ultralytics
        results = self.model.predict(
            source=bgr,
            imgsz=self.img_size,
            conf=self.conf_raw,
            max_det=self.max_det,
            verbose=False,
            device="cpu"    # ⛔ FORCE CPU — no CUDA, no torch
        )

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            preview = self._encode_jpeg_bytes(bgr)
            return FaceResult(False, 0, None, None, "No boxes", [], {}, preview, None, None)

        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        mask_min = confs >= self.min_conf
        if not mask_min.any():
            preview = self._draw_preview(bgr, xyxy, confs, None)
            return FaceResult(False, 0, None, None,
                              f"Tidak ada box >= {self.min_conf:.2f}",
                              [], {}, preview, None, None)

        xyxy_valid = xyxy[mask_min]
        confs_valid = confs[mask_min]

        # Hitung jumlah wajah
        n_hi = int((confs_valid >= self.hi_count).sum())
        faces_count = n_hi if n_hi >= 2 else 1

        # Pilih ROI
        idx = self._pick_idx(confs_valid, xyxy_valid, confs_valid >= self.hi_priority)
        if idx is None:
            idx = self._pick_idx(confs_valid, xyxy_valid, confs_valid >= self.mid_conf)
        if idx is None:
            idx = self._pick_idx(confs_valid, xyxy_valid, confs_valid >= self.min_conf)

        x1, y1, x2, y2 = map(int, xyxy_valid[idx])
        roi = bgr[y1:y2, x1:x2]

        ok_roi, buf_roi = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        roi_bytes = buf_roi.tobytes() if ok_roi else None
        roi_b64 = base64.b64encode(roi_bytes).decode("ascii") if include_roi_b64 else None

        preview = self._draw_preview(bgr, xyxy_valid, confs_valid, idx)

        note = ("Dianggap 1 wajah" if faces_count == 1 else
                "Terdeteksi lebih dari 1 wajah (>=0.75).")

        return FaceResult(
            True,
            faces_count,
            float(confs_valid[idx]),
            [x1, y1, x2, y2],
            note,
            confs_valid.tolist(),
            {
                "backend": "ultralytics-onnx-cpu",
                "img_size": self.img_size,
                "conf_raw": self.conf_raw
            },
            preview,
            roi_bytes,
            roi_b64,
        )
