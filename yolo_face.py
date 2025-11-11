import base64, io, os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch


@dataclass
class FaceResult:
    ok: bool
    faces_count: int
    chosen_conf: float | None
    box: list[int] | None
    note: str
    kept_confs_ge_min: list[float]
    meta: Dict[str, Any]
    roi_b64: str | None


class CatFaceDetector:
    """
    Detektor wajah kucing (cat-head) ONNX via Ultralytics.
    Aturan:
      - faces_count > 1 hanya jika >= 2 box dengan conf >= HI_COUNT (default 0.75)
      - ROI dipilih dengan prioritas:
          1) conf >= HI_PRIORITY (0.80) → pilih conf tertinggi (tie-break area terbesar)
          2) kalau tidak ada, pilih dari conf >= MID_CONF (0.50)
          3) kalau tidak ada, pilih dari conf >= MIN_CONF (0.40)
    """

    def __init__(self,
                 onnx_path: str,
                 classes_path: str | None = None,
                 img_size: int = 768,
                 conf_raw: float = 0.20,
                 min_conf: float = 0.40,
                 mid_conf: float = 0.50,
                 hi_count: float = 0.75,
                 hi_priority: float = 0.80,
                 max_det: int = 5):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Cat-Head ONNX not found: {onnx_path}")
        self.model = YOLO(onnx_path)  # Ultralytics → onnxruntime backend otomatis
        self.classes_path = classes_path
        self.img_size = int(img_size)
        self.conf_raw = float(conf_raw)
        self.min_conf = float(min_conf)
        self.mid_conf = float(mid_conf)
        self.hi_count = float(hi_count)
        self.hi_priority = float(hi_priority)
        self.max_det = int(max_det)
        self.device = 0 if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
        # Decode via PIL → RGB → BGR (lebih robust terhadap beragam format)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]  # RGB->BGR
        return arr

    @staticmethod
    def _encode_jpeg_b64(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _pick_idx(self, confs: np.ndarray, xyxy: np.ndarray, mask: np.ndarray | list[int]):
        idxs = np.where(mask)[0] if isinstance(mask, np.ndarray) else np.array(mask, dtype=int)
        if len(idxs) == 0:
            return None
        best_conf = confs[idxs].max()
        best_idxs = idxs[confs[idxs] == best_conf]
        if len(best_idxs) == 1:
            return int(best_idxs[0])
        areas = (xyxy[best_idxs, 2] - xyxy[best_idxs, 0]) * (xyxy[best_idxs, 3] - xyxy[best_idxs, 1])
        return int(best_idxs[np.argmax(areas)])

    def detect(self, image_bytes: bytes, include_roi_b64: bool = True) -> FaceResult:
        bgr = self._bytes_to_bgr(image_bytes)

        results = self.model.predict(
            source=[bgr],
            conf=self.conf_raw,
            imgsz=self.img_size,
            max_det=self.max_det,
            verbose=False,
            device=self.device
        )
        r = results[0]

        # Tidak ada box sama sekali
        if r.boxes is None or len(r.boxes) == 0:
            return FaceResult(
                ok=False,
                faces_count=0,
                chosen_conf=None,
                box=None,
                note="No boxes",
                kept_confs_ge_min=[],
                meta={"img_size": self.img_size, "conf_raw": self.conf_raw},
                roi_b64=None
            )

        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)

        keep_min = confs >= self.min_conf
        if not keep_min.any():
            return FaceResult(
                ok=False,
                faces_count=0,
                chosen_conf=None,
                box=None,
                note=f"Tidak ada box >= {self.min_conf:.2f}",
                kept_confs_ge_min=[],
                meta={"img_size": self.img_size, "conf_raw": self.conf_raw},
                roi_b64=None
            )

        # Hitung jumlah wajah (aturan revisi)
        n_hi = int((confs >= self.hi_count).sum())
        faces_count = n_hi if n_hi >= 2 else 1

        # Pilih ROI (prioritas berjenjang)
        idx = self._pick_idx(confs, xyxy, confs >= self.hi_priority)
        if idx is None:
            idx = self._pick_idx(confs, xyxy, confs >= self.mid_conf)
        if idx is None:
            idx = self._pick_idx(confs, xyxy, confs >= self.min_conf)

        x1, y1, x2, y2 = map(int, xyxy[idx])
        roi_b64 = None
        if include_roi_b64:
            roi = r.orig_img[y1:y2, x1:x2]
            roi_b64 = self._encode_jpeg_b64(roi)

        note = "Dianggap 1 wajah (tidak ada >= 0.75 kedua)." if faces_count == 1 \
               else "Terdeteksi lebih dari 1 wajah (>= 0.75). ROI diambil dari box prioritas."

        return FaceResult(
            ok=True,
            faces_count=faces_count,
            chosen_conf=float(confs[idx]),
            box=[x1, y1, x2, y2],
            note=note,
            kept_confs_ge_min=confs[keep_min].tolist(),
            meta={
                "img_size": self.img_size,
                "conf_raw": self.conf_raw,
                "min_conf": self.min_conf,
                "mid_conf": self.mid_conf,
                "hi_count": self.hi_count,
                "hi_priority": self.hi_priority
            },
            roi_b64=roi_b64
        )
