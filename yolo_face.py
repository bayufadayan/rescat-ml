"""Cat face detector using YOLO model."""

import base64
import io
import os
from dataclasses import dataclass
from typing import Dict, Any

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
    preview_jpeg: bytes | None
    roi_jpeg: bytes | None
    roi_b64: str | None


class CatFaceDetector:
    """YOLO-based cat face detector with smart ROI selection."""

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
        max_det: int = 5,
    ):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Cat-Head ONNX not found: {onnx_path}")
        self.model = YOLO(onnx_path)
        self.classes_path = classes_path
        self.img_size = int(img_size)
        self.conf_raw = float(conf_raw)
        self.min_conf = float(min_conf)
        self.mid_conf = float(mid_conf)
        self.hi_count = float(hi_count)
        self.hi_priority = float(hi_priority)
        self.max_det = int(max_det)
        self.device = 0 if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to BGR numpy array."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]
        return arr

    @staticmethod
    def _encode_jpeg_b64(bgr: np.ndarray) -> str:
        """Encode BGR image to base64 JPEG string."""
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _encode_jpeg_bytes(bgr: np.ndarray) -> bytes | None:
        """Encode BGR image to JPEG bytes."""
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return buf.tobytes() if ok else None

    def _pick_idx(
        self, confs: np.ndarray, xyxy: np.ndarray, mask: np.ndarray | list[int]
    ):
        idxs = (
            np.where(mask)[0]
            if isinstance(mask, np.ndarray)
            else np.array(mask, dtype=int)
        )
        if len(idxs) == 0:
            return None
        best_conf = confs[idxs].max()
        best_idxs = idxs[confs[idxs] == best_conf]
        if len(best_idxs) == 1:
            return int(best_idxs[0])
        areas = (xyxy[best_idxs, 2] - xyxy[best_idxs, 0]) * (
            xyxy[best_idxs, 3] - xyxy[best_idxs, 1]
        )
        return int(best_idxs[np.argmax(areas)])

    def _draw_preview(
        self,
        bgr: np.ndarray,
        xyxy: np.ndarray,
        confs: np.ndarray,
        chosen_idx: int | None,
        min_conf: float,
    ) -> bytes | None:
        """Draw bounding boxes on image and return as JPEG bytes."""
        canvas = bgr.copy()
        for i, (x1, y1, x2, y2) in enumerate(xyxy.astype(int)):
            conf = float(confs[i])
            if conf < min_conf:
                continue
            color = (0, 255, 0)
            thickness = 2
            if chosen_idx is not None and i == int(chosen_idx):
                color = (0, 0, 255)
                thickness = 3
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            label = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return self._encode_jpeg_bytes(canvas)

    def detect(self, image_bytes: bytes, include_roi_b64: bool = True) -> FaceResult:
        """Detect cat faces and return structured result."""
        bgr = self._bytes_to_bgr(image_bytes)

        results = self.model.predict(
            source=[bgr],
            conf=self.conf_raw,
            imgsz=self.img_size,
            max_det=self.max_det,
            verbose=False,
            device=self.device,
        )
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            preview = self._encode_jpeg_bytes(bgr)
            return FaceResult(
                ok=False,
                faces_count=0,
                chosen_conf=None,
                box=None,
                note="No boxes",
                kept_confs_ge_min=[],
                meta={"img_size": self.img_size, "conf_raw": self.conf_raw},
                preview_jpeg=preview,
                roi_jpeg=None,
                roi_b64=None,
            )

        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)

        keep_min = confs >= self.min_conf
        if not keep_min.any():
            preview = self._draw_preview(bgr, xyxy, confs, None, self.min_conf)
            return FaceResult(
                ok=False,
                faces_count=0,
                chosen_conf=None,
                box=None,
                note=f"No boxes >= {self.min_conf:.2f}",
                kept_confs_ge_min=[],
                meta={"img_size": self.img_size, "conf_raw": self.conf_raw},
                preview_jpeg=preview,
                roi_jpeg=None,
                roi_b64=None,
            )

        n_hi = int((confs >= self.hi_count).sum())
        faces_count = n_hi if n_hi >= 2 else 1

        idx = self._pick_idx(confs, xyxy, confs >= self.hi_priority)
        if idx is None:
            idx = self._pick_idx(confs, xyxy, confs >= self.mid_conf)
        if idx is None:
            idx = self._pick_idx(confs, xyxy, confs >= self.min_conf)

        x1, y1, x2, y2 = map(int, xyxy[idx])
        roi_jpeg = None
        roi_b64 = None
        roi = r.orig_img[y1:y2, x1:x2]
        ok_roi, buf_roi = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok_roi:
            roi_bytes = buf_roi.tobytes()
            roi_jpeg = roi_bytes
            if include_roi_b64:
                roi_b64 = base64.b64encode(roi_bytes).decode("ascii")

        preview = self._draw_preview(bgr, xyxy, confs, idx, self.min_conf)

        note = (
            "Single face detected"
            if faces_count == 1
            else f"Multiple faces detected ({faces_count})"
        )

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
                "hi_priority": self.hi_priority,
            },
            preview_jpeg=preview,
            roi_jpeg=roi_jpeg,
            roi_b64=roi_b64,
        )
