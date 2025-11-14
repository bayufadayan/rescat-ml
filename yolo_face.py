# yolo_face.py (VERSI LITE, TANPA TORCH / ULTRALYTICS)

import base64
import io
import os
from dataclasses import dataclass
from typing import Dict, Any, List

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort


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
    Versi ringan: inference YOLO ONNX via onnxruntime CPU saja (tanpa torch/ultralytics).
    Output dan API disamakan dengan versi sebelumnya.
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
        max_det: int = 5,
    ):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Cat-Head ONNX not found: {onnx_path}")

        self.img_size = int(img_size)
        self.conf_raw = float(conf_raw)
        self.min_conf = float(min_conf)
        self.mid_conf = float(mid_conf)
        self.hi_count = float(hi_count)
        self.hi_priority = float(hi_priority)
        self.max_det = int(max_det)
        self.classes_path = classes_path

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    # ---------- util kecil ----------
    @staticmethod
    def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
        return arr

    @staticmethod
    def _encode_jpeg_bytes(bgr: np.ndarray) -> bytes | None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return buf.tobytes() if ok else None

    @staticmethod
    def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.float32)
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter_w = np.clip(x2 - x1, 0, None)
        inter_h = np.clip(y2 - y1, 0, None)
        inter = inter_w * inter_h
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter + 1e-6
        return inter / union

    def _nms_xyxy(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=int)
        idxs = scores.argsort()[::-1]
        keep: List[int] = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(int(i))
            if idxs.size == 1:
                break
            ious = self._iou_xyxy(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious < iou_thr]
        return np.array(keep, dtype=int)

    def _preprocess_for_onnx(self, bgr: np.ndarray):
        """Letterbox ke (img_size,img_size), return tensor + scale + pad + orig_size."""
        h0, w0 = bgr.shape[:2]
        size = self.img_size
        scale = min(size / w0, size / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x = (size - nw) // 2
        pad_y = (size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        img = canvas.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,H,W)
        return img, scale, pad_x, pad_y, (w0, h0), canvas

    def _draw_preview(
        self,
        bgr_orig: np.ndarray,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        chosen_idx: int | None,
    ) -> bytes | None:
        canvas = bgr_orig.copy()
        for i, box in enumerate(boxes_xyxy.astype(int)):
            conf = float(confs[i])
            if conf < self.min_conf:
                continue
            x1, y1, x2, y2 = box
            color = (0, 255, 0)
            thickness = 2
            if chosen_idx is not None and i == int(chosen_idx):
                color = (0, 0, 255)
                thickness = 3
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            label = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return self._encode_jpeg_bytes(canvas)

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

    # ---------- main detect ----------
    def detect(self, image_bytes: bytes, include_roi_b64: bool = True) -> FaceResult:
        bgr_orig = self._bytes_to_bgr(image_bytes)
        h0, w0 = bgr_orig.shape[:2]

        # preprocess untuk onnx
        inp, scale, pad_x, pad_y, (w0, h0), _ = self._preprocess_for_onnx(bgr_orig)

        outputs = self.session.run(None, {self.input_name: inp})[0]  # (1,C,N) atau (1,N,C)
        if outputs.ndim != 3:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")

        # jadi (N, C)
        if outputs.shape[1] < outputs.shape[2]:
            # (1, C, N) -> (N, C)
            preds = outputs[0].transpose(1, 0)
        else:
            # (1, N, C) -> (N, C)
            preds = outputs[0]

        if preds.shape[1] <= 4:
            raise ValueError(f"Unexpected channels in output: {preds.shape}")

        boxes = preds[:, :4]
        scores = preds[:, 4:]

        # untuk 1 class, scores.shape[1] == 1; untuk multi, ambil max class prob
        confs = scores.max(axis=1)

        # filter berdasarkan conf_raw
        mask_raw = confs >= self.conf_raw
        if not mask_raw.any():
            preview = self._encode_jpeg_bytes(bgr_orig)
            return FaceResult(
                ok=False,
                faces_count=0,
                chosen_conf=None,
                box=None,
                note=f"No boxes >= conf_raw {self.conf_raw:.2f}",
                kept_confs_ge_min=[],
                meta={"img_size": self.img_size, "conf_raw": self.conf_raw},
                preview_jpeg=preview,
                roi_jpeg=None,
                roi_b64=None,
            )

        boxes = boxes[mask_raw]
        confs = confs[mask_raw]

        # cx,cy,w,h -> x1,y1,x2,y2 di space input model
        x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2

        # balikkan letterbox -> koordinat original
        x1 = (x1 - pad_x) / scale
        x2 = (x2 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        y2 = (y2 - pad_y) / scale

        x1 = np.clip(x1, 0, w0 - 1)
        y1 = np.clip(y1, 0, h0 - 1)
        x2 = np.clip(x2, 0, w0 - 1)
        y2 = np.clip(y2, 0, h0 - 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS untuk kurangi duplikasi box
        keep_nms = self._nms_xyxy(boxes_xyxy, confs, iou_thr=0.45)
        boxes_xyxy = boxes_xyxy[keep_nms]
        confs = confs[keep_nms]

        if boxes_xyxy.shape[0] == 0:
            preview = self._encode_jpeg_bytes(bgr_orig)
            return FaceResult(
                ok=False,
                faces_count=0,
                chosen_conf=None,
                box=None,
                note="No boxes after NMS",
                kept_confs_ge_min=[],
                meta={"img_size": self.img_size, "conf_raw": self.conf_raw},
                preview_jpeg=preview,
                roi_jpeg=None,
                roi_b64=None,
            )

        # filter min_conf untuk definisi "wajah valid"
        keep_min = confs >= self.min_conf
        kept_confs = confs[keep_min].tolist()
        if not keep_min.any():
            preview = self._draw_preview(bgr_orig, boxes_xyxy, confs, None)
            return FaceResult(
                ok=False,
                faces_count=0,
                chosen_conf=None,
                box=None,
                note=f"Tidak ada box >= {self.min_conf:.2f}",
                kept_confs_ge_min=[],
                meta={"img_size": self.img_size, "conf_raw": self.conf_raw},
                preview_jpeg=preview,
                roi_jpeg=None,
                roi_b64=None,
            )

        boxes_valid = boxes_xyxy[keep_min]
        confs_valid = confs[keep_min]

        # hitung jumlah wajah
        n_hi = int((confs_valid >= self.hi_count).sum())
        faces_count = n_hi if n_hi >= 2 else 1

        # pilih ROI
        idx = self._pick_idx(confs_valid, boxes_valid, confs_valid >= self.hi_priority)
        if idx is None:
            idx = self._pick_idx(confs_valid, boxes_valid, confs_valid >= self.mid_conf)
        if idx is None:
            idx = self._pick_idx(confs_valid, boxes_valid, confs_valid >= self.min_conf)

        x1, y1, x2, y2 = map(int, boxes_valid[idx])
        roi = bgr_orig[y1:y2, x1:x2]
        roi_jpeg = None
        roi_b64 = None
        if roi.size > 0:
            ok_roi, buf_roi = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok_roi:
                roi_bytes = buf_roi.tobytes()
                roi_jpeg = roi_bytes
                if include_roi_b64:
                    roi_b64 = base64.b64encode(roi_bytes).decode("ascii")

        preview = self._draw_preview(bgr_orig, boxes_valid, confs_valid, idx)

        note = (
            "Dianggap 1 wajah (tidak ada >= 0.75 kedua)."
            if faces_count == 1
            else "Terdeteksi lebih dari 1 wajah (>= 0.75). ROI diambil dari box prioritas."
        )

        return FaceResult(
            ok=True,
            faces_count=faces_count,
            chosen_conf=float(confs_valid[idx]),
            box=[x1, y1, x2, y2],
            note=note,
            kept_confs_ge_min=kept_confs,
            meta={
                "img_size": self.img_size,
                "conf_raw": self.conf_raw,
                "min_conf": self.min_conf,
                "mid_conf": self.mid_conf,
                "hi_count": self.hi_count,
                "hi_priority": self.hi_priority,
                "backend": "onnxruntime-cpu",
            },
            preview_jpeg=preview,
            roi_jpeg=roi_jpeg,
            roi_b64=roi_b64,
        )
