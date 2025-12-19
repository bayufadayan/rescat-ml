# ResCat ML API Contract Documentation

**Service Name:** ResCat ML API  
**Version:** v1  
**Base URL:** `http://localhost:5000` (Development)  
**Production URL:** `https://ml.rescat.life` (if applicable)

---

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Response Format](#common-response-format)
4. [Error Codes](#error-codes)
5. [API Endpoints](#api-endpoints)
6. [Data Models](#data-models)
7. [Storage Buckets](#storage-buckets)
8. [Environment Configuration](#environment-configuration)

---

## Overview

ResCat ML API adalah service berbasis Flask untuk deteksi, klasifikasi, dan analisis wajah kucing menggunakan multiple AI models. Service ini menyediakan 5 endpoint utama untuk:
- Validasi dan klasifikasi gambar kucing
- Deteksi wajah kucing (face detection)
- Penghapusan background
- Deteksi landmark wajah kucing (mata, telinga, mulut)
- Klasifikasi area wajah dengan Grad-CAM visualization

---

## Authentication

**Current Version:** No authentication required  
API bersifat open untuk saat ini. Pastikan untuk menambahkan API key atau Bearer token authentication untuk production.

---

## Common Response Format

### Success Response
```json
{
  "ok": true,
  "data": {},
  "...additional_fields": "..."
}
```

### Error Response
```json
{
  "ok": false,
  "error": "ERROR_CODE",
  "message": "Human readable error message"
}
```

### Response Headers
Setiap response akan memiliki header berikut:
- `X-Request-ID`: Unique request identifier (8 characters)
- `X-Response-Time-ms`: Response time in milliseconds

---

## Error Codes

| Error Code | HTTP Status | Description |
|-----------|-------------|-------------|
| `FILE_TOO_LARGE` | 413 | File size exceeds maximum limit (default 8MB) |
| `INVALID_FILE` | 400 | File not provided or invalid |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | File is not a valid image format |
| `MODEL_NOT_READY` | 503 | ML model not loaded/initialized |
| `INFERENCE_ERROR` | 500 | Error during model inference |
| `UPLOAD_FAILED` | 502 | Failed to upload result to storage |
| `UPLOAD_ERROR` | 502 | Error during upload process |
| `UPLOAD_INVALID_RESPONSE` | 502 | Storage service returned invalid response |
| `FETCH_FAILED` | 400 | Failed to fetch image from URL |
| `FETCH_URL_ERROR` | 400 | Error while downloading image from URL |
| `REMOVE_BG_ERROR` | 500 | Background removal failed |
| `VALIDATION_ERROR` | 400 | Input validation failed |
| `IMAGE_DECODE_ERROR` | 400 | Cannot decode image |
| `MISSING_INPUT` | 400 | Required input parameter missing |
| `INVALID_URL` | 400 | Invalid or empty URL provided |
| `INVALID_REQUEST` | 400 | Request format invalid |
| `MISSING_AREAS` | 400 | Missing required area parameters |
| `INVALID_IMAGE` | 400 | Image validation failed |
| `PROCESSING_ERROR` | 500 | General processing error |
| `CLASSIFIER_NOT_READY` | 503 | Area classifier not loaded |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## API Endpoints

### 1. Health Check

#### `GET /healthz`

Check service health and model availability.

**Request:**
```http
GET /healthz HTTP/1.1
```

**Response:**
```json
{
  "ok": true,
  "recognizer_loaded": true,
  "face_loaded": true,
  "landmark_loaded": true,
  "classifier_loaded": true
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Always true for success |
| `recognizer_loaded` | boolean | Cat breed classifier model status |
| `face_loaded` | boolean | Face detector model status |
| `landmark_loaded` | boolean | Landmark detector model status |
| `classifier_loaded` | boolean | Area classifier model status |

---

### 2. Root & Version Info

#### `GET /` or `GET /v1`

Display service information page (HTML).

**Request:**
```http
GET /v1 HTTP/1.1
```

**Response:**
HTML page with service status and model availability.

---

### 3. Cat Recognition

#### `POST /v1/cat/recognize`

Melakukan validasi gambar, klasifikasi breed kucing, dan deteksi wajah sekaligus.

**Request:**
```http
POST /v1/cat/recognize HTTP/1.1
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="file"; filename="cat.jpg"
Content-Type: image/jpeg

[binary image data]
--boundary--
```

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file (JPEG, PNG, WebP) max 8MB |

**Success Response:**
```json
{
  "ok": true,
  "request_id": "a1b2c3d4",
  "label": "cat",
  "cat_prob": 0.9823,
  "threshold": 0.50,
  "topk": 3,
  "meta": {
    "predictions": [
      ["n02123045 tabby", 0.45],
      ["n02124075 Egyptian_cat", 0.32],
      ["n02123159 tiger_cat", 0.15]
    ],
    "input_shape": [1, 3, 224, 224],
    "inference_time_ms": 45,
    "api_latency_ms": 123
  },
  "faces": {
    "ok": true,
    "faces_count": 1,
    "chosen_conf": 0.87,
    "box": [120, 85, 340, 305],
    "note": "Selected face with highest confidence",
    "kept_confs_ge_min": [0.87],
    "meta": {
      "total_raw_detections": 1,
      "after_nms": 1,
      "inference_time_ms": 67
    },
    "preview": {
      "id": "file_abc123",
      "filename": "1703012345678-a1b2c3d4-preview.jpg",
      "url": "https://storage.rescat.life/preview-bounding-box/1703012345678-a1b2c3d4-preview.jpg"
    },
    "roi": {
      "id": "file_def456",
      "filename": "1703012345678-a1b2c3d4-roi.jpg",
      "url": "https://storage.rescat.life/roi-face-cat/1703012345678-a1b2c3d4-roi.jpg"
    },
    "preview_error": null,
    "roi_error": null
  }
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Success status |
| `request_id` | string | Unique request identifier |
| `label` | string | Classification label ("cat" or ImageNet class) |
| `cat_prob` | float | Probability that image contains a cat |
| `threshold` | float | Classification threshold used |
| `topk` | integer | Number of top predictions |
| `meta.predictions` | array | Top-K predictions with scores |
| `meta.input_shape` | array | Model input tensor shape |
| `meta.inference_time_ms` | integer | Model inference duration |
| `meta.api_latency_ms` | integer | Total API processing time |
| `faces.ok` | boolean | Face detection success status |
| `faces.faces_count` | integer | Number of faces detected |
| `faces.chosen_conf` | float | Confidence of selected face |
| `faces.box` | array | Bounding box [x1, y1, x2, y2] |
| `faces.note` | string | Selection note |
| `faces.kept_confs_ge_min` | array | Confidence scores above min threshold |
| `faces.preview` | object | Uploaded preview image with bounding box |
| `faces.roi` | object | Uploaded cropped face ROI image |

**Error Responses:**

*Model Not Ready:*
```json
{
  "ok": false,
  "error": "MODEL_NOT_READY",
  "message": "Classifier not loaded"
}
```

*Invalid File:*
```json
{
  "ok": false,
  "error": "INVALID_FILE",
  "message": "Expect form-data field 'file'. Found: []"
}
```

*File Too Large:*
```json
{
  "ok": false,
  "error": "FILE_TOO_LARGE",
  "message": "Max 8 MB"
}
```

---

### 4. Face Detection

#### `POST /v1/cat/faces`

Deteksi wajah kucing pada gambar dan upload hasil ke storage.

**Request:**
```http
POST /v1/cat/faces HTTP/1.1
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="file"; filename="cat.jpg"
Content-Type: image/jpeg

[binary image data]
--boundary--
```

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file (JPEG, PNG, WebP) max 8MB |

**Success Response:**
```json
{
  "ok": true,
  "request_id": "b2c3d4e5",
  "faces_count": 2,
  "chosen_conf": 0.92,
  "box": [150, 120, 380, 350],
  "note": "Selected face with highest confidence",
  "kept_confs_ge_min": [0.92, 0.76],
  "meta": {
    "total_raw_detections": 3,
    "after_nms": 2,
    "inference_time_ms": 54
  },
  "preview": {
    "id": "file_xyz789",
    "filename": "1703012345678-b2c3d4e5-preview.jpg",
    "url": "https://storage.rescat.life/preview-bounding-box/1703012345678-b2c3d4e5-preview.jpg"
  },
  "roi": {
    "id": "file_uvw456",
    "filename": "1703012345678-b2c3d4e5-roi.jpg",
    "url": "https://storage.rescat.life/roi-face-cat/1703012345678-b2c3d4e5-roi.jpg"
  },
  "preview_error": null,
  "roi_error": null
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Success status |
| `request_id` | string | Unique request identifier |
| `faces_count` | integer | Number of cat faces detected |
| `chosen_conf` | float | Confidence score of chosen face |
| `box` | array[int] | Bounding box coordinates [x1, y1, x2, y2] |
| `note` | string | Selection algorithm note |
| `kept_confs_ge_min` | array[float] | All confidence scores >= MIN_CONF |
| `meta.total_raw_detections` | integer | Raw detections before filtering |
| `meta.after_nms` | integer | Detections after NMS |
| `meta.inference_time_ms` | integer | Model inference time |
| `preview.id` | string | Storage file ID for preview image |
| `preview.filename` | string | Filename in storage |
| `preview.url` | string | Public URL for preview image |
| `roi.id` | string | Storage file ID for ROI image |
| `roi.filename` | string | Filename in storage |
| `roi.url` | string | Public URL for ROI image |

**Error Responses:**

*Model Not Ready:*
```json
{
  "ok": false,
  "error": "MODEL_NOT_READY",
  "message": "Face detector not loaded"
}
```

---

### 5. Background Removal

#### `POST /v1/cat/remove-bg`

Menghilangkan background dari gambar kucing menggunakan rembg library.

**Request Option 1: Multipart Form Data**
```http
POST /v1/cat/remove-bg HTTP/1.1
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="file"; filename="cat.jpg"
Content-Type: image/jpeg

[binary image data]
--boundary--
```

**Request Option 2: JSON with URL**
```http
POST /v1/cat/remove-bg HTTP/1.1
Content-Type: application/json

{
  "url": "https://example.com/cat.jpg"
}
```

**Request Option 3: Form Data with URL**
```http
POST /v1/cat/remove-bg HTTP/1.1
Content-Type: application/x-www-form-urlencoded

url=https://example.com/cat.jpg
```

**Request Option 4: Query Parameter**
```http
POST /v1/cat/remove-bg?url=https://example.com/cat.jpg HTTP/1.1
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Conditional | Image file (if not using URL) |
| `url` | string | Conditional | Image URL (if not using file) |

**Success Response (Cached):**
```json
{
  "ok": true,
  "id": "file_removed123",
  "bucket": "remove-bg",
  "url": "https://storage.rescat.life/remove-bg/1703012345678-c3d4e5f6.png",
  "filename": "1703012345678-c3d4e5f6.png",
  "hash": "64f44363fd9e3c16aa19c69e113ba3c04c9e5d0566bcd3b3653f2f819441eba3",
  "cached": true
}
```

**Success Response (New):**
```json
{
  "ok": true,
  "id": "file_removed456",
  "bucket": "remove-bg",
  "url": "https://storage.rescat.life/remove-bg/1703012345789-d4e5f6g7.png",
  "filename": "1703012345789-d4e5f6g7.png",
  "hash": "74f44363fd9e3c16aa19c69e113ba3c04c9e5d0566bcd3b3653f2f819441eba3",
  "cached": false
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Success status |
| `id` | string | Storage file ID |
| `bucket` | string | Storage bucket name |
| `url` | string | Public URL of processed image |
| `filename` | string | Filename in storage |
| `hash` | string | SHA256 hash of original image |
| `cached` | boolean | Whether result was from cache |

**Cache System:**
- Images are cached based on SHA256 hash
- Cache directory: `cache/remove-bg/`
- Cache files: `{hash}.json`

**Error Responses:**

*Invalid Input:*
```json
{
  "ok": false,
  "error": "INVALID_INPUT",
  "message": "Provide either form-data 'file' or 'url' (JSON/form/query)"
}
```

*Fetch Failed:*
```json
{
  "ok": false,
  "error": "FETCH_FAILED",
  "message": "Failed to fetch image from URL (status 404)"
}
```

*Remove BG Error:*
```json
{
  "ok": false,
  "error": "REMOVE_BG_ERROR",
  "message": "Failed to remove background: {error_details}"
}
```

---

### 6. Landmark Detection

#### `POST /v1/cat/landmark`

Mendeteksi landmark wajah kucing (mata, telinga, mulut) dan melakukan cropping pada setiap area.

**Request Option 1: Multipart Form Data**
```http
POST /v1/cat/landmark HTTP/1.1
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="file"; filename="cat.jpg"
Content-Type: image/jpeg

[binary image data]
--boundary--
```

**Request Option 2: JSON with URL**
```http
POST /v1/cat/landmark HTTP/1.1
Content-Type: application/json

{
  "file": "https://storage.rescat.life/roi-face-cat/cat-face.jpg"
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file/string | Yes | Image file or URL to cat face image |

**Success Response:**
```json
{
  "ok": true,
  "landmark": {
    "img_landmark_id": "file_landmarked123",
    "img_landmark_url": "https://storage.rescat.life/landmarked_face/landmarked_1703012345678-e5f6g7h8.jpg"
  },
  "right_eye": {
    "img_right_eye_id": "file_righteye123",
    "img_right_eye_url": "https://storage.rescat.life/right_eye_crop/1703012345678-e5f6g7h8.jpg"
  },
  "left_eye": {
    "img_left_eye_id": "file_lefteye123",
    "img_left_eye_url": "https://storage.rescat.life/left_eye_crop/1703012345678-e5f6g7h8.jpg"
  },
  "mouth": {
    "img_mouth_id": "file_mouth123",
    "img_mouth_url": "https://storage.rescat.life/mouth_crop/1703012345678-e5f6g7h8.jpg"
  },
  "right_ear": {
    "img_right_ear_id": "file_rightear123",
    "img_right_ear_url": "https://storage.rescat.life/right_ear_crop/1703012345678-e5f6g7h8.jpg"
  },
  "left_ear": {
    "img_left_ear_id": "file_leftear123",
    "img_left_ear_url": "https://storage.rescat.life/left_ear_crop/1703012345678-e5f6g7h8.jpg"
  }
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Success status |
| `landmark.img_landmark_id` | string | File ID of annotated face with landmarks |
| `landmark.img_landmark_url` | string | Public URL of annotated face |
| `right_eye.img_right_eye_id` | string | File ID of right eye crop |
| `right_eye.img_right_eye_url` | string | Public URL of right eye crop |
| `left_eye.img_left_eye_id` | string | File ID of left eye crop |
| `left_eye.img_left_eye_url` | string | Public URL of left eye crop |
| `mouth.img_mouth_id` | string | File ID of mouth crop |
| `mouth.img_mouth_url` | string | Public URL of mouth crop |
| `right_ear.img_right_ear_id` | string | File ID of right ear crop |
| `right_ear.img_right_ear_url` | string | Public URL of right ear crop |
| `left_ear.img_left_ear_id` | string | File ID of left ear crop |
| `left_ear.img_left_ear_url` | string | Public URL of left ear crop |

**Landmark Points Detected:**
- Right Eye (2 points): outer corner, inner corner
- Left Eye (2 points): inner corner, outer corner  
- Mouth (2 points): left corner, right corner
- Right Ear (3 points): base, middle, tip
- Left Ear (3 points): base, middle, tip

**Error Responses:**

*Model Not Ready:*
```json
{
  "ok": false,
  "error": "MODEL_NOT_READY",
  "message": "Landmark detector not initialized"
}
```

*Missing Input:*
```json
{
  "ok": false,
  "error": "MISSING_INPUT",
  "message": "Provide 'file' in form-data or JSON with image URL"
}
```

*Validation Error:*
```json
{
  "ok": false,
  "error": "VALIDATION_ERROR",
  "message": "No face detected in image"
}
```

---

### 7. Area Classification with Grad-CAM

#### `POST /v1/cat/area-check`

Mengklasifikasikan setiap area wajah kucing (normal/abnormal) dan menghasilkan Grad-CAM visualization untuk interpretasi model.

**Request:**
```http
POST /v1/cat/area-check HTTP/1.1
Content-Type: application/json

{
  "right_eye": "https://storage.rescat.life/right_eye_crop/image1.jpg",
  "left_eye": "https://storage.rescat.life/left_eye_crop/image2.jpg",
  "mouth": "https://storage.rescat.life/mouth_crop/image3.jpg",
  "right_ear": "https://storage.rescat.life/right_ear_crop/image4.jpg",
  "left_ear": "https://storage.rescat.life/left_ear_crop/image5.jpg"
}
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `right_eye` | string | Yes | URL to right eye cropped image |
| `left_eye` | string | Yes | URL to left eye cropped image |
| `mouth` | string | Yes | URL to mouth cropped image |
| `right_ear` | string | Yes | URL to right ear cropped image |
| `left_ear` | string | Yes | URL to left ear cropped image |

**Success Response:**
```json
{
  "ok": true,
  "classification": {
    "right_eye": {
      "label": "normal",
      "confidence": 0.9523
    },
    "left_eye": {
      "label": "abnormal",
      "confidence": 0.8734
    },
    "mouth": {
      "label": "normal",
      "confidence": 0.9102
    },
    "right_ear": {
      "label": "abnormal",
      "confidence": 0.7891
    },
    "left_ear": {
      "label": "normal",
      "confidence": 0.8967
    }
  },
  "gradcam": {
    "right_eye": {
      "img_right_eye_gradcam_id": "file_gradcam_re123",
      "img_right_eye_gradcam_url": "https://storage.rescat.life/right_eye_gradcam/1703012345678-f6g7h8i9.jpg"
    },
    "left_eye": {
      "img_left_eye_gradcam_id": "file_gradcam_le123",
      "img_left_eye_gradcam_url": "https://storage.rescat.life/left_eye_gradcam/1703012345678-f6g7h8i9.jpg"
    },
    "mouth": {
      "img_mouth_gradcam_id": "file_gradcam_m123",
      "img_mouth_gradcam_url": "https://storage.rescat.life/mouth_gradcam/1703012345678-f6g7h8i9.jpg"
    },
    "right_ear": {
      "img_right_ear_gradcam_id": "file_gradcam_rr123",
      "img_right_ear_gradcam_url": "https://storage.rescat.life/right_ear_gradcam/1703012345678-f6g7h8i9.jpg"
    },
    "left_ear": {
      "img_left_ear_gradcam_id": "file_gradcam_lr123",
      "img_left_ear_gradcam_url": "https://storage.rescat.life/left_ear_gradcam/1703012345678-f6g7h8i9.jpg"
    }
  }
}
```

**Response Fields:**

*Classification Object:*
| Field | Type | Description |
|-------|------|-------------|
| `{area}.label` | string | Classification result: "normal" or "abnormal" |
| `{area}.confidence` | float | Model confidence score (0.0 - 1.0) |

*Grad-CAM Object:*
| Field | Type | Description |
|-------|------|-------------|
| `{area}.img_{area}_gradcam_id` | string | File ID of Grad-CAM visualization |
| `{area}.img_{area}_gradcam_url` | string | Public URL of Grad-CAM image |

**Grad-CAM Explanation:**
Grad-CAM (Gradient-weighted Class Activation Mapping) adalah visualization technique yang menampilkan area mana pada gambar yang paling berpengaruh dalam keputusan klasifikasi model. Semakin merah/panas area pada heatmap, semakin besar pengaruhnya.

**Error Responses:**

*Classifier Not Ready:*
```json
{
  "ok": false,
  "error": "CLASSIFIER_NOT_READY",
  "message": "Area classifier not loaded"
}
```

*Invalid Request:*
```json
{
  "ok": false,
  "error": "INVALID_REQUEST",
  "message": "Request must be JSON"
}
```

*Missing Areas:*
```json
{
  "ok": false,
  "error": "MISSING_AREAS",
  "message": "Missing areas: right_ear, left_ear"
}
```

*Fetch Error:*
```json
{
  "ok": false,
  "error": "FETCH_URL_ERROR",
  "message": "Failed to fetch right_eye from URL: Connection timeout"
}
```

---

## Data Models

### 1. Face Detection Result

```typescript
interface FaceDetectionResult {
  ok: boolean;
  faces_count: number;
  chosen_conf: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  note: string;
  kept_confs_ge_min: number[];
  meta: {
    total_raw_detections: number;
    after_nms: number;
    inference_time_ms: number;
  };
  preview?: UploadedImage;
  roi?: UploadedImage;
  preview_error?: string | null;
  roi_error?: string | null;
}
```

### 2. Uploaded Image

```typescript
interface UploadedImage {
  id: string;
  filename: string;
  url: string;
}
```

### 3. Classification Result

```typescript
interface ClassificationResult {
  ok: boolean;
  request_id: string;
  label: string;
  cat_prob: number;
  threshold: number;
  topk: number;
  meta: {
    predictions: Array<[string, number]>;
    input_shape: number[];
    inference_time_ms: number;
    api_latency_ms: number;
  };
  faces?: FaceDetectionResult | null;
}
```

### 4. Background Removal Result

```typescript
interface BackgroundRemovalResult {
  ok: boolean;
  id: string;
  bucket: string;
  url: string;
  filename: string;
  hash: string;
  cached: boolean;
}
```

### 5. Landmark Detection Result

```typescript
interface LandmarkDetectionResult {
  ok: boolean;
  landmark: {
    img_landmark_id: string;
    img_landmark_url: string;
  };
  right_eye: {
    img_right_eye_id: string;
    img_right_eye_url: string;
  };
  left_eye: {
    img_left_eye_id: string;
    img_left_eye_url: string;
  };
  mouth: {
    img_mouth_id: string;
    img_mouth_url: string;
  };
  right_ear: {
    img_right_ear_id: string;
    img_right_ear_url: string;
  };
  left_ear: {
    img_left_ear_id: string;
    img_left_ear_url: string;
  };
}
```

### 6. Area Classification Result

```typescript
interface AreaClassification {
  label: "normal" | "abnormal";
  confidence: number;
}

interface GradCAMImage {
  [key: `img_${string}_gradcam_id`]: string;
  [key: `img_${string}_gradcam_url`]: string;
}

interface AreaCheckResult {
  ok: boolean;
  classification: {
    right_eye: AreaClassification;
    left_eye: AreaClassification;
    mouth: AreaClassification;
    right_ear: AreaClassification;
    left_ear: AreaClassification;
  };
  gradcam: {
    right_eye: GradCAMImage;
    left_eye: GradCAMImage;
    mouth: GradCAMImage;
    right_ear: GradCAMImage;
    left_ear: GradCAMImage;
  };
}
```

### 7. Error Response

```typescript
interface ErrorResponse {
  ok: false;
  error: string;
  message: string;
  [key: string]: any; // Additional error fields
}
```

---

## Storage Buckets

Service ini menggunakan external storage API untuk menyimpan hasil processing. Berikut adalah bucket-bucket yang digunakan:

| Bucket Name | Purpose | Content Type |
|------------|---------|--------------|
| `preview-bounding-box` | Face detection preview with bounding box | image/jpeg |
| `roi-face-cat` | Cropped face region of interest | image/jpeg |
| `remove-bg` | Images with background removed | image/png |
| `right_eye_crop` | Cropped right eye region | image/jpeg |
| `left_eye_crop` | Cropped left eye region | image/jpeg |
| `mouth_crop` | Cropped mouth region | image/jpeg |
| `right_ear_crop` | Cropped right ear region | image/jpeg |
| `left_ear_crop` | Cropped left ear region | image/jpeg |
| `landmarked_face` | Face with landmark annotations | image/jpeg |
| `right_eye_gradcam` | Right eye Grad-CAM visualization | image/jpeg |
| `left_eye_gradcam` | Left eye Grad-CAM visualization | image/jpeg |
| `mouth_gradcam` | Mouth Grad-CAM visualization | image/jpeg |
| `right_ear_gradcam` | Right ear Grad-CAM visualization | image/jpeg |
| `left_ear_gradcam` | Left ear Grad-CAM visualization | image/jpeg |

**Storage API:**
- Base URL: `https://storage.rescat.life`
- Upload Endpoint: `POST /api/files`
- Upload Method: multipart/form-data
- Required Fields: `file`, `bucket`

---

## Environment Configuration

### Environment Variables

```bash
# Model Configuration
THRESHOLD=0.50                    # Cat classification threshold
TOPK=3                           # Number of top predictions to return
PORT=5000                        # Server port
MAX_FILE_MB=8                    # Maximum upload file size in MB
CORS_ENABLED=false               # Enable CORS

# Model Paths - Validation/Recognition
ONNX_PATH=models/validation_model/mobilenetv3_small.onnx
CLASSES_PATH=models/validation_model/imagenet_classes.txt

# Model Paths - Face Detection
CAT_HEAD_ONNX=models/validation_model/cat_head_model.onnx
CAT_HEAD_CLASSES=models/validation_model/cat_head_classes.json
IMG_SIZE=640                     # YOLO input image size
CONF_RAW=0.20                    # Raw detection confidence threshold
MIN_CONF=0.40                    # Minimum confidence to keep
MID_CONF=0.50                    # Mid confidence threshold
HI_COUNT=0.75                    # High count threshold
HI_PRIORITY=0.80                 # High priority threshold
MAX_DET=5                        # Maximum detections

# Model Paths - Landmark Detection
LANDMARK_BBOX_ONNX=models/landmark_model/frederic_bbox.onnx
LANDMARK_ONNX=models/landmark_model/frederic_landmarks.onnx

# Model Paths - Area Classification
EYE_MODEL_PATH=models/classification_model/efficientnet_cat_eye_model_b1.h5
EAR_MODEL_PATH=models/classification_model/efficientnet_cat_ear_model_b1.h5
MOUTH_MODEL_PATH=models/classification_model/efficientnet_cat_mouth_model_b1.h5

# Storage Configuration
BUCKET_PREVIEW=preview-bounding-box
BUCKET_ROI=roi-face-cat
BUCKET_LANDMARK=landmark-crops
BUCKET_LANDMARKED_FACE=landmarked_face

# Storage API Configuration
CONTENT_API_BASE=https://storage.rescat.life
CONTENT_API_TIMEOUT_CONNECT=5    # Connection timeout in seconds
CONTENT_API_TIMEOUT_READ=30      # Read timeout in seconds

# Cache Configuration
REMOVEBG_CACHE_DIR=cache/remove-bg
```

---

## Database Tables (Recommended)

Meskipun ML service ini tidak menggunakan database langsung, berikut adalah rekomendasi struktur database untuk sistem lengkap:

### Table: `detection_results`

Menyimpan hasil deteksi dan klasifikasi.

```sql
CREATE TABLE detection_results (
    id VARCHAR(36) PRIMARY KEY,
    request_id VARCHAR(8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Classification
    label VARCHAR(100),
    cat_prob DECIMAL(5,4),
    threshold DECIMAL(3,2),
    
    -- Face Detection
    faces_count INT DEFAULT 0,
    chosen_conf DECIMAL(5,4),
    box_x1 INT,
    box_y1 INT,
    box_x2 INT,
    box_y2 INT,
    
    -- Uploaded Images
    preview_id VARCHAR(100),
    preview_url TEXT,
    roi_id VARCHAR(100),
    roi_url TEXT,
    
    -- Metadata
    inference_time_ms INT,
    api_latency_ms INT,
    
    INDEX idx_request_id (request_id),
    INDEX idx_created_at (created_at)
);
```

### Table: `landmark_results`

Menyimpan hasil deteksi landmark.

```sql
CREATE TABLE landmark_results (
    id VARCHAR(36) PRIMARY KEY,
    request_id VARCHAR(8) NOT NULL,
    detection_result_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Landmark Image
    landmark_id VARCHAR(100),
    landmark_url TEXT,
    
    -- Right Eye
    right_eye_id VARCHAR(100),
    right_eye_url TEXT,
    
    -- Left Eye
    left_eye_id VARCHAR(100),
    left_eye_url TEXT,
    
    -- Mouth
    mouth_id VARCHAR(100),
    mouth_url TEXT,
    
    -- Right Ear
    right_ear_id VARCHAR(100),
    right_ear_url TEXT,
    
    -- Left Ear
    left_ear_id VARCHAR(100),
    left_ear_url TEXT,
    
    FOREIGN KEY (detection_result_id) REFERENCES detection_results(id),
    INDEX idx_request_id (request_id),
    INDEX idx_detection_result_id (detection_result_id)
);
```

### Table: `area_classifications`

Menyimpan hasil klasifikasi area dan Grad-CAM.

```sql
CREATE TABLE area_classifications (
    id VARCHAR(36) PRIMARY KEY,
    request_id VARCHAR(8) NOT NULL,
    landmark_result_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Right Eye
    right_eye_label VARCHAR(20),
    right_eye_confidence DECIMAL(5,4),
    right_eye_gradcam_id VARCHAR(100),
    right_eye_gradcam_url TEXT,
    
    -- Left Eye
    left_eye_label VARCHAR(20),
    left_eye_confidence DECIMAL(5,4),
    left_eye_gradcam_id VARCHAR(100),
    left_eye_gradcam_url TEXT,
    
    -- Mouth
    mouth_label VARCHAR(20),
    mouth_confidence DECIMAL(5,4),
    mouth_gradcam_id VARCHAR(100),
    mouth_gradcam_url TEXT,
    
    -- Right Ear
    right_ear_label VARCHAR(20),
    right_ear_confidence DECIMAL(5,4),
    right_ear_gradcam_id VARCHAR(100),
    right_ear_gradcam_url TEXT,
    
    -- Left Ear
    left_ear_label VARCHAR(20),
    left_ear_confidence DECIMAL(5,4),
    left_ear_gradcam_id VARCHAR(100),
    left_ear_gradcam_url TEXT,
    
    FOREIGN KEY (landmark_result_id) REFERENCES landmark_results(id),
    INDEX idx_request_id (request_id),
    INDEX idx_landmark_result_id (landmark_result_id)
);
```

### Table: `background_removals`

Menyimpan hasil background removal.

```sql
CREATE TABLE background_removals (
    id VARCHAR(36) PRIMARY KEY,
    request_id VARCHAR(8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Result
    file_id VARCHAR(100) NOT NULL,
    bucket VARCHAR(100) NOT NULL,
    url TEXT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    
    -- Cache Info
    image_hash VARCHAR(64) NOT NULL UNIQUE,
    cached BOOLEAN DEFAULT FALSE,
    
    INDEX idx_request_id (request_id),
    INDEX idx_image_hash (image_hash),
    INDEX idx_created_at (created_at)
);
```

### Table: `api_logs`

Menyimpan log semua request API.

```sql
CREATE TABLE api_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    request_id VARCHAR(8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Request Info
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    
    -- Response Info
    status_code INT NOT NULL,
    response_time_ms INT NOT NULL,
    
    -- Client Info
    user_agent TEXT,
    ip_address VARCHAR(45),
    
    -- Error Info (if applicable)
    error_code VARCHAR(50),
    error_message TEXT,
    
    INDEX idx_request_id (request_id),
    INDEX idx_endpoint (endpoint),
    INDEX idx_created_at (created_at),
    INDEX idx_status_code (status_code)
);
```

---

## API Flow Diagram

### Complete Cat Analysis Flow

```
┌─────────────────┐
│  Upload Image   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ POST /v1/cat/recognize  │ ◄── Step 1: Validate & Detect Face
└────────┬────────────────┘
         │
         ├──► cat_prob < threshold? ───► Return "Not a cat"
         │
         ▼
    cat_prob >= threshold
         │
         ├──► Face detected? ───► Get ROI image URL
         │
         ▼
┌─────────────────────────┐
│ POST /v1/cat/remove-bg  │ ◄── Step 2: Remove Background (Optional)
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ POST /v1/cat/landmark   │ ◄── Step 3: Detect Landmarks
└────────┬────────────────┘
         │
         ├──► Get cropped areas:
         │    - right_eye_url
         │    - left_eye_url
         │    - mouth_url
         │    - right_ear_url
         │    - left_ear_url
         │
         ▼
┌──────────────────────────┐
│ POST /v1/cat/area-check  │ ◄── Step 4: Classify Areas + Grad-CAM
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│   Final Results:         │
│   - Classification       │
│   - Grad-CAM Images      │
│   - Normal/Abnormal      │
└──────────────────────────┘
```

---

## Rate Limiting (Recommendation)

Untuk production, disarankan menambahkan rate limiting:

```python
# Example using Flask-Limiter
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)

@app.route("/v1/cat/recognize", methods=["POST"])
@limiter.limit("10 per minute")
def recognize_cat():
    # ...
```

**Recommended Limits:**
- `/v1/cat/recognize`: 10 req/min, 100 req/hour
- `/v1/cat/faces`: 15 req/min, 150 req/hour
- `/v1/cat/remove-bg`: 5 req/min, 50 req/hour (heavy operation)
- `/v1/cat/landmark`: 10 req/min, 100 req/hour
- `/v1/cat/area-check`: 5 req/min, 50 req/hour (heavy operation)

---

## Example Usage

### Complete Workflow with cURL

```bash
# Step 1: Recognize cat and detect face
curl -X POST http://localhost:5000/v1/cat/recognize \
  -F "file=@cat.jpg" \
  -o step1_result.json

# Extract ROI URL from response
ROI_URL=$(jq -r '.faces.roi.url' step1_result.json)

# Step 2: Detect landmarks
curl -X POST http://localhost:5000/v1/cat/landmark \
  -H "Content-Type: application/json" \
  -d "{\"file\": \"$ROI_URL\"}" \
  -o step2_result.json

# Extract area URLs
RIGHT_EYE=$(jq -r '.right_eye.img_right_eye_url' step2_result.json)
LEFT_EYE=$(jq -r '.left_eye.img_left_eye_url' step2_result.json)
MOUTH=$(jq -r '.mouth.img_mouth_url' step2_result.json)
RIGHT_EAR=$(jq -r '.right_ear.img_right_ear_url' step2_result.json)
LEFT_EAR=$(jq -r '.left_ear.img_left_ear_url' step2_result.json)

# Step 3: Classify areas with Grad-CAM
curl -X POST http://localhost:5000/v1/cat/area-check \
  -H "Content-Type: application/json" \
  -d "{
    \"right_eye\": \"$RIGHT_EYE\",
    \"left_eye\": \"$LEFT_EYE\",
    \"mouth\": \"$MOUTH\",
    \"right_ear\": \"$RIGHT_EAR\",
    \"left_ear\": \"$LEFT_EAR\"
  }" \
  -o step3_result.json

# Display final results
cat step3_result.json | jq '.'
```

### Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:5000"

def analyze_cat(image_path):
    """Complete cat analysis workflow."""
    
    # Step 1: Recognize and detect face
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/v1/cat/recognize",
            files={'file': f}
        )
    
    result1 = response.json()
    
    if not result1['ok']:
        print(f"Error: {result1['error']}")
        return None
    
    if result1['cat_prob'] < result1['threshold']:
        print("Not a cat image")
        return None
    
    if not result1.get('faces') or not result1['faces']['ok']:
        print("No face detected")
        return None
    
    roi_url = result1['faces']['roi']['url']
    print(f"Face detected, ROI: {roi_url}")
    
    # Step 2: Detect landmarks
    response = requests.post(
        f"{BASE_URL}/v1/cat/landmark",
        json={'file': roi_url}
    )
    
    result2 = response.json()
    
    if not result2['ok']:
        print(f"Landmark error: {result2['error']}")
        return None
    
    # Step 3: Classify areas
    response = requests.post(
        f"{BASE_URL}/v1/cat/area-check",
        json={
            'right_eye': result2['right_eye']['img_right_eye_url'],
            'left_eye': result2['left_eye']['img_left_eye_url'],
            'mouth': result2['mouth']['img_mouth_url'],
            'right_ear': result2['right_ear']['img_right_ear_url'],
            'left_ear': result2['left_ear']['img_left_ear_url']
        }
    )
    
    result3 = response.json()
    
    if not result3['ok']:
        print(f"Classification error: {result3['error']}")
        return None
    
    # Print results
    print("\n=== Classification Results ===")
    for area, result in result3['classification'].items():
        print(f"{area}: {result['label']} ({result['confidence']:.2%})")
    
    print("\n=== Grad-CAM URLs ===")
    for area, urls in result3['gradcam'].items():
        url = urls[f'img_{area}_gradcam_url']
        print(f"{area}: {url}")
    
    return {
        'recognition': result1,
        'landmark': result2,
        'classification': result3
    }

# Usage
if __name__ == "__main__":
    results = analyze_cat("path/to/cat.jpg")
    
    if results:
        # Save complete results
        with open('analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
```

---

## Troubleshooting

### Common Issues

1. **Model Not Ready (503)**
   - Check if model files exist in correct paths
   - Verify ONNX/H5 model file integrity
   - Check logs for model loading errors

2. **File Too Large (413)**
   - Reduce image size or quality
   - Increase `MAX_FILE_MB` environment variable
   - Compress image before upload

3. **Upload Failed (502)**
   - Check storage API availability
   - Verify `CONTENT_API_BASE` configuration
   - Check network connectivity

4. **No Face Detected**
   - Ensure image contains visible cat face
   - Try different image angle or lighting
   - Adjust confidence thresholds in config

5. **Landmark Detection Failed**
   - Ensure input is cat face ROI (not full image)
   - Check face quality and resolution
   - Verify landmark model files

---

## Performance Considerations

### Response Times (Typical)

| Endpoint | Average | P95 | P99 |
|----------|---------|-----|-----|
| `/v1/cat/recognize` | 150ms | 300ms | 500ms |
| `/v1/cat/faces` | 80ms | 150ms | 250ms |
| `/v1/cat/remove-bg` | 2000ms | 4000ms | 6000ms |
| `/v1/cat/landmark` | 200ms | 400ms | 600ms |
| `/v1/cat/area-check` | 3000ms | 5000ms | 8000ms |

### Resource Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 2GB

**Recommended:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- GPU: Optional (for faster inference)

---

## Security Considerations

1. **Input Validation**
   - File type checking
   - File size limits
   - Image format verification

2. **Rate Limiting**
   - Implement per-IP rate limits
   - Add API key authentication

3. **CORS Configuration**
   - Configure allowed origins
   - Set appropriate CORS headers

4. **Error Handling**
   - Don't expose internal paths
   - Sanitize error messages

5. **Resource Limits**
   - Set timeout for external requests
   - Limit concurrent processing

---

## Change Log

### Version 1.0.0 (Current)
- Initial API release
- Support for 5 main endpoints
- Multi-model inference pipeline
- Storage integration
- Grad-CAM visualization

---

## Support & Contact

For issues or questions:
- GitHub Issues: [repository-url]
- Email: support@rescat.life
- Documentation: [docs-url]

---

**Last Updated:** December 19, 2025  
**API Version:** v1  
**Document Version:** 1.0.0
