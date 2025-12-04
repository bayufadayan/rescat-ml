"""Image upload utilities for storage API."""

import os
import requests
import logging
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

log = logging.getLogger("uploader")

CONTENT_API_BASE = os.getenv("CONTENT_API_BASE", "https://storage.rescat.life").rstrip("/")
TIMEOUT_CONNECT = float(os.getenv("CONTENT_API_TIMEOUT_CONNECT", "5"))
TIMEOUT_READ = float(os.getenv("CONTENT_API_TIMEOUT_READ", "30"))
TIMEOUT = (TIMEOUT_CONNECT, TIMEOUT_READ)


def _session():
    """Create requests session with retry configuration."""
    s = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.4,
        allowed_methods=frozenset(["POST"]),
        status_forcelist=[502, 503, 504],
        raise_on_status=False,
        raise_on_redirect=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "ResCat-Flask/1.0",
        "Connection": "close",
    })
    return s

_S = _session()


def upload_image_bytes(image_bytes: bytes, bucket: str, original_name: str) -> dict:
    """Upload image bytes to storage API."""
    url = f"{CONTENT_API_BASE}/api/files"
    files = {
        "file": (original_name, image_bytes, "image/jpeg"),
    }
    data = {"bucket": bucket}
    try:
        resp = _S.post(url, files=files, data=data, timeout=TIMEOUT, allow_redirects=True)
        status = resp.status_code
        ctype = resp.headers.get("content-type", "")
        j = resp.json() if ctype.startswith("application/json") else {}
        if status >= 400:
            log.warning("UPLOAD HTTP %s body=%s", status, resp.text[:300])
            return {"ok": False, "status_code": status, "message": f"HTTP {status}", "raw": j or {"text": resp.text[:300]}}
        if not j.get("ok"):
            return {"ok": False, "status_code": status, "message": "Upload returned ok=false", "raw": j}
        d = j.get("data", {})
        return {
            "ok": True,
            "id": d.get("id"),
            "bucket": d.get("bucket"),
            "filename": d.get("filename"),
            "url": d.get("url"),
            "raw": j,
        }
    except requests.exceptions.ChunkedEncodingError as e:
        # server tidak menerima transfer chunked
        return {"ok": False, "message": f"UPLOAD_ERROR: chunked not supported: {e}"}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "message": f"UPLOAD_ERROR: connection error: {e}"}
    except Exception as e:
        return {"ok": False, "message": f"UPLOAD_ERROR: {e}"}