"""File upload validation utilities."""

import os
from werkzeug.datastructures import FileStorage

def get_env_csv(name: str, default_csv: str) -> set:
    """Parse comma-separated env variable into set."""
    raw = os.getenv(name, default_csv)
    return {x.strip().lower() for x in raw.split(",") if x.strip()}

ALLOWED_EXT = get_env_csv("ALLOWED_EXT", "jpg,jpeg,png")
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "8"))

def _has_allowed_ext(filename: str) -> bool:
    """Check if filename has allowed extension."""
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXT

def _infer_size_bytes(file: FileStorage) -> int:
    """Get file size from content_length or stream."""
    if file.content_length is not None:
        return int(file.content_length)
    pos = file.stream.tell()
    file.stream.seek(0, 2)
    size = file.stream.tell()
    file.stream.seek(pos)
    return int(size)

def validate_upload(file: FileStorage):
    """Validate uploaded file for size and extension."""
    if not file:
        return False, "INVALID_FILE", "No file part"
    if not file.filename:
        return False, "INVALID_FILE", "Empty filename"
    if not _has_allowed_ext(file.filename):
        return False, "UNSUPPORTED_EXTENSION", f"Allowed: {', '.join(sorted(ALLOWED_EXT))}"
    size_bytes = _infer_size_bytes(file)
    if size_bytes > MAX_FILE_MB * 1024 * 1024:
        return False, "FILE_TOO_LARGE", f"Max {MAX_FILE_MB} MB"
    return True, None, None
