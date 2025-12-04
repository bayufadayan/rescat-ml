"""Flask JSON response utilities."""

from flask import jsonify

def ok(data: dict, status: int = 200):
    """Return success JSON response."""
    payload = {"ok": True}
    payload.update(data or {})
    return jsonify(payload), status

def err(code: str, message: str = "", status: int = 400, extra: dict | None = None):
    """Return error JSON response."""
    payload = {"ok": False, "error": code}
    if message:
        payload["message"] = message
    if extra:
        payload.update(extra)
    return jsonify(payload), status
