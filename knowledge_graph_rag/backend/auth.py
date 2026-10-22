"""Bring-your-own-key auth.

Every request carries the user's NVIDIA API key in the `X-NVIDIA-API-Key` header.
The key is verified against NVIDIA once (then cached), is never stored or logged, and an
HMAC of it selects a private workspace: its own data folder and Milvus collection.
"""
import hashlib
import hmac
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from fastapi import Header, HTTPException

import config

_verified = {}  # sha256(key) -> expiry timestamp
_lock = threading.Lock()


@dataclass(frozen=True)
class Workspace:
    api_key: str
    id: str

    @property
    def data_dir(self) -> Path:
        path = config.DATA_DIR / "workspaces" / self.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def uploads_dir(self) -> Path:
        path = self.data_dir / "uploads"
        path.mkdir(exist_ok=True)
        return path

    @property
    def collection(self) -> str:
        return f"kg_{self.id}"

    def file(self, name) -> Path:
        return self.data_dir / name


def _verify_with_nvidia(api_key):
    """A one-token embeddings call: rejects bad keys with 401/403 in ~0.2s."""
    try:
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": config.KEY_CHECK_MODEL, "input": ["ping"], "input_type": "query"},
            timeout=20,
        )
    except requests.RequestException:
        raise HTTPException(503, "Could not reach NVIDIA to verify the API key. Try again.")
    if resp.status_code in (401, 403):
        raise HTTPException(401, "NVIDIA rejected this API key.")
    if resp.status_code != 200:
        raise HTTPException(503, f"NVIDIA key check failed with HTTP {resp.status_code}.")


def _ensure_verified(api_key):
    digest = hashlib.sha256(api_key.encode()).hexdigest()
    now = time.time()
    with _lock:
        if _verified.get(digest, 0) > now:
            return
    _verify_with_nvidia(api_key)
    with _lock:
        _verified[digest] = now + config.KEY_CACHE_TTL_S


def workspace_id(api_key):
    return hmac.new(config.WORKSPACE_SECRET.encode(), api_key.encode(), hashlib.sha256).hexdigest()[:16]


def get_workspace(x_nvidia_api_key: Optional[str] = Header(default=None)) -> Workspace:
    """FastAPI dependency: resolve the caller's workspace or fail with 401."""
    api_key = (x_nvidia_api_key or "").strip()
    if not api_key:
        if config.ALLOW_SERVER_KEY and config.SERVER_API_KEY:
            api_key = config.SERVER_API_KEY
        else:
            raise HTTPException(401, "Missing NVIDIA API key. Add yours in the sidebar.")
    if not api_key.startswith("nvapi-"):
        raise HTTPException(401, "NVIDIA API keys start with 'nvapi-'.")
    _ensure_verified(api_key)
    return Workspace(api_key=api_key, id=workspace_id(api_key))
