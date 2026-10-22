from fastapi import APIRouter, Depends

import config
from auth import Workspace, get_workspace

router = APIRouter()


@router.get("/config")
def auth_config():
    """Tells the UI whether it may run without a user key (server key fallback)."""
    return {"server_key_available": bool(config.ALLOW_SERVER_KEY and config.SERVER_API_KEY)}


@router.post("/verify")
def verify(ws: Workspace = Depends(get_workspace)):
    # get_workspace already verified the key with NVIDIA
    return {"workspace": ws.id[:8]}
