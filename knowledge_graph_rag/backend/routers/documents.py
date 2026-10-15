"""Upload, list and delete the PDFs in the caller's workspace."""
import re
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

import config
from auth import Workspace, get_workspace

router = APIRouter()

CHUNK = 1024 * 1024


def _safe_name(filename):
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", (filename or "").split("/")[-1].split("\\")[-1]).strip(" .")
    if not name.lower().endswith(".pdf") or name.lower() == ".pdf":
        raise HTTPException(400, f"'{filename}' is not a PDF file.")
    return name


@router.get("/")
def list_documents(ws: Workspace = Depends(get_workspace)):
    files = sorted(ws.uploads_dir.glob("*.pdf"))
    return {"documents": [{"name": f.name, "size_kb": round(f.stat().st_size / 1024)} for f in files]}


@router.post("/")
def upload_documents(files: List[UploadFile] = File(...), ws: Workspace = Depends(get_workspace)):
    limit = config.MAX_UPLOAD_MB * CHUNK
    saved = []
    for upload in files:
        name = _safe_name(upload.filename)
        head = upload.file.read(5)
        if head != b"%PDF-":
            raise HTTPException(400, f"'{upload.filename}' does not look like a PDF.")
        target = ws.uploads_dir / name
        size = len(head)
        with open(target, "wb") as out:
            out.write(head)
            while chunk := upload.file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, f"'{name}' is larger than {config.MAX_UPLOAD_MB} MB.")
                out.write(chunk)
        saved.append(name)
    return {"uploaded": saved}


@router.delete("/{name}")
def delete_document(name: str, ws: Workspace = Depends(get_workspace)):
    target = ws.uploads_dir / _safe_name(name)
    if not target.exists():
        raise HTTPException(404, f"'{name}' not found.")
    target.unlink()
    return {"deleted": target.name}
