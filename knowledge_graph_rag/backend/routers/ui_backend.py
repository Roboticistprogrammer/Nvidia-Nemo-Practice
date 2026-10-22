# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from auth import Workspace, get_workspace
from utils.lc_graph import process_documents, save_triples_to_csvs, build_graphml
from utils.models import get_chat_models, make_llm, without_reasoning
from utils.rag import graph_path
from vectorstore.search import SearchHandler

logger = logging.getLogger(__name__)

# Initialize a FastAPI router
router = APIRouter()

STATUS_FILE = "status.json"
# a "running" status older than this is treated as stale (e.g. the server restarted mid-job)
STALE_AFTER_S = 3 * 3600


class ProcessRequest(BaseModel):
    model_id: str


def _write_status(ws, state, progress=0.0, message=""):
    ws.file(STATUS_FILE).write_text(json.dumps(
        {"state": state, "progress": progress, "message": message, "updated": time.time()}
    ))


def _read_status(ws):
    path = ws.file(STATUS_FILE)
    if not path.exists():
        return {"state": "idle", "progress": 0.0, "message": ""}
    return json.loads(path.read_text())


# Define an endpoint to get available models
@router.get("/get-models/")
def get_models():
    return {"models": get_chat_models()}


# Define an endpoint to process the documents uploaded to the caller's workspace
@router.post("/process-documents/")
def process_documents_endpoint(request: ProcessRequest, background_tasks: BackgroundTasks,
                               ws: Workspace = Depends(get_workspace)):
    if not any(ws.uploads_dir.glob("*.pdf")):
        raise HTTPException(400, "No PDFs uploaded yet. Upload at least one PDF first.")
    status = _read_status(ws)
    if status["state"] == "running" and time.time() - status.get("updated", 0) < STALE_AFTER_S:
        raise HTTPException(409, "Documents are already being processed in this workspace.")

    llm = make_llm(ws, request.model_id)
    _write_status(ws, "running", 0.0, "Loading and splitting PDFs")

    def update_progress(completed, total):
        _write_status(ws, "running", completed / total, f"Extracting triples: {completed}/{total} chunks")

    def background_task():
        try:
            documents, results = process_documents(
                ws.uploads_dir, without_reasoning(llm), ws.data_dir, update_progress=update_progress,
                max_workers=config.EXTRACTION_WORKERS,
            )
            if not results:
                raise RuntimeError("No triples were extracted. Check that the PDFs contain text "
                                   "and that the selected model is available.")
            _write_status(ws, "running", 1.0, "Indexing chunks in the vector store")
            search_handler = SearchHandler(ws.collection, use_bge_m3=True, use_reranker=True)
            search_handler.reset()
            search_handler.insert_data(documents)
            save_triples_to_csvs(results, ws.data_dir)
            build_graphml(ws.data_dir)
            _write_status(ws, "done", 1.0, f"Graph built from {len(documents)} chunks and {len(results)} triples.")
        except Exception as e:
            logger.exception("Document processing failed")
            _write_status(ws, "error", 0.0, str(e))

    background_tasks.add_task(background_task)
    return {"message": "Processing started"}


# Define an endpoint to get the progress of the background task
@router.get("/progress/")
def get_progress(ws: Workspace = Depends(get_workspace)):
    return _read_status(ws)


# Define an endpoint to check if the GraphML file exists
@router.get("/check-file-exists/")
def check_file_exists(ws: Workspace = Depends(get_workspace)):
    return {"file_exists": graph_path(ws).exists()}


# Download the workspace's knowledge graph (e.g. to open it in Gephi Lite)
@router.get("/graph/")
def download_graph(ws: Workspace = Depends(get_workspace)):
    path = graph_path(ws)
    if not path.exists():
        raise HTTPException(404, "No knowledge graph yet. Process your documents first.")
    return FileResponse(path, media_type="application/xml", filename="knowledge_graph.graphml")
