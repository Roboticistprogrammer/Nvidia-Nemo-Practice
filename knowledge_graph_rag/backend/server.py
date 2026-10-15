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

import logging
import os

import config  # loads .env before anything reads the environment
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from routers import auth, documents, ui_backend, chat, evaluation, visualization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Knowledge Graph RAG")


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    # Surface upstream errors (e.g. a retired NVIDIA model) to the UI instead of a bare 500
    logger.exception(f"Unhandled error on {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/health")
def health():
    return {"status": "ok"}


# Include routers from different modules
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(ui_backend.router, prefix="/ui")
app.include_router(chat.router, prefix="/chat")
app.include_router(evaluation.router, prefix="/evaluation")
app.include_router(visualization.router, prefix="/visualization", tags=["visualization"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
