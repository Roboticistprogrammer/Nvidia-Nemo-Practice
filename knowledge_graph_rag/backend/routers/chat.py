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

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth import Workspace, get_workspace
from utils.models import get_chat_models, make_llm
from utils import rag

logger = logging.getLogger(__name__)

router = APIRouter()

KG_UNAVAILABLE = "The knowledge graph is currently unavailable. Please try again later."


# Define a Pydantic model for the chat request body
class ChatRequest(BaseModel):
    user_input: str
    use_kg: bool
    model_id: str


# Define an endpoint to get available models
@router.get("/get-models/")
def get_models():
    return {"models": get_chat_models()}


# Define an endpoint for the chat interface
@router.post("/chat/")
def chat_endpoint(request: ChatRequest, ws: Workspace = Depends(get_workspace)):
    response_data = {"user_input": request.user_input, "use_kg": request.use_kg}
    llm = make_llm(ws, request.model_id)

    if not request.use_kg:
        response_data["context"] = ""
        response_data["assistant_response"] = llm.invoke(request.user_input).content or ""
        return response_data

    graph = rag.load_graph(ws)
    if graph is None:
        logger.info(f"No knowledge graph yet for workspace {ws.id[:8]}")
        return {"assistant_response": KG_UNAVAILABLE}

    try:
        entities = rag.extract_entities(llm, request.user_input)
        context = rag.passages_context(rag.search_passages(ws, request.user_input, k=5))
        context += "\n\n" + rag.triplets_context(rag.graph_triplets(graph, entities))
    except Exception as e:
        logger.warning(f"Retrieval failed, answering without graph context: {e}")
        context = rag.NO_TRIPLES_CONTEXT
    response_data["context"] = context
    response_data["assistant_response"] = rag.answer_with_context(llm, context, request.user_input)
    return response_data
