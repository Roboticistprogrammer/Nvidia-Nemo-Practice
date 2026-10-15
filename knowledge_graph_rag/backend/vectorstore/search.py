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

import random
import threading
from types import SimpleNamespace

import numpy as np
from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,
)

import config

# BGE-M3 and the reranker take ~3 GB of RAM and several seconds to load, so they are
# loaded once per process and shared by all requests (the lock also caps memory use).
_model_lock = threading.Lock()
_embedding_fn = None
_reranker_fn = None

TEXT_MAX_BYTES = 2048
INSERT_BATCH = 64


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        _embedding_fn = BGEM3EmbeddingFunction(use_fp16=False, device=config.EMBED_DEVICE)
    return _embedding_fn


def _get_reranker_fn():
    global _reranker_fn
    if _reranker_fn is None:
        from pymilvus.model.reranker import BGERerankFunction
        _reranker_fn = BGERerankFunction(device=config.EMBED_DEVICE)
    return _reranker_fn


def _connect():
    if not connections.has_connection("default"):
        connections.connect("default", uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)


def _truncate(text):
    # the VARCHAR field is limited to 2048 bytes
    return text.encode("utf-8")[:TEXT_MAX_BYTES].decode("utf-8", errors="ignore")


class SearchHandler:
    def __init__(self, collection_name, use_bge_m3=True, use_reranker=True):
        self.use_bge_m3 = use_bge_m3
        self.use_reranker = use_reranker
        self.collection_name = collection_name
        self.dense_dim = None
        self.ef = None
        self.setup_embeddings()
        self.setup_milvus_collection()

    def setup_embeddings(self):
        if self.use_bge_m3:
            with _model_lock:
                self.ef = _get_embedding_fn()
            self.dense_dim = self.ef.dim["dense"]
        else:
            self.dense_dim = 768
            self.ef = self.random_embedding

    def random_embedding(self, texts):
        rng = np.random.default_rng()
        return {
            "dense": np.random.rand(len(texts), self.dense_dim),
            "sparse": [{d: rng.random() for d in random.sample(range(1000), random.randint(20, 30))} for _ in texts],
        }

    def _embed(self, texts):
        with _model_lock:
            return self.ef(texts)

    def setup_milvus_collection(self):
        _connect()
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=TEXT_MAX_BYTES),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
            ]
            schema = CollectionSchema(fields, "")
            self.collection = Collection(self.collection_name, schema, consistency_level="Strong")

            sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.collection.create_index("sparse_vector", sparse_index)
            dense_index = {"index_type": "FLAT", "metric_type": "IP"}
            self.collection.create_index("dense_vector", dense_index)
        self.collection.load()

    def reset(self):
        """Drop all stored chunks, e.g. before re-processing a workspace's documents."""
        utility.drop_collection(self.collection_name)
        self.setup_milvus_collection()

    def insert_data(self, docs):
        texts = [_truncate(doc.page_content) for doc in docs]
        for start in range(0, len(texts), INSERT_BATCH):
            batch = texts[start:start + INSERT_BATCH]
            embeddings = self._embed(batch)
            self.collection.insert([batch, embeddings["sparse"], embeddings["dense"]])
        self.collection.flush()

    def search_and_rerank(self, query, k=2):
        """Return up to k passages; each result has a `.text` attribute."""
        query_embeddings = self._embed([query])

        sparse_search_params = {"metric_type": "IP"}
        sparse_req = AnnSearchRequest(query_embeddings["sparse"], "sparse_vector", sparse_search_params, limit=k)
        dense_search_params = {"metric_type": "IP"}
        dense_req = AnnSearchRequest(query_embeddings["dense"], "dense_vector", dense_search_params, limit=k)

        res = self.collection.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(), limit=k, output_fields=['text'])
        result_texts = [hit.fields["text"] for hit in res[0]]
        if not result_texts:
            return []

        if self.use_reranker:
            with _model_lock:
                return _get_reranker_fn()(query, result_texts, top_k=k)
        return [SimpleNamespace(text=t) for t in result_texts]
