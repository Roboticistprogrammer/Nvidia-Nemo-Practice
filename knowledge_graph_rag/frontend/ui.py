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

import time

import streamlit as st

from common import api, error_detail, model_selector, require_api_key

st.set_page_config(page_title="Knowledge Graph RAG")
require_api_key()

st.title("Knowledge Graph RAG")
llm = model_selector("ui")


def upload_section():
    st.subheader("1. Upload PDFs")
    files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Upload"):
        resp = api("POST", "/documents/",
                   files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in files])
        if resp.ok:
            st.success(f"Uploaded: {', '.join(resp.json()['uploaded'])}")
        else:
            st.error(error_detail(resp))

    resp = api("GET", "/documents/")
    documents = resp.json()["documents"] if resp.ok else []
    if not documents:
        st.caption("No documents in your workspace yet.")
    for doc in documents:
        col_name, col_size, col_delete = st.columns([6, 2, 1])
        col_name.write(doc["name"])
        col_size.write(f"{doc['size_kb']} KB")
        if col_delete.button("✕", key=f"delete-{doc['name']}", help="Delete"):
            api("DELETE", f"/documents/{doc['name']}")
            st.rerun()
    return documents


def show_progress():
    """Poll the backend until the current processing job finishes."""
    progress_bar = st.progress(0.0)
    progress_text = st.empty()
    while True:
        resp = api("GET", "/ui/progress/")
        if not resp.ok:
            st.error(error_detail(resp))
            return
        status = resp.json()
        progress_bar.progress(min(float(status.get("progress", 0)), 1.0))
        progress_text.text(status.get("message", ""))
        if status["state"] == "done":
            st.success(status["message"] + " Saved as knowledge_graph.graphml (see the visualization page).")
            return
        if status["state"] == "error":
            st.error(f"Processing failed: {status['message']}")
            return
        if status["state"] == "idle":
            return
        time.sleep(2)


def process_section(documents):
    st.subheader("2. Build the knowledge graph")
    st.caption("Extracts entity/relation triples from every chunk with the selected LLM and indexes "
               "the chunks for hybrid search. Large PDFs can take a while.")
    if st.button("Process Documents", disabled=not documents):
        resp = api("POST", "/ui/process-documents/", json={"model_id": llm})
        if not resp.ok:
            st.error(error_detail(resp))
            return
        show_progress()
    else:
        status = api("GET", "/ui/progress/")
        # e.g. the page was reloaded while a job was running
        if status.ok and status.json().get("state") == "running":
            show_progress()


documents = upload_section()
process_section(documents)
