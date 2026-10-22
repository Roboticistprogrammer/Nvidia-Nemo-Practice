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

import pandas as pd
import requests
import streamlit as st

from common import api, error_detail, model_selector, require_api_key

st.set_page_config(layout="wide")
require_api_key()

# Custom CSS to change heading font sizes
st.markdown("""
    <style>
    h1, h2, h3, h4, h5, h6 {
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Evaluations")

if 'documents' not in st.session_state:
    st.session_state['documents'] = None

llm_selectbox = model_selector("evaluation")
with st.sidebar:
    num_data = st.slider("How many Q&A pairs to generate?", 10, 100, 50, step=10)

eval_config = api("GET", "/evaluation/config").json()


def stream_rows(path, payload, total, table=None, show_each=False):
    """POST to a streaming endpoint and collect the NDJSON rows it returns."""
    rows = []
    progress_bar = st.progress(0.0)
    placeholder = st.empty()
    try:
        resp = api("POST", path, json=payload, stream=True)
        if not resp.ok:
            st.error(error_detail(resp))
            return rows
        for line in resp.iter_lines():
            if not line:
                continue
            row = json.loads(line.decode('utf-8'))
            if "error" in row:
                st.error(row["error"])
                continue
            rows.append(row)
            if show_each:
                st.write(row)
            elif table:
                placeholder.dataframe(pd.DataFrame(rows))
            progress_bar.progress(min(len(rows) / max(total, 1), 1.0))
    except requests.exceptions.ChunkedEncodingError as e:
        st.error(f"Streaming error: {e}")
    progress_bar.progress(1.0)
    return rows


def get_records(path):
    resp = api("GET", path)
    return resp.json()["records"] if resp.ok else None


def csv_download(label, rows, file_name):
    st.download_button(label, pd.DataFrame(rows).to_csv(index=False), file_name=file_name, mime="text/csv")


def app():
    st.caption("Uses the PDFs uploaded to your workspace on the main page.")

    # Process Documents
    with st.container():
        st.markdown("### 1. Process Documents")
        if st.button("Process Documents"):
            with st.spinner("Splitting documents..."):
                resp = api("POST", "/evaluation/process-documents/", json={"model_id": llm_selectbox})
            if resp.ok:
                st.session_state["documents"] = resp.json().get("documents_processed")
                st.success(f"Finished splitting documents! Number of documents processed: {st.session_state['documents']}")
            else:
                st.error(error_detail(resp))

    # Create Q&A pairs
    with st.container():
        st.markdown("### 2. Create synthetic Q&A pairs from large document chunks")
        if st.session_state["documents"] is not None and st.button("Create Q&A pairs"):
            pairs = stream_rows("/evaluation/create-qa-pairs/",
                                {"num_data": num_data, "model_id": llm_selectbox}, num_data, show_each=True)
            if pairs:
                st.success("Q&A pair generation completed.")

    # Run Evaluation
    qa_records = get_records("/evaluation/qa-data/")
    if qa_records:
        with st.container():
            st.markdown("### 3. Load Q&A data and run evaluations of text vs graph vs text+graph RAG")
            if st.button("Run Evaluation"):
                results = stream_rows("/evaluation/run-evaluation/", {
                    "questions_list": [r["question"] for r in qa_records],
                    "answers_list": [r["answer"] for r in qa_records],
                    "model_id": llm_selectbox,
                }, len(qa_records), table=True)
                if results:
                    st.success("Combined results saved to 'combined_results.csv'")
                    csv_download("Download combined_results.csv", results, "combined_results.csv")

    # Run Scoring
    combined_results = get_records("/evaluation/combined-results/")
    if combined_results:
        with st.container():
            st.markdown("### 4. Run comparative evals for saved Q&A data")
            options = {
                "LLM-AS-A-JUDGE": ("/evaluation/run-scoring_llm_as_a_judge/",
                                   f"{eval_config['judge_model']} will be used as Judge"),
                "RAGAS": ("/evaluation/run-scoring-RAGAS/",
                          f"Evaluations will be done using: {eval_config['judge_model']} LLM and "
                          f"{eval_config['embed_model']} embedding model"),
            }
            if eval_config.get("reward_model"):
                options["Nemotron"] = ("/evaluation/run-scoring/",
                                       f"Evaluations will be done using: {eval_config['reward_model']}")
            eval_select = st.selectbox("Choose evaluation type ? ", list(options))
            path, description = options[eval_select]
            st.write(description)
            if st.button("Run Scoring"):
                results = stream_rows(path, {"combined_results": combined_results}, len(combined_results), table=True)
                if results:
                    score_file = f"combined_results_with_scores_{eval_select}.csv"
                    st.success("Scoring completed and results saved to " + score_file)
                    csv_download(f"Download {score_file}", results, score_file)


app()
