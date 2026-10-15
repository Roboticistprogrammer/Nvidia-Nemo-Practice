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

import csv
import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.schema import Document
from openai import OpenAI

from datasets import Dataset
from ragas.metrics import answer_relevancy, context_precision
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

import config
from auth import Workspace, get_workspace
from utils.lc_graph import process_documents
from utils.models import get_chat_models, make_llm, make_embeddings
from utils.preprocessor import generate_qa_pair, judge_prompt_template
from utils import rag

logger = logging.getLogger(__name__)

router = APIRouter()

DOCUMENTS_CSV = "documents.csv"
QA_CSV = "qa_data.csv"
COMBINED_CSV = "combined_results.csv"
SCORE_COLUMNS = ['gt', 'textRAG', 'graphRAG', 'combinedRAG']


class ProcessRequest(BaseModel):
    model_id: str


class QAPairsRequest(BaseModel):
    num_data: int
    model_id: str


class QARequest(BaseModel):
    questions_list: list
    answers_list: list
    model_id: str


class ScoreRequest(BaseModel):
    combined_results: list


def _stream(rows):
    """NDJSON streaming; a plain generator runs in FastAPI's threadpool, not the event loop."""
    return StreamingResponse((json.dumps(row) + "\n" for row in rows), media_type="text/event-stream")


def _read_records(ws, name, missing_msg):
    path = ws.file(name)
    if not path.exists():
        raise HTTPException(404, missing_msg)
    return pd.read_csv(path).fillna("").to_dict(orient="records")


@router.get("/get-models/")
def get_models():
    return {"models": get_chat_models()}


@router.get("/config")
def eval_config():
    """Which scoring methods are enabled, and with which models (shown in the UI)."""
    return {
        "judge_model": get_chat_models()[0],
        "embed_model": config.EVAL_EMBED_MODEL,
        "reward_model": config.REWARD_MODEL or None,
    }


# ---- RAG variants compared by the evaluation ----

def get_text_RAG_response(ws, question, llm):
    passages = rag.search_passages(ws, question, k=5)
    answer = rag.answer_with_context(llm, rag.passages_context(passages), question)
    return answer, passages


def get_graph_RAG_response(ws, question, llm, graph):
    try:
        triplets = rag.graph_triplets(graph, rag.extract_entities(llm, question))
        context = rag.triplets_context(triplets)
        context_return = triplets or ["no relationship found"]
    except Exception:
        context, context_return = rag.NO_TRIPLES_CONTEXT, []
    return rag.answer_with_context(llm, context, question), context_return


def get_combined_RAG_response(ws, question, llm, graph):
    try:
        entities = rag.extract_entities(llm, question)
        passages = rag.search_passages(ws, question, k=5)
        triplets = rag.graph_triplets(graph, entities)
        context = rag.passages_context(passages) + "\n\n" + rag.triplets_context(triplets)
        context_return = passages + triplets
    except Exception:
        context, context_return = rag.NO_TRIPLES_CONTEXT, []
    return rag.answer_with_context(llm, context, question), context_return


def process_question(ws, question, answer, llm, graph):
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_text = executor.submit(get_text_RAG_response, ws, question, llm)
        future_graph = executor.submit(get_graph_RAG_response, ws, question, llm, graph)
        future_combined = executor.submit(get_combined_RAG_response, ws, question, llm, graph)

        text_RAG_response, text_RAG_context_response = future_text.result()
        graph_RAG_response, graph_RAG_context_response = future_graph.result()
        combined_RAG_response, combined_RAG_context_response = future_combined.result()

    return {
        "question": question,
        "gt_answer": answer,
        "textRAG_answer": text_RAG_response,
        "graphRAG_answer": graph_RAG_response,
        "combined_answer": combined_RAG_response,
        "text_RAG_context_response": text_RAG_context_response,
        "graph_RAG_context_response": graph_RAG_context_response,
        "combined_RAG_context_response": combined_RAG_context_response
    }


# ---- Pipeline endpoints ----

@router.post("/process-documents/")
def process_documents_endpoint(request: ProcessRequest, ws: Workspace = Depends(get_workspace)):
    if not any(ws.uploads_dir.glob("*.pdf")):
        raise HTTPException(400, "No PDFs uploaded yet. Upload at least one PDF first.")
    llm = make_llm(ws, request.model_id)
    documents, _ = process_documents(ws.uploads_dir, llm, ws.data_dir, triplets=False,
                                     chunk_size=2000, chunk_overlap=200)
    logger.info(f"Split {len(documents)} evaluation chunks for workspace {ws.id[:8]}")
    return {"message": "Documents processed", "documents_processed": len(documents)}


@router.post("/create-qa-pairs/")
def create_qa_pairs(request: QAPairsRequest, ws: Workspace = Depends(get_workspace)):
    llm = make_llm(ws, request.model_id)
    df = pd.DataFrame(_read_records(ws, DOCUMENTS_CSV, "Documents not found. Please process documents first."))
    documents = [Document(page_content=str(row['content'])) for _, row in df.iterrows()]
    qa_docs = random.sample(documents, min(request.num_data, len(documents)))

    def generate():
        json_list = []
        for doc in qa_docs:
            try:
                res = generate_qa_pair(doc, llm)
                if res:
                    json_list.append(res)
                    yield res
            except Exception as e:
                logger.error(f"Error generating Q&A pair: {e}")
                yield {"error": str(e)}
        if json_list:
            pd.DataFrame(json_list).to_csv(ws.file(QA_CSV), index=False)
        else:
            yield {"error": "No Q&A pairs generated"}

    return _stream(generate())


@router.get("/qa-data/")
def get_qa_data(ws: Workspace = Depends(get_workspace)):
    return {"records": _read_records(ws, QA_CSV, "No Q&A data yet. Create Q&A pairs first.")}


@router.post("/run-evaluation/")
def run_evaluation(request: QARequest, ws: Workspace = Depends(get_workspace)):
    llm = make_llm(ws, request.model_id)
    graph = rag.load_graph(ws)
    if graph is None:
        raise HTTPException(404, "No knowledge graph yet. Build it on the main page first.")
    combined_path = ws.file(COMBINED_CSV)
    combined_path.unlink(missing_ok=True)

    def evaluate_all():
        for question, answer in zip(request.questions_list, request.answers_list):
            result = process_question(ws, question, answer, llm, graph)
            # Save the result to the CSV file incrementally
            with open(combined_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow(result)
            yield result

    return _stream(evaluate_all())


@router.get("/combined-results/")
def get_combined_results(ws: Workspace = Depends(get_workspace)):
    return {"records": _read_records(ws, COMBINED_CSV, "No evaluation results yet. Run the evaluation first.")}


# ---- Scoring ----

def _score_rows(ws, rows, score_fn, out_name):
    """Score every answer variant of each row, stream rows back and save them when done."""
    scored = []
    for row in rows:
        try:
            answers = [row["gt_answer"], row["textRAG_answer"], row["graphRAG_answer"], row["combined_answer"]]
            contexts = [row["combined_RAG_context_response"], row["text_RAG_context_response"],
                        row["graph_RAG_context_response"], row["combined_RAG_context_response"]]
            for score_type, answer, context in zip(SCORE_COLUMNS, answers, contexts):
                res = score_fn(row, answer, context)
                for metric, value in (res or {}).items():
                    row[f'{score_type}_{metric}'] = value
            scored.append(row)
            yield row
        except Exception as e:
            yield {"error": str(e)}
    if scored:
        pd.DataFrame(scored).to_csv(ws.file(out_name), index=False)


def get_reward_scores(reward_client, question, answer):
    completion = reward_client.chat.completions.create(
        model=config.REWARD_MODEL,
        messages=[{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    )
    try:
        content = completion.choices[0].message[0].content
        res = content.split(",")
        content_dict = {}
        for item in res:
            name, val = item.split(":")
            content_dict[name] = float(val)
        return content_dict
    except Exception:
        return None


@router.post("/run-scoring/")
def run_scoring(request: ScoreRequest, ws: Workspace = Depends(get_workspace)):
    if not config.REWARD_MODEL:
        raise HTTPException(400, "Reward-model scoring is disabled on this server (REWARD_MODEL is not set).")
    reward_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=ws.api_key)
    metrics = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']

    def score(row, answer, _context):
        res = get_reward_scores(reward_client, row["question"], answer)
        return {m: res.get(m) for m in metrics} if res else None

    return _stream(_score_rows(ws, request.combined_results, score, "combined_results_with_scores_nemotron.csv"))


def get_RAGAS_evaluation(question, rag_answer, context, gt_answer, llm, embeddings, metrics):
    list_items = [item.strip() for item in str(context).split(',')]
    d_eval_dataset = Dataset.from_dict({
        "question": [question],
        "answer": [rag_answer],
        "contexts": [list_items],
        "ground_truth": [gt_answer]
    })
    result = evaluate(d_eval_dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    context_result = {}
    for score in result.scores:
        for m, value in score.items():
            context_result[f"{m}"] = value
    return context_result


@router.post("/run-scoring-RAGAS/")
def run_scoring_RAGAS(request: ScoreRequest, ws: Workspace = Depends(get_workspace)):
    # RAGAS evaluation uses your own LLM and embeddings model
    llm = LangchainLLMWrapper(langchain_llm=make_llm(ws, temperature=0.2))
    embeddings = LangchainEmbeddingsWrapper(make_embeddings(ws))
    metrics = [answer_relevancy, context_precision]

    def score(row, answer, context):
        return get_RAGAS_evaluation(row['question'], answer, context, row['gt_answer'], llm, embeddings, metrics)

    return _stream(_score_rows(ws, request.combined_results, score, "combined_results_with_scores_RAGAS.csv"))


def get_llm_as_a_judge_scores(question, answer, llm, QA_PROMPT):
    result = llm.invoke(QA_PROMPT.format(question=question, answer=answer))
    res = result.content or ""
    match = re.search(r'Evaluation:(.*?)Total rating:', res, flags=re.DOTALL)
    eval_res = match.group(1).strip() if match else "no match found"
    match = re.search(r'Total rating:\s*(\d+(?:\.\d+)?)', res, flags=re.DOTALL)
    eval_score = float(match.group(1).strip()) if match else 0.0
    return {'llm_judge_evaluation': eval_res, 'llm_judge_score': eval_score}


@router.post("/run-scoring_llm_as_a_judge/")
def run_scoring_llm_as_a_judge(request: ScoreRequest, ws: Workspace = Depends(get_workspace)):
    llm = make_llm(ws, temperature=0.2)
    QA_PROMPT = judge_prompt_template()

    def score(row, answer, _context):
        return get_llm_as_a_judge_scores(row["question"], answer, llm, QA_PROMPT)

    return _stream(_score_rows(ws, request.combined_results, score, "combined_results_with_scores_LLM-AS-A-JUDGE.csv"))
