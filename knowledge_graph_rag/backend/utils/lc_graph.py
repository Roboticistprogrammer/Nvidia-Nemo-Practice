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

import concurrent.futures
import logging
import os

import networkx as nx
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.preprocessor import extract_triples

logger = logging.getLogger(__name__)


# function to process a single document (many of these run in parallel)
def process_document(doc, llm):
    try:
        return extract_triples(doc, llm)
    except Exception as e:
        logger.warning(f"Error processing document chunk: {e}")
        return []


def process_documents(directory, llm, data_dir, update_progress=None, triplets=True,
                      chunk_size=500, chunk_overlap=100, max_workers=4):
    """Split the PDFs in `directory`, save the chunks to documents.csv in `data_dir` and
    (optionally) extract knowledge-graph triples from each chunk."""
    loader = DirectoryLoader(str(directory), glob="*.pdf")
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(raw_docs)
    logger.info(f"Loaded {len(raw_docs)} PDFs -> {len(documents)} chunks")

    df = pd.DataFrame([{"id": i, "content": doc.page_content} for i, doc in enumerate(documents)])
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "documents.csv"), index=False)

    if not triplets or not documents:
        return documents, []

    # LLM calls are network-bound, so threads are enough (and safe inside containers)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_document, doc, llm) for doc in documents]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                results.extend(future.result())
            except Exception as e:
                logger.warning(f"Error collecting result: {e}")
            if update_progress:
                update_progress(completed, len(futures))

    logger.info(f"Processing complete. Total triples extracted: {len(results)}")
    return documents, results


def save_triples_to_csvs(triples, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    # Define paths for the CSV files
    entities_csv_path = os.path.join(data_dir, 'entities.csv')
    relations_csv_path = os.path.join(data_dir, 'relations.csv')
    triples_csv_path = os.path.join(data_dir, 'triples.csv')
    # Create the triples DataFrame
    triples_df = pd.DataFrame(triples, columns=['subject', 'subject_type', 'relation', 'object', 'object_type'])

    # Create the relations DataFrame
    relations_df = pd.DataFrame({'relation_id': range(len(triples_df['relation'].unique())), 'relation_name': triples_df['relation'].unique()})

    # Get unique entities (subjects and objects) from triples_df
    entities = pd.concat([triples_df['subject'], triples_df['object']]).unique()

    entities_df = pd.DataFrame({
    'entity_name': entities,
    'entity_type': [
        triples_df.loc[triples_df['subject'] == entity, 'subject_type'].iloc[0]
        if entity in triples_df['subject'].values
        else triples_df.loc[triples_df['object'] == entity, 'object_type'].dropna().iloc[0]
             if not triples_df.loc[triples_df['object'] == entity, 'object_type'].empty
             else 'Unknown'
        for entity in entities
        ]
    })
    entities_df = entities_df.reset_index().rename(columns={'index': 'entity_id'})

    # Merge triples_df with entities_df for subject
    triples_with_ids = triples_df.merge(entities_df[['entity_id', 'entity_name']], left_on='subject', right_on='entity_name', how='left')
    triples_with_ids = triples_with_ids.rename(columns={'entity_id': 'entity_id_1'}).drop(columns=['entity_name', 'subject', 'subject_type'])

    # Merge triples_with_ids with entities_df for object
    triples_with_ids = triples_with_ids.merge(entities_df[['entity_id', 'entity_name']], left_on='object', right_on='entity_name', how='left')
    triples_with_ids = triples_with_ids.rename(columns={'entity_id': 'entity_id_2'}).drop(columns=['entity_name', 'object', 'object_type'])

    # Merge triples_with_ids with relations_df to get relation IDs
    triples_with_ids = triples_with_ids.merge(relations_df, left_on='relation', right_on='relation_name', how='left').drop(columns=['relation', 'relation_name'])

    # Select necessary columns and ensure correct data types
    result_df = triples_with_ids[['entity_id_1', 'relation_id', 'entity_id_2']].astype({'entity_id_1': int, 'relation_id': int, 'entity_id_2': int})

    entities_df.to_csv(entities_csv_path, index=False)
    relations_df.to_csv(relations_csv_path, index=False)
    result_df.to_csv(triples_csv_path, index=False)


def build_graphml(data_dir):
    """Build the knowledge graph from the saved CSVs and write knowledge_graph.graphml."""
    triples_df = pd.read_csv(os.path.join(data_dir, "triples.csv"))
    entities_df = pd.read_csv(os.path.join(data_dir, "entities.csv"))
    relations_df = pd.read_csv(os.path.join(data_dir, "relations.csv"))

    entity_name_map = entities_df.set_index("entity_id")["entity_name"].to_dict()
    relation_name_map = relations_df.set_index("relation_id")["relation_name"].to_dict()

    G = nx.from_pandas_edgelist(
        triples_df,
        source="entity_id_1",
        target="entity_id_2",
        edge_attr="relation_id",
        create_using=nx.DiGraph,
    )
    G = nx.relabel_nodes(G, entity_name_map)
    edge_attributes = nx.get_edge_attributes(G, "relation_id")
    nx.set_edge_attributes(G, {
        (u, v): relation_name_map[edge_attributes[(u, v)]]
        for u, v in G.edges()
        if edge_attributes[(u, v)] in relation_name_map
    }, "relation")

    graphml_path = os.path.join(data_dir, "knowledge_graph.graphml")
    nx.write_graphml(G, graphml_path)
    return graphml_path


if __name__ == "__main__":
    # Standalone run: python -m utils.lc_graph (uses NVIDIA_API_KEY from the environment)
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    import config

    llm = ChatNVIDIA(model=config.LLM_MODELS[0])
    out_dir = "data"
    documents, results = process_documents("papers/", llm, out_dir)
    save_triples_to_csvs(results, out_dir)
    print("Graph written to", build_graphml(out_dir))
