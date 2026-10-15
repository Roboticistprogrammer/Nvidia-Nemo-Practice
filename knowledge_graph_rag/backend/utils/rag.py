"""Retrieval helpers shared by the chat and evaluation routers, scoped to one workspace."""
import ast
import json
import re

import networkx as nx
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

from utils.models import without_reasoning
from vectorstore.search import SearchHandler

SYSTEM_PROMPT = "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."

ENTITY_PROMPT = """Return a JSON with a single key 'entities' and list of entities within this user query. Each element in your list MUST BE part of the user's query. Do not provide any explanation. If the returned list is not parseable in Python, you will be heavily penalized. For example, input: 'What is the difference between Apple and Google?' output: ['Apple', 'Google']. Always follow this output format. Here's the user query: """

NO_TRIPLES_CONTEXT = "No graph triples were available to extract from the knowledge graph. Always provide a disclaimer if you know the answer to the user's question, since it is not grounded in the knowledge you are provided from the graph."


def graph_path(ws):
    return ws.file("knowledge_graph.graphml")


def load_graph(ws):
    """The workspace's knowledge graph, or None if it hasn't been built yet."""
    path = graph_path(ws)
    if not path.exists():
        return None
    return NetworkxEntityGraph(nx.read_graphml(path))


def extract_entities(llm, query):
    """Ask the LLM for the entities in `query`; tolerant of code fences and list-only replies."""
    raw = without_reasoning(llm).invoke(ENTITY_PROMPT + query).content or ""
    obj = re.search(r"\{.*\}", raw, re.DOTALL)
    if obj:
        try:
            return list(json.loads(obj.group(0)).get("entities", []))
        except (json.JSONDecodeError, AttributeError):
            pass
    lst = re.search(r"\[.*\]", raw, re.DOTALL)
    if lst:
        try:
            return [str(e) for e in ast.literal_eval(lst.group(0))]
        except (ValueError, SyntaxError):
            pass
    return []


def search_passages(ws, query, k=5):
    handler = SearchHandler(ws.collection, use_bge_m3=True, use_reranker=True)
    return [item.text for item in handler.search_and_rerank(query, k=k)]


def graph_triplets(graph, entities, depth=2):
    triplets = []
    for entity in entities:
        triplets.extend(graph.get_entity_knowledge(entity, depth=depth))
    return triplets


def passages_context(passages):
    return "Here are the relevant passages from the knowledge base: \n\n" + "\n".join(passages)


def triplets_context(triplets):
    return "Here are the relationships from the knowledge graph: " + "\n".join(triplets)


def answer_with_context(llm, context, question):
    reply = llm.invoke([("system", SYSTEM_PROMPT), ("user", f"Context: {context}\n\nUser query: {question}")])
    return reply.content or ""
