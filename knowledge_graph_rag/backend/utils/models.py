import warnings

from fastapi import HTTPException
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

import config

# The pinned client only knows a mid-2024 model table; models missing from it still work
# if NVIDIA's live listing has them, but each construction warns about it.
warnings.filterwarnings("ignore", message=".*type is unknown and inference may fail.*")


def get_chat_models():
    return list(config.LLM_MODELS)


def make_llm(ws, model_id=None, **kwargs):
    """ChatNVIDIA bound to the caller's own API key."""
    model = model_id or config.LLM_MODELS[0]
    if model not in config.LLM_MODELS:
        raise HTTPException(400, f"Model '{model}' is not enabled on this server.")
    # reasoning models spend hundreds of hidden tokens thinking before they answer; with the
    # client's default 1024 limit, long prompts come back empty
    kwargs.setdefault("max_tokens", 4096)
    return ChatNVIDIA(model=model, api_key=ws.api_key, **kwargs)


def without_reasoning(llm):
    """For structured extraction (triples, entities): skip the model's hidden reasoning.
    ~7x faster with nemotron-3 and the output is just as parseable."""
    if not config.DISABLE_REASONING_FOR_EXTRACTION:
        return llm
    return llm.bind(chat_template_kwargs={"enable_thinking": False})


def make_embeddings(ws):
    return NVIDIAEmbeddings(model=config.EVAL_EMBED_MODEL, api_key=ws.api_key)
