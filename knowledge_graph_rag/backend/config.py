"""Runtime settings, read once from environment variables (and .env for local dev)."""
import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _str(name, default=""):
    # treat empty values (e.g. `LLM_MODELS=` in .env) as unset
    return os.getenv(name) or default


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Root folder for all per-user workspaces (uploads, CSVs, graphml);
# relative paths are taken from the project root, not the current directory
DATA_DIR = (PROJECT_ROOT / _str("DATA_DIR", "backend/data")).resolve()

# Milvus: a local/compose instance (http://standalone:19530) or Zilliz Cloud (uri + token)
MILVUS_URI = _str("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = _str("MILVUS_TOKEN")

# Comma-separated hosted NVIDIA models offered in the UI; the first one is the default.
LLM_MODELS = [m.strip() for m in _str(
    "LLM_MODELS", "nvidia/nemotron-3-super-120b-a12b,nvidia/nemotron-3-ultra-550b-a55b"
).split(",") if m.strip()]
EVAL_EMBED_MODEL = _str("EVAL_EMBED_MODEL", "nvidia/nemotron-3-embed-1b")
# Reward-model scoring is disabled unless a live reward model is configured
REWARD_MODEL = _str("REWARD_MODEL")

# Auth: users bring their own NVIDIA API key. The server's own NVIDIA_API_KEY is only used
# as a fallback when ALLOW_SERVER_KEY=true (handy for local dev, never for a public deploy).
ALLOW_SERVER_KEY = _bool("ALLOW_SERVER_KEY")
SERVER_API_KEY = _str("NVIDIA_API_KEY")
# Secret mixed into workspace ids so they can't be derived from a key by someone else
WORKSPACE_SECRET = _str("WORKSPACE_SECRET")
KEY_CHECK_MODEL = _str("KEY_CHECK_MODEL", EVAL_EMBED_MODEL)
KEY_CACHE_TTL_S = int(_str("KEY_CACHE_TTL_S", "900"))

MAX_UPLOAD_MB = int(_str("MAX_UPLOAD_MB", "50"))
# Parallel LLM calls during triple extraction; keep low for free-tier rate limits
EXTRACTION_WORKERS = int(_str("EXTRACTION_WORKERS", "4"))
# Send chat_template_kwargs.enable_thinking=false for triple/entity extraction. Set to false
# if you configure a model that rejects that parameter.
DISABLE_REASONING_FOR_EXTRACTION = _bool("DISABLE_REASONING_FOR_EXTRACTION", True)
EMBED_DEVICE = _str("EMBED_DEVICE", "cpu")

if not WORKSPACE_SECRET:
    WORKSPACE_SECRET = "dev-only-insecure-secret"
    logger.warning("WORKSPACE_SECRET is not set; using an insecure default. Set it in production.")
