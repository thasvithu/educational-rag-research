"""Prepare LangChain's ChatGroq client without making a network request."""

import json
from pathlib import Path

from langchain_groq import ChatGroq

from ..common.config import ROOT
from ..common.load_corpus import require


def create_chat_model(model_id: str | None = None,
                      config_path: Path = ROOT / "rag_pipeline/configs/models.json") -> ChatGroq:
    """Create the future generation client; calling invoke() is a separate action.

    Select an available free-tier model at the generation stage and put the API
    key in GROQ_API_KEY. Never store keys in configuration or source code.
    The free-tier setting records our budget policy; it cannot verify the tier of
    a Groq account. Use a free-tier account, not a paid account with the same model.
    """
    settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
    require(settings.get("schema_version") == 1, "Unsupported model configuration")
    generation = settings["generation"]
    require(generation["provider"] == "groq" and generation["integration"] == "ChatGroq",
            "This adapter supports the configured Groq provider only")
    require(generation["account_tier"] == "free" and generation["paid_budget"] == 0,
            "This research is configured for Groq free tier and zero paid budget")
    require(generation["api_key_environment_variable"] == "GROQ_API_KEY",
            "ChatGroq reads its key from GROQ_API_KEY")
    selected_model = model_id or generation["model_id"]
    require(isinstance(selected_model, str) and bool(selected_model.strip()),
            "Choose an available Groq free-tier model before generation; no default model is assumed")
    # LangChain supplies the provider client and basic retry handling. The later
    # experiment runner will add request caching and daily-budget scheduling.
    # This pinned ChatGroq version converts temperature=0 to 1e-8 internally.
    # Store the actual value explicitly so saved configuration matches the client.
    return ChatGroq(model=selected_model, temperature=generation["temperature"],
                    max_tokens=generation["max_output_tokens"],
                    timeout=generation["timeout_seconds"], max_retries=generation["max_retries"])
