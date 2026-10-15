"""Prepare LangChain's ChatOpenAI client without making a network request."""

import json
from pathlib import Path

from langchain_openai import ChatOpenAI

from ..common.config import ROOT
from ..common.load_corpus import require


def create_chat_model(model_id: str | None = None,
                      config_path: Path = ROOT / "rag_pipeline/configs/models.json") -> ChatOpenAI:
    """Create the future generation client; calling invoke() is a separate action.

    Select an available model at the generation stage and put the API key in
    OPENAI_API_KEY. Never store keys in configuration or source code.
    """
    settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
    require(settings.get("schema_version") == 1, "Unsupported model configuration")
    generation = settings["generation"]
    require(generation["provider"] == "openai" and generation["integration"] == "ChatOpenAI",
            "This adapter supports the configured OpenAI provider only")
    require(generation["account_tier"] == "paid",
            "This research is configured for a paid OpenAI account")
    require(generation["api_key_environment_variable"] == "OPENAI_API_KEY",
            "ChatOpenAI reads its key from OPENAI_API_KEY")
    selected_model = model_id or generation["model_id"]
    require(isinstance(selected_model, str) and bool(selected_model.strip()),
            "Choose an available OpenAI model before generation; no default model is assumed")
    # LangChain supplies the provider client and basic retry handling. The later
    # experiment runner will add request caching and daily-budget scheduling.
    return ChatOpenAI(model=selected_model, temperature=generation["temperature"],
                      max_tokens=generation["max_output_tokens"],
                      timeout=generation["timeout_seconds"], max_retries=generation["max_retries"])
