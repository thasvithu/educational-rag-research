"""Strict Phase 1 configuration; paths are relative to the project root."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def within(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Expected a nonempty relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or ".." in Path(relative).parts:
        raise ValueError(f"Path escapes its root: {relative!r}")
    return path


@dataclass(frozen=True)
class Config:
    schema_version: int
    corpus_dir: str
    source_dir: str
    output_root: str
    expected_documents: int
    expected_pages: int
    expected_members: dict[str, int]
    gap_policy: str
    sample_characters: int
    seed: int

    def validate(self, root: Path) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported configuration schema")
        for key in ("expected_documents", "expected_pages", "sample_characters"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.expected_members, dict) or not self.expected_members:
            raise ValueError("expected_members must be a nonempty mapping")
        if any(not isinstance(k, str) or not k or type(v) is not int or v <= 0
               for k, v in self.expected_members.items()):
            raise ValueError("Invalid member counts")
        if sum(self.expected_members.values()) != self.expected_documents:
            raise ValueError("Member counts do not sum to expected_documents")
        if self.gap_policy != "excluded_and_blank_pages":
            raise ValueError("Only excluded_and_blank_pages is supported in Phase 1")
        corpus, source, output = (within(root, getattr(self, name))
                                  for name in ("corpus_dir", "source_dir", "output_root"))
        if corpus == source or corpus.is_relative_to(source) or source.is_relative_to(corpus):
            raise ValueError("Corpus and raw source directories must be separate")
        if not output.is_relative_to((root / "data/outputs/rag").resolve()):
            raise ValueError("Outputs must be inside data/outputs/rag")
        if output.is_relative_to(corpus) or corpus.is_relative_to(output) or source.is_relative_to(output):
            raise ValueError("Output path overlaps input data")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path, root: Path = ROOT) -> Config:
    try:
        config = Config(**json.loads(path.read_text(encoding="utf-8")))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    config.validate(root)
    return config


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Run ID must contain 1–80 letters, digits, underscores or hyphens")
    return value
