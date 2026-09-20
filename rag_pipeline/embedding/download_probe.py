"""Download the exact probe bundle, verifying every pinned file before use."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from ..common.config import ROOT, within
from ..common.load_corpus import file_hash, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    require(bundle.is_relative_to(ROOT / "data/outputs/rag"), "Bundle must remain under ignored data/outputs/rag")
    manifest = json.loads(Path(__file__).with_name("probe_bundle.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = within(bundle, name)
        if path.exists():
            require(file_hash(path) == expected, f"Existing bundle file differs: {name}")
            continue
        prefix, relative = name.split("/", 1)
        repo = "jinaai/jina-embeddings-v2-small-en" if prefix == "model" else "jinaai/jina-bert-implementation"
        url = f"https://huggingface.co/{repo}/resolve/{manifest['repos'][repo]}/{relative}"
        with urlopen(url, timeout=120) as response:
            data = response.read()
        require(hashlib.sha256(data).hexdigest() == expected, f"Downloaded hash mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        print(f"Verified {name} ({len(data)} bytes)", flush=True)
    print(f"Bundle ready: {bundle}")


if __name__ == "__main__":
    main()
