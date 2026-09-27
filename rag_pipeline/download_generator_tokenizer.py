"""Download the pinned generation tokenizer files only (no model weights)."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

from .common.config import ROOT, within
from .common.load_corpus import file_hash, require
from .common.run_manifest import write_json


def download(config_path):
    identity = json.loads(Path(config_path).read_text())['tokenizer']
    folder = within(ROOT, identity['directory'])
    require(folder.is_relative_to(ROOT / 'data/outputs/rag'), 'Tokenizer must be saved under ignored research outputs')
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in identity['files'].items():
        require(name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'), 'Unexpected tokenizer file')
        target = within(folder, name)
        if target.exists():
            require(file_hash(target) == expected, 'Existing tokenizer file differs; preserve and investigate it')
            continue
        url = f'https://huggingface.co/{identity["repository"]}/resolve/{identity["revision"]}/{name}'
        data = urllib.request.urlopen(url, timeout=60).read()
        require(sha256(data).hexdigest() == expected, 'Downloaded tokenizer checksum differs')
        # Exclusive creation prevents silently replacing a concurrent download.
        with target.open('xb') as handle:
            handle.write(data)
    manifest = {'schema_version': 1, **{k: identity[k] for k in ('repository', 'revision', 'files')}}
    path = folder / 'manifest.json'
    if path.exists():
        require(json.loads(path.read_text()) == manifest, 'Existing tokenizer manifest differs')
    else:
        write_json(path, manifest)
    print(f'Pinned tokenizer verified: {folder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-config', type=Path, default=ROOT / 'rag_pipeline/configs/generation.json')
    download(parser.parse_args().generation_config)
