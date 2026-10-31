"""Download only real CSI recordings, pinned to a public RoboFiSense commit.

Git blob hashes verify each file. Existing verified files are reused. --per-class
limits download size for a basic experiment; default downloads all 1103 recordings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
from common import ROOT, save_json

COMMIT = '7cc84af1229b62d48f82392d1fed1422f3c9223b'
REPO = 'SiamiLab/RoboFiSense'


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'zen-noifi-basic-experiment'}), timeout=120) as response:
        return response.read()


def blob_hash(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-class', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = ROOT/'data/RoboFiSense'; root.mkdir(parents=True, exist_ok=True)
    tree_path = root/'source_tree.json'
    if not tree_path.exists():
        tree = json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'))
        if tree.get('truncated'):
            raise RuntimeError('Incomplete source tree')
        save_json(tree_path, tree)
    tree = json.loads(tree_path.read_text())
    files = [r for r in tree['tree'] if r['path'].endswith('.cmplx')]
    if args.per_class:
        import random
        rng = random.Random(42)
        selected = []
        for label in sorted({r['path'].split('/')[2] for r in files}):
            candidates = sorted([r for r in files if r['path'].split('/')[2] == label], key=lambda r: r['path'])
            selected.extend(rng.sample(candidates, min(args.per_class, len(candidates))))
        files = selected
    print(f'Downloading {len(files)} CSI recordings ({sum(r["size"] for r in files)/1e9:.2f} GB)', flush=True)
    save_json(root/'source.json', {'repository': f'https://github.com/{REPO}', 'commit': COMMIT,
                                  'selected_recordings': len(files), 'download_status': 'in_progress'})

    def download(item):
        path = root/item['path']
        if path.is_file() and blob_hash(path.read_bytes()) == item['sha']:
            return
        for attempt in range(4):
            try:
                data = fetch(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{quote(item["path"])}')
                if len(data) != item['size'] or blob_hash(data) != item['sha']:
                    raise ValueError(f'Hash mismatch: {path}')
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.download')
                temp.write_bytes(data); temp.replace(path)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item) for item in files]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 50 == 0 or count == len(files):
                print(f'Verified {count}/{len(files)} recordings', flush=True)
    save_json(root/'source.json', {'repository': f'https://github.com/{REPO}', 'commit': COMMIT,
                                  'selected_recordings': len(files), 'download_status': 'completed'})


if __name__ == '__main__':
    main()
