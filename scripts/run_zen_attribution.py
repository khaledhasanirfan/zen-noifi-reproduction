"""Run upstream ZEN attribution on the authors' supplied recovered CMRs.

No raw RAM recovery or Python 3.8 bytecode execution is claimed here. The only
adaptation is rebasing artifact paths to this checkout. Similarity is upstream code.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np

from common import ROOT, save_csv, save_json

ZEN = ROOT/'third_party/ZEN/volatility3'
BASES = ['yolov5', 'yolov7', 'resnet', 'mobilenetv2', 'nanoGPT', 'llama2', 'yolov8']
QUERIES = {'yolov5-involution': 'yolov5', 'yolov5-KD': 'yolov5', 'yolov7-CA': 'yolov7',
           'yolov7-GAM': 'yolov7', 'resnet-se': 'resnet', 'resnet-ghost': 'resnet',
           'mobilenetv2-former': 'mobilenetv2', 'mobilenetv2-ghost': 'mobilenetv2',
           'nanoGPT-LORA': 'nanoGPT', 'nanoGPTRWKV': 'nanoGPT', 'llama2-pt': 'llama2',
           'yolov8-Gold': 'yolov8', 'yolov8-spd': 'yolov8'}


def load_cmr(module, name):
    raw = (ZEN/'CMRs'/f'{name}_CMR').read_text()
    pr, mr = map(json.loads, raw.split('____DIVIDE_BETWEEN_PR_AND_MR____'))
    # Upstream serializes ./PR/... paths relative to volatility3 cwd.
    for key in ['CO_paths', 'CO_code_paths', 'DC_paths']:
        for func, value in pr[key].items():
            if value.startswith('./'):
                value = value[2:]
            if value.startswith('PR/'):
                artifact = ZEN/value
            else:
                raise ValueError(f'Unexpected upstream artifact path: {value}')
            pr[key][func] = str(artifact)
    instance = module.CMR.__new__(module.CMR)
    instance.model_name = name
    instance.MR = module.MR(None, None, MR_dict=mr)
    instance.PR = module.PR(None, None, PR_dict=pr)
    instance.mr_sim_weights = np.array([0.25, 0.50, 0.25])
    instance.pr_sim_weights = np.array([0.80, 0.10, 0.10])
    instance.cmr_sim_weights = np.array([0.25, 0.75])
    return instance


def main():
    sys.dont_write_bytecode = True
    out = ROOT/'results/zen-attribution'; out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').unlink(missing_ok=True)
    source = ZEN/'utils/MAI_cmr_gen.py'
    spec = importlib.util.spec_from_file_location('zen_cmr', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows, predictions, inventory, coverage = [], [], [], []
    for name in list(QUERIES)+BASES:
        cmr = load_cmr(module, name)
        paths = list(cmr.PR.CO_code_paths.values())
        coverage.append({'model': name, 'referenced_bytecode_files': len(paths),
                         'available_bytecode_files': sum(Path(p).is_file() for p in paths),
                         'missing_bytecode_files': sum(not Path(p).is_file() for p in paths)})
    for query, expected in QUERIES.items():
        sims = []
        for base in BASES:
            # Upstream comparison consumes base PR entries; reload each pair.
            a, b = load_cmr(module, query), load_cmr(module, base)
            with contextlib.redirect_stdout(io.StringIO()):
                result = a.compare_CMRs(b)
            mr, pr, cmr, identical, changed, added, _, _ = result
            if not np.isfinite([mr, pr, cmr]).all():
                raise ValueError('Non-finite attribution score')
            rows.append({'query': query, 'base': base, 'mr_similarity': float(mr),
                         'pr_similarity': float(pr), 'cmr_similarity': float(cmr),
                         'identical_functions': len(identical), 'changed_functions': len(changed),
                         'added_functions': len(added)})
            sims.append((float(cmr), base))
        sims.sort(reverse=True)
        query_rows = [row for row in rows if row['query'] == query]
        mr_top = max(query_rows, key=lambda row: row['mr_similarity'])['base']
        pr_top = max(query_rows, key=lambda row: row['pr_similarity'])['base']
        # Simple extension: record top-two margin as an ambiguity indicator.
        predictions.append({'query': query, 'expected_base': expected, 'predicted_base': sims[0][1],
                            'correct': sims[0][1] == expected, 'top1_score': sims[0][0],
                            'top2_base': sims[1][1], 'top2_score': sims[1][0],
                            'margin': sims[0][0]-sims[1][0],
                            'mr_only_prediction': mr_top, 'mr_only_correct': mr_top == expected,
                            'pr_only_prediction': pr_top, 'pr_only_correct': pr_top == expected})
        original = load_cmr(module, query)
        inventory.append({'query': query, 'layers': original.MR.layer_count,
                          'tensors': original.MR.tensor_count, 'weights': original.MR.weight_count})
        print(f'{query}: {sims[0][1]}, margin={sims[0][0]-sims[1][0]:.4f}', flush=True)
    save_csv(out/'similarity.csv', rows)
    save_csv(out/'predictions.csv', predictions)
    save_csv(out/'inventory.csv', inventory)
    save_csv(out/'artifact_coverage.csv', coverage)
    save_json(out/'summary.json', {'status': 'completed', 'scope': 'Attribution from supplied recovered artifacts',
              'queries': len(predictions), 'base_candidates': BASES,
              'correct': sum(r['correct'] for r in predictions),
              'mr_only_correct': sum(r['mr_only_correct'] for r in predictions),
              'pr_only_correct': sum(r['pr_only_correct'] for r in predictions),
              'accuracy': sum(r['correct'] for r in predictions)/len(predictions),
              'algorithm_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'limitations': ['Authors supplied these recovered artifacts; no fresh memory recovery performed.',
                'Top-two margin is an ambiguity indicator, not a calibrated confidence probability.',
                'Upstream artifacts contain missing bytecode/decompilation files; original missing-file fallbacks retained. See artifact_coverage.csv.',
                'Full supplied-dump recovery and YOLO patch inference require separate compatible setup.']})


if __name__ == '__main__':
    main()
