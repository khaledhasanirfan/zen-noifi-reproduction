"""Compact NoiFi implementation of paper Eq. 8 on real RoboFiSense recordings.

This uses a smaller CNN/DAE and 64x64 inputs to fit a laptop. It is a reduced
experiment, not a reproduction of the paper's full architecture or headline scores.
"""
import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import ROOT, save_csv, save_json

CLASSES = ['Arc', 'Elbow', 'Rectangle', 'Silence', 'SLFW', 'SLRL', 'SLUD', 'Triangle']
NULLS = [0, 1, 2, 3, 4, 115, 116, 117, 118, 119, 120, 122, 123, 124,
         125, 126, 127, 128, 129, 130, 131, 253, 254, 255]


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(32*8*8, 128), nn.ReLU())
        self.head = nn.Linear(128, 8)

    def feature(self, x):
        return self.features(self.conv(x))

    def forward(self, x):
        return self.head(self.feature(x))


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4096, 256), nn.ReLU(),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(),
                                 nn.Linear(256, 4096), nn.Sigmoid(), nn.Unflatten(1, (1, 64, 64)))

    def forward(self, x):
        return self.net(x)


def load_data(data_dir, per_class, seed):
    """One sample per .cmplx recording, never split a recording across partitions."""
    directory = data_dir / 'RoboFiSense Dataset'
    if not directory.is_dir():
        raise FileNotFoundError(f'Dataset missing: {directory}. Run bash scripts/download_data.sh')
    rng = np.random.default_rng(seed)
    paths = []
    for label in CLASSES:
        candidates = sorted(directory.glob(f'*/{label}/*/*.cmplx'))
        if len(candidates) < 12:
            raise ValueError(f'Need at least 12 recordings for {label}; found {len(candidates)}')
        ids = rng.permutation(len(candidates))[:per_class] if per_class else np.arange(len(candidates))
        paths.extend((candidates[i], CLASSES.index(label)) for i in ids)
    xs, ys, manifest = [], [], []
    for path, y in paths:
        raw = path.read_bytes()
        # Same public pickle recordings read by upstream get_robofi.py.
        csi = pickle.loads(raw)[0]['complex_csi']
        amplitude = np.abs(csi).astype(np.float32)
        if amplitude.ndim != 2 or amplitude.shape[1] != 256:
            raise ValueError(f'Unexpected CSI shape {amplitude.shape}: {path}')
        amplitude = np.delete(amplitude, NULLS, axis=1)
        amplitude = np.pad(amplitude, ((0, 0), (64, 64))).T.copy()
        low, high = amplitude.min(), amplitude.max()
        if not np.isfinite(amplitude).all() or high <= low:
            raise ValueError(f'Invalid recording: {path}')
        x = torch.from_numpy((amplitude-low)/(high-low))[None, None]
        x = F.interpolate(x, (64, 64), mode='bilinear', align_corners=False, antialias=True)
        xs.append(x[0]); ys.append(y)
        manifest.append({'recording_id': len(manifest), 'path': str(path.relative_to(data_dir)),
                         'label': CLASSES[y], 'sha256': hashlib.sha256(raw).hexdigest()})
    hashes = [row['sha256'] for row in manifest]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate source recordings found; deduplicate before splitting')
    return torch.stack(xs), torch.tensor(ys), manifest


def train(model, x, y, vx, vy, epochs, device, denoise=False):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    history, best_loss, best_state = [], float('inf'), None
    for epoch in range(epochs):
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            inp = (xb + 0.05*torch.randn_like(xb)).clamp(0, 1) if denoise else xb
            output = model(inp)
            loss = F.mse_loss(output, xb) if denoise else F.cross_entropy(output, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vout = model(vx.to(device))
            val_loss = (F.mse_loss(vout, vx.to(device)) if denoise
                        else F.cross_entropy(vout, vy.to(device))).item()
            accuracy = None if denoise else (vout.argmax(1).cpu() == vy).float().mean().item()
        history.append({'epoch': epoch+1, 'train_loss': float(np.mean(losses)),
                        'validation_loss': val_loss, 'validation_accuracy': accuracy})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f'{"DAE" if denoise else "CNN"} {epoch+1}/{epochs}: loss={val_loss:.5f}, accuracy={accuracy}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return history


def attack(model, x, y, name, eps, generator):
    original = x.detach()
    if name == 'FGSM':
        candidate = original.clone().requires_grad_()
        grad = torch.autograd.grad(F.cross_entropy(model(candidate), y), candidate)[0]
        return (original + eps*grad.sign()).clamp(0, 1).detach()
    delta = (torch.rand(original.shape, generator=generator)*2-1).to(x.device)*eps
    candidate = (original+delta).clamp(0, 1)
    for _ in range(20):
        candidate = candidate.detach().requires_grad_()
        grad = torch.autograd.grad(F.cross_entropy(model(candidate), y), candidate)[0]
        candidate = candidate + (eps/5)*grad.sign()
        candidate = torch.maximum(torch.minimum(candidate, original+eps), original-eps).clamp(0, 1)
    return candidate.detach()


@torch.no_grad()
def score_noifi(model, dae, x, n, seed):
    """MED between original residual features and each sample's N permutations."""
    rng = torch.Generator().manual_seed(seed)
    scores = []
    start = time.perf_counter()
    for sample in x.split(32):
        residual = sample-dae(sample)
        features = model.feature(residual)
        distances = []
        for _ in range(n):
            # Each recording gets its own permutation; all N share the same residual.
            shuffled = torch.stack([r.flatten()[torch.randperm(r.numel(), generator=rng).to(r.device)]
                                    .reshape_as(r) for r in residual])
            distances.append(torch.linalg.vector_norm(features-model.feature(shuffled), dim=1))
        scores.append(torch.stack(distances).mean(0).cpu().numpy())
    return np.concatenate(scores), (time.perf_counter()-start)*1000/len(x)


def metrics(benign, adversarial, calibration, fpr_target):
    threshold = float(np.quantile(calibration, 1-fpr_target, method='higher'))
    labels = np.r_[np.zeros(len(benign)), np.ones(len(adversarial))]
    scores = np.r_[benign, adversarial]
    return {'auroc': float(roc_auc_score(labels, scores)),
            'auprc': float(average_precision_score(labels, scores)),
            'recall_at_calibrated_threshold': float(np.mean(adversarial > threshold)),
            'observed_test_fpr': float(np.mean(benign > threshold)), 'threshold': threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/RoboFiSense')
    parser.add_argument('--output', type=Path, default=ROOT/'results/noifi-basic')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--ae-epochs', type=int, default=30)
    parser.add_argument('--per-class', type=int, default=0, help='0 uses all recordings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eps', type=float, default=0.10)
    parser.add_argument('--n-values', type=int, nargs='+', default=[1, 3, 5, 10, 20])
    parser.add_argument('--score-seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--reuse', action='store_true', help='Reuse checkpoints only when experiment config matches')
    args = parser.parse_args()
    if min(args.epochs, args.ae_epochs) < 1 or any(n < 1 for n in args.n_values):
        parser.error('Epochs and N must be positive')
    if not 0 < args.eps <= 1:
        parser.error('epsilon must be in (0, 1]')
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    # A stale completed marker must not survive a failed rerun.
    (out/'summary.json').unlink(missing_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cpu')
    print('Loading real CSI recordings...', flush=True)
    x, y, manifest = load_data(args.data, args.per_class, args.seed)
    ids = np.arange(len(x))
    train_ids, rest = train_test_split(ids, test_size=0.4, random_state=args.seed, stratify=y.numpy())
    valid_ids, rest = train_test_split(rest, test_size=0.75, random_state=args.seed, stratify=y[rest].numpy())
    cal_ids, test_ids = train_test_split(rest, test_size=2/3, random_state=args.seed, stratify=y[rest].numpy())
    splits = {'train': train_ids, 'validation': valid_ids, 'calibration': cal_ids, 'test': test_ids}
    for name, indices in splits.items():
        for i in indices:
            manifest[i]['split'] = name
    save_csv(out/'recordings.csv', manifest)
    if (args.data/'source.json').is_file():
        source = json.loads((args.data/'source.json').read_text())
        if source['download_status'] != 'completed':
            raise ValueError('Dataset download is incomplete')
        dataset_commit = source['commit']
    else:
        dataset_commit = subprocess.check_output(['git', '-C', str(args.data), 'rev-parse', 'HEAD'], text=True).strip()
    config = {'seed': args.seed, 'epochs': args.epochs, 'ae_epochs': args.ae_epochs,
              'per_class': args.per_class, 'input_shape': [1, 64, 64], 'epsilon': args.eps,
              'batch_size': 32, 'learning_rate': 0.001, 'dae_noise_std': 0.05,
              'calibration_fpr_target': 0.05, 'pgd_steps': 20, 'pgd_step_size': args.eps/5,
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'n_values': args.n_values, 'score_seeds': args.score_seeds, 'device': str(device),
              'split_counts': {k: len(v) for k, v in splits.items()},
              'dataset_commit': dataset_commit,
              'manifest_sha256': hashlib.sha256((out/'recordings.csv').read_bytes()).hexdigest()}
    models = out/'models'; models.mkdir(exist_ok=True)
    config_path = out/'config.json'
    reuse = args.reuse and config_path.exists() and json.loads(config_path.read_text()) == config
    if args.reuse and not reuse:
        raise ValueError('Checkpoint configuration differs or is missing; rerun without --reuse')
    save_json(config_path, config)
    cnn, dae = Classifier(), Denoiser()
    for name, model, denoise in [('classifier', cnn, False), ('dae', dae, True)]:
        checkpoint = models/f'{name}.pt'
        if reuse:
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); model.eval()
        else:
            # DAE uses only the training partition; validation chooses checkpoints.
            history = train(model, x[train_ids], y[train_ids], x[valid_ids], y[valid_ids],
                            args.ae_epochs if denoise else args.epochs, device, denoise)
            save_csv(out/f'{name}_training.csv', history)
            torch.save(model.state_dict(), checkpoint)
    tx, ty = x[test_ids], y[test_ids]
    calx = x[cal_ids]
    with torch.no_grad():
        pred = cnn(tx).argmax(1)
        clean_accuracy = (pred == ty).float().mean().item()
        clean_correct = pred == ty
    if not clean_correct.any():
        raise RuntimeError('No correctly classified test recordings; ASR is undefined')
    rng = torch.Generator().manual_seed(args.seed+100)
    benign = (tx + (torch.rand(tx.shape, generator=rng)*2-1)*args.eps).clamp(0, 1)
    calben = (calx + (torch.rand(calx.shape, generator=rng)*2-1)*args.eps).clamp(0, 1)
    with torch.no_grad():
        benign_accuracy = (cnn(benign).argmax(1) == ty).float().mean().item()
    attacks = {name: attack(cnn, tx, ty, name, args.eps, rng) for name in ['FGSM', 'PGD']}
    attacks_info = []
    for name, adv in attacks.items():
        with torch.no_grad():
            adv_pred = cnn(adv).argmax(1)
        asr = (adv_pred[clean_correct] != ty[clean_correct]).float().mean().item()
        attacks_info.append({'attack': name, 'clean_accuracy': clean_accuracy,
                             'adversarial_accuracy': (adv_pred == ty).float().mean().item(),
                             'attack_success_rate_on_clean_correct': asr,
                             'clean_correct_count': int(clean_correct.sum()), 'test_count': len(tx),
                             'uniform_noise_control_accuracy': benign_accuracy,
                             'max_linf': (adv-tx).abs().max().item()})
        if (adv-tx).abs().max().item() > args.eps+1e-6:
            raise AssertionError('Attack exceeds epsilon bound')
    save_csv(out/'attack_metrics.csv', attacks_info)
    rows, score_rows, curve_rows = [], [], []
    with torch.no_grad():
        baseline_start = time.perf_counter()
        mse_cal = (calben-dae(calben)).square().flatten(1).mean(1).numpy()
        mse_ben = (benign-dae(benign)).square().flatten(1).mean(1).numpy()
        baseline_ms = (time.perf_counter()-baseline_start)*1000/(len(calben)+len(benign))
        for name, adv in attacks.items():
            baseline_start = time.perf_counter()
            mse_adv = (adv-dae(adv)).square().flatten(1).mean(1).numpy()
            adv_ms = (time.perf_counter()-baseline_start)*1000/len(adv)
            rows.append({'attack': name, 'detector': 'Reconstruction MSE', 'n': 0, 'score_seed': -1,
                         **metrics(mse_ben, mse_adv, mse_cal, 0.05), 'ms_per_sample': (baseline_ms+adv_ms)/2})
    for n in args.n_values:
        for seed in args.score_seeds:
            cal_score, _ = score_noifi(cnn, dae, calben, n, seed)
            ben_score, ben_ms = score_noifi(cnn, dae, benign, n, seed)
            for name, adv in attacks.items():
                adv_score, adv_ms = score_noifi(cnn, dae, adv, n, seed)
                row = {'attack': name, 'detector': 'NoiFi MED', 'n': n, 'score_seed': seed,
                       **metrics(ben_score, adv_score, cal_score, 0.05),
                       'ms_per_sample': (ben_ms+adv_ms)/2}
                rows.append(row)
                print(f'N={n}, seed={seed}, {name}: AUROC={row["auroc"]:.3f}, '
                      f'recall={row["recall_at_calibrated_threshold"]:.3f}', flush=True)
                for kind, vals in [('benign', ben_score), ('adversarial', adv_score)]:
                    score_rows.extend({'attack': name, 'n': n, 'score_seed': seed, 'kind': kind,
                                       'recording_id': int(test_ids[i]), 'score': float(s)} for i, s in enumerate(vals))
                if seed == args.score_seeds[0] and n == 5:
                    fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(tx)), np.ones(len(tx))], np.r_[ben_score, adv_score])
                    curve_rows.extend({'attack': name, 'fpr': float(a), 'tpr': float(b)} for a, b in zip(fpr, tpr))
    save_csv(out/'detection_metrics.csv', rows)
    save_csv(out/'scores.csv', score_rows)
    if curve_rows:
        save_csv(out/'roc.csv', curve_rows)
    # Actual per-sample variability across seeds, then average over samples.
    stability = []
    for name in attacks:
        for n in args.n_values:
            for kind in ['benign', 'adversarial']:
                arrays = [[r['score'] for r in score_rows if r['attack'] == name and r['n'] == n
                           and r['kind'] == kind and r['score_seed'] == seed] for seed in args.score_seeds]
                stability.append({'attack': name, 'n': n, 'kind': kind,
                                  'mean_sample_score_std': float(np.std(arrays, axis=0).mean())})
    save_csv(out/'stability.csv', stability)
    save_json(out/'summary.json', {'status': 'completed', 'scope': 'Reduced real-data NoiFi experiment',
              'clean_accuracy': clean_accuracy, 'recordings': len(x), 'config': config,
              'uniform_noise_control_accuracy': benign_accuracy,
              'hardware': platform.machine(), 'python': platform.python_version(), 'torch': torch.__version__,
              'limitations': ['64x64 resized inputs; compact CNN and dense DAE, unlike full paper architecture.',
                'Recording-level stratified split mixes locations; no cross-location generalization claim.',
                'All test attacks are evaluated, including unsuccessful attacks; digital white-box only.',
                'Uniform random perturbations serve as the benign control; their classifier accuracy is separately recorded.',
                '5% calibration FPR target; empirical test FPR is reported separately.',
                'Three scoring seeds share one trained model; they do not measure retraining variability.',
                'CPU timings are amortized batch inference, not single-request or paper GPU latency.']})
    print(f'Completed: clean accuracy={clean_accuracy:.3f}; evidence in {out}', flush=True)


if __name__ == '__main__':
    main()
