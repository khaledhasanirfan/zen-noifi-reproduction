# ZEN + NoiFi: basic experiments

Small, runnable implementations and measured comparisons on an M5 Mac.

- **NoiFi:** real RoboFiSense CSI -> compact CNN/DAE -> FGSM/PGD -> residual randomization -> MED detection. Compare with reconstruction MSE.
- **Extension:** N = 1, 3, 5, 10, 20 versus AUROC, score stability and CPU latency.
- **ZEN:** run upstream attribution on supplied recovered artifacts; compare top-two candidate margins. A separate local bytecode demo checks output agreement after patching.
- **Scope:** reduced experiments. Full ZEN Linux dump recovery and official YOLO patch inference are not run. The papers solve different problems, so their metrics are not ranked against each other.

## Run

```bash
git submodule update --init --recursive
bash scripts/setup_mac.sh
bash scripts/run_all.sh
```

Python 3.10 installs through Homebrew; dependencies stay in `.venv-noifi`.
The downloader fetches only CSI recordings (3.26 GB) and verifies Git blob hashes.
For a quick connectivity check: `bash scripts/run_noifi_smoke.sh` after downloading.
To repeat the N-study with the same checkpoints: `bash scripts/run_noifi_n_study.sh`.

## Results

See [measured status](reports/status.md), [PDF](output/pdf/experiment_report.pdf),
[slide notes](output/slide_notes.md), [figures](output/figures), and [raw CSVs](results).
NoiFi uses separate train, validation, calibration and test recordings. This compact
64x64 model is not the full paper architecture. ZEN artifact coverage is reported explicitly.

## Sources / downloads

- [ZEN code + official Linux instructions](https://github.com/CyFI-Lab-Public/ZEN)
- [ZEN supplied memory dump](https://drive.google.com/file/d/1nsTN716aGoT7AU32HmuFQjCYYXmakjHR/view)
- [NoiFi code](https://github.com/shahriar0651/NoiFi)
- [RoboFiSense dataset](https://github.com/SiamiLab/RoboFiSense)

Upstream commits: `reports/upstream-commits.txt`. Source recordings, checkpoints and
logs are excluded from Git. Our scripts: MIT; upstream code retains its own license.
