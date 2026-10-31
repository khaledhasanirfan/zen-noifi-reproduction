# Experiment status

- **Python:** 3.10.22 installed with an isolated CPU environment; imports and pip check pass.
- **Dataset:** all 1,103 real RoboFiSense recordings downloaded (3.26 GB), pinned to commit 7cc84af1229b62d48f82392d1fed1422f3c9223b. Git blob hashes verified; all recording SHA-256 hashes are unique.
- **NoiFi basic:** completed. 661 train / 110 validation / 110 calibration / 222 test recordings. CNN: 40 epochs (best validation checkpoint: epoch 36). DAE: 30 epochs (best: epoch 11). Clean test accuracy: 91.44%; uniform-noise control accuracy: 90.54%.
- **Attacks:** FGSM and 20-step PGD, epsilon 0.10. ASR on 203 clean-correct test recordings: 96.06% and 100.00%.
- **NoiFi extension:** completed N = 1, 3, 5, 10, 20, with permutation seeds 7, 17, 27 and the same trained models/attacks.

| Attack | MSE AUROC | MED AUROC (N=5) | MED recall | MED test FPR |
|---|---:|---:|---:|---:|
| FGSM | 0.9546 | 0.9257 | 59.46% | 4.95% |
| PGD | 0.9578 | 0.9677 | 77.18% | 4.95% |

- **Interpretation:** MSE is better on FGSM; MED has higher AUROC on PGD. Increasing N changes AUROC little but reduces score variability and increases computation. No universal performance improvement is claimed.
- **ZEN attribution:** completed using upstream code and supplied recovered artifacts. MR-only: 11/13 correct; PR-only and combined MR+PR: 13/13. Missing referenced bytecode files: 1,861 across 20 inspected models; coverage is recorded.
- **ZEN extensions:** candidate margin comparison completed. Local bytecode patch demo: maximum error falls from 0.613492 to 0 over 64 inputs; numerical consistency checker passes.
- **Full ZEN memory recovery / official YOLO patch inference:** not run. Requires the supplied LiME dump and compatible Linux x86-64 / Python 3.8.18 setup. Attribution artifacts and the local patch demo do not substitute for fresh recovery.
- **Verification:** four scientific invariant/regression tests pass. Python syntax, shell syntax and dependency checks pass.
- **Deliverables:** output/pdf/experiment_report.pdf, output/figures/*.png and *.svg, output/slide_notes.md, raw CSV/JSON files under results/. Generated dates/times are omitted.
