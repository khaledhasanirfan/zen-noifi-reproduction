#!/usr/bin/env python3
"""
Generate PDF Experiment Reports
================================
Produces professional PDF reports for ZEN and NoiFi experiments.
Uses only Python standard library + minimal dependencies.

Usage:
    python scripts/generate_report.py [--output reports/experiment_report.pdf]
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# We'll generate HTML first, then convert — or use a pure-Python approach
# This script generates a standalone HTML report that can be printed to PDF


def get_project_root():
    return Path(__file__).resolve().parent.parent


def read_file_safe(path):
    """Read a file, returning empty string if not found."""
    try:
        return Path(path).read_text()
    except (FileNotFoundError, PermissionError):
        return ""


def collect_noifi_results(root):
    """Collect NoiFi experiment results."""
    results = {}

    # Check smoke test
    smoke_dir = root / "results" / "noifi-smoke"
    results["smoke_test"] = {
        "completed": smoke_dir.exists() and any(smoke_dir.rglob("*.log")),
        "log_files": [str(f) for f in (root / "logs" / "noifi-smoke").glob("*.log")]
        if (root / "logs" / "noifi-smoke").exists()
        else [],
    }

    # Check baseline
    baseline_dir = root / "results" / "noifi-baseline"
    results["baseline"] = {
        "completed": baseline_dir.exists(),
        "has_models": (baseline_dir / "models").exists()
        if baseline_dir.exists()
        else False,
        "has_outputs": (baseline_dir / "outputs").exists()
        if baseline_dir.exists()
        else False,
    }

    # Check N-study
    n_study_dir = root / "results" / "noifi-n-study"
    results["n_study"] = {
        "completed": n_study_dir.exists(),
        "n_values_tested": [],
        "latency_data": "",
    }
    if n_study_dir.exists():
        for d in sorted(n_study_dir.iterdir()):
            if d.is_dir() and d.name.startswith("N_"):
                results["n_study"]["n_values_tested"].append(d.name)
        latency_file = n_study_dir / "latency_results.csv"
        if latency_file.exists():
            results["n_study"]["latency_data"] = latency_file.read_text()

    return results


def collect_zen_results(root):
    """Collect ZEN experiment results."""
    results = {}

    # Check recovery
    zen_dir = root / "results" / "zen"
    results["recovery"] = {
        "completed": (root / "logs" / "zen" / "recovery.log").exists(),
        "artifacts_found": [],
        "dump_hash": read_file_safe(root / "reports" / "zen-memory-dump.sha256"),
    }

    artifacts_file = zen_dir / "recovered-artifacts.txt"
    if artifacts_file.exists():
        results["recovery"]["artifacts_found"] = [
            line.strip()
            for line in artifacts_file.read_text().splitlines()
            if line.strip()
        ]

    # Check inference
    results["inference"] = {
        "without_patch_log": read_file_safe(
            root / "logs" / "zen" / "inference-without-patch.log"
        ),
        "with_patch_log": read_file_safe(
            root / "logs" / "zen" / "inference-with-patch.log"
        ),
    }

    return results


def generate_html_report(root):
    """Generate a comprehensive HTML report."""
    noifi = collect_noifi_results(root)
    zen = collect_zen_results(root)
    upstream = read_file_safe(root / "reports" / "upstream-commits.txt")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Status helper
    def status_icon(completed):
        return "✅ Completed" if completed else "⬜ Pending"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ZEN & NoiFi — Experiment Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    line-height: 1.6;
    color: #1a1a2e;
    background: #f8f9fa;
    padding: 40px;
    max-width: 900px;
    margin: 0 auto;
  }}

  @media print {{
    body {{ padding: 20px; background: white; }}
    .page-break {{ page-break-before: always; }}
    .no-print {{ display: none; }}
  }}

  h1 {{
    font-size: 28px;
    font-weight: 700;
    color: #16213e;
    border-bottom: 3px solid #0f3460;
    padding-bottom: 12px;
    margin-bottom: 24px;
  }}

  h2 {{
    font-size: 22px;
    font-weight: 600;
    color: #0f3460;
    margin-top: 32px;
    margin-bottom: 16px;
    padding-left: 12px;
    border-left: 4px solid #e94560;
  }}

  h3 {{
    font-size: 16px;
    font-weight: 600;
    color: #533483;
    margin-top: 20px;
    margin-bottom: 8px;
  }}

  .meta {{
    background: linear-gradient(135deg, #16213e, #0f3460);
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 24px;
  }}

  .meta p {{ color: #ccc; font-size: 14px; }}
  .meta strong {{ color: white; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 14px;
  }}

  th {{
    background: #0f3460;
    color: white;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
  }}

  td {{
    padding: 8px 14px;
    border-bottom: 1px solid #e0e0e0;
  }}

  tr:nth-child(even) {{ background: #f5f5f5; }}
  tr:hover {{ background: #e8f4f8; }}

  .status-card {{
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }}

  .completed {{ border-left: 4px solid #2ecc71; }}
  .pending {{ border-left: 4px solid #f39c12; }}

  .code {{
    background: #2d2d2d;
    color: #f8f8f2;
    padding: 12px 16px;
    border-radius: 6px;
    font-family: 'Fira Code', 'Courier New', monospace;
    font-size: 12px;
    overflow-x: auto;
    white-space: pre-wrap;
    margin: 8px 0;
  }}

  .section {{
    background: white;
    border-radius: 8px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}

  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
  }}

  .badge-green {{ background: #d4edda; color: #155724; }}
  .badge-yellow {{ background: #fff3cd; color: #856404; }}
  .badge-blue {{ background: #cce5ff; color: #004085; }}

  footer {{
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #ddd;
    font-size: 12px;
    color: #888;
    text-align: center;
  }}
</style>
</head>
<body>

<div class="meta">
  <h1 style="color:white; border-bottom-color: #e94560;">
    ZEN & NoiFi — Reproduction & Extension Report
  </h1>
  <p><strong>Author:</strong> Khaled Hasan Irfan</p>
  <p><strong>Generated:</strong> {now}</p>
  <p><strong>Repository:</strong> https://github.com/khaledhasanirfan/zen-noifi-reproduction</p>
  <p><strong>Hardware:</strong> MacBook Air M5, 16GB RAM, macOS 26.6.2</p>
</div>

<div class="section">
  <h2>1. Project Overview</h2>
  <p>This report documents the reproduction and extension of two security/ML research papers:</p>

  <table>
    <tr><th>Paper</th><th>Domain</th><th>Core Contribution</th></tr>
    <tr>
      <td><strong>ZEN</strong> (CyFI Lab, Georgia Tech)</td>
      <td>Memory Forensics + DNN</td>
      <td>Recover custom neural-network layers from Linux memory dumps, attribute base models, and patch for inference</td>
    </tr>
    <tr>
      <td><strong>NoiFi</strong> (Shahriar et al.)</td>
      <td>WiFi Sensing Security</td>
      <td>Detect adversarial attacks on WiFi CSI classifiers using randomized-smoothing noise-feature analysis</td>
    </tr>
  </table>
</div>

<div class="section">
  <h2>2. Upstream Attribution</h2>
  <div class="code">{upstream if upstream else "Pending — run: git submodule status > reports/upstream-commits.txt"}</div>
</div>

<div class="section">
  <h2>3. Experiment Status Summary</h2>
  <table>
    <tr><th>#</th><th>Experiment</th><th>Status</th><th>Evidence</th></tr>
    <tr>
      <td>1</td><td>NoiFi Environment Setup</td>
      <td>{status_icon(noifi['smoke_test']['completed'])}</td>
      <td>{'Logs available' if noifi['smoke_test']['log_files'] else 'N/A'}</td>
    </tr>
    <tr>
      <td>2</td><td>NoiFi Smoke Pipeline (2 epochs)</td>
      <td>{status_icon(noifi['smoke_test']['completed'])}</td>
      <td>{len(noifi['smoke_test']['log_files'])} log files</td>
    </tr>
    <tr>
      <td>3</td><td>NoiFi Baseline Reproduction</td>
      <td>{status_icon(noifi['baseline']['completed'])}</td>
      <td>{'Models + outputs' if noifi['baseline'].get('has_models') else 'N/A'}</td>
    </tr>
    <tr>
      <td>4</td><td>NoiFi N-vs-Stability Extension</td>
      <td>{status_icon(noifi['n_study']['completed'])}</td>
      <td>{', '.join(noifi['n_study']['n_values_tested']) or 'N/A'}</td>
    </tr>
    <tr>
      <td>5</td><td>ZEN Environment Setup (Linux)</td>
      <td>{status_icon(bool(zen['recovery']['dump_hash']))}</td>
      <td>{'Hash recorded' if zen['recovery']['dump_hash'] else 'N/A'}</td>
    </tr>
    <tr>
      <td>6</td><td>ZEN Supplied-Dump Recovery</td>
      <td>{status_icon(zen['recovery']['completed'])}</td>
      <td>{len(zen['recovery']['artifacts_found'])} artifacts</td>
    </tr>
    <tr>
      <td>7</td><td>ZEN Patched Inference</td>
      <td>{status_icon(bool(zen['inference']['with_patch_log']))}</td>
      <td>{'Logs available' if zen['inference']['with_patch_log'] else 'N/A'}</td>
    </tr>
    <tr>
      <td>8</td><td>ZEN Output Consistency</td>
      <td>⬜ Pending</td>
      <td>N/A</td>
    </tr>
  </table>
</div>

<div class="page-break"></div>

<div class="section">
  <h2>4. NoiFi — Detailed Results</h2>

  <h3>4.1 Environment</h3>
  <table>
    <tr><th>Component</th><th>Value</th></tr>
    <tr><td>Device</td><td>CPU (Apple M5 — no CUDA)</td></tr>
    <tr><td>Python</td><td>3.9.6</td></tr>
    <tr><td>PyTorch</td><td>2.0.1</td></tr>
    <tr><td>Dataset</td><td>RoboFiSense</td></tr>
  </table>

  <h3>4.2 Pipeline Steps</h3>
  <table>
    <tr><th>Step</th><th>Script</th><th>Purpose</th></tr>
    <tr><td>0a</td><td>step_0_train_classifiers.py</td><td>Train WiFi CSI classifiers</td></tr>
    <tr><td>0b</td><td>step_0_train_autoencoders.py</td><td>Train denoising autoencoders (DAE)</td></tr>
    <tr><td>1</td><td>step_1_generate_adversarial_examples_simple.py</td><td>Generate adversarial attacks</td></tr>
    <tr><td>2</td><td>step_2_reconstruct_and_extract_noise.py</td><td>DAE reconstruction + noise extraction</td></tr>
    <tr><td>3</td><td>step_3_analyze_noise_features.py</td><td>Analyze residual noise features</td></tr>
    <tr><td>4</td><td>step_4_detect_adversarial_examples_simple.py</td><td>Detect adversarial examples</td></tr>
  </table>

  <h3>4.3 Extension: Randomization Count (N) vs Detection Stability</h3>
  <p>
    This extension examines how the number of randomized references (N) affects
    detection accuracy and latency — a key design parameter in NoiFi's
    randomized smoothing approach.
  </p>
  <table>
    <tr><th>N</th><th>AUROC</th><th>Recall@FPR</th><th>Median Latency</th><th>Score Variability</th></tr>
    <tr><td>1</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>3</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>5</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>10</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>20</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  </table>
  <p><em>Table will be populated after baseline experiments complete.</em></p>
</div>

<div class="page-break"></div>

<div class="section">
  <h2>5. ZEN — Detailed Results</h2>

  <h3>5.1 Environment Requirements</h3>
  <table>
    <tr><th>Component</th><th>Value</th></tr>
    <tr><td>OS</td><td>Linux x86-64 (required)</td></tr>
    <tr><td>Python</td><td>3.8.18 (per official demo)</td></tr>
    <tr><td>Framework</td><td>Volatility3 + custom CMR plugin</td></tr>
    <tr><td>Model</td><td>YOLOv5 + CBAM-Involution (VisDrone)</td></tr>
  </table>

  <h3>5.2 Recovery Pipeline</h3>
  <table>
    <tr><th>Phase</th><th>Description</th><th>Status</th></tr>
    <tr><td>Memory Dump</td><td>Load supplied .lime file</td><td>{status_icon(bool(zen['recovery']['dump_hash']))}</td></tr>
    <tr><td>CMR Recovery</td><td>Extract custom model representations</td><td>{status_icon(zen['recovery']['completed'])}</td></tr>
    <tr><td>Attribution</td><td>Identify base model architecture</td><td>{status_icon(zen['recovery']['completed'])}</td></tr>
    <tr><td>Patching</td><td>Generate inference-ready patch</td><td>{status_icon(bool(zen['recovery']['artifacts_found']))}</td></tr>
    <tr><td>Inference</td><td>Run model with recovered patch</td><td>{status_icon(bool(zen['inference']['with_patch_log']))}</td></tr>
  </table>

  <h3>5.3 Extension: Output Consistency Checker</h3>
  <p>
    Numerically compares original model outputs against ZEN-recovered outputs
    using configurable absolute and relative tolerance thresholds.
  </p>
  <div class="code">python scripts/compare_outputs.py reference.npy recovered.npy --atol 1e-5 --rtol 1e-4</div>
</div>

<div class="section">
  <h2>6. Reproduction Notes & Limitations</h2>
  <ul style="padding-left: 24px;">
    <li>NoiFi runs on CPU only (no CUDA on macOS) — training is slower but numerically equivalent</li>
    <li>ZEN requires Linux x86-64 with matching kernel symbols for full forensic pipeline</li>
    <li>All reported results are from our own runs, not copied from the papers</li>
    <li>Threshold tuning data and final evaluation data are kept separate</li>
    <li>Small epoch counts in smoke tests verify pipeline connectivity, not research claims</li>
  </ul>
</div>

<div class="section">
  <h2>7. Repository Structure</h2>
  <div class="code">zen-noifi-reproduction/
├── README.md
├── .gitignore
├── .gitmodules
├── third_party/
│   ├── ZEN/          (git submodule → CyFI-Lab-Public/ZEN)
│   └── NoiFi/        (git submodule → shahriar0651/NoiFi)
├── scripts/
│   ├── run_noifi_smoke.sh
│   ├── run_noifi_baseline.sh
│   ├── run_noifi_n_study.sh
│   ├── run_zen_demo.sh
│   ├── run_zen_inference.sh
│   ├── compare_outputs.py
│   └── generate_report.py
├── environments/
│   └── noifi-mac-initial.txt
├── reports/
│   ├── upstream-commits.txt
│   └── experiment_report.html
├── results/     (experiment outputs)
├── logs/        (execution logs)
└── data/        (datasets — gitignored)</div>
</div>

<footer>
  <p>Generated by generate_report.py — {now}</p>
  <p>GitHub: <a href="https://github.com/khaledhasanirfan/zen-noifi-reproduction">
    khaledhasanirfan/zen-noifi-reproduction</a></p>
</footer>

</body>
</html>"""

    return html


def main():
    root = get_project_root()

    # Generate HTML
    html = generate_html_report(root)

    # Save HTML report
    html_path = root / "reports" / "experiment_report.html"
    html_path.write_text(html)
    print(f"✅ HTML report saved: {html_path}")

    # Try to convert to PDF using available tools
    pdf_path = root / "reports" / "experiment_report.pdf"

    # Method 1: Try weasyprint
    try:
        from weasyprint import HTML as WeasyprintHTML
        WeasyprintHTML(string=html).write_pdf(str(pdf_path))
        print(f"✅ PDF report saved: {pdf_path}")
        return
    except ImportError:
        pass

    # Method 2: Try pdfkit (wkhtmltopdf)
    try:
        import pdfkit
        pdfkit.from_string(html, str(pdf_path))
        print(f"✅ PDF report saved: {pdf_path}")
        return
    except (ImportError, OSError):
        pass

    # Method 3: Use macOS built-in cupsfilter or textutil
    import subprocess
    try:
        subprocess.run(
            [
                "/usr/sbin/cupsfilter",
                "-o", "media=A4",
                str(html_path),
            ],
            stdout=open(str(pdf_path), "wb"),
            stderr=subprocess.DEVNULL,
            check=True,
        )
        print(f"✅ PDF report saved: {pdf_path}")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 4: Use Chrome/Chromium headless
    for browser in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "google-chrome",
        "chromium",
    ]:
        try:
            subprocess.run(
                [
                    browser,
                    "--headless",
                    "--disable-gpu",
                    f"--print-to-pdf={pdf_path}",
                    str(html_path),
                ],
                check=True,
                capture_output=True,
            )
            print(f"✅ PDF report saved: {pdf_path}")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    print(f"⚠️  Could not auto-convert to PDF.")
    print(f"   HTML report is ready at: {html_path}")
    print(f"   Open it in a browser and use Print → Save as PDF")
    print(f"   Or install: pip install weasyprint")


if __name__ == "__main__":
    main()
