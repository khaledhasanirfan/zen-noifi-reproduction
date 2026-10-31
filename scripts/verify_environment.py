#!/usr/bin/env python3
"""
Environment Verification Script
================================
Checks that all required dependencies are installed and working.
Run this after setting up the virtual environment.

Usage:
    python scripts/verify_environment.py
"""

import sys
import platform


def check_import(module_name, display_name=None):
    """Try importing a module and report status."""
    display = display_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✅ {display:30s} {version}")
        return True
    except ImportError as e:
        print(f"  ❌ {display:30s} MISSING — {e}")
        return False


def main():
    print("=" * 60)
    print(" Environment Verification")
    print("=" * 60)
    print()

    # System info
    print("📋 System Information:")
    print(f"  Platform:    {platform.platform()}")
    print(f"  Machine:     {platform.machine()}")
    print(f"  Python:      {sys.version}")
    print(f"  Executable:  {sys.executable}")
    print()

    # Core ML
    print("📦 Core ML Packages:")
    results = []
    results.append(check_import("torch", "PyTorch"))
    results.append(check_import("torchvision", "TorchVision"))
    results.append(check_import("torchaudio", "TorchAudio"))
    print()

    # PyTorch details
    try:
        import torch
        print("🔧 PyTorch Details:")
        print(f"  CUDA available:  {torch.cuda.is_available()}")
        print(f"  MPS available:   {torch.backends.mps.is_available()}")
        print(f"  CPU threads:     {torch.get_num_threads()}")
        print()
    except ImportError:
        pass

    # NoiFi dependencies
    print("📦 NoiFi Dependencies:")
    results.append(check_import("hydra", "Hydra"))
    results.append(check_import("omegaconf", "OmegaConf"))
    results.append(check_import("pyod", "PyOD"))
    results.append(check_import("torchattacks", "TorchAttacks"))
    results.append(check_import("sklearn", "scikit-learn"))
    results.append(check_import("scipy", "SciPy"))
    results.append(check_import("numpy", "NumPy"))
    results.append(check_import("pandas", "Pandas"))
    results.append(check_import("matplotlib", "Matplotlib"))
    results.append(check_import("seaborn", "Seaborn"))
    results.append(check_import("cv2", "OpenCV"))
    results.append(check_import("h5py", "h5py"))
    results.append(check_import("tqdm", "tqdm"))
    results.append(check_import("art", "ART (Adversarial Robustness)"))
    print()

    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    if passed == total:
        print(f" 🎉 All {total} packages verified!")
    else:
        print(f" ⚠️  {passed}/{total} packages OK, {total - passed} missing")
    print("=" * 60)

    # pip check
    print()
    print("Running pip check...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  ✅ No broken dependencies")
    else:
        print(f"  ⚠️  Dependency issues found:")
        print(f"  {result.stdout}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
