"""Verify the compact local experiment environment."""
import importlib
import platform
import subprocess
import sys


def main():
    print(f'Python {platform.python_version()} / {platform.machine()}')
    ok = sys.version_info[:2] == (3, 10)
    for name in ['torch', 'numpy', 'scipy', 'sklearn', 'matplotlib', 'reportlab', 'fuzzywuzzy', 'Levenshtein', 'pypdf']:
        try:
            module = importlib.import_module(name)
            print(f'{name}: {getattr(module, "__version__", "available")}')
        except ImportError as exc:
            print(f'{name}: {exc}'); ok = False
    import torch
    x = torch.randn(2, 4, requires_grad=True)
    x.square().mean().backward()
    ok = ok and x.grad is not None
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], check=False)
    return 0 if ok and check.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
