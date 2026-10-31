"""Small controlled patch/re-execution demo, inspired by ZEN, using local bytecode.

Demonstrates why state_dict alone is insufficient for custom computation. This is
not Volatility memory recovery or the official YOLO patch experiment.
"""
import marshal
import types
import numpy as np
import torch
from torch import nn
from common import ROOT, save_json


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return torch.tanh(self.linear(x)) * 1.25 + 0.1


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return self.linear(x)


def main():
    torch.manual_seed(42)
    out = ROOT/'results/zen-patch-demo'; out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').unlink(missing_ok=True)
    original, base = CustomModel().eval(), BaseModel().eval()
    base.load_state_dict(original.state_dict())
    x = torch.randn(64, 16)
    with torch.no_grad():
        reference = original(x).numpy()
        before = base(x).numpy()
    # Only our own known function bytecode, serialized/reloaded within Python 3.10.
    code = marshal.dumps(CustomModel.forward.__code__)
    recovered_forward = types.FunctionType(marshal.loads(code), {'torch': torch}, 'forward')
    base.forward = types.MethodType(recovered_forward, base)
    with torch.no_grad():
        after = base(x).numpy()
    np.save(out/'reference.npy', reference); np.save(out/'before.npy', before); np.save(out/'recovered.npy', after)
    result = {'status': 'completed', 'scope': 'Controlled local bytecode patch demo',
              'input_count': len(x), 'before_max_absolute_error': float(np.abs(reference-before).max()),
              'after_max_absolute_error': float(np.abs(reference-after).max()),
              'before_allclose': bool(np.allclose(reference, before)),
              'after_allclose': bool(np.allclose(reference, after)),
              'limitations': ['Locally captured function bytecode; no memory dump or YOLO model used.',
                              'Agreement covers only these 64 inputs.']}
    if result['before_allclose'] or not result['after_allclose']:
        raise AssertionError('Patch demonstration failed')
    save_json(out/'summary.json', result)
    print(result)


if __name__ == '__main__':
    main()
