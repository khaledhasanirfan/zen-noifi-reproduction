"""
ZEN Output Consistency Checker
==============================
Compares the numerical output of the original model against
the ZEN-recovered (patched) model to verify reconstruction
fidelity.

Usage:
    python compare_outputs.py reference.npy recovered.npy [--atol 1e-5] [--rtol 1e-4]

Exit codes:
    0 — outputs match within tolerance
    1 — outputs differ beyond tolerance
"""

import argparse
import json
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs ZEN-recovered model outputs"
    )
    parser.add_argument("reference", help="Path to reference output (.npy)")
    parser.add_argument("recovered", help="Path to recovered output (.npy)")
    parser.add_argument(
        "--atol", type=float, default=1e-5,
        help="Absolute tolerance for np.allclose (default: 1e-5)"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-4,
        help="Relative tolerance for np.allclose (default: 1e-4)"
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Optional path to save results as JSON"
    )
    args = parser.parse_args()

    # Load arrays
    reference = np.load(args.reference, allow_pickle=False)
    recovered = np.load(args.recovered, allow_pickle=False)

    # Shape check
    if reference.shape != recovered.shape:
        print(
            f"❌ Shape mismatch: reference={reference.shape} "
            f"vs recovered={recovered.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Empty check
    if reference.size == 0:
        print("❌ Empty outputs cannot establish equivalence.", file=sys.stderr)
        sys.exit(1)

    # Finite check
    ref_finite = np.isfinite(reference).all()
    rec_finite = np.isfinite(recovered).all()
    if not (ref_finite and rec_finite):
        print(
            f"⚠️  Non-finite values detected — "
            f"reference finite: {ref_finite}, recovered finite: {rec_finite}",
            file=sys.stderr,
        )

    # Compute metrics
    difference = np.abs(
        reference.astype(np.float64) - recovered.astype(np.float64)
    )
    passed = bool(
        np.allclose(reference, recovered, atol=args.atol, rtol=args.rtol)
    )

    results = {
        "shape": list(reference.shape),
        "num_elements": int(reference.size),
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "median_absolute_error": float(np.median(difference)),
        "std_absolute_error": float(difference.std()),
        "percentile_95": float(np.percentile(difference, 95)),
        "percentile_99": float(np.percentile(difference, 99)),
        "atol": args.atol,
        "rtol": args.rtol,
        "allclose": passed,
        "reference_finite": bool(ref_finite),
        "recovered_finite": bool(rec_finite),
    }

    output = json.dumps(results, indent=2)
    print(output)

    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(output + "\n")
        print(f"\nResults saved to {args.output_json}")

    if passed:
        print("\n✅ PASS — outputs match within tolerance")
    else:
        print("\n❌ FAIL — outputs differ beyond tolerance")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
