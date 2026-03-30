"""
extract_threshold.py
──────────────────────
One-time offline script to extract the optimal detection threshold
from your existing models/metrics.json and save it to models/threshold.json.

No GPU, no retraining needed.

Usage:
    cd backend
    python extract_threshold.py

Output:
    models/threshold.json  →  {"threshold": <value>, "epoch": <best_epoch>, "recall": <recall>}

The predictor loads this file automatically on startup.
If this file does not exist, predictor uses settings.CONFIDENCE_THRESHOLD (0.4) as fallback.
"""

import json
from pathlib import Path


def main():
    metrics_path = Path("models/metrics.json")
    out_path = Path("models/threshold.json")

    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found. Run training first.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    if not metrics:
        print("ERROR: metrics.json is empty.")
        return

    # Strategy 1: pick epoch with best recall (FN minimization priority)
    best_by_recall = max(metrics, key=lambda e: e.get("recall", 0.0))

    # Strategy 2: pick epoch with best F1
    best_by_f1 = max(metrics, key=lambda e: e.get("f1", 0.0))

    print("\n── Best by Recall (FN minimization) ──")
    print(f"  Epoch     : {best_by_recall.get('epoch')}")
    print(f"  Recall    : {best_by_recall.get('recall', 0):.4f}")
    print(f"  Precision : {best_by_recall.get('precision', 0):.4f}")
    print(f"  F1        : {best_by_recall.get('f1', 0):.4f}")
    print(f"  FN        : {best_by_recall.get('fn', '?')}")
    print(f"  Threshold : {best_by_recall.get('threshold', 'N/A')}")

    print("\n── Best by F1 ──")
    print(f"  Epoch     : {best_by_f1.get('epoch')}")
    print(f"  Recall    : {best_by_f1.get('recall', 0):.4f}")
    print(f"  F1        : {best_by_f1.get('f1', 0):.4f}")
    print(f"  Threshold : {best_by_f1.get('threshold', 'N/A')}")

    # Use recall-optimized threshold (clinical safety priority)
    chosen = best_by_recall
    threshold = chosen.get("threshold", None)

    if threshold is None:
        print("\nWARNING: No threshold in metrics.json. Using default 0.315 (PR-based safe minimum).")
        threshold = 0.315

    # Enforce minimum floor for FN safety
    threshold = max(float(threshold), 0.30)

    result = {
        "threshold": round(threshold, 6),
        "epoch": chosen.get("epoch"),
        "recall": round(chosen.get("recall", 0.0), 6),
        "f1": round(chosen.get("f1", 0.0), 6),
        "fn": chosen.get("fn"),
        "source": "metrics.json (recall-optimized)"
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Saved threshold to {out_path}")
    print(f"   threshold = {result['threshold']}")
    print("   Predictor will load this automatically on next startup.")


if __name__ == "__main__":
    main()
