"""
Evaluate the rule fallback against the physics simulated dataset and dump a
confusion matrix and per class metrics. The rule classifier is used as the
lower bound baseline. The fine tuned SmolLM2-135M Instruct should comfortably
beat it on the same prompts, which is the comparison reported in the paper.
"""

import csv
import json
from pathlib import Path

from physics_sim import LABELS
from slm_classifier import RuleFallback

DATA = Path(__file__).parent / "dataset" / "imu_actions_1000.csv"
OUT = Path(__file__).parent / "dataset" / "rule_eval.json"

TRIGGERED = {"hard_kick", "repeated_kick"}


def main():
    rule = RuleFallback()
    cm = {a: {b: 0 for b in LABELS} for a in LABELS}
    correct = 0
    total = 0
    trig_tp = trig_fp = trig_fn = trig_tn = 0
    with open(DATA, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            feats = {k: float(r[k]) for k in r if k not in ("label", "trigger")}
            y_true = r["label"]
            y_pred = rule.classify(feats)
            cm[y_true][y_pred] += 1
            total += 1
            if y_pred == y_true:
                correct += 1
            true_trig = y_true == "repeated_kick"
            pred_trig = y_pred == "repeated_kick"
            if true_trig and pred_trig:
                trig_tp += 1
            elif true_trig and not pred_trig:
                trig_fn += 1
            elif not true_trig and pred_trig:
                trig_fp += 1
            else:
                trig_tn += 1

    acc = correct / total
    prec = trig_tp / max(1, trig_tp + trig_fp)
    rec = trig_tp / max(1, trig_tp + trig_fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    out = {
        "accuracy": acc,
        "trigger_precision": prec,
        "trigger_recall": rec,
        "trigger_f1": f1,
        "confusion_matrix": cm,
        "labels": LABELS,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("accuracy", "trigger_precision", "trigger_recall", "trigger_f1")}, indent=2))


if __name__ == "__main__":
    main()
