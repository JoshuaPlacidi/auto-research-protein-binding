"""Fixed validation evaluator. Append all results to ONE shared research log.

Training may use any loss; validation_loss is always probability MSE (Brier
score), so changing a training objective cannot change the evaluation scale.
No training, model selection, file restoration or agent API calls happen here.
"""
import argparse
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED = ("TCR_full", "Peptide", "HLA_sequence", "binding_flag", "tcra_len", "tcrb_len")
CATEGORY_COLUMNS = ("category", "Category", "split_category", "novelty_category",
                    "generalisation_category", "generalization_category")


def append_log(path, record):
    """One JSON object per line; never replace earlier history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


def read_history(path):
    if not Path(path).exists():
        return []
    # Preserve human-written rationale too: only JSON model-evaluation events
    # participate in numerical comparisons. The agent must read the whole file.
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        except json.JSONDecodeError:
            continue
    return records


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_rows(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    alpha = pd.to_numeric(frame.tcra_len, errors="coerce")
    beta = pd.to_numeric(frame.tcrb_len, errors="coerce")
    if (alpha.isna() | beta.isna() | (alpha < 0) | (beta < 0)
            | (alpha % 1 != 0) | (beta % 1 != 0)).any():
        raise ValueError("Chain lengths must be non-negative integers")
    mask = (alpha > 0) & (beta > 0)
    selected = frame.loc[mask].copy()
    for column in ("TCR_full", "Peptide", "HLA_sequence"):
        if not selected[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"Missing/empty {column} in full-chain cohort")
    if not (alpha[mask] + beta[mask] == selected.TCR_full.str.len()).all():
        raise ValueError("Chain lengths do not match TCR_full; do not infer boundaries")
    if not selected.binding_flag.isin([0, 1]).all():
        raise ValueError("binding_flag must be 0 or 1")
    return selected.reset_index(drop=True), np.flatnonzero(mask.to_numpy())


def metric_summary(labels, scores):
    y, p = np.asarray(labels, dtype=float), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("Expected one score and label per evaluated row")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be finite binary values")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Model must return finite binding probabilities in [0, 1]")
    n = len(y)
    result = {"n": n, "n_positive": int(y.sum()), "n_negative": int(n - y.sum()),
              "positive_fraction": float(y.mean()) if n else None}
    mse = float(np.mean((p - y) ** 2)) if n else None
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    result.update(validation_loss=mse, mse=mse,
                  binary_log_loss=float(-np.mean(y * np.log(clipped) + (1-y) * np.log(1-clipped))) if n else None,
                  accuracy=float(np.mean((p >= .5) == y)) if n else None,
                  auroc=None, auprc=None, undefined_reason=None)
    if not n:
        result["undefined_reason"] = "empty group"
    elif len(np.unique(y)) < 2:
        result["undefined_reason"] = "single-class group; AUROC and AUPRC not reported"
    else:
        result["auroc"] = float(roc_auc_score(y, p))
        # Average precision, not trapezoidal integration of the PR curve.
        result["auprc"] = float(average_precision_score(y, p))
    return result


def grouped_metrics(frame, scores, column):
    result = {}
    for value, indices in frame.groupby(column, dropna=False, sort=True).indices.items():
        name = "<missing>" if pd.isna(value) else str(value)
        result[name] = metric_summary(frame.iloc[indices].binding_flag, scores[indices])
    return result


def novelty_masks(frame, train):
    """Exact sequence identities against the eligible training cohort; overlaps allowed."""
    def identities(data):
        tcr = [(s[:int(a)], s[int(a):]) for s, a in zip(data.TCR_full, data.tcra_len)]
        pmhc = list(zip(data.Peptide, data.HLA_sequence))
        return tcr, pmhc
    train_tcr, train_pmhc = identities(train)
    tcr, pmhc = identities(frame)
    def unseen(values, reference):
        known = set(reference)
        return np.array([value not in known for value in values], dtype=bool)
    t = unseen(tcr, train_tcr)
    p = unseen(frame.Peptide, train.Peptide)
    h = unseen(frame.HLA_sequence, train.HLA_sequence)
    m = unseen(pmhc, train_pmhc)
    return {"seen_tcr": ~t, "unseen_tcr": t, "seen_peptide": ~p,
            "unseen_peptide": p, "seen_hla": ~h, "unseen_hla": h,
            "seen_pmhc": ~m, "unseen_pmhc": m,
            "fully_seen_components": ~t & ~p & ~h,
            "unseen_tcr_and_pmhc": t & m,
            "completely_unseen_tcr_peptide_hla": t & p & h}


def build_report(frame, scores, train=None, category_column=None):
    scores = np.asarray(scores, dtype=float)
    overall = metric_summary(frame.binding_flag, scores)
    per_peptide = grouped_metrics(frame, scores, "Peptide")
    valid = [v for v in per_peptide.values() if v["auroc"] is not None]
    column = category_column
    if column and column not in frame:
        raise ValueError(f"Category column {column!r} is missing")
    if column is None:
        candidates = [c for c in CATEGORY_COLUMNS if c in frame]
        if len(candidates) > 1:
            raise ValueError("Multiple category columns; specify --category-column")
        column = candidates[0] if candidates else None
    derived = None
    if train is not None:
        derived = {name: metric_summary(frame.binding_flag[mask], scores[mask])
                   for name, mask in novelty_masks(frame, train).items()}
    return {"overall": overall, "per_peptide": per_peptide,
            "peptide_macro_auroc": float(np.mean([v["auroc"] for v in valid])) if valid else None,
            "peptide_macro_auprc": float(np.mean([v["auprc"] for v in valid])) if valid else None,
            "peptide_coverage": {"total": len(per_peptide), "both_labels": len(valid),
                                 "excluded_single_class": len(per_peptide) - len(valid)},
            "category_column": column,
            "provided_categories": grouped_metrics(frame, scores, column) if column else None,
            "provided_categories_note": None if column else "No recognised category column; use --category-column for a different name",
            "derived_categories": derived,
            "derived_categories_note": "Overlapping exact-identity groups against eligible training rows; TCR identity is the ordered full alpha/beta pair, HLA identity is HLA_sequence. Strict completely-unseen means TCR, peptide AND HLA unseen. Supplied categories are reported separately without reinterpretation." if train is not None else "Unavailable: TRAIN_PATH was not supplied; provided categories are still reported"}


def evaluate_model(model, test_loader, device, return_predictions=False):
    """Retain the old three-value interface; optionally return ordered scores."""
    import torch
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for peptide, hla, tcr, binding_flag in test_loader:
            pred = model(peptide.to(device), hla.to(device), tcr.to(device))
            pred = pred.reshape(-1).detach().cpu().numpy()
            target = binding_flag.reshape(-1).cpu().numpy()
            if len(pred) != len(target):
                raise ValueError("Model must return exactly one probability per row")
            preds.extend(pred.tolist())
            labels.extend(target.tolist())
    scores = np.asarray(preds)
    metrics = metric_summary(labels, scores)
    if return_predictions:
        return np.asarray(labels), scores
    return metrics["validation_loss"], metrics["auroc"], metrics["auprc"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--log-path", default="experiments/evaluation.log")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--category-column")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--change", default="")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_final_test:
        parser.error("Test evaluation requires --allow-final-test after freezing the final model")
    started = time.perf_counter()
    record = {"event": "model_evaluation", "timestamp": datetime.now(timezone.utc).isoformat(),
              "experiment_id": args.experiment_id or uuid.uuid4().hex,
              "experiment_dir": args.experiment_dir, "split": args.split,
              "hypothesis": args.hypothesis, "change": args.change,
              "loss_definition": "mean squared error on binding probabilities; sample-weighted Brier score",
              "status": "failed"}
    try:
        import torch
        import ml
        # Import ml before reading paths: the existing ml.py loads .env.
        path = os.getenv("VALIDATION_PATH" if args.split == "validation" else "TEST_PATH")
        if not path:
            raise ValueError(f"Set {'VALIDATION_PATH' if args.split == 'validation' else 'TEST_PATH'}")
        checkpoint = Path(args.experiment_dir) / "model.pth"
        record.update(ml_sha256=sha256(ml.__file__), evaluator_sha256=sha256(__file__),
                      checkpoint_sha256=sha256(checkpoint), evaluation_data_sha256=sha256(path))
        raw = pd.read_csv(path, low_memory=False)
        frame, indices = eligible_rows(raw)
        if not len(frame):
            raise ValueError("No eligible full-chain evaluation rows")
        dataset = ml.Dataset(path)
        # The fixed evaluator owns labels, membership and row order. Do not
        # silently accept a Dataset that filters, reorders or relabels the CSV.
        if not hasattr(dataset, "data") or not dataset.data[list(REQUIRED)].reset_index(drop=True).equals(raw[list(REQUIRED)].reset_index(drop=True)):
            raise ValueError("Dataset.data must preserve CSV row order, labels and inputs; filtering is owned by eval.py")
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices.tolist()), batch_size=128, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ml.Model().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        labels, scores = evaluate_model(model, loader, device, return_predictions=True)
        if len(scores) != len(frame) or not np.array_equal(labels, frame.binding_flag.to_numpy()):
            raise ValueError("Returned labels/order do not match the fixed eligible cohort")
        train = None
        train_path = os.getenv("TRAIN_PATH")
        if train_path:
            record["training_data_sha256"] = sha256(train_path)
            train_raw = pd.read_csv(train_path, low_memory=False)
            train, _ = eligible_rows(train_raw)
            if not len(train):
                raise ValueError("No eligible full-chain training rows")
            record["training_cohort"] = {"rows": len(train_raw), "eligible": len(train)}
        report = build_report(frame, scores, train, args.category_column)
        # Only compare the same data, evaluator and split; changing evaluation
        # membership/protocol must not look like a model improvement.
        history = read_history(args.log_path)
        previous = [r for r in history if r.get("event") == "model_evaluation" and r.get("status") == "valid"
                    and r.get("split") == args.split
                    and r.get("evaluation_data_sha256") == record["evaluation_data_sha256"]
                    and r.get("evaluator_sha256") == record["evaluator_sha256"]
                    and r.get("training_data_sha256") == record.get("training_data_sha256")]
        best = min((r["metrics"]["overall"]["validation_loss"] for r in previous), default=None)
        auc_best = max((r["metrics"]["overall"]["auroc"] for r in previous if r["metrics"]["overall"]["auroc"] is not None), default=None)
        loss = report["overall"]["validation_loss"]
        auc = report["overall"]["auroc"]
        record.update(status="valid", metrics=report,
                      cohort={"rows": len(raw), "eligible": len(frame), "excluded_missing_chain": len(raw)-len(frame)},
                      agent_evaluation={"execution_valid": True, "hypothesis_recorded": bool(args.hypothesis.strip()),
                          "change_recorded": bool(args.change.strip()), "comparable_previous_runs": len(previous),
                          "previous_best_validation_loss": best,
                          "loss_improvement_over_previous_best": best-loss if best is not None else None,
                          "previous_best_auroc": auc_best,
                          "auroc_change_from_previous_best": auc-auc_best if auc is not None and auc_best is not None else None,
                          "new_lowest_validation_loss": best is None or loss < best,
                          "decision": "pending_agent_interpretation",
                          "reason": "Agent must append keep/discard/investigate with scientific justification; no automatic single-metric acceptance."})
    except Exception as exc:
        record.update(error=f"{type(exc).__name__}: {exc}",
                      agent_evaluation={"execution_valid": False, "decision": "failed"})
    record["evaluation_runtime_seconds"] = time.perf_counter() - started
    append_log(args.log_path, record)
    print(json.dumps(record, indent=2, allow_nan=False))
    if record["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
