"""Evaluate a frozen recorded model on a user-supplied TEST CSV, without Argon.

    python final_eval.py --test-path /absolute/path/to/test.csv
    python final_eval.py --test-path /path/test.csv --experiment-id SESSION_0001

Defaults to the last scientifically selected model. Takes the shared log lock,
freezes this history against further research, and appends final test results.
Never overwrites the current ml.py, trains a model or calls a model API.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def read_records(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        except json.JSONDecodeError:
            pass
    return records


def append_log(path, event, **fields):
    record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    return record


def resolve_model(records, experiment_id=None):
    """Resolve source and verified checkpoint from validation history only."""
    incumbent = None
    for record in records:
        if record.get("event") == "selection":
            incumbent = record.get("incumbent")
    experiment_id = experiment_id or (incumbent or {}).get("experiment_id")
    if not experiment_id:
        raise ValueError("No selected model; specify a recorded --experiment-id")
    proposals = [r for r in records if r.get("event") == "proposal" and r.get("experiment_id") == experiment_id]
    evaluations = [r for r in records if r.get("event") == "model_evaluation"
                   and r.get("experiment_id") == experiment_id and r.get("split") == "validation"
                   and r.get("status") == "valid"]
    if not proposals or not evaluations:
        raise ValueError("Model requires both recorded source and a valid validation evaluation")
    proposal, evaluation = proposals[-1], evaluations[-1]
    source = proposal["ml_py"]
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_sha != evaluation["ml_sha256"]:
        raise ValueError("Recorded source does not match the evaluated model")
    directory = Path(evaluation["experiment_dir"])
    if not directory.is_absolute():
        directory = ROOT / directory
    checkpoint = directory / "model.pth"
    from eval import sha256
    if not checkpoint.is_file() or sha256(checkpoint) != evaluation["checkpoint_sha256"]:
        raise ValueError("Saved validation checkpoint is missing or changed")
    return experiment_id, source, checkpoint, evaluation


def load_frozen_module(source, directory):
    """Execute trusted recorded model code in memory, leaving working ml.py alone."""
    module = types.ModuleType("frozen_research_ml")
    module.__file__ = str(ROOT / "ml.py")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "<recorded ml.py>", "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module.__name__, None)
        raise
    module.EXPERIMENT_DIR = str(directory)
    return module


def evaluate_final(args):
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    log = ROOT / "experiments/evaluation.log"
    if not log.is_file():
        raise ValueError("Research log is missing; no model/source history can be verified")
    test_path = Path(args.test_path).expanduser().resolve()
    if not test_path.is_file():
        raise ValueError("--test-path must point to an existing labelled CSV")
    with log.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Controller is active: stop it before final evaluation") from None
        records = read_records(log)
        experiment_id, source, checkpoint, validation = resolve_model(records, args.experiment_id)
        from eval import sha256
        # Reject an accidental evaluation of a supplied development partition.
        data_sha = sha256(test_path)
        development_hashes = {r.get(key) for r in records if r.get("event") == "model_evaluation"
                              for key in ("evaluation_data_sha256", "training_data_sha256")}
        if data_sha in development_hashes:
            raise ValueError("Test CSV matches a recorded training/validation file")
        if sha256(ROOT / "eval.py") != validation["evaluator_sha256"]:
            raise ValueError("Fixed evaluator differs from the validation protocol")
        train_path = args.train_path or os.getenv("TRAIN_PATH")
        if train_path and sha256(train_path) != validation.get("training_data_sha256"):
            raise ValueError("Training CSV differs from the recorded training reference")
        # Freeze BEFORE inference/test-label access. Even a failed final test run
        # must not allow test feedback to reach another research decision.
        if not any(r.get("event") == "research_finalized" for r in records):
            append_log(log, "research_finalized", experiment_id=experiment_id,
                       reason="Operator started final test evaluation; research on this history cannot resume")
        started = time.monotonic()
        record = {"experiment_id": experiment_id, "split": "test", "status": "failed",
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                  "ml_sha256": validation["ml_sha256"], "test_data_sha256": data_sha,
                  "evaluator_sha256": validation["evaluator_sha256"],
                  "loss_definition": "probability MSE/Brier score; common subgroup field validation_loss denotes the same fixed loss on TEST rows"}
        try:
            import numpy as np
            import pandas as pd
            import torch
            from eval import eligible_rows, evaluate_model, build_report
            raw = pd.read_csv(test_path, low_memory=False)
            cohort, indices = eligible_rows(raw)
            if cohort.empty:
                raise ValueError("No eligible full-chain test rows")
            module = load_frozen_module(source, checkpoint.parent)
            dataset = module.Dataset(str(test_path))
            required = ["TCR_full", "Peptide", "HLA_sequence", "binding_flag", "tcra_len", "tcrb_len"]
            if not dataset.data[required].reset_index(drop=True).equals(raw[required].reset_index(drop=True)):
                raise ValueError("Recorded Dataset changed CSV row order, inputs or labels")
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices.tolist()),
                                                 batch_size=128, shuffle=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = module.Model().to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            labels, scores = evaluate_model(model, loader, device, return_predictions=True)
            if len(scores) != len(cohort) or not np.array_equal(labels, cohort.binding_flag.to_numpy()):
                raise ValueError("Test predictions/labels do not match the fixed full-chain cohort")
            train = None
            if train_path:
                train, _ = eligible_rows(pd.read_csv(train_path, low_memory=False))
            report = build_report(cohort, scores, train, args.category_column)
            record.update(status="valid", test_loss=report["overall"]["mse"], metrics=report,
                          cohort={"rows": len(raw), "eligible": len(cohort),
                                  "excluded_missing_chain": len(raw)-len(cohort)})
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            sys.modules.pop("frozen_research_ml", None)
        record["evaluation_runtime_seconds"] = time.monotonic() - started
        append_log(log, "final_model_evaluation", **record)
        print(json.dumps(record, indent=2, allow_nan=False))
        if record["status"] != "valid":
            raise RuntimeError("Final evaluation failed; details are in the shared log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", required=True, help="User-supplied labelled test CSV")
    parser.add_argument("--experiment-id", help="Default: last kept model; must have a valid validation report")
    parser.add_argument("--train-path", help="Original training CSV for derived novelty categories; defaults to TRAIN_PATH")
    parser.add_argument("--category-column")
    args = parser.parse_args()
    try:
        evaluate_final(args)
    except Exception as exc:
        print(f"Final evaluation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
