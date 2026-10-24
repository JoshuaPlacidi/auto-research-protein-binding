"""Evaluate a frozen recorded model on a user-supplied TEST CSV, without Argon.

    python final_eval.py --test-path /absolute/path/to/test.csv
    python final_eval.py --test-path /path/test.csv --experiment-id SESSION_0001

Defaults to the latest experiment with a valid validation record, searching
every session. Pass `--session` to stay inside one trace. Takes the controller
lock, freezes that session against further research, and appends final test
results to the experiment log. Never overwrites repository ml.py, trains a
model or calls a model API.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import types
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
LOCK_PATH = EXPERIMENTS / ".lock"


def read_records(path):
    records = []
    if not Path(path).is_file():
        return records
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        except json.JSONDecodeError:
            pass
    return records


def append_log(path, event, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    return record


def session_directories():
    if not EXPERIMENTS.is_dir():
        return []
    return [path for path in sorted(EXPERIMENTS.iterdir())
            if path.is_dir() and (path / "controller.jsonl").is_file()]


def experiment_dirs(session=None):
    sessions = []
    if session:
        directory = EXPERIMENTS / session
        if not (directory / "controller.jsonl").is_file():
            raise ValueError(f"No session {session}")
        sessions = [directory]
    else:
        sessions = session_directories()
    found = []
    for session_dir in sessions:
        found.extend(path for path in sorted(session_dir.iterdir())
                     if path.is_dir() and (path / "ml.py").is_file())
    return found


def valid_evaluations(directory):
    return [r for r in read_records(directory / "log.jsonl")
            if r.get("event") == "model_evaluation" and r.get("status") == "valid"
            and r.get("split") == "validation"]


def resolve_model(experiment_id=None, session=None):
    """Resolve source and verified checkpoint from a session's experiment directory."""
    directories = experiment_dirs(session)
    if experiment_id:
        matches = [path for path in directories if path.name == experiment_id]
        if not matches:
            raise ValueError(f"Unknown experiment {experiment_id}")
        if len(matches) > 1:
            raise ValueError(f"Experiment {experiment_id} exists in multiple sessions; pass --session")
        directory = matches[0]
        evaluations = valid_evaluations(directory)
        if not evaluations:
            raise ValueError("Model requires a valid validation evaluation")
        evaluation = evaluations[-1]
    else:
        ranked = []
        for directory in directories:
            for record in valid_evaluations(directory):
                ranked.append((record.get("timestamp", ""), directory, record))
        if not ranked:
            raise ValueError("No experiment with a valid validation record; specify --experiment-id")
        ranked.sort()
        _, directory, evaluation = ranked[-1]
        experiment_id = directory.name
    source_path = directory / "ml.py"
    source = source_path.read_text(encoding="utf-8")
    from eval import sha256
    if sha256(source_path) != evaluation["ml_sha256"]:
        raise ValueError("Experiment source does not match the evaluated model")
    checkpoint = directory / "model.pth"
    if not checkpoint.is_file() or sha256(checkpoint) != evaluation["checkpoint_sha256"]:
        raise ValueError("Saved validation checkpoint is missing or changed")
    return experiment_id, source, checkpoint, evaluation, directory


def load_frozen_module(source, directory):
    """Execute trusted recorded model code in memory, leaving repository ml.py alone."""
    module = types.ModuleType("frozen_research_ml")
    module.__file__ = str(directory / "ml.py")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "<recorded ml.py>", "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module.__name__, None)
        raise
    module.EXPERIMENT_DIR = str(directory)
    return module


def development_hashes():
    hashes = set()
    for directory in experiment_dirs():
        for record in read_records(directory / "log.jsonl"):
            if record.get("event") == "model_evaluation":
                hashes.update(record.get(key) for key in ("evaluation_data_sha256", "training_data_sha256"))
    hashes.discard(None)
    return hashes


def evaluate_final(args):
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    test_path = Path(args.test_path).expanduser().resolve()
    if not test_path.is_file():
        raise ValueError("--test-path must point to an existing labelled CSV")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Controller is active: stop it before final evaluation") from None
        experiment_id, source, checkpoint, validation, directory = resolve_model(args.experiment_id, args.session)
        session_log = directory.parent / "controller.jsonl"
        records = read_records(session_log)
        from eval import sha256
        data_sha = sha256(test_path)
        if data_sha in development_hashes():
            raise ValueError("Test CSV matches a recorded training/validation file")
        if sha256(ROOT / "eval.py") != validation["evaluator_sha256"]:
            raise ValueError("Fixed evaluator differs from the validation protocol")
        train_path = args.train_path or os.getenv("TRAIN_PATH")
        if train_path and sha256(train_path) != validation.get("training_data_sha256"):
            raise ValueError("Training CSV differs from the recorded training reference")
        if not any(r.get("event") == "research_finalized" for r in records):
            append_log(session_log, "research_finalized", experiment_id=experiment_id,
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
            module = load_frozen_module(source, directory)
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
        append_log(directory / "log.jsonl", "final_model_evaluation", **record)
        print(json.dumps(record, indent=2, allow_nan=False))
        if record["status"] != "valid":
            raise RuntimeError("Final evaluation failed; details are in the experiment log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", required=True, help="User-supplied labelled test CSV")
    parser.add_argument("--session", help="Session directory name under experiments/. Default searches every session")
    parser.add_argument("--experiment-id", help="Default: latest experiment with a valid validation report")
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
