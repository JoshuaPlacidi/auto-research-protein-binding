#!/usr/bin/env python3
"""Build a read-only replay file from one recorded research session.

Reads experiments/<session>/summary.json, controller.jsonl and the per-experiment
logs. Writes docs/data/experiments.json. Does not modify the session, call a
model, or copy checkpoints.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
DEFAULT_OUTPUT = REPO / "docs" / "data" / "experiments.json"

NOT_RECORDED = "Not recorded"
DECISION_NOTE = (
    "This session does not record a post-evaluation keep, discard, investigate, "
    "or failed decision. The text below is the proposal written before training."
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def choose_session(root: Path, requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        if not path.is_dir():
            path = root / requested
        if not path.is_dir():
            raise SystemExit(f"Session not found: {requested}")
        return path
    active = root / "active"
    if active.is_file():
        name = active.read_text(encoding="utf-8").strip()
        if name and (root / name).is_dir():
            return root / name
    sessions = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not sessions:
        raise SystemExit(f"No session directories in {root}")
    return sessions[-1]


def last_event(records: list[dict], event: str) -> dict | None:
    found = None
    for record in records:
        if record.get("event") == event:
            found = record
    return found


def directory_status(directory: Path) -> str:
    records = load_jsonl(directory / "log.jsonl")
    if not records:
        return "incomplete"
    evaluation = last_event(records, "model_evaluation")
    failure = last_event(records, "experiment_failure")
    failed_after = bool(
        failure and evaluation and str(failure.get("timestamp", "")) > str(evaluation.get("timestamp", ""))
    )
    if evaluation and evaluation.get("status") == "valid" and not failed_after:
        return "valid"
    if failure or evaluation:
        return "failed"
    return "incomplete"


def diff_line_counts(previous: Path | None, current: Path | None) -> dict | None:
    if previous is None or current is None or not previous.is_file() or not current.is_file():
        return None
    before = previous.read_text(encoding="utf-8").splitlines()
    after = current.read_text(encoding="utf-8").splitlines()
    added = removed = 0
    for line in difflib.ndiff(before, after):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return {"lines_added": added, "lines_removed": removed}


def proposal_prompt_tokens(records: list[dict]) -> int | None:
    chosen = None
    for record in records:
        if record.get("event") == "api_response" and record.get("phase") == "proposal":
            chosen = record
    count = ((chosen or {}).get("token_usage") or {}).get("prompt_token_count")
    return count if isinstance(count, int) else None


def public_error(error) -> str | None:
    if not error:
        return None
    text = re.sub(r"/[\w./-]+", "[path]", str(error))
    if "RESOURCE_EXHAUSTED" in text or "spending cap" in text:
        return "API request failed: billing spending cap reached (429)."
    if "Protected input changed" in text:
        return text.split(";")[0][:220]
    return text[:220]


def annotate_lineage(replay: list[dict], ordered_ids: list[str]) -> None:
    position = {experiment_id: index for index, experiment_id in enumerate(ordered_ids)}
    for row in replay:
        parent = row.get("parent")
        index = position.get(row.get("experiment_id"))
        previous = ordered_ids[index - 1] if index else None
        row["previous_experiment_id"] = previous
        if parent == "starter":
            row["lineage"] = "starter"
            row["backtrack"] = False
        elif isinstance(parent, str) and parent.isdigit():
            row["backtrack"] = parent != previous
            row["lineage"] = "backtrack" if row["backtrack"] else "continued"
        else:
            row["parent"] = None
            row["lineage"] = "not recorded"
            row["backtrack"] = False


def public_experiment(row: dict, directory: Path, parent_ml: Path | None) -> dict:
    metrics = row.get("metrics") or {}
    training = row.get("training") or {}
    groups = row.get("groups") or {}
    records = load_jsonl(directory / "log.jsonl")
    unseen_hla = groups.get("unseen_hla") or {}
    return {
        "experiment_id": row.get("experiment_id"),
        "timestamp": row.get("timestamp"),
        "status": row.get("status") or "incomplete",
        "parent": row.get("parent"),
        "hypothesis": row.get("hypothesis") or NOT_RECORDED,
        "proposal_evidence": row.get("evidence") or NOT_RECORDED,
        "change_summary": row.get("change") or NOT_RECORDED,
        "n_edits": row.get("n_edits"),
        "source_diff": diff_line_counts(parent_ml, directory / "ml.py"),
        "validation_loss": metrics.get("validation_loss"),
        "validation_auroc": metrics.get("auroc"),
        "validation_auprc": metrics.get("auprc"),
        "accuracy": metrics.get("accuracy"),
        "peptide_macro_auroc": metrics.get("peptide_macro_auroc"),
        "peptide_macro_auprc": metrics.get("peptide_macro_auprc"),
        "unseen_hla_auroc": unseen_hla.get("auroc"),
        "unseen_hla_n": unseen_hla.get("n"),
        "evaluation_rows": metrics.get("n"),
        "decision": NOT_RECORDED,
        "scientific_justification": NOT_RECORDED,
        "decision_note": DECISION_NOTE,
        "runtime_seconds": training.get("runtime_seconds"),
        "epochs_completed": training.get("epochs_completed"),
        "epochs_planned": training.get("epochs_planned"),
        "recorded_training_loss": training.get("loss") or NOT_RECORDED,
        "best_so_far_loss": row.get("best_validation_loss"),
        "best_so_far_auroc": row.get("best_auroc"),
        "loss_vs_parent": row.get("loss_vs_parent"),
        "auroc_vs_parent": row.get("auroc_vs_parent"),
        "proposal_prompt_tokens": row.get("context_tokens") if isinstance(row.get("context_tokens"), int) else proposal_prompt_tokens(records),
        "error": public_error(row.get("error")),
    }


def controller_meta(session: Path) -> dict:
    records = load_jsonl(session / "controller.jsonl")
    start = next((row for row in records if row.get("event") == "session_start"), {})
    budgets = start.get("budgets") or {}
    return {
        "recorded_provider": budgets.get("provider") or NOT_RECORDED,
        "recorded_model": budgets.get("model") or NOT_RECORDED,
        "session_started_at": start.get("timestamp") or NOT_RECORDED,
        "full_history_included": NOT_RECORDED,
    }


def build(session: Path) -> dict:
    summary_path = session / "summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {"experiments": []}
    recorded = [row for row in summary.get("experiments", []) if isinstance(row, dict)]
    by_id = {row.get("experiment_id"): row for row in recorded}
    directories = [path for path in sorted(session.iterdir()) if path.is_dir() and path.name.isdigit()]

    replay = []
    excluded = []
    for directory in directories:
        experiment_id = directory.name
        row = by_id.get(experiment_id)
        if row is None:
            status = directory_status(directory)
            excluded.append({
                "experiment_id": experiment_id,
                "status": status,
                "reason": "Not in summary.json. Excluded from the replay.",
            })
            continue
        parent = row.get("parent")
        parent_ml = session / parent / "ml.py" if isinstance(parent, str) and parent.isdigit() else None
        status = row.get("status") or directory_status(directory)
        if status != "valid":
            item = public_experiment(row, directory, parent_ml)
            item["status"] = status
            if status == "failed":
                replay.append(item)
            else:
                excluded.append({
                    "experiment_id": experiment_id,
                    "status": status,
                    "reason": "Incomplete experiment excluded from the replay.",
                })
            continue
        replay.append(public_experiment(row, directory, parent_ml))
    annotate_lineage(replay, [directory.name for directory in directories])

    valid = [row for row in replay if row["status"] == "valid" and row["validation_loss"] is not None]
    best_loss = min(valid, key=lambda row: (row["validation_loss"], row["experiment_id"])) if valid else None
    ranked_auroc = [row for row in valid if row["validation_auroc"] is not None]
    best_auroc = max(ranked_auroc, key=lambda row: (row["validation_auroc"], row["experiment_id"])) if ranked_auroc else None
    return {
        "session_id": session.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_definition": (
            "validation_auroc is the fixed evaluator's AUROC. Higher is better. "
            "The replay tracks this metric so sessions can be compared."
        ),
        "decision_recording": DECISION_NOTE,
        "running_best_definition": (
            "best_so_far_auroc is the running maximum validation AUROC among valid experiments, "
            "in chronological order. It is not a keep decision."
        ),
        "backtrack_definition": (
            "A backtrack is recorded when parent is an earlier experiment id, "
            "not the immediately previous experiment. The controller copies that parent's ml.py."
        ),
        **controller_meta(session),
        "completed_experiments": len(replay),
        "valid_experiments": len(valid),
        "failed_experiments": sum(1 for row in replay if row["status"] == "failed"),
        "backtracks": [
            {"experiment_id": row["experiment_id"], "parent": row.get("parent"), "previous_experiment_id": row.get("previous_experiment_id")}
            for row in replay if row.get("backtrack")
        ],
        "recorded_keep_decisions": 0,
        "best_validation_loss_experiment": None if best_loss is None else {
            "experiment_id": best_loss["experiment_id"],
            "validation_loss": best_loss["validation_loss"],
            "validation_auroc": best_loss["validation_auroc"],
        },
        "highest_auroc_experiment": None if best_auroc is None else {
            "experiment_id": best_auroc["experiment_id"],
            "validation_auroc": best_auroc["validation_auroc"],
            "validation_loss": best_auroc["validation_loss"],
        },
        "excluded": excluded,
        "experiments": replay,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="Session directory name or path. Defaults to experiments/active.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    session = choose_session(EXPERIMENTS, args.session)
    payload = build(session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    best = payload["highest_auroc_experiment"] or {}
    print(
        f"session {payload['session_id']}: {payload['completed_experiments']} in replay, "
        f"{len(payload['excluded'])} excluded, "
        f"highest AUROC {best.get('experiment_id')} {best.get('validation_auroc')}"
    )
    print(args.output)


if __name__ == "__main__":
    main()
