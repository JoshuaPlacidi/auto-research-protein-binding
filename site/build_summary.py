#!/usr/bin/env python3
"""Build a committed dashboard summary from local experiment sessions.

Reads experiments/<session>/summary.json and the first session_start event in
controller.jsonl. Writes site/data/sessions.json. Does not copy checkpoints,
model files, per-experiment logs, or controller fields that contain local paths.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = REPO / "experiments"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "sessions.json"

METRIC = (
    "validation_auroc is the fixed evaluator's AUROC. Higher is better. "
    "The x-axis is experiment number within a session."
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


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def public_error(error) -> str | None:
    if not error:
        return None
    text = re.sub(r"(?:/[\w.~+-]+)+", "[path]", str(error))
    return text[:240] or None


def first_session_start(session: Path) -> dict:
    for row in load_jsonl(session / "controller.jsonl"):
        if row.get("event") == "session_start":
            return row
    return {}


def context_tokens(rows: list[dict]) -> int | None:
    """Largest proposal prompt sent in the session. Each later call already includes that history."""
    counts = [row.get("context_tokens") for row in rows if isinstance(row.get("context_tokens"), int)]
    return max(counts) if counts else None


def public_experiment(row: dict) -> dict:
    metrics = row.get("metrics") or {}
    training = row.get("training") or {}
    return {
        "id": row.get("experiment_id"),
        "timestamp": row.get("timestamp"),
        "status": row.get("status") or "incomplete",
        "parent": row.get("parent"),
        "title": row.get("title") or None,
        "hypothesis": row.get("hypothesis") or None,
        "change": row.get("change") or None,
        "validation_auroc": number(metrics.get("auroc")),
        "validation_auprc": number(metrics.get("auprc")),
        "validation_loss": number(metrics.get("validation_loss")),
        "auroc_vs_parent": number(row.get("auroc_vs_parent")),
        "runtime_seconds": number(training.get("runtime_seconds")),
        "error": public_error(row.get("error")),
    }


def public_session(directory: Path) -> dict | None:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return None
    summary = load_json(summary_path)
    start = first_session_start(directory)
    budgets = start.get("budgets") or {}
    rows = [row for row in summary.get("experiments") or [] if isinstance(row, dict)]
    experiments = [public_experiment(row) for row in rows]
    return {
        "id": summary.get("session_id") or directory.name,
        "started_at": start.get("timestamp") or summary.get("updated_at"),
        "provider": budgets.get("provider"),
        "model": budgets.get("model"),
        "context_tokens": context_tokens(rows),
        "experiments": experiments,
    }


def session_directories(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def build(root: Path) -> dict:
    sessions = []
    for directory in session_directories(root):
        session = public_session(directory)
        if session is not None:
            sessions.append(session)
    sessions.sort(key=lambda row: (row.get("started_at") or "", row["id"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": METRIC,
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.experiments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    n_experiments = sum(len(session["experiments"]) for session in payload["sessions"])
    print(
        f"{len(payload['sessions'])} sessions, {n_experiments} experiments -> {args.output}"
    )


if __name__ == "__main__":
    main()
