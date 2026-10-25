#!/usr/bin/env python3
"""Build a committed dashboard summary from local experiment sessions.

Reads experiments/<session>/summary.json and the first session_start event in
controller.jsonl. Writes site/data/sessions.json. Does not copy checkpoints,
model files, per-experiment logs, or controller fields that contain local paths.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = REPO / "experiments"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "sessions.json"
STARTER_ML = REPO / "ml.py"
GEMINI_MODEL = "gemini-4-argon"

METRIC = (
    "validation_auroc is the fixed evaluator's AUROC. Higher is better. "
    "The x-axis is experiment number within a session."
)
CONCLUSION_PROMPT = """Write one short paragraph, three to five sentences, for a researcher skimming this autonomous modelling session.
Name the ideas that actually raised validation AUROC, and state the best AUROC and AUPRC.
Do not mention file paths, token counts, experiment ids, or that you are a model.

Research model: {model}

Improvements in chronological order:
{improvements}
"""


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


def parent_ml(directory: Path, parent) -> Path | None:
    if parent == "starter":
        return STARTER_ML if STARTER_ML.is_file() else None
    if isinstance(parent, str) and parent.isdigit():
        path = directory / parent / "ml.py"
        return path if path.is_file() else None
    return None


def ml_diff(directory: Path, experiment_id, parent) -> str | None:
    current = directory / str(experiment_id) / "ml.py"
    previous = parent_ml(directory, parent)
    if not current.is_file() or previous is None:
        return None
    before = previous.read_text(encoding="utf-8").splitlines()
    after = current.read_text(encoding="utf-8").splitlines()
    lines = list(difflib.unified_diff(
        before,
        after,
        fromfile=f"{parent}/ml.py",
        tofile=f"{experiment_id}/ml.py",
        lineterm="",
    ))
    return "\n".join(lines) if lines else None


def public_experiment(row: dict, parent_row: dict | None) -> dict:
    metrics = row.get("metrics") or {}
    parent_metrics = (parent_row or {}).get("metrics") or {}
    training = row.get("training") or {}
    auprc = number(metrics.get("auprc"))
    parent_auprc = number(parent_metrics.get("auprc"))
    return {
        "id": row.get("experiment_id"),
        "timestamp": row.get("timestamp"),
        "status": row.get("status") or "incomplete",
        "parent": row.get("parent"),
        "title": row.get("title") or None,
        "hypothesis": row.get("hypothesis") or None,
        "change": row.get("change") or None,
        "validation_auroc": number(metrics.get("auroc")),
        "validation_auprc": auprc,
        "validation_loss": number(metrics.get("validation_loss")),
        "auroc_vs_parent": number(row.get("auroc_vs_parent")),
        "auprc_vs_parent": None if auprc is None or parent_auprc is None else auprc - parent_auprc,
        "runtime_seconds": number(training.get("runtime_seconds")),
        "error": public_error(row.get("error")),
        "improved": False,
    }


def mark_improvements(experiments: list[dict], directory: Path) -> None:
    best = None
    for experiment in experiments:
        auroc = experiment["validation_auroc"] if experiment["status"] == "valid" else None
        improved = auroc is not None and best is not None and auroc > best
        experiment["improved"] = improved
        if improved:
            experiment["diff"] = ml_diff(directory, experiment["id"], experiment["parent"])
        if auroc is not None:
            best = auroc if best is None else max(best, auroc)


def conclusion_key(experiments: list[dict]) -> str:
    payload = [
        {
            "id": row["id"],
            "auroc": row["validation_auroc"],
            "auprc": row["validation_auprc"],
            "change": row.get("change"),
        }
        for row in experiments
        if row.get("improved")
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def improvement_brief(experiments: list[dict]) -> str:
    by_id = {row["id"]: row for row in experiments}
    lines = []
    for row in experiments:
        if not row.get("improved"):
            continue
        parent = by_id.get(row.get("parent")) or {}
        lines.append(
            f"- {row.get('title') or 'Untitled'}: {row.get('change') or 'No change recorded.'} "
            f"AUROC {parent.get('validation_auroc')} -> {row.get('validation_auroc')}. "
            f"AUPRC {parent.get('validation_auprc')} -> {row.get('validation_auprc')}."
        )
    return "\n".join(lines)


def load_previous_conclusions(path: Path) -> dict[str, tuple[str, str]]:
    if not path.is_file():
        return {}
    try:
        data = load_json(path)
    except json.JSONDecodeError:
        return {}
    found = {}
    for session in data.get("sessions") or []:
        session_id = session.get("id")
        key = session.get("conclusion_key")
        text = session.get("conclusion")
        if session_id and key and text:
            found[session_id] = (key, text)
    return found


def ask_gemini(prompt: str) -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY missing; session conclusions left blank")
        return None
    from google import genai
    client = genai.Client()
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (response.text or "").strip()
        return text or None
    finally:
        client.close()


def fill_conclusion(session: dict, previous: dict[str, tuple[str, str]]) -> None:
    key = conclusion_key(session["experiments"])
    session["conclusion_key"] = key
    cached = previous.get(session["id"])
    if cached and cached[0] == key:
        session["conclusion"] = cached[1]
        return
    brief = improvement_brief(session["experiments"])
    if not brief:
        session["conclusion"] = None
        return
    prompt = CONCLUSION_PROMPT.format(model=session.get("model") or "unknown", improvements=brief)
    try:
        session["conclusion"] = ask_gemini(prompt)
    except Exception as exc:
        print(f"{session['id']}: Gemini conclusion failed ({exc})")
        session["conclusion"] = None


def public_session(directory: Path) -> dict | None:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return None
    summary = load_json(summary_path)
    start = first_session_start(directory)
    budgets = start.get("budgets") or {}
    rows = [row for row in summary.get("experiments") or [] if isinstance(row, dict)]
    by_id = {row.get("experiment_id"): row for row in rows}
    experiments = [
        public_experiment(row, by_id.get(row.get("parent")) if isinstance(row.get("parent"), str) else None)
        for row in rows
    ]
    mark_improvements(experiments, directory)
    return {
        "id": summary.get("session_id") or directory.name,
        "started_at": start.get("timestamp") or summary.get("updated_at"),
        "provider": budgets.get("provider"),
        "model": budgets.get("model"),
        "context_tokens": context_tokens(rows),
        "conclusion": None,
        "conclusion_key": None,
        "experiments": experiments,
    }


def session_directories(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def build(root: Path, previous: dict[str, tuple[str, str]]) -> dict:
    sessions = []
    for directory in session_directories(root):
        session = public_session(directory)
        if session is None:
            continue
        fill_conclusion(session, previous)
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
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    payload = build(args.experiments, load_previous_conclusions(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    n_experiments = sum(len(session["experiments"]) for session in payload["sessions"])
    n_improved = sum(1 for session in payload["sessions"] for row in session["experiments"] if row.get("improved"))
    n_conclusions = sum(1 for session in payload["sessions"] if session.get("conclusion"))
    print(
        f"{len(payload['sessions'])} sessions, {n_experiments} experiments, "
        f"{n_improved} AUROC improvements, {n_conclusions} conclusions -> {args.output}"
    )


if __name__ == "__main__":
    main()
