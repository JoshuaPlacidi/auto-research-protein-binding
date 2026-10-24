"""Bounded research controller; run from the repository root.

    python agent/agent.py --dry-run
    python agent/agent.py --max-experiments 2 --run-timeout 900
    python agent/agent.py --resume --max-experiments 2
    python agent/agent.py --provider openrouter --model vendor/name --max-input-tokens 100000 --max-output-tokens 8192

A launch creates experiments/<session>/ and keeps that trace's copies, logs and
checkpoints inside it. summary.json and summary.csv in that directory are rebuilt
from the experiment logs after each attempt: what changed, and the headline
validation metrics. Another launch starts a separate trace. --resume continues
the latest trace; --session NAME continues a named one. The repository ml.py is
never written. This is an orchestration boundary, not an OS sandbox for generated Python.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
LOCK_PATH = EXPERIMENTS / ".lock"
ACTIVE_PATH = EXPERIMENTS / "active"
STARTER = ROOT / "ml.py"
SESSION_DIR = None
SESSION_STARTED_AT = None
MAX_EDITS = 8
MAX_REPAIRS = 5
HIGHLIGHT_GROUPS = (
    "unseen_tcr", "unseen_peptide", "unseen_hla", "unseen_pmhc",
    "unseen_tcr_and_pmhc", "completely_unseen_tcr_peptide_hla", "fully_seen_components",
)
_GROUP_METRICS = ("auroc", "auprc")
_SUMMARY_COLUMNS = (
    "session_id", "experiment_id", "timestamp", "status", "parent", "n_edits",
    "hypothesis", "change",
    "validation_loss", "auroc", "auprc", "accuracy", "binary_log_loss",
    "peptide_macro_auroc", "peptide_macro_auprc",
    "n", "n_positive", "n_negative",
    "best_validation_loss", "best_auroc", "loss_vs_parent", "auroc_vs_parent",
    "epochs_completed", "epochs_planned", "learning_rate", "training_objective",
    "train_runtime_seconds", "final_train_loss", "context_tokens",
    *[f"{name}_{metric}" for name in HIGHLIGHT_GROUPS for metric in _GROUP_METRICS],
    "error",
)


class StopRequested(Exception):
    """A configured budget or operator request ended the search."""


def controller_log():
    return SESSION_DIR / "controller.jsonl"


def check_stop():
    if SESSION_STARTED_AT is None or SESSION_DIR is None:
        return
    for record in reversed(read_records(controller_log())):
        if record.get("event") == "stop_requested" and record.get("timestamp", "") >= SESSION_STARTED_AT:
            raise StopRequested("Operator requested a stop through the controller log")


def say(text=""):
    print(text, flush=True)


def preview_lines(text, limit=8):
    lines = text.splitlines() or [""]
    shown = []
    for line in lines[:limit]:
        shown.append(line if len(line) <= 160 else line[:157] + "...")
    if len(lines) > limit:
        shown.append(f"... {len(lines) - limit} more lines")
    return shown


def show_proposal(proposal):
    say(f"  parent  {proposal['parent']}")
    brief = " ".join(proposal["change"].split())
    say(f"  change  {brief if len(brief) <= 240 else brief[:237] + '...'}")
    edits = proposal.get("edits") or []
    if not edits:
        say("  edits   none")
        return
    say(f"  edits   {len(edits)}")
    for index, edit in enumerate(edits, 1):
        say(f"  --- {index}")
        for line in preview_lines(edit.get("old", "")):
            say(f"  - {line}")
        for line in preview_lines(edit.get("new", "")):
            say(f"  + {line}")


def metric_text(value):
    return "—" if value is None else f"{value:.4f}"


def prompt_context(directory):
    """Provider prompt-token count, or the preflight count when the response has none."""
    request = response = None
    for record in read_records(directory / "log.jsonl"):
        if record.get("phase") != "proposal":
            continue
        if record.get("event") == "api_request":
            request = record
        elif record.get("event") == "api_response":
            response = record
    actual = ((response or {}).get("token_usage") or {}).get("prompt_token_count")
    if isinstance(actual, int):
        return actual
    estimate = (request or {}).get("input_tokens")
    return estimate if isinstance(estimate, int) else None


def show_result(experiment_id, record):
    if record:
        overall = (record.get("metrics") or {}).get("overall") or {}
        metrics = record.get("metrics") or {}
        say(f"  result  loss {metric_text(overall.get('validation_loss'))}"
            f"  auroc {metric_text(overall.get('auroc'))}"
            f"  auprc {metric_text(overall.get('auprc'))}"
            f"  macro {metric_text(metrics.get('peptide_macro_auroc'))}/{metric_text(metrics.get('peptide_macro_auprc'))}")
    say("  trace")
    for directory in experiment_dirs():
        evaluations = [r for r in read_records(directory / "log.jsonl")
                       if r.get("event") == "model_evaluation" and r.get("status") == "valid"]
        if evaluations:
            row = evaluations[-1].get("metrics", {}).get("overall", {})
            mark = "*" if directory.name == experiment_id else " "
            say(f"   {mark} {directory.name}  loss {metric_text(row.get('validation_loss'))}"
                f"  auroc {metric_text(row.get('auroc'))}  auprc {metric_text(row.get('auprc'))}"
                f"  ctx {prompt_context(directory) or '—'}")
        elif directory.name == experiment_id or any(
                r.get("event") == "experiment_failure" for r in read_records(directory / "log.jsonl")):
            mark = "*" if directory.name == experiment_id else " "
            ctx = prompt_context(directory)
            say(f"   {mark} {directory.name}  failed" + (f"  ctx {ctx}" if ctx else ""))


def append_log(path, event, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    return record


def read_full_history(path):
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_records(path):
    records = []
    for line in read_full_history(path).splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        except json.JSONDecodeError:
            pass
    return records


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def session_directories():
    if not EXPERIMENTS.is_dir():
        return []
    return [path for path in sorted(EXPERIMENTS.iterdir())
            if path.is_dir() and (path / "controller.jsonl").is_file()]


def experiment_dirs():
    if SESSION_DIR is None or not SESSION_DIR.is_dir():
        return []
    return [path for path in sorted(SESSION_DIR.iterdir())
            if path.is_dir() and (path / "ml.py").is_file()]


def _last_event(records, event, **match):
    found = None
    for record in records:
        if record.get("event") != event:
            continue
        if all(record.get(key) == value for key, value in match.items()):
            found = record
    return found


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return None
    return value


def _first_timestamp(*records):
    for record in records:
        if record and record.get("timestamp"):
            return record["timestamp"]
    return None


def _group_metrics(summary):
    if not isinstance(summary, dict):
        return None
    return {key: _number(summary.get(key)) for key in ("n", "validation_loss", "auroc", "auprc")}


def summarize_experiment(directory):
    """One experiment row: the model's stated change and headline validation metrics."""
    records = read_records(directory / "log.jsonl")
    proposal = _last_event(records, "proposal") or {}
    evaluation = _last_event(records, "model_evaluation")
    failure = _last_event(records, "experiment_failure")
    config = _last_event(records, "training_configuration") or {}
    epochs = [record for record in records if record.get("event") == "training_epoch"]
    training_run = _last_event(records, "execution", stage="training")
    failed_after = failure and evaluation and failure.get("timestamp", "") > evaluation.get("timestamp", "")
    valid = bool(evaluation and evaluation.get("status") == "valid" and not failed_after)
    if valid:
        status, error = "valid", None
    elif failure or evaluation:
        status = "failed"
        error = (failure or evaluation).get("error")
    else:
        status, error = "incomplete", None
    report = (evaluation or {}).get("metrics") or {}
    overall = report.get("overall") or {}
    metrics = groups = provided = None
    if valid:
        metrics = {
            "validation_loss": _number(overall.get("validation_loss")),
            "auroc": _number(overall.get("auroc")),
            "auprc": _number(overall.get("auprc")),
            "accuracy": _number(overall.get("accuracy")),
            "binary_log_loss": _number(overall.get("binary_log_loss")),
            "peptide_macro_auroc": _number(report.get("peptide_macro_auroc")),
            "peptide_macro_auprc": _number(report.get("peptide_macro_auprc")),
            "n": overall.get("n"),
            "n_positive": overall.get("n_positive"),
            "n_negative": overall.get("n_negative"),
        }
        derived = report.get("derived_categories") or {}
        groups = {name: _group_metrics(derived[name])
                  for name in HIGHLIGHT_GROUPS if isinstance(derived.get(name), dict)} or None
        raw_provided = report.get("provided_categories") or {}
        if isinstance(raw_provided, dict) and raw_provided:
            provided = {name: _group_metrics(value) for name, value in raw_provided.items()
                        if isinstance(value, dict)} or None
    final_epoch = epochs[-1] if epochs else {}
    training = None
    if config or epochs or training_run:
        training = {
            "epochs_planned": config.get("epochs"),
            "epochs_completed": len(epochs),
            "learning_rate": config.get("learning_rate"),
            "loss": config.get("loss"),
            "batch_size": config.get("batch_size"),
            "seed": config.get("seed"),
            "runtime_seconds": None if training_run is None else training_run.get("runtime_seconds"),
            "final_train_loss": _number(final_epoch.get("train_loss")),
            "final_validation_loss": _number(final_epoch.get("validation_loss")),
        }
    edits = proposal.get("edits")
    return {
        "experiment_id": directory.name,
        "timestamp": _first_timestamp(evaluation, failure, proposal, records[-1] if records else None),
        "status": status,
        "parent": proposal.get("parent"),
        "hypothesis": proposal.get("hypothesis"),
        "evidence": proposal.get("evidence"),
        "change": proposal.get("change"),
        "n_edits": len(edits) if isinstance(edits, list) else None,
        "error": error,
        "training": training,
        "metrics": metrics,
        "groups": groups,
        "provided_categories": provided,
        "context_tokens": prompt_context(directory),
    }


def _annotate_summary(rows):
    by_id = {row["experiment_id"]: row for row in rows}
    best_loss = best_auroc = None
    for row in rows:
        metrics = row.get("metrics") or {}
        loss, auroc = metrics.get("validation_loss"), metrics.get("auroc")
        if row["status"] == "valid":
            if loss is not None:
                best_loss = loss if best_loss is None else min(best_loss, loss)
            if auroc is not None:
                best_auroc = auroc if best_auroc is None else max(best_auroc, auroc)
        parent_metrics = (by_id.get(row.get("parent")) or {}).get("metrics") or {}
        parent_loss, parent_auroc = parent_metrics.get("validation_loss"), parent_metrics.get("auroc")
        row["best_validation_loss"] = best_loss
        row["best_auroc"] = best_auroc
        row["loss_vs_parent"] = None if loss is None or parent_loss is None else loss - parent_loss
        row["auroc_vs_parent"] = None if auroc is None or parent_auroc is None else auroc - parent_auroc
    return rows


def _flat_summary_row(session_id, row):
    metrics = row.get("metrics") or {}
    training = row.get("training") or {}
    groups = row.get("groups") or {}
    flat = {
        "session_id": session_id,
        "experiment_id": row["experiment_id"],
        "timestamp": row["timestamp"],
        "status": row["status"],
        "parent": row["parent"],
        "n_edits": row["n_edits"],
        "hypothesis": row.get("hypothesis"),
        "change": row.get("change"),
        "validation_loss": metrics.get("validation_loss"),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "accuracy": metrics.get("accuracy"),
        "binary_log_loss": metrics.get("binary_log_loss"),
        "peptide_macro_auroc": metrics.get("peptide_macro_auroc"),
        "peptide_macro_auprc": metrics.get("peptide_macro_auprc"),
        "n": metrics.get("n"),
        "n_positive": metrics.get("n_positive"),
        "n_negative": metrics.get("n_negative"),
        "best_validation_loss": row.get("best_validation_loss"),
        "best_auroc": row.get("best_auroc"),
        "loss_vs_parent": row.get("loss_vs_parent"),
        "auroc_vs_parent": row.get("auroc_vs_parent"),
        "epochs_completed": training.get("epochs_completed"),
        "epochs_planned": training.get("epochs_planned"),
        "learning_rate": training.get("learning_rate"),
        "training_objective": training.get("loss"),
        "train_runtime_seconds": training.get("runtime_seconds"),
        "final_train_loss": training.get("final_train_loss"),
        "context_tokens": row.get("context_tokens"),
        "error": row.get("error"),
    }
    for name in HIGHLIGHT_GROUPS:
        group = groups.get(name) or {}
        for metric in _GROUP_METRICS:
            flat[f"{name}_{metric}"] = group.get(metric)
    return flat


def _atomic_text(path, text):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_session_summary(session_dir=None):
    """Rebuild summary.json and summary.csv from the experiment logs in one session.

    The per-experiment logs remain the research record. This table is derived:
    one row per experiment, in id order, safe to rebuild on resume.
    """
    session_dir = Path(session_dir) if session_dir is not None else SESSION_DIR
    if session_dir is None or not session_dir.is_dir():
        return None
    directories = [path for path in sorted(session_dir.iterdir())
                   if path.is_dir() and path.name.isdigit()]
    rows = _annotate_summary([summarize_experiment(path) for path in directories])
    payload = {"session_id": session_dir.name,
               "updated_at": datetime.now(timezone.utc).isoformat(),
               "experiments": rows}
    _atomic_text(session_dir / "summary.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = io.StringIO()
    writer = csv.DictWriter(lines, fieldnames=_SUMMARY_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        flat = _flat_summary_row(session_dir.name, row)
        writer.writerow({key: "" if flat.get(key) is None else flat[key] for key in _SUMMARY_COLUMNS})
    csv_path = session_dir / "summary.csv"
    _atomic_text(csv_path, lines.getvalue())
    return csv_path


def publish_session_summary(announce=False):
    try:
        path = write_session_summary()
    except Exception as exc:
        say(f"summary failed  {exc}")
        return
    if announce and path is not None:
        say(f"summary  {path.relative_to(ROOT)}")


def next_experiment_id():
    numbers = []
    if SESSION_DIR is not None and SESSION_DIR.is_dir():
        for path in SESSION_DIR.iterdir():
            if path.is_dir() and path.name.isdigit():
                numbers.append(int(path.name))
    return f"{(max(numbers) if numbers else 0) + 1:04d}"


def inspect_environment():
    """Local CSV audit; only aggregate statistics enter the model context."""
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from eval import eligible_rows
    summary, paths = {}, []
    for variable in ("TRAIN_PATH", "VALIDATION_PATH"):
        value = os.getenv(variable)
        if not value:
            raise ValueError(f"Set {variable} in the environment or repository .env")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{variable} does not point to an existing CSV")
        frame = pd.read_csv(path, low_memory=False)
        cohort, _ = eligible_rows(frame)
        if cohort.empty or cohort.binding_flag.nunique() < 2:
            raise ValueError(f"{variable} requires eligible full-chain rows with both labels")
        summary[variable] = {
            "rows": len(frame), "eligible_full_chain": len(cohort),
            "positive": int(cohort.binding_flag.sum()),
            "negative": int(len(cohort) - cohort.binding_flag.sum()),
            "columns": list(frame.columns),
            "max_lengths": {c: int(cohort[c].str.len().max())
                            for c in ("TCR_full", "Peptide", "HLA_sequence")},
            "sha256": digest(path),
        }
        paths.append(path)
    if paths[0] == paths[1] or summary["TRAIN_PATH"]["sha256"] == summary["VALIDATION_PATH"]["sha256"]:
        raise ValueError("Training and validation must be different datasets")
    return summary, paths


CONTROLLER_FILES = ("ml.py", "final_eval.py", "objective.md", "agent/agent.py", "pyproject.toml", "uv.lock", ".env")


def protected_hashes(data_paths):
    paths = [ROOT / "eval.py", *[ROOT / name for name in CONTROLLER_FILES]]
    return {str(p): digest(p) for p in paths + data_paths if p.is_file()}


def experiment_contract(protocol):
    """Evaluator and dataset hashes. A controller or objective edit can still resume."""
    ignored = {str(ROOT / name) for name in CONTROLLER_FILES}
    return {path: sha for path, sha in protocol.items() if path not in ignored}


def assert_unchanged(expected):
    for path, sha in expected.items():
        if not Path(path).is_file() or digest(path) != sha:
            raise RuntimeError(f"Protected input changed: {Path(path).name}; stopping")


def research_dirs():
    if SESSION_DIR is None or not SESSION_DIR.is_dir():
        return []
    return [path for path in sorted(SESSION_DIR.iterdir())
            if path.is_dir() and (path / "log.jsonl").is_file()]


def prompt_data_summary(summary):
    keys = ("rows", "eligible_full_chain", "positive", "negative", "max_lengths")
    return {name: {key: info[key] for key in keys if key in info} for name, info in summary.items()}


def compact_group(row):
    if not isinstance(row, dict):
        return row
    return {key: row.get(key) for key in ("n", "n_positive", "validation_loss", "auroc", "auprc")}


def compact_groups(groups):
    if not isinstance(groups, dict):
        return groups
    return {name: compact_group(row) for name, row in groups.items()}


def experiment_history(directory):
    """Proposal, epoch losses and summary metrics. Per-peptide tables stay on disk."""
    entry = {"id": directory.name}
    epochs = []
    for record in read_records(directory / "log.jsonl"):
        event = record.get("event")
        if event == "proposal":
            for key in ("parent", "hypothesis", "change", "edits"):
                if key in record:
                    entry[key] = record[key]
        elif event == "training_configuration":
            epochs = []
        elif event == "training_epoch":
            epochs.append({key: record.get(key) for key in ("epoch", "train_loss", "validation_loss")})
        elif event == "repair":
            lines = [line.strip() for line in (record.get("error") or "").splitlines() if line.strip()]
            entry.setdefault("repairs", []).append({
                "change": record.get("change"),
                "error": lines[-1][:240] if lines else None,
            })
        elif event == "model_evaluation":
            metrics = record.get("metrics") or {}
            overall = metrics.get("overall") if isinstance(metrics.get("overall"), dict) else {}
            entry["metrics"] = {
                "overall": {key: overall.get(key) for key in ("validation_loss", "auroc", "auprc")},
                "peptide_macro_auroc": metrics.get("peptide_macro_auroc"),
                "peptide_macro_auprc": metrics.get("peptide_macro_auprc"),
                "peptide_coverage": metrics.get("peptide_coverage"),
                "provided_categories": compact_groups(metrics.get("provided_categories")),
                "derived_categories": compact_groups(metrics.get("derived_categories")),
            }
        elif event == "experiment_failure":
            entry["error"] = record.get("error")
    if epochs:
        entry["epochs"] = epochs
    return entry


def build_context(task, data_summary):
    """Objective, source files the model may patch, and a condensed experiment history."""
    sources = {"starter": STARTER.read_text(encoding="utf-8")}
    history = []
    for directory in research_dirs():
        history.append(experiment_history(directory))
        source = directory / "ml.py"
        if source.is_file():
            sources[directory.name] = source.read_text(encoding="utf-8")
    return json.dumps({
        "task": task,
        "objective": (ROOT / "objective.md").read_text(encoding="utf-8"),
        "sources": sources,
        "history": history,
        "data_summary": prompt_data_summary(data_summary),
        "controller_contract": (
            "Respond only in the requested JSON schema. You have no shell or file tools. "
            "The controller copies the named parent ml.py into a new experiment directory and applies edits. "
            "parent is starter or an existing experiment id. edits is an ordered list of {old, new} strings; "
            "each old must occur exactly once in the parent file; at most 8 edits; an empty list copies the parent unchanged. "
            f"If that run fails because of the code, you can repair the same file up to {MAX_REPAIRS} times. "
            f"After repair {MAX_REPAIRS} the experiment is abandoned. "
            "Do not change evaluator/data/dependencies/controller, access test data, "
            "or make API/network calls from ml.py. "
            "Keep Dataset.data in original CSV order; select eligible full-chain rows with Subset for training. "
            "Validate chain boundaries and use complete untruncated alpha+beta, peptide and HLA sequences. "
            "Maintain Model(), Dataset(path), main(), and forward(peptide,hla,tcr) interfaces. "
            "main() must use the module EXPERIMENT_DIR, set by the controller, and save model.pth there. "
            "Seed Python, numpy and torch; record actual hyperparameters and epoch metrics. "
            "Append aggregate-only training events to RESEARCH_LOG_PATH. Never print/log raw rows, "
            "patient information, sequence tables, secrets, or create training.log/summary/plot files. "
            "CPU/GPU use must respect the operator's environment and scheduler allocation. "
            "Base the next parent choice on all reported metrics and full history, not solely the lowest MSE. "
            "No training occurs during imports. No package installs or subprocess launches in generated ml.py. "
            "Preserve checkpoint saving: initial_model.pth after initialisation, last_model.pth every epoch "
            "and in a finally block, model.pth for the chosen validation checkpoint. Save atomically. "
            "Never delete or overwrite another experiment directory or checkpoint."
        ),
    }, ensure_ascii=False)


class ArgonClient:
    """One research client. Gemini counts tokens then requests JSON; OpenRouter sends one structured chat completion."""
    def __init__(self, args, deadline):
        self.args, self.deadline, self.calls = args, deadline, 0
        self.provider = args.provider
        self.client = None
        self._openrouter_key = None
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(http_options={"timeout": int(args.api_timeout * 1000),
                                                    "retry_options": {"attempts": 1}})
        elif self.provider == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if not key:
                raise RuntimeError("Set OPENROUTER_API_KEY in the environment or repository .env")
            self._openrouter_key = key
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def close(self):
        if self.client is not None:
            self.client.close()

    def _begin_call(self):
        check_stop()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise StopRequested("Session runtime budget exhausted")
        self.calls += 1
        return remaining

    def call(self, method, **kwargs):
        remaining = self._begin_call()
        kwargs.setdefault("config", {})["http_options"] = {
            "timeout": max(1, int(min(self.args.api_timeout, remaining) * 1000))}
        return getattr(self.client.models, method)(model=self.args.model, **kwargs)

    def ask(self, prompt, schema, phase, experiment_id, log_path):
        if self.provider == "openrouter" and not self.args.openrouter_strict:
            prompt += "\n\nReturn one JSON object matching this schema and no other text:\n" + json.dumps(schema)
        tokens, count_method = self._count_input(prompt)
        self._check_input_budget(tokens)
        say(f"  context  {'~' if count_method == 'estimate_utf8_div3' else ''}{tokens}")
        append_log(log_path, "api_request", experiment_id=experiment_id, phase=phase, provider=self.provider,
                   input_tokens=tokens, input_token_count_method=count_method,
                   context_sha256=source_hash(prompt), api_calls_used=self.calls,
                   thinking_level=self.args.thinking_level if self.provider == "gemini" else None,
                   max_output_tokens=self.args.max_output_tokens)
        text, finishes, token_usage = self._generate(prompt, schema, phase)
        append_log(log_path, "api_response", experiment_id=experiment_id, phase=phase, provider=self.provider,
                   finish_reasons=finishes, token_usage=token_usage)
        who = "Argon" if self.provider == "gemini" else "OpenRouter"
        if self._output_truncated(finishes):
            raise ValueError(f"{who} reached its thinking/output cap; no partial model code will be applied")
        if not text:
            raise ValueError(f"{who} returned no JSON text")
        if self.provider == "openrouter":
            text = self._unwrap_json(text)
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError(f"{who} response must be a JSON object")
        return result

    def _count_input(self, prompt):
        if self.provider == "openrouter":
            return max(1, (len(prompt.encode("utf-8")) + 2) // 3), "estimate_utf8_div3"
        return self.call("count_tokens", contents=prompt).total_tokens, "gemini_count_tokens"

    def _check_input_budget(self, tokens):
        over_cap = tokens is None or tokens + 2048 > self.args.max_input_tokens
        over_window = (self.provider == "gemini" and
                       (tokens is None or tokens + 2048 + self.args.max_output_tokens > 1048576))
        if over_cap or over_window:
            raise RuntimeError("Full history exceeds input-token budget; no truncation performed")

    def _generate(self, prompt, schema, phase):
        if self.provider == "openrouter":
            return self._generate_openrouter(prompt, schema, phase)
        return self._generate_gemini(prompt, schema)

    def _generate_gemini(self, prompt, schema):
        response = self.call("generate_content", contents=prompt, config={
            "response_mime_type": "application/json", "response_json_schema": schema,
            "max_output_tokens": self.args.max_output_tokens,
            "thinking_config": {"thinking_level": self.args.thinking_level},
        })
        usage = getattr(response, "usage_metadata", None)
        finishes = [str(getattr(candidate, "finish_reason", ""))
                    for candidate in (getattr(response, "candidates", None) or [])]
        token_usage = {key: getattr(usage, key, None) for key in
                       ("prompt_token_count", "candidates_token_count",
                        "thoughts_token_count", "total_token_count")}
        return response.text, finishes, token_usage

    def _openrouter_payload(self, prompt, schema, phase):
        if self.args.openrouter_strict:
            wired = dict(schema)
            wired["additionalProperties"] = False
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "proposal" if phase == "proposal" else "decision",
                    "strict": True,
                    "schema": wired,
                },
            }
        else:
            response_format = {"type": "json_object"}
        return {
            "model": self.args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.args.max_output_tokens,
            "response_format": response_format,
            "provider": {"require_parameters": True},
            "plugins": [{"id": "context-compression", "enabled": False}],
        }

    def _generate_openrouter(self, prompt, schema, phase):
        remaining = self._begin_call()
        timeout = max(1.0, min(self.args.api_timeout, remaining))
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(self._openrouter_payload(prompt, schema, phase), ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._openrouter_key}",
                "Content-Type": "application/json",
                "X-Title": "auto-research-protein-binding",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ValueError(f"OpenRouter HTTP {exc.code}: {detail}") from None
        except TimeoutError:
            raise ValueError("OpenRouter request timed out") from None
        except urllib.error.URLError as exc:
            raise ValueError(f"OpenRouter request failed: {exc.reason}") from None
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError("OpenRouter returned no JSON text") from None
        return self._openrouter_result(body)

    def _openrouter_result(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("OpenRouter response must be a JSON object")
        error = payload.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else error
            raise ValueError(f"OpenRouter error: {message}")
        choices = payload.get("choices") or []
        usage = self._openrouter_usage(payload)
        if not choices or not isinstance(choices[0], dict):
            return "", [], usage
        choice = choices[0]
        finish = choice.get("finish_reason")
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        return self._message_text(message.get("content")), ["" if finish is None else str(finish)], usage

    @staticmethod
    def _openrouter_usage(payload):
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        details = usage.get("completion_tokens_details")
        details = details if isinstance(details, dict) else {}
        return {
            "prompt_token_count": usage.get("prompt_tokens"),
            "candidates_token_count": usage.get("completion_tokens"),
            "thoughts_token_count": details.get("reasoning_tokens"),
            "total_token_count": usage.get("total_tokens"),
        }

    @staticmethod
    def _unwrap_json(text):
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
        return stripped

    @staticmethod
    def _message_text(content):
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)

    @staticmethod
    def _output_truncated(finishes):
        return any("MAX_TOKENS" in reason or reason.lower() == "length" for reason in finishes)


REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "change": {"type": "string"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["old", "new"]}},
    },
    "required": ["change", "edits"],
}


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "parent": {"type": "string"},
        "hypothesis": {"type": "string"},
        "evidence": {"type": "string"},
        "change": {"type": "string"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["old", "new"]}},
    },
    "required": ["parent", "hypothesis", "evidence", "change", "edits"],
}


def require_strings(value, keys):
    for key in keys:
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"Missing/non-empty string required: {key}")


def parent_source(parent):
    if parent == "starter":
        return STARTER.read_text(encoding="utf-8")
    path = SESSION_DIR / parent / "ml.py"
    if not path.is_file():
        raise ValueError(f"Unknown parent {parent}; use starter or an existing experiment id")
    return path.read_text(encoding="utf-8")


def apply_edits(source, edits):
    if not isinstance(edits, list):
        raise ValueError("edits must be a list")
    if len(edits) > MAX_EDITS:
        raise ValueError(f"At most {MAX_EDITS} edits")
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict) or not isinstance(edit.get("old"), str) or not isinstance(edit.get("new"), str):
            raise ValueError(f"Edit {index} must have old and new strings")
        if source.count(edit["old"]) != 1:
            raise ValueError(f"Edit {index}: old must occur exactly once")
        source = source.replace(edit["old"], edit["new"], 1)
    return source


def write_experiment_source(directory, source):
    tree = ast.parse(source, filename="ml.py")
    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not {"Model", "Dataset"} <= classes or "main" not in functions:
        raise ValueError("ml.py must define Model, Dataset and main")
    compile(tree, "ml.py", "exec")
    (directory / "ml.py").write_text(source, encoding="utf-8")


def propose_experiment(client, data_summary, experiment_id, log_path):
    say("  planning the next change")
    available = ["starter", *[path.name for path in experiment_dirs()]]
    if len(available) == 1:
        task = ("Establish a valid baseline from the starter ml.py. Set parent to starter. "
                "Use an empty edits list unless you must correct full-chain filtering/padding, "
                "seeding, shared logging or output-interface issues.")
    else:
        task = ("Propose a scientifically motivated change based on the complete history. "
                "Set parent to starter or an existing experiment id. Return search-replace edits.")
    task += f" Experiment ID: {experiment_id}. Available parents: {available}."
    proposal = client.ask(build_context(task, data_summary), PROPOSAL_SCHEMA,
                          "proposal", experiment_id, log_path)
    require_strings(proposal, ["parent", "hypothesis", "evidence", "change"])
    proposal["parent"] = proposal["parent"].strip()
    return proposal


def materialize_experiment(directory, proposal):
    source = apply_edits(parent_source(proposal["parent"]), proposal.get("edits"))
    write_experiment_source(directory, source)
    return source


def propose_repair(client, experiment_id, log_path, source, error, attempt):
    task = (
        f"Repair {attempt} of {MAX_REPAIRS}. "
        "The current ml.py failed before a valid evaluation. "
        "Return search-replace edits that fix this failure and keep the experiment's intended change. "
        "A failed repair uses one of the remaining attempts, and the experiment is abandoned after the last one.\n\n"
        f"Error:\n{error}\n\n"
        f"ml.py:\n{source}"
    )
    repair = client.ask(task, REPAIR_SCHEMA, "repair", experiment_id, log_path)
    require_strings(repair, ["change"])
    if not repair.get("edits"):
        raise ValueError("Repair returned no edits")
    return repair


def run_command(command, experiment_id, stage, env, timeout, log_path):
    """Fixed argv; poll for stop requests, then allow checkpoint saving before kill."""
    say(f"  {stage}")
    start = time.monotonic()
    announced = start
    code = None
    err_path = log_path.with_name(f".{stage}.err")
    err_handle = err_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stderr=err_handle,
                               start_new_session=(os.name == "posix"))
    try:
        try:
            while True:
                check_stop()
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = process.wait(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - announced >= 15:
                        say(f"  {stage} · {time.monotonic() - start:.0f}s")
                        announced = time.monotonic()
                    continue
        except BaseException:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGINT)
                else:
                    process.terminate()
                process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
            append_log(log_path, "execution", experiment_id=experiment_id, stage=stage,
                       status="interrupted_or_timeout", runtime_seconds=time.monotonic()-start)
            raise
        append_log(log_path, "execution", experiment_id=experiment_id, stage=stage,
                   returncode=code, runtime_seconds=time.monotonic()-start)
        if code:
            err_handle.flush()
            detail = err_path.read_text(encoding="utf-8", errors="replace").strip()
            tail = detail[-4000:] if detail else ""
            raise RuntimeError(f"{stage} exited with code {code}" + (f"\n{tail}" if tail else ""))
    finally:
        err_handle.close()
        err_path.unlink(missing_ok=True)


def run_training(experiment_id, directory, args, deadline, log_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               EXPERIMENT_DIR=str(directory), RESEARCH_LOG_PATH=str(log_path),
               RESEARCH_EXPERIMENT_ID=experiment_id)
    launcher = ("import os, sys\n"
                "sys.path.insert(0, os.environ['EXPERIMENT_DIR'])\n"
                "import ml\n"
                "ml.EXPERIMENT_DIR = os.environ['EXPERIMENT_DIR']\n"
                "ml.main()\n")
    try:
        run_command([sys.executable, "-B", "-c", launcher], experiment_id, "training", env,
                    max(0.01, min(args.run_timeout, deadline-time.monotonic())), log_path)
    finally:
        append_log(log_path, "checkpoint_inventory", experiment_id=experiment_id,
                   experiment_dir=str(directory),
                   checkpoints={p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                                for p in directory.glob("*.pth")})
    if not (directory / "model.pth").is_file():
        raise RuntimeError("Training did not produce model.pth in its assigned directory")


def run_evaluation(experiment_id, directory, proposal, args, deadline, log_path):
    argv = ["--experiment_dir", str(directory), "--experiment-id", experiment_id,
            "--log-path", str(log_path), "--hypothesis", proposal["hypothesis"],
            "--change", proposal["change"]]
    if args.category_column:
        argv += ["--category-column", args.category_column]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               EXPERIMENT_DIR=str(directory), EVAL_ARGV=json.dumps(argv))
    launcher = ("import json, os, sys\n"
                "sys.path.insert(0, os.environ['EXPERIMENT_DIR'])\n"
                "sys.argv = ['eval.py'] + json.loads(os.environ['EVAL_ARGV'])\n"
                "import eval as E\n"
                "E.main()\n")
    run_command([sys.executable, "-B", "-c", launcher], experiment_id, "validation", env,
                max(0.01, min(args.run_timeout, deadline-time.monotonic())), log_path)
    records = [r for r in read_records(log_path) if r.get("event") == "model_evaluation"
               and r.get("experiment_id") == experiment_id]
    record = records[-1] if records else None
    if record is None or record.get("status") != "valid":
        raise RuntimeError("Expected a valid fixed-evaluator report")
    if record.get("split") != "validation" or record.get("ml_sha256") != digest(directory / "ml.py"):
        raise RuntimeError("Evaluation split/source identity mismatch")
    return record


def open_session(args, protocol):
    """A fresh launch creates a new trace. Resume reopens one existing trace."""
    global SESSION_DIR
    if args.resume or args.session:
        if args.session:
            directory = EXPERIMENTS / args.session
            if not (directory / "controller.jsonl").is_file():
                raise ValueError(f"No session {args.session}")
        else:
            found = session_directories()
            if not found:
                raise ValueError("No session to resume")
            directory = max(found, key=lambda path: (path / "controller.jsonl").stat().st_mtime)
        records = read_records(directory / "controller.jsonl")
        if any(r.get("event") == "research_finalized" for r in records):
            raise ValueError("This session is frozen for final test evaluation; autonomous optimisation cannot resume")
        starts = [r for r in records if r.get("event") == "session_start"]
        stored = starts[-1].get("protocol") or {} if starts else {}
        if starts and experiment_contract(stored) != experiment_contract(protocol):
            raise ValueError("Evaluator or dataset changed; cannot resume this session")
        SESSION_DIR = directory
        return directory, True
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    directory = EXPERIMENTS / f"{stamp}-{uuid.uuid4().hex[:4]}"
    directory.mkdir(parents=True)
    SESSION_DIR = directory
    return directory, False


def run_experiment(client, directory, proposal, args, deadline, log_path, protected):
    """Train and evaluate. A code failure gets up to MAX_REPAIRS edits of this same file."""
    experiment_id = directory.name
    last_error = None
    for attempt in range(MAX_REPAIRS + 1):
        try:
            if attempt:
                say(f"  repair  {attempt}/{MAX_REPAIRS}")
                source = ((directory / "ml.py").read_text(encoding="utf-8")
                          if (directory / "ml.py").is_file() else parent_source(proposal["parent"]))
                repair = propose_repair(client, experiment_id, log_path, source, last_error, attempt)
                show_proposal({"parent": experiment_id, "change": repair["change"], "edits": repair.get("edits")})
                source = apply_edits(source, repair.get("edits"))
                write_experiment_source(directory, source)
                append_log(log_path, "repair", experiment_id=experiment_id, attempt=attempt,
                           error=last_error, ml_sha256=source_hash(source), **repair)
            else:
                source = materialize_experiment(directory, proposal)
                append_log(log_path, "proposal", experiment_id=experiment_id,
                           ml_sha256=source_hash(source), **proposal)
            run_training(experiment_id, directory, args, deadline, log_path)
            assert_unchanged(protected)
            record = run_evaluation(experiment_id, directory, proposal, args, deadline, log_path)
            assert_unchanged(protected)
            return record
        except StopRequested:
            raise
        except Exception as exc:
            if str(exc).startswith("Protected input changed"):
                raise
            last_error = str(exc)
            if len(last_error) > 4000:
                last_error = last_error[-4000:]
            lines = last_error.splitlines() or [last_error]
            say(f"  failed  {lines[0]}")
            for line in lines[1:40]:
                say(f"    {line}")
    append_log(log_path, "experiment_failure", experiment_id=experiment_id, error=last_error)
    return None


def _run_agent_loop(args, client_factory=ArgonClient):
    global SESSION_STARTED_AT
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    data_summary, data_paths = inspect_environment()
    protected = protected_hashes(data_paths)
    if args.dry_run:
        say(json.dumps({"data_summary": data_summary, "provider": args.provider, "model": args.model,
                        "max_experiments": args.max_experiments, "api_called": False}, indent=2))
        return
    session_dir, resumed = open_session(args, protected)
    deadline = time.monotonic() + args.session_timeout
    session_id = session_dir.name
    SESSION_STARTED_AT = datetime.now(timezone.utc).isoformat()
    ACTIVE_PATH.write_text(session_id, encoding="utf-8")
    append_log(controller_log(), "session_start", session_id=session_id, protocol=protected,
               resumed=resumed, budgets=vars(args), data_summary=data_summary)
    say(f"session  {session_dir.relative_to(ROOT)}")
    say(f"model    {args.provider} · {args.model}")
    say(f"budget   {args.max_experiments} experiments · {'resuming' if resumed else 'new trace'}")
    publish_session_summary(announce=True)
    client = None
    latest = None
    try:
        client = client_factory(args, deadline)
        for _ in range(args.max_experiments):
            check_stop()
            if time.monotonic() >= deadline:
                raise StopRequested("Session runtime budget exhausted")
            assert_unchanged(protected)
            experiment_id = next_experiment_id()
            directory = session_dir / experiment_id
            directory.mkdir(parents=True, exist_ok=False)
            log_path = directory / "log.jsonl"
            say("")
            say(f"experiment {experiment_id}")
            try:
                proposal = propose_experiment(client, data_summary, experiment_id, log_path)
                show_proposal(proposal)
                record = run_experiment(client, directory, proposal, args, deadline, log_path, protected)
                if record:
                    latest = directory
                    show_result(experiment_id, record)
                else:
                    show_result(experiment_id, None)
            except StopRequested:
                raise
            except Exception as exc:
                append_log(log_path, "experiment_failure", experiment_id=experiment_id,
                           error=f"{type(exc).__name__}: {exc}")
                assert_unchanged(protected)
                say(f"  failed  {exc}")
                show_result(experiment_id, None)
            finally:
                publish_session_summary()
    except BaseException as exc:
        append_log(controller_log(), "session_stopped", session_id=session_id,
                   error=f"{type(exc).__name__}: {exc}", api_calls_used=client.calls if client else 0)
        if isinstance(exc, StopRequested):
            say(f"stopped  {exc}")
            if latest:
                say(f"latest   {latest.relative_to(ROOT)}/model.pth")
        else:
            raise
    else:
        append_log(controller_log(), "session_complete", session_id=session_id,
                   api_calls_used=client.calls)
        if latest:
            say(f"latest   {latest.relative_to(ROOT)}/model.pth")
    finally:
        close = getattr(client, "close", None)
        if close:
            close()
        if ACTIVE_PATH.is_file() and ACTIVE_PATH.read_text(encoding="utf-8").strip() == session_id:
            ACTIVE_PATH.unlink()
        SESSION_STARTED_AT = None


def run_agent_loop(args, client_factory=ArgonClient):
    if args.dry_run:
        return _run_agent_loop(args, client_factory)
    import fcntl
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller holds the research lock") from None
        return _run_agent_loop(args, client_factory)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-stop", action="store_true",
                        help="Append a manual stop request for the active controller; no API/data access")
    parser.add_argument("--provider", choices=("gemini", "openrouter"), default="gemini")
    parser.add_argument("--model", default="gemini-4-argon",
                        help="Gemini model id, or vendor/name when --provider openrouter")
    parser.add_argument("--openrouter-strict", action=argparse.BooleanOptionalAction, default=False,
                        help="Require an OpenRouter endpoint that enforces a JSON schema. Default sends JSON-object mode so smaller models can be routed")
    parser.add_argument("--max-experiments", type=int, default=2,
                        help="Total attempts this invocation, including initial baseline setup")
    parser.add_argument("--run-timeout", type=float, default=900,
                        help="Seconds per training/evaluation subprocess")
    parser.add_argument("--session-timeout", type=float, default=3600)
    parser.add_argument("--api-timeout", type=float, default=600)
    parser.add_argument("--thinking-level", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--max-input-tokens", type=int, default=900000,
                        help="Full-prompt input cap, including safety margin; never silently truncated")
    parser.add_argument("--max-output-tokens", type=int, default=65536,
                        help="Combined thinking and answer cap; synchronous maximum is 65536")
    parser.add_argument("--category-column")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the latest session, or the session named by --session")
    parser.add_argument("--session", help="Session directory name under experiments/ to resume")
    parser.add_argument("--dry-run", action="store_true", help="Local data preflight only; no API/training")
    args = parser.parse_args(argv)
    for name in ("max_experiments", "run_timeout", "session_timeout", "api_timeout",
                 "max_input_tokens", "max_output_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_output_tokens > 65536:
        parser.error("Synchronous controller supports up to 65536 thinking/output tokens; longer decoding requires the background API")
    if args.max_input_tokens + args.max_output_tokens > 1048576:
        parser.error("Input and thinking/output budgets must fit the shared 1048576-token context")
    if args.provider == "openrouter" and "/" not in args.model:
        parser.error("Pass an OpenRouter model id like vendor/name")
    return args


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt("Received SIGTERM")
    signal.signal(signal.SIGTERM, terminate)
    try:
        args = parse_args()
        if args.request_stop:
            if not ACTIVE_PATH.is_file():
                print("No active session to stop.", file=sys.stderr)
                return 1
            session_dir = EXPERIMENTS / ACTIVE_PATH.read_text(encoding="utf-8").strip()
            append_log(session_dir / "controller.jsonl", "stop_requested", reason="Operator CLI request")
            print(f"Stop requested for {session_dir.name}; it will stop at its next checkpoint (or after its current API request).")
            return 0
        run_agent_loop(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Controller stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
