"""Bounded Argon research controller; run from the repository root.

    python agent/agent.py --dry-run
    python agent/agent.py --max-experiments 2 --run-timeout 900
    python agent/agent.py --resume --max-experiments 2

Two experiments means baseline setup/training plus ONE proposed modification.
Only this controller writes model proposals to ml.py; all history stays in
experiments/evaluation.log. Checkpoints are generated training outputs.
This is an orchestration boundary, not an OS sandbox for generated Python.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "experiments/evaluation.log"


def append_log(event, **fields):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    return record


def read_full_history():
    """Never summarise, select recent entries or silently truncate this text."""
    return LOG.read_text(encoding="utf-8") if LOG.exists() else ""


def read_records():
    records = []
    for line in read_full_history().splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        except json.JSONDecodeError:
            pass  # Human-written log lines still enter every API prompt.
    return records


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


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
    # Test data is neither opened nor included in the API context.
    return summary, paths


def protected_hashes(data_paths):
    paths = [ROOT / name for name in ("eval.py", "objective.md", "agent/agent.py",
                                      "pyproject.toml", "uv.lock", ".env")]
    return {str(p): digest(p) for p in paths + data_paths if p.is_file()}


def assert_unchanged(expected):
    for path, sha in expected.items():
        if not Path(path).is_file() or digest(path) != sha:
            raise RuntimeError(f"Protected input changed: {Path(path).name}; stopping")


def build_context(task, data_summary, incumbent):
    """Fresh request, including objective, fixed evaluator, source and ENTIRE log."""
    return json.dumps({
        "task": task,
        "objective": (ROOT / "objective.md").read_text(encoding="utf-8"),
        "fixed_evaluator": (ROOT / "eval.py").read_text(encoding="utf-8"),
        "current_ml_py": (ROOT / "ml.py").read_text(encoding="utf-8"),
        "data_summary_only": data_summary,
        "incumbent": incumbent,
        "full_untruncated_research_log": read_full_history(),
        "controller_contract": (
            "Respond only in the requested JSON schema. You have no shell or file tools. "
            "The controller applies only the ml_py string to ml.py. Do not generate other source files, "
            "change evaluator/data/dependencies/controller, access test data, or make API/network calls from ml.py. "
            "Keep Dataset.data in original CSV order; select eligible full-chain rows with Subset for training. "
            "Validate chain boundaries and use complete untruncated alpha+beta, peptide and HLA sequences. "
            "Maintain Model(), Dataset(path), main(), and forward(peptide,hla,tcr) interfaces. "
            "main() must use the module EXPERIMENT_DIR, set by the controller, and save model.pth there. "
            "Seed Python, numpy and torch from RESEARCH_SEED; record actual hyperparameters and epoch metrics. "
            "Append aggregate-only training events to RESEARCH_LOG_PATH. Never print/log raw rows, "
            "patient information, sequence tables, secrets, or create training.log/summary/plot files. "
            "CPU/GPU use must respect the operator's environment and scheduler allocation. "
            "Base scientific decisions on all reported metrics and full history, not solely the lowest MSE. "
            "For investigate/discard/failed, the controller restores the incumbent; keep promotes the candidate. "
            "Investigate means record the uncertainty and propose a controlled follow-up next. "
            "No training occurs during imports. No package installs or subprocess launches in generated ml.py."
        ),
    }, ensure_ascii=False)


class ArgonClient:
    """Count context first, then request structured JSON, with a hard API-call cap."""
    def __init__(self, args, deadline):
        from google import genai
        self.client = genai.Client(http_options={"timeout": int(args.api_timeout * 1000),
                                                "retry_options": {"attempts": 1}})
        self.args, self.deadline, self.calls = args, deadline, 0

    def close(self):
        self.client.close()

    def call(self, method, **kwargs):
        if self.calls >= self.args.max_api_calls:
            raise RuntimeError("API-call budget exhausted")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Session runtime budget exhausted")
        # Request-level timeout is also bounded by the remaining session time.
        kwargs.setdefault("config", {})["http_options"] = {
            "timeout": max(1, int(min(self.args.api_timeout, remaining) * 1000))}
        self.calls += 1
        return getattr(self.client.models, method)(model=self.args.model, **kwargs)

    def ask(self, prompt, schema, phase, experiment_id):
        tokens = self.call("count_tokens", contents=prompt).total_tokens
        if tokens is None or tokens + 2048 > self.args.max_input_tokens:
            raise RuntimeError("Full history exceeds input-token budget; no truncation performed")
        append_log("api_request", experiment_id=experiment_id, phase=phase,
                   input_tokens=tokens, context_sha256=source_hash(prompt),
                   full_history_included=True, api_calls_used=self.calls)
        response = self.call("generate_content", contents=prompt, config={
            "response_mime_type": "application/json", "response_json_schema": schema,
            "max_output_tokens": self.args.max_output_tokens,
        })
        if not response.text:
            raise ValueError("Argon returned no JSON text")
        result = json.loads(response.text)
        if not isinstance(result, dict):
            raise ValueError("Argon response must be a JSON object")
        return result


PROPOSAL_SCHEMA = {"type": "object", "properties": {
    key: {"type": "string"} for key in
    ("hypothesis", "evidence", "change", "training_configuration", "selection_rationale", "ml_py")},
    "required": ["hypothesis", "evidence", "change", "training_configuration", "selection_rationale", "ml_py"]}
DECISION_SCHEMA = {"type": "object", "properties": {
    "decision": {"type": "string", "enum": ["keep", "discard", "investigate", "failed"]},
    **{k: {"type": "string"} for k in ("reason", "limitations", "next_experiment")}},
    "required": ["decision", "reason", "limitations", "next_experiment"]}


def require_strings(value, keys):
    for key in keys:
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"Missing/non-empty string required: {key}")


def propose_experiment(client, data_summary, incumbent, experiment_id):
    task = ("Establish a valid baseline: preserve the starter model architecture and training loss; "
            "only correct full-chain filtering/padding, seeding, shared logging and output-interface issues."
            if incumbent is None else
            "Propose ONE scientifically motivated model/training change based on the complete history.")
    task += f" Experiment ID: {experiment_id}. Return the entire replacement ml.py, not a diff."
    proposal = client.ask(build_context(task, data_summary, incumbent), PROPOSAL_SCHEMA,
                          "proposal", experiment_id)
    require_strings(proposal, PROPOSAL_SCHEMA["required"])
    return proposal


def apply_model_change(source):
    """Validate without executing generated code; write exactly one source path."""
    tree = ast.parse(source, filename="ml.py")
    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not {"Model", "Dataset"} <= classes or "main" not in functions:
        raise ValueError("ml.py must define Model, Dataset and main")
    compile(tree, "ml.py", "exec")
    (ROOT / "ml.py").write_text(source, encoding="utf-8")


def run_command(command, experiment_id, stage, env, timeout):
    """Fixed argv only, no shell. Kill the process group on timeout/interruption."""
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=(os.name == "posix"))
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()
        append_log("execution", experiment_id=experiment_id, stage=stage,
                   status="interrupted_or_timeout", runtime_seconds=time.monotonic()-start)
        raise
    append_log("execution", experiment_id=experiment_id, stage=stage,
               returncode=code, runtime_seconds=time.monotonic()-start)
    if code:
        raise RuntimeError(f"{stage} exited with code {code}; see console and shared log")


def run_training(experiment_id, directory, args, deadline):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               EXPERIMENT_DIR=str(directory), RESEARCH_LOG_PATH=str(LOG),
               RESEARCH_EXPERIMENT_ID=experiment_id, RESEARCH_SEED=str(args.seed))
    directory.mkdir(parents=True, exist_ok=False)
    # Import in a fresh process and set the existing module variable explicitly.
    launcher = "import os, ml; ml.EXPERIMENT_DIR=os.environ['EXPERIMENT_DIR']; ml.main()"
    run_command([sys.executable, "-B", "-c", launcher], experiment_id, "training", env,
                max(0.01, min(args.run_timeout, deadline-time.monotonic())))
    if not (directory / "model.pth").is_file():
        raise RuntimeError("Training did not produce model.pth in its assigned directory")


def run_evaluation(experiment_id, directory, proposal, args, deadline):
    command = [sys.executable, "-B", str(ROOT / "eval.py"), "--experiment_dir", str(directory),
               "--experiment-id", experiment_id, "--log-path", str(LOG),
               "--hypothesis", proposal["hypothesis"], "--change", proposal["change"]]
    if args.category_column:
        command += ["--category-column", args.category_column]
    run_command(command, experiment_id, "validation", dict(os.environ),
                max(0.01, min(args.run_timeout, deadline-time.monotonic())))
    records = [r for r in read_records() if r.get("event") == "model_evaluation"
               and r.get("experiment_id") == experiment_id]
    if len(records) != 1 or records[0].get("status") != "valid":
        raise RuntimeError("Expected exactly one valid fixed-evaluator report")
    if records[0].get("split") != "validation" or records[0].get("ml_sha256") != digest(ROOT / "ml.py"):
        raise RuntimeError("Evaluation split/source identity mismatch")
    return records[0]


def decide_next_action(client, data_summary, incumbent, experiment_id, valid):
    task = (f"Interpret experiment {experiment_id}. Execution_valid={valid}. "
            "Compare full validation evidence with the incumbent and justify keep/discard/investigate/failed. "
            "A failed execution cannot be kept. For a first valid baseline, keep it unless scientifically invalid. "
            "Do not propose source changes in this response; provide a specific next experiment.")
    decision = client.ask(build_context(task, data_summary, incumbent), DECISION_SCHEMA,
                          "interpretation", experiment_id)
    require_strings(decision, DECISION_SCHEMA["required"])
    if decision["decision"] not in {"keep", "discard", "investigate", "failed"}:
        raise ValueError("Unknown decision")
    if not valid and decision["decision"] != "failed":
        raise ValueError("Failed execution must receive a failed decision")
    return decision


def restore_or_keep(decision, source, previous_source, incumbent, experiment_id, directory):
    """Promote explicitly; all other states restore the incumbent source."""
    if decision["decision"] == "keep":
        incumbent = {"experiment_id": experiment_id, "experiment_dir": str(directory),
                     "ml_sha256": source_hash(source),
                     "checkpoint_sha256": digest(directory / "model.pth")}
        previous_source = source
    (ROOT / "ml.py").write_text(previous_source, encoding="utf-8")
    append_log("selection", experiment_id=experiment_id, incumbent=incumbent,
               restored_source_sha256=source_hash(previous_source))
    return previous_source, incumbent


def recover_selection(resume, protocol):
    """Recover source from the single ledger; never create revision files."""
    records = read_records()
    sessions = [r for r in records if r.get("event") == "session_start"]
    if sessions and not resume:
        raise ValueError("History exists: use --resume to continue it")
    if resume and not sessions:
        raise ValueError("No controller session to resume")
    if not sessions:
        return (ROOT / "ml.py").read_text(encoding="utf-8"), None
    last = sessions[-1]
    if last.get("protocol") != protocol:
        raise ValueError("Data/evaluator/objective/controller changed; cannot resume this protocol")
    selections = [r for r in records if r.get("event") == "selection"]
    selected = selections[-1].get("incumbent") if selections else None
    sources = {r["experiment_id"]: r["ml_py"] for r in records
               if r.get("event") == "proposal" and "ml_py" in r}
    source = sources[selected["experiment_id"]] if selected else last["initial_ml_py"]
    if selected and (source_hash(source) != selected["ml_sha256"] or
                     not (Path(selected["experiment_dir"]) / "model.pth").is_file()):
        raise ValueError("Selected source or checkpoint unavailable")
    if selected and digest(Path(selected["experiment_dir"]) / "model.pth") != selected["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash mismatch")
    (ROOT / "ml.py").write_text(source, encoding="utf-8")
    return source, selected


def _run_agent_loop(args, client_factory=ArgonClient):
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    data_summary, data_paths = inspect_environment()
    protected = protected_hashes(data_paths)
    if args.dry_run:
        print(json.dumps({"data_summary": data_summary, "model": args.model,
                          "max_experiments": args.max_experiments,
                          "log_path": str(LOG), "api_called": False}, indent=2))
        return
    previous_source, incumbent = recover_selection(args.resume, protected)
    deadline = time.monotonic() + args.session_timeout
    session_id = uuid.uuid4().hex[:12]
    append_log("session_start", session_id=session_id, protocol=protected,
               initial_ml_py=previous_source, incumbent=incumbent,
               budgets=vars(args), data_summary=data_summary)
    client = None
    try:
        client = client_factory(args, deadline)
        for index in range(args.max_experiments):
            if time.monotonic() >= deadline:
                raise RuntimeError("Session runtime budget exhausted")
            # Reserve count+generate for BOTH proposal and interpretation.
            if client.calls + 4 > args.max_api_calls:
                raise RuntimeError("Insufficient API budget for a complete experiment")
            assert_unchanged(protected)
            experiment_id = f"{session_id}_{index+1:04d}"
            directory = ROOT / "experiments" / experiment_id
            proposal = propose_experiment(client, data_summary, incumbent, experiment_id)
            append_log("proposal", experiment_id=experiment_id, parent=incumbent,
                       seed=args.seed, **proposal)
            source, valid = proposal["ml_py"], False
            try:
                apply_model_change(source)
                run_training(experiment_id, directory, args, deadline)
                assert_unchanged(protected)
                run_evaluation(experiment_id, directory, proposal, args, deadline)
                assert_unchanged(protected)
                valid = True
            except Exception as exc:
                append_log("experiment_failure", experiment_id=experiment_id,
                           error=f"{type(exc).__name__}: {exc}")
                assert_unchanged(protected)
            decision = decide_next_action(client, data_summary, incumbent, experiment_id, valid)
            append_log("agent_decision", experiment_id=experiment_id, **decision)
            previous_source, incumbent = restore_or_keep(
                decision, source, previous_source, incumbent, experiment_id, directory)
            print(f"{experiment_id}: {decision['decision']} — {decision['reason']}", flush=True)
        if incumbent is None:
            raise RuntimeError("No valid model was selected; inspect the shared log")
    except BaseException as exc:
        (ROOT / "ml.py").write_text(previous_source, encoding="utf-8")
        append_log("session_stopped", session_id=session_id, incumbent=incumbent,
                   error=f"{type(exc).__name__}: {exc}", api_calls_used=client.calls if client else 0)
        raise
    else:
        append_log("session_complete", session_id=session_id, incumbent=incumbent,
                   api_calls_used=client.calls)
        print(f"Selected checkpoint: {incumbent['experiment_dir']}/model.pth")
    finally:
        close = getattr(client, "close", None)
        if close:
            close()


def run_agent_loop(args, client_factory=ArgonClient):
    if args.dry_run:
        return _run_agent_loop(args, client_factory)
    # Cooperating controllers share the same log lock; eval.py can still append.
    import fcntl
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller holds the shared research log") from None
        return _run_agent_loop(args, client_factory)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-4-argon")
    parser.add_argument("--max-experiments", type=int, default=2,
                        help="Total attempts this invocation, including initial baseline setup")
    parser.add_argument("--run-timeout", type=float, default=900,
                        help="Seconds per training/evaluation subprocess")
    parser.add_argument("--session-timeout", type=float, default=3600)
    parser.add_argument("--api-timeout", type=float, default=180)
    parser.add_argument("--max-api-calls", type=int, default=8,
                        help="Includes count_tokens; four calls per complete experiment")
    parser.add_argument("--max-input-tokens", type=int, default=100000)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--category-column")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Local data preflight only; no API/training")
    args = parser.parse_args(argv)
    for name in ("max_experiments", "run_timeout", "session_timeout", "api_timeout",
                 "max_api_calls", "max_input_tokens", "max_output_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main():
    try:
        run_agent_loop(parse_args())
    except KeyboardInterrupt:
        print("Interrupted; incumbent ml.py restored.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Controller stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
