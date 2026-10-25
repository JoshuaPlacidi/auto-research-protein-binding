const title = document.querySelector("#title");
const meta = document.querySelector("#meta");
const conclusion = document.querySelector("#conclusion");
const improvements = document.querySelector("#improvements");
const progress = document.querySelector("#progress");
const prev = document.querySelector("#prev");
const next = document.querySelector("#next");

const state = {
  session: null,
  rows: [],
  index: 0,
};

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function fmtDelta(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(3)}`;
}

function experimentTitle(experiment) {
  return experiment.title || experiment.change || "Untitled experiment";
}

function diffClass(line) {
  if (line.startsWith("+++") || line.startsWith("---")) return "file";
  if (line.startsWith("@@")) return "hunk";
  if (line.startsWith("+")) return "add";
  if (line.startsWith("-")) return "del";
  return "ctx";
}

function renderDiff(text) {
  if (!text) return `<p class="placeholder">No recorded code change.</p>`;
  const lines = text.split("\n").map((line) => (
    `<span class="${diffClass(line)}">${esc(line) || " "}</span>`
  ));
  return `<pre class="diff">${lines.join("\n")}</pre>`;
}

function render() {
  const total = state.rows.length;
  prev.disabled = state.index <= 0;
  next.disabled = state.index >= total - 1;
  progress.textContent = total ? `${state.index + 1} / ${total}` : "No AUROC improvements";
  improvements.replaceChildren();
  state.rows.forEach((experiment, index) => {
    const item = document.createElement("li");
    const open = index === state.index;
    item.className = open ? "is-open" : "";
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<span>${esc(experiment.id)} · ${esc(experimentTitle(experiment))}</span><span class="score">AUROC ${esc(fmt(experiment.validation_auroc))} (${esc(fmtDelta(experiment.auroc_vs_parent))})</span>`;
    button.addEventListener("click", () => {
      state.index = index;
      render();
    });
    item.append(button);
    if (open) {
      const panel = document.createElement("div");
      panel.className = "expand";
      panel.innerHTML = `
        <p class="meta">AUROC ${esc(fmt(experiment.validation_auroc))} (${esc(fmtDelta(experiment.auroc_vs_parent))}) · AUPRC ${esc(fmt(experiment.validation_auprc))} (${esc(fmtDelta(experiment.auprc_vs_parent))}) · parent ${esc(experiment.parent || "—")}</p>
        <h3>Change</h3>
        <p>${esc(experiment.change || "Not recorded.")}</p>
        <h3>ml.py</h3>
        ${renderDiff(experiment.diff)}
      `;
      item.append(panel);
    }
    improvements.append(item);
  });
}

function move(step) {
  if (!state.rows.length) return;
  state.index = Math.min(state.rows.length - 1, Math.max(0, state.index + step));
  render();
}

prev.addEventListener("click", () => move(-1));
next.addEventListener("click", () => move(1));
window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") move(-1);
  if (event.key === "ArrowRight") move(1);
});

const sessionId = new URLSearchParams(window.location.search).get("id");

fetch("./data/sessions.json")
  .then((response) => {
    if (!response.ok) throw new Error(response.statusText);
    return response.json();
  })
  .then((data) => {
    const session = (data.sessions || []).find((row) => row.id === sessionId);
    if (!session) {
      title.textContent = "Session not found";
      conclusion.textContent = "That session is not in the published summary.";
      return;
    }
    state.session = session;
    state.rows = session.experiments.filter((row) => row.improved);
    document.title = `${session.model || "Session"} · ${session.id}`;
    title.textContent = session.model || "Unknown model";
    meta.textContent = session.id;
    conclusion.textContent = session.conclusion || "No session conclusion recorded.";
    render();
  })
  .catch(() => {
    title.textContent = "Could not load sessions";
    conclusion.innerHTML = `Run <code>python site/build_summary.py</code> from the repo root, then open this page from the site folder.`;
  });
