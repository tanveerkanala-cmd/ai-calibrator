"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
let current = null; // current project name

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch("/api" + path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function banner(msg) {
  const b = $("#banner");
  if (!msg) { b.classList.add("hidden"); return; }
  b.textContent = msg;
  b.classList.remove("hidden");
}

async function loadAuth() {
  try {
    const st = await api("GET", "/auth");
    $("#auth").textContent = st.map((s) => `${s.configured ? "✓" : "·"} ${s.provider}`).join("   ");
  } catch { /* ignore */ }
}

async function loadProjects() {
  const names = await api("GET", "/projects");
  const ul = $("#projects");
  ul.innerHTML = "";
  for (const name of names) {
    const li = document.createElement("li");
    li.textContent = name;
    if (name === current) li.classList.add("active");
    li.onclick = () => selectProject(name);
    ul.appendChild(li);
  }
}

async function action(fn) {
  banner("");
  try { return await fn(); }
  catch (e) { banner(e.message); }
}

async function selectProject(name) {
  current = name;
  await loadProjects();
  const state = await action(() => api("GET", `/projects/${encodeURIComponent(name)}`));
  if (state) renderPanel(state);
}

function stepButton(label, handler, disabled) {
  const b = document.createElement("button");
  b.textContent = label;
  b.disabled = !!disabled;
  b.onclick = () => action(handler);
  return b;
}

function renderPanel(s) {
  const p = $("#panel");
  p.innerHTML = "";
  const name = encodeURIComponent(s.name);

  const h = document.createElement("h2");
  h.textContent = s.name;
  p.appendChild(h);
  p.insertAdjacentHTML("beforeend", `<p class="muted">${escapeHtml(s.goal)}</p>`);

  // pipeline progress pills
  const done = {
    materials: s.materials.length > 0, gaps: s.gaps.length > 0,
    answered: s.interview.some((i) => i.answer), spec: s.has_spec, tests: s.tests > 0,
  };
  p.insertAdjacentHTML("beforeend",
    `<p>${Object.entries(done).map(([k, v]) => `<span class="pill">${v ? "✓" : "·"} ${k}</span>`).join(" ")}</p>`);

  // material upload + step buttons
  const steps = document.createElement("div");
  steps.className = "steps";
  const upl = document.createElement("input");
  upl.type = "file";
  upl.onchange = () => action(async () => {
    const fd = new FormData(); fd.append("file", upl.files[0]);
    const res = await fetch(`/api/projects/${name}/materials`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || "upload failed");
    selectProject(s.name);
  });
  steps.appendChild(upl);
  steps.appendChild(stepButton("Ingest", () => run(`/projects/${name}/ingest`)));
  steps.appendChild(stepButton("Interview", () => run(`/projects/${name}/interview`), !done.gaps));
  steps.appendChild(stepButton("Compile", () => run(`/projects/${name}/compile`), !done.answered));
  steps.appendChild(stepButton("Eval", () => runEval(name, false), !done.spec));
  steps.appendChild(stepButton("Eval + Refine", () => runEval(name, true), !done.spec));
  steps.appendChild(stepButton("Export", () => run(`/projects/${name}/export`), !done.spec));
  p.appendChild(steps);

  // analysis & tuning actions (the M4+ capabilities)
  const tools = document.createElement("div");
  tools.className = "steps";
  tools.appendChild(stepButton("Teach", () => startTeach(name, s.name)));
  tools.appendChild(stepButton("Coverage", () => showCoverage(name), !done.spec));
  tools.appendChild(stepButton("Report", () => showReport(name), !done.spec));
  tools.appendChild(stepButton("Red-team", () => showRedteam(name), !done.spec));
  tools.appendChild(stepButton("Rightsize", () => showRightsize(name), !done.tests));
  tools.appendChild(stepButton("Drift", () => showDrift(name), !done.tests));
  p.appendChild(tools);

  if (s.gaps.length) {
    p.insertAdjacentHTML("beforeend",
      `<div class="card"><strong>Gaps</strong><ul>${s.gaps.map((g) => `<li>${escapeHtml(g.dimension)}</li>`).join("")}</ul></div>`);
  }

  if (s.interview.length) renderInterview(p, s, name);
}

function renderInterview(p, s, name) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = "<strong>Interview</strong>";
  for (const item of s.interview) {
    const q = document.createElement("div");
    q.className = "q";
    q.style.margin = "0.8rem 0";
    q.innerHTML =
      `<div class="dim">[${escapeHtml(item.dimension || "")}]</div>` +
      `<div>${escapeHtml(item.question)}</div>` +
      (item.rationale ? `<div class="why">why: ${escapeHtml(item.rationale)}</div>` : "");
    const ta = document.createElement("textarea");
    ta.value = item.answer || item.draft_answer || "";
    ta.dataset.qid = item.id;
    q.appendChild(ta);
    card.appendChild(q);
  }
  const save = document.createElement("button");
  save.textContent = "Save answers";
  save.onclick = () => action(async () => {
    const answers = {};
    card.querySelectorAll("textarea").forEach((t) => { answers[t.dataset.qid] = t.value; });
    await api("POST", `/projects/${name}/answers`, { answers });
    selectProject(s.name);
  });
  card.appendChild(save);
  p.appendChild(card);
}

async function run(path) {
  await api("POST", path);
  selectProject(current);
}

async function runEval(name, refine) {
  const res = await api("POST", `/projects/${name}/eval`, { refine, rounds: 3, threshold: 0.8 });
  const p = $("#panel");
  const last = res.rounds[res.rounds.length - 1];
  const pct = Math.round(last.pass_rate * 100);
  const rounds = res.rounds.map((r, i) => `round ${i + 1} [${r.run_id}]: ${Math.round(r.pass_rate * 100)}%`).join("<br>");
  p.insertAdjacentHTML("beforeend",
    `<div class="card"><strong>Eval</strong><div class="bar"><span style="width:${pct}%"></span></div>` +
    `<p>Final pass rate: <b>${pct}%</b></p><p class="muted">${rounds}</p></div>`);
}

// --- M4+ analysis & tuning views -------------------------------------------

function resultCard(title) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<strong>${title}</strong>`;
  $("#panel").appendChild(card);
  return card;
}

const pctText = (x) => `${Math.round((x || 0) * 100)}%`;

async function showCoverage(name) {
  const d = await api("GET", `/projects/${name}/coverage`);
  const card = resultCard("Coverage");
  card.insertAdjacentHTML("beforeend",
    `<p>${pctText(d.coverage_rate)} of criteria targeted (${d.covered}/${d.total_criteria}).</p>` +
    (d.uncovered.length ? `<p class="muted">Untested: ${d.uncovered.map((c) => escapeHtml(c.id)).join(", ")}</p>` : "") +
    d.warnings.map((w) => `<div class="why">⚠ ${escapeHtml(w)}</div>`).join(""));
}

async function showReport(name) {
  const d = await api("GET", `/projects/${name}/report`);
  const card = resultCard("Calibration report");
  const pct = Math.round((d.confidence || 0) * 100);
  card.insertAdjacentHTML("beforeend",
    `<div class="bar"><span style="width:${pct}%"></span></div><p>Calibration Confidence: <b>${pct}%</b></p>`);
  const pre = document.createElement("pre");
  pre.style.whiteSpace = "pre-wrap";
  pre.textContent = d.markdown;
  card.appendChild(pre);
}

async function showRedteam(name) {
  const card = resultCard("Red-team");
  card.insertAdjacentHTML("beforeend", `<p class="muted">probing…</p>`);
  const d = await api("POST", `/projects/${name}/redteam`, { max_probes: 8, add_tests: false });
  card.innerHTML = `<strong>Red-team</strong><p>Held ${pctText(d.hold_rate)} — ${d.violations}/${d.probes} probe(s) caused a violation.</p>`;
  for (const r of d.results.filter((x) => x.violated)) {
    card.insertAdjacentHTML("beforeend",
      `<div class="q"><div class="dim">[${escapeHtml(r.severity)}] ${escapeHtml(r.target)}</div>` +
      `<div class="why">probe: ${escapeHtml(r.input)}</div></div>`);
  }
}

async function showRightsize(name) {
  const card = resultCard("Rightsize");
  card.insertAdjacentHTML("beforeend", `<p class="muted">evaluating models (runs your tests N×)…</p>`);
  const d = await api("POST", `/projects/${name}/rightsize`, { threshold: 0.8 });
  const rows = d.results.map((r) => r.error
    ? `<tr><td>${escapeHtml(r.spec)}</td><td>—</td><td>error</td></tr>`
    : `<tr><td>${escapeHtml(r.spec)}</td><td>${pctText(r.pass_rate)}</td><td>${r.in_price != null ? "$" + r.in_price + "/" + r.out_price : "?"}</td></tr>`
  ).join("");
  card.innerHTML = `<strong>Rightsize</strong>` +
    `<table><tr><th align="left">model</th><th>pass</th><th>$ in/out</th></tr>${rows}</table>` +
    (d.recommended ? `<p>→ Recommended: <b>${escapeHtml(d.recommended)}</b></p>` : `<p class="muted">No model met the bar.</p>`);
}

async function showDrift(name) {
  const card = resultCard("Drift");
  card.insertAdjacentHTML("beforeend", `<p class="muted">re-evaluating…</p>`);
  const d = await api("POST", `/projects/${name}/drift`, {});
  card.innerHTML = `<strong>Drift</strong>` +
    `<p>${escapeHtml(d.baseline_run)}: ${pctText(d.baseline_rate)} → ${escapeHtml(d.candidate_run)}: ${pctText(d.candidate_rate)} (Δ ${pctText(d.delta)})</p>` +
    (d.regressed_tests.length ? `<div class="why">✗ regressed: ${d.regressed_tests.map(escapeHtml).join(", ")}</div>` : "") +
    (d.fixed_tests.length ? `<div class="why">✓ improved: ${d.fixed_tests.map(escapeHtml).join(", ")}</div>` : "") +
    (d.regressed ? `<p><b>⚠ Drift detected.</b></p>` : `<p>✓ No drift.</p>`);
}

async function startTeach(name, projName) {
  const card = resultCard("Teach by example");
  card.insertAdjacentHTML("beforeend", `<p class="muted">generating samples to judge…</p>`);
  const d = await api("POST", `/projects/${name}/teach/draft`, { n: 4 });
  card.innerHTML = `<strong>Teach by example</strong>` +
    `<p class="muted">Approve or reject each output; add a reason if you like.</p>`;
  const rows = [];
  for (const cand of d.candidates) {
    const q = document.createElement("div");
    q.className = "q";
    q.style.margin = "0.8rem 0";
    q.innerHTML = `<div class="dim">INPUT</div><div>${escapeHtml(cand.input)}</div>` +
                  `<div class="dim">OUTPUT</div><div>${escapeHtml(cand.output)}</div>`;
    const sel = document.createElement("select");
    sel.innerHTML = `<option value="approve">👍 approve</option><option value="reject">👎 reject</option>`;
    const reason = document.createElement("input");
    reason.type = "text";
    reason.placeholder = "why? (optional)";
    q.appendChild(sel);
    q.appendChild(reason);
    card.appendChild(q);
    rows.push({ cand, sel, reason });
  }
  const learn = document.createElement("button");
  learn.textContent = "Learn from my judgments";
  learn.onclick = () => action(async () => {
    const judgments = rows.map((r) => ({
      input: r.cand.input, output: r.cand.output,
      approved: r.sel.value === "approve", reason: r.reason.value || null,
    }));
    const res = await api("POST", `/projects/${name}/teach/learn`, { judgments });
    banner(`Learned ${res.standards_added} standard(s) + ${res.do_not_added} never-rule(s).`);
    selectProject(projName);
  });
  card.appendChild(learn);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

$("#new-project").onsubmit = (e) => {
  e.preventDefault();
  const f = e.target;
  // Explicit control access (f.name would collide with HTMLFormElement.name);
  // capture BEFORE reset() clears the inputs.
  const name = f.elements.namedItem("name").value;
  const goal = f.elements.namedItem("goal").value;
  action(async () => {
    const created = await api("POST", "/projects", { name, goal });
    f.reset();
    await loadProjects();
    selectProject(created.name); // use the server's canonical name
  });
};

loadAuth();
loadProjects();
