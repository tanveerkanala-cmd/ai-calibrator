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
  if (!res.ok) throw new Error(detailMessage(data.detail) || res.statusText);
  return data;
}

// FastAPI validation errors arrive as detail=[{loc,msg,...}]; a plain
// `new Error(array)` renders "[object Object]". Turn it into readable text.
function detailMessage(detail) {
  if (Array.isArray(detail)) {
    return detail.map((e) => (e && e.msg) ? e.msg : JSON.stringify(e)).join("; ");
  }
  return typeof detail === "string" ? detail : (detail ? JSON.stringify(detail) : "");
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
  // An empty list is ambiguous on its own, so name the directory that was read —
  // the server and `calibrate init` share it, so this is also where a project
  // created below will appear on disk.
  if (names.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    let root = "";
    try { root = (await api("GET", "/health")).projects_root || ""; } catch { /* ignore */ }
    li.textContent = root
      ? `No projects in ${root} — create one below, or start the server in another directory (\`calibrate serve --projects <dir>\`).`
      : "No projects yet — create one below.";
    ul.appendChild(li);
    return;
  }
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
  const cert = document.createElement("span");
  cert.className = "pill";
  cert.textContent = "… certification";
  h.appendChild(document.createTextNode(" "));
  h.appendChild(cert);
  p.appendChild(h);
  p.insertAdjacentHTML("beforeend", `<p class="muted">${escapeHtml(s.goal)}</p>`);
  api("GET", `/projects/${name}/certification`).then((c) => {
    const marks = { pass: "✓ certified", fail: "✗ gate failing", stale: "⚠ stale — re-run ci", none: "· ungated" };
    cert.textContent = marks[c.status] || c.status;
    cert.title = c.detail || "";
  }).catch(() => { cert.textContent = ""; });

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
  tools.appendChild(stepButton("Try & flag", () => startFlywheel(name, s.name), !done.spec));
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
  const data = await api("POST", path);
  // Reload FIRST — selectProject's own action() clears the banner, so the
  // summary has to be announced after the panel has been rebuilt.
  await selectProject(current);
  const note = runSummary(data);
  if (note) banner(note);
  return data;
}

// Every step route says in its body what actually happened — documents that
// could not be read, gaps that got no question, where the export landed. run()
// used to throw all of it away, so a rejected document never reached the owner.
function runSummary(d) {
  if (!d || typeof d !== "object") return "";
  const parts = [];
  if (Array.isArray(d.skipped) && d.skipped.length) {
    parts.push(`⚠ ${d.skipped.length} file(s) could NOT be read and were left out: ` +
      d.skipped.map((s) => `${s.path} (${s.reason})`).join("; "));
  }
  if (typeof d.analyzed === "number" && typeof d.materials === "number" && d.analyzed < d.materials) {
    parts.push(`⚠ only ${d.analyzed} of ${d.materials} file(s) fit the analysis window — ` +
      `the facts and gaps come from those alone.`);
  }
  if (Array.isArray(d.uncovered_gaps) && d.uncovered_gaps.length) {
    parts.push(`⚠ ${d.uncovered_gaps.length} gap(s) got no question: ${d.uncovered_gaps.join(", ")}`);
  }
  if (typeof d.criteria === "number") {
    parts.push(`Compiled ${d.criteria} criteria and ${d.tests} test(s).`);
  }
  if (d.bundle_dir) parts.push(`Exported to ${d.bundle_dir}`);
  return parts.join("   ");
}

async function runEval(name, refine) {
  const res = await api("POST", `/projects/${name}/eval`, { refine, rounds: 3, threshold: 0.8 });
  // Refining rewrote the spec, so the certification pill and the pipeline counts
  // on screen now describe a config that no longer exists. Re-render before
  // appending the result — a stale "certified" badge must never sit above it.
  if (refine) await selectProject(current);
  const p = $("#panel");
  const last = res.rounds[res.rounds.length - 1];
  const width = Math.round(last.pass_rate * 100);  // bar width only — the NUMBER is honest
  const rounds = res.rounds.map((r, i) => `round ${i + 1} [${escapeHtml(r.run_id)}]: ${pctText(r.pass_rate)}`).join("<br>");
  p.insertAdjacentHTML("beforeend",
    `<div class="card"><strong>Eval</strong><div class="bar"><span style="width:${width}%"></span></div>` +
    `<p>Final pass rate: <b>${pctText(last.pass_rate)}</b></p><p class="muted">${rounds}</p></div>`);
}

// --- M4+ analysis & tuning views -------------------------------------------

function resultCard(title) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<strong>${title}</strong>`;
  $("#panel").appendChild(card);
  return card;
}

// Honest percent (mirrors calibrator/fmt.py): never "100%" unless exactly 1,
// never "0%" unless exactly 0 — 249/250 must not render as a perfect score.
function pctText(x) {
  x = x || 0;
  if (x === 0) return "0%";
  if (x === 1) return "100%";
  const r = Math.round(x * 100);
  if (r >= 100) return ">99%";
  if (r <= 0) return "<1%";
  return `${r}%`;
}

function pctDelta(x) {
  if (!x) return "±0%";
  const s = `${x > 0 ? "+" : "-"}${Math.abs(x * 100).toFixed(1)}%`;
  return s === "+0.0%" || s === "-0.0%" ? (x > 0 ? "+<0.1%" : "-<0.1%") : s;
}

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
  const width = Math.round((d.confidence || 0) * 100);
  card.insertAdjacentHTML("beforeend",
    `<div class="bar"><span style="width:${width}%"></span></div><p>Calibration Confidence: <b>${pctText(d.confidence)}</b></p>`);
  const body = document.createElement("div");
  body.className = "md";
  body.innerHTML = renderMarkdown(d.markdown);
  card.appendChild(body);
}

// The calibration report is the artifact this whole tool points at, and the one
// a non-technical owner is most likely to read or share. It was written into a
// <pre> as raw source, so the person the product is FOR saw `## Coverage`,
// `**67%**` and backticks instead of a report.
//
// A deliberately small subset — headings, bold, italic, inline code, bullets —
// rendered here rather than pulled from a library: the UI ships as three static
// files with no build step, and a markdown dependency is a far larger surface
// than the rules below.
//
// SECURITY: escape FIRST, then add markup. Everything in this document is
// untrusted — the goal, the criteria descriptions and the judge's rationales
// are written by a model from ingested documents — so by the time any rule
// runs, every `<` is already `&lt;` and no rule can emit an element the report
// author chose. Link syntax is deliberately NOT supported: it is the one
// construct that would let untrusted text choose a URL (`javascript:`), and
// the report contains no links.
function renderMarkdown(src) {
  const lines = escapeHtml(String(src || "")).split("\n");
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    // Single-asterisk italics AFTER bold, so `**x**` is already consumed and
    // cannot be mistaken for two emphasis runs. The report uses both spellings.
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(/(^|[\s(])_([^_\n]+)_/g, "$1<em>$2</em>");

  const out = [];
  let depth = 0;                                   // open <ul> count
  const closeLists = () => { while (depth > 0) { out.push("</ul>"); depth--; } };

  for (const line of lines) {
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      closeLists();
      const level = Math.min(heading[1].length + 1, 5);   // `#` -> h2; the page owns h1
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }
    const bullet = /^(\s*)[-*]\s+(.*)$/.exec(line);
    if (bullet) {
      const want = bullet[1].length >= 2 ? 2 : 1;
      while (depth < want) { out.push("<ul>"); depth++; }
      while (depth > want) { out.push("</ul>"); depth--; }
      out.push(`<li>${inline(bullet[2])}</li>`);
      continue;
    }
    closeLists();
    if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  closeLists();
  return out.join("");
}

async function showRedteam(name) {
  const card = resultCard("Red-team");
  card.insertAdjacentHTML("beforeend", `<p class="muted">probing…</p>`);
  const d = await api("POST", `/projects/${name}/redteam`, { max_probes: 8, add_tests: false });
  card.innerHTML = `<strong>Red-team</strong><p>Held ${pctText(d.hold_rate)} — ${d.violations}/${d.probes} probe(s) caused a violation.</p>`;
  if (d.ungraded) {
    // Probes the judge could not grade sit outside the hold rate, so say so here
    // too — otherwise the number reads as covering every probe that ran.
    card.insertAdjacentHTML("beforeend",
      `<p class="muted">${d.ungraded}/${d.probes} probe(s) could not be judged — not counted as held.</p>`);
  }
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
    // A local candidate is free, and `recommended` now prefers it — showing "?"
    // for its price left the table unable to explain the recommendation.
    : `<tr><td>${escapeHtml(r.spec)}</td><td>${pctText(r.pass_rate)}</td><td>${r.local ? "local" : r.in_price != null ? "$" + r.in_price + "/" + r.out_price : "?"}</td></tr>`
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
    `<p>${escapeHtml(d.baseline_run)}: ${pctText(d.baseline_rate)} → ${escapeHtml(d.candidate_run)}: ${pctText(d.candidate_rate)} (Δ ${pctDelta(d.delta)})</p>` +
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
    await selectProject(projName);  // reload first — its action() clears the banner
    banner(`Learned ${res.standards_added} standard(s) + ${res.do_not_added} never-rule(s).`);
  });
  card.appendChild(learn);
}

// --- the flywheel: try the calibrated AI, thumb its answers, absorb ---------

async function startFlywheel(name, projName) {
  const card = resultCard("Try & flag (flywheel)");
  card.insertAdjacentHTML("beforeend",
    `<p class="muted">Ask your calibrated AI, thumb the answer; absorb turns flags into
     examples + pinned tests (the certification then goes stale until re-proven).</p>`);

  const pendingLine = document.createElement("p");
  pendingLine.className = "muted";
  card.appendChild(pendingLine);
  const absorbBtn = document.createElement("button");
  absorbBtn.textContent = "Absorb into spec & tests";
  absorbBtn.disabled = true;  // until the pending count arrives — never absorb blind
  absorbBtn.onclick = () => action(async () => {
    const r = await api("POST", `/projects/${name}/absorb`);
    // reload FIRST — selectProject's own action() clears the banner, so
    // announcing before the reload made the message flash and vanish
    await selectProject(projName);
    banner(`Absorbed ${r.ups + r.downs} record(s): +${r.examples_added} example(s), ` +
           `+${r.tests_added} pinned test(s)${r.tests_added ? " — re-run Eval/ci to re-certify" : ""}.`);
  });
  card.appendChild(absorbBtn);

  const refreshPending = () => api("GET", `/projects/${name}/feedback`).then((d) => {
    pendingLine.textContent = d.pending
      ? `${d.pending} feedback record(s) waiting to be absorbed.`
      : "No feedback waiting.";
    absorbBtn.disabled = !d.pending;
  }).catch(() => {});
  refreshPending();

  const ask = document.createElement("div");
  ask.className = "q";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "ask your AI something…";
  input.style.width = "70%";
  const go = document.createElement("button");
  go.textContent = "Ask";
  ask.appendChild(input);
  ask.appendChild(go);
  card.appendChild(ask);
  const answers = document.createElement("div");
  card.appendChild(answers);

  go.onclick = () => action(async () => {
    const message = input.value.trim();
    if (!message) return;
    go.disabled = true;
    go.textContent = "asking…";
    try {
      const d = await api("POST", `/projects/${name}/try`, { message });
      renderExchange(answers, name, d, refreshPending);
    } finally {
      go.disabled = false;
      go.textContent = "Ask";
    }
  });
}

function renderExchange(container, name, d, refreshPending) {
  const q = document.createElement("div");
  q.className = "q";
  q.style.margin = "0.8rem 0";
  q.innerHTML = `<div class="dim">YOU</div><div>${escapeHtml(d.turns[d.turns.length - 1])}</div>` +
                `<div class="dim">AI</div><div>${escapeHtml(d.output)}</div>`;
  const up = document.createElement("button");
  up.textContent = "👍";
  const down = document.createElement("button");
  down.textContent = "👎";
  const detail = document.createElement("div");
  q.appendChild(up);
  q.appendChild(down);
  q.appendChild(detail);
  container.prepend(q);

  const send = (verdict, correction, reason) => action(async () => {
    await api("POST", `/projects/${name}/feedback`,
              { turns: d.turns, output: d.output, verdict, correction, reason });
    up.disabled = down.disabled = true;
    detail.innerHTML = `<div class="why">recorded ${verdict === "up" ? "👍" : "👎"} — absorb when ready</div>`;
    refreshPending();
  });

  up.onclick = () => send("up", null, null);
  down.onclick = () => {
    detail.innerHTML = "";
    const corr = document.createElement("input");
    corr.type = "text";
    corr.placeholder = "what SHOULD it have said? (optional)";
    corr.style.width = "60%";
    const why = document.createElement("input");
    why.type = "text";
    why.placeholder = "why? (optional)";
    const ok = document.createElement("button");
    ok.textContent = "Send 👎";
    ok.onclick = () => send("down", corr.value || null, why.value || null);
    detail.appendChild(corr);
    detail.appendChild(why);
    detail.appendChild(ok);
  };
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
