"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  browsePath: window.ROOT,
  project: null,          // { project, name, pairs, library, versions }
  selectedSources: new Set(),
  edl: null,              // { version, cuts: [{cut, enabled, added}], allChunks }
  picker: null,           // { gap, showAll, checked, reasons, aiStatus }
  pollTimer: null,
  activeJob: null,
  busy: false,
  lastJob: null,          // snapshot of the most recently finished job
  jobHistory: [],         // merged in-memory + on-disk jobs for the open project
  drawerSelection: "live", // "live" or a history index
  elapsedTimer: null,
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text.replace(/<[^>]*>/g, " ").trim() || res.statusText);
  }
  return res.json();
}

function esc(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Folder browser
// ---------------------------------------------------------------------------
async function browse(path) {
  const data = await api(`/api/browse?path=${encodeURIComponent(path)}`);
  state.browsePath = data.path;
  $("#crumbs").innerHTML = data.parent
    ? `<button class="linkish" id="crumb-up">&#8593; up</button> <span>${esc(data.path)}</span>`
    : `<span>${esc(data.path)}</span>`;
  if (data.parent) $("#crumb-up").onclick = () => browse(data.parent);
  const ul = $("#folders");
  ul.innerHTML = "";
  for (const dir of data.dirs) {
    const li = document.createElement("li");
    li.innerHTML = `<button class="linkish dir">${esc(dir.name)}</button>` +
      (dir.has_pairs
        ? ` <button class="open">open${dir.has_autoedit ? " *" : ""}</button>`
        : "");
    li.querySelector(".dir").onclick = () => browse(dir.path);
    const open = li.querySelector(".open");
    if (open) open.onclick = () => openProject(dir.path);
    ul.appendChild(li);
  }
}

// ---------------------------------------------------------------------------
// Project
// ---------------------------------------------------------------------------
async function openProject(path) {
  const data = await api("/api/project/open", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  state.project = data;
  state.selectedSources = new Set();
  resetGraph();
  $("#no-project").classList.add("hidden");
  $("#project").classList.remove("hidden");
  $("#project-name").textContent = data.name;
  const lib = data.library;
  const changed = lib.changed_clips
    ? ` (${lib.changed_clips} changed on disk)` : "";
  if (lib.exists) {
    const kg = lib.kg_clips < lib.total_clips
      ? `, graph ${lib.kg_clips}/${lib.total_clips} clips` : ", graph ready";
    $("#library-status").textContent = `library: ${lib.chunks} chunks${kg}`;
  } else if (lib.partial) {
    $("#library-status").textContent =
      `analyzed ${lib.analyzed_clips}/${lib.total_clips} clips${changed} - analyze to finish`;
  } else {
    $("#library-status").textContent = "not analyzed yet" + changed;
  }
  $("#library-status").className = lib.exists ? "ok" : "warn";
  $("#btn-analyze").textContent =
    lib.partial || lib.changed_clips ? "Analyze new/changed clips" : "Analyze footage";
  showTab("clips");
  renderVersions();
  await renderClips();
  refreshHistory();
}

async function renderClips() {
  const data = await api(`/api/project/clips?path=${encodeURIComponent(state.project.project)}`);
  const grid = $("#clip-grid");
  grid.innerHTML = "";
  for (const clip of data.clips) {
    const card = document.createElement("div");
    card.className = "clip-card";
    const chunkRows = clip.chunks.map((c) =>
      `<div class="chunk"><span class="t">${c.start.toFixed(0)}-${c.end.toFixed(0)}s</span> ` +
      `${esc(c.summary)} <span class="i">i=${(c.interest || 0).toFixed(2)}</span>` +
      (c.transcript ? `<div class="speech">&ldquo;${esc(c.transcript)}&rdquo;</div>` : "") +
      `</div>`).join("");
    card.innerHTML =
      `<label class="clip-head"><input type="checkbox" data-source="${esc(clip.source)}"> ` +
      `<img loading="lazy" src="/api/thumb?path=${encodeURIComponent(state.project.project)}&clip=${encodeURIComponent(clip.source)}" alt="">` +
      `<span>${esc(clip.source)}</span></label>` +
      `<div class="clip-status" data-clip="${esc(clip.source)}"></div>` +
      (clip.chunks.length
        ? `<details><summary>${clip.chunks.length} chunks</summary><div class="chunks">${chunkRows}</div></details>`
        : clip.analyzed
          ? `<div class="hint">analyzed - no notable chunks</div>`
          : `<div class="hint clip-status-hint" data-clip-hint="${esc(clip.source)}">` +
            `${clip.changed ? "changed on disk - re-analysis needed" : "not analyzed"}</div>`);
    card.querySelector("input").onchange = (e) => {
      if (e.target.checked) state.selectedSources.add(clip.source);
      else state.selectedSources.delete(clip.source);
      renderChips();
    };
    grid.appendChild(card);
  }
  renderChips();
}

function renderChips() {
  const box = $("#selected-chips");
  const names = [...state.selectedSources];
  box.innerHTML = names.length
    ? "Composing from: " + names.map((n) => `<span class="chip">${esc(n)}</span>`).join(" ")
    : "Composing from: <em>all clips</em>";
}

// ---------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------
function renderVersions() {
  const tbody = $("#versions-table tbody");
  tbody.innerHTML = "";
  for (const v of [...state.project.versions].reverse()) {
    const tr = document.createElement("tr");
    const files = [v.has_preview ? "preview" : null, v.has_final ? "final" : null]
      .filter(Boolean).join(", ") || "-";
    tr.innerHTML =
      `<td>v${v.version}</td><td>${esc(v.created)}</td><td>${esc(v.origin)}</td>` +
      `<td class="brief-cell">${esc(v.brief || "")}</td><td>${v.cut_count}</td><td>${files}</td>` +
      `<td>` +
      (v.has_preview ? `<button data-act="play">play</button>` : "") +
      (v.has_final ? `<button data-act="play-final">play final</button>` : "") +
      `<button data-act="edit">edit EDL</button>` +
      `<button data-act="preview">render preview</button>` +
      `<button data-act="finalize">finalize</button>` +
      `</td>`;
    tr.querySelectorAll("button").forEach((b) => {
      b.onclick = () => versionAction(b.dataset.act, v.version);
    });
    tbody.appendChild(tr);
  }
}

async function versionAction(act, version) {
  if (act === "play") return playVideo(`preview_v${version}.mp4`);
  if (act === "play-final") return playVideo(`final_v${version}.mp4`);
  if (act === "edit") return loadEdl(version);
  if (act === "preview" || act === "finalize") {
    return startJob(act === "preview" ? "preview" : "finalize", { version });
  }
}

async function refreshProject() {
  if (state.project) await openProject(state.project.project);
}

// ---------------------------------------------------------------------------
// EDL editor
// ---------------------------------------------------------------------------
async function loadEdl(version) {
  const data = await api(`/api/edl?path=${encodeURIComponent(state.project.project)}&version=${version}`);
  const clipData = await api(`/api/project/clips?path=${encodeURIComponent(state.project.project)}`);
  const allChunks = [];
  for (const clip of clipData.clips) {
    for (const c of clip.chunks) allChunks.push({ ...c, source: clip.source });
  }
  allChunks.sort((a, b) => posCmp([a.source, a.start], [b.source, b.start]));
  state.edl = {
    version,
    cuts: data.cuts.map((cut) => ({ cut, enabled: true })),
    allChunks,
  };
  state.picker = null;
  $("#edl-tab").disabled = false;
  $("#edl-title").textContent = `Editing v${version} (${data.cuts.length} cuts)`;
  renderEdl();
  showTab("edl");
}

function posCmp(a, b) {
  return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] - b[1];
}

function lrfName(source) {
  const pair = state.project.pairs.find((p) => p.source === source);
  return pair ? pair.lrf.split(/[\\/]/).pop() : null;
}

function renderEdl() {
  const tbody = $("#edl-table tbody");
  tbody.innerHTML = "";
  tbody.appendChild(insertRow(0));
  state.edl.cuts.forEach((row, idx) => {
    const c = row.cut;
    const tr = document.createElement("tr");
    tr.className = (row.enabled ? "" : "disabled-cut") + (row.added ? " added-cut" : "");
    tr.draggable = true;
    tr.dataset.idx = idx;
    tr.innerHTML =
      `<td class="drag">&#8942;&#8942;</td>` +
      `<td><input type="checkbox" ${row.enabled ? "checked" : ""}></td>` +
      `<td>${esc(c.source || "")}</td>` +
      `<td><input class="num" type="number" step="0.5" value="${c.start.toFixed(1)}"></td>` +
      `<td><input class="num" type="number" step="0.5" value="${c.end.toFixed(1)}"></td>` +
      `<td>${esc(c.role || "")}${c.beat ? " / " + esc(c.beat) : ""}</td>` +
      `<td class="summary-cell">${esc(c.summary || "")}` +
      (c.speech ? `<div class="speech">&ldquo;${esc(c.speech)}&rdquo;</div>` : "") + `</td>` +
      `<td><button class="cut-play" title="preview this cut from the LRF proxy">&#9654;</button></td>`;
    tr.querySelector('input[type="checkbox"]').onchange = (e) => {
      row.enabled = e.target.checked;
      tr.className = (row.enabled ? "" : "disabled-cut") + (row.added ? " added-cut" : "");
    };
    const [startInput, endInput] = tr.querySelectorAll(".num");
    startInput.onchange = () => { c.start = parseFloat(startInput.value); };
    endInput.onchange = () => { c.end = parseFloat(endInput.value); };
    tr.querySelector(".cut-play").onclick = () => {
      const file = c.lrf_file ? c.lrf_file.split(/[\\/]/).pop() : lrfName(c.source);
      if (!file) return alert("No LRF proxy known for this cut.");
      playVideo(file, c.start, c.end);
    };
    tr.ondragstart = (e) => e.dataTransfer.setData("text/plain", idx);
    tr.ondragover = (e) => e.preventDefault();
    tr.ondrop = (e) => {
      e.preventDefault();
      const from = parseInt(e.dataTransfer.getData("text/plain"), 10);
      const to = idx;
      if (from === to) return;
      const [moved] = state.edl.cuts.splice(from, 1);
      state.edl.cuts.splice(to, 0, moved);
      if (state.picker) state.picker = null;
      renderEdl();
    };
    tbody.appendChild(tr);
    tbody.appendChild(insertRow(idx + 1));
  });
  renderPicker();
}

function insertRow(gap) {
  const tr = document.createElement("tr");
  tr.className = "insert-row" + (state.picker && state.picker.gap === gap ? " active" : "");
  const cols = $("#edl-table thead tr").children.length;
  tr.innerHTML = `<td colspan="${cols}"><button class="linkish">+ add clip here</button></td>`;
  tr.querySelector("button").onclick = () => openPicker(gap);
  return tr;
}

// ---------------------------------------------------------------------------
// Add-clip picker
// ---------------------------------------------------------------------------
function pickerWindow(gap) {
  const enabled = state.edl.cuts
    .map((row, idx) => ({ row, idx }))
    .filter((e) => e.row.enabled);
  const prev = [...enabled].reverse().find((e) => e.idx < gap);
  const next = enabled.find((e) => e.idx >= gap);
  const lo = prev ? [prev.row.cut.source, prev.row.cut.end] : null;
  const hi = next ? [next.row.cut.source, next.row.cut.start] : null;
  return { lo, hi, prev: prev && prev.row.cut, next: next && next.row.cut };
}

function chunkInEdit(chunk) {
  return state.edl.cuts.some((r) => r.enabled && r.cut.source === chunk.source
    && r.cut.start < chunk.end && chunk.start < r.cut.end);
}

function pickerCandidates(gap, showAll) {
  if (showAll) return state.edl.allChunks.filter((c) => !chunkInEdit(c));
  const { lo, hi } = pickerWindow(gap);
  if (lo && hi && posCmp(lo, hi) >= 0) return [];
  return state.edl.allChunks.filter((c) =>
    (!lo || posCmp([c.source, c.start], lo) >= 0) &&
    (!hi || posCmp([c.source, c.end], hi) <= 0));
}

function openPicker(gap) {
  state.picker = { gap, showAll: false, checked: new Set(), reasons: {}, roles: {},
                   aiStatus: "", query: "" };
  renderEdl();
  $("#edl-picker").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderPicker() {
  const box = $("#edl-picker");
  const p = state.picker;
  if (!p) { box.classList.add("hidden"); box.innerHTML = ""; return; }
  box.classList.remove("hidden");
  const { prev, next } = pickerWindow(p.gap);
  const candidates = pickerCandidates(p.gap, p.showAll);
  const where = p.showAll
    ? "anywhere (unused chunks)"
    : `between ${prev ? `${esc(prev.source)} @${prev.end.toFixed(0)}s` : "the start"}` +
      ` and ${next ? `${esc(next.source)} @${next.start.toFixed(0)}s` : "the end"}`;

  box.innerHTML =
    `<div class="picker-head"><strong>Add clip ${where}</strong>` +
    `<button id="picker-close" class="linkish">close</button></div>` +
    `<div class="picker-ai">` +
    `<input id="picker-query" type="text" value="${esc(p.query)}" placeholder="brief for this gap, e.g. 'the coffee stop, keep it light'">` +
    `<button id="picker-suggest" data-empty="${candidates.length ? 0 : 1}"` +
    `${candidates.length && !state.busy ? "" : " disabled"}` +
    `${state.busy ? ' title="another job is running"' : ""}>Compose gap fill</button>` +
    `<span id="picker-ai-status" class="hint">${esc(p.aiStatus)}</span></div>` +
    (candidates.length
      ? `<div class="picker-grid">` + candidates.map((c) => {
          const checked = p.checked.has(c.library_id);
          const reason = p.reasons[c.library_id];
          return `<div class="pick-card${checked ? " picked" : ""}" data-lid="${c.library_id}">` +
            `<img loading="lazy" src="/api/thumb?path=${encodeURIComponent(state.project.project)}&clip=${encodeURIComponent(c.source)}" alt="">` +
            `<div class="pick-info">` +
            `<div><span class="t">${esc(c.source)} ${c.start.toFixed(0)}-${c.end.toFixed(0)}s</span>` +
            ` <span class="i">i=${(c.interest || 0).toFixed(2)}</span>` +
            (chunkInEdit(c) ? ` <span class="badge waiting">in edit</span>` : "") + `</div>` +
            `<div>${esc(c.summary)}</div>` +
            (c.transcript ? `<div class="speech">&ldquo;${esc(c.transcript)}&rdquo;</div>` : "") +
            (reason ? `<div class="ai-reason">${p.roles[c.library_id] ? esc(p.roles[c.library_id]) + ": " : ""}${esc(reason)}</div>` : "") +
            `</div>` +
            `<div class="pick-actions">` +
            `<button class="pick-play" title="preview">&#9654;</button>` +
            `<input type="checkbox" ${checked ? "checked" : ""}>` +
            `</div></div>`;
        }).join("") + `</div>`
      : `<div class="hint">No unused chunks fall between these two cuts.</div>`) +
    `<div class="picker-foot">` +
    `<label class="inline"><input id="picker-showall" type="checkbox" ${p.showAll ? "checked" : ""}> show all unused chunks</label>` +
    `<button id="picker-add" ${p.checked.size ? "" : "disabled"}>Add selected (${p.checked.size})</button>` +
    `</div>`;

  $("#picker-close").onclick = () => { state.picker = null; renderEdl(); };
  $("#picker-showall").onchange = (e) => { p.showAll = e.target.checked; renderPicker(); };
  $("#picker-query").oninput = (e) => { p.query = e.target.value; };
  $("#picker-suggest").onclick = () => runSuggest(candidates);
  $("#picker-add").onclick = () => addPicked(candidates);
  box.querySelectorAll(".pick-card").forEach((card) => {
    const lid = parseInt(card.dataset.lid, 10);
    const chunk = candidates.find((c) => c.library_id === lid);
    card.querySelector(".pick-play").onclick = () => {
      const file = lrfName(chunk.source);
      if (!file) return alert("No LRF proxy known for this clip.");
      playVideo(file, chunk.start, chunk.end);
    };
    card.querySelector('input[type="checkbox"]').onchange = (e) => {
      if (e.target.checked) p.checked.add(lid);
      else p.checked.delete(lid);
      renderPicker();
    };
  });
}

async function runSuggest(candidates) {
  const p = state.picker;
  const brief = p.query.trim();
  if (!brief) return alert("Give a brief for this gap first.");
  p.query = brief;
  p.aiStatus = "submitting…";
  renderPicker();
  const syncPicker = () => {
    if (state.picker !== p) return false;
    renderPicker();
    return true;
  };
  try {
    const data = await api("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "suggest", project: state.project.project,
        params: {
          brief,
          candidate_ids: candidates.map((c) => c.library_id),
        },
      }),
    });
    watchJob(data.job_id, "suggest", {
      onUpdate(j) {
        const status = j.stage || j.status;
        if (status === p.aiStatus) return;
        p.aiStatus = status;
        syncPicker();
      },
      onDone(j) {
        if (j.status === "error") {
          p.aiStatus = "failed: " + j.error;
        } else {
          const picks = (j.result && j.result.picks) || [];
          p.aiStatus = picks.length
            ? `composed ${picks.length} cut(s) for "${p.query}" - review below`
            : `nothing matched "${p.query}"`;
          for (const pick of picks) {
            p.checked.add(pick.library_id);
            p.reasons[pick.library_id] = pick.reason;
            p.roles[pick.library_id] = pick.role;
          }
        }
        syncPicker();
      },
    });
  } catch (e) {
    p.aiStatus = "failed: " + e.message;
    renderPicker();
  }
}

function addPicked(candidates) {
  const p = state.picker;
  const chosen = candidates
    .filter((c) => p.checked.has(c.library_id))
    .sort((a, b) => posCmp([a.source, a.start], [b.source, b.start]));
  const pairBySource = Object.fromEntries(state.project.pairs.map((x) => [x.source, x]));
  const rows = chosen.map((c) => {
    const pair = pairBySource[c.source] || {};
    return {
      enabled: true,
      added: true,
      cut: {
        source_file: pair.hires || "", lrf_file: pair.lrf || "",
        source: c.source, start: c.start, end: c.end,
        role: p.roles[c.library_id] || "manual", beat: "",
        summary: c.summary || "", speech: c.transcript || "",
      },
    };
  });
  state.edl.cuts.splice(p.gap, 0, ...rows);
  state.picker = null;
  renderEdl();
}

async function saveEdl() {
  const cuts = state.edl.cuts.filter((r) => r.enabled).map((r) => r.cut);
  if (!cuts.length) return alert("All cuts are disabled - nothing to save.");
  try {
    const data = await api("/api/edl", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.project.project, base_version: state.edl.version, cuts }),
    });
    state.picker = null;
    $("#edl-picker").classList.add("hidden");
    await refreshProject();
    showTab("versions");
    alert(`Saved as v${data.version} - render a preview or finalize it from the Versions tab.`);
  } catch (e) {
    alert("Save failed: " + e.message);
  }
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------
async function startJob(type, params) {
  try {
    const data = await api("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, project: state.project.project, params }),
    });
    watchJob(data.job_id, type);
  } catch (e) {
    alert("Could not start job: " + e.message);
  }
}

function watchJob(jobId, type, hooks = {}) {
  state.activeJob = { id: jobId, type, logOffset: 0, started: Date.now() };
  state.drawerSelection = "live";
  if (type !== "suggest") $("#job-drawer").classList.remove("hidden");
  $("#job-title").textContent = `${type} job #${jobId} - queued`;
  $("#job-log").textContent = "";
  setBusy(true);
  renderJobHistory();
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${jobId}?log_offset=${state.activeJob.logOffset}`);
      if (j.started) state.activeJob.started = Date.parse(j.started);
      if (j.log_delta) {
        if (state.drawerSelection === "live") {
          $("#job-log").textContent += j.log_delta;
          $("#job-log").scrollTop = $("#job-log").scrollHeight;
        }
        state.activeJob.logOffset = j.log_offset;
      }
      $("#job-title").textContent = `${j.type} job #${jobId} - ${j.stage}`;
      updateStatusBar(j);
      if (j.type === "analyze" && j.progress) updateClipStatuses(j.progress);
      if (hooks.onUpdate) hooks.onUpdate(j);
      if (j.status === "done" || j.status === "error") {
        clearInterval(state.pollTimer);
        state.lastJob = j;
        state.activeJob = null;
        setBusy(false);
        updateStatusBar(null);
        if (j.status === "error") $("#job-title").textContent += ` - FAILED: ${j.error}`;
        if (hooks.onDone) hooks.onDone(j);
        await refreshHistory();
        if (j.type !== "suggest") {
          await refreshProject();
          if (j.status === "done" && j.result && j.result.version !== undefined
              && j.type !== "finalize") {
            showTab("versions");
          }
        }
      }
    } catch (e) {
      clearInterval(state.pollTimer);
      state.activeJob = null;
      setBusy(false);
      updateStatusBar(null);
    }
  }, 1000);
}

function setBusy(busy) {
  state.busy = busy;
  document.querySelectorAll("#btn-analyze, #btn-compose, #versions-table button")
    .forEach((b) => {
      if (["preview", "finalize"].includes(b.dataset.act) || !b.dataset.act) b.disabled = busy;
    });
  const suggest = $("#picker-suggest");
  if (suggest) suggest.disabled = busy || suggest.dataset.empty === "1";
}

function fmtDuration(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
}

function fmtAgo(iso) {
  const s = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function jobDuration(j) {
  if (!j.started || !j.finished) return "";
  return fmtDuration(Date.parse(j.finished) - Date.parse(j.started));
}

function updateStatusBar(j) {
  const bar = $("#status-bar");
  const spinner = $("#status-spinner");
  const text = $("#status-text");
  const elapsed = $("#status-elapsed");
  if (j && (j.status === "running" || j.status === "queued")) {
    bar.className = "running";
    spinner.classList.remove("hidden");
    text.textContent = `${j.type} #${j.id} — ${j.stage}`;
    return;
  }
  spinner.classList.add("hidden");
  const last = state.lastJob;
  if (last) {
    const ok = last.status === "done";
    bar.className = ok ? "" : "error";
    text.textContent = (ok ? "✓" : "✗") + ` ${last.type} #${last.id} ` +
      (ok ? `finished${last.finished ? " " + fmtAgo(last.finished) : ""}` +
            (last.summary ? ` — ${last.summary}` : "")
          : `failed — ${last.error}`);
    elapsed.textContent = jobDuration(last);
  } else {
    bar.className = "";
    text.textContent = "no job running";
    elapsed.textContent = "";
  }
}

function tickElapsed() {
  const a = state.activeJob;
  if (a) {
    $("#status-elapsed").textContent = fmtDuration(Date.now() - a.started);
  } else if (state.lastJob && state.lastJob.finished) {
    updateStatusBar(null); // refresh the "Xm ago" text
  }
}

async function refreshHistory() {
  if (!state.project) return;
  try {
    const data = await api(`/api/jobs?path=${encodeURIComponent(state.project.project)}`);
    state.jobHistory = data.jobs.filter((j) => j.status === "done" || j.status === "error");
    if (!state.lastJob && state.jobHistory.length) state.lastJob = state.jobHistory[0];
    renderJobHistory();
    if (!state.activeJob) updateStatusBar(null);
  } catch (e) {
    /* history is best-effort */
  }
}

function renderJobHistory() {
  const ul = $("#job-history");
  ul.innerHTML = "";
  if (state.activeJob) {
    const li = document.createElement("li");
    li.className = state.drawerSelection === "live" ? "active" : "";
    li.innerHTML = `<span class="glyph">⟳</span>` +
      `<span>${esc(state.activeJob.type)} #${state.activeJob.id}</span>` +
      `<span class="dur">running</span>`;
    li.onclick = () => { state.drawerSelection = "live"; selectLiveJob(); };
    ul.appendChild(li);
  }
  state.jobHistory.forEach((j, idx) => {
    const ok = j.status === "done";
    const li = document.createElement("li");
    li.className = state.drawerSelection === idx ? "active" : "";
    li.title = [j.params_summary, j.summary || j.error].filter(Boolean).join("\n");
    li.innerHTML = `<span class="glyph ${ok ? "done" : "error"}">${ok ? "✓" : "✗"}</span>` +
      `<span>${esc(j.type)} #${j.id}</span>` +
      (j.summary ? `<span class="jsum">${esc(j.summary)}</span>` : "") +
      `<span class="dur">${jobDuration(j)}</span>`;
    li.onclick = () => selectHistoryJob(idx);
    ul.appendChild(li);
  });
  if (!state.activeJob && !state.jobHistory.length) {
    ul.innerHTML = `<li class="hint">no jobs yet</li>`;
  }
}

function selectLiveJob() {
  renderJobHistory();
  const a = state.activeJob;
  $("#job-log").textContent = "";
  if (a) {
    a.logOffset = 0; // next poll re-fetches the full log
    $("#job-title").textContent = `${a.type} job #${a.id}`;
  }
}

async function selectHistoryJob(idx) {
  const j = state.jobHistory[idx];
  state.drawerSelection = idx;
  renderJobHistory();
  $("#job-title").textContent = `${j.type} job #${j.id} - ${j.status}` +
    (j.summary ? ` - ${j.summary}` : "") + (j.error ? ` - ${j.error}` : "");
  if (j.from_history) {
    $("#job-log").textContent = j.log || "(no log)";
  } else {
    try {
      const full = await api(`/api/jobs/${j.id}`);
      $("#job-log").textContent = full.log_delta || "(no log)";
    } catch (e) {
      $("#job-log").textContent = "(log unavailable: " + e.message + ")";
    }
  }
}

function toggleDrawer() {
  const drawer = $("#job-drawer");
  const opening = drawer.classList.contains("hidden");
  drawer.classList.toggle("hidden");
  if (opening) refreshHistory();
}

function updateClipStatuses(progress) {
  document.querySelectorAll(".clip-status").forEach((el) => {
    const clip = el.dataset.clip;
    const hint = document.querySelector(`[data-clip-hint="${CSS.escape(clip)}"]`);
    if (progress.done.includes(clip)) {
      el.innerHTML = `<span class="badge done">analyzed &#10003;</span>`;
      if (hint) hint.remove();
    } else if (progress.current === clip) {
      el.innerHTML = `<span class="badge running">analyzing&hellip;</span>`;
      if (hint) hint.textContent = "analyzing now";
    } else {
      el.innerHTML = `<span class="badge waiting">waiting</span>`;
    }
  });
}

// ---------------------------------------------------------------------------
// Video modal
// ---------------------------------------------------------------------------
let segmentWatcher = null;

function playVideo(file, start, end) {
  const url = `/api/media?path=${encodeURIComponent(state.project.project)}&file=${encodeURIComponent(file)}`;
  const player = $("#player");
  if (segmentWatcher) {
    player.removeEventListener("timeupdate", segmentWatcher);
    segmentWatcher = null;
  }
  player.src = url;
  if (start !== undefined) {
    player.addEventListener("loadedmetadata", () => { player.currentTime = start; },
      { once: true });
    if (end !== undefined) {
      segmentWatcher = () => { if (player.currentTime >= end) player.pause(); };
      player.addEventListener("timeupdate", segmentWatcher);
    }
  }
  $("#video-modal").classList.remove("hidden");
  player.play().catch(() => {});
}

// ---------------------------------------------------------------------------
// Knowledge graph explorer
// ---------------------------------------------------------------------------
const GRAPH_COLORS = {
  thread: "#e8b350", place: "#7ab7ff", activity: "#7dd87d",
  person: "#e88ad0", object: "#9aa3af", mood: "#d6c37a",
};

const graph = {
  data: null, nodes: [], edges: [], byId: {},
  selected: null, search: "",
  panX: 0, panY: 0, zoom: 1,
  animFrame: null, loading: false,
};

function resetGraph() {
  if (graph.animFrame) cancelAnimationFrame(graph.animFrame);
  graph.data = null;
  graph.nodes = [];
  graph.edges = [];
  graph.byId = {};
  graph.selected = null;
  graph.search = "";
  graph.panX = 0;
  graph.panY = 0;
  graph.zoom = 1;
  graph.animFrame = null;
  graph.loading = false;
  const search = $("#graph-search");
  if (search) search.value = "";
  const panel = $("#graph-panel");
  if (panel) { panel.classList.add("hidden"); panel.innerHTML = ""; }
}

async function loadGraph() {
  if (graph.loading || !state.project) return;
  graph.loading = true;
  try {
    const data = await api(`/api/project/graph?path=${encodeURIComponent(state.project.project)}`);
    graph.data = data;
    $("#graph-empty").classList.toggle("hidden", data.nodes.length > 0);
    buildGraphLayout(data);
    renderGraphLegend(data);
    startGraphSim();
  } catch (e) {
    $("#graph-empty").textContent = "Graph unavailable: " + e.message;
    $("#graph-empty").classList.remove("hidden");
  } finally {
    graph.loading = false;
  }
}

function buildGraphLayout(data) {
  const canvas = $("#graph-canvas");
  const w = canvas.clientWidth || 800;
  const h = canvas.clientHeight || 500;
  const elements = new Set(data.elements || []);
  graph.nodes = data.nodes.map((n, i) => {
    const angle = (i / Math.max(1, data.nodes.length)) * 2 * Math.PI;
    return {
      ...n,
      element: elements.has(n.id),
      x: w / 2 + Math.cos(angle) * Math.min(w, h) * 0.33,
      y: h / 2 + Math.sin(angle) * Math.min(w, h) * 0.33,
      vx: 0, vy: 0,
      r: Math.min(26, 7 + Math.sqrt(n.chunk_count) * 3),
    };
  });
  graph.byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  graph.edges = (data.edges || [])
    .filter((e) => graph.byId[e.a] && graph.byId[e.b]);
  graph.panX = 0;
  graph.panY = 0;
  graph.zoom = 1;
}

function startGraphSim() {
  if (graph.animFrame) cancelAnimationFrame(graph.animFrame);
  const canvas = $("#graph-canvas");
  let ticks = 0;
  const step = () => {
    const w = canvas.clientWidth || 800;
    const h = canvas.clientHeight || 500;
    if (ticks < 300) {
      simTick(w, h);
      ticks += 1;
    }
    drawGraph();
    graph.animFrame = ticks < 300 ? requestAnimationFrame(step) : null;
  };
  graph.animFrame = requestAnimationFrame(step);
}

function simTick(w, h) {
  const nodes = graph.nodes;
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    // gentle pull to center keeps disconnected nodes on screen
    a.vx += (w / 2 - a.x) * 0.0015;
    a.vy += (h / 2 - a.y) * 0.0015;
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      let dx = a.x - b.x, dy = a.y - b.y;
      let d2 = dx * dx + dy * dy;
      if (d2 < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 1; }
      const force = 900 / d2;
      const d = Math.sqrt(d2);
      const fx = (dx / d) * force, fy = (dy / d) * force;
      a.vx += fx; a.vy += fy;
      b.vx -= fx; b.vy -= fy;
    }
  }
  for (const e of graph.edges) {
    const a = graph.byId[e.a], b = graph.byId[e.b];
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 1;
    const pull = (d - 90) * 0.002 * Math.min(3, e.weight);
    const fx = (dx / d) * pull, fy = (dy / d) * pull;
    a.vx += fx; a.vy += fy;
    b.vx -= fx; b.vy -= fy;
  }
  for (const n of nodes) {
    n.vx *= 0.85; n.vy *= 0.85;
    n.x += n.vx; n.y += n.vy;
  }
}

function graphMatches(n) {
  if (!graph.search) return true;
  return n.name.includes(graph.search) || n.type.includes(graph.search);
}

function drawGraph() {
  const canvas = $("#graph-canvas");
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.translate(graph.panX, graph.panY);
  ctx.scale(graph.zoom, graph.zoom);

  for (const e of graph.edges) {
    const a = graph.byId[e.a], b = graph.byId[e.b];
    const lit = graphMatches(a) && graphMatches(b);
    ctx.strokeStyle = lit ? "rgba(122, 183, 255, 0.25)" : "rgba(122, 183, 255, 0.06)";
    ctx.lineWidth = Math.min(4, 0.5 + e.weight * 0.5);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  for (const n of graph.nodes) {
    const lit = graphMatches(n);
    ctx.globalAlpha = lit ? 1 : 0.18;
    ctx.fillStyle = GRAPH_COLORS[n.type] || "#9aa3af";
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, 2 * Math.PI);
    ctx.fill();
    if (n.id === graph.selected) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (n.element) {
      ctx.strokeStyle = "rgba(255,255,255,0.45)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.fillStyle = lit ? "#e6e9ef" : "#9aa3af";
    ctx.font = "11px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(n.name, n.x, n.y + n.r + 12);
    ctx.globalAlpha = 1;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function graphNodeAt(clientX, clientY) {
  const rect = $("#graph-canvas").getBoundingClientRect();
  const x = (clientX - rect.left - graph.panX) / graph.zoom;
  const y = (clientY - rect.top - graph.panY) / graph.zoom;
  for (let i = graph.nodes.length - 1; i >= 0; i--) {
    const n = graph.nodes[i];
    const dx = x - n.x, dy = y - n.y;
    if (dx * dx + dy * dy <= (n.r + 4) * (n.r + 4)) return n;
  }
  return null;
}

function renderGraphLegend(data) {
  const counts = {};
  for (const n of data.nodes) counts[n.type] = (counts[n.type] || 0) + 1;
  $("#graph-legend").innerHTML = Object.entries(GRAPH_COLORS)
    .filter(([type]) => counts[type])
    .map(([type, color]) =>
      `<span class="legend-item"><span class="legend-dot" style="background:${color}"></span>${type} (${counts[type]})</span>`)
    .join("");
}

function renderGraphPanel(node) {
  const panel = $("#graph-panel");
  if (!node) {
    panel.classList.add("hidden");
    panel.innerHTML = "";
    return;
  }
  panel.classList.remove("hidden");
  const links = graph.edges
    .filter((e) => e.a === node.id || e.b === node.id)
    .map((e) => graph.byId[e.a === node.id ? e.b : e.a])
    .filter(Boolean);
  panel.innerHTML =
    `<div class="gp-head"><span class="legend-dot" style="background:${GRAPH_COLORS[node.type]}"></span>` +
    `<strong>${esc(node.name)}</strong> <span class="hint">${esc(node.type)}</span>` +
    `<button id="gp-close" class="linkish">close</button></div>` +
    (links.length
      ? `<div class="gp-links">appears with: ` +
        links.slice(0, 8).map((l) => `<span class="chip gp-link" data-id="${esc(l.id)}">${esc(l.name)}</span>`).join(" ") + `</div>`
      : "") +
    `<div class="gp-chunks">` + node.chunks.map((c) =>
      `<div class="gp-chunk" data-source="${esc(c.source)}" data-start="${c.start}" data-end="${c.end}">` +
      `<img loading="lazy" src="/api/thumb?path=${encodeURIComponent(state.project.project)}&clip=${encodeURIComponent(c.source)}" alt="">` +
      `<div><span class="t">${esc(c.source)} ${c.start.toFixed(0)}-${c.end.toFixed(0)}s</span>` +
      ` <span class="i">i=${(c.interest || 0).toFixed(2)}</span>` +
      `<div>${esc(c.summary)}</div></div></div>`).join("") + `</div>`;
  $("#gp-close").onclick = () => {
    graph.selected = null;
    renderGraphPanel(null);
    drawGraph();
  };
  panel.querySelectorAll(".gp-link").forEach((el) => {
    el.onclick = () => selectGraphNode(el.dataset.id);
  });
  panel.querySelectorAll(".gp-chunk").forEach((el) => {
    el.onclick = () => {
      const file = lrfName(el.dataset.source);
      if (!file) return alert("No LRF proxy known for this clip.");
      playVideo(file, parseFloat(el.dataset.start), parseFloat(el.dataset.end));
    };
  });
}

function selectGraphNode(id) {
  graph.selected = id;
  renderGraphPanel(graph.byId[id] || null);
  drawGraph();
}

function wireGraphCanvas() {
  const canvas = $("#graph-canvas");
  let dragging = false, moved = false, lastX = 0, lastY = 0;
  canvas.onmousedown = (e) => {
    dragging = true; moved = false;
    lastX = e.clientX; lastY = e.clientY;
  };
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    graph.panX += dx;
    graph.panY += dy;
    lastX = e.clientX; lastY = e.clientY;
    drawGraph();
  });
  window.addEventListener("mouseup", (e) => {
    if (!dragging) return;
    dragging = false;
    if (!moved && e.target === canvas) {
      const node = graphNodeAt(e.clientX, e.clientY);
      selectGraphNode(node ? node.id : null);
      if (!node) renderGraphPanel(null);
    }
  });
  canvas.onwheel = (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const next = Math.max(0.25, Math.min(4, graph.zoom * factor));
    // zoom around the cursor
    graph.panX = mx - (mx - graph.panX) * (next / graph.zoom);
    graph.panY = my - (my - graph.panY) * (next / graph.zoom);
    graph.zoom = next;
    drawGraph();
  };
  $("#graph-search").oninput = (e) => {
    graph.search = e.target.value.trim().toLowerCase();
    drawGraph();
  };
}
wireGraphCanvas();

// ---------------------------------------------------------------------------
// Tabs + wiring
// ---------------------------------------------------------------------------
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.id === `tab-${name}`));
  if (name === "graph" && !graph.data) loadGraph();
  else if (name === "graph") drawGraph();
}

document.querySelectorAll("#tabs button").forEach((b) => {
  b.onclick = () => { if (!b.disabled) showTab(b.dataset.tab); };
});

$("#btn-analyze").onclick = () => startJob("analyze", {});
$("#btn-compose").onclick = () => {
  const minutes = parseFloat($("#target-len").value);
  startJob("compose", {
    brief: $("#brief").value.trim(),
    target_len: minutes > 0 ? minutes * 60 : null,
    margin: parseFloat($("#margin").value) || 3,
    sources: [...state.selectedSources],
  });
};
$("#btn-edl-save").onclick = saveEdl;
$("#job-close").onclick = () => $("#job-drawer").classList.add("hidden");
$("#status-bar").onclick = toggleDrawer;
state.elapsedTimer = setInterval(tickElapsed, 1000);
$("#video-close").onclick = () => {
  const player = $("#player");
  if (segmentWatcher) {
    player.removeEventListener("timeupdate", segmentWatcher);
    segmentWatcher = null;
  }
  player.pause();
  player.src = "";
  $("#video-modal").classList.add("hidden");
};

// On page load, re-attach to any job still queued/running on the server
// (jobs survive refreshes - only the browser's handle is lost).
async function reconnectJobs() {
  try {
    const data = await api("/api/jobs");
    const active = data.jobs.find((j) => j.status === "running" || j.status === "queued");
    if (!active) {
      state.lastJob = data.jobs.find(
        (j) => j.status === "done" || j.status === "error") || null;
      updateStatusBar(null);
      return;
    }
    if (!state.project) await openProject(active.project);
    watchJob(active.id, active.type);
  } catch (e) {
    /* no jobs yet or server restarted - nothing to re-attach */
  }
}

browse(window.ROOT).catch((e) => alert("Browse failed: " + e.message));
reconnectJobs();
