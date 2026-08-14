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
  $("#no-project").classList.add("hidden");
  $("#project").classList.remove("hidden");
  $("#project-name").textContent = data.name;
  const lib = data.library;
  const changed = lib.changed_clips
    ? ` (${lib.changed_clips} changed on disk)` : "";
  if (lib.exists) {
    $("#library-status").textContent = `library: ${lib.chunks} chunks`;
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
  state.picker = { gap, showAll: false, checked: new Set(), reasons: {}, roles: {}, aiStatus: "" };
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
    `<input id="picker-query" type="text" placeholder="brief for this gap, e.g. 'the coffee stop, keep it light'">` +
    `<label class="inline">cuts <input id="picker-cuts" class="num" type="number" value="3" min="1" max="10"></label>` +
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
  const brief = $("#picker-query").value.trim();
  if (!brief) return alert("Give a brief for this gap first.");
  p.query = brief;
  p.aiStatus = "submitting…";
  renderPicker();
  $("#picker-query").value = brief;
  const syncPicker = () => {
    if (state.picker !== p) return false;
    renderPicker();
    if (p.query) $("#picker-query").value = p.query;
    return true;
  };
  try {
    const data = await api("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "suggest", project: state.project.project,
        params: {
          brief,
          target_cuts: parseInt($("#picker-cuts").value, 10) || 3,
          candidate_ids: candidates.map((c) => c.library_id),
        },
      }),
    });
    watchJob(data.job_id, "suggest", {
      onUpdate(j) {
        p.aiStatus = j.stage || j.status;
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
// Tabs + wiring
// ---------------------------------------------------------------------------
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.id === `tab-${name}`));
}

document.querySelectorAll("#tabs button").forEach((b) => {
  b.onclick = () => { if (!b.disabled) showTab(b.dataset.tab); };
});

$("#btn-analyze").onclick = () => startJob("analyze", {});
$("#btn-compose").onclick = () => startJob("compose", {
  brief: $("#brief").value.trim(),
  target_cuts: parseInt($("#target-cuts").value, 10) || 15,
  margin: parseFloat($("#margin").value) || 3,
  sources: [...state.selectedSources],
});
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
