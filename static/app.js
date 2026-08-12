"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  browsePath: window.ROOT,
  project: null,          // { project, name, pairs, library, versions }
  selectedSources: new Set(),
  edl: null,              // { version, cuts: [{cut, enabled}] }
  pollTimer: null,
  activeJob: null,
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
  $("#library-status").textContent = lib.exists
    ? `library: ${lib.chunks} chunks`
    : lib.stale ? "footage changed - re-analysis needed" : "not analyzed yet";
  $("#library-status").className = lib.exists ? "ok" : "warn";
  showTab("clips");
  renderVersions();
  await renderClips();
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
      (clip.chunks.length
        ? `<details><summary>${clip.chunks.length} chunks</summary><div class="chunks">${chunkRows}</div></details>`
        : `<div class="hint">not analyzed</div>`);
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
  state.edl = { version, cuts: data.cuts.map((cut) => ({ cut, enabled: true })) };
  $("#edl-tab").disabled = false;
  $("#edl-title").textContent = `Editing v${version} (${data.cuts.length} cuts)`;
  renderEdl();
  showTab("edl");
}

function renderEdl() {
  const tbody = $("#edl-table tbody");
  tbody.innerHTML = "";
  state.edl.cuts.forEach((row, idx) => {
    const c = row.cut;
    const tr = document.createElement("tr");
    tr.className = row.enabled ? "" : "disabled-cut";
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
      (c.speech ? `<div class="speech">&ldquo;${esc(c.speech)}&rdquo;</div>` : "") + `</td>`;
    tr.querySelector('input[type="checkbox"]').onchange = (e) => {
      row.enabled = e.target.checked;
      tr.className = row.enabled ? "" : "disabled-cut";
    };
    const [startInput, endInput] = tr.querySelectorAll(".num");
    startInput.onchange = () => { c.start = parseFloat(startInput.value); };
    endInput.onchange = () => { c.end = parseFloat(endInput.value); };
    tr.ondragstart = (e) => e.dataTransfer.setData("text/plain", idx);
    tr.ondragover = (e) => e.preventDefault();
    tr.ondrop = (e) => {
      e.preventDefault();
      const from = parseInt(e.dataTransfer.getData("text/plain"), 10);
      const to = idx;
      if (from === to) return;
      const [moved] = state.edl.cuts.splice(from, 1);
      state.edl.cuts.splice(to, 0, moved);
      renderEdl();
    };
    tbody.appendChild(tr);
  });
}

async function saveEdl() {
  const cuts = state.edl.cuts.filter((r) => r.enabled).map((r) => r.cut);
  if (!cuts.length) return alert("All cuts are disabled - nothing to save.");
  try {
    const data = await api("/api/edl", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.project.project, base_version: state.edl.version, cuts }),
    });
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

function watchJob(jobId, type) {
  state.activeJob = { id: jobId, logOffset: 0 };
  $("#job-drawer").classList.remove("hidden");
  $("#job-title").textContent = `${type} job #${jobId} - queued`;
  $("#job-log").textContent = "";
  setJobButtons(true);
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${jobId}?log_offset=${state.activeJob.logOffset}`);
      if (j.log_delta) {
        $("#job-log").textContent += j.log_delta;
        $("#job-log").scrollTop = $("#job-log").scrollHeight;
        state.activeJob.logOffset = j.log_offset;
      }
      $("#job-title").textContent = `${j.type} job #${jobId} - ${j.stage}`;
      if (j.status === "done" || j.status === "error") {
        clearInterval(state.pollTimer);
        setJobButtons(false);
        if (j.status === "error") $("#job-title").textContent += ` - FAILED: ${j.error}`;
        await refreshProject();
        if (j.status === "done" && j.result && j.result.version !== undefined
            && j.type !== "finalize") {
          showTab("versions");
        }
      }
    } catch (e) {
      clearInterval(state.pollTimer);
      setJobButtons(false);
    }
  }, 1000);
}

function setJobButtons(busy) {
  document.querySelectorAll("#btn-analyze, #btn-compose, #versions-table button")
    .forEach((b) => {
      if (["preview", "finalize"].includes(b.dataset.act) || !b.dataset.act) b.disabled = busy;
    });
}

// ---------------------------------------------------------------------------
// Video modal
// ---------------------------------------------------------------------------
function playVideo(file) {
  const url = `/api/media?path=${encodeURIComponent(state.project.project)}&file=${encodeURIComponent(file)}`;
  $("#player").src = url;
  $("#video-modal").classList.remove("hidden");
  $("#player").play().catch(() => {});
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
$("#video-close").onclick = () => {
  $("#player").pause();
  $("#player").src = "";
  $("#video-modal").classList.add("hidden");
};

// On page load, re-attach to any job still queued/running on the server
// (jobs survive refreshes - only the browser's handle is lost).
async function reconnectJobs() {
  try {
    const data = await api("/api/jobs");
    const active = data.jobs.find((j) => j.status === "running" || j.status === "queued");
    if (!active) return;
    if (!state.project) await openProject(active.project);
    watchJob(active.id, active.type);
  } catch (e) {
    /* no jobs yet or server restarted - nothing to re-attach */
  }
}

browse(window.ROOT).catch((e) => alert("Browse failed: " + e.message));
reconnectJobs();
