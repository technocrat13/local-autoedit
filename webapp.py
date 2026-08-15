"""Local single-user web GUI for the auto-editor.

Flask serves a single page; all GPU work (analyze / compose / render) runs on ONE
background worker thread so HTTP requests never block and VRAM is never contended.
Each project folder gets a visible `autoedit/` subfolder holding the chunk library,
versioned EDLs, previews, thumbnails, final renders, and a state.json index.
"""
import os
import io
import json
import queue
import argparse
import threading
import subprocess
import contextlib
from datetime import datetime

from flask import Flask, jsonify, request, render_template, send_file, abort

import betterautoeditor as engine

app = Flask(__name__)

ROOT = None  # set in main(); every path from the client must resolve inside it

AUTOEDIT_DIR = "autoedit"
STATE_SCHEMA = 1

# ----------------------------------------------------------------------------
# Path safety
# ----------------------------------------------------------------------------
def safe_path(raw):
    """Resolve a client-supplied path and require it to be inside ROOT."""
    if not raw:
        abort(400, "missing path")
    resolved = os.path.realpath(raw)
    if resolved != ROOT and not resolved.startswith(ROOT + os.sep):
        abort(403, "path outside the configured root")
    return resolved


def project_paths(project):
    ae = os.path.join(project, AUTOEDIT_DIR)
    return {
        "autoedit": ae,
        "library": os.path.join(ae, "library.json"),
        "state": os.path.join(ae, "state.json"),
        "thumbs": os.path.join(ae, "thumbs"),
        "jobs": os.path.join(ae, "jobs.json"),
        "kg": os.path.join(ae, "kg.json"),
    }


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def edl_path(project, version):
    return os.path.join(project, AUTOEDIT_DIR, f"edl_v{version}.json")


def render_path(project, version, kind):
    return os.path.join(project, AUTOEDIT_DIR, f"{kind}_v{version}.mp4")


# ----------------------------------------------------------------------------
# state.json - version index per project
# ----------------------------------------------------------------------------
STATE_LOCK = threading.Lock()


def load_state(project):
    path = project_paths(project)["state"]
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("schema") == STATE_SCHEMA:
                return state
        except (OSError, json.JSONDecodeError):
            pass
    return {"schema": STATE_SCHEMA, "next_version": 1, "versions": []}


def save_state(project, state):
    paths = project_paths(project)
    os.makedirs(paths["autoedit"], exist_ok=True)
    tmp = paths["state"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, paths["state"])


def add_version(project, origin, meta):
    with STATE_LOCK:
        state = load_state(project)
        version = state["next_version"]
        state["next_version"] = version + 1
        entry = {"version": version, "created": datetime.now().isoformat(timespec="seconds"),
                 "origin": origin, **meta}
        state["versions"].append(entry)
        save_state(project, state)
    return version


def versions_with_files(project):
    state = load_state(project)
    out = []
    for entry in state["versions"]:
        v = entry["version"]
        out.append({**entry,
                    "has_edl": os.path.exists(edl_path(project, v)),
                    "has_preview": os.path.exists(render_path(project, v, "preview")),
                    "has_final": os.path.exists(render_path(project, v, "final"))})
    return out


# ----------------------------------------------------------------------------
# Clip-subset view over the library (global library_ids preserved)
# ----------------------------------------------------------------------------
class LibraryView(engine.ChunkLibrary):
    # Base by_id indexes by list position, which is wrong on a filtered list.
    def __init__(self, chunks):
        super().__init__()
        self.chunks = list(chunks)
        self._map = {c["library_id"]: c for c in self.chunks}

    def by_id(self, library_id):
        return self._map.get(library_id)


# ----------------------------------------------------------------------------
# Job worker - one thread, one queue, sequential GPU jobs
# ----------------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_QUEUE = queue.Queue()
_job_counter = 0
WORKDIR = None  # created in main(); shared across jobs (sequential worker)


class _JobLog(io.StringIO):
    """StringIO whose writes are visible mid-job under the jobs lock."""


def submit_job(job_type, project, params):
    global _job_counter
    with JOBS_LOCK:
        _job_counter += 1
        job_id = _job_counter
        JOBS[job_id] = {"id": job_id, "type": job_type, "project": project,
                        "params": params, "status": "queued", "stage": "queued",
                        "log": _JobLog(), "result": None, "error": None,
                        "progress": None, "created": now_iso(), "started": None,
                        "finished": None, "summary": None}
    JOB_QUEUE.put(job_id)
    return job_id


def set_stage(job, stage):
    with JOBS_LOCK:
        job["stage"] = stage
    print(f"--- {stage} ---")


PREVIEW_KWARGS = {"source_key": "lrf_file", "preset": "ultrafast",
                  "bitrate": engine.PREVIEW_BITRATE}


def job_analyze(job):
    project = job["project"]
    paths = project_paths(project)
    pairs = engine.get_video_pairs(project)
    if not pairs:
        raise RuntimeError("No LRF/MP4 pairs found in this folder.")
    engine.load_model()
    os.makedirs(paths["autoedit"], exist_ok=True)
    library, status = engine.ChunkLibrary.load(paths["library"], pairs)
    kg_store, _ = engine.KnowledgeGraphStore.load(paths["kg"], pairs)
    preprocessor = engine.VideoPreprocessor(WORKDIR)
    analyzer = engine.ChunkAnalyzer()
    transcriber = engine.SpeechTranscriber()
    done = sorted(s for s, st in status.items() if st == "analyzed")
    for i, pair in enumerate(pairs):
        source = os.path.basename(pair["hires"])
        if status[source] == "analyzed":
            continue
        with JOBS_LOCK:
            job["progress"] = {"done": list(done), "current": source,
                               "total": len(pairs), "chunks": len(library.chunks)}
        set_stage(job, f"analyzing {source} ({i + 1}/{len(pairs)})")
        engine.analyze_clip(pair, preprocessor, analyzer, transcriber,
                            library, WORKDIR, None)
        library.commit_clip(pair)
        done = sorted(done + [source])
        # Incremental checkpoint: a stopped job resumes from the next clip.
        library.save(paths["library"])
        set_stage(job, f"graphing {source} ({i + 1}/{len(pairs)})")
        # Extracts this clip's entities; also backfills any earlier clip
        # analyzed before the knowledge graph existed.
        engine.sync_knowledge_graph(kg_store, paths["kg"], library, pairs)
    with JOBS_LOCK:
        job["progress"] = {"done": done, "current": None,
                           "total": len(pairs), "chunks": len(library.chunks)}
    transcriber.unload()
    if not library.chunks:
        raise RuntimeError("No analyzable chunks found in the footage.")
    # All clips may have been analyzed already - still bring the graph current.
    set_stage(job, "syncing knowledge graph")
    engine.sync_knowledge_graph(kg_store, paths["kg"], library, pairs)
    return {"chunks": len(library.chunks),
            "kg_clips": len(kg_store.clip_entries)}


def job_compose(job):
    project = job["project"]
    params = job["params"]
    paths = project_paths(project)
    pairs = engine.get_video_pairs(project)
    library, status = engine.ChunkLibrary.load(paths["library"], pairs)
    if not library.chunks:
        raise RuntimeError("No analyzed clips yet - run Analyze first.")

    sources = params.get("sources") or []
    if sources:
        pending = [s for s in sources if status.get(s) != "analyzed"]
        if pending:
            raise RuntimeError("Selected clips not analyzed yet: "
                               + ", ".join(pending))
        chunks = [c for c in library.chunks if c["source"] in sources]
        if not chunks:
            raise RuntimeError("Selected clips have no analyzed chunks.")
        library = LibraryView(chunks)
    else:
        pending = [s for s, st in status.items() if st != "analyzed"]
        if pending:
            print(f"Note: composing from analyzed clips only - "
                  f"{len(pending)} clip(s) still pending analysis.")
    brief = params.get("brief") or engine.DEFAULT_BRIEF
    target_len = params.get("target_len")
    target_len = float(target_len) if target_len else None
    target_cuts = params.get("target_cuts")
    target_cuts = int(target_cuts) if target_cuts else None
    margin = max(2.0, min(5.0, float(params.get("margin") or 3.0)))

    engine.load_model()
    set_stage(job, f"composing story from {len(library.chunks)} chunks")
    story = engine.StoryAgent().parse_user_brief(brief)
    composer = engine.StoryComposer()
    selections = []
    if target_cuts is None:
        kg_store, _ = engine.KnowledgeGraphStore.load(paths["kg"], pairs)
        graph = engine.KnowledgeGraph.build(kg_store, library)
        if graph.entities:
            selections = composer.compose_with_graph(
                library, story, graph, target_len=target_len, margin=margin)
            if not selections:
                print("Graph composition unusable - trying the legacy composer.")
        else:
            print("No knowledge graph yet - using the legacy composer "
                  "(re-run Analyze to build one).")
    if not selections:
        selections = composer.compose(library, story, target_cuts or 15)
    if not selections:
        print("StoryComposer output unusable - falling back to deterministic scoring.")
        selections = engine.FallbackSelector().select(library, story, target_cuts or 15)
    if not selections:
        raise RuntimeError("No clips chosen - try a different brief.")

    durations = {c["source"]: c["clip_duration"] for c in library.chunks}
    lrf_by_source = {os.path.basename(p["hires"]): p["lrf"] for p in pairs}
    editor = engine.FinalEditor(WORKDIR)
    cuts = editor.build_cuts(selections, margin, durations, lrf_by_source)
    edit_plan = editor.merge_overlapping(cuts)

    version = add_version(project, "compose", {
        "parent": None, "brief": brief, "target_cuts": target_cuts,
        "target_len": target_len,
        "margin": margin, "sources": sources, "cut_count": len(edit_plan)})
    engine.FinalEditor.write_edl(edit_plan, edl_path(project, version))

    set_stage(job, f"rendering LRF preview v{version}")
    preview = render_path(project, version, "preview")
    if not editor.render(edit_plan, preview, **PREVIEW_KWARGS):
        raise RuntimeError("Preview render produced no clips.")
    return {"version": version, "cut_count": len(edit_plan)}


def job_render(job, kind):
    project = job["project"]
    version = int(job["params"]["version"])
    src_edl = edl_path(project, version)
    out = render_path(project, version, kind)
    set_stage(job, f"rendering {kind} v{version}")
    kwargs = PREVIEW_KWARGS if kind == "preview" else {}
    if not engine.render_from_edl(src_edl, out, WORKDIR, **kwargs):
        raise RuntimeError(f"Render from {os.path.basename(src_edl)} failed.")
    return {"version": version, "output": out}


def job_suggest(job):
    """Gap-fill: run the normal compose pipeline (brief -> story -> composer,
    with fallback) on just the chunks between two cuts, so the EDL editor can
    weave a small video into a gap the first pass missed."""
    project = job["project"]
    params = job["params"]
    paths = project_paths(project)
    pairs = engine.get_video_pairs(project)
    library, _ = engine.ChunkLibrary.load(paths["library"], pairs)
    candidate_ids = {int(i) for i in params.get("candidate_ids") or []}
    chunks = [c for c in library.chunks if c["library_id"] in candidate_ids]
    if not chunks:
        raise RuntimeError("No analyzed chunks in this window to pick from.")
    brief = (params.get("brief") or "").strip()
    if not brief:
        raise RuntimeError("Describe what should fill this gap.")
    target_cuts = params.get("target_cuts")
    target_cuts = max(1, min(10, int(target_cuts))) if target_cuts else None
    view = LibraryView(chunks)

    engine.load_model()
    set_stage(job, f"composing gap fill from {len(chunks)} chunks")
    story = engine.StoryAgent().parse_user_brief(brief)
    composer = engine.StoryComposer()
    selections = []
    kg_store, _ = engine.KnowledgeGraphStore.load(paths["kg"], pairs)
    subgraph = engine.KnowledgeGraph.build(kg_store, library).subgraph(candidate_ids)
    if subgraph.entities:
        selections = composer.compose_with_graph(
            view, story, subgraph, max_total=target_cuts or 10)
    if not selections:
        if subgraph.entities:
            print("Graph gap fill unusable - trying the legacy composer.")
        selections = composer.compose(view, story, target_cuts or 3)
    if not selections:
        print("StoryComposer output unusable - falling back to deterministic scoring.")
        selections = engine.FallbackSelector().select(view, story, target_cuts or 3)
    if not selections:
        raise RuntimeError("Nothing matched - try a different brief.")

    selections.sort(key=lambda s: (s["chunk"]["source"], s["chunk"]["start"]))
    picks = [{"library_id": s["chunk"]["library_id"], "role": s["role"],
              "reason": s.get("reason", ""), "beat": s.get("beat", "")}
             for s in selections[:target_cuts or 10]]
    print(f"Gap fill: {len(picks)} pick(s) for \"{brief}\".")
    return {"picks": picks}


JOB_HANDLERS = {
    "analyze": job_analyze,
    "compose": job_compose,
    "preview": lambda job: job_render(job, "preview"),
    "finalize": lambda job: job_render(job, "final"),
    "suggest": job_suggest,
}


def worker_loop():
    while True:
        job_id = JOB_QUEUE.get()
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "running"
            job["stage"] = "starting"
        execute_job(job)


def execute_job(job):
    """Run a job to completion, stamp timestamps/summary, persist history."""
    log = job["log"]
    with JOBS_LOCK:
        job["started"] = now_iso()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = JOB_HANDLERS[job["type"]](job)
        with JOBS_LOCK:
            job["status"] = "done"
            job["stage"] = "done"
            job["result"] = result
            job["finished"] = now_iso()
            job["summary"] = summarize_result(job["type"], result)
    except Exception as e:  # job errors must never kill the worker
        log.write(f"\nERROR: {e}\n")
        with JOBS_LOCK:
            job["status"] = "error"
            job["stage"] = "error"
            job["error"] = str(e)
            job["finished"] = now_iso()
    persist_job(job)


def summarize_result(job_type, result):
    """One-line human summary of a finished job; persists into history."""
    try:
        if job_type == "analyze":
            return f"{result['chunks']} chunks in library"
        if job_type == "compose":
            return f"EDL v{result['version']}, {result['cut_count']} cuts"
        if job_type in ("preview", "finalize"):
            return os.path.basename(result["output"])
        if job_type == "suggest":
            return f"{len(result['picks'])} pick(s)"
    except (TypeError, KeyError):
        pass
    return ""


def summarize_params(job_type, params):
    try:
        if job_type in ("compose", "suggest"):
            brief = (params.get("brief") or "").strip()
            if len(brief) > 60:
                brief = brief[:57] + "..."
            if params.get("target_len"):
                extra = f", ~{float(params['target_len']) / 60:.0f}m"
            elif params.get("target_cuts"):
                extra = f", {params['target_cuts']} cuts"
            else:
                extra = ""
            return f"brief: {brief}{extra}" if brief else ""
        if job_type in ("preview", "finalize"):
            return f"version {params.get('version')}"
    except (TypeError, AttributeError):
        pass
    return ""


JOBS_HISTORY_SCHEMA = 1
JOBS_HISTORY_MAX = 50
JOBS_LOG_MAX_CHARS = 20_000
HISTORY_LOCK = threading.Lock()


def load_job_history(project):
    path = project_paths(project)["jobs"]
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history.get("schema") == JOBS_HISTORY_SCHEMA:
                return history
        except (OSError, json.JSONDecodeError):
            pass
    return {"schema": JOBS_HISTORY_SCHEMA, "jobs": []}


def persist_job(job):
    """Append a finished job to the project's jobs.json (best-effort)."""
    try:
        with JOBS_LOCK:
            record = {"id": job["id"], "type": job["type"], "status": job["status"],
                      "created": job["created"], "started": job["started"],
                      "finished": job["finished"],
                      "params_summary": summarize_params(job["type"], job["params"]),
                      "summary": job["summary"], "error": job["error"],
                      "log": job["log"].getvalue()[-JOBS_LOG_MAX_CHARS:]}
            project = job["project"]
        with HISTORY_LOCK:
            paths = project_paths(project)
            os.makedirs(paths["autoedit"], exist_ok=True)
            history = load_job_history(project)
            history["jobs"].append(record)
            history["jobs"] = history["jobs"][-JOBS_HISTORY_MAX:]
            tmp = paths["jobs"] + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            os.replace(tmp, paths["jobs"])
    except OSError as e:
        print(f"Could not persist job history: {e}")


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html", root=ROOT)


@app.get("/api/browse")
def api_browse():
    path = safe_path(request.args.get("path") or ROOT)
    if not os.path.isdir(path):
        abort(404, "not a directory")
    dirs = []
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name.lower())
    except OSError as e:
        abort(400, str(e))
    for entry in entries:
        if entry.is_dir() and not entry.name.startswith(".") and entry.name != AUTOEDIT_DIR:
            sub = entry.path
            dirs.append({"name": entry.name, "path": sub,
                         "has_pairs": bool(engine.get_video_pairs(sub)),
                         "has_autoedit": os.path.isdir(os.path.join(sub, AUTOEDIT_DIR))})
    parent = os.path.dirname(path) if path != ROOT else None
    return jsonify({"path": path, "parent": parent, "dirs": dirs})


@app.post("/api/project/open")
def api_project_open():
    project = safe_path((request.json or {}).get("path"))
    pairs = engine.get_video_pairs(project)
    paths = project_paths(project)
    library, status = (engine.ChunkLibrary.load(paths["library"], pairs)
                       if pairs else (engine.ChunkLibrary(), {}))
    total = len(pairs)
    analyzed = sum(1 for st in status.values() if st == "analyzed")
    changed = sum(1 for st in status.values() if st == "changed")
    kg_store, kg_status = (engine.KnowledgeGraphStore.load(paths["kg"], pairs)
                           if pairs else (engine.KnowledgeGraphStore(), {}))
    kg_clips = sum(1 for st in kg_status.values() if st == "analyzed")
    return jsonify({
        "project": project,
        "name": os.path.basename(project),
        "pairs": [{"source": os.path.basename(p["hires"]),
                   "lrf": p["lrf"], "hires": p["hires"]} for p in pairs],
        "library": {"exists": bool(pairs) and analyzed == total,
                    "partial": 0 < analyzed < total,
                    "analyzed_clips": analyzed, "total_clips": total,
                    "changed_clips": changed,
                    "chunks": len(library.chunks),
                    "kg_clips": kg_clips},
        "versions": versions_with_files(project),
    })


@app.get("/api/project/graph")
def api_project_graph():
    project = safe_path(request.args.get("path"))
    pairs = engine.get_video_pairs(project)
    paths = project_paths(project)
    library, _ = engine.ChunkLibrary.load(paths["library"], pairs)
    kg_store, _ = engine.KnowledgeGraphStore.load(paths["kg"], pairs)
    graph = engine.KnowledgeGraph.build(kg_store, library)
    return jsonify(graph.to_json(library))


@app.get("/api/project/clips")
def api_project_clips():
    project = safe_path(request.args.get("path"))
    pairs = engine.get_video_pairs(project)
    paths = project_paths(project)
    library, status = (engine.ChunkLibrary.load(paths["library"], pairs)
                       if pairs else (engine.ChunkLibrary(), {}))
    by_source = {}
    for c in library.chunks:
        by_source.setdefault(c["source"], []).append({
            "library_id": c["library_id"], "start": c["start"], "end": c["end"],
            "scene": c.get("scene", ""), "summary": c.get("one_line_summary", ""),
            "interest": c.get("interest_score", 0.0),
            "transcript": c.get("speech", {}).get("transcript", "")})
    clips = []
    for p in pairs:
        source = os.path.basename(p["hires"])
        chunks = by_source.get(source, [])
        clips.append({"source": source,
                      "duration": chunks[-1]["end"] if chunks else None,
                      "analyzed": status.get(source) == "analyzed",
                      "changed": status.get(source) == "changed",
                      "chunks": chunks})
    analyzed = sum(1 for c in clips if c["analyzed"])
    return jsonify({"clips": clips, "analyzed_clips": analyzed,
                    "total_clips": len(pairs)})


@app.get("/api/thumb")
def api_thumb():
    project = safe_path(request.args.get("path"))
    clip = os.path.basename(request.args.get("clip") or "")
    pair = next((p for p in engine.get_video_pairs(project)
                 if os.path.basename(p["hires"]) == clip), None)
    if pair is None:
        abort(404, "unknown clip")
    paths = project_paths(project)
    thumb = os.path.join(paths["thumbs"], os.path.splitext(clip)[0] + ".jpg")
    if not os.path.exists(thumb):
        os.makedirs(paths["thumbs"], exist_ok=True)
        ffmpeg, _ = engine.get_media_tools()
        if not ffmpeg:
            abort(500, "ffmpeg not found")
        duration = engine.VideoPreprocessor.probe_duration(pair["lrf"])
        subprocess.run(
            [ffmpeg, "-y", "-ss", f"{max(0.0, duration / 2):.1f}", "-i", pair["lrf"],
             "-frames:v", "1", "-vf", "scale=320:-2", thumb],
            capture_output=True, check=False,
        )
    if not os.path.exists(thumb):
        abort(500, "thumbnail generation failed")
    return send_file(thumb, mimetype="image/jpeg", conditional=True)


@app.post("/api/jobs")
def api_jobs_create():
    payload = request.json or {}
    job_type = payload.get("type")
    if job_type not in JOB_HANDLERS:
        abort(400, f"unknown job type: {job_type}")
    project = safe_path(payload.get("project"))
    params = payload.get("params") or {}
    if job_type in ("preview", "finalize"):
        try:
            version = int(params.get("version"))
        except (TypeError, ValueError):
            abort(400, "preview/finalize jobs need a version")
        if not os.path.exists(edl_path(project, version)):
            abort(404, f"no EDL for version {version}")
    job_id = submit_job(job_type, project, params)
    return jsonify({"job_id": job_id})


@app.get("/api/jobs/<int:job_id>")
def api_jobs_get(job_id):
    offset = int(request.args.get("log_offset") or 0)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404, "unknown job")
        text = job["log"].getvalue()
        return jsonify({"id": job_id, "type": job["type"], "status": job["status"],
                        "stage": job["stage"], "result": job["result"],
                        "error": job["error"], "progress": job["progress"],
                        "created": job["created"], "started": job["started"],
                        "finished": job["finished"], "summary": job["summary"],
                        "log_delta": text[offset:], "log_offset": len(text)})


@app.get("/api/jobs")
def api_jobs_list():
    path = request.args.get("path")
    project = safe_path(path) if path else None
    with JOBS_LOCK:
        busy = any(j["status"] in ("queued", "running") for j in JOBS.values())
        mem = [{"id": j["id"], "type": j["type"], "status": j["status"],
                "stage": j["stage"], "project": j["project"],
                "created": j["created"], "started": j["started"],
                "finished": j["finished"], "summary": j["summary"],
                "error": j["error"]}
               for j in sorted(JOBS.values(), key=lambda j: j["id"], reverse=True)
               if project is None or j["project"] == project][:20]
    jobs = mem
    if project is not None:
        seen = {(j["id"], j["finished"]) for j in mem}
        disk = [dict(entry, project=project, from_history=True)
                for entry in reversed(load_job_history(project)["jobs"])
                if (entry["id"], entry["finished"]) not in seen]
        jobs = mem + disk
    return jsonify({"busy": busy, "jobs": jobs})


@app.get("/api/edl")
def api_edl_get():
    project = safe_path(request.args.get("path"))
    try:
        version = int(request.args.get("version"))
    except (TypeError, ValueError):
        abort(400, "version required")
    path = edl_path(project, version)
    if not os.path.exists(path):
        abort(404, "no such EDL version")
    with open(path, "r", encoding="utf-8") as f:
        cuts = json.load(f)
    meta = next((v for v in versions_with_files(project) if v["version"] == version), None)
    return jsonify({"version": version, "cuts": cuts, "meta": meta})


@app.post("/api/edl")
def api_edl_save():
    payload = request.json or {}
    project = safe_path(payload.get("path"))
    cuts = payload.get("cuts")
    if not isinstance(cuts, list) or not cuts:
        abort(400, "cuts must be a non-empty list")
    cleaned = []
    for cut in cuts:
        if not isinstance(cut, dict):
            abort(400, "each cut must be an object")
        try:
            cut["start"] = float(cut["start"])
            cut["end"] = float(cut["end"])
        except (KeyError, TypeError, ValueError):
            abort(400, "cuts need numeric start/end")
        if cut["end"] <= cut["start"]:
            abort(400, f"cut has end <= start ({cut['start']}-{cut['end']})")
        cleaned.append(cut)
    try:
        parent = int(payload.get("base_version"))
    except (TypeError, ValueError):
        parent = None
    version = add_version(project, "manual_edit", {
        "parent": parent, "brief": "", "target_cuts": None, "margin": None,
        "sources": [], "cut_count": len(cleaned)})
    engine.FinalEditor.write_edl(cleaned, edl_path(project, version))
    return jsonify({"version": version, "cut_count": len(cleaned)})


@app.get("/api/media")
def api_media():
    project = safe_path(request.args.get("path"))
    name = os.path.basename(request.args.get("file") or "")
    candidates = [os.path.join(project, AUTOEDIT_DIR, name), os.path.join(project, name)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return send_file(candidate, conditional=True)
    abort(404, "file not found")


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------
def main():
    global ROOT, WORKDIR
    parser = argparse.ArgumentParser(description="Web GUI for the auto-editor.")
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(engine.FOOTAGE_DIR)),
                        help="Top folder the UI is allowed to browse (default: parent of FOOTAGE_DIR).")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    ROOT = os.path.realpath(args.root)
    WORKDIR = engine.WorkDir()

    threading.Thread(target=worker_loop, daemon=True, name="gpu-worker").start()
    print(f"Auto-editor web GUI: http://127.0.0.1:{args.port}  (root: {ROOT})")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
