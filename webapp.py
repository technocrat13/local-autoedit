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
    }


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
                        "progress": None}
    JOB_QUEUE.put(job_id)
    return job_id


def set_stage(job, stage):
    with JOBS_LOCK:
        job["stage"] = stage
    print(f"--- {stage} ---")


def faststart_remux(path):
    """Move the moov atom to the front so browsers start playback instantly."""
    ffmpeg, _ = engine.get_media_tools()
    if not ffmpeg or not os.path.exists(path):
        return
    tmp = path + ".faststart.mp4"
    result = subprocess.run(
        [ffmpeg, "-y", "-i", path, "-c", "copy", "-movflags", "+faststart", tmp],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, path)
    elif os.path.exists(tmp):
        os.remove(tmp)


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
    with JOBS_LOCK:
        job["progress"] = {"done": done, "current": None,
                           "total": len(pairs), "chunks": len(library.chunks)}
    transcriber.unload()
    if not library.chunks:
        raise RuntimeError("No analyzable chunks found in the footage.")
    return {"chunks": len(library.chunks)}


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
    target_cuts = int(params.get("target_cuts") or 15)
    margin = max(2.0, min(5.0, float(params.get("margin") or 3.0)))

    engine.load_model()
    set_stage(job, f"composing story from {len(library.chunks)} chunks")
    story = engine.StoryAgent().parse_user_brief(brief)
    selections = engine.StoryComposer().compose(library, story, target_cuts)
    if not selections:
        print("StoryComposer output unusable - falling back to deterministic scoring.")
        selections = engine.FallbackSelector().select(library, story, target_cuts)
    if not selections:
        raise RuntimeError("No clips chosen - try a different brief.")

    durations = {c["source"]: c["clip_duration"] for c in library.chunks}
    lrf_by_source = {os.path.basename(p["hires"]): p["lrf"] for p in pairs}
    editor = engine.FinalEditor(WORKDIR)
    cuts = editor.build_cuts(selections, margin, durations, lrf_by_source)
    edit_plan = editor.merge_overlapping(cuts)

    version = add_version(project, "compose", {
        "parent": None, "brief": brief, "target_cuts": target_cuts,
        "margin": margin, "sources": sources, "cut_count": len(edit_plan)})
    engine.FinalEditor.write_edl(edit_plan, edl_path(project, version))

    set_stage(job, f"rendering LRF preview v{version}")
    preview = render_path(project, version, "preview")
    if not editor.render(edit_plan, preview, **PREVIEW_KWARGS):
        raise RuntimeError("Preview render produced no clips.")
    faststart_remux(preview)
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
    if kind == "preview":
        faststart_remux(out)
    return {"version": version, "output": out}


def job_suggest(job):
    """Pick chunks matching a free-text query from a candidate window - used by
    the EDL editor's add-clip picker to recover moments the first pass missed."""
    project = job["project"]
    params = job["params"]
    paths = project_paths(project)
    pairs = engine.get_video_pairs(project)
    library, _ = engine.ChunkLibrary.load(paths["library"], pairs)
    candidate_ids = {int(i) for i in params.get("candidate_ids") or []}
    candidates = [c for c in library.chunks if c["library_id"] in candidate_ids]
    if not candidates:
        raise RuntimeError("No analyzed chunks in this window to pick from.")
    query = (params.get("query") or "").strip()
    if not query:
        raise RuntimeError("Describe the moment you want to add.")
    max_picks = max(1, min(10, int(params.get("max_picks") or 3)))

    engine.load_model()
    set_stage(job, f"searching {len(candidates)} chunks")
    digest = "\n".join(engine.StoryComposer._digest_line(c) for c in candidates)
    user = (
        "An editor wants to add a small moment to an existing vlog edit. Their "
        f'request: "{query}"\n'
        f"Below are the available shots in chronological order. Pick up to "
        f"{max_picks} shots that best match the request - fewer is fine, and "
        "pick none if nothing matches.\n"
        "Rules:\n"
        "- Only use id values that appear below.\n"
        "- Prefer shots whose summary, tags, or speech directly match the request.\n"
        'Output ONLY a JSON object: {"selections": [list of objects, each with '
        '"chunk_id" (id number) and "reason" (one short sentence)]}.\n\n'
        f"SHOTS:\n{digest}"
    )
    parsed = engine.parse_json_response(
        engine._llm_text(engine.StoryComposer.SYSTEM, user, 400))
    picks = []
    seen = set()
    if isinstance(parsed, dict) and isinstance(parsed.get("selections"), list):
        for item in parsed["selections"]:
            if not isinstance(item, dict):
                continue
            try:
                library_id = int(item.get("chunk_id"))
            except (TypeError, ValueError):
                continue
            if library_id in candidate_ids and library_id not in seen:
                seen.add(library_id)
                picks.append({"library_id": library_id,
                              "reason": str(item.get("reason") or "")})
            if len(picks) >= max_picks:
                break
    print(f"Suggest: {len(picks)} pick(s) for \"{query}\".")
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
        log = job["log"]
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = JOB_HANDLERS[job["type"]](job)
            with JOBS_LOCK:
                job["status"] = "done"
                job["stage"] = "done"
                job["result"] = result
        except Exception as e:  # job errors must never kill the worker
            log.write(f"\nERROR: {e}\n")
            with JOBS_LOCK:
                job["status"] = "error"
                job["stage"] = "error"
                job["error"] = str(e)


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
    return jsonify({
        "project": project,
        "name": os.path.basename(project),
        "pairs": [{"source": os.path.basename(p["hires"]),
                   "lrf": p["lrf"], "hires": p["hires"]} for p in pairs],
        "library": {"exists": bool(pairs) and analyzed == total,
                    "partial": 0 < analyzed < total,
                    "analyzed_clips": analyzed, "total_clips": total,
                    "changed_clips": changed,
                    "chunks": len(library.chunks)},
        "versions": versions_with_files(project),
    })


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
                        "log_delta": text[offset:], "log_offset": len(text)})


@app.get("/api/jobs")
def api_jobs_list():
    with JOBS_LOCK:
        busy = any(j["status"] in ("queued", "running") for j in JOBS.values())
        jobs = [{"id": j["id"], "type": j["type"], "status": j["status"],
                 "stage": j["stage"], "project": j["project"]}
                for j in sorted(JOBS.values(), key=lambda j: j["id"], reverse=True)[:20]]
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
