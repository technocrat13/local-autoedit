"""Stubbed sanity test for the v6 Flask web GUI (macOS, no GPU / heavy deps).
Fakes the heavy modules, builds a temp project with fake LRF/MP4 pairs, and
drives the API with Flask's test_client."""
import sys
import json
import time
import types
import tempfile
import os

# ---- stub heavy modules before importing engine/webapp ----------------------
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

class _FakeNp(types.ModuleType):
    def __getattr__(self, name):
        return type(name, (), {})

np_stub = _FakeNp("numpy")
np_stub.asarray = lambda x, dtype=None: x
np_stub.int64 = int
np_stub.bool_ = bool
np_stub.integer = int
np_stub.floating = float
sys.modules["numpy"] = np_stub

RENDERS = []
OPENED = []

class FakeClip:
    duration = 10_000.0
    def subclipped(self, start, end):
        return ("sub", start, end)
    def close(self):
        pass

class FakeFinal:
    def write_videofile(self, output_path, **kwargs):
        RENDERS.append({"output": output_path, **kwargs})
        with open(output_path, "wb") as f:
            f.write(b"\x00" * 4096)  # real bytes so /api/media can serve it
    def close(self):
        pass

_stub("torch", cuda=types.SimpleNamespace(empty_cache=lambda: None,
                                          is_available=lambda: False))
_stub("moviepy", VideoFileClip=lambda src: (OPENED.append(src), FakeClip())[1],
      concatenate_videoclips=lambda clips, method=None: FakeFinal())
_stub("transformers", Qwen3_5ForConditionalGeneration=object, AutoProcessor=object,
      BitsAndBytesConfig=object)
_stub("faster_whisper", WhisperModel=None)
tc = _stub("torchcodec")
_stub("torchcodec.decoders", VideoDecoder=type("VD", (), {"get_frames_at": lambda self, i: None}))
tc.decoders = sys.modules["torchcodec.decoders"]

import betterautoeditor as ae
import webapp

failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)

# ---- fake filesystem: root/dayA project with 2 fake pairs -------------------
root = os.path.realpath(tempfile.mkdtemp(prefix="v6root_"))
project = os.path.join(root, "dayA")
os.makedirs(project)
for stem in ("DJI_0001", "DJI_0002"):
    for ext in (".LRF", ".MP4"):
        with open(os.path.join(project, stem + ext), "wb") as f:
            f.write(b"\x00" * 128)

webapp.ROOT = root
webapp.WORKDIR = ae.WorkDir()
webapp.faststart_remux = lambda path: None  # no real ffmpeg on this box

# fake library on disk matching the pairs' fingerprint
pairs = ae.get_video_pairs(project)
check("pairs discovered", len(pairs) == 2)
lib = ae.ChunkLibrary()
for i in range(8):
    src = "DJI_0001.MP4" if i < 4 else "DJI_0002.MP4"
    lib.add({"source": src, "hires_path": os.path.join(project, src),
             "start": (i % 4) * 5.0, "end": (i % 4) * 5.0 + 5, "scene": "beach",
             "visual_tags": ["x"], "actions": [], "people_count": 1, "emotions": [],
             "transition_flag": False, "one_line_summary": f"moment {i}",
             "interest_score": 0.5 + 0.05 * i, "clip_duration": 20.0,
             "audio": {"loudness_peak_db": -12.0},
             "speech": {"transcript": "", "has_speech": False, "language": "en"}})
paths = webapp.project_paths(project)
os.makedirs(paths["autoedit"], exist_ok=True)
lib.save(paths["library"], pairs)

# fake compose: record the library object it saw, return chunk 0 + 5
SEEN = {}
def fake_compose(self, library, story, target_cuts):
    SEEN["library"] = library
    SEEN["brief"] = story["brief"]
    ids = [c["library_id"] for c in library.chunks]
    return [{"chunk": library.by_id(ids[0]), "role": "setting", "reason": "r"},
            {"chunk": library.by_id(ids[-1]), "role": "peak", "reason": "r2"}]
ae.StoryComposer.compose = fake_compose
ae.StoryAgent.parse_user_brief = lambda self, brief: {
    "brief": brief, "themes": [], "emotional_targets": [], "must_capture": [], "avoid": []}
ae.load_model = lambda: None

client = webapp.app.test_client()

def run_job(payload):
    """Submit a job and run the worker synchronously for determinism."""
    res = client.post("/api/jobs", json=payload)
    assert res.status_code == 200, res.get_data(as_text=True)
    job_id = res.get_json()["job_id"]
    jid = webapp.JOB_QUEUE.get()
    job = webapp.JOBS[jid]
    job["status"] = "running"
    import contextlib
    try:
        with contextlib.redirect_stdout(job["log"]):
            job["result"] = webapp.JOB_HANDLERS[job["type"]](job)
        job["status"] = "done"
    except Exception as e:
        job["log"].write(f"\nERROR: {e}\n")
        job["status"] = "error"
        job["error"] = str(e)
    return client.get(f"/api/jobs/{job_id}").get_json()

# ---- 1. path safety ----------------------------------------------------------
check("browse root ok", client.get(f"/api/browse?path={root}").status_code == 200)
check("traversal .. rejected",
      client.get(f"/api/browse?path={root}/../..").status_code == 403)
check("absolute outside root rejected",
      client.get("/api/browse?path=/etc").status_code == 403)
data = client.get(f"/api/browse?path={root}").get_json()
check("browse flags project folder",
      any(d["name"] == "dayA" and d["has_pairs"] for d in data["dirs"]))

# ---- 2. project open ----------------------------------------------------------
data = client.post("/api/project/open", json={"path": project}).get_json()
check("project open reports library", data["library"]["exists"]
      and data["library"]["chunks"] == 8 and data["versions"] == [])
clips = client.get(f"/api/project/clips?path={project}").get_json()
check("clips grouped per source",
      len(clips["clips"]) == 2 and len(clips["clips"][0]["chunks"]) == 4)

# ---- 3. compose job with clip subset ------------------------------------------
OPENED.clear(); RENDERS.clear()
j = run_job({"type": "compose", "project": project,
             "params": {"brief": "beach only", "target_cuts": 10, "margin": 3,
                        "sources": ["DJI_0002.MP4"]}})
check("compose job done", j["status"] == "done" and j["result"]["version"] == 1)
check("composer saw only subset chunks",
      isinstance(SEEN["library"], webapp.LibraryView)
      and all(c["source"] == "DJI_0002.MP4" for c in SEEN["library"].chunks)
      and SEEN["library"].by_id(4) is not None      # global ids preserved
      and SEEN["library"].by_id(0) is None)          # excluded source
check("tailored brief reached composer", SEEN["brief"] == "beach only")
check("EDL v1 written", os.path.exists(webapp.edl_path(project, 1)))
check("preview v1 rendered from LRF",
      RENDERS and RENDERS[-1]["output"].endswith("preview_v1.mp4")
      and RENDERS[-1].get("preset") == "ultrafast"
      and all(o.endswith(".LRF") for o in OPENED))
check("job log captured engine prints", "EDL written" in j["log_delta"])
state = webapp.load_state(project)
check("state.json tracks version", state["versions"][0]["version"] == 1
      and state["versions"][0]["sources"] == ["DJI_0002.MP4"])

# ---- 4. manual EDL edit -> v2 --------------------------------------------------
with open(webapp.edl_path(project, 1)) as f:
    v1_cuts = json.load(f)
v1_raw = open(webapp.edl_path(project, 1)).read()
edited = [dict(v1_cuts[0], start=1.0, end=4.5)]  # dropped one cut, nudged times
res = client.post("/api/edl", json={"path": project, "base_version": 1, "cuts": edited})
check("manual edit saved as v2", res.get_json()["version"] == 2)
check("v1 EDL untouched", open(webapp.edl_path(project, 1)).read() == v1_raw)
state = webapp.load_state(project)
v2 = [v for v in state["versions"] if v["version"] == 2][0]
check("v2 origin/parent recorded", v2["origin"] == "manual_edit" and v2["parent"] == 1)
check("invalid cuts rejected",
      client.post("/api/edl", json={"path": project, "base_version": 2,
                                    "cuts": [{"start": 5, "end": 5}]}).status_code == 400)

# ---- 5. finalize v2 renders hi-res from disk EDL -------------------------------
OPENED.clear(); RENDERS.clear()
j = run_job({"type": "finalize", "project": project, "params": {"version": 2}})
check("finalize v2 done", j["status"] == "done")
check("finalize used hi-res sources at final bitrate",
      RENDERS[-1]["output"].endswith("final_v2.mp4")
      and RENDERS[-1]["bitrate"] == ae.FINAL_BITRATE
      and all(o.endswith(".MP4") for o in OPENED))
check("finalize honored nudged times",
      json.load(open(webapp.edl_path(project, 2)))[0]["start"] == 1.0)
check("finalize on missing version 404s",
      client.post("/api/jobs", json={"type": "finalize", "project": project,
                                     "params": {"version": 99}}).status_code == 404)

# ---- 6. media serving with Range ------------------------------------------------
res = client.get(f"/api/media?path={project}&file=preview_v1.mp4",
                 headers={"Range": "bytes=0-99"})
check("media Range request returns 206", res.status_code == 206
      and len(res.data) == 100)
check("media path traversal blocked",
      client.get(f"/api/media?path={project}&file=../../etc/passwd").status_code == 404)

# ---- 7. versions report file existence ------------------------------------------
data = client.post("/api/project/open", json={"path": project}).get_json()
vmap = {v["version"]: v for v in data["versions"]}
check("version badges correct",
      vmap[1]["has_preview"] and not vmap[1]["has_final"]
      and vmap[2]["has_final"] and not vmap[2]["has_preview"])

# ---- 8. real worker thread: queued job stays queued while first runs ------------
import threading
gate = threading.Event()
webapp.JOB_HANDLERS["slow"] = lambda job: (gate.wait(5), {"ok": True})[1]
a = webapp.submit_job("slow", project, {})
b = webapp.submit_job("slow", project, {})
threading.Thread(target=webapp.worker_loop, daemon=True).start()
time.sleep(0.3)
ja = client.get(f"/api/jobs/{a}").get_json()
jb = client.get(f"/api/jobs/{b}").get_json()
sequential_ok = ja["status"] == "running" and jb["status"] == "queued"
gate.set()
time.sleep(0.5)
# checked after the redirect_stdout in worker_loop has released real stdout
check("sequential worker: first running, second queued", sequential_ok)
check("both jobs complete after gate",
      client.get(f"/api/jobs/{a}").get_json()["status"] == "done"
      and client.get(f"/api/jobs/{b}").get_json()["status"] == "done")

# ---- 9. incremental library: partial save + resume -------------------------------
projectB = os.path.join(root, "dayB")
os.makedirs(projectB)
for stem in ("DJI_0101", "DJI_0102", "DJI_0103"):
    for ext in (".LRF", ".MP4"):
        with open(os.path.join(projectB, stem + ext), "wb") as f:
            f.write(b"\x00" * 128)
pairsB = ae.get_video_pairs(projectB)
pathsB = webapp.project_paths(projectB)
os.makedirs(pathsB["autoedit"], exist_ok=True)
libB = ae.ChunkLibrary()
libB.add({"source": "DJI_0101.MP4",
          "hires_path": os.path.join(projectB, "DJI_0101.MP4"),
          "start": 0.0, "end": 5.0, "scene": "street", "visual_tags": [],
          "actions": [], "people_count": 0, "emotions": [],
          "transition_flag": False, "one_line_summary": "walk",
          "interest_score": 0.4, "clip_duration": 20.0,
          "audio": {"loudness_peak_db": -12.0},
          "speech": {"transcript": "", "has_speech": False, "language": "en"}})
libB.save(pathsB["library"], pairsB, analyzed_sources={"DJI_0101.MP4"})

check("partial library rejected by load_if_valid",
      ae.ChunkLibrary.load_if_valid(pathsB["library"], pairsB) is None)
rlib, rdone = ae.ChunkLibrary.load_resumable(pathsB["library"], pairsB)
check("load_resumable returns chunks + done set",
      len(rlib.chunks) == 1 and rdone == {"DJI_0101.MP4"})

data = client.post("/api/project/open", json={"path": projectB}).get_json()
check("project open reports partial library",
      not data["library"]["exists"] and data["library"]["partial"]
      and data["library"]["analyzed_clips"] == 1
      and data["library"]["total_clips"] == 3)
clipsB = client.get(f"/api/project/clips?path={projectB}").get_json()
byname = {c["source"]: c for c in clipsB["clips"]}
check("per-clip analyzed flags",
      byname["DJI_0101.MP4"]["analyzed"]
      and not byname["DJI_0102.MP4"]["analyzed"]
      and clipsB["analyzed_clips"] == 1 and clipsB["total_clips"] == 3)

# resume: analyze skips done clip, saves after each remaining clip
ANALYZED = []
def fake_analyze(pair, preprocessor, analyzer, transcriber, library, *a, **k):
    src = os.path.basename(pair["hires"])
    ANALYZED.append(src)
    library.add({"source": src, "hires_path": pair["hires"],
                 "start": 0.0, "end": 5.0, "scene": "s", "visual_tags": [],
                 "actions": [], "people_count": 0, "emotions": [],
                 "transition_flag": False, "one_line_summary": "m",
                 "interest_score": 0.5, "clip_duration": 20.0,
                 "audio": {"loudness_peak_db": -12.0},
                 "speech": {"transcript": "", "has_speech": False, "language": "en"}})
ae.analyze_clip = fake_analyze
ae.VideoPreprocessor = lambda *a, **k: None
ae.ChunkAnalyzer = lambda *a, **k: None
class _FakeTranscriber:
    def unload(self): pass
ae.SpeechTranscriber = lambda *a, **k: _FakeTranscriber()
j = run_job({"type": "analyze", "project": projectB, "params": {}})
check("resume analyzes only remaining clips", j["status"] == "done"
      and ANALYZED == ["DJI_0102.MP4", "DJI_0103.MP4"])
check("library complete after resume",
      ae.ChunkLibrary.load_if_valid(pathsB["library"], pairsB) is not None)
data = client.post("/api/project/open", json={"path": projectB}).get_json()
check("project open reports complete after resume",
      data["library"]["exists"] and not data["library"]["partial"]
      and data["library"]["chunks"] == 3)

print(f"\n{len(failures)} failures")
sys.exit(1 if failures else 0)
