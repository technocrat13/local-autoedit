"""Stubbed sanity test for the v5 preview/iterate/finalize flow (runs on macOS,
no GPU / heavy deps). Fakes moviepy to record what each render opened/encoded."""
import sys
import json
import types
import tempfile
import os

# ---- stub heavy modules before importing the script ------------------------
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

RENDERS = []          # one dict per write_videofile call
OPENED = []           # source paths opened per render (reset by test)

class FakeClip:
    duration = 10_000.0
    def subclipped(self, start, end):
        return ("sub", start, end)
    def close(self):
        pass

def fake_videofileclip(src):
    OPENED.append(src)
    return FakeClip()

class FakeFinal:
    def write_videofile(self, output_path, **kwargs):
        RENDERS.append({"output": output_path, **kwargs})
    def close(self):
        pass

_stub("torch", cuda=types.SimpleNamespace(empty_cache=lambda: None,
                                          is_available=lambda: False))
_stub("moviepy", VideoFileClip=fake_videofileclip,
      concatenate_videoclips=lambda clips, method=None: FakeFinal())
_stub("transformers", Qwen3_5ForConditionalGeneration=object, AutoProcessor=object,
      BitsAndBytesConfig=object)
_stub("faster_whisper", WhisperModel=None)
tc = _stub("torchcodec")
_stub("torchcodec.decoders", VideoDecoder=type("VD", (), {"get_frames_at": lambda self, i: None}))
tc.decoders = sys.modules["torchcodec.decoders"]

import betterautoeditor as ae

failures = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)

tmp = tempfile.mkdtemp(prefix="v5test_")
workdir = ae.WorkDir()

# ---- 1. build_cuts emits both source paths ----------------------------------
selections = [{
    "chunk": {"source": "DJI_0001.MP4", "hires_path": "/fake/DJI_0001.MP4",
              "start": 20.0, "end": 25.0, "visual_tags": ["beach"],
              "one_line_summary": "s", "speech": {"transcript": "hi"}},
    "role": "peak", "reason": "r", "beat": "b",
}]
lrf_map = {"DJI_0001.MP4": "/fake/DJI_0001.LRF"}
cuts = ae.FinalEditor.build_cuts(selections, 3, {"DJI_0001.MP4": 100.0}, lrf_map)
check("cut has hires source_file", cuts[0]["source_file"] == "/fake/DJI_0001.MP4")
check("cut has lrf_file", cuts[0]["lrf_file"] == "/fake/DJI_0001.LRF")

# ---- 2. preview vs final render settings ------------------------------------
editor = ae.FinalEditor(workdir)
plan = [
    {"source_file": "/fake/A.MP4", "lrf_file": "/fake/A.LRF", "start": 0.0, "end": 5.0},
    {"source_file": "/fake/B.MP4", "lrf_file": "/fake/B.LRF", "start": 3.0, "end": 9.0},
]
OPENED.clear(); RENDERS.clear()
editor.render(plan, "prev.mp4", source_key="lrf_file", preset="ultrafast",
              bitrate=ae.PREVIEW_BITRATE)
check("preview opens LRF sources", OPENED == ["/fake/A.LRF", "/fake/B.LRF"])
check("preview uses ultrafast + preview bitrate",
      RENDERS[-1].get("preset") == "ultrafast"
      and RENDERS[-1]["bitrate"] == ae.PREVIEW_BITRATE)

OPENED.clear(); RENDERS.clear()
editor.render(plan, "final.mp4")
check("final opens hi-res sources", OPENED == ["/fake/A.MP4", "/fake/B.MP4"])
check("final uses 55Mbps and no preset override",
      RENDERS[-1]["bitrate"] == ae.FINAL_BITRATE and "preset" not in RENDERS[-1])

# ---- 3. render_from_edl on a hand-edited EDL ---------------------------------
edl_path = os.path.join(tmp, "test_edl.json")
hand_edited = [
    {"source_file": "/fake/B.MP4", "lrf_file": "/fake/B.LRF",
     "start": 3.5, "end": 8.0},                       # nudged times, reordered first
    {"source_file": "/fake/A.MP4", "lrf_file": "/fake/A.LRF",
     "start": 5.0, "end": 5.0},                       # invalid: end <= start
    {"source_file": "/fake/A.MP4", "lrf_file": "/fake/A.LRF",
     "start": "1.0", "end": "4.0"},                   # string times coerced
    "not a dict",
]
with open(edl_path, "w") as f:
    json.dump(hand_edited, f)
OPENED.clear(); RENDERS.clear()
ok = ae.render_from_edl(edl_path, "final2.mp4", workdir)
check("render_from_edl renders valid cuts in file order",
      ok and OPENED == ["/fake/B.MP4", "/fake/A.MP4"])
check("render_from_edl missing file errors cleanly",
      ae.render_from_edl(os.path.join(tmp, "nope.json"), "x.mp4", workdir) is False)

# ---- 4. --finalize never loads the model -------------------------------------
model_loads = []
ae.load_model = lambda: model_loads.append(1)
args = types.SimpleNamespace(
    brief="b", footage_dir="/nowhere", output=os.path.join(tmp, "test.mp4"),
    max_chunks=None, margin=3.0, target_cuts=15, reanalyze=False, finalize=True)
with open(os.path.join(tmp, "test_edl.json"), "w") as f:
    json.dump(plan, f)
OPENED.clear(); RENDERS.clear()
ae.run_pipeline(args)
check("--finalize renders hi-res from EDL", OPENED == ["/fake/A.MP4", "/fake/B.MP4"])
check("--finalize never loads model", model_loads == [])

# ---- 5. interactive loop ------------------------------------------------------
# Build a tiny fake library + monkeypatch the pipeline pieces.
lib = ae.ChunkLibrary()
for i in range(6):
    lib.add({"source": "DJI_0001.MP4", "hires_path": "/fake/DJI_0001.MP4",
             "start": i * 5.0, "end": i * 5.0 + 5, "scene": "beach",
             "visual_tags": [], "actions": [], "people_count": 1, "emotions": [],
             "transition_flag": False, "one_line_summary": f"m{i}",
             "interest_score": 0.5, "clip_duration": 60.0,
             "audio": {"loudness_peak_db": -12.0},
             "speech": {"transcript": "", "has_speech": False, "language": "en"}})

ae.get_video_pairs = lambda d: [{"lrf": "/fake/DJI_0001.LRF",
                                 "hires": "/fake/DJI_0001.MP4"}]
ae.ChunkLibrary.load_if_valid = staticmethod(lambda path, pairs: lib)

composed = []
def fake_compose(self, library, story, target_cuts):
    composed.append((story["brief"], target_cuts))
    return [{"chunk": library.chunks[0], "role": "setting", "reason": "r"}]
ae.StoryComposer.compose = fake_compose
ae.StoryAgent.parse_user_brief = lambda self, brief: {
    "brief": brief, "themes": [], "emotional_targets": [],
    "must_capture": [], "avoid": []}

ae.sys = types.SimpleNamespace(stdin=types.SimpleNamespace(isatty=lambda: True))
script = iter(["make it about the beach", "cuts 10", "preview", "final"])
ae.input = lambda prompt="": next(script)

args.finalize = False
OPENED.clear(); RENDERS.clear(); model_loads.clear()
ae.run_pipeline(args)

check("model loaded once for interactive run", model_loads == [1])
check("initial + brief + cuts re-compositions",
      composed == [("b", 15), ("make it about the beach", 15),
                   ("make it about the beach", 10)])
preview_renders = [r for r in RENDERS if r["output"].endswith("_preview.mp4")]
final_renders = [r for r in RENDERS if r["output"] == args.output]
check("4 preview renders (initial, brief, cuts, preview cmd)",
      len(preview_renders) == 4)
check("all previews are ultrafast LRF renders",
      all(r.get("preset") == "ultrafast" and r["bitrate"] == ae.PREVIEW_BITRATE
          for r in preview_renders))
check("'final' renders hi-res once at 55Mbps",
      len(final_renders) == 1 and final_renders[0]["bitrate"] == ae.FINAL_BITRATE)
check("preview cmd re-read EDL from disk (LRF opened)", "/fake/DJI_0001.LRF" in OPENED)

# EOF quits cleanly without hi-res render
def raise_eof(prompt=""):
    raise EOFError
ae.input = raise_eof
OPENED.clear(); RENDERS.clear()
ae.run_pipeline(args)
check("EOF quits without hi-res render",
      all(r["output"].endswith("_preview.mp4") for r in RENDERS))

print(f"\n{len(failures)} failures")
sys.exit(1 if failures else 0)
