"""Stubbed sanity test for the v4 hierarchical StoryComposer (runs on macOS,
no GPU / heavy deps needed). Tests pure-Python composition logic only."""
import sys
import json
import types
import random

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

_stub("torch", cuda=types.SimpleNamespace(empty_cache=lambda: None,
                                          is_available=lambda: False))
_stub("moviepy", VideoFileClip=object, concatenate_videoclips=lambda *a, **k: None)
_stub("transformers", Qwen3_5ForConditionalGeneration=object, AutoProcessor=object,
      BitsAndBytesConfig=object)
_stub("faster_whisper", WhisperModel=None)
tc = _stub("torchcodec")
_stub("torchcodec.decoders", VideoDecoder=type("VD", (), {"get_frames_at": lambda self, i: None}))
tc.decoders = sys.modules["torchcodec.decoders"]

import betterautoeditor as ae

# ---- synthetic 223-chunk library --------------------------------------------
random.seed(7)
SCENES = ["home", "car", "beach", "restaurant", "street", "sunset point"]
lib = ae.ChunkLibrary()
for i in range(223):
    has_speech = i % 5 == 0
    lib.add({
        "source": f"DJI_{i // 40:04d}",
        "hires_path": f"/fake/DJI_{i // 40:04d}.MP4",
        "start": float((i % 40) * 5),
        "end": float((i % 40) * 5 + 5),
        "scene": SCENES[(i // 37) % len(SCENES)],
        "visual_tags": ["people", "outdoor", "sunny"],
        "actions": ["walking", "talking"],
        "people_count": 2,
        "emotions": ["happy"] if i % 3 == 0 else [],
        "transition_flag": i % 20 == 0,
        "one_line_summary": f"moment number {i} where something happens at the location",
        "interest_score": round(random.random(), 2),
        "audio": {"loudness_peak_db": -10.0 + (i % 8)},
        "speech": {"transcript": f"hey check this out clip {i}" if has_speech else "",
                   "has_speech": has_speech, "language": "en"},
    })

story = {"brief": "day in the life", "themes": ["friends", "beach"],
         "emotional_targets": ["happy"], "must_capture": [], "avoid": []}

composer = ae.StoryComposer()
failures = []
checks = 0

def check(name, cond):
    global checks
    checks += 1
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)

# ---- 1. digest budgets -------------------------------------------------------
outline_digest = composer._build_outline_digest(lib.chunks)
check("outline digest fits budget", len(outline_digest) <= ae.DIGEST_CHAR_BUDGET)
check("outline digest covers many chunks", outline_digest.count("\nid=") > 100)

beat_chunks = lib.chunks[0:35]
lines = [composer._digest_line(c) for c in beat_chunks]
check("per-beat full digest fits budget",
      sum(len(l) + 1 for l in lines) <= ae.DIGEST_CHAR_BUDGET)

# ---- 2. outline validation ---------------------------------------------------
def fake_llm(reply):
    def _f(system, user, max_tokens):
        return reply if isinstance(reply, str) else reply(system, user, max_tokens)
    return _f

# overlapping / gappy / reversed beats -> normalized contiguous coverage
messy = json.dumps({"story_title": "Beach Day", "beats": [
    {"name": "morning", "narrative": "waking up.", "start_id": 10, "end_id": 0, "target_shots": 3},
    {"name": "drive", "narrative": "heading out.", "start_id": 5, "end_id": 80, "target_shots": 4},
    {"name": "beach fun", "narrative": "peak of the day.", "start_id": 120, "end_id": 300, "target_shots": 20},
    {"name": "dinner", "narrative": "winding down.", "start_id": 150, "end_id": 200, "target_shots": 2},
    {"name": "goodbye", "narrative": "closing.", "start_id": 190, "end_id": 210, "target_shots": 2},
]})
ae._llm_text = fake_llm(messy)
composer_llm = composer
outline = composer._compose_outline(lib, story, 15)
check("messy outline parsed", outline is not None)
if outline:
    beats = outline["beats"]
    check("first beat starts at 0", beats[0]["start_id"] == 0)
    check("last beat ends at 222", beats[-1]["end_id"] == 222)
    contiguous = all(beats[i + 1]["start_id"] == beats[i]["end_id"] + 1
                     for i in range(len(beats) - 1))
    check("beats contiguous non-overlapping", contiguous)
    total_shots = sum(b["target_shots"] for b in beats)
    check("target_shots rescaled near target", 10 <= total_shots <= 22)
    check("per-beat shots within 1..8",
          all(1 <= b["target_shots"] <= 8 for b in beats))

# too few beats -> outline rejected
ae._llm_text = fake_llm(json.dumps({"story_title": "x", "beats": [
    {"name": "a", "narrative": "", "start_id": 0, "end_id": 222, "target_shots": 15}]}))
check("too-few-beats outline rejected", composer._compose_outline(lib, story, 15) is None)

# garbage -> rejected
ae._llm_text = fake_llm("not json at all")
check("garbage outline rejected", composer._compose_outline(lib, story, 15) is None)

# ---- 3. beat selection -------------------------------------------------------
beat = {"name": "beach fun", "narrative": "peak of the day",
        "start_id": 100, "end_id": 140, "target_shots": 4}
bchunks = [c for c in lib.chunks if 100 <= c["library_id"] <= 140]

# ids outside beat range rejected, dedupe works
ae._llm_text = fake_llm(json.dumps({"selections": [
    {"chunk_id": 105, "role": "peak", "reason": "great moment."},
    {"chunk_id": 105, "role": "peak", "reason": "dup."},
    {"chunk_id": 5, "role": "setting", "reason": "outside range."},
    {"chunk_id": 130, "role": "weirdo-role", "reason": "role coerced."},
]}))
picks = composer._select_for_beat(beat, bchunks, story, "")
ids = [p["chunk"]["library_id"] for p in picks]
check("beat selection valid ids only", ids == [105, 130])
check("bad role coerced to buildup", picks[1]["role"] == "buildup")

# failed beat call -> heuristic picks
ae._llm_text = fake_llm("garbage")
picks = composer._select_for_beat(beat, bchunks, story, "")
check("failed beat falls back to heuristic", len(picks) == 4)
check("heuristic picks within beat range",
      all(100 <= p["chunk"]["library_id"] <= 140 for p in picks))

# ---- 4. end-to-end compose() with fake LLM ------------------------------------
good_outline = json.dumps({"story_title": "Pickle & Masti", "beats": [
    {"name": "waking up", "narrative": "the day starts.", "start_id": 0, "end_id": 50, "target_shots": 3},
    {"name": "the drive", "narrative": "heading to the beach.", "start_id": 51, "end_id": 110, "target_shots": 4},
    {"name": "beach peak", "narrative": "the best moments.", "start_id": 111, "end_id": 180, "target_shots": 5},
    {"name": "sunset close", "narrative": "the day ends.", "start_id": 181, "end_id": 222, "target_shots": 3},
]})
call_log = []
prior_summaries = []

def scripted_llm(system, user, max_tokens):
    call_log.append(max_tokens)
    if "story beats" in user:
        return good_outline
    m = None
    import re as _re
    m = _re.search(r"SHOTS:\n(id=\d+)", user)
    first_id = int(m.group(1).split("=")[1])
    sf = _re.search(r"STORY SO FAR:\n(.*?)\n\nEDITING BRIEF", user, _re.S)
    prior_summaries.append(sf.group(1) if sf else "")
    sels = [{"chunk_id": first_id + k * 3, "role": "peak" if k == 1 else "setting",
             "reason": f"a full one-sentence reason for shot {first_id + k * 3}."}
            for k in range(3)]
    return json.dumps({"selections": sels})

ae._llm_text = scripted_llm
selections = composer.compose(lib, story, 15)
check("compose returns selections", len(selections) == 12)
srt = sorted(selections, key=lambda s: (s["chunk"]["source"], s["chunk"]["start"]))
check("compose output chronological", selections == srt)
check("selections carry beat names", all(s.get("beat") for s in selections))
check("one outline call + 4 beat calls", len(call_log) == 5)
check("outline call used OUTLINE_MAX_TOKENS", call_log[0] == ae.OUTLINE_MAX_TOKENS)
check("beat calls used BEAT_MAX_TOKENS",
      all(t == ae.BEAT_MAX_TOKENS for t in call_log[1:]))
check("prior_summary empty for first beat, grows after",
      "Nothing selected yet" in prior_summaries[0]
      and all("waking up:" in s for s in prior_summaries[1:]))

# EDL carries beat
cuts = ae.FinalEditor.build_cuts(selections, 3, {}, {})
check("EDL cuts carry beat name", all(c["beat"] for c in cuts))

# ---- 5. garbage LLM everywhere -> compose falls through to [] -----------------
ae._llm_text = fake_llm("total garbage every time")
selections = composer.compose(lib, story, 15)
check("garbage LLM -> empty (FallbackSelector takes over in pipeline)",
      selections == [])

print("\n%d checks, %d failures" % (checks, len(failures)))
sys.exit(1 if failures else 0)
