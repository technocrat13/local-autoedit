"""Stubbed sanity test for the knowledge graph layer (runs on macOS, no GPU /
heavy deps needed): GraphExtractor validation, KnowledgeGraphStore persistence
and fingerprint invalidation, KnowledgeGraph build/subgraph/elements_of_day,
and compose_with_graph auto cut counts + fallback."""
import os
import sys
import json
import types
import random
import tempfile

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

failures = []
checks = 0

def check(name, cond):
    global checks
    checks += 1
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)

def fake_llm(reply):
    def _f(system, user, max_tokens):
        return reply if isinstance(reply, str) else reply(system, user, max_tokens)
    return _f

# ---- synthetic project: 3 clips, 10 chunks each ------------------------------
tmp = tempfile.mkdtemp(prefix="kgtest_")
random.seed(11)
SOURCES = ["DJI_0001.MP4", "DJI_0002.MP4", "DJI_0003.MP4"]
pairs = []
for i, source in enumerate(SOURCES):
    lrf = os.path.join(tmp, source.replace(".MP4", ".LRF"))
    with open(lrf, "wb") as f:
        f.write(b"x" * (100 + i))  # distinct sizes = distinct fingerprints
    pairs.append({"lrf": lrf, "hires": os.path.join(tmp, source)})

lib = ae.ChunkLibrary()
for ci, source in enumerate(SOURCES):
    for k in range(10):
        lib.add({
            "source": source, "hires_path": pairs[ci]["hires"],
            "start": float(k * 5), "end": float(k * 5 + 5),
            "scene": "beach" if ci else "home",
            "visual_tags": ["outdoor"], "actions": ["walking"],
            "people_count": 1, "emotions": [], "transition_flag": False,
            "one_line_summary": f"clip {ci} moment {k}",
            "interest_score": round(random.random(), 2),
            "clip_duration": 50.0,
            "audio": {"loudness_peak_db": -12.0},
            "speech": {"transcript": "", "has_speech": False, "language": "en"},
        })
for pair in pairs:
    lib.commit_clip(pair)

clip_chunks = {s: [c for c in lib.chunks if c["source"] == s] for s in SOURCES}

# ---- 1. GraphExtractor validation ---------------------------------------------
extractor = ae.GraphExtractor()

ae._llm_text = fake_llm(json.dumps({"entities": [
    {"type": "place", "name": "  The   Beach ", "chunk_ids": [0, 1, 2]},
    {"type": "person", "name": "man in red cap", "chunk_ids": [3, "4", 99, -1]},
    {"type": "place", "name": "the beach", "chunk_ids": [5]},   # dup after normalize
    {"type": "spaceship", "name": "ufo", "chunk_ids": [0]},     # unknown type
    {"type": "mood", "name": "chill", "chunk_ids": []},         # no valid ids
    "not a dict",
]}))
ents = extractor.extract_clip(clip_chunks[SOURCES[0]], {t: [] for t in ae.KG_ENTITY_TYPES})
check("extractor normalizes names", any(e["name"] == "the beach" for e in ents))
check("extractor keeps valid entities only", len(ents) == 2)
person = next(e for e in ents if e["type"] == "person")
check("extractor clamps + coerces chunk_ids", person["chunk_ids"] == [3, 4])

ae._llm_text = fake_llm("total garbage")
check("garbage extraction -> []",
      extractor.extract_clip(clip_chunks[SOURCES[0]], {}) == [])

vocab_seen = {}
def vocab_llm(system, user, max_tokens):
    vocab_seen["user"] = user
    return json.dumps({"entities": [
        {"type": "place", "name": "the beach", "chunk_ids": [0]}]})
ae._llm_text = vocab_llm
extractor.extract_clip(clip_chunks[SOURCES[1]],
                       {"place": ["the beach"], "person": ["man in red cap"]})
check("vocabulary appears in prompt",
      "the beach" in vocab_seen["user"] and "man in red cap" in vocab_seen["user"])

# ---- 2. KnowledgeGraphStore persistence + fingerprints -------------------------
store = ae.KnowledgeGraphStore()
store.commit_clip(pairs[0], [
    {"type": "place", "name": "the beach", "chunk_ids": [0, 1, 2]},
    {"type": "person", "name": "man in red cap", "chunk_ids": [3, 4]},
    {"type": "thread", "name": "finding the lost drone", "chunk_ids": [5, 6]},
])
store.commit_clip(pairs[1], [
    {"type": "place", "name": "the beach", "chunk_ids": [0, 1]},
    {"type": "activity", "name": "swimming", "chunk_ids": [2, 3, 4]},
])
kg_path = os.path.join(tmp, "kg.json")
store.save(kg_path)

vocab = store.vocabulary()
check("vocabulary dedupes across clips", vocab["place"] == ["the beach"])
check("vocabulary keyed by type", "swimming" in vocab["activity"])

loaded, status = ae.KnowledgeGraphStore.load(kg_path, pairs)
check("load restores entities",
      len(loaded.clip_entries[SOURCES[0]]["entities"]) == 3)
check("load status analyzed/new",
      status[SOURCES[0]] == "analyzed" and status[SOURCES[2]] == "new")
check("store is_current tracks fingerprint",
      loaded.is_current(pairs[0]) and not loaded.is_current(pairs[2]))

with open(pairs[1]["lrf"], "ab") as f:
    f.write(b"more")  # lrf grows -> fingerprint changes
_, status2 = ae.KnowledgeGraphStore.load(kg_path, pairs)
check("changed lrf invalidates only that clip",
      status2[SOURCES[1]] == "changed" and status2[SOURCES[0]] == "analyzed")

with open(kg_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
payload["schema"] = 99
with open(kg_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
_, status3 = ae.KnowledgeGraphStore.load(kg_path, pairs)
check("wrong schema -> everything new",
      all(st == "new" for st in status3.values()))
store.save(kg_path)  # restore good file

# ---- 3. sync_knowledge_graph backfill ------------------------------------------
sync_calls = []
def sync_llm(system, user, max_tokens):
    sync_calls.append(user)
    return json.dumps({"entities": [
        {"type": "mood", "name": "golden hour", "chunk_ids": [7, 8]}]})
ae._llm_text = sync_llm
store2, _ = ae.KnowledgeGraphStore.load(kg_path, pairs)
ae.sync_knowledge_graph(store2, kg_path, lib, pairs)
check("sync extracts only stale/missing clips", len(sync_calls) == 2)
check("sync commits the new clips",
      all(store2.is_current(p) for p in pairs))
sync_calls.clear()
ae.sync_knowledge_graph(store2, kg_path, lib, pairs)
check("sync is a no-op when current", sync_calls == [])

# ---- 4. KnowledgeGraph build: merge, edges, chronology -------------------------
store3 = ae.KnowledgeGraphStore()
store3.commit_clip(pairs[0], [
    {"type": "place", "name": "the beach", "chunk_ids": [1, 0]},
    {"type": "person", "name": "man in red cap", "chunk_ids": [0, 3]},
    {"type": "thread", "name": "finding the lost drone", "chunk_ids": [3, 5]},
])
store3.commit_clip(pairs[1], [
    {"type": "place", "name": "the beach", "chunk_ids": [0, 2]},
    {"type": "activity", "name": "swimming", "chunk_ids": [2, 3, 4]},
])
store3.commit_clip(pairs[2], [
    {"type": "mood", "name": "golden hour", "chunk_ids": [8, 9, 99]},  # 99 dropped
])
graph = ae.KnowledgeGraph.build(store3, lib)

beach = graph.entities["place:the beach"]
lid = {(c["source"], c["start"]): c["library_id"] for c in lib.chunks}
check("cross-clip merge by type:name",
      beach["chunk_ids"] == [lid[(SOURCES[0], 0.0)], lid[(SOURCES[0], 5.0)],
                             lid[(SOURCES[1], 0.0)], lid[(SOURCES[1], 10.0)]])
check("out-of-range clip-local ids dropped",
      len(graph.entities["mood:golden hour"]["chunk_ids"]) == 2)
pair_key = frozenset(("place:the beach", "person:man in red cap"))
check("co-occurrence edge from shared chunk",
      graph.co_occurs.get(pair_key) == 1)
check("no self edges", all(len(p) == 2 for p in graph.co_occurs))

removed_lib = ae.ChunkLibrary()
for c in lib.chunks:
    if c["source"] != SOURCES[0]:
        removed_lib.add(dict(c))
graph_removed = ae.KnowledgeGraph.build(store3, removed_lib)
check("removed clip drops from graph",
      "person:man in red cap" not in graph_removed.entities
      and "place:the beach" in graph_removed.entities)

# ---- 5. elements_of_day + subgraph ---------------------------------------------
elements = graph.elements_of_day()
keys = [e["key"] for e in elements]
check("threads outrank places",
      keys.index("thread:finding the lost drone") < keys.index("place:the beach"))
check("all substantial entities covered",
      {"place:the beach", "activity:swimming", "mood:golden hour"} <= set(keys))

dup_store = ae.KnowledgeGraphStore()
dup_store.commit_clip(pairs[0], [
    {"type": "place", "name": "the beach", "chunk_ids": [0, 1, 2]},
    {"type": "activity", "name": "beach walk", "chunk_ids": [0, 1, 2]},  # same chunks
])
dup_graph = ae.KnowledgeGraph.build(dup_store, lib)
dup_keys = [e["key"] for e in dup_graph.elements_of_day()]
check("redundant element filtered by freshness", len(dup_keys) == 1)

window = set(beach["chunk_ids"][:2])
sub = graph.subgraph(window)
check("subgraph restricts chunk_ids",
      sub.entities["place:the beach"]["chunk_ids"] == sorted(window))
check("subgraph prunes empty entities",
      "mood:golden hour" not in sub.entities)
check("subgraph keeps edges with surviving endpoints",
      all(all(k in sub.entities for k in p) for p in sub.co_occurs))

payload = graph.to_json(lib)
check("to_json nodes carry chunk refs",
      all("chunks" in n and "chunk_count" in n for n in payload["nodes"]))
check("to_json edges well formed",
      all(e["a"] < e["b"] for e in payload["edges"]))
check("to_json lists elements", set(payload["elements"]) <=
      {n["id"] for n in payload["nodes"]})

# ---- 6. compose_with_graph ------------------------------------------------------
story = {"brief": "day at the beach", "themes": ["beach"],
         "emotional_targets": [], "must_capture": [], "avoid": []}
composer = ae.StoryComposer()

def element_llm(system, user, max_tokens):
    import re as _re
    ids = [int(m) for m in _re.findall(r"id=(\d+)", user)]
    sels = [{"chunk_id": i, "role": "peak", "reason": "a good moment for this element."}
            for i in ids[:3]]
    return json.dumps({"selections": sels})

ae._llm_text = element_llm
sels = composer.compose_with_graph(lib, story, graph)
check("graph compose returns selections", len(sels) >= ae.MIN_AUTO_CUTS // 2)
srt = sorted(sels, key=lambda s: (s["chunk"]["source"], s["chunk"]["start"]))
check("graph compose chronological", sels == srt)
check("selections carry element beats", all(s.get("beat") for s in sels))
ids = [s["chunk"]["library_id"] for s in sels]
check("no duplicate chunks", len(ids) == len(set(ids)))
first = min(lib.chunks, key=lambda c: (c["source"], c["start"]))
last = max(lib.chunks, key=lambda c: (c["source"], c["start"]))
check("day anchors present",
      first["library_id"] in ids and last["library_id"] in ids)

sels_short = composer.compose_with_graph(lib, story, graph, target_len=60,
                                         margin=3.0)
sels_long = composer.compose_with_graph(lib, story, graph, target_len=600,
                                        margin=3.0)
check("target_len scales cut count", len(sels_short) < len(sels_long))

sels_capped = composer.compose_with_graph(lib, story, graph, max_total=3)
check("max_total caps and drops anchors", len(sels_capped) <= 3 + 2
      and all(s["beat"] not in ("setting", "outro") or s.get("role") for s in sels_capped))

empty = composer.compose_with_graph(lib, story, ae.KnowledgeGraph())
check("empty graph -> [] for legacy fallback", empty == [])

ae._llm_text = fake_llm("garbage")
sels_fb = composer.compose_with_graph(lib, story, graph)
check("garbage element LLM still yields heuristic picks or []",
      isinstance(sels_fb, list))

print("\n%d checks, %d failures" % (checks, len(failures)))
sys.exit(1 if failures else 0)
