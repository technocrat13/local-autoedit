"""Stubbed sanity test for the knowledge graph layer (runs on macOS, no GPU /
heavy deps needed): GraphExtractor validation (seeds, clip-level themes/places,
scene-seed folding, salvage, retry), KnowledgeGraphStore persistence and
fingerprint invalidation, KnowledgeGraph build/subgraph/elements_of_day with
object/mood demotion, and compose_with_graph auto cut counts + fallback."""
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

# ---- 1. GraphExtractor: seeds, ranges, canonicalize, semantic, retry ----------
extractor = ae.GraphExtractor()

def mk(i, scene="beach", actions=(), tags=(), emotions=(), summary=None, speech=""):
    return {"source": "X.MP4", "hires_path": "/fake/X.MP4",
            "start": i * 5.0, "end": i * 5.0 + 5, "scene": scene,
            "visual_tags": list(tags), "actions": list(actions),
            "people_count": 1, "emotions": list(emotions),
            "transition_flag": False,
            "one_line_summary": summary or f"moment {i}",
            "interest_score": 0.5, "clip_duration": 60.0,
            "audio": {"loudness_peak_db": -12.0},
            "speech": {"transcript": speech, "has_speech": bool(speech),
                       "language": "en"}}

# 1a. deterministic seeds - no LLM involved
seed_chunks = ([mk(i, scene="Sandy  Beach", actions=["walking"], tags=["drone"])
                for i in range(2)]
               + [mk(2, scene="unknown", actions=["juggling"], tags=["drone"])]
               + [mk(i, scene="sandy beach", emotions=["joy"])
                  for i in range(3, 8)])
seeds = ae.GraphExtractor._seed_terms(seed_chunks)
check("seed place from scene, normalized",
      seeds.get(("place", "sandy beach")) == {0, 1, 3, 4, 5, 6, 7})
check("seed skips unknown scene",
      not any(k == ("place", "unknown") for k in seeds))
check("seed activity/object need 2 chunks",
      ("activity", "walking") in seeds and ("object", "drone") in seeds
      and ("activity", "juggling") not in seeds)
check("seed mood from emotions", seeds[("mood", "joy")] == {3, 4, 5, 6, 7})

small = [mk(0, actions=["surfing"]), mk(1), mk(2)]
check("small clip keeps single-chunk terms",
      ("activity", "surfing") in ae.GraphExtractor._seed_terms(small))

mood_chunks = [mk(i, emotions=[f"mood{i % 7}"]) for i in range(14)]
mood_seeds = ae.GraphExtractor._seed_terms(mood_chunks)
check("mood seeds capped at 5",
      sum(1 for k in mood_seeds if k[0] == "mood") == 5)

# 1b. parse_id_ranges
check("parse_id_ranges ranges + singles",
      ae.parse_id_ranges("3-7,12", 20) == [3, 4, 5, 6, 7, 12])
check("parse_id_ranges clamps + skips garbage",
      ae.parse_id_ranges("8-15, x, -2, 3", 10) == [3, 8, 9])
check("parse_id_ranges reversed range",
      ae.parse_id_ranges("7-3", 10) == [3, 4, 5, 6, 7])
check("parse_id_ranges accepts int list",
      ae.parse_id_ranges([1, "2", 99], 10) == [1, 2])

# 1c. full pipeline: canned merges + semantic entities
LONG = ("a man in a red cap wanders up and down the shoreline scanning "
        "the waves for any sign of his missing drone")
clip = ([mk(i, scene="sandy beach", actions=["swimming"], summary=LONG,
            speech="where did the drone go") for i in range(4)]
        + [mk(i, scene="sandy beach") for i in range(4, 8)])

prompts = {"merge": [], "semantic": []}
def scripted(merge_replies, semantic_replies):
    m, s = iter(merge_replies), iter(semantic_replies)
    def _f(system, user, max_tokens):
        if "THIS CLIP'S TERMS" in user:
            prompts["merge"].append(user)
            return next(m)
        prompts["semantic"].append(user)
        return next(s)
    return _f

good_merge = json.dumps({"merges": [
    {"type": "place", "from": "sandy beach", "to": "The  Beach"},
    {"type": "spaceship", "from": "x", "to": "y"},          # unknown type
    {"type": "activity", "from": "not a term", "to": "z"},  # unknown from
]})
good_semantic = json.dumps({
    "themes": [{"name": "Hunt For The  Lost Drone", "chunks": "junk"}],  # bad ids
    "places": [{"name": "the beach", "chunks": "0-7"}],
    "people": [{"name": " Man In  Red Cap ", "chunks": "0-2,3"}],
    "threads": [{"name": "finding the lost drone", "chunks": "0-3"},
                {"name": "", "chunks": "1"},                # no name
                {"name": "ghost", "chunks": "50-60"}]})     # ids out of range

ae._llm_text = scripted([good_merge], [good_semantic])
ents, enriched = extractor.extract_clip(clip, {"place": ["the beach"]})
by_key = {(e["type"], e["name"]): e for e in ents}
check("merge renames onto canonical name",
      by_key.get(("place", "the beach"), {}).get("chunk_ids") == list(range(8)))
check("invalid merges ignored, seeds kept", ("activity", "swimming") in by_key)
check("theme with garbage chunk spec spans whole clip",
      by_key.get(("theme", "hunt for the lost drone"), {}).get("chunk_ids")
      == list(range(8)))
check("people/threads parsed from range strings",
      by_key.get(("person", "man in red cap"), {}).get("chunk_ids") == [0, 1, 2, 3]
      and ("thread", "finding the lost drone") in by_key)
check("nameless/out-of-range semantic entities dropped",
      ("thread", "ghost") not in by_key
      and not any(t == "thread" and not n for t, n in by_key))
check("good replies -> enriched", enriched is True)
check("vocabulary appears in merge prompt", "the beach" in prompts["merge"][0])

# 1c-bis. clip-level places fold fragmented scene seeds
kart_clip = ([mk(i, scene="racetrack") for i in range(6)]
             + [mk(i, scene="paddock") for i in range(6, 10)])
fold_semantic = json.dumps({
    "themes": [{"name": "go karting day", "chunks": "0-9"}],
    "places": [{"name": "go kart track", "chunks": "0-5"}],
    "people": [], "threads": []})
ae._llm_text = scripted([json.dumps({"merges": []})], [fold_semantic])
fents, _ = extractor.extract_clip(kart_clip, {})
fkeys = {(e["type"], e["name"]): e for e in fents}
check("covered scene seed folds into clip place",
      ("place", "racetrack") not in fkeys
      and fkeys.get(("place", "go kart track"), {}).get("chunk_ids") == [0, 1, 2, 3, 4, 5])
check("uncovered scene seed keeps its remaining ids",
      fkeys.get(("place", "paddock"), {}).get("chunk_ids") == [6, 7, 8, 9])
check("theme entity carried through", ("theme", "go karting day") in fkeys)

# 1c-ter. semantic salvage classifies objects by nearest preceding group key
trunc_sem = ('{"themes": [{"name": "go karting day", "chunks": "0-9"}], '
             '"people": [{"name": "man in red cap", "chunks": "2-4"}, '
             '{"name": "pit crew", "chunks": "5')
salv = ae.GraphExtractor._salvage_semantic(trunc_sem)
check("salvage assigns objects to nearest preceding group",
      salv["themes"][0]["name"] == "go karting day"
      and [p["name"] for p in salv["people"]] == ["man in red cap"]
      and salv["places"] == [] and salv["threads"] == [])
orphan = ('{"name": "go karting", "chunks": "0-9"} "people": ['
          '{"name": "driver", "chunks": "1-3"}]')
salv2 = ae.GraphExtractor._salvage_semantic(orphan)
check("salvage puts objects before any key into themes",
      salv2["themes"][0]["name"] == "go karting"
      and salv2["people"][0]["name"] == "driver")

# 1d. truncated merge reply -> salvage, no retry
truncated = ('{"merges": [{"type": "place", "from": "sandy beach", '
             '"to": "the beach"}, {"type": "activity", "from": "swim')
prompts["merge"].clear(); prompts["semantic"].clear()
ae._llm_text = scripted([truncated], [good_semantic])
ents2, enriched2 = extractor.extract_clip(clip, {"place": ["the beach"]})
keys2 = {(e["type"], e["name"]) for e in ents2}
check("truncated merge reply salvaged",
      ("place", "the beach") in keys2 and enriched2 is True)
check("salvage does not retry", len(prompts["merge"]) == 1)
check("unmerged terms keep raw names", ("activity", "swimming") in keys2)

# 1e. one self-retry with strictly less output pressure
prompts["merge"].clear(); prompts["semantic"].clear()
ae._llm_text = scripted(["total garbage", json.dumps({"merges": []})],
                        ["also garbage", good_semantic])
ents3, enriched3 = extractor.extract_clip(clip, {})
check("merge retried once on garbage", len(prompts["merge"]) == 2)
check("merge retry prompt is smaller",
      len(prompts["merge"][1]) < len(prompts["merge"][0]))
check("semantic retry prompt is smaller",
      len(prompts["semantic"][1]) < len(prompts["semantic"][0]))
check("retry success -> enriched", enriched3 is True)

# 1f. both LLM calls fail -> seeds survive, marked unenriched
ae._llm_text = fake_llm("garbage")
ents4, enriched4 = extractor.extract_clip(clip, {})
check("double LLM failure still returns seeds",
      any(e["type"] == "place" for e in ents4))
check("double LLM failure -> enriched False", enriched4 is False)

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
payload["schema"] = 2  # pre-theme schema is rebuilt from scratch
with open(kg_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
_, status3 = ae.KnowledgeGraphStore.load(kg_path, pairs)
check("old schema -> everything new",
      all(st == "new" for st in status3.values()))
store.save(kg_path)  # restore good file

# ---- 3. sync_knowledge_graph backfill + enrichment retry ------------------------
sync_calls = []
def sync_llm(system, user, max_tokens):
    sync_calls.append(user)
    return json.dumps({"merges": [], "themes": [], "places": [],
                       "people": [], "threads": []})
ae._llm_text = sync_llm
store2, _ = ae.KnowledgeGraphStore.load(kg_path, pairs)
check("sync extracts only stale/missing clips (2 LLM calls each)",
      (ae.sync_knowledge_graph(store2, kg_path, lib, pairs) or True)
      and len(sync_calls) == 4)
check("sync commits the new clips",
      all(store2.is_current(p) for p in pairs))
check("sync marks clips enriched",
      all(store2.is_enriched(p) for p in pairs))
sync_calls.clear()
ae.sync_knowledge_graph(store2, kg_path, lib, pairs)
check("sync is a no-op when current", sync_calls == [])

# a clip whose LLM calls all failed is committed unenriched and retried
ae._llm_text = fake_llm("garbage")
store2.commit_clip(pairs[2], [], enriched=False)
ae.sync_knowledge_graph(store2, kg_path, lib, pairs)
check("failed enrichment stays unenriched", not store2.is_enriched(pairs[2]))
check("seeds still committed for unenriched clip",
      len(store2.clip_entries[SOURCES[2]]["entities"]) > 0)
sync_calls.clear()
ae._llm_text = sync_llm
ae.sync_knowledge_graph(store2, kg_path, lib, pairs)
check("unenriched clip retried on next sync",
      len(sync_calls) == 2 and store2.is_enriched(pairs[2]))
check("enriched clips untouched by retry sync",
      store2.is_enriched(pairs[0]) and store2.is_enriched(pairs[1]))

# ---- 4. KnowledgeGraph build: merge, edges, chronology -------------------------
store3 = ae.KnowledgeGraphStore()
store3.commit_clip(pairs[0], [
    {"type": "theme", "name": "beach day", "chunk_ids": [0, 1, 2]},
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
check("themes outrank threads outrank places",
      keys.index("theme:beach day")
      < keys.index("thread:finding the lost drone")
      < keys.index("place:the beach"))
check("all substantial element-type entities covered",
      {"place:the beach", "activity:swimming"} <= set(keys))
check("object/mood demoted out of elements",
      "mood:golden hour" not in keys
      and not any(k.split(":")[0] in ("object", "mood") for k in keys))
check("demoted types stay in the graph for retrieval",
      "mood:golden hour" in graph.entities)

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
check("theme element becomes a beat",
      any(s.get("beat") == "beach day" for s in sels))
check("demoted types never become beats",
      not any(s.get("beat") == "golden hour" for s in sels))

ae._llm_text = element_llm
sels_must = composer.compose_with_graph(
    lib, story, graph, must_include=["mood:golden hour", "bogus:nope"])
check("must_include forces a demoted entity into the beats",
      any(s.get("beat") == "golden hour" for s in sels_must))
check("unknown must_include keys ignored",
      not any(s.get("beat") == "nope" for s in sels_must))
srt_must = sorted(sels_must, key=lambda s: (s["chunk"]["source"], s["chunk"]["start"]))
check("must_include keeps chronology", sels_must == srt_must)
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
# 600s at ~11s/cut wants ~55 cuts; the library only has 30 chunks, so the
# top-up must exhaust the whole library instead of stopping at element quotas
check("top-up approaches long targets", len(sels_long) == len(lib.chunks))
check("top-up picks are flagged for pruning",
      any(s["beat"] == "top-up" for s in sels_long))
check("short target adds no top-up",
      not any(s["beat"] == "top-up" for s in sels_short))
srt_long = sorted(sels_long, key=lambda s: (s["chunk"]["source"], s["chunk"]["start"]))
check("top-up keeps chronology", sels_long == srt_long)

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
