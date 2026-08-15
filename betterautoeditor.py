import os
import re
import sys
import glob
import json
import wave
import atexit
import shutil
import argparse
import tempfile
import subprocess
import time

# Reduce CUDA fragmentation after hundreds of sequential video-inference calls;
# must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# ----------------------------------------------------------------------------
# Windows/torchcodec compatibility patch
# ----------------------------------------------------------------------------
# transformers samples frame indices with np.arange(..., dtype=int), which is
# 32-bit on Windows. torchcodec's VideoDecoder.get_frames_at expects 64-bit
# (Long) indices, so on Windows this raises:
#   RuntimeError: expected scalar type Long but found Int
# The processor doesn't expose a `video_load_backend` override for this model,
# so we patch the actual call site instead - this fixes it regardless of which
# internal code path built the indices.
try:
    from torchcodec.decoders import VideoDecoder as _VideoDecoder

    _original_get_frames_at = _VideoDecoder.get_frames_at

    def _patched_get_frames_at(self, indices, *args, **kwargs):
        indices = np.asarray(indices, dtype=np.int64)
        return _original_get_frames_at(self, indices, *args, **kwargs)

    _VideoDecoder.get_frames_at = _patched_get_frames_at
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
FOOTAGE_DIR = r".\pickle + masti 20260802 - Copy"
HI_RES_EXT = ".MP4"
MODEL_ID = "Qwen/Qwen3.5-9B"      # natively multimodal, confirmed real model on HF
ANALYSIS_FPS = 1.0                # how densely the model samples the LRF proxy
CHUNK_DURATION = 5                # seconds per model analysis chunk

DEFAULT_BRIEF = (
    "This is a day-in-the-life vlog. Capture meals, travel, key conversations, "
    "and any main events or performances. Highlight excitement, laughter, and "
    "scene transitions."
)

# Token guardrails: small caps keep attention scratch space tiny on 16GB VRAM.
PASS1_MAX_TOKENS = 350      # frames -> structured chunk metadata JSON
STORY_MAX_TOKENS = 300      # text-only brief parsing
COMPOSER_MAX_TOKENS = 2000  # single-call composition (fallback path only)
# Prompt-size guardrails. A 16GB card running 4-bit Qwen3.5-9B comfortably
# prefills ~6-8k prompt tokens; beyond that the KV cache + attention scratch
# space triggers CUDA OOM. ~4 chars/token.
DIGEST_CHAR_BUDGET = 24000
# Hierarchical composition: one small outline call over a condensed digest of
# the whole day, then one detailed selection call per story beat.
OUTLINE_MAX_TOKENS = 700
BEAT_MAX_TOKENS = 900
MIN_BEATS = 4
MAX_BEATS = 8

# Preview renders come from the low-res LRF proxies (1:1 timeline with the
# hi-res MP4s) so iteration is fast; the final render uses the MP4s.
PREVIEW_SUFFIX = "_preview.mp4"
PREVIEW_BITRATE = "6000k"
FINAL_BITRATE = "55000k"

FALLBACK_SCORE_WEIGHTS = (0.35, 0.25, 0.2, 0.3)
FALLBACK_SCORE_THRESHOLD = 0.5
LOUDNESS_BONUS_DB = -6.0

# Speech transcription (faster-whisper). "small" int8 uses ~1GB VRAM alongside
# the ~8GB 4-bit Qwen - fits a 16GB card since all calls are sequential.
WHISPER_MODEL_ID = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "int8"
TRANSCRIPT_STORE_CHARS = 500   # cap stored per-chunk transcript length
TRANSCRIPT_DIGEST_CHARS = 200  # cap transcript length inside the composer digest

LIBRARY_SCHEMA = 3  # v3 stores chunks per clip fingerprint (sticky analysis)

# Knowledge graph: entity membership is derived deterministically from the
# pass-1 chunk metadata; the LLM only merges near-duplicate names and spots
# people/threads, so its replies stay tiny and cannot truncate mid-JSON.
# Cached per clip with the same lrf_size fingerprint as the library.
KG_SCHEMA = 2
KG_MERGE_MAX_TOKENS = 350
KG_ENTITY_MAX_TOKENS = 300
ELEMENT_MAX_TOKENS = 500
MAX_ELEMENTS = 12        # coverage units the graph composer weaves together
MIN_AUTO_CUTS = 8
MAX_AUTO_CUTS = 30

# Qwen3.5's hybrid Gated DeltaNet + Attention architecture has open bitsandbytes
# compatibility reports as of mid-2026 (load failures / bad output on the 27B
# checkpoint; no official 4-bit checkpoints from the Qwen team yet). If loading
# errors out or you get garbage JSON, flip this to False - the 9B dense model
# fits comfortably in bf16 on a 16GB 5080 anyway.
USE_4BIT = True

model = None
processor = None


def load_model():
    global model, processor
    if model is not None:
        return
    print("Spinning up Qwen3.5 on your RTX 5080 VRAM...")

    quantization_config = None
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)


# ----------------------------------------------------------------------------
# WorkDir - single session folder for every temp artifact, removed at exit
# ----------------------------------------------------------------------------
class WorkDir:
    def __init__(self):
        self.path = tempfile.mkdtemp(prefix="autoedit_")
        atexit.register(self.cleanup)
        print(f"Session work folder: {self.path}")

    def file(self, name):
        return os.path.join(self.path, name)

    def remove(self, path):
        try:
            os.remove(path)
        except OSError:
            pass

    def cleanup(self):
        shutil.rmtree(self.path, ignore_errors=True)


# ----------------------------------------------------------------------------
# Shared LLM helpers (the only code that touches the model directly)
# ----------------------------------------------------------------------------
def _run_generate(messages, max_new_tokens):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            do_sample=False,
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip() if decoded else ""


def _llm_text(system, user, max_new_tokens):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    return _run_generate(messages, max_new_tokens)


def _llm_video(video_path, system, user, max_new_tokens):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": ANALYSIS_FPS},
            {"type": "text", "text": user},
        ]},
    ]
    try:
        return _run_generate(messages, max_new_tokens)
    finally:
        # Video calls allocate the big vision buffers - release them before the
        # next chunk so peak VRAM stays flat across the whole clip.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _json_default(o):
    """Last-resort coercion for non-native types (np.bool_, np.float64, ...)."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def parse_json_response(output_text):
    """Robustly pulls the first JSON object or list out of the model's reply."""
    clean = output_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                continue

    print(f"Formatting warning. Raw text was: {output_text}")
    return None


def salvage_selections(output_text):
    """Recover complete selection objects from a truncated composer reply.

    When generation hits max_new_tokens mid-JSON, the reply is unparseable as a
    whole but usually contains many complete {"chunk_id":..,"role":..,"reason":..}
    objects before the cut-off - extract those individually instead of
    discarding everything."""
    items = []
    for match in re.finditer(r"\{[^{}]*\"chunk_id\"[^{}]*\}", output_text, re.DOTALL):
        try:
            item = json.loads(match.group(0))
        except Exception:
            continue
        if isinstance(item, dict) and "chunk_id" in item:
            items.append(item)
    if not items:
        return None
    title_match = re.search(r'"story_title"\s*:\s*"([^"]*)"', output_text)
    return {
        "story_title": title_match.group(1) if title_match else "",
        "selections": items,
    }


def parse_id_ranges(spec, n):
    """Compact id spec -> sorted clip-local ids clamped to 0..n-1.

    Accepts "3-7,12" style strings (the KG prompt asks for these so replies
    stay tiny) or a plain list of ints; garbage tokens are skipped."""
    ids = set()
    if isinstance(spec, (list, tuple)):
        tokens = spec
    else:
        tokens = str(spec).split(",")
    for token in tokens:
        token = str(token).strip()
        if not token:
            continue
        m = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", token)
        try:
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                ids.update(range(max(0, lo), min(n - 1, hi) + 1))
            else:
                cid = int(token)
                if 0 <= cid < n:
                    ids.add(cid)
        except (TypeError, ValueError):
            continue
    return sorted(ids)


# ----------------------------------------------------------------------------
# File discovery
# ----------------------------------------------------------------------------
def get_video_pairs(folder):
    # On case-insensitive filesystems (Windows, default macOS) "*.LRF" and "*.lrf"
    # match the exact same files, so combining both patterns double-counted every
    # clip (and processed each one twice). A single bracket-expansion pattern
    # avoids that on any OS, and we still dedupe defensively below.
    lrf_files = glob.glob(os.path.join(folder, "*.[Ll][Rr][Ff]"))

    seen = set()
    pairs = []
    for lrf_path in sorted(lrf_files):
        key = os.path.normcase(os.path.abspath(lrf_path))
        if key in seen:
            continue
        seen.add(key)

        base_name, _ = os.path.splitext(lrf_path)
        hi_res_path = base_name + HI_RES_EXT
        if not os.path.exists(hi_res_path):
            hi_res_path = base_name + HI_RES_EXT.lower()
        if os.path.exists(hi_res_path):
            pairs.append({"lrf": lrf_path, "hires": hi_res_path})
    return pairs


def _find_media_tools():
    """Locate ffmpeg/ffprobe: PATH first, then moviepy's bundled imageio-ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            if not ffprobe:
                sibling = os.path.join(
                    os.path.dirname(ffmpeg),
                    "ffprobe" + (".exe" if os.name == "nt" else ""),
                )
                if os.path.exists(sibling):
                    ffprobe = sibling
        except ImportError:
            pass
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install it and add it to PATH (or pip install imageio-ffmpeg)."
        )
    return ffmpeg, ffprobe


FFMPEG_BIN, FFPROBE_BIN = None, None


def get_media_tools():
    global FFMPEG_BIN, FFPROBE_BIN
    if FFMPEG_BIN is None:
        FFMPEG_BIN, FFPROBE_BIN = _find_media_tools()
    return FFMPEG_BIN, FFPROBE_BIN


_NVENC_AVAILABLE = None


def nvenc_available():
    """Functional NVENC probe (encoder can be compiled in but lack a driver)."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        ffmpeg, _ = get_media_tools()
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, check=False,
        )
        _NVENC_AVAILABLE = result.returncode == 0
        print("NVENC hardware encoder: "
              + ("available - rendering on the GPU." if _NVENC_AVAILABLE
                 else "not available - rendering on the CPU (libx264)."))
    return _NVENC_AVAILABLE


# ----------------------------------------------------------------------------
# ChunkLibrary - persistent metadata library covering every chunk of every clip
# ----------------------------------------------------------------------------
class ChunkLibrary:
    """Per-clip sticky analysis: each clip's chunks live under that clip's own
    fingerprint, so adding/removing/renaming one clip never invalidates the
    others. Entries for clips no longer on disk are kept so the knowledge
    survives if the clip comes back."""

    def __init__(self):
        self.chunks = []        # active flat list for the current pairs
        self.clip_entries = {}  # source -> {"lrf_name", "lrf_size", "chunks"}

    def add(self, meta):
        meta["library_id"] = len(self.chunks)
        self.chunks.append(meta)

    def by_id(self, library_id):
        if 0 <= library_id < len(self.chunks):
            return self.chunks[library_id]
        return None

    @staticmethod
    def clip_fingerprint(pair):
        return os.path.getsize(pair["lrf"])

    def commit_clip(self, pair):
        """Sticky one clip's finished analysis to its fingerprint."""
        source = os.path.basename(pair["hires"])
        self.clip_entries[source] = {
            "lrf_name": os.path.basename(pair["lrf"]),
            "lrf_size": self.clip_fingerprint(pair),
            "chunks": [c for c in self.chunks if c["source"] == source],
        }

    def save(self, path):
        payload = {
            "schema": LIBRARY_SCHEMA,
            "analysis_fps": ANALYSIS_FPS,
            "chunk_duration": CHUNK_DURATION,
            "clips": self.clip_entries,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        total = sum(len(e["chunks"]) for e in self.clip_entries.values())
        print(f"Chunk library saved to {path} "
              f"({len(self.clip_entries)} clips, {total} chunks)")

    @classmethod
    def load(cls, path, pairs):
        """(library, status) for the current pairs. status maps each source to
        'analyzed' | 'changed' | 'new'; the active chunk list covers only
        'analyzed' clips, renumbered in pairs order."""
        lib = cls()
        payload = None
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as e:
                print(f"Could not read library cache {path}: {e}")
        if payload and (payload.get("analysis_fps") != ANALYSIS_FPS
                        or payload.get("chunk_duration") != CHUNK_DURATION):
            print("Library cache ignored: analysis settings changed.")
            payload = None
        if payload:
            schema = payload.get("schema")
            if schema == LIBRARY_SCHEMA:
                lib.clip_entries = payload.get("clips", {})
            elif schema == 2:
                lib.clip_entries = cls._migrate_v2(payload)
                if lib.clip_entries:
                    print(f"Migrated library cache to per-clip format "
                          f"({len(lib.clip_entries)} clips carried over).")
            else:
                print("Library cache ignored: incompatible schema - re-analyzing.")
        status = {}
        for pair in pairs:
            source = os.path.basename(pair["hires"])
            entry = lib.clip_entries.get(source)
            if entry is None:
                status[source] = "new"
            elif entry.get("lrf_size") != cls.clip_fingerprint(pair):
                status[source] = "changed"
            else:
                status[source] = "analyzed"
                for meta in entry["chunks"]:
                    lib.add(meta)
        return lib, status

    @staticmethod
    def _migrate_v2(payload):
        """Old format: one global fingerprint + flat chunk list. Regroup per
        clip so existing analysis carries over."""
        sizes = {os.path.splitext(fp["name"])[0]: fp
                 for fp in payload.get("fingerprint", [])}
        by_source = {}
        for meta in payload.get("chunks", []):
            by_source.setdefault(meta["source"], []).append(meta)
        all_sources = list(by_source)
        entries = {}
        for source in payload.get("analyzed_sources", all_sources):
            fp = sizes.get(os.path.splitext(source)[0])
            if fp is None:
                continue
            entries[source] = {"lrf_name": fp["name"], "lrf_size": fp["lrf_size"],
                               "chunks": by_source.get(source, [])}
        return entries

    @classmethod
    def load_if_valid(cls, path, pairs):
        """A library covering every current clip, or None."""
        if not pairs:
            return None
        lib, status = cls.load(path, pairs)
        pending = [s for s, st in status.items() if st != "analyzed"]
        if pending:
            done = len(pairs) - len(pending)
            if done:
                print(f"Library covers {done}/{len(pairs)} clips - "
                      "the rest still need analysis.")
            return None
        print(f"Library cache hit: {path} ({len(lib.chunks)} chunks) "
              "- skipping video analysis.")
        return lib


# ----------------------------------------------------------------------------
# Knowledge graph - entities per clip (LLM pass), derived edges, coverage units
# ----------------------------------------------------------------------------
KG_ENTITY_TYPES = ("person", "place", "activity", "object", "mood", "thread")
# priority when choosing the day's coverage units (threads/story first)
KG_TYPE_PRIORITY = {"thread": 0, "place": 1, "activity": 2,
                    "person": 3, "object": 4, "mood": 5}


def kg_normalize(name):
    return " ".join(str(name).lower().split())


class GraphExtractor:
    """Per-clip entity extraction with tiny LLM replies.

    Entity membership (the bulky data) is derived deterministically from the
    pass-1 chunk metadata; the LLM only (1) merges near-duplicate names into
    the day vocabulary and (2) names people/threads with compact id ranges.
    Both replies are a handful of small objects, so they cannot blow through
    the token budget the way full chunk_id arrays did."""

    SYSTEM = (
        "You are an archivist building a knowledge graph of one day of vlog "
        "footage. You only output raw JSON objects. Never talk to the user, "
        "never use markdown."
    )

    # per-type caps for deterministic seeds (place is never capped)
    SEED_CAPS = {"activity": 10, "object": 8, "mood": 5}

    @staticmethod
    def _seed_terms(chunks):
        """{(type, name): set(clip_local_ids)} straight from pass-1 fields -
        no LLM involved, so an analyzed clip always has graph entries."""
        terms = {}

        def note(etype, raw, idx):
            name = kg_normalize(raw)
            if name and name != "unknown":
                terms.setdefault((etype, name), set()).add(idx)

        for i, chunk in enumerate(chunks):
            note("place", chunk.get("scene", ""), i)
            for action in chunk.get("actions") or []:
                note("activity", action, i)
            for tag in chunk.get("visual_tags") or []:
                note("object", tag, i)
            for emotion in chunk.get("emotions") or []:
                note("mood", emotion, i)

        min_chunks = 1 if len(chunks) < 6 else 2
        terms = {k: v for k, v in terms.items() if len(v) >= min_chunks}
        for etype, cap in GraphExtractor.SEED_CAPS.items():
            typed = sorted((k for k in terms if k[0] == etype),
                           key=lambda k: (-len(terms[k]), k[1]))
            for key in typed[cap:]:
                del terms[key]
        return terms

    @staticmethod
    def _salvage_merges(output_text):
        items = []
        for match in re.finditer(r"\{[^{}]*\"from\"[^{}]*\}", output_text, re.DOTALL):
            try:
                item = json.loads(match.group(0))
            except Exception:
                continue
            if isinstance(item, dict) and "from" in item:
                items.append(item)
        return {"merges": items} if items else None

    @staticmethod
    def _salvage_semantic(output_text):
        items = []
        for match in re.finditer(r"\{[^{}]*\"chunks\"[^{}]*\}", output_text, re.DOTALL):
            try:
                item = json.loads(match.group(0))
            except Exception:
                continue
            if isinstance(item, dict) and "chunks" in item:
                items.append((match.start(), item))
        if not items:
            return None
        # objects after the "threads" key belong to the threads list
        threads_at = output_text.find('"threads"')
        people, threads = [], []
        for pos, item in items:
            if threads_at != -1 and pos > threads_at:
                threads.append(item)
            else:
                people.append(item)
        return {"people": people, "threads": threads}

    def _ask(self, user, max_tokens, salvage):
        """One LLM call -> parsed dict, trying the salvager before giving up."""
        try:
            raw = _llm_text(self.SYSTEM, user, max_tokens)
        except Exception as e:
            print(f"  graph extraction call failed: {e}")
            return None
        clean = raw.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        salvaged = salvage(raw)
        if salvaged is not None:
            return salvaged
        print(f"Formatting warning. Raw text was: {raw}")
        return None

    def _canonicalize(self, terms, vocabulary):
        """LLM call 1: merge near-duplicate term names (tiny output - only the
        merges themselves). Returns (merged_terms, ok)."""
        if not terms:
            return {}, True
        known = "\n".join(f"{t}: {', '.join(names)}"
                          for t, names in vocabulary.items() if names) or "none yet"

        def term_lines(keys):
            by_type = {}
            for etype, name in keys:
                by_type.setdefault(etype, []).append(name)
            return "\n".join(f"{t}: {', '.join(sorted(names))}"
                             for t, names in sorted(by_type.items()))

        def prompt(keys):
            return (
                "Below are entity terms detected in ONE video clip, plus the "
                "names already known from other clips of the same day. Merge "
                "near-duplicate names so the day's graph uses one canonical "
                "name per real-world entity (e.g. 'sandy beach' -> 'the "
                "beach').\n"
                "Rules:\n"
                "- Only output merges. A term without a merge keeps its name.\n"
                "- 'to' should be a KNOWN name when the term means the same "
                "thing; otherwise a cleaner short lowercase name.\n"
                "- Do not merge terms that mean different things. No merges "
                "is a fine answer.\n"
                'Output ONLY a JSON object: {"merges": [{"type", "from", '
                '"to"}, ...]} (empty list if nothing to merge).\n\n'
                f"KNOWN NAMES:\n{known}\n\n"
                f"THIS CLIP'S TERMS:\n{term_lines(keys)}"
            )

        parsed = self._ask(prompt(list(terms)), KG_MERGE_MAX_TOKENS,
                           self._salvage_merges)
        if parsed is None:
            # retry once with strictly less output pressure: only the biggest
            # half of the terms are offered for merging
            top = sorted(terms, key=lambda k: -len(terms[k]))
            top = top[:max(1, len(top) // 2)]
            parsed = self._ask(prompt(top), KG_MERGE_MAX_TOKENS,
                               self._salvage_merges)
        ok = parsed is not None

        merged = {}
        renames = {}
        if ok:
            for item in parsed.get("merges") or []:
                if not isinstance(item, dict):
                    continue
                etype = str(item.get("type", "")).strip().lower()
                src = kg_normalize(item.get("from", ""))
                dst = kg_normalize(item.get("to", ""))
                if etype in KG_ENTITY_TYPES and dst and (etype, src) in terms:
                    renames[(etype, src)] = (etype, dst)
        for key, ids in terms.items():
            target = renames.get(key, key)
            merged.setdefault(target, set()).update(ids)
        return merged, ok

    def _semantic_entities(self, chunks, vocabulary):
        """LLM call 2: people + threads from summaries/transcripts, with
        compact range strings for membership. Returns (entities, ok)."""
        known = [n for t in ("person", "thread") for n in vocabulary.get(t, [])]
        known_part = ", ".join(known) or "none yet"

        def prompt(char_cap, people_cap, thread_cap):
            lines = []
            for i, c in enumerate(chunks):
                speech = c.get("speech", {}).get("transcript", "")
                line = f"id={i} {c.get('one_line_summary', '')[:char_cap]}"
                if speech:
                    line += f' says:"{speech[:char_cap]}"'
                lines.append(line)
            return (
                "Below are the shots of ONE video clip, in order. Identify "
                "recurring PEOPLE and story THREADS (an ongoing storyline, "
                "e.g. 'finding the lost drone').\n"
                "Rules:\n"
                f"- At most {people_cap} people and {thread_cap} threads; only "
                "recurring/nameable ones. Empty lists are a fine answer.\n"
                "- Reuse a KNOWN name when it is the same person/thread.\n"
                "- Names are short lowercase phrases (1-4 words).\n"
                "- 'chunks' is a compact id spec like \"3-7,12\" - ranges and "
                "single ids, no spaces.\n"
                'Output ONLY a JSON object: {"people": [{"name", "chunks"}], '
                '"threads": [{"name", "chunks"}]}.\n\n'
                f"KNOWN NAMES: {known_part}\n\n"
                "SHOTS:\n" + "\n".join(lines)
            )

        parsed = self._ask(prompt(200, 4, 2), KG_ENTITY_MAX_TOKENS,
                           self._salvage_semantic)
        if parsed is None:
            parsed = self._ask(prompt(60, 2, 1), KG_ENTITY_MAX_TOKENS,
                               self._salvage_semantic)
        ok = parsed is not None

        entities = []
        if ok:
            for etype, group in (("person", "people"), ("thread", "threads")):
                for item in parsed.get(group) or []:
                    if not isinstance(item, dict):
                        continue
                    name = kg_normalize(item.get("name", ""))
                    ids = parse_id_ranges(item.get("chunks", ""), len(chunks))
                    if name and ids:
                        entities.append({"type": etype, "name": name,
                                         "chunk_ids": ids})
        return entities, ok

    def extract_clip(self, chunks, vocabulary):
        """chunks: this clip's chunks in order. vocabulary: {type: [names]} of
        entities already known from other clips. Returns (entities, enriched):
        the validated entity list with clip-local chunk_ids, and whether both
        LLM calls produced usable output (False -> retried on next sync)."""
        if not chunks:
            return [], True
        vocabulary = vocabulary or {}
        seeds = self._seed_terms(chunks)
        merged, merge_ok = self._canonicalize(seeds, vocabulary)
        semantic, semantic_ok = self._semantic_entities(chunks, vocabulary)

        entities = []
        seen = set()
        for (etype, name), ids in merged.items():
            key = (etype, name)
            if key in seen or not ids:
                continue
            seen.add(key)
            entities.append({"type": etype, "name": name,
                             "chunk_ids": sorted(ids)})
        for ent in semantic:
            key = (ent["type"], ent["name"])
            if key in seen:
                continue
            seen.add(key)
            entities.append(ent)
        return entities, merge_ok and semantic_ok


class KnowledgeGraphStore:
    """Persistence twin of ChunkLibrary: per-clip entity entries keyed by the
    same lrf_size fingerprint, so graph knowledge is sticky and resumes."""

    def __init__(self):
        self.clip_entries = {}  # source -> {"lrf_name", "lrf_size", "entities", "enriched"}

    def commit_clip(self, pair, entities, enriched=True):
        source = os.path.basename(pair["hires"])
        self.clip_entries[source] = {
            "lrf_name": os.path.basename(pair["lrf"]),
            "lrf_size": ChunkLibrary.clip_fingerprint(pair),
            "entities": entities,
            "enriched": bool(enriched),
        }

    def is_current(self, pair):
        entry = self.clip_entries.get(os.path.basename(pair["hires"]))
        return bool(entry) and entry.get("lrf_size") == ChunkLibrary.clip_fingerprint(pair)

    def is_enriched(self, pair):
        entry = self.clip_entries.get(os.path.basename(pair["hires"]))
        return bool(entry) and bool(entry.get("enriched"))

    def vocabulary(self):
        """{type: [names]} of everything known so far, for prompt reuse."""
        vocab = {t: [] for t in KG_ENTITY_TYPES}
        seen = set()
        for entry in self.clip_entries.values():
            for ent in entry["entities"]:
                key = (ent["type"], ent["name"])
                if key not in seen:
                    seen.add(key)
                    vocab[ent["type"]].append(ent["name"])
        return vocab

    def save(self, path):
        payload = {"schema": KG_SCHEMA, "analysis_fps": ANALYSIS_FPS,
                   "chunk_duration": CHUNK_DURATION, "clips": self.clip_entries}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        total = sum(len(e["entities"]) for e in self.clip_entries.values())
        print(f"Knowledge graph saved to {path} "
              f"({len(self.clip_entries)} clips, {total} entities)")

    @classmethod
    def load(cls, path, pairs):
        """(store, status) like ChunkLibrary.load: each source is 'analyzed' |
        'changed' | 'new' against the current fingerprints."""
        store = cls()
        payload = None
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as e:
                print(f"Could not read knowledge graph cache {path}: {e}")
        if payload and (payload.get("schema") != KG_SCHEMA
                        or payload.get("analysis_fps") != ANALYSIS_FPS
                        or payload.get("chunk_duration") != CHUNK_DURATION):
            print("Knowledge graph cache ignored: settings or schema changed.")
            payload = None
        if payload:
            store.clip_entries = payload.get("clips", {})
        status = {}
        for pair in pairs:
            source = os.path.basename(pair["hires"])
            entry = store.clip_entries.get(source)
            if entry is None:
                status[source] = "new"
            elif entry.get("lrf_size") != ChunkLibrary.clip_fingerprint(pair):
                status[source] = "changed"
            else:
                status[source] = "analyzed"
        return store, status


class KnowledgeGraph:
    """The assembled day: entity nodes merged across clips (by type+name),
    chunk references resolved to library_ids, derived edges computed on build.
    Nothing here is persisted - it is rebuilt from the store + library, so it
    can never drift out of sync."""

    def __init__(self):
        self.entities = {}   # key "type:name" -> {key, type, name, chunk_ids}
        self.co_occurs = {}  # frozenset({key_a, key_b}) -> weight

    @classmethod
    def build(cls, store, library):
        graph = cls()
        by_clip = {}
        for chunk in library.chunks:
            by_clip.setdefault(chunk["source"], []).append(chunk)
        for source, entry in store.clip_entries.items():
            clip_chunks = by_clip.get(source)
            if not clip_chunks:
                continue  # clip removed or not in the active library
            for ent in entry["entities"]:
                lids = [clip_chunks[i]["library_id"] for i in ent["chunk_ids"]
                        if 0 <= i < len(clip_chunks)]
                if not lids:
                    continue
                key = f"{ent['type']}:{ent['name']}"
                node = graph.entities.setdefault(
                    key, {"key": key, "type": ent["type"], "name": ent["name"],
                          "chunk_ids": []})
                node["chunk_ids"].extend(lids)
        order = {c["library_id"]: (c["source"], c["start"])
                 for c in library.chunks}
        for node in graph.entities.values():
            node["chunk_ids"] = sorted(set(node["chunk_ids"]),
                                       key=lambda i: order[i])
        # co-occurrence: entities sharing a chunk
        by_chunk = {}
        for node in graph.entities.values():
            for lid in node["chunk_ids"]:
                by_chunk.setdefault(lid, []).append(node["key"])
        for keys in by_chunk.values():
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    pair = frozenset((a, b))
                    graph.co_occurs[pair] = graph.co_occurs.get(pair, 0) + 1
        return graph

    def elements_of_day(self, max_elements=MAX_ELEMENTS):
        """The day's coverage units: the most substantial entities, story
        threads first, each with its chunks in chronological order."""
        nodes = sorted(
            self.entities.values(),
            key=lambda n: (KG_TYPE_PRIORITY[n["type"]], -len(n["chunk_ids"])))
        picked, covered = [], set()
        for node in nodes:
            if len(picked) >= max_elements:
                break
            fresh = [i for i in node["chunk_ids"] if i not in covered]
            # an element must add real coverage, not restate another element
            if len(fresh) < max(1, len(node["chunk_ids"]) // 3):
                continue
            picked.append(node)
            covered.update(node["chunk_ids"])
        return picked

    def chunks_for_element(self, key):
        node = self.entities.get(key)
        return list(node["chunk_ids"]) if node else []

    def neighbors(self, key):
        out = []
        for pair, weight in self.co_occurs.items():
            if key in pair:
                (other,) = pair - {key}
                out.append((other, weight))
        out.sort(key=lambda x: -x[1])
        return out

    def subgraph(self, candidate_ids):
        """The graph restricted to a window of chunks (gap fill)."""
        allowed = set(candidate_ids)
        sub = KnowledgeGraph()
        for key, node in self.entities.items():
            kept = [i for i in node["chunk_ids"] if i in allowed]
            if kept:
                sub.entities[key] = dict(node, chunk_ids=kept)
        for pair, weight in self.co_occurs.items():
            if all(k in sub.entities for k in pair):
                sub.co_occurs[pair] = weight
        return sub

    def to_json(self, library):
        """Explorer payload: nodes/edges plus chunk refs for the side panel."""
        chunk_meta = {
            c["library_id"]: {"library_id": c["library_id"], "source": c["source"],
                              "start": c["start"], "end": c["end"],
                              "summary": c["one_line_summary"],
                              "interest": c.get("interest_score", 0)}
            for c in library.chunks}
        nodes = []
        for node in sorted(self.entities.values(), key=lambda n: n["key"]):
            nodes.append({"id": node["key"], "type": node["type"],
                          "name": node["name"],
                          "chunk_count": len(node["chunk_ids"]),
                          "chunks": [chunk_meta[i] for i in node["chunk_ids"]
                                     if i in chunk_meta]})
        edges = [{"a": min(pair), "b": max(pair), "weight": weight}
                 for pair, weight in self.co_occurs.items()]
        edges.sort(key=lambda e: (e["a"], e["b"]))
        element_keys = [n["key"] for n in self.elements_of_day()]
        return {"nodes": nodes, "edges": edges, "elements": element_keys}


# ----------------------------------------------------------------------------
# VideoPreprocessor - chunk bounds, ffmpeg chunk extraction, audio metrics
# ----------------------------------------------------------------------------
# Pass 1 deliberately avoids moviepy: in moviepy 2.x subclipped() shares the
# parent clip's ffmpeg reader processes, so closing a subclip kills the parent
# reader and every later audio read fails with "'NoneType' has no attribute
# 'stdout'". Direct ffmpeg calls have no shared state (and read LRF directly,
# so no .mp4 alias files are needed either).
class VideoPreprocessor:
    def __init__(self, workdir):
        self.workdir = workdir

    @staticmethod
    def probe_duration(path):
        _, ffprobe = get_media_tools()
        if not ffprobe:
            raise RuntimeError("ffprobe not found. Install ffmpeg and add it to PATH.")
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=False,
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            print(f"ffprobe could not read duration of {path}: {result.stderr.strip()}")
            return 0.0

    def extract_rolling_chunks(self, duration, chunk_duration=CHUNK_DURATION, max_chunks=None):
        num_chunks = int(np.ceil(duration / chunk_duration))
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_duration
            chunks.append({
                "chunk_id": i,
                "start": float(start),
                "end": float(min(start + chunk_duration, duration)),
            })
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        return chunks

    def extract_chunk_media(self, lrf_path, chunk_id, start, end):
        """One ffmpeg call -> video-only mp4 for the vision model + 16kHz mono
        wav for whisper/loudness. Returns (mp4_path, wav_path_or_None)."""
        ffmpeg, _ = get_media_tools()
        mp4_path = self.workdir.file(f"chunk_{chunk_id:04d}.mp4")
        wav_path = self.workdir.file(f"chunk_{chunk_id:04d}.wav")
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", lrf_path,
            "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-an", mp4_path,
            "-map", "0:a:0?", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not os.path.exists(mp4_path):
            raise RuntimeError(f"ffmpeg chunk extraction failed: {result.stderr.strip()}")
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) <= 44:  # empty WAV header
            self.workdir.remove(wav_path)
            wav_path = None
        return mp4_path, wav_path

    @staticmethod
    def measure_audio(wav_path):
        result = {"has_audio": False, "loudness_peak_db": -120.0, "speech_likely": False}
        if wav_path is None:
            return result
        try:
            with wave.open(wav_path, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16)
            if samples.size == 0:
                return result
            peak = float(np.max(np.abs(samples))) / 32768.0
            if peak > 0:
                # Cast to native Python types: np.float64/np.bool_ leak into the
                # library dict otherwise and json.dump chokes on np.bool_
                # ("Object of type bool_ is not JSON serializable").
                result["has_audio"] = True
                result["loudness_peak_db"] = float(round(20.0 * np.log10(peak), 1))
                result["speech_likely"] = bool(result["loudness_peak_db"] > -30.0)
        except Exception as e:
            print(f"Audio measurement failed for {wav_path}: {e}")
        return result


# ----------------------------------------------------------------------------
# SpeechTranscriber - faster-whisper on the chunk WAVs (the composer can't hear)
# ----------------------------------------------------------------------------
class SpeechTranscriber:
    def __init__(self):
        self._model = None
        self._warned = False

    def _ensure_loaded(self):
        if self._model is not None:
            return True
        if WhisperModel is None:
            if not self._warned:
                print("faster-whisper not installed (pip install faster-whisper); "
                      "continuing without speech transcription.")
                self._warned = True
            return False
        print(f"Loading whisper '{WHISPER_MODEL_ID}' ({WHISPER_DEVICE}/{WHISPER_COMPUTE})...")
        try:
            self._model = WhisperModel(
                WHISPER_MODEL_ID, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
            )
        except Exception as e:
            print(f"Whisper load failed ({e}); retrying on CPU.")
            try:
                self._model = WhisperModel(WHISPER_MODEL_ID, device="cpu", compute_type="int8")
            except Exception as e2:
                print(f"Whisper CPU load also failed ({e2}); continuing without speech.")
                self._warned = True
                return False
        return True

    def transcribe(self, wav_path):
        empty = {"transcript": "", "language": "", "has_speech": False}
        if wav_path is None or not self._ensure_loaded():
            return empty
        try:
            segments, info = self._model.transcribe(wav_path, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments)
            text = re.sub(r"\s+", " ", text).strip()[:TRANSCRIPT_STORE_CHARS]
            return {
                "transcript": text,
                "language": getattr(info, "language", "") or "",
                "has_speech": bool(text),
            }
        except Exception as e:
            print(f"Transcription failed for {wav_path}: {e}")
            return empty

    def unload(self):
        """Free whisper's VRAM after pass 1 so the composer's long text prompt
        has maximum headroom."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Whisper unloaded from VRAM.")


# ----------------------------------------------------------------------------
# StoryAgent - parses the user brief into structured criteria
# ----------------------------------------------------------------------------
class StoryAgent:
    SYSTEM = (
        "You are a video editing story planner. You only output raw JSON objects. "
        "Never talk to the user."
    )

    def parse_user_brief(self, brief):
        user = (
            "Convert this vlog editing brief into a JSON object with exactly these keys: "
            '"themes" (list of short lowercase keywords), '
            '"emotional_targets" (list of short lowercase emotion words), '
            '"must_capture" (list of specific moments to include), '
            '"avoid" (list of things to skip). '
            "Output ONLY the JSON object, no explanation or markdown.\n\n"
            f"Brief: {brief}"
        )
        parsed = parse_json_response(_llm_text(self.SYSTEM, user, STORY_MAX_TOKENS))
        if not isinstance(parsed, dict):
            print("StoryAgent: brief parsing failed, using brief verbatim as theme.")
            parsed = {}
        story = {
            "brief": brief,
            "themes": [str(t).lower() for t in parsed.get("themes", []) if t] or [brief.lower()],
            "emotional_targets": [str(e).lower() for e in parsed.get("emotional_targets", [])],
            "must_capture": [str(m) for m in parsed.get("must_capture", [])],
            "avoid": [str(a).lower() for a in parsed.get("avoid", [])],
        }
        print(f"StoryAgent criteria: {json.dumps(story, indent=2)}")
        return story


# ----------------------------------------------------------------------------
# ChunkAnalyzer - the only agent that ever sees pixels (pass 1 only)
# ----------------------------------------------------------------------------
class ChunkAnalyzer:
    VISION_SYSTEM = (
        "You are a video shot logger. You watch short clips and output raw JSON "
        "metadata. Never talk to the user, never use markdown."
    )

    @staticmethod
    def _neutral_metadata():
        return {
            "visual_tags": [],
            "actions": [],
            "people_count": 0,
            "emotions": [],
            "scene": "unknown",
            "transition_flag": False,
            "one_line_summary": "unreadable chunk",
            "interest_score": 0.0,
        }

    def analyze_chunk(self, chunk_path):
        """Frames at ANALYSIS_FPS -> structured metadata JSON."""
        user = (
            "Watch this short clip and describe it as a JSON object with exactly "
            "these keys:\n"
            '"visual_tags": list of short lowercase keywords for setting/objects,\n'
            '"actions": list of short lowercase phrases for what people are doing,\n'
            '"people_count": integer number of visible people,\n'
            '"emotions": list of visible emotions (e.g. "happy", "laughter", "excited"),\n'
            '"scene": short lowercase location description,\n'
            '"transition_flag": true if the shot shows movement between locations '
            "(walking out a door, boarding a vehicle, entering a venue),\n"
            '"one_line_summary": one sentence describing the moment,\n'
            '"interest_score": float 0.0-1.0 for how visually engaging this clip is.\n'
            "Output ONLY the JSON object."
        )
        try:
            parsed = parse_json_response(
                _llm_video(chunk_path, self.VISION_SYSTEM, user, PASS1_MAX_TOKENS)
            )
        except Exception as e:
            print(f"Chunk analysis failed for {chunk_path}: {e}")
            parsed = None

        meta = self._neutral_metadata()
        if isinstance(parsed, dict):
            meta["visual_tags"] = [str(t).lower() for t in parsed.get("visual_tags", []) if t]
            meta["actions"] = [str(a).lower() for a in parsed.get("actions", []) if a]
            try:
                meta["people_count"] = int(parsed.get("people_count", 0))
            except (TypeError, ValueError):
                pass
            meta["emotions"] = [str(e).lower() for e in parsed.get("emotions", []) if e]
            meta["scene"] = str(parsed.get("scene", "unknown")).lower()
            meta["transition_flag"] = bool(parsed.get("transition_flag", False))
            meta["one_line_summary"] = str(parsed.get("one_line_summary", ""))
            try:
                meta["interest_score"] = max(0.0, min(1.0, float(parsed.get("interest_score", 0.0))))
            except (TypeError, ValueError):
                pass
        return meta


# ----------------------------------------------------------------------------
# StoryComposer - reads the ENTIRE library and builds one coherent story
# ----------------------------------------------------------------------------
class StoryComposer:
    SYSTEM = (
        "You are a documentary film editor composing a vlog from a shot library. "
        "You only output raw JSON objects. Never talk to the user, never use markdown."
    )
    ROLES = ("setting", "buildup", "peak", "transition", "outro")

    @staticmethod
    def _digest_line(chunk):
        speech = chunk.get("speech", {}).get("transcript", "")
        speech_part = f'"{speech[:TRANSCRIPT_DIGEST_CHARS]}"' if speech else "none"
        return (
            f"id={chunk['library_id']} [{chunk['source']} "
            f"{chunk['start']:.0f}-{chunk['end']:.0f}s] "
            f"scene={chunk['scene']} "
            f"tags={','.join(chunk['visual_tags'][:6]) or 'none'} "
            f"emotions={','.join(chunk['emotions']) or 'none'} "
            f"people={chunk['people_count']} "
            f"transition={'Y' if chunk['transition_flag'] else 'N'} "
            f"audio={chunk['audio']['loudness_peak_db']:.0f}dB "
            f"interest={chunk['interest_score']:.2f} "
            f"speech={speech_part} "
            f":: \"{chunk['one_line_summary']}\""
        )

    @staticmethod
    def _prefilter_heuristic(chunk):
        score = chunk["interest_score"]
        score += 0.2 * min(len(chunk["emotions"]), 2)
        if chunk["transition_flag"]:
            score += 0.2
        if chunk.get("speech", {}).get("has_speech"):
            score += 0.25
        if chunk["audio"]["loudness_peak_db"] > LOUDNESS_BONUS_DB:
            score += 0.1
        return score

    @staticmethod
    def _outline_line(chunk, summary_chars=80):
        speech_flag = "Y" if chunk.get("speech", {}).get("has_speech") else "N"
        actions = ",".join(chunk.get("actions", [])[:2]) or "none"
        trans = " | T" if chunk["transition_flag"] else ""
        return (
            f"id={chunk['library_id']} [{chunk['source']} "
            f"{chunk['start']:.0f}-{chunk['end']:.0f}s] {chunk['scene']} | {actions} "
            f"| speech:{speech_flag}{trans} | i={chunk['interest_score']:.1f} "
            f":: \"{chunk['one_line_summary'][:summary_chars]}\""
        )

    def _build_outline_digest(self, chunks):
        """Condensed whole-day digest for the outline call. If even that is over
        budget, coarsen deterministically: shorter summaries first, then drop the
        weakest chunks - but never transitions or speech (they anchor the arc)."""
        def total(lines):
            return sum(len(line) + 1 for line in lines)

        lines = [self._outline_line(c) for c in chunks]
        if total(lines) <= DIGEST_CHAR_BUDGET:
            return "\n".join(lines)

        summary_chars = 40
        included = list(chunks)
        lines = [self._outline_line(c, summary_chars) for c in included]
        droppable = sorted(
            (c for c in included
             if not c["transition_flag"] and not c.get("speech", {}).get("has_speech")),
            key=self._prefilter_heuristic,
        )
        while droppable and total(lines) > DIGEST_CHAR_BUDGET:
            included.remove(droppable.pop(0))
            lines = [self._outline_line(c, summary_chars) for c in included]
        while len(included) > 1 and total(lines) > DIGEST_CHAR_BUDGET:
            included.remove(min(included, key=self._prefilter_heuristic))
            lines = [self._outline_line(c, summary_chars) for c in included]
        print(f"Outline digest coarsened: {len(included)}/{len(chunks)} chunks shown "
              "(beats still cover every chunk via id ranges).")
        return "\n".join(lines)

    def _compose_outline(self, library, story, target_cuts):
        """Stage 1: one small call over the condensed digest -> 4-8 story beats."""
        chunks = library.chunks
        if not chunks:
            return None
        min_id = min(c["library_id"] for c in chunks)
        max_id = max(c["library_id"] for c in chunks)
        digest = self._build_outline_digest(chunks)
        user = (
            "Below is a condensed shot library for one day of footage, in "
            "chronological order, plus the editing brief. Plan the narrative arc of "
            f"the vlog as a sequence of {MIN_BEATS}-{MAX_BEATS} story beats.\n"
            "Rules:\n"
            "- Beats are chronological and together cover the entire day: contiguous "
            "id ranges from the first id to the last.\n"
            "- The first beat opens/establishes the day; the last beat closes it.\n"
            f"- Distribute target_shots so they sum to about {target_cuts}; give more "
            "shots to beats with conversations and exciting peaks.\n"
            "Output ONLY a JSON object with exactly these keys:\n"
            '"story_title": short title,\n'
            '"beats": list of objects, each with "name" (2-4 words), "narrative" '
            '(one sentence describing what this beat tells), "start_id" and "end_id" '
            '(id range from the library), "target_shots" (integer).\n\n'
            f"EDITING BRIEF:\n{json.dumps(story)}\n\n"
            f"SHOT LIBRARY:\n{digest}"
        )
        try:
            parsed = parse_json_response(_llm_text(self.SYSTEM, user, OUTLINE_MAX_TOKENS))
        except Exception as e:
            print(f"Outline call failed: {e}")
            return None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("beats"), list):
            return None

        beats = []
        for item in parsed["beats"]:
            if not isinstance(item, dict):
                continue
            try:
                start_id = int(item.get("start_id"))
                end_id = int(item.get("end_id"))
            except (TypeError, ValueError):
                continue
            try:
                target = int(item.get("target_shots", 3))
            except (TypeError, ValueError):
                target = 3
            if end_id < start_id:
                start_id, end_id = end_id, start_id
            name = str(item.get("name", "")).strip() or f"beat {len(beats) + 1}"
            beats.append({
                "name": name,
                "narrative": str(item.get("narrative", "")).strip(),
                "start_id": max(min_id, min(start_id, max_id)),
                "end_id": max(min_id, min(end_id, max_id)),
                "target_shots": max(1, min(target, 8)),
            })
        beats.sort(key=lambda b: (b["start_id"], b["end_id"]))
        beats = beats[:MAX_BEATS]

        # Snap beats into contiguous, non-overlapping coverage of every chunk.
        normalized = []
        next_start = min_id
        for i, beat in enumerate(beats):
            if next_start > max_id:
                break
            beat["start_id"] = next_start
            end = max(beat["start_id"], min(beat["end_id"], max_id))
            # Don't let an overshooting beat swallow the ones after it.
            if i + 1 < len(beats):
                next_claim = beats[i + 1]["start_id"]
                if next_claim > beat["start_id"]:
                    end = min(end, next_claim - 1)
            beat["end_id"] = end
            normalized.append(beat)
            next_start = beat["end_id"] + 1
        if not normalized or len(normalized) < MIN_BEATS:
            return None
        normalized[-1]["end_id"] = max_id

        total = sum(b["target_shots"] for b in normalized)
        if total and target_cuts:
            scale = target_cuts / total
            for beat in normalized:
                beat["target_shots"] = max(1, min(8, round(beat["target_shots"] * scale)))
        return {"story_title": str(parsed.get("story_title", "")), "beats": normalized}

    def _select_for_beat(self, beat, chunks, story, prior_summary):
        """Stage 2: pick this beat's shots from its full-detail chunk digest."""
        ranked = sorted(chunks, key=self._prefilter_heuristic, reverse=True)
        included = list(chunks)
        lines = [self._digest_line(c) for c in included]
        while len(included) > 1 and sum(len(line) + 1 for line in lines) > DIGEST_CHAR_BUDGET:
            included.remove(min(included, key=self._prefilter_heuristic))
            lines = [self._digest_line(c) for c in included]
        digest = "\n".join(lines)
        so_far = prior_summary or "Nothing selected yet - this is the opening beat."
        user = (
            f'You are selecting shots for one beat of a vlog: "{beat["name"]}" - '
            f"{beat['narrative']}\n"
            "Below are this beat's shots in chronological order, the editing brief, "
            f"and the story so far. Pick about {beat['target_shots']} shots that tell "
            "this beat well.\n"
            "Rules:\n"
            "- Balance context and excitement: quieter setting shots matter as much "
            "as high-interest peaks.\n"
            "- Speech matters: prioritize meaningful conversation, jokes, reactions, "
            "and quotes that carry the story.\n"
            "- Do not repeat moments already covered in the story so far, and avoid "
            "near-duplicate shots within this beat.\n"
            "- Only use id values that appear below.\n"
            'Output ONLY a JSON object: {"selections": [list of objects, each with '
            '"chunk_id" (id number), "role" (one of "setting", "buildup", "peak", '
            '"transition", "outro"), and "reason" (one short sentence)]}.\n\n'
            f"STORY SO FAR:\n{so_far}\n\n"
            f"EDITING BRIEF:\n{json.dumps(story)}\n\n"
            f"SHOTS:\n{digest}"
        )
        try:
            parsed = parse_json_response(_llm_text(self.SYSTEM, user, BEAT_MAX_TOKENS))
        except Exception as e:
            print(f"  beat \"{beat['name']}\" call failed: {e}")
            parsed = None

        by_id = {c["library_id"]: c for c in chunks}
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
                chunk = by_id.get(library_id)
                if chunk is None or library_id in seen:
                    continue
                seen.add(library_id)
                role = item.get("role") if item.get("role") in self.ROLES else "buildup"
                picks.append({"chunk": chunk, "role": role,
                              "reason": str(item.get("reason", ""))})
        if not picks:
            print(f"  beat \"{beat['name']}\": no usable picks - using heuristic top shots.")
            picks = [{"chunk": c, "role": "buildup",
                      "reason": "strongest moment in this beat (heuristic fallback)"}
                     for c in ranked[:beat["target_shots"]]]
            picks.sort(key=lambda p: p["chunk"]["library_id"])
        return picks[:beat["target_shots"] + 2]

    def compose_with_graph(self, library, story, graph, target_len=None,
                           margin=3.0, max_total=None):
        """Coverage-driven composition: the knowledge graph's elements of the
        day are the beats; each element gets a small bounded LLM call over its
        own chunks. Cut count emerges from coverage (optionally scaled to an
        approximate target length in seconds). Returns [] if the graph is
        unusable so callers can fall back to the legacy compose path."""
        if not library.chunks:
            return []
        # ---- auto cut count ---------------------------------------------------
        avg_cut = CHUNK_DURATION + 2 * margin
        target_total = (max(MIN_AUTO_CUTS // 2, round(target_len / avg_cut))
                        if target_len else None)
        # long targets need more coverage units, not just more cuts per unit
        max_elements = MAX_ELEMENTS if not target_total else \
            max(MAX_ELEMENTS, min(2 * MAX_ELEMENTS, target_total // 3))
        elements = graph.elements_of_day(max_elements)
        if not elements:
            return []
        order = {c["library_id"]: (c["source"], c["start"]) for c in library.chunks}
        # brief matching: wanted elements gain weight, avoided ones fade
        wanted = [kg_normalize(w) for w in
                  (story.get("themes") or []) + (story.get("must_capture") or [])]
        avoided = [kg_normalize(a) for a in story.get("avoid") or []]

        def brief_bias(el):
            name = el["name"]
            if any(a and (a in name or name in a) for a in avoided):
                return 0.3
            if any(w and (w in name or name in w) for w in wanted):
                return 2.0
            return 1.0

        elements = sorted(elements,
                          key=lambda e: order.get(e["chunk_ids"][0], ("", 0)))

        total = target_total or max(MIN_AUTO_CUTS,
                                    min(MAX_AUTO_CUTS, 2 * len(elements)))
        if max_total:
            total = min(total, max_total)
        weights = [len(e["chunk_ids"]) * brief_bias(e) for e in elements]
        wsum = sum(weights) or 1
        # per-element cap scales with the target so long videos are reachable
        quota_cap = max(4, -(-total // len(elements)) + 1)
        quotas = [max(1, min(quota_cap, round(total * w / wsum))) for w in weights]
        print(f"\nStoryComposer: weaving {len(elements)} elements of the day "
              f"(~{sum(quotas)} cuts"
              + (f", targeting ~{target_len:.0f}s" if target_len else ", auto")
              + ")...")
        for el, q in zip(elements, quotas):
            print(f"  [{el['type']:>8}] {el['name']} "
                  f"({len(el['chunk_ids'])} chunks, ~{q} cuts)")

        by_id = {c["library_id"]: c for c in library.chunks}
        selections, seen, prior_lines = [], set(), []
        for el, quota in zip(elements, quotas):
            chunks = [by_id[i] for i in el["chunk_ids"] if i in by_id]
            if not chunks:
                continue
            ranked = sorted(chunks, key=self._prefilter_heuristic, reverse=True)
            shortlist = sorted(ranked[:max(15, 3 * quota)],
                               key=lambda c: c["library_id"])
            beat = {"name": el["name"],
                    "narrative": f"the day's {el['type']}: {el['name']}",
                    "target_shots": quota}
            picks = self._select_for_beat(beat, shortlist, story,
                                          "\n".join(prior_lines))
            for pick in picks[:quota + 1]:
                c = pick["chunk"]
                if c["library_id"] in seen:
                    continue
                seen.add(c["library_id"])
                pick["beat"] = el["name"]
                selections.append(pick)
                print(f"  [{pick['role']:>10}] {c['source']} "
                      f"{c['start']:.0f}-{c['end']:.0f}s: {pick['reason']}")
            el_picks = [p for p in selections if p.get("beat") == el["name"]]
            if el_picks:
                moments = "; ".join(
                    p["chunk"]["one_line_summary"][:60] for p in el_picks[:3])
                prior_lines.append(f"{el['name']}: {moments}")

        if not selections:
            return []
        if max_total is not None:
            selections = selections[:max_total]
        # top-up: element quotas often under-deliver (LLM returns fewer picks
        # than asked); when a target length was given, fill the shortfall with
        # the strongest unused chunks, spread across the day
        if target_len and max_total is None and len(selections) < total:
            spare = sorted((c for c in library.chunks
                            if c["library_id"] not in seen),
                           key=self._prefilter_heuristic, reverse=True)
            fills = sorted(spare[:total - len(selections)],
                           key=lambda c: order[c["library_id"]])
            for c in fills:
                seen.add(c["library_id"])
                selections.append({"chunk": c, "role": "buildup",
                                   "reason": "length top-up (heuristic pick)",
                                   "beat": "top-up"})
            if fills:
                print(f"  topped up with {len(fills)} heuristic pick(s) to "
                      f"approach the target length.")
        # anchor the day: guarantee the chronological open and close
        first = min(library.chunks, key=lambda c: order[c["library_id"]])
        last = max(library.chunks, key=lambda c: order[c["library_id"]])
        for anchor, role in ((first, "setting"), (last, "outro")):
            if anchor["library_id"] not in seen and (max_total is None):
                seen.add(anchor["library_id"])
                selections.append({"chunk": anchor, "role": role,
                                   "reason": "anchors the day", "beat": role})
        selections.sort(key=lambda s: (s["chunk"]["source"], s["chunk"]["start"]))
        est = len(selections) * avg_cut
        print(f"\nStoryComposer: {len(selections)} cuts covering "
              f"{len(elements)} elements of the day "
              f"(~{est / 60:.1f} min estimated"
              + (f" of ~{target_len / 60:.1f} min target" if target_len else "")
              + ").")
        return selections

    def compose(self, library, story, target_cuts):
        print(f"\nStoryComposer: outlining the day from {len(library.chunks)} chunks...")
        outline = self._compose_outline(library, story, target_cuts)
        if outline:
            title = outline["story_title"]
            print(f"Story outline: \"{title}\" - {len(outline['beats'])} beats")
            for beat in outline["beats"]:
                print(f"  [{beat['start_id']:>3}-{beat['end_id']:>3}] {beat['name']} "
                      f"(~{beat['target_shots']} shots): {beat['narrative']}")

            selections = []
            seen = set()
            prior_lines = []
            for beat in outline["beats"]:
                beat_chunks = [c for c in library.chunks
                               if beat["start_id"] <= c["library_id"] <= beat["end_id"]]
                if not beat_chunks:
                    continue
                print(f"\nSelecting shots for beat \"{beat['name']}\"...")
                picks = self._select_for_beat(beat, beat_chunks, story,
                                              "\n".join(prior_lines))
                for pick in picks:
                    c = pick["chunk"]
                    if c["library_id"] in seen:
                        continue
                    seen.add(c["library_id"])
                    pick["beat"] = beat["name"]
                    selections.append(pick)
                    print(f"  [{pick['role']:>10}] {c['source']} "
                          f"{c['start']:.0f}-{c['end']:.0f}s: {pick['reason']}")
                beat_picks = [p for p in selections if p.get("beat") == beat["name"]]
                if beat_picks:
                    moments = "; ".join(
                        p["chunk"]["one_line_summary"][:60] for p in beat_picks[:4])
                    prior_lines.append(f"{beat['name']}: {moments}")

            if selections:
                selections.sort(key=lambda s: (s["chunk"]["source"], s["chunk"]["start"]))
                cap = int(target_cuts * 1.5)
                if len(selections) > cap:
                    selections = selections[:cap]
                print(f"\nStoryComposer: \"{title}\" - {len(selections)} shots across "
                      f"{len(outline['beats'])} beats.")
                return selections
            print("StoryComposer: outline produced no usable shots.")
        else:
            print("StoryComposer: outline unusable - trying single-call composition.")
        return self._compose_single_call(library, story, target_cuts)

    def _compose_single_call(self, library, story, target_cuts):
        digest = self._build_outline_digest(library.chunks)
        user = (
            "Below is the shot library for one day of footage, in "
            "chronological order, followed by the editing brief. Compose a coherent "
            f"vlog story using roughly {target_cuts} shots.\n"
            "Rules:\n"
            "- Tell the full story of the day in order: open with establishing/"
            "setting shots, build up to the key moments, include transitions "
            "between locations, and end with a closing shot.\n"
            "- Balance context and excitement: quieter setting shots are as "
            "important as high-interest peaks. Do not just pick the highest "
            "interest scores.\n"
            "- Avoid redundant near-duplicate moments (same scene, same action).\n"
            "- Speech matters: prioritize moments with meaningful conversation, "
            "jokes, reactions, and quotes that carry the story; a quiet shot with "
            "a great line can outrank a flashy shot with nothing said.\n"
            "- Only use id values that appear in the library.\n"
            "Output a JSON object with exactly these keys:\n"
            '"story_title": short title,\n'
            '"selections": list of objects, each with "chunk_id" (the id number), '
            '"role" (one of "setting", "buildup", "peak", "transition", "outro"), '
            'and "reason" (5 words maximum).\n'
            "Keep every reason under 5 words so the full list fits in your reply. "
            "Output ONLY the JSON object, no other text.\n\n"
            f"EDITING BRIEF:\n{json.dumps(story)}\n\n"
            f"SHOT LIBRARY:\n{digest}"
        )
        print(f"\nStoryComposer: composing story from {len(library.chunks)} chunks...")
        try:
            raw = _llm_text(self.SYSTEM, user, COMPOSER_MAX_TOKENS)
        except Exception as e:
            print(f"StoryComposer call failed: {e}")
            raw = ""
        parsed = parse_json_response(raw) if raw else None
        if parsed is None and raw:
            parsed = salvage_selections(raw)
            if parsed:
                print(f"StoryComposer reply was truncated mid-JSON; salvaged "
                      f"{len(parsed['selections'])} complete selections from it.")

        selections = self._validate(parsed, library, target_cuts)
        if selections:
            title = parsed.get("story_title", "") if isinstance(parsed, dict) else ""
            print(f"StoryComposer: \"{title}\" - {len(selections)} shots selected.")
            for sel in selections:
                c = sel["chunk"]
                print(f"  [{sel['role']:>10}] {c['source']} {c['start']:.0f}-{c['end']:.0f}s: {sel['reason']}")
        return selections

    def _validate(self, parsed, library, target_cuts):
        """Deterministic guardrails: real ids only, deduped, chronological, capped."""
        if not isinstance(parsed, dict) or not isinstance(parsed.get("selections"), list):
            return []
        seen = set()
        selections = []
        for item in parsed["selections"]:
            if not isinstance(item, dict):
                continue
            try:
                library_id = int(item.get("chunk_id"))
            except (TypeError, ValueError):
                continue
            chunk = library.by_id(library_id)
            if chunk is None or library_id in seen:
                continue
            seen.add(library_id)
            role = item.get("role") if item.get("role") in self.ROLES else "buildup"
            selections.append({
                "chunk": chunk,
                "role": role,
                "reason": str(item.get("reason", "")),
            })
        selections.sort(key=lambda s: (s["chunk"]["source"], s["chunk"]["start"]))
        cap = int(target_cuts * 1.5)
        if len(selections) > cap:
            selections = selections[:cap]
        return selections


# ----------------------------------------------------------------------------
# FallbackSelector - deterministic scoring, used only if the composer fails
# ----------------------------------------------------------------------------
class FallbackSelector:
    def __init__(self, weights=FALLBACK_SCORE_WEIGHTS):
        self.w_theme, self.w_emotion, self.w_action, self.w_novelty = weights

    @staticmethod
    def _word_set(chunk):
        words = set(chunk["visual_tags"])
        for action in chunk["actions"]:
            words.update(action.split())
        words.update(chunk["scene"].split())
        words.update(chunk.get("speech", {}).get("transcript", "").lower().split())
        return words

    def _theme_match(self, chunk, story):
        if not story["themes"]:
            return 0.0
        text = " ".join(self._word_set(chunk) | {chunk["one_line_summary"].lower()})
        hits = sum(1 for theme in story["themes"] if any(t in text for t in theme.split()))
        return hits / len(story["themes"])

    def _emotion_score(self, chunk, story):
        if not chunk["emotions"]:
            return 0.0
        if any(e in story["emotional_targets"] for e in chunk["emotions"]):
            return 1.0
        return min(1.0, len(chunk["emotions"]) / 3.0)

    def _action_score(self, chunk):
        score = min(1.0, len(chunk["actions"]) / 3.0)
        if chunk["transition_flag"]:
            score += 0.3
        return min(1.0, score)

    def _avoid_penalty(self, chunk, story):
        if not story["avoid"]:
            return 0.0
        text = " ".join(self._word_set(chunk) | {chunk["one_line_summary"].lower()})
        return 0.5 if any(a and a in text for a in story["avoid"]) else 0.0

    @staticmethod
    def _jaccard(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def select(self, library, story, target_cuts, threshold=FALLBACK_SCORE_THRESHOLD):
        candidates = []
        for chunk in library.chunks:
            prelim = (
                self.w_theme * self._theme_match(chunk, story)
                + self.w_emotion * self._emotion_score(chunk, story)
                + self.w_action * self._action_score(chunk)
                + chunk["interest_score"]
                - self._avoid_penalty(chunk, story)
            )
            if chunk["audio"]["loudness_peak_db"] > LOUDNESS_BONUS_DB:
                prelim += 0.1
            if chunk.get("speech", {}).get("has_speech"):
                prelim += 0.1
            candidates.append((prelim, chunk))

        candidates.sort(key=lambda item: item[0], reverse=True)

        selections = []
        selected_tags = []
        for prelim, chunk in candidates:
            if len(selections) >= target_cuts:
                break
            tags = set(chunk["visual_tags"])
            novelty_penalty = max(
                (self._jaccard(tags, prev) for prev in selected_tags), default=0.0
            )
            if prelim - self.w_novelty * novelty_penalty < threshold:
                continue
            selections.append({"chunk": chunk, "role": "buildup", "reason": "fallback score"})
            selected_tags.append(tags)

        selections.sort(key=lambda s: (s["chunk"]["source"], s["chunk"]["start"]))
        return selections


# ----------------------------------------------------------------------------
# FinalEditor - margins, merging, hi-res render, EDL export
# ----------------------------------------------------------------------------
class FinalEditor:
    def __init__(self, workdir):
        self.workdir = workdir

    @staticmethod
    def build_cuts(selections, margin, durations, lrf_by_source):
        """Selections -> per-source margined intervals, chronological."""
        cuts = []
        for sel in selections:
            chunk = sel["chunk"]
            duration = durations.get(chunk["source"], chunk["end"])
            cuts.append({
                "source_file": chunk["hires_path"],
                "lrf_file": lrf_by_source.get(chunk["source"], ""),
                "source": chunk["source"],
                "start": max(0.0, chunk["start"] - margin),
                "end": min(duration, chunk["end"] + margin),
                "role": sel["role"],
                "reason": sel["reason"],
                "beat": sel.get("beat", ""),
                "tags": list(chunk["visual_tags"]),
                "summary": chunk["one_line_summary"],
                "speech": chunk.get("speech", {}).get("transcript", ""),
            })
        return cuts

    @staticmethod
    def merge_overlapping(cuts):
        """Merge overlapping/adjacent intervals per source file, keep order."""
        if not cuts:
            return []
        cuts = sorted(cuts, key=lambda c: (c["source"], c["start"]))
        merged = [cuts[0]]
        for cur in cuts[1:]:
            last = merged[-1]
            if cur["source"] == last["source"] and cur["start"] <= last["end"]:
                last["end"] = max(last["end"], cur["end"])
                last["tags"] = sorted(set(last["tags"]) | set(cur["tags"]))
                if cur["role"] == "peak":
                    last["role"] = "peak"
                if cur["summary"] and cur["summary"] not in last["summary"]:
                    last["summary"] = f"{last['summary']} / {cur['summary']}".strip(" /")
                if cur["speech"] and cur["speech"] not in last["speech"]:
                    last["speech"] = f"{last['speech']} {cur['speech']}".strip()
            else:
                merged.append(cur)
        return merged

    def render(self, edit_plan, output_path, source_key="source_file",
               preset=None, bitrate=FINAL_BITRATE):
        """Pure-ffmpeg render: encode each cut as its own segment (NVENC on the
        GPU when available), then concat losslessly. Replaces moviepy, which
        piped every frame through Python and encoded 4K on the CPU (hours for
        minutes of video)."""
        ffmpeg, _ = get_media_tools()
        if nvenc_available():
            # p1 = fastest NVENC preset (previews), p5 = quality (finals)
            video_args = ["-c:v", "h264_nvenc", "-preset", "p1" if preset else "p5",
                          "-rc", "vbr", "-b:v", bitrate]
        else:
            video_args = ["-c:v", "libx264", "-preset", preset or "medium",
                          "-b:v", bitrate]
        # Uniform audio so the concat demuxer accepts every segment.
        audio_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

        t0 = time.time()
        durations = {}
        segments = []
        for i, cut in enumerate(edit_plan):
            src = cut.get(source_key)
            if not src:
                continue
            if src not in durations:
                durations[src] = VideoPreprocessor.probe_duration(src)
            duration = durations[src]
            if duration and cut["start"] >= duration:
                continue
            end = min(cut["end"], duration) if duration else cut["end"]
            seg = self.workdir.file(f"seg_{i:04d}.mp4")
            print(f"Encoding cut {i + 1}/{len(edit_plan)}: "
                  f"{os.path.basename(src)} {cut['start']:.1f}-{end:.1f}s")
            result = subprocess.run(
                [ffmpeg, "-y", "-v", "error",
                 "-ss", f"{cut['start']:.3f}", "-t", f"{end - cut['start']:.3f}",
                 "-i", src, *video_args, *audio_args,
                 "-avoid_negative_ts", "make_zero", seg],
                capture_output=True, text=True, check=False,
            )
            if result.returncode != 0 or not os.path.exists(seg):
                print(f"  Segment failed, skipping: {result.stderr.strip()}")
                continue
            segments.append(seg)

        if not segments:
            print("\nNo clips survived selection.")
            return False

        print(f"\nStitching {len(segments)} cuts into {output_path}...")
        list_path = self.workdir.file("concat.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")
        result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", "-movflags", "+faststart", output_path],
            capture_output=True, text=True, check=False,
        )
        for seg in segments:
            self.workdir.remove(seg)
        self.workdir.remove(list_path)
        if result.returncode != 0:
            print(f"Concat failed: {result.stderr.strip()}")
            return False
        print(f"Render finished in {time.time() - t0:.0f}s.")
        return True

    @staticmethod
    def write_edl(edit_plan, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(edit_plan, f, indent=2, default=_json_default)
        print(f"EDL written to {path}")


def load_edl(edl_path):
    """Read + minimally validate the EDL from disk. Tolerates hand edits:
    deleted cuts, nudged times, reordered entries (file order = render order)."""
    if not os.path.exists(edl_path):
        print(f"Error: no EDL found at {edl_path} - run a composition pass first.")
        return None
    with open(edl_path, "r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: EDL at {edl_path} is not valid JSON ({e}).")
            return None
    if not isinstance(raw, list):
        print(f"Error: EDL at {edl_path} should be a JSON list of cuts.")
        return None
    edit_plan = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            item["start"] = float(item["start"])
            item["end"] = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if item["end"] <= item["start"]:
            continue
        edit_plan.append(item)
    if len(edit_plan) < len(raw):
        print(f"EDL: skipped {len(raw) - len(edit_plan)} invalid entries.")
    if not edit_plan:
        print(f"Error: EDL at {edl_path} contains no usable cuts.")
        return None
    return edit_plan


def render_from_edl(edl_path, output_path, workdir, **render_kwargs):
    """Render straight from the EDL on disk - the source of truth, so any manual
    edits to the JSON are honored. Needs no model, library, or composition."""
    edit_plan = load_edl(edl_path)
    if edit_plan is None:
        return False
    return FinalEditor(workdir).render(edit_plan, output_path, **render_kwargs)


# ----------------------------------------------------------------------------
# Pipeline orchestration
# ----------------------------------------------------------------------------
def analyze_clip(pair, preprocessor, analyzer, transcriber, library, workdir, max_chunks):
    """Pass 1 for one LRF proxy: sequential vision + speech analysis into the library."""
    lrf_path = pair["lrf"]
    source_name = os.path.basename(pair["hires"])
    print(f"\n=== Analyzing proxy: {os.path.basename(lrf_path)} ===")

    duration = preprocessor.probe_duration(lrf_path)
    if duration <= 0:
        print("Warning: clip duration is zero or invalid.")
        return

    chunks = preprocessor.extract_rolling_chunks(duration, max_chunks=max_chunks)
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_id'] + 1}/{len(chunks)} "
              f"({chunk['start']:.1f}-{chunk['end']:.1f}s)")
        t0 = time.time()
        try:
            mp4_path, wav_path = preprocessor.extract_chunk_media(
                lrf_path, chunk["chunk_id"], chunk["start"], chunk["end"]
            )
        except RuntimeError as e:
            print(f"  Skipping chunk: {e}")
            continue
        try:
            meta = analyzer.analyze_chunk(mp4_path)
            meta["audio"] = preprocessor.measure_audio(wav_path)
            meta["speech"] = transcriber.transcribe(wav_path)
        finally:
            workdir.remove(mp4_path)
            if wav_path:
                workdir.remove(wav_path)
        if meta["speech"]["has_speech"]:
            meta["audio"]["speech_likely"] = True
        meta.update(chunk)
        meta["source"] = source_name
        meta["hires_path"] = pair["hires"]
        meta["clip_duration"] = duration
        library.add(meta)
        speech_preview = meta["speech"]["transcript"][:60]
        print(f"  -> {meta['one_line_summary']} "
              f"(interest {meta['interest_score']:.2f}, {time.time() - t0:.1f}s)"
              + (f'\n     speech: "{speech_preview}"' if speech_preview else ""))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_knowledge_graph(store, kg_path, library, pairs):
    """Extract entities for every clip whose KG entry is missing, stale, or
    was committed without usable LLM output (enriched=False), from the
    already-cached chunks (no video re-analysis). Saves after each clip so an
    interrupted run resumes."""
    by_clip = {}
    for chunk in library.chunks:
        by_clip.setdefault(chunk["source"], []).append(chunk)
    extractor = None
    for pair in pairs:
        source = os.path.basename(pair["hires"])
        clip_chunks = by_clip.get(source)
        if not clip_chunks or (store.is_current(pair) and store.is_enriched(pair)):
            continue
        if extractor is None:
            extractor = GraphExtractor()
        entities, enriched = extractor.extract_clip(clip_chunks, store.vocabulary())
        print(f"Knowledge graph: {source} -> {len(entities)} entities"
              + ("" if enriched else " (LLM naming failed - will retry next sync)"))
        store.commit_clip(pair, entities, enriched=enriched)
        store.save(kg_path)


def run_pipeline(args):
    output_stem = os.path.splitext(args.output)[0]
    edl_path = output_stem + "_edl.json"
    preview_path = output_stem + PREVIEW_SUFFIX

    if args.finalize:
        workdir = WorkDir()
        if render_from_edl(edl_path, args.output, workdir):
            print("\nSuccess! Final hi-res vlog rendered from the EDL.")
        return

    pairs = get_video_pairs(args.footage_dir)
    if not pairs:
        print(f"Error: Make sure your DJI clips are dropped into the '{args.footage_dir}' folder.")
        return
    print(f"Found {len(pairs)} matching video clip pairs.")

    workdir = WorkDir()
    library_path = output_stem + "_library.json"
    kg_path = output_stem + "_kg.json"

    library = None
    if not args.reanalyze:
        library = ChunkLibrary.load_if_valid(library_path, pairs)

    load_model()

    if library is None:
        library, status = ChunkLibrary.load(
            library_path if not args.reanalyze else "", pairs)
        preprocessor = VideoPreprocessor(workdir)
        analyzer = ChunkAnalyzer()
        transcriber = SpeechTranscriber()
        for pair in pairs:
            source = os.path.basename(pair["hires"])
            if status[source] == "analyzed":
                print(f"\n=== Skipping {source}: already in the library ===")
                continue
            if status[source] == "changed":
                print(f"\n=== {source} changed on disk - re-analyzing ===")
            analyze_clip(pair, preprocessor, analyzer, transcriber, library, workdir, args.max_chunks)
            library.commit_clip(pair)
            # Incremental checkpoint: an interrupted run resumes from here.
            library.save(library_path)
        transcriber.unload()
        if not library.chunks:
            print("\nNo analyzable chunks found in the footage.")
            return

    kg_store, _ = KnowledgeGraphStore.load(kg_path, pairs)
    sync_knowledge_graph(kg_store, kg_path, library, pairs)
    graph = KnowledgeGraph.build(kg_store, library)

    durations = {c["source"]: c["clip_duration"] for c in library.chunks}
    lrf_by_source = {os.path.basename(p["hires"]): p["lrf"] for p in pairs}
    editor = FinalEditor(workdir)
    preview_kwargs = {"source_key": "lrf_file", "preset": "ultrafast",
                      "bitrate": PREVIEW_BITRATE}

    def compose_and_preview(story, target_len=None, target_cuts=None):
        composer = StoryComposer()
        selections = []
        if target_cuts is None and graph.entities:
            selections = composer.compose_with_graph(
                library, story, graph, target_len=target_len, margin=args.margin)
        if not selections:
            cuts = target_cuts or args.target_cuts
            if target_cuts is None and graph.entities:
                print("Graph composition unusable - trying the legacy composer.")
            selections = composer.compose(library, story, cuts)
            if not selections:
                print("StoryComposer output unusable - falling back to deterministic scoring.")
                selections = FallbackSelector().select(library, story, cuts)
        if not selections:
            print("\nNo clips chosen. Try a different brief or --reanalyze.")
            return False
        cuts = editor.build_cuts(selections, args.margin, durations, lrf_by_source)
        edit_plan = editor.merge_overlapping(cuts)
        editor.write_edl(edit_plan, edl_path)
        if editor.render(edit_plan, preview_path, **preview_kwargs):
            print(f"\nFast LRF preview ready: {preview_path} ({len(edit_plan)} cuts)")
            return True
        return False

    story = StoryAgent().parse_user_brief(args.brief)
    target_len = args.target_len * 60 if args.target_len else None
    compose_and_preview(story, target_len=target_len)

    if not sys.stdin.isatty():
        print(f"\nPreview + EDL written. Run with --finalize to render the "
              f"hi-res version from {edl_path}.")
        return

    target_cuts = None
    print(
        "\nIterate on the preview:\n"
        "  final      render the hi-res vlog from the EDL on disk and exit\n"
        "  preview    re-render the LRF preview from the EDL on disk\n"
        "  length N   target roughly N minutes and re-compose (0 = auto)\n"
        "  cuts N     legacy: fixed cut count instead of graph coverage\n"
        "  quit / q   exit without the hi-res render\n"
        "  <text>     anything else is a new brief - re-compose + preview\n"
        f"(you can also hand-edit {edl_path} between commands)"
    )
    while True:
        try:
            cmd = input("\nauto-edit> ").strip()
        except (EOFError, KeyboardInterrupt):
            cmd = "quit"
        if not cmd:
            continue
        lowered = cmd.lower()
        if lowered in ("quit", "q", "exit"):
            print(f"Exiting. Run with --finalize later to render hi-res from {edl_path}.")
            return
        if lowered == "final":
            if render_from_edl(edl_path, args.output, workdir):
                print("\nSuccess! Final hi-res vlog rendered from the EDL.")
            return
        if lowered == "preview":
            render_from_edl(edl_path, preview_path, workdir, **preview_kwargs)
            continue
        if lowered.startswith("length"):
            try:
                minutes = float(cmd.split()[1])
            except (IndexError, ValueError):
                print("Usage: length N (approximate minutes, e.g. length 5; 0 = auto)")
                continue
            target_len = minutes * 60 if minutes > 0 else None
            target_cuts = None
            compose_and_preview(story, target_len=target_len)
            continue
        if lowered.startswith("cuts"):
            try:
                target_cuts = int(cmd.split()[1])
            except (IndexError, ValueError):
                print("Usage: cuts N (e.g. cuts 12)")
                continue
            compose_and_preview(story, target_cuts=target_cuts)
            continue
        story = StoryAgent().parse_user_brief(cmd)
        compose_and_preview(story, target_len=target_len, target_cuts=target_cuts)


def parse_args():
    parser = argparse.ArgumentParser(description="Library-based multi-agent auto video editor.")
    parser.add_argument("--brief", default=DEFAULT_BRIEF,
                        help="Narrative brief describing what the final vlog should capture.")
    parser.add_argument("--footage-dir", default=FOOTAGE_DIR,
                        help="Folder containing DJI .LRF proxy + .MP4 hi-res pairs.")
    parser.add_argument("--output", default="qwen35_native_vlog.mp4",
                        help="Output video path (EDL and library JSON are written alongside it).")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Debug cap on chunks per clip (default: process everything).")
    parser.add_argument("--margin", type=float, default=3.0,
                        help="Safety margin seconds added before/after each cut (clamped 2-5).")
    parser.add_argument("--target-len", type=float, default=None,
                        help="Approximate target video length in minutes "
                             "(default: auto from knowledge-graph coverage).")
    parser.add_argument("--target-cuts", type=int, default=15,
                        help="Legacy fixed cut count, used only when the "
                             "knowledge graph is unavailable.")
    parser.add_argument("--reanalyze", action="store_true",
                        help="Ignore the cached chunk library and redo the vision pass.")
    parser.add_argument("--finalize", action="store_true",
                        help="Skip analysis/composition and render the hi-res vlog "
                             "straight from the EDL JSON next to --output.")
    args = parser.parse_args()
    args.margin = max(2.0, min(5.0, args.margin))
    return args


if __name__ == "__main__":
    run_pipeline(parse_args())
