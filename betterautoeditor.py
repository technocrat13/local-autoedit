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
from moviepy import VideoFileClip, concatenate_videoclips
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

LIBRARY_SCHEMA = 2  # v2 adds per-chunk "speech"; older caches are re-analyzed

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


# ----------------------------------------------------------------------------
# ChunkLibrary - persistent metadata library covering every chunk of every clip
# ----------------------------------------------------------------------------
class ChunkLibrary:
    def __init__(self):
        self.chunks = []

    def add(self, meta):
        meta["library_id"] = len(self.chunks)
        self.chunks.append(meta)

    def by_id(self, library_id):
        if 0 <= library_id < len(self.chunks):
            return self.chunks[library_id]
        return None

    @staticmethod
    def fingerprint(pairs):
        return [
            {"name": os.path.basename(p["lrf"]), "lrf_size": os.path.getsize(p["lrf"])}
            for p in pairs
        ]

    def save(self, path, pairs):
        payload = {
            "schema": LIBRARY_SCHEMA,
            "fingerprint": self.fingerprint(pairs),
            "analysis_fps": ANALYSIS_FPS,
            "chunk_duration": CHUNK_DURATION,
            "chunks": self.chunks,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        print(f"Chunk library saved to {path} ({len(self.chunks)} chunks)")

    @classmethod
    def load_if_valid(cls, path, pairs):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"Could not read library cache {path}: {e}")
            return None
        if payload.get("schema") != LIBRARY_SCHEMA:
            print("Library cache ignored: older schema (no speech data) - re-analyzing.")
            return None
        if payload.get("analysis_fps") != ANALYSIS_FPS or payload.get("chunk_duration") != CHUNK_DURATION:
            print("Library cache ignored: analysis settings changed.")
            return None
        if payload.get("fingerprint") != cls.fingerprint(pairs):
            print("Library cache ignored: footage changed.")
            return None
        lib = cls()
        for meta in payload.get("chunks", []):
            lib.chunks.append(meta)
        print(f"Library cache hit: {path} ({len(lib.chunks)} chunks) - skipping video analysis.")
        return lib


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
        final_clips = []
        open_sources = {}
        try:
            for cut in edit_plan:
                src = cut.get(source_key)
                if not src:
                    continue
                if src not in open_sources:
                    open_sources[src] = VideoFileClip(src)
                main_clip = open_sources[src]
                if cut["start"] >= main_clip.duration:
                    continue
                end = min(cut["end"], main_clip.duration)
                final_clips.append(main_clip.subclipped(cut["start"], end))

            if not final_clips:
                print("\nNo clips survived selection.")
                return False

            print(f"\nStitching {len(final_clips)} cuts into {output_path}...")
            vlog = concatenate_videoclips(final_clips, method="compose")
            extra = {"preset": preset} if preset else {}
            try:
                vlog.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    bitrate=bitrate,
                    temp_audiofile=self.workdir.file("render_temp_audio.m4a"),
                    remove_temp=True,
                    threads=4,
                    **extra,
                )
            finally:
                vlog.close()
            return True
        finally:
            for clip in open_sources.values():
                clip.close()

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

    library = None
    if not args.reanalyze:
        library = ChunkLibrary.load_if_valid(library_path, pairs)

    load_model()

    if library is None:
        library = ChunkLibrary()
        preprocessor = VideoPreprocessor(workdir)
        analyzer = ChunkAnalyzer()
        transcriber = SpeechTranscriber()
        for pair in pairs:
            analyze_clip(pair, preprocessor, analyzer, transcriber, library, workdir, args.max_chunks)
        transcriber.unload()
        if not library.chunks:
            print("\nNo analyzable chunks found in the footage.")
            return
        library.save(library_path, pairs)

    durations = {c["source"]: c["clip_duration"] for c in library.chunks}
    lrf_by_source = {os.path.basename(p["hires"]): p["lrf"] for p in pairs}
    editor = FinalEditor(workdir)
    preview_kwargs = {"source_key": "lrf_file", "preset": "ultrafast",
                      "bitrate": PREVIEW_BITRATE}

    def compose_and_preview(story, target_cuts):
        selections = StoryComposer().compose(library, story, target_cuts)
        if not selections:
            print("StoryComposer output unusable - falling back to deterministic scoring.")
            selections = FallbackSelector().select(library, story, target_cuts)
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
    compose_and_preview(story, args.target_cuts)

    if not sys.stdin.isatty():
        print(f"\nPreview + EDL written. Run with --finalize to render the "
              f"hi-res version from {edl_path}.")
        return

    target_cuts = args.target_cuts
    print(
        "\nIterate on the preview:\n"
        "  final      render the hi-res vlog from the EDL on disk and exit\n"
        "  preview    re-render the LRF preview from the EDL on disk\n"
        "  cuts N     change target cut count and re-compose\n"
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
        if lowered.startswith("cuts"):
            try:
                target_cuts = int(cmd.split()[1])
            except (IndexError, ValueError):
                print("Usage: cuts N (e.g. cuts 12)")
                continue
            compose_and_preview(story, target_cuts)
            continue
        story = StoryAgent().parse_user_brief(cmd)
        compose_and_preview(story, target_cuts)


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
    parser.add_argument("--target-cuts", type=int, default=15,
                        help="How many story moments the composer should aim for.")
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
