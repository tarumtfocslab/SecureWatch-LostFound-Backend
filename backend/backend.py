# backend/backend.py
# ------------------------------------------------------------
# Lost & Found Backend (FastAPI)
#
# ✅ upload/            = Dashboard + Live source (serves /videos)
# ✅ offline_upload/    = Offline Upload Page source (serves /offline)
#
# ------------------------------------------------------------
from __future__ import annotations
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
from urllib.parse import urlsplit, urlunsplit, quote, unquote
import signal
import cv2
from collections import deque
from queue import Queue, Empty
from threading import Lock
import hashlib


LIVE_VIEW_SESSIONS: Dict[str, Any] = {}
LIVE_VIEW_LOCK = threading.Lock()

cv2.setNumThreads(1)
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi import Query
from pydantic import BaseModel

# Local imports
try:
    from lostfound_backend import lostandfound as lf
    from lostfound_backend.backend.live_hub import LiveHub
except ModuleNotFoundError:
    import lostandfound as lf
    from backend.live_hub import LiveHub
import re, datetime


# ============================================================
# Configuration and Constants
# ============================================================
# Strict console style
def _system(msg: str) -> None:
    print(f"[SYSTEM] {msg}")


# Paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "upload"  # dashboard/live source
OFFLINE_UPLOAD_DIR = BASE_DIR / "offline_upload"  # offline upload page source
GRID_DIR = UPLOAD_DIR / "_grid"
OUTPUTS_LF_DIR = BASE_DIR / "outputs" / "lost_and_found"
PROJECT_ROOT = BASE_DIR.parent
PROJECT_OUTPUTS_LF_DIR = PROJECT_ROOT / "outputs" / "lost_and_found"
PROJECT_OUTPUTS_LF_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = BASE_DIR / "_tmp"  # ✅ all temp artifacts MUST be here
os.environ.setdefault("LF_BACKEND", "1")
# Create directories
for p in (UPLOAD_DIR, OFFLINE_UPLOAD_DIR, GRID_DIR, OUTPUTS_LF_DIR, TMP_DIR):
    p.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OFFLINE_UPLOAD_DIR / "_manifest.json"
SETTINGS_PATH = OUTPUTS_LF_DIR / "_settings.json"

# Video extensions
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Fisheye configuration
FISHEYE_GROUPS = getattr(
    lf,
    "FISHEYE_GROUPS",
    [
        ["middle_row", "front_right_row", "front_left_row", "front_corridor"],
        ["back_right_row", "back_left_row", "back_corridor", "entrance"],
    ],
)
FISHEYE_VIEW_CONFIGS = getattr(lf, "FISHEYE_VIEW_CONFIGS", [])

# Default settings
DEFAULT_SETTINGS: Dict[str, Any] = {
    "notifications_enabled": True,
    "notifications_sound_enabled": False,
    "play_sound": False,
    "camera_enabled": True,
    "data_retention_enabled": True,
    "data_retention_days": 90,
}

BOOLEAN_SETTING_KEYS = {
    "notifications_enabled",
    "notifications_sound_enabled",
    "play_sound",
    "camera_enabled",
    "data_retention_enabled",
}
INT_SETTING_KEYS = {
    "data_retention_days",
}
ALLOWED_SETTING_KEYS = BOOLEAN_SETTING_KEYS | INT_SETTING_KEYS

# Threading locks and caches
_settings_lock = threading.Lock()
_settings_cache: Dict[str, Any] = {}
_manifest_lock = threading.Lock()
_video_type_lock = threading.Lock()
_video_type_cache: Dict[str, str] = {}
_offline_pre_lock = threading.Lock()
_offline_pre_cache: Dict[str, Any] = {}
_offline_jobs_lock = threading.Lock()
_offline_jobs: Dict[str, Dict[str, Any]] = {}
_offline_frame_cache_lock = threading.Lock()
_offline_frame_cache: Dict[str, Dict[str, Any]] = {}  # {stem: { "ts": float, "frame": np.ndarray }}
_live_pre_lock = threading.Lock()
_live_pre_cache: Dict[str, Any] = {}
_grid_lock = threading.Lock()
_grid_threads: Dict[str, threading.Thread] = {}
_grid_requested_once: Set[str] = set()
_analyze_queue_lock = threading.Lock()
_analyze_queued: Set[str] = set()
OFFLINE_LOCKS = defaultdict(threading.Lock)

# Live components
hub = LiveHub()
pipelines_live: Dict[str, Any] = {}
pipelines_settings: Dict[str, Any] = {}
pipelines: Dict[str, Any] = {}
_live_pipe_lock = threading.Lock()
# ✅ MUST exist before _get_live_detector() is ever called
_live_detectors: Dict[str, Any] = {}
_live_detector_lock = threading.Lock()

# ============================================================
# Helper Functions
# ============================================================
LIVE_ROOT = OUTPUTS_LF_DIR / "live"
OFFLINE_ROOT = OUTPUTS_LF_DIR / "offline"
LIVE_ROOT.mkdir(parents=True, exist_ok=True)
OFFLINE_ROOT.mkdir(parents=True, exist_ok=True)


def _outputs_dir(mode: str, id_: str) -> Path:
    mode = (mode or "").lower().strip()
    root = LIVE_ROOT if mode == "live" else OFFLINE_ROOT
    out_dir = root / id_
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _ensure_roi_file(mode: str, id_: str) -> Path:
    out_dir = _outputs_dir(mode, id_)
    roi_path = out_dir / "roi_config.json"
    if not roi_path.exists():
        roi_path.write_text(
            json.dumps({"bounding_polygons": [], "fisheye_polygons": {}}, indent=2),
            encoding="utf-8",
        )
    return roi_path

def _reset_live_run_files(cam_id: str) -> None:
    """
    On every backend restart, we want LIVE detection to start fresh:
    - overwrite event_log
    - overwrite lost_items json/csv
    - optionally clear snapshots
    ROI must NOT be deleted.
    """
    cam_id = _live_id(cam_id)

    # (A) Files written by video_pipeline.py today:
    # outputs/lost_and_found/<cam_id>/*
    out_dir = OUTPUTS_LF_DIR / cam_id
    for name in ("event_log.jsonl", "lost_items.json", "lost_items.csv"):
        try:
            (out_dir / name).unlink(missing_ok=True)
        except Exception:
            pass

    # (B) If you ever moved them into outputs/lost_and_found/live/<cam_id>/*
    live_dir = LIVE_ROOT / cam_id
    for name in ("event_log.jsonl", "lost_items.json", "lost_items.csv"):
        try:
            (live_dir / name).unlink(missing_ok=True)
        except Exception:
            pass


def _human_mb(nbytes: int) -> str:
    try:
        return f"{(nbytes / (1024 * 1024)):.0f} MB"
    except Exception:
        return "0 MB"


def _safe_stem(filename: str) -> str:
    return Path(filename).stem


def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def _wait_file_stable(path: Path, checks: int = 3, sleep_s: float = 0.5) -> None:
    last = -1
    for _ in range(checks):
        cur = path.stat().st_size if path.exists() else 0
        if cur == last and cur > 0:
            return
        last = cur
        time.sleep(sleep_s)


def _blank_bgr(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _force_640x480(bgr: np.ndarray) -> np.ndarray:
    if bgr is None:
        return bgr
    h, w = bgr.shape[:2]
    if (w, h) == (640, 480):
        return bgr
    return cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)


def _jpg_bytes(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise HTTPException(status_code=500, detail="jpg encode failed")
    return buf.tobytes()


def _draw_label_bar(img: np.ndarray, text: str) -> np.ndarray:
    if img is None:
        return img
    out = img.copy()
    cv2.rectangle(out, (6, 6), (320, 36), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_2x2_grid(
        imgs: List[Optional[np.ndarray]],
        cell_h: int = 480,
        cell_w: int = 640
) -> np.ndarray:
    """
    Build a 2x2 BGR grid from up to 4 images.
    Missing images are replaced with blank cells.
    """

    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    tiles: List[np.ndarray] = []

    for i in range(4):
        im = imgs[i] if i < len(imgs) else None

        if im is None:
            tiles.append(blank)
            continue

        # Resize only if size mismatch
        if im.shape[0] != cell_h or im.shape[1] != cell_w:
            im = cv2.resize(im, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        tiles.append(im)

    top = np.hstack((tiles[0], tiles[1]))
    bottom = np.hstack((tiles[2], tiles[3]))

    return np.vstack((top, bottom))


def _is_original_upload_file(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.name.startswith("__upload_tmp__"):
        return False
    if p.name.lower().endswith(".ffmpeg.log"):
        return False
    if p.suffix.lower() not in VIDEO_EXTS:
        return False
    # IMPORTANT: do NOT treat converted file as "uploaded item"
    if p.name.lower().endswith("_h264.mp4"):
        return False
    return True


def _safe_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    try:
        if dst.exists():
            dst.unlink(missing_ok=True)
        src.replace(dst)
    except Exception:
        shutil.copy2(str(src), str(dst))
        src.unlink(missing_ok=True)


def _det_label(det: dict) -> str:
    s = (det.get("label") or det.get("class_name") or det.get("cls") or det.get("name") or "").strip()
    return s.lower()


def _base_id(s: str) -> str:
    s = (s or "").strip()
    return s[:-5] if s.lower().endswith("_h264") else s


def _cam_id_from_h264_file(p: Path) -> str:
    return _base_id(p.stem)  # p.stem may be "..._h264"


# ============================================================
# Video Processing Helpers
# ============================================================
def _video_duration_hms(path: Path) -> str:
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return "00:00:00"
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps <= 0.0:
            return "00:00:00"
        sec = int(frames / fps)
        hh = sec // 3600
        mm = (sec % 3600) // 60
        ss = sec % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    except Exception:
        return "00:00:00"


def _assert_video_readable(path: Path) -> None:
    # 1) ffprobe check
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=nk=1:nw=1",
            str(path),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not (p.stdout or "").strip():
            raise RuntimeError("ffprobe failed")
        return
    except Exception:
        pass

    # 2) OpenCV fallback check
    cap = cv2.VideoCapture(str(path))
    ok, fr = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    if not ok or fr is None:
        raise RuntimeError("OpenCV cannot read frames")


def _read_any_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Try grab a stable frame (not black) by sampling a few positions.
    """
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    candidates = [0]
    if total > 0:
        candidates += [max(0, total // 10), max(0, total // 2), max(0, total - 5)]

    for idx in candidates:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        except Exception:
            pass
        ok, fr = cap.read()
        if ok and fr is not None and fr.size > 0:
            return fr
    return None


def _is_h264_mp4(path: Path) -> bool:
    """
    Checks if the first video stream is h264 using ffprobe.
    If ffprobe missing, returns False (forces conversion).
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return False
        codec = (p.stdout or "").strip().lower()
        return codec == "h264"
    except Exception:
        return False


def _ffprobe_readable(path: Path) -> bool:
    try:
        cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams", str(path)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return p.returncode == 0
    except Exception:
        return False


def _write_ffmpeg_fail_log(log_path: Path, cmd: List[str], stdout: str, stderr: str) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write("CMD:\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("STDOUT:\n")
            f.write(stdout or "")
            f.write("\n\nSTDERR:\n")
            f.write(stderr or "")
    except Exception:
        pass


def _ffmpeg_faststart_copy(src: Path, tmp_out: Path, fail_log: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c", "copy",
        "-movflags", "+faststart",
        str(tmp_out),
    ]
    _system(f"FFMPEG remux faststart: {src.name}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg remux timeout (>900s)")

    if p.returncode != 0:
        tmp_out.unlink(missing_ok=True)
        _write_ffmpeg_fail_log(fail_log, cmd, p.stdout, p.stderr)
        raise RuntimeError(f"ffmpeg remux failed. See {fail_log}")


def _ffmpeg_to_h264(src: Path, tmp_out: Path, fail_log: Path) -> None:
    # fast preset + CRF for speed
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp_out),
    ]
    _system(f"FFMPEG encode h264: {src.name}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg h264 timeout (>900s)")

    if p.returncode != 0:
        tmp_out.unlink(missing_ok=True)
        _write_ffmpeg_fail_log(fail_log, cmd, p.stdout, p.stderr)
        raise RuntimeError(f"ffmpeg h264 failed. See {fail_log}")


def _ffmpeg_avi_to_h264_tmp(avi_path: Path, tmp_mp4_out: Path, fail_log: Path) -> None:
    # ✅ fast start + frequent keyframes to avoid long black buffering
    cmd = [
        "ffmpeg", "-y",
        "-i", str(avi_path),

        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",

        # ✅ keyframe tuning (important for quick display)
        "-g", "25",
        "-keyint_min", "25",
        "-sc_threshold", "0",

        "-movflags", "+faststart",
        str(tmp_mp4_out),
    ]
    _system(f"FFMPEG grid convert (tmp): {avi_path.name}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tmp_mp4_out.unlink(missing_ok=True)
        _write_ffmpeg_fail_log(fail_log, cmd, p.stdout, p.stderr)
        raise RuntimeError(f"ffmpeg grid convert failed. See {fail_log}")


def ensure_h264_promoted(stem: str, original_path: Path, target_dir: Path) -> Path:
    """
    Guarantees: <target_dir>/<stem>_h264.mp4 exists and is playable.

    ✅ No temp video/log in upload/ or offline_upload/.
       We write temp into backend/_tmp/ then rename into final.
    """
    _assert_video_readable(original_path)
    if not _ffprobe_readable(original_path):
        raise RuntimeError(f"Input video is not readable/corrupted (ffprobe failed): {original_path.name}")

    target_dir.mkdir(parents=True, exist_ok=True)
    final_dst = target_dir / f"{stem}_h264.mp4"

    if final_dst.exists() and final_dst.stat().st_size > 256 * 1024:
        return final_dst

    tmp_out = TMP_DIR / f"{stem}_h264__{int(time.time() * 1000)}.mp4"
    tmp_out.unlink(missing_ok=True)

    fail_log = TMP_DIR / f"{stem}_h264_fail.ffmpeg.log"
    fail_log.unlink(missing_ok=True)

    if original_path.suffix.lower() == ".mp4" and _is_h264_mp4(original_path):
        _ffmpeg_faststart_copy(original_path, tmp_out, fail_log)
    else:
        _ffmpeg_to_h264(original_path, tmp_out, fail_log)

    try:
        # ✅ atomic replace: do NOT delete first (prevents "half file" / missing file window)
        os.replace(str(tmp_out), str(final_dst))
    finally:
        tmp_out.unlink(missing_ok=True)

    return final_dst


# ============================================================
# Manifest Management
# ============================================================
def _read_manifest() -> Dict[str, Any]:
    with _manifest_lock:
        if not MANIFEST_PATH.exists():
            return {"videos": {}}
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) or {"videos": {}}
        except Exception:
            return {"videos": {}}


def _write_manifest(m: Dict[str, Any]) -> None:
    with _manifest_lock:
        MANIFEST_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")


def _manifest_upsert(stem: str, patch: Dict[str, Any]) -> None:
    m = _read_manifest()
    vids = m.get("videos") or {}
    rec = dict(vids.get(stem) or {})
    rec.update(patch or {})
    vids[stem] = rec
    m["videos"] = vids
    _write_manifest(m)


def _manifest_all() -> List[Dict[str, Any]]:
    m = _read_manifest()
    vids = m.get("videos") or {}
    out: List[Dict[str, Any]] = []
    for stem, rec in vids.items():
        r = dict(rec or {})
        r["id"] = stem
        out.append(r)
    out.sort(key=lambda x: int(x.get("uploadDate") or 0), reverse=True)
    return out


def reconcile_manifest_with_offline_folder() -> None:
    """
    Ensures:
    - manifest contains ONLY original uploaded files in offline_upload/
    - removes records whose original file no longer exists
    - updates size/uploadDate
    - updates h264_ready and roi_ready from actual files
    """
    # scan originals in folder
    originals: Dict[str, Path] = {}
    for f in OFFLINE_UPLOAD_DIR.iterdir():
        if _is_original_upload_file(f):
            originals[f.stem] = f

    m = _read_manifest()
    vids = m.get("videos") or {}

    # remove stale manifest entries (original missing)
    stale = []
    for stem, rec in list(vids.items()):
        if stem not in originals:
            stale.append(stem)
    for stem in stale:
        vids.pop(stem, None)

    # upsert from folder
    for stem, f in originals.items():
        promoted = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"
        roi_path = _outputs_dir("offline", stem) / "roi_config.json"
        roi_ready = False
        if roi_path.exists():
            try:
                roi_obj = json.loads(roi_path.read_text(encoding="utf-8"))
                roi_ready = (not _roi_is_empty(roi_obj))
            except Exception:
                roi_ready = False

        rec = dict(vids.get(stem) or {})
        rec.update({
            "name": f.name,
            "size": _human_mb(f.stat().st_size),
            "uploadDate": int(f.stat().st_mtime),
            "h264_name": f"{stem}_h264.mp4",
            "h264_ready": bool(promoted.exists()),
            "roi_ready": bool(rec.get("roi_ready", False)) or roi_ready,
        })

        # status logic
        if rec.get("status") in ("failed",):
            pass
        else:
            rec["status"] = "ready" if rec["h264_ready"] else "processing"

        vids[stem] = rec

    m["videos"] = vids
    _write_manifest(m)


# ============================================================
# Settings Management
# ============================================================
def load_lf_settings() -> Dict[str, Any]:
    global _settings_cache
    with _settings_lock:
        if _settings_cache:
            return dict(_settings_cache)

        raw: Dict[str, Any] = {}
        try:
            if SETTINGS_PATH.exists():
                raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            raw = {}

        out = dict(DEFAULT_SETTINGS)
        out.update(raw or {})

        _settings_cache = dict(out)
        return dict(out)


def save_lf_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    global _settings_cache
    out = dict(DEFAULT_SETTINGS)
    out.update(load_lf_settings() or {})
    out.update(data or {})

    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    with _settings_lock:
        _settings_cache = dict(out)
    return out

def _normalize_lf_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    cur = load_lf_settings()
    out = dict(DEFAULT_SETTINGS)
    out.update(cur)

    payload = payload or {}

    for key in BOOLEAN_SETTING_KEYS:
        if key in payload:
            out[key] = bool(payload.get(key))

    if "data_retention_days" in payload:
        try:
            days = int(payload.get("data_retention_days"))
        except Exception:
            days = int(out.get("data_retention_days", 90) or 90)
        out["data_retention_days"] = max(1, min(days, 3650))

    # backward compatibility
    if "play_sound" not in payload and "notifications_sound_enabled" in payload:
        out["play_sound"] = bool(payload.get("notifications_sound_enabled"))

    if "notifications_sound_enabled" not in payload and "play_sound" in payload:
        out["notifications_sound_enabled"] = bool(payload.get("play_sound"))

    return out


def _item_ts_seconds(item: Dict[str, Any]) -> float:
    """
    Return a REAL wall-clock timestamp only.
    Ignore relative video seconds like 100, 451, 917.
    """
    MODERN_TS_MIN = 1_700_000_000  # around year 2023+

    for key in ("updatedAt", "createdAt", "timestamp", "lastSeenTs", "firstSeenTs"):
        try:
            val = item.get(key)
            if val is None:
                continue

            ts = float(val)

            # Ignore relative/video timeline values
            if ts < MODERN_TS_MIN:
                continue

            return ts
        except Exception:
            continue

    return 0.0

def _prune_json_store_shards(root: Path, prefix: str, cutoff_ts: float, per_file: int) -> int:
    data = _lf_load_all_shards(root, prefix)
    if not isinstance(data, dict):
        return 0

    original_count = len(data)
    kept = {}

    for key, value in data.items():
        if not isinstance(value, dict):
            kept[key] = value
            continue

        ts = _item_ts_seconds(value)

        # IMPORTANT:
        # if no valid wall-clock timestamp, keep it
        if ts <= 0:
            kept[key] = value
            continue

        if ts < cutoff_ts:
            continue

        kept[key] = value

    if len(kept) != original_count:
        _lf_rewrite_all_shards(root, prefix, kept, per_file)

    return original_count - len(kept)

LF_EVENT_FILE_NAMES = {
    "lost_items.json",
    "lost_items.csv",
    "event_log.jsonl",
    "progress.json",
    "offline_analyze.log",
}

LF_EVENT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _is_safe_lf_event_file(p: Path) -> bool:
    if not p.is_file():
        return False

    # ONLY allow deletion of log/json/csv files
    if p.name in LF_EVENT_FILE_NAMES:
        return True

    # NEVER delete snapshot/evidence images automatically
    return False


def _prune_old_output_files(root: Path, cutoff_ts: float) -> int:
    if not root.exists():
        return 0

    removed = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        # IMPORTANT:
        # _is_safe_lf_event_file now deletes ONLY json/csv/log files.
        # Snapshot images are intentionally preserved.
        if not _is_safe_lf_event_file(p):
            continue

        try:
            mtime = float(p.stat().st_mtime)
        except Exception:
            continue

        if mtime < cutoff_ts:
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass

    return removed

def run_data_retention_cleanup() -> Dict[str, Any]:
    settings = load_lf_settings()

    enabled = bool(settings.get("data_retention_enabled", True))
    days = int(settings.get("data_retention_days", 90) or 90)
    days = max(1, min(days, 3650))

    if not enabled:
        return {
            "ok": True,
            "data_retention_enabled": False,
            "data_retention_days": days,
            "removed_items": 0,
            "removed_overrides": 0,
            "removed_files": 0,
            "message": "Data retention disabled",
        }

    cutoff_ts = time.time() - (days * 86400)

    removed_items = 0
    removed_overrides = 0
    removed_files = 0

    with _items_store_lock:
        removed_items += _prune_json_store_shards(
            LF_STORE_ITEMS_DIR,
            LF_STORE_ITEMS_PREFIX,
            cutoff_ts,
            LF_ITEMS_PER_FILE,
        )

    with _overrides_lock:
        removed_overrides += _prune_json_store_shards(
            LF_STORE_OVERRIDES_DIR,
            LF_STORE_OVERRIDES_PREFIX,
            cutoff_ts,
            LF_OVERRIDES_PER_FILE,
        )

    removed_files += _prune_old_output_files(OUTPUTS_LF_DIR, cutoff_ts)
    removed_files += _prune_old_output_files(PROJECT_OUTPUTS_LF_DIR, cutoff_ts)

    result = {
        "ok": True,
        "data_retention_enabled": True,
        "data_retention_days": days,
        "cutoff_ts": cutoff_ts,
        "removed_items": removed_items,
        "removed_overrides": removed_overrides,
        "removed_files": removed_files,
    }
    _system(
        f"DATA RETENTION cleanup enabled={enabled} days={days} removed_items={removed_items} "
        f"removed_overrides={removed_overrides} removed_files={removed_files}"
    )
    return result

def data_retention_loop():
    _system("DATA RETENTION worker started")
    while True:
        try:
            run_data_retention_cleanup()
        except Exception as e:
            _system(f"DATA RETENTION worker error: {e}")
        time.sleep(3600)



# ============================================================
# Video Type Detection
# ============================================================
_video_type_cache: Dict[str, Dict[str, Any]] = {}  # stem -> {"type": str, "timestamp": float, "hits": int}
_video_type_cache_lock = threading.Lock()
VIDEO_TYPE_CACHE_TTL = 300  # 5 minutes cache lifetime


def detect_video_type_cached(stem: str, src: Path, force_refresh: bool = False) -> str:
    """
    Detect video type with improved caching to prevent repeated detection.
    Only re-detects if file changed or cache expired.

    ✅ For RTSP sources: returns stored type from _rtsp_sources.json immediately.
       Never tries to stat() or probe an rtsp:// URL as a file.
    """
    # ✅ RTSP short-circuit: if this cam has an RTSP entry, return stored type immediately.
    # This prevents stat() crash and avoids slow stream probing on every request.
    rtsp_entry = _get_rtsp_entry(stem)
    if rtsp_entry is not None:
        _system(f"VideoType RTSP short-circuit stem={stem} type={rtsp_entry['type']}")
        return rtsp_entry["type"]

    with _video_type_cache_lock:
        # Check cache first
        now = time.time()
        cached = _video_type_cache.get(stem)

        if not force_refresh and cached:
            # Check if cache is still valid
            cache_age = now - cached.get("timestamp", 0)
            if cache_age < VIDEO_TYPE_CACHE_TTL:
                # Increment hit counter for debugging
                cached["hits"] = cached.get("hits", 0) + 1
                if cached["hits"] % 100 == 0:  # Log every 100 hits
                    _system(f"VideoType cache HIT stem={stem} hits={cached['hits']} type={cached['type']}")
                return cached["type"]
            else:
                _system(f"VideoType cache EXPIRED stem={stem} age={cache_age:.1f}s")

        # Detect video type (expensive operation — only runs for upload files)
        _system(f"VideoType DETECTING stem={stem} (force={force_refresh})")
        vtype = "normal"
        try:
            if hasattr(lf, "STRICT_CONSOLE"):
                lf.STRICT_CONSOLE = True
            if hasattr(lf, "configure_logging"):
                lf.configure_logging(debug=False, mute_info=True)
            vtype = lf.detect_video_type(str(src))
        except Exception as e:
            _system(f"lf.detect_video_type ERROR stem={stem} err={e}")
            vtype = "normal"

        # Update cache
        _video_type_cache[stem] = {
            "type": vtype,
            "timestamp": now,
            "hits": 0,
            "path": str(src),
            "mtime": src.stat().st_mtime if src.exists() else 0
        }

        _system(f"VideoType CACHED stem={stem} type={vtype}")
        return vtype


def invalidate_video_type_cache(stem: str = None):
    """Invalidate cache for specific stem or all if stem=None"""
    with _video_type_cache_lock:
        if stem:
            _video_type_cache.pop(stem, None)
            _system(f"VideoType cache invalidated for {stem}")
        else:
            _video_type_cache.clear()
            _system("VideoType cache cleared")


# ============================================================
# Upload Management
# ============================================================
def list_upload_videos_h264_only() -> List[Path]:
    out: List[Path] = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and f.suffix.lower() == ".mp4" and f.name.lower().endswith("_h264.mp4"):
            out.append(f)
    out.sort(key=lambda p: p.name.lower())
    return out


def find_upload_by_stem(stem: str) -> Optional[Path]:
    want = _base_id(stem)
    for p in list_upload_videos_h264_only():
        if _cam_id_from_h264_file(p) == want:
            return p
    return None


def cam_name_from_file(file: Path) -> str:
    s = file.stem.replace("_h264", "")
    short = s.split("_")[0]
    tail = s[-8:]
    return f"{short}-{tail}"


# ============================================================
# Fisheye View Management
# ============================================================
def _fisheye_order_names() -> List[str]:
    """
    MUST match frontend ordering:
      [...GroupA, ...GroupB]
    """
    a = list(FISHEYE_GROUPS[0])
    b = list(FISHEYE_GROUPS[1])
    return a + b


def _group_view_indices(group: str) -> List[int]:
    """
    CORRECT mapping for 8 views:
    Group A: views 0,1,2,3 (middle_row, front_right_row, front_left_row, front_corridor)
    Group B: views 4,5,6,7 (back_right_row, back_left_row, back_corridor, entrance)
    """
    g = (group or "A").upper()
    if g == "A":
        return [0, 1, 2, 3]
    else:  # Group B
        return [4, 5, 6, 7]  # ✅ FIXED: Use correct indices 4,5,6,7


def _group_names(group: str) -> List[str]:
    g = (group or "A").upper()
    return list(FISHEYE_GROUPS[0]) if g == "A" else list(FISHEYE_GROUPS[1])


def _get_views_by_names(pre, frame: np.ndarray, names_ordered: List[str]) -> List[Dict[str, Any]]:
    # ✅ always ask for exactly the names we want
    try:
        all_views = pre.get_views(frame, allowed_names=names_ordered)
    except TypeError:
        all_views = pre.get_views(frame)

    by_name = {}
    for v in (all_views or []):
        nm = v.get("name")
        if nm:
            by_name[nm] = v

    out = []
    for nm in names_ordered:
        vv = by_name.get(nm)
        if vv is not None:
            out.append(vv)
    return out


# ============================================================
# Offline Frame Processing
# ============================================================
def _offline_get_cached_frame(stem: str, h264_path: Path, max_age_s: float = 0.5) -> Optional[np.ndarray]:
    """
    Cache 1 latest frame per stem to avoid opening VideoCapture on every request.
    """
    now = time.time()
    with _offline_frame_cache_lock:
        rec = _offline_frame_cache.get(stem)
        if rec:
            ts = float(rec.get("ts") or 0.0)
            fr = rec.get("frame")
            if fr is not None and (now - ts) <= max_age_s:
                return fr

    cap = cv2.VideoCapture(str(h264_path))
    try:
        fr = _read_any_frame(cap)
    finally:
        cap.release()

    if fr is None:
        return None

    with _offline_frame_cache_lock:
        _offline_frame_cache[stem] = {"ts": now, "frame": fr}
    return fr


def _offline_h264_path(stem: str) -> Path:
    p = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"
    if not p.exists():
        raise HTTPException(status_code=404, detail="offline h264 not ready")
    return p


def _get_or_make_offline_fisheye_pre(stem: str, h264_path: Path):
    """
    Reuse lostandfound.py FisheyePreprocessor if available.
    Cache per stem, invalidate if file changes.
    """
    if not hasattr(lf, "FisheyePreprocessor"):
        raise RuntimeError("lostandfound.py missing FisheyePreprocessor")
    if not FISHEYE_VIEW_CONFIGS:
        raise RuntimeError("lostandfound.py missing FISHEYE_VIEW_CONFIGS")

    mtime = float(h264_path.stat().st_mtime)

    with _offline_pre_lock:
        rec = _offline_pre_cache.get(stem)
        if rec and rec.get("path") == str(h264_path) and float(rec.get("mtime") or 0) == mtime:
            return rec["pre"]

        # create new preprocessor
        pre = lf.FisheyePreprocessor(
            view_configs=FISHEYE_VIEW_CONFIGS,
            output_size=(480, 640),
            input_fov_deg=180.0,
            logger=None,
        )
        pre.open(str(h264_path))

        _offline_pre_cache[stem] = {"pre": pre, "lock": threading.Lock(), "path": str(h264_path), "mtime": mtime}
        return pre


def _offline_build_view_image(stem: str, view_idx: int) -> np.ndarray:
    """
    Returns a SINGLE view image.
    - Normal: return an original frame
    - Fisheye: return dewarped 640x480 view (fast using cached frame)
    """
    h264_path = _offline_h264_path(stem)
    vtype = detect_video_type_cached(stem, h264_path)

    # NORMAL
    if vtype != "fisheye":
        cap = cv2.VideoCapture(str(h264_path))
        try:
            fr = _read_any_frame(cap)
        finally:
            cap.release()
        if fr is None:
            raise HTTPException(status_code=404, detail="cannot read offline frame")
        return fr

    # FISHEYE (FAST)
    pre = _get_or_make_offline_fisheye_pre(stem, h264_path)

    # ✅ cached frame (avoid opening cap every request)
    fr = _offline_get_cached_frame(stem, h264_path, max_age_s=0.5)
    if fr is None:
        raise HTTPException(status_code=404, detail="cannot read offline fisheye frame")

    wanted_names = _fisheye_order_names()
    vi = int(view_idx)
    if vi < 0:
        vi = 0
    if vi >= len(wanted_names):
        vi = 0
    name = wanted_names[vi]

    rec = _offline_pre_cache.get(stem) or {}
    lock = rec.get("lock")

    # ✅ IMPORTANT FIX: ask ONLY for this view (prevents duplicates)
    if lock:
        with lock:
            try:
                views = pre.get_views(fr, allowed_names=[name])
            except TypeError:
                views = pre.get_views(fr)
    else:
        try:
            views = pre.get_views(fr, allowed_names=[name])
        except TypeError:
            views = pre.get_views(fr)

    chosen = None
    for v in (views or []):
        if (v.get("name") or "") == name:
            chosen = v
            break

    # ✅ DO NOT fallback to another view (that causes copies)
    if chosen is None:
        raise HTTPException(status_code=404, detail=f"fisheye view missing: {name}")

    img = chosen.get("image")
    if img is None:
        raise HTTPException(status_code=404, detail=f"fisheye view image missing: {name}")

    return img


# ============================================================
# Live Frame Processing
# ============================================================
def _get_or_make_live_fisheye_pre(cam_id: str, src_path: Path):
    if not hasattr(lf, "FisheyePreprocessor"):
        raise RuntimeError("lostandfound.py missing FisheyePreprocessor")
    if not FISHEYE_VIEW_CONFIGS:
        raise RuntimeError("lostandfound.py missing FISHEYE_VIEW_CONFIGS")

    mtime = float(src_path.stat().st_mtime)

    with _live_pre_lock:
        rec = _live_pre_cache.get(cam_id)
        if rec and rec.get("path") == str(src_path) and float(rec.get("mtime") or 0) == mtime:
            return rec["pre"]

        pre = lf.FisheyePreprocessor(
            view_configs=FISHEYE_VIEW_CONFIGS,
            output_size=(480, 640),
            input_fov_deg=180.0,
            logger=None,
        )
        pre.open(str(src_path))
        _live_pre_cache[cam_id] = {"pre": pre, "path": str(src_path), "mtime": mtime}
        return pre


# ============================================================
# ROI Geometry Helpers
# ============================================================
def _polygons_equal(poly1: List[Dict[str, float]], poly2: List[Dict[str, float]], tolerance: float = 2.0) -> bool:
    """Check if two polygons are equal within tolerance"""
    if len(poly1) != len(poly2):
        return False

    # Sort points to handle different orders
    def normalize_poly(poly):
        # Find the point with smallest x, then smallest y as start
        points = [(p["x"], p["y"]) for p in poly]
        # Find the lexicographically smallest point
        min_point = min(points)
        min_idx = points.index(min_point)

        # Reorder starting from that point
        normalized = points[min_idx:] + points[:min_idx]
        return normalized

    norm1 = normalize_poly(poly1)
    norm2 = normalize_poly(poly2)

    # Compare all points
    for (x1, y1), (x2, y2) in zip(norm1, norm2):
        if abs(x1 - x2) > tolerance or abs(y1 - y2) > tolerance:
            return False

    return True


def _poly_area(pts: List[Dict[str, float]]) -> float:
    # Shoelace formula
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]["x"], pts[i]["y"]
        x2, y2 = pts[(i + 1) % len(pts)]["x"], pts[(i + 1) % len(pts)]["y"]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _dedupe_close_points(pts, eps=1.0):
    if not pts:
        return pts
    out = [pts[0]]
    eps2 = eps * eps
    for p in pts[1:]:
        dx = float(p["x"]) - float(out[-1]["x"])
        dy = float(p["y"]) - float(out[-1]["y"])
        if (dx * dx + dy * dy) > eps2:
            out.append(p)
    # drop last if same as first (distance)
    if len(out) >= 2:
        dx = float(out[0]["x"]) - float(out[-1]["x"])
        dy = float(out[0]["y"]) - float(out[-1]["y"])
        if (dx * dx + dy * dy) <= eps2:
            out.pop()
    return out


def _is_xy_dict(p: Any) -> bool:
    return isinstance(p, dict) and ("x" in p) and ("y" in p)


def _is_xy_pair(p: Any) -> bool:
    return (
            isinstance(p, (list, tuple))
            and len(p) == 2
            and isinstance(p[0], (int, float))
            and isinstance(p[1], (int, float))
    )


ROI_CANVAS_W = 640.0
ROI_CANVAS_H = 480.0


def _poly_to_xy_dicts(poly: Any, w: float, h: float) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    if not isinstance(poly, list) or not poly:
        return out

    def _convert_xy(x: float, y: float) -> Tuple[float, float]:
        # Case 1: normalized (0–1)
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x * float(w), y * float(h)

        # Case 2: already in canvas resolution (640x480 exactly)
        if int(w) == int(ROI_CANVAS_W) and int(h) == int(ROI_CANVAS_H):
            return x, y

        # Case 3: coordinates look like 640x480 canvas but frame is larger
        if x <= ROI_CANVAS_W and y <= ROI_CANVAS_H and (w != ROI_CANVAS_W or h != ROI_CANVAS_H):
            return (x / ROI_CANVAS_W) * float(w), (y / ROI_CANVAS_H) * float(h)

        # Case 4: already correct pixel coords
        return x, y

    # dict points [{x,y}, ...]
    if _is_xy_dict(poly[0]):
        xs, ys = [], []
        for p in poly:
            try:
                xs.append(float(p["x"]))
                ys.append(float(p["y"]))
            except Exception:
                pass
        if not xs:
            return out
        mx, my = max(xs), max(ys)
        for x, y in zip(xs, ys):
            xx, yy = _convert_xy(x, y, mx, my)
            out.append({"x": xx, "y": yy})
        return out

    # pair points [[x,y], ...]
    if _is_xy_pair(poly[0]):
        pairs = []
        for p in poly:
            if _is_xy_pair(p):
                pairs.append((float(p[0]), float(p[1])))
        if not pairs:
            return out
        mx = max(x for x, _ in pairs)
        my = max(y for _, y in pairs)
        for x, y in pairs:
            xx, yy = _convert_xy(x, y, mx, my)
            out.append({"x": xx, "y": yy})
        return out

    return out


def _polys_to_xy_dicts(polys: Any, w: float, h: float) -> List[List[Dict[str, float]]]:
    """
    Accepts polygons in ANY of these forms:
      1) [ [ {x,y}, ... ], [ {x,y}, ... ] ]          (list of polygons)
      2) [ {x,y}, {x,y}, ... ]                      (single polygon points)  ✅ NEW
      3) [ [x,y], [x,y], ... ]                      (single polygon as pairs) ✅ NEW
      4) [ [ [x,y], ... ], [ [x,y], ... ] ]         (list of polygons as pairs)
    Returns: list of polygons, each polygon is [{x,y}, ...] in pixels.
    """
    if not isinstance(polys, list) or not polys:
        return []

    # ✅ Case: single polygon as point dicts: [{x,y}, ...]
    if _is_xy_dict(polys[0]):
        pts = _poly_to_xy_dicts(polys, w, h)
        return [pts] if len(pts) >= 3 else []

    # ✅ Case: single polygon as pairs: [[x,y], ...]
    if _is_xy_pair(polys[0]):
        pts = _poly_to_xy_dicts(polys, w, h)
        return [pts] if len(pts) >= 3 else []

    # Original: list of polygons
    out: List[List[Dict[str, float]]] = []
    for poly in polys:
        pts = _poly_to_xy_dicts(poly, w, h)
        if len(pts) >= 3:
            out.append(pts)
    return out


def _sanitize_one_fisheye_poly(poly: List[Dict[str, float]], W: int = 640, H: int = 480) -> List[Dict[str, float]]:
    pts = []
    for p in (poly or []):
        if not (isinstance(p, dict) and "x" in p and "y" in p):
            continue
        x = _clamp(p["x"], 0, W)
        y = _clamp(p["y"], 0, H)
        pts.append({"x": x, "y": y})

    pts = _dedupe_close_points(pts, eps=1.0)
    if len(pts) < 3:
        return []

    # reject ultra-small polygons (acts like "empty ROI")
    area = _poly_area(pts)
    if area < 500:  # adjust if needed; 500px² is tiny
        return []

    return pts


def _poly_signature(poly: List[Dict[str, float]], decimals: int = 1) -> str:
    """
    Make a stable signature for a polygon so duplicates can be removed.
    Rounds to 0.1px by default to tolerate tiny float differences.
    """
    pts = []
    for p in (poly or []):
        if not isinstance(p, dict) or "x" not in p or "y" not in p:
            continue
        pts.append([round(float(p["x"]), decimals), round(float(p["y"]), decimals)])
    # keep order (your UI usually draws in order); convert to JSON string
    return json.dumps(pts, separators=(",", ":"))


def _dedupe_fisheye_polygons(fis: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each view_name: remove duplicate polygons.
    Keeps only the first polygon for identical signatures.
    """
    if not isinstance(fis, dict):
        return {}

    out: Dict[str, Any] = {}
    for view_name, poly_list in fis.items():
        if not isinstance(poly_list, list):
            out[view_name] = []
            continue

        seen = set()
        uniq_polys = []
        for poly in poly_list:
            if not isinstance(poly, list):
                continue
            sig = _poly_signature(poly, decimals=1)
            if sig in seen:
                continue
            seen.add(sig)
            uniq_polys.append(poly)

        out[view_name] = uniq_polys

    return out


# ============================================================
# ROI Management
# ============================================================
def _roi_is_empty(roi: Dict[str, Any]) -> bool:
    # bounding_polygons: list of polygons, each polygon is list of points
    polys = (roi or {}).get("bounding_polygons") or []
    if isinstance(polys, list) and len(polys) > 0:
        # ensure at least 1 polygon has >= 3 points
        for poly in polys:
            if isinstance(poly, list) and len(poly) >= 3:
                # area check
                try:
                    pts = [{"x": float(p["x"]), "y": float(p["y"])} for p in poly if
                           isinstance(p, dict) and "x" in p and "y" in p]
                    if _poly_area(pts) >= 500:
                        return False
                except Exception:
                    pass

    # fisheye_polygons: dict(view_name -> list of polygons)
    fis = (roi or {}).get("fisheye_polygons") or {}
    if isinstance(fis, dict):
        for _, poly_list in fis.items():
            if isinstance(poly_list, list) and len(poly_list) > 0:
                # poly_list is list of polygons
                for poly in poly_list:
                    if isinstance(poly, list) and len(poly) >= 3:
                        return False

    return True


def _merge_fisheye_groups_to_flat(fisheye_polys: Any) -> Dict[str, Any]:
    """
    Accept fisheye_polygons in either:
      A) {"A": {...}, "B": {...}} (your UI newer format)
      B) {"middle_row": [...], "front_right_row": [...]} (original)
    Return ONLY the flat/original style:
      {"middle_row": [...], ...}
    """
    if not isinstance(fisheye_polys, dict):
        return {}

    # already flat/original if it contains known view keys
    has_group_keys = ("A" in fisheye_polys) or ("B" in fisheye_polys)
    if not has_group_keys:
        return dict(fisheye_polys)

    flat: Dict[str, Any] = {}
    for g in ("A", "B"):
        part = fisheye_polys.get(g)
        if not isinstance(part, dict):
            continue
        for view_name, poly_list in part.items():
            # merge: if same view name appears twice, append
            if view_name not in flat:
                flat[view_name] = poly_list
            else:
                if isinstance(flat[view_name], list) and isinstance(poly_list, list):
                    flat[view_name].extend(poly_list)
    return flat


def _infer_normal_frame_size_for_offline(stem: str) -> Tuple[int, int]:
    """
    Reads one frame from offline h264 to get (w,h) so normalized ROI can be converted to pixels.
    """
    h264_path = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"
    cap = cv2.VideoCapture(str(h264_path))
    try:
        fr = _read_any_frame(cap)
    finally:
        cap.release()
    if fr is None:
        # fallback
        return (1920, 1080)
    h, w = fr.shape[:2]
    return (int(w), int(h))


def _infer_normal_frame_size_for_live(cam_id: str, src: Path) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(src))
    try:
        fr = _read_any_frame(cap)
    finally:
        cap.release()
    if fr is None:
        return (1920, 1080)
    h, w = fr.shape[:2]
    return (int(w), int(h))


def _normalize_roi_payload_to_original(stem_or_cam: str, payload: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    payload = payload or {}

    # ---------------------------------------------------------
    # Decide video type using unified resolver
    # ---------------------------------------------------------
    vtype = "normal"

    if mode == "offline":
        h264 = OFFLINE_UPLOAD_DIR / f"{stem_or_cam}_h264.mp4"
        if h264.exists():
            try:
                vtype = detect_video_type_cached(stem_or_cam, h264)
            except Exception:
                pass
    else:
        try:
            src_str = resolve_live_source(stem_or_cam)
            if src_str:
                vtype = resolve_settings_video_type(stem_or_cam, str(src_str))
        except Exception:
            pass

    # ---------------------------------------------------------
    # NORMAL
    # ---------------------------------------------------------
    if vtype != "fisheye":
        if mode == "offline":
            w, h = _infer_normal_frame_size_for_offline(stem_or_cam)
        else:
            src = find_upload_by_stem(stem_or_cam)
            w, h = _infer_normal_frame_size_for_live(stem_or_cam, src) if src else (1920, 1080)

        bounding_in = payload.get("bounding_polygons", [])

        if bounding_in and len(bounding_in) > 0:
            if isinstance(bounding_in[0], list) and len(bounding_in[0]) > 0:
                if isinstance(bounding_in[0][0], dict) and "x" in bounding_in[0][0]:
                    bounding_out = bounding_in
                else:
                    bounding_out = _polys_to_xy_dicts(bounding_in, w, h)
            else:
                bounding_out = _polys_to_xy_dicts(bounding_in, w, h)
        else:
            bounding_out = []

        return {
            "bounding_polygons": bounding_out,
            "fisheye_polygons": {},
        }

    # ---------------------------------------------------------
    # FISHEYE
    # ---------------------------------------------------------
    F_W, F_H = 640, 480
    fis_in = payload.get("fisheye_polygons", {})

    if isinstance(fis_in, dict):
        is_correct_format = True
        for view_name, poly_list in fis_in.items():
            if not isinstance(poly_list, list):
                is_correct_format = False
                break
            for poly in poly_list:
                if not isinstance(poly, list):
                    is_correct_format = False
                    break
                for point in poly:
                    if not isinstance(point, dict) or "x" not in point or "y" not in point:
                        is_correct_format = False
                        break
                if not is_correct_format:
                    break
            if not is_correct_format:
                break

        if is_correct_format:
            fis_out = dict(fis_in)
        else:
            flat = _merge_fisheye_groups_to_flat(fis_in)
            fis_out = {}
            for view_name, poly_list in flat.items():
                fis_out[view_name] = _polys_to_xy_dicts(poly_list, F_W, F_H)
    else:
        fis_out = {}

    for name in _fisheye_order_names():
        if name not in fis_out:
            fis_out[name] = []
        elif not isinstance(fis_out[name], list):
            fis_out[name] = []

    return {
        "bounding_polygons": [],
        "fisheye_polygons": fis_out,
    }


# ============================================================
# Grid Video Management
# ============================================================
def _grid_folder(stem: str) -> Path:
    return GRID_DIR / stem


def grid_path_for_group(stem: str, group: str) -> Path:
    g = (group or "A").upper()
    folder = _grid_folder(stem)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{g}_grid_h264.mp4"


def grid_ready_for_groups(stem: str) -> Tuple[bool, bool]:
    a = grid_path_for_group(stem, "A").exists()
    b = grid_path_for_group(stem, "B").exists()
    return a, b


def build_fisheye_group_grid_video(src: Path, out_mp4: Path, group: str) -> None:
    group = (group or "A").upper()
    if group not in ("A", "B"):
        group = "A"

    if not hasattr(lf, "FisheyePreprocessor"):
        raise RuntimeError("lostandfound.py missing FisheyePreprocessor")
    if not FISHEYE_VIEW_CONFIGS:
        raise RuntimeError("lostandfound.py missing FISHEYE_VIEW_CONFIGS")

    allowed_names = FISHEYE_GROUPS[0] if group == "A" else FISHEYE_GROUPS[1]

    pre = lf.FisheyePreprocessor(
        view_configs=FISHEYE_VIEW_CONFIGS,
        output_size=(480, 640),
        input_fov_deg=180.0,
        logger=None,
    )
    pre.open(str(src))

    cap = pre.cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = float(fps) if fps > 1.0 else 25.0

    cell_h, cell_w = pre.out_h, pre.out_w
    grid_h, grid_w = cell_h * 2, cell_w * 2

    # ✅ tmp AVI goes to _tmp (NOT in upload/_grid)
    tmp_avi = TMP_DIR / f"{src.stem}__{group}__grid__{int(time.time() * 1000)}.avi"
    tmp_avi.unlink(missing_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(str(tmp_avi), fourcc, fps, (grid_w, grid_h))
    if not vw.isOpened():
        pre.release()
        raise RuntimeError("VideoWriter failed (XVID).")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            view_dicts = _get_views_by_names(pre, frame, allowed_names)
            imgs: List[np.ndarray] = [v.get("image") for v in view_dicts]
            while len(imgs) < 4:
                imgs.append(None)
            grid = make_2x2_grid(imgs[:4], cell_h, cell_w)
            vw.write(grid)
    finally:
        try:
            vw.release()
        except Exception:
            pass
        pre.release()

    # ✅ tmp MP4 goes to _tmp too, then move to final out_mp4
    tmp_mp4 = TMP_DIR / f"{src.stem}__{group}__grid__{int(time.time() * 1000)}.mp4"
    tmp_mp4.unlink(missing_ok=True)
    fail_log = TMP_DIR / f"{src.stem}__{group}__grid_fail.ffmpeg.log"
    fail_log.unlink(missing_ok=True)

    try:
        _ffmpeg_avi_to_h264_tmp(tmp_avi, tmp_mp4, fail_log)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        if out_mp4.exists():
            out_mp4.unlink(missing_ok=True)
        tmp_mp4.replace(out_mp4)
    finally:
        tmp_avi.unlink(missing_ok=True)
        tmp_mp4.unlink(missing_ok=True)


def _grid_worker(src: Path) -> None:
    stem = src.stem
    try:
        _system(f"GRID START stem={stem}")
        outA = grid_path_for_group(stem, "A")
        outB = grid_path_for_group(stem, "B")
        if not outA.exists():
            build_fisheye_group_grid_video(src, outA, "A")
        if not outB.exists():
            build_fisheye_group_grid_video(src, outB, "B")
        _system(f"GRID DONE stem={stem}")
    except Exception as e:
        _system(f"GRID FAIL stem={stem} err={e}")
        traceback.print_exc()


def trigger_grid_async(stem: str) -> bool:
    src = find_upload_by_stem(stem)
    if not src:
        raise HTTPException(status_code=404, detail=f"Source not found for stem={stem}")

    with _grid_lock:
        t = _grid_threads.get(stem)
        if t and t.is_alive():
            return False
        th = threading.Thread(target=_grid_worker, args=(src,), daemon=True)
        _grid_threads[stem] = th
        th.start()
        return True


_grid_last_check: Dict[str, float] = {}
_grid_check_lock = threading.Lock()
GRID_CHECK_INTERVAL = 60  # Only check grid every 60 seconds


def maybe_auto_trigger_grid(stem: str, src: Path) -> None:
    """Throttled grid trigger to prevent excessive checks"""
    now = time.time()

    with _grid_check_lock:
        last_check = _grid_last_check.get(stem, 0)
        if now - last_check < GRID_CHECK_INTERVAL:
            return
        _grid_last_check[stem] = now

    hasA, hasB = grid_ready_for_groups(stem)
    if hasA and hasB:
        return

    # Use cached detection
    if detect_video_type_cached(stem, src, force_refresh=False) != "fisheye":
        return

    with _grid_lock:
        if stem in _grid_requested_once:
            return
        _grid_requested_once.add(stem)

    try:
        if trigger_grid_async(stem):
            _system(f"AutoGRID triggered stem={stem}")
    except Exception as e:
        _system(f"AutoGRID ERROR stem={stem} err={e}")


# Add these with other global variables
_live_video_monitor_lock = threading.Lock()
_live_video_ended: Dict[str, float] = {}  # Track when videos ended
_live_video_restart_lock = threading.Lock()
_live_video_restart_pending: Set[
    str] = set()  # Videos pending restart                                          def check_and_restart_ended_videos():

def check_and_restart_ended_videos():
    """
    Restart only FILE/upload videos that truly ended.

    IMPORTANT:
    - RTSP cameras must NOT use file-style "ended" detection.
    - RTSP can briefly stall / queue-empty / reconnect without actually ending.
    - Restarting RTSP here breaks tracking continuity and makes event_log look stuck.
    """
    with _live_video_monitor_lock:
        current_time = time.time()
        videos_to_restart = []

        for cam_id, pipeline in list(pipelines_live.items()):
            try:
                src = get_live_source(cam_id)
                is_rtsp = str(src).startswith("rtsp://") or str(src).startswith("rtsps://")

                # Skip RTSP here
                if is_rtsp:
                    continue

                video_ended = False

                if hasattr(pipeline, "eof_reached"):
                    try:
                        video_ended = bool(pipeline.eof_reached)
                    except Exception:
                        video_ended = False

                elif hasattr(pipeline, "is_running"):
                    try:
                        video_ended = not bool(pipeline.is_running())
                    except Exception:
                        video_ended = False

                elif hasattr(pipeline, "frame_queue"):
                    try:
                        video_ended = pipeline.frame_queue.empty() and bool(getattr(pipeline, "_eof", False))
                    except Exception:
                        video_ended = False

                if video_ended:
                    last_ended = _live_video_ended.get(cam_id, 0)
                    if current_time - last_ended > 10:
                        _live_video_ended[cam_id] = current_time
                        videos_to_restart.append(cam_id)
                        _system(f"LIVE: Video ended for {cam_id}, queued for restart")

            except Exception as e:
                _system(f"Error checking video end for {cam_id}: {e}")

        for cam_id in videos_to_restart:
            restart_single_live_camera(cam_id)

def restart_single_live_camera(cam_id: str) -> bool:
    """
    Restart one LIVE camera pipeline.
    Supports BOTH:
    - upload/file source
    - RTSP source

    Important:
    - keeps ROI file
    - resets live run json/csv logs
    - reapplies detection state after restart
    """
    with _live_video_restart_lock:
        if cam_id in _live_video_restart_pending:
            _system(f"LIVE: Restart already pending for {cam_id}")
            return False
        _live_video_restart_pending.add(cam_id)

    try:
        _system(f"LIVE: Attempting to restart {cam_id}")

        # ---------------------------------
        # 1) remember current detection state
        # ---------------------------------
        detection_config = load_detection_config()
        detection_enabled = bool(detection_config.get(cam_id, True))

        # ---------------------------------
        # 2) stop old pipeline if exists
        # ---------------------------------
        old_p = pipelines_live.get(cam_id)
        if old_p:
            try:
                if hasattr(old_p, "stop"):
                    old_p.stop()
                if hasattr(old_p, "join"):
                    old_p.join(timeout=3.0)
                _system(f"LIVE: Stopped pipeline for {cam_id}")
            except Exception as e:
                _system(f"LIVE: Error stopping pipeline for {cam_id}: {e}")

        try:
            pipelines_live.pop(cam_id, None)
        except Exception:
            pass

        # ---------------------------------
        # 3) resolve source
        # ---------------------------------
        src = get_live_source(cam_id)
        if not src:
            _system(f"LIVE: Source not found for {cam_id}, cannot restart")
            return False

        is_rtsp = str(src).startswith("rtsp://") or str(src).startswith("rtsps://")

        if (not is_rtsp) and (not Path(src).exists()):
            _system(f"LIVE: File source does not exist for {cam_id}: {src}")
            return False

        from video_pipeline import PipelineConfig, VideoPipeline

        # ---------------------------------
        # 4) prepare common paths
        # ---------------------------------
        roi_path = _ensure_roi_file("live", cam_id)

        # only reset for FILE source, not RTSP restart
        if not is_rtsp:
            _reset_live_run_files(cam_id)

        forced_video_type = None
        source_kind = "RTSP" if is_rtsp else "FILE"

        # ---------------------------------
        # 5) branch: RTSP
        # ---------------------------------
        if is_rtsp:
            try:
                rtsp_store = load_rtsp_sources() or {}
                rec = _rtsp_store_get(rtsp_store, cam_id)

                vt = None
                if isinstance(rec, dict):
                    vt = str(rec.get("video_type") or rec.get("type") or "auto").strip().lower()

                if not vt or vt == "auto":
                    vt = detect_rtsp_video_type(cam_id, src)

                if vt not in ("normal", "fisheye"):
                    vt = "normal"

                forced_video_type = vt
            except Exception as e:
                _system(f"LIVE: RTSP type resolve failed for {cam_id}: {e}")
                forced_video_type = "normal"

            cfg = PipelineConfig(
                camera_id=cam_id,
                src= src,
                roi_config_path=str(roi_path),
                num_workers=1,
                max_skip=0.8,

                desired_fps_fisheye=1.0,
                desired_fps_normal=1.0,

                window_scale=0.80,
                display_fps=2.5,
                show_ui=False,
                enable_detection=detection_enabled,
                force_video_type=forced_video_type,
                source_kind="RTSP",

                base_frame_skip_fisheye_rtsp=0,
                base_frame_skip_normal_rtsp=0,

                base_frame_skip_fisheye_file=0,
                base_frame_skip_normal_file=0,

                drop_old_detection_jobs=False,
                latest_only_tracking=False,
            )

        # ---------------------------------
        # 6) branch: upload/file
        # ---------------------------------
        else:
            try:
                forced_video_type = detect_video_type_cached(cam_id, Path(src), force_refresh=False)
            except Exception as e:
                _system(f"LIVE: FILE type resolve failed for {cam_id}: {e}")
                forced_video_type = None

            cfg = PipelineConfig(
                camera_id=cam_id,
                src= str(src),
                roi_config_path=str(roi_path),
                num_workers=1,
                max_skip=0,

                desired_fps_fisheye=6.0,
                desired_fps_normal=10.0,
                window_scale=0.80,
                display_fps=8.0,

                show_ui=False,
                enable_detection=True,
                force_video_type=forced_video_type,
                source_kind="FILE",

                base_frame_skip_fisheye_rtsp=0,
                base_frame_skip_normal_rtsp=0,

                base_frame_skip_fisheye_file=0,
                base_frame_skip_normal_file=0,

                drop_old_detection_jobs=False,
                latest_only_tracking=False,
            )
        # ---------------------------------
        # 7) start new pipeline
        # ---------------------------------
        det_group = _live_detector_group_for_cam(cam_id)
        cfg.detector_group = det_group
        detector = _get_live_detector(det_group)
        new_pipeline = VideoPipeline(cfg, detector)

        if not new_pipeline.start():
            _system(f"LIVE: Failed to start new pipeline for {cam_id}")
            return False

        pipelines_live[cam_id] = new_pipeline

        # ---------------------------------
        # 8) re-apply ROI + detection state
        # ---------------------------------
        try:
            setattr(new_pipeline, "_roi_dirty", True)
        except Exception:
            pass

        try:
            if hasattr(new_pipeline, "reload_roi"):
                new_pipeline.reload_roi()
        except Exception as e:
            _system(f"LIVE: ROI reload failed for {cam_id}: {e}")

        try:
            if hasattr(new_pipeline, "set_detection_enabled"):
                new_pipeline.set_detection_enabled(detection_enabled)
        except Exception as e:
            _system(f"LIVE: detection state reapply failed for {cam_id}: {e}")

        _system(
            f"LIVE: Successfully restarted {cam_id} "
            f"(src={src}, source_kind={source_kind}, type={forced_video_type}, detection={detection_enabled})"
        )
        return True

    except Exception as e:
        _system(f"LIVE: Error creating new pipeline for {cam_id}: {e}")
        traceback.print_exc()
        return False

    finally:
        with _live_video_restart_lock:
            _live_video_restart_pending.discard(cam_id)

def monitor_live_videos_loop():
    """
    Background thread to monitor live videos and restart ended ones
    """
    _system("LIVE video monitor thread started")
    check_interval = 2  # Check every 2 seconds

    while True:
        try:
            check_and_restart_ended_videos()
        except Exception as e:
            _system(f"Error in video monitor loop: {e}")
            traceback.print_exc()

        time.sleep(check_interval)


# ============================================================
# Live Pipeline Management
# ============================================================
def _live_id(cam_id: str) -> str:
    cid = (cam_id or "").strip()
    if cid.startswith("rtsp_"):
        return cid  # keep exact RTSP id
    return _base_id(cid)  # only normalize upload ids


def _build_live_detector():
    try:
        if hasattr(lf, "configure_logging"):
            lf.configure_logging(debug=False, mute_info=True)
    except Exception:
        pass

    try:
        device = lf.DeviceManager.get_device()
    except Exception:
        device = None

    return lf.YoloDetector(
        items_weights=getattr(lf, "WEIGHTS_ITEMS", None),
        coco_weights=getattr(lf, "WEIGHTS_PERSON", None),
        device=device,
        item_capture_conf=0.25,
        coco_capture_conf=0.25,
        person_conf=0.45,
        track_win=10,
        confirm_k=3,
        hold_frames=15,
    )


def _get_live_detector(group_key: str):
    global _live_detectors
    g = str(group_key or "A").upper()

    with _live_detector_lock:
        det = _live_detectors.get(g)
        if det is not None:
            return det

        det = _build_live_detector()
        _live_detectors[g] = det
        return det

# ============================================================
# Detector Group Assignment (Hybrid: manual override + auto)
# ============================================================
#MPORTANT:
#Keep heavy cameras split across A and B.
MANUAL_DETECTOR_GROUPS: Dict[str, str] = {
    # -----------------------------
    # Upload videos (preserve your original sequence)
    # -----------------------------
    "B001G_B_Block_B_Block_20251110103703_20251110104059_41001909": "A",
    "B100D_B_Block_B_Block_20251110101000_20251110104057_40572930": "B",
    "B001G_B_Block_B_Block_20251110083105_20251110084059_39975898": "A",
    "B100D_B_Block_B_Block_20251110081000_20251110084058_39454624": "B",

    # -----------------------------
    # RTSP cameras (preserve your original sequence)
    # Put known heavier fisheye streams on different groups
    # -----------------------------
    "rtsp_1773308594748_819df8013cb51": "A",  # fisheye
    "rtsp_1773308648388_52d4c9fa4e4108": "B",
    "rtsp_1773308690931_36df724b6f90b8": "B",
    "rtsp_1773308703234_4910a4da0c6978": "A",

    "rtsp_1773559227781_07f05b889a7958": "B",  # fisheye
    "rtsp_1773559352792_4bfa714dc33fb8": "A",
    "rtsp_1773559379913_24d01fbe8bd7b": "A",
    "rtsp_1773559393966_d07025f0cb29f8": "B",  # fisheye
}


def _stable_auto_detector_group(cam_id: str) -> str:
    """
    Stable A/B assignment for cameras not listed in MANUAL_DETECTOR_GROUPS.
    Unlike Python hash(), md5-based assignment is stable across restarts.
    """
    cid = str(cam_id or "").strip()
    h = hashlib.md5(cid.encode("utf-8")).hexdigest()
    return "A" if (int(h, 16) % 2 == 0) else "B"


def _live_detector_group_for_cam(cam_id: str) -> str:
    """
    Hybrid strategy:
    1) Manual override for known heavy/special cameras
    2) Stable auto-balance for all others
    """
    cid = str(cam_id or "").strip()

    if cid in MANUAL_DETECTOR_GROUPS:
        group = str(MANUAL_DETECTOR_GROUPS[cid]).upper().strip()
        if group not in ("A", "B"):
            group = "A"
        _system(f"GROUP RESOLVE cam_id={cid} resolved_group={group} mode=MANUAL")
        return group

    group = _stable_auto_detector_group(cid)
    _system(f"GROUP RESOLVE cam_id={cid} resolved_group={group} mode=AUTO")
    return group

def start_live_pipelines(limit_normal: int = 999, limit_fisheye: int = 999):
    """
    Start LIVE pipelines for:
    - Upload videos
    - RTSP sources
    """
    global pipelines_live
    from video_pipeline import PipelineConfig, VideoPipeline

    with _live_pipe_lock:
        enabled_upload = load_cameras_enabled() or {}

        # -----------------------------
        # 1) Upload videos
        # -----------------------------
        files = list_upload_videos_h264_only()
        normals, fisheyes = [], []
        for f in files:
            cam_id = _base_id(_cam_id_from_h264_file(f))
            if enabled_upload.get(cam_id, True) is False:
                continue

            t = detect_video_type_cached(f.stem, f, force_refresh=False)
            (fisheyes if t == "fisheye" else normals).append(f)

        normals = normals[:max(0, int(limit_normal))]
        fisheyes = fisheyes[:max(0, int(limit_fisheye))]

        # -----------------------------
        # 2) RTSP sources enabled list + type
        # -----------------------------
        rtsp_list: List[Dict[str, Any]] = []
        try:
            rtsp = load_rtsp_sources() or {}
            for cam_id, rec in (rtsp.items() if isinstance(rtsp, dict) else []):
                if not isinstance(rec, dict):
                    continue
                url = (rec.get("url") or "").strip()
                if not url:
                    continue
                if not bool(rec.get("enabled", True)):
                    continue

                cam_id = str(cam_id).strip()
                url = _encode_rtsp_credentials(url)

                # ✅ Decide type:
                vt = (rec.get("video_type") or "auto").strip().lower()
                if vt == "auto":
                    vt = detect_rtsp_video_type(cam_id, url)
                elif vt not in ("normal", "fisheye"):
                    vt = "normal"

                rtsp_list.append({"id": cam_id, "url": url, "video_type": vt})
        except Exception as e:
            _system(f"LIVE: load_rtsp_sources error: {e}")

        # -----------------------------
        # 3) stop old pipelines
        # -----------------------------
        for _, p in list(pipelines_live.items()):
            try:
                if hasattr(p, "stop"):
                    p.stop()
                if hasattr(p, "join"):
                    p.join(timeout=3.0)
            except Exception:
                pass
        pipelines_live = {}



        # -----------------------------
        # 4) Start upload pipelines
        # -----------------------------
        for f in (normals + fisheyes):
            cam_id = _base_id(_cam_id_from_h264_file(f))
            roi_path = _ensure_roi_file("live", cam_id)
            _reset_live_run_files(cam_id)

            cfg = PipelineConfig(
                camera_id=cam_id,
                src=str(f),
                roi_config_path=str(roi_path),
                num_workers=1,
                max_skip=0,

                desired_fps_fisheye=6.0,
                desired_fps_normal=10.0,
                window_scale=0.80,
                display_fps=8.0,

                show_ui=False,
                enable_detection=True,
                force_video_type=None,
                source_kind="FILE",

                base_frame_skip_fisheye_rtsp=0,
                base_frame_skip_normal_rtsp=0,

                base_frame_skip_fisheye_file=0,
                base_frame_skip_normal_file=0,

                drop_old_detection_jobs=False,
                latest_only_tracking=False,
            )

            det_group = _live_detector_group_for_cam(cam_id)
            cfg.detector_group = det_group
            detector = _get_live_detector(det_group)
            p = VideoPipeline(cfg, detector)
            if not p.start():
                raise RuntimeError(f"pipeline start failed cam_id={cam_id} mode=live")

            pipelines_live[cam_id] = p
            _system(f"LIVE started cam_id={cam_id} (UPLOAD)")

        # -----------------------------
        # 5) Start RTSP pipelines (WITH type)
        # -----------------------------
        for rec in rtsp_list:
            cam_id = rec["id"]
            if cam_id in pipelines_live:
                continue

            roi_path = _ensure_roi_file("live", cam_id)

            cfg = PipelineConfig(
                camera_id=cam_id,
                src=rec["url"],
                roi_config_path=str(roi_path),
                num_workers=1,
                max_skip=0.8,

                desired_fps_fisheye=1.0,
                desired_fps_normal=1.0,

                window_scale=0.80,
                display_fps=2.5,
                show_ui=False,
                enable_detection=True,
                force_video_type=rec["video_type"],
                source_kind="RTSP",

                base_frame_skip_fisheye_rtsp=0,
                base_frame_skip_normal_rtsp=0,

                base_frame_skip_fisheye_file=0,
                base_frame_skip_normal_file=0,

                drop_old_detection_jobs=False,
                latest_only_tracking=False,
            )
            det_group = _live_detector_group_for_cam(cam_id)
            cfg.detector_group = det_group
            detector = _get_live_detector(det_group)
            p = VideoPipeline(cfg, detector)
            if not p.start():
                _system(f"LIVE: pipeline start failed cam_id={cam_id} (RTSP)")
                continue

            pipelines_live[cam_id] = p
            _system(f"LIVE started cam_id={cam_id} (RTSP) type={rec['video_type']}")


def get_zones_for_normal_video(roi_cfg):
    """
    Extract zones from normal video ROI configuration.

    Args:
        roi_cfg: ROI configuration dictionary with 'bounding_polygons'

    Returns:
        List of zone dictionaries with shape, points, and roi_id
    """
    if not roi_cfg:
        return []

    # Get bounding polygons from config
    polygons = roi_cfg.get("bounding_polygons", [])
    if not polygons:
        return []

    zones = []
    for i, poly in enumerate(polygons):
        # Convert polygon points to list of (x, y) tuples
        points = []
        for point in poly:
            if isinstance(point, dict) and "x" in point and "y" in point:
                points.append((float(point["x"]), float(point["y"])))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))

        # Only add if we have at least 3 points (valid polygon)
        if len(points) >= 3:
            zones.append({
                "shape": "polygon",
                "points": points,
                "roi_id": i + 1,
                "zone_id": i + 1
            })

    return zones


def draw_zones_on_view(img, zones, color=(0, 255, 0), thickness=2, show_id=True):
    """
    Draw ROI zones on an image.

    Args:
        img: BGR image
        zones: List of zone dictionaries
        color: BGR color tuple
        thickness: Line thickness
        show_id: Whether to show ROI IDs

    Returns:
        Image with zones drawn
    """
    if img is None or not zones:
        return img

    out = img.copy()
    H, W = out.shape[:2]

    for zone in zones:
        if zone.get("shape") != "polygon":
            continue

        points = zone.get("points", [])
        if len(points) < 3:
            continue

        # Convert points to numpy array for drawing
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        # Draw polygon
        cv2.polylines(out, [pts], True, color, thickness)

        # Draw ROI ID if requested
        if show_id:
            roi_id = zone.get("roi_id") or zone.get("zone_id") or "?"
            # Find a good position for the label (use first point)
            if points:
                x, y = int(points[0][0]), int(points[0][1])
                # Add background for text
                cv2.rectangle(out, (x, y - 25), (x + 60, y), color, -1)
                cv2.putText(
                    out,
                    f"ROI {roi_id}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

    return out


# Update the live_pump function to respect detection toggles
def live_pump() -> None:
    """Main live update loop with detection toggle support"""
    _system("LIVE pump started (with detection toggle support)")
    last_config_load = 0
    detection_config = {}

    while True:
        try:
            settings = load_lf_settings()
            enabled_cams = settings.get("cameras_enabled", {}) or {}
            enabled_classes = settings.get("enabled_classes", {}) or {}

            # Reload detection config every 5 seconds to pick up changes
            now = time.time()
            if now - last_config_load > 5:
                detection_config = load_detection_config()
                last_config_load = now

            # Process each live pipeline
            for cam_id, p in list(pipelines_live.items()):
                if not hasattr(p, "pull_latest_for_api"):
                    continue

                # Check if detection is enabled for this camera
                detection_enabled = detection_config.get(cam_id, True)

                # Update pipeline's detection state if it has the method
                if hasattr(p, "set_detection_enabled"):
                    try:
                        p.set_detection_enabled(detection_enabled)
                    except Exception:
                        pass

                # If detection disabled, clear detections before sending to hub
                if not detection_enabled:
                    try:
                        views_payload = p.pull_latest_for_api() or []
                        for v in views_payload:
                            if not isinstance(v, dict):
                                continue
                            v["zones"] = []   # IMPORTANT: clear ROI overlay too
                            v["dets"] = []
                            v["detections"] = []
                            v["detection_enabled"] = False

                        hub.update(cam_id, views_payload, lost_items=[])
                    except Exception as e:
                        _system(f"Error processing disabled detection for {cam_id}: {e}")
                    continue

                # Normal processing with detection enabled
                try:
                    views_payload = p.pull_latest_for_api() or []

                    # ================= FIX: normal empty ROI must stay empty =================
                    src = find_upload_by_stem(cam_id)
                    if src:
                        vtype = detect_video_type_cached(cam_id, src)

                        if vtype != "fisheye" and views_payload:
                            roi_path = _ensure_roi_file("live", cam_id)
                            roi_config = {}
                            if roi_path.exists():
                                try:
                                    roi_config = json.loads(roi_path.read_text(encoding="utf-8"))
                                except Exception:
                                    pass

                            zones = get_zones_for_normal_video(roi_config) or []

                            # Always set zones explicitly
                            if len(views_payload) > 0 and isinstance(views_payload[0], dict):
                                views_payload[0]["zones"] = zones

                                # IMPORTANT:
                                # if no ROI, clear stale detections from pipeline payload
                                if not zones:
                                    views_payload[0]["dets"] = []
                                    views_payload[0]["detections"] = []
                    # =========================================================================

                    # Add detection enabled flag
                    for v in views_payload:
                        if not isinstance(v, dict):
                            continue
                        v["detection_enabled"] = True
                        v["zones"] = v.get("zones", []) or []

                except Exception as e:
                    _system(f"Error pulling views for {cam_id}: {e}")
                    continue

                # Get lost items
                try:
                    lost_items = (
                        p.pull_lost_items_for_api()
                        if hasattr(p, "pull_lost_items_for_api")
                        else []
                    )
                    if not isinstance(lost_items, list):
                        lost_items = []
                except Exception:
                    lost_items = []

                cam_on = bool(enabled_cams.get(cam_id, True))

                # Process detections
                for group_idx, v in enumerate(views_payload or []):
                    if not isinstance(v, dict):
                        continue

                    dets = v.get("dets", None)
                    if dets is None:
                        dets = v.get("detections", None)
                    if dets is None:
                        dets = []
                    if not isinstance(dets, list):
                        dets = [dets]

                    try:
                        gi = v.get("gi", v.get("group_idx", v.get("group", group_idx)))
                        gi = int(gi) if gi is not None else group_idx
                    except Exception:
                        gi = group_idx

                    norm = []
                    for d in dets:
                        if not isinstance(d, dict):
                            continue

                        dd = dict(d)
                        dd["label"] = _det_label(dd)

                        local = dd.get("view_id", dd.get("view_idx", dd.get("view", dd.get("local_id", 0))))
                        try:
                            local = int(local)
                        except Exception:
                            local = 0

                        if gi == 0:
                            global_view_id = int(local) % 4
                        else:
                            if 4 <= int(local) <= 7:
                                global_view_id = int(local)
                            else:
                                global_view_id = 4 + (int(local) % 4)

                        dd["view_id"] = global_view_id
                        dd.setdefault("img_w", dd.get("img_w") or 640)
                        dd.setdefault("img_h", dd.get("img_h") or 480)
                        norm.append(dd)

                    if not cam_on:
                        norm = []
                        lost_items = []
                    else:
                        kept = []
                        for dd in norm:
                            lab = (dd.get("label") or "").lower().strip()
                            if enabled_classes.get(lab, True):
                                kept.append(dd)
                        norm = kept
                        # keep only top detections to reduce frontend / hub load
                        try:
                            norm.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                        except Exception:
                            pass

                        MAX_DETS_PER_VIEW = 15
                        norm = norm[:MAX_DETS_PER_VIEW]

                    # IMPORTANT:
                    # if normal video and zones empty, force no detections
                    try:
                        src = find_upload_by_stem(cam_id)
                        if src:
                            vtype = detect_video_type_cached(cam_id, src)
                            if vtype != "fisheye":
                                zones = v.get("zones", []) or []
                                if not zones:
                                    norm = []
                    except Exception:
                        pass

                    v["dets"] = norm
                    v["detections"] = norm
                    v["gi"] = gi
                    v["detection_enabled"] = detection_enabled

                # IMPORTANT:
                # if camera off, no lost items
                if not cam_on:
                    lost_items = []

                # Update hub
                hub.update(cam_id, views_payload, lost_items=lost_items)

        except Exception as e:
            _system(f"Error in live_pump: {e}")
            traceback.print_exc()

        time.sleep(0.05)


def live_auto_toggle_fisheye(period_sec: float = 1.5) -> None:
    _system("LIVE auto-toggle started (fisheye A<->B)")
    while True:
        try:
            # ✅ LIVE ONLY
            for cam_id, p in list(pipelines_live.items()):
                try:
                    if getattr(p, "is_fisheye", False) and hasattr(p, "active_views"):
                        p.active_views.toggle()
                        # optional queue drain
                        try:
                            from video_pipeline import drain_queue
                            drain_queue(p.frame_queue)
                            drain_queue(p.bundle_job_queue)
                            drain_queue(p.det_out_queue)
                            drain_queue(p.out_queue)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

        time.sleep(period_sec)


# ============================================================
# Offline Conversion Management
# ============================================================
def _ensure_offline_record_for_file(f: Path) -> None:
    stem = f.stem
    promoted = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"
    st = "ready" if promoted.exists() else "processing"

    _manifest_upsert(
        stem,
        {
            "name": f.name,
            "size": _human_mb(f.stat().st_size),
            "uploadDate": int(f.stat().st_mtime),
            "status": st,
            "h264_ready": bool(promoted.exists()),
            "h264_name": f"{stem}_h264.mp4",
            "error": None,
        },
    )


def _convert_and_promote_worker(stem: str, original_path: Path, target_dir: Path) -> None:
    """
    target_dir:
      - OFFLINE_UPLOAD_DIR for offline uploads
      - UPLOAD_DIR for upload-scan conversions
    """
    try:
        _wait_file_stable(original_path)

        _manifest_upsert(stem, {"status": "processing", "error": None, "h264_ready": False})

        promoted = ensure_h264_promoted(stem, original_path, target_dir)

        invalidate_video_type_cache(stem)

        _manifest_upsert(
            stem,
            {
                "status": "ready",
                "h264_ready": True,
                "duration": _video_duration_hms(promoted),
                "error": None,
            },
        )

        try:
            if target_dir.resolve() == UPLOAD_DIR.resolve():
                if detect_video_type_cached(stem, promoted, force_refresh=False) == "fisheye":
                    maybe_auto_trigger_grid(stem, promoted)
        except Exception:
            pass

    except Exception as e:
        _manifest_upsert(stem, {"status": "failed", "error": str(e), "h264_ready": False})
        _system(f"CONVERT FAIL stem={stem} err={e}")
        traceback.print_exc()

def _startup_scan_offline_folder() -> None:
    """
    On startup:
    - read existing files in offline_upload/
    - ensure they exist in manifest
    - if offline_upload/<stem>_h264.mp4 missing -> start background conversion (STRICT offline)
    """
    try:
        for f in OFFLINE_UPLOAD_DIR.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith("__upload_tmp__"):
                continue
            if f.name.lower().endswith(".ffmpeg.log"):
                continue
            if f.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
                continue
            if f.name.lower().endswith("_h264.mp4"):
                continue

            _ensure_offline_record_for_file(f)
            stem = f.stem
            promoted = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"
            if not promoted.exists():
                threading.Thread(
                    target=_convert_and_promote_worker,
                    args=(stem, f, OFFLINE_UPLOAD_DIR),
                    daemon=True,
                ).start()

    except Exception as e:
        _system(f"Startup offline scan error: {e}")


def _startup_scan_upload_folder() -> None:
    """
    Scan upload/ for any video that is NOT *_h264.mp4 and convert it into upload/<stem>_h264.mp4.

    ✅ Ensures file like:
    C:\\Users\\admin\\Documents\\LostAnd\\backend\\upload\\B100D_....mp4
    will be auto converted to:
    upload\\B100D_...._h264.mp4
    """
    try:
        for f in UPLOAD_DIR.iterdir():
            if not f.is_file():
                continue

            # ignore grid cache folder outputs
            if f.name.startswith("_"):
                continue
            if f.name.lower().endswith(".ffmpeg.log"):
                continue

            ext = f.suffix.lower()
            if ext not in (".mp4", ".avi", ".mov", ".mkv"):
                continue

            name_lower = f.name.lower()

            # Case A: file is already *_h264.mp4 but maybe wrong codec
            if name_lower.endswith("_h264.mp4"):
                stem = f.stem
                base_stem = stem[:-5] if stem.lower().endswith("_h264") else stem

                if _is_h264_mp4(f):
                    continue

                # ✅ Stability: do NOT rebuild existing _h264 at startup (may be in use by LIVE)
                _system(f"UPLOAD scan: detected non-h264 _h264 file but skipping rebuild (stability): {f.name}")
                continue

            # Case B: original/non-h264 file dropped into upload/
            base_stem = f.stem
            promoted = UPLOAD_DIR / f"{base_stem}_h264.mp4"

            if promoted.exists() and promoted.stat().st_size > 256 * 1024:
                continue

            # ensure manifest record exists immediately (optional)
            _manifest_upsert(
                base_stem,
                {
                    "name": f.name,
                    "size": _human_mb(f.stat().st_size),
                    "uploadDate": int(f.stat().st_mtime),
                    "status": "processing",
                    "h264_ready": False,
                    "h264_name": f"{base_stem}_h264.mp4",
                    "error": None,
                },
            )

            _system(f"UPLOAD scan: converting -> {base_stem}_h264.mp4")
            threading.Thread(
                target=_convert_and_promote_worker,
                args=(base_stem, f, UPLOAD_DIR),
                daemon=True,
            ).start()

    except Exception as e:
        _system(f"Startup upload scan error: {e}")


# ============================================================
# Offline Analyze Management
# ============================================================
def _clear_outputs_keep_roi(stem: str) -> None:
    folder = _outputs_dir("offline", stem)
    folder.mkdir(parents=True, exist_ok=True)

    keep = {"roi_config.json"}
    for p in folder.iterdir():
        if p.is_file() and p.name not in keep:
            p.unlink(missing_ok=True)


def _is_run_dir_for_stem(p: Path, stem: str) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    name = p.name
    return name == stem or name.startswith(stem + "_")


def _pick_latest_run_dir(stem: str) -> Optional[Path]:
    roots = [
        OFFLINE_ROOT,  # backend offline outputs
        PROJECT_OUTPUTS_LF_DIR / "offline",  # project offline outputs (if any)
        OUTPUTS_LF_DIR,  # legacy fallback
        PROJECT_OUTPUTS_LF_DIR,  # legacy fallback
    ]

    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        # If root has direct stem folder
        d = root / stem
        if _is_run_dir_for_stem(d, stem):
            candidates.append(d)
        # Also scan children in case scripts create stem_timestamp folders
        try:
            for ch in root.iterdir():
                if _is_run_dir_for_stem(ch, stem):
                    candidates.append(ch)
        except Exception:
            pass

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return candidates[0]


def _consolidate_outputs_to_stem_folder(stem: str) -> None:
    target = _outputs_dir("offline", stem)
    target.mkdir(parents=True, exist_ok=True)

    run_dir = _pick_latest_run_dir(stem)
    if not run_dir:
        return
    if run_dir.resolve() == target.resolve():
        return

    names = [
        "roi_config.json",
        "progress.json",
        "lost_items.json",
        "lost_items.csv",
        "event_log.jsonl",
        "offline_analyze.log",
    ]

    for name in names:
        _safe_move(run_dir / name, target / name)

    try:
        if run_dir.exists() and run_dir.is_dir() and not any(run_dir.iterdir()):
            run_dir.rmdir()
    except Exception:
        pass


def _mirror_run_outputs(stem: str, stop_flag: threading.Event):
    target = _outputs_dir("offline", stem)
    target.mkdir(parents=True, exist_ok=True)

    names = ["progress.json", "event_log.jsonl", "lost_items.json", "lost_items.csv"]

    while not stop_flag.is_set():
        run_dir = _pick_latest_run_dir(stem)
        if run_dir and run_dir.exists():
            for name in names:
                src = run_dir / name
                dst = target / name
                if src.exists():
                    try:
                        dst.write_bytes(src.read_bytes())
                    except Exception:
                        pass
        time.sleep(0.5)


def _run_lostfound_subprocess(stem: str, h264_path: Path) -> None:
    try:
        with _offline_jobs_lock:
            _offline_jobs[stem] = {"status": "processing", "error": None, "started_at": time.time()}
        _manifest_upsert(stem, {"status": "analyzing", "error": None})

        roi_path = _ensure_roi_file("offline", stem)
        _clear_outputs_keep_roi(stem)
        env = dict(os.environ)
        env["LF_VIDEO_PATH"] = str(h264_path.resolve())
        env["LF_HEADLESS"] = "1"
        env["LF_AUTO_ROI"] = "1"  # ROI must come from UI
        env["LF_BACKEND"] = "1"
        env["LF_ROI_CONFIG"] = str(roi_path)
        env["LF_RUN_PREFIX"] = stem
        env["LF_FISHEYE_MODE"] = "A"  # start with A
        env["LF_AUTO_TOGGLE_SEC"] = "30"
        env["LF_OUTPUT_DIR"] = str(_outputs_dir("offline", stem).resolve())

        project_root = BASE_DIR.parent
        lf_script = project_root / "lostandfound.py"

        cmd = [os.sys.executable, str(lf_script)]
        log_dir = _outputs_dir("offline", stem)
        log_path = log_dir / "offline_analyze.log"

        _system(f"OFFLINE Analyze START stem={stem}")
        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
            project_root = BASE_DIR.parent

            # unbuffered so offline_analyze.log updates immediately
            env["PYTHONUNBUFFERED"] = "1"

            stop_flag = threading.Event()
            threading.Thread(target=_mirror_run_outputs, args=(stem, stop_flag), daemon=True).start()

            p = subprocess.Popen(
                [os.sys.executable, "-u", str(lf_script)],
                cwd=str(project_root),  # ✅ IMPORTANT
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            assert p.stdout is not None
            for line in p.stdout:
                f.write(line)
                f.flush()  # ✅ so you can see updates while running

            rc = p.wait()
            stop_flag.set()

        if rc != 0:
            raise RuntimeError(f"lostandfound.py failed. See {log_path}")

        try:
            _consolidate_outputs_to_stem_folder(stem)
        except Exception as e:
            _system(f"Consolidate outputs failed stem={stem} err={e}")

        with _offline_jobs_lock:
            j = _offline_jobs.get(stem) or {}
            j.update({"status": "done", "error": None, "ended_at": time.time()})
            _offline_jobs[stem] = j

        _manifest_upsert(stem, {"status": "ready", "error": None})

    except Exception as e:
        with _offline_jobs_lock:
            j = _offline_jobs.get(stem) or {}
            j.update({"status": "failed", "error": str(e), "ended_at": time.time()})
            _offline_jobs[stem] = j

        _manifest_upsert(stem, {"status": "failed", "error": str(e)})


def _wait_h264_and_start_analyze(stem: str) -> None:
    """
    Waits up to 120s total for h264 file to exist then launches analysis.
    """
    start = time.time()
    try:
        _system(f"QUEUE: waiting for h264 stem={stem}")
        while time.time() - start < 120:
            h264_path = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"
            if h264_path.exists() and h264_path.stat().st_size > 256 * 1024:
                _system(f"QUEUE: h264 ready -> start analyze stem={stem}")
                _run_lostfound_subprocess(stem, h264_path)
                return
            time.sleep(0.4)

        # timeout
        _manifest_upsert(stem, {"status": "failed", "error": "Analyze queued but h264 not ready within 120s"})
        _system(f"QUEUE: timeout stem={stem}")
    finally:
        with _analyze_queue_lock:
            _analyze_queued.discard(stem)


def queue_analyze_if_needed(stem: str) -> None:
    with _analyze_queue_lock:
        if stem in _analyze_queued:
            return
        _analyze_queued.add(stem)
    threading.Thread(target=_wait_h264_and_start_analyze, args=(stem,), daemon=True).start()


# ============================================================
# FastAPI App Setup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    def boot_job():
        _system("Startup: scanning upload/ for non-h264 videos (auto-convert)...")
        _startup_scan_upload_folder()

        _system("Startup: scanning offline_upload (manifest + auto-convert missing h264)...")
        _startup_scan_offline_folder()

        try:
            run_data_retention_cleanup()
        except Exception as e:
            _system(f"Startup: data retention cleanup failed: {e}")

        _system("Startup: scanning uploads for fisheye grid (non-blocking)...")
        files = list_upload_videos_h264_only()
        _system(f"Startup: found {len(files)} h264 uploads")
        for f in files:
            stem = f.stem
            hasA, hasB = grid_ready_for_groups(stem)
            if hasA and hasB:
                continue
            try:
                if detect_video_type_cached(stem, f) == "fisheye":
                    trigger_grid_async(stem)
            except Exception:
                pass

    # Update the live_boot function to initialize detection config
    def live_boot():
        """Start live pipelines and monitoring"""
        try:
            _system("Startup: starting LIVE pipelines (detection)...")
            start_live_pipelines(limit_normal=8, limit_fisheye=4)

            # Initialize detection config from pipelines
            init_detection_config_from_pipelines()

            # Set pipelines to use LIVE pipelines as source of truth
            global pipelines
            pipelines = pipelines_live

            # Start background threads
            threading.Thread(target=live_pump, daemon=True).start()
            threading.Thread(target=live_auto_toggle_fisheye, daemon=True).start()

            # START THE VIDEO MONITOR THREAD
            threading.Thread(target=monitor_live_videos_loop, daemon=True).start()
            threading.Thread(target=data_retention_loop, daemon=True).start()

            _system(f"LIVE pipelines started: {len(pipelines_live)} cameras")
            _system("LIVE video monitor started")

            # Log cache stats
            with _video_type_cache_lock:
                _system(f"Video type cache initialized with {len(_video_type_cache)} entries")

        except Exception as e:
            _system(f"Startup: LIVE failed err={e}")
            traceback.print_exc()

    # ✅ STARTUP (this is what you missed)
    threading.Thread(target=boot_job, daemon=True).start()
    threading.Thread(target=live_boot, daemon=True).start()

    yield

    # ✅ SHUTDOWN
    _system("Shutdown: backend stopping")

# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(title="Lost & Found Backend", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://qy248.github.io",
        "https://jinxuan-wong.github.io",
        "https://tarumtfocslab.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# media
app.mount("/videos", StaticFiles(directory=str(UPLOAD_DIR)), name="videos")
app.mount("/grid", StaticFiles(directory=str(GRID_DIR)), name="grid")
app.mount("/offline", StaticFiles(directory=str(OFFLINE_UPLOAD_DIR)), name="offline")
app.mount("/lf_outputs", StaticFiles(directory=str(OUTPUTS_LF_DIR)), name="lf_outputs")


# ============================================================
# API Routes
# ============================================================
@app.get("/")
def root():
    return {
        "ok": True,
        "hint": "Use /api/offline/videos, /api/offline/upload, /api/offline/roi/{id}, /api/offline/analyze, /api/lostfound/cameras, /api/live/state",
        "upload_dir": str(UPLOAD_DIR),
        "offline_upload_dir": str(OFFLINE_UPLOAD_DIR),
        "outputs_lf_dir": str(OUTPUTS_LF_DIR),
        "tmp_dir": str(TMP_DIR),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/health")
def render_health():
    return {"status": "ok"}


@app.get("/api/lostfound/alerts")
def lostfound_alerts(request: Request, limit: int = 50):
    lim = max(1, min(int(limit or 50), 200))

    overrides = _read_overrides()

    store = _read_items_store()
    if not isinstance(store, dict):
        store = {}

    items = []
    for _, it in store.items():
        if not isinstance(it, dict):
            continue
        it = _apply_override(dict(it), overrides)
        if it.get("deleted") is True:
            continue
        if (it.get("status") or "lost") != "lost":
            continue
        items.append(it)

    def _ts(x: dict) -> float:
        t = x.get("lastSeenTs")
        if t is None:
            t = x.get("firstSeenTs")
        try:
            return float(t or 0)
        except Exception:
            return 0.0

    items.sort(key=_ts, reverse=True)
    items = items[:lim]

    base = str(request.base_url).rstrip("/")
    out = []

    for it in items:
        iid = str(it.get("id") or "")
        cam_id = str(it.get("cameraId") or it.get("videoId") or it.get("source") or "")
        cam_name = str(it.get("location") or cam_id or "Unknown")

        label = str(it.get("label") or "item")
        loc = str(it.get("location") or cam_id or "unknown location")

        ts = _ts(it)
        severity = "medium"
        if label.lower() in ("wallet", "mobile_phone", "laptop", "tablet"):
            severity = "high"

        snap = (
            it.get("snapshot")
            or it.get("snapshot_path")
            or it.get("image_path")
            or None
        )

        # IMPORTANT:
        # rebuild fresh URL from local snapshot path first
        img = None
        if snap:
            try:
                img = to_image_url(str(snap), base)
            except Exception:
                img = None

        # fallback only if no snapshot-based url available
        if not img:
            img = (
                it.get("imageUrl")
                or it.get("thumbUrl")
                or it.get("evidence_url")
                or it.get("snapshot_url")
                or None
            )

        if img is not None:
            img = str(img)

        out.append(
            {
                "id": iid or f"lf_{int(ts)}",
                "cameraId": cam_id or "unknown",
                "cameraName": cam_name,
                "type": "lost-found",
                "timestamp": ts,
                "severity": severity,
                "message": f"Lost item detected: {label} ({loc})",
                "imageUrl": img,
                "thumbUrl": img,
            }
        )

    return {"alerts": out}


@app.post("/api/lostfound/alerts/{alert_id}/dismiss")
def dismiss_lostfound_alert(alert_id: str):
    store = _read_items_store()
    if not isinstance(store, dict):
        store = {}

    found = False

    for key, it in store.items():
        if not isinstance(it, dict):
            continue

        iid = str(it.get("id") or "")
        if iid == str(alert_id):
            it["deleted"] = True
            store[key] = it
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="Alert not found")

    _write_items_store(store)
    return {"ok": True, "id": alert_id}

_lf_notif_subscribers: List[Queue] = []
_lf_notif_lock = Lock()
_lf_emitted_ids: Set[str] = set()

def _lf_broadcast_notification(payload: Dict[str, Any]) -> None:
    with _lf_notif_lock:
        dead = []
        for q in _lf_notif_subscribers:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                _lf_notif_subscribers.remove(q)
            except Exception:
                pass


def _lf_register_subscriber() -> Queue:
    q = Queue()
    with _lf_notif_lock:
        _lf_notif_subscribers.append(q)
    return q


def _lf_unregister_subscriber(q: Queue) -> None:
    with _lf_notif_lock:
        try:
            _lf_notif_subscribers.remove(q)
        except ValueError:
            pass

@app.get("/api/lostfound/notifications/stream")
def lostfound_notifications_stream():
    def event_gen():
        q = _lf_register_subscriber()
        try:
            while True:
                try:
                    payload = q.get(timeout=15.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except Empty:
                    yield ": keepalive\n\n"
        finally:
            _lf_unregister_subscriber(q)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ---------- settings ----------
@app.get("/api/lostfound/settings")
def get_lf_settings_api():
    return load_lf_settings()


@app.post("/api/lostfound/settings")
def save_lf_settings_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save Lost & Found UI settings, including numeric data retention days.
    """
    clean_settings = _normalize_lf_settings(payload)
    saved = save_lf_settings(clean_settings)

    try:
        if "data_retention_days" in (payload or {}):
            run_data_retention_cleanup()
    except Exception as e:
        _system(f"LF SETTINGS retention cleanup error: {e}")

    _system(f"LF SETTINGS saved keys: {list(saved.keys())}")
    return saved


@app.post("/api/lostfound/settings/retention/run")
def run_lf_retention_cleanup_api() -> Dict[str, Any]:
    return run_data_retention_cleanup()

@app.post("/api/lostfound/events/clear_all")
def clear_all_lostfound_events() -> Dict[str, Any]:
    removed_files = 0

    try:
        with _items_store_lock:
            _lf_rewrite_all_shards(
                LF_STORE_ITEMS_DIR,
                LF_STORE_ITEMS_PREFIX,
                {},
                LF_ITEMS_PER_FILE,
            )

        with _overrides_lock:
            _lf_rewrite_all_shards(
                LF_STORE_OVERRIDES_DIR,
                LF_STORE_OVERRIDES_PREFIX,
                {},
                LF_OVERRIDES_PER_FILE,
            )

        for base_dir in [OUTPUTS_LF_DIR, PROJECT_OUTPUTS_LF_DIR]:
            if not base_dir.exists():
                continue

            for p in base_dir.rglob("*"):
                if not p.is_file():
                    continue

                if not _is_safe_lf_event_file(p):
                    continue

                try:
                    p.unlink(missing_ok=True)
                    removed_files += 1
                except Exception:
                    pass

        _system(f"LF EVENTS clear_all removed_files={removed_files}")
        return {
            "ok": True,
            "removed_files": removed_files,
            "message": "All Lost & Found events cleared",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"clear_all failed: {e}")

# ---------- dashboard cameras (from upload/*_h264.mp4) ----------
@app.get("/api/lostfound/cameras")
def get_lostfound_cameras(request: Request, start: str = "A"):
    base = str(request.base_url).rstrip("/")
    cams: List[Dict[str, Any]] = []

    start = (start or "A").upper()
    if start not in ("A", "B"):
        start = "A"

    # -----------------------------
    # Upload cameras (existing behavior)
    # -----------------------------
    files = list_upload_videos_h264_only()

    for file in files:
        stem = file.stem
        cam_id = _cam_id_from_h264_file(file)  # removes _h264
        cam_name = cam_name_from_file(file)
        vtype = detect_video_type_cached(stem, file)

        if vtype == "fisheye":
            outA = grid_path_for_group(stem, "A")
            outB = grid_path_for_group(stem, "B")

            if outA.exists() and outB.exists():
                a_rel = outA.relative_to(GRID_DIR).as_posix()
                b_rel = outB.relative_to(GRID_DIR).as_posix()

                views = [
                    {"id": f"{stem}__A", "name": f"{cam_name} (Group A)", "videoUrl": f"{base}/grid/{a_rel}",
                     "filename": a_rel, "order": 0},
                    {"id": f"{stem}__B", "name": f"{cam_name} (Group B)", "videoUrl": f"{base}/grid/{b_rel}",
                     "filename": b_rel, "order": 1},
                ]
                if start == "B":
                    views = [views[1], views[0]]

                cams.append(
                    {
                        "id": stem,
                        "name": cam_name,
                        "groupId": stem,
                        "order": 0,
                        "classroomId": cam_name,
                        "location": "Block B - Uploaded",
                        "status": "online",
                        "recording": True,
                        "videoUrl": views[0]["videoUrl"],
                        "filename": views[0]["filename"],
                        "views": views,
                    }
                )
            else:
                cams.append(
                    {
                        "id": stem,
                        "name": cam_name,
                        "groupId": stem,
                        "order": 0,
                        "classroomId": cam_name,
                        "location": "Block B - Uploaded",
                        "status": "online",
                        "recording": True,
                        "videoUrl": f"{base}/videos/{file.name}",
                        "filename": file.name,
                        "views": [
                            {"id": f"{stem}__A", "name": f"{cam_name} (Group A)", "order": 0},
                            {"id": f"{stem}__B", "name": f"{cam_name} (Group B)", "order": 1},
                        ],
                    }
                )
                maybe_auto_trigger_grid(stem, file)
            continue

        cams.append(
            {
                "id": stem,
                "name": cam_name,
                "groupId": stem,
                "order": 0,
                "classroomId": cam_name,
                "location": "Block B - Uploaded",
                "status": "online",
                "recording": True,
                "videoUrl": f"{base}/videos/{file.name}",
                "filename": file.name,
            }
        )

    # -----------------------------
    # ✅ RTSP cameras (with video type)
    # -----------------------------
    try:
        rtsp = load_rtsp_sources() or {}
        for rid, rec in rtsp.items():
            if not isinstance(rec, dict):
                continue
            url = (rec.get("url") or "").strip()
            if not url:
                continue

            rid = str(rid).strip()
            name = (rec.get("name") or rid).strip()
            enabled = bool(rec.get("enabled", True))
            url = _encode_rtsp_credentials(url)

            # ✅ determine video type
            vt = (rec.get("video_type") or "auto").strip().lower()
            if vt == "auto":
                vt = detect_rtsp_video_type(rid, url)
            elif vt not in ("normal", "fisheye"):
                vt = "normal"

            views = [
                {"id": f"{rid}__A", "name": f"{name} (Group A)", "order": 0,
                 "mjpegUrl": f"{base}/api/live/mjpeg/{rid}/A"},
                {"id": f"{rid}__B", "name": f"{name} (Group B)", "order": 1,
                 "mjpegUrl": f"{base}/api/live/mjpeg/{rid}/B"},
            ]
            if start == "B":
                views = [views[1], views[0]]

            cams.append({
                "id": rid,
                "name": name,
                "groupId": rid,
                "order": 0,
                "classroomId": name,
                "location": "RTSP - Live",
                "status": "online" if enabled else "offline",
                "recording": enabled,
                "videoUrl": "",
                "filename": "",
                "is_rtsp": True,

                # ✅ NEW (frontend should trust this)
                "videoType": vt,
                "isFisheye": (vt == "fisheye"),

                "mjpegUrl": views[0]["mjpegUrl"],
                "views": views,
            })
    except Exception as e:
        _system(f"cameras: load_rtsp_sources error: {e}")

    cams.sort(key=lambda c: (str(c.get("classroomId", "")), str(c.get("groupId", ""))))
    return cams


# ---------- LIVE ----------
@app.get("/api/live/state")
def api_live_state():
    # ✅ Only compute state when user is on live/dashboard page
    st = get_ui_focus()
    if focus_alive() and st.get("page") not in ("live", "dashboard"):
        return JSONResponse({"cameras": {}, "skipped": True, "reason": "not focused"})

    cams: Dict[str, Any] = {}

    for cam_id, p in (pipelines_live or {}).items():
        # default empty
        views_payload = []
        lost_items = []

        # pull raw
        try:
            views_payload = p.pull_latest_for_api() or []  # This now includes ROI filtering
        except Exception:
            views_payload = []

        try:
            lost_items = p.pull_lost_items_for_api() or []
        except Exception:
            lost_items = []

        # ✅ sanitize to JSON-safe only
        safe_views: List[Dict[str, Any]] = []

        for v in (views_payload or []):
            if not isinstance(v, dict):
                continue

            # only keep keys that frontend needs (NO frames / NO numpy arrays)
            vv: Dict[str, Any] = {}

            # keep some meta if exists
            for k in ("view_id", "view_idx", "name", "group", "mode", "ts", "img_w", "img_h"):
                if k in v and v[k] is not None:
                    vv[k] = v[k]

            # detections - these are already ROI-filtered from pull_latest_for_api
            dets = v.get("dets") or v.get("detections") or []
            safe_dets: List[Dict[str, Any]] = []

            if isinstance(dets, list):
                for d in dets:
                    if not isinstance(d, dict):
                        continue

                    dd: Dict[str, Any] = {}

                    # label/conf
                    lab = d.get("label") or d.get("class_name") or d.get("name") or ""
                    dd["label"] = str(lab).lower().strip()

                    conf = d.get("conf")
                    try:
                        dd["conf"] = float(conf) if conf is not None else None
                    except Exception:
                        dd["conf"] = None

                    # bbox (try common formats)
                    box = d.get("bbox") or d.get("box") or d.get("xyxy") or d.get("xywh")
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        try:
                            dd["bbox"] = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                        except Exception:
                            pass
                    else:
                        keys = ("x1", "y1", "x2", "y2")
                        if all(k in d for k in keys):
                            try:
                                dd["bbox"] = [float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])]
                            except Exception:
                                pass

                    # view_id
                    try:
                        dd["view_id"] = int(d.get("view_id", vv.get("view_id", 0)) or 0)
                    except Exception:
                        dd["view_id"] = 0

                    # img_w/img_h if present
                    for k in ("img_w", "img_h"):
                        if k in d and d[k] is not None:
                            try:
                                dd[k] = int(d[k])
                            except Exception:
                                pass

                    safe_dets.append(dd)

            vv["dets"] = safe_dets
            vv["detections"] = safe_dets

            safe_views.append(vv)

        # ✅ lost_items also sanitize
        safe_lost: List[Dict[str, Any]] = []
        if isinstance(lost_items, list):
            for it in lost_items:
                if not isinstance(it, dict):
                    continue
                ii: Dict[str, Any] = {}
                for k in ("id", "item_id", "label", "status", "snapshot_path", "snapshot", "image_path", "ts"):
                    if k in it and it[k] is not None:
                        ii[k] = it[k]
                safe_lost.append(ii)

        cams[str(cam_id)] = {
            "updated_at": float(time.time()),
            "views": safe_views,
            "lost_items": safe_lost,
        }

    return JSONResponse({"cameras": cams})


@app.get("/api/live/frame/{cam_id}/{view_idx}")
def live_frame(cam_id: str, view_idx: int):
    """
    ✅ FAST PATH:
    Use VideoPipeline cached frames (RAM). No VideoCapture.
    ✅ FALLBACK:
    If pipeline not running / cache empty, use get_live_source() which returns
    RTSP URL or file path string — both work with cv2.VideoCapture and
    lf.FisheyePreprocessor.open().
    """

    cam_id = _base_id(cam_id)

    # -----------------------------
    # Decide whether FAST PATH is safe
    # IMPORTANT:
    # For normal video with empty ROI, skip FAST PATH because cached JPG may
    # already contain fake full-frame ROI from pipeline/preprocessor.
    # -----------------------------
    allow_fast_path = True
    src0 = get_live_source(cam_id)
    if src0:
        try:
            is_rtsp0 = src0.startswith("rtsp://") or src0.startswith("rtsps://")
            vtype0 = detect_video_type_cached(cam_id, Path(src0) if not is_rtsp0 else Path("."))

            if vtype0 != "fisheye":
                roi_path0 = _ensure_roi_file("live", cam_id)
                roi_config0 = {}
                if roi_path0.exists():
                    try:
                        roi_config0 = json.loads(roi_path0.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                zones0 = get_zones_for_normal_video(roi_config0) or []
                if not zones0:
                    allow_fast_path = False
        except Exception:
            pass

    # -----------------------------
    # 1) FAST PATH (pipeline cache)
    # -----------------------------
    p = pipelines_live.get(cam_id)
    if allow_fast_path and p is not None and hasattr(p, "pull_single_view_jpg"):
        try:
            jpg = p.pull_single_view_jpg(int(view_idx))
        except Exception:
            jpg = None

        if jpg:
            return Response(
                content=jpg,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
            )
    
    # -----------------------------
    # 2) FALLBACK (slow method) — supports both file path and rtsp://
    # -----------------------------
    src = get_live_source(cam_id)
    if not src:
        raise HTTPException(status_code=404, detail="live source not found")

    is_rtsp = src.startswith("rtsp://") or src.startswith("rtsps://")

    # For RTSP, allow fallback even if focus state is not perfect.
    # Dashboard often needs fallback when cache is not ready yet.
    if (not is_rtsp) and (not focus_is(page="live", active_id=cam_id)) and (not focus_is(page="dashboard", active_id=cam_id)):
        raise HTTPException(status_code=404, detail="not focused (skip slow fallback)")

    src = get_live_source(cam_id)
    if not src:
        raise HTTPException(status_code=404, detail="live source not found")

    is_rtsp = src.startswith("rtsp://") or src.startswith("rtsps://")
    vtype = detect_video_type_cached(cam_id, Path(src) if not is_rtsp else Path("."))

    # ============== NORMAL VIDEO PATH ==============
    if vtype != "fisheye":
        roi_path = _ensure_roi_file("live", cam_id)
        roi_config = {}
        if roi_path.exists():
            try:
                roi_config = json.loads(roi_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        zones = get_zones_for_normal_video(roi_config) or []

        cap = cv2.VideoCapture(src)
        try:
            fr = _read_any_frame(cap)
        finally:
            cap.release()

        if fr is None:
            raise HTTPException(status_code=404, detail="no frame")

        if zones:
           det_on = bool(getattr(p, "_detection_enabled", True)) if p is not None else True

        if overlay == 1 and det_on and zones:
            fr = draw_zones_on_view(fr, zones, color=(0, 255, 0), thickness=2, show_id=True)

        return Response(
            content=_jpg_bytes(fr, quality=85),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    # ===============================================

    # ✅ FISHEYE PATH — build/reuse lf.FisheyePreprocessor with RTSP URL or file path
    try:
        pre = _get_or_make_live_fisheye_pre(cam_id, src)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fisheye preprocessor error: {e}")

    cap = cv2.VideoCapture(src)
    try:
        fr = _read_any_frame(cap)
    finally:
        cap.release()

    if fr is None:
        raise HTTPException(status_code=404, detail="no frame")

    wanted_names = _fisheye_order_names()
    vi = int(view_idx)
    if vi < 0:
        vi = 0
    if vi >= len(wanted_names):
        vi = 0
    name = wanted_names[vi]

    try:
        views = pre.get_views(fr, allowed_names=[name])
    except TypeError:
        views = pre.get_views(fr)

    chosen = None
    for v in (views or []):
        if (v.get("name") or "") == name:
            chosen = v
            break

    if chosen is None:
        raise HTTPException(status_code=404, detail=f"fisheye view missing: {name}")

    img = chosen.get("image") if chosen else None
    if img is None:
        raise HTTPException(status_code=404, detail=f"fisheye view image missing: {name}")

    return Response(
        content=_jpg_bytes(img, quality=85),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/live/group_frame/{cam_id}/{group}")
def live_group_frame(cam_id: str, group: str):
    """
    ✅ FAST PATH:
    Use VideoPipeline cached group grid JPG (RAM). No VideoCapture.
    ✅ FALLBACK:
    If cache empty, use get_live_source() — works for both file and rtsp://.
    """

    cam_id = _base_id(cam_id)

    # -----------------------------
    # Decide whether FAST PATH is safe
    # IMPORTANT:
    # For normal video with empty ROI, skip FAST PATH because cached JPG may
    # already contain fake full-frame ROI from pipeline/preprocessor.
    # -----------------------------
    allow_fast_path = True
    src0 = get_live_source(cam_id)
    if src0:
        try:
            is_rtsp0 = src0.startswith("rtsp://") or src0.startswith("rtsps://")
            vtype0 = detect_video_type_cached(cam_id, Path(src0) if not is_rtsp0 else Path("."))

            if vtype0 != "fisheye":
                roi_path0 = _ensure_roi_file("live", cam_id)
                roi_config0 = {}
                if roi_path0.exists():
                    try:
                        roi_config0 = json.loads(roi_path0.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                zones0 = get_zones_for_normal_video(roi_config0) or []
                if not zones0:
                    allow_fast_path = False
        except Exception:
            pass

    # -----------------------------
    # 1) FAST PATH (pipeline cache)
    # -----------------------------
    p = pipelines_live.get(cam_id)
    if allow_fast_path and p is not None and hasattr(p, "pull_group_grid_jpg"):
        try:
            jpg = p.pull_group_grid_jpg(group)
            if jpg:
                return Response(
                    content=jpg,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
                )
        except Exception:
            pass

    # -----------------------------
    # 2) FALLBACK — supports file and rtsp://
    # -----------------------------
    if not focus_is(page="live", active_id=cam_id) and not focus_is(page="dashboard", active_id=cam_id):
        raise HTTPException(status_code=404, detail="not focused (skip slow fallback)")

    src = get_live_source(cam_id)
    if not src:
        raise HTTPException(status_code=404, detail="live source not found")

    is_rtsp = src.startswith("rtsp://") or src.startswith("rtsps://")
    vtype = detect_video_type_cached(cam_id, Path(src) if not is_rtsp else Path("."))

    # ============== NORMAL VIDEO PATH ==============
    if vtype != "fisheye":
        roi_path = _ensure_roi_file("live", cam_id)
        roi_config = {}
        if roi_path.exists():
            try:
                roi_config = json.loads(roi_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        zones = get_zones_for_normal_video(roi_config) or []

        cap = cv2.VideoCapture(src)
        try:
            fr = _read_any_frame(cap)
        finally:
            cap.release()

        if fr is None:
            raise HTTPException(status_code=404, detail="no frame")

        det_on = bool(getattr(p, "_detection_enabled", True)) if p is not None else True

        if overlay == 1 and det_on and zones:
            fr = draw_zones_on_view(fr, zones, color=(0, 255, 0), thickness=2, show_id=True)

        return Response(
            content=_jpg_bytes(fr, quality=85),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    # ===============================================

    try:
        pre = _get_or_make_live_fisheye_pre(cam_id, src)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fisheye preprocessor error: {e}")

    cap = cv2.VideoCapture(src)
    try:
        fr = _read_any_frame(cap)
    finally:
        cap.release()

    if fr is None:
        raise HTTPException(status_code=404, detail="no frame")

    g = (group or "A").upper().strip()
    if g not in ("A", "B"):
        g = "A"

    group_names = _group_names(g)

    try:
        group_views = pre.get_views(fr, allowed_names=group_names)
    except TypeError:
        group_views = pre.get_views(fr)

    name_to_img: Dict[str, np.ndarray] = {}
    for v in (group_views or []):
        nm = v.get("name", "")
        im = v.get("image")
        if nm and im is not None:
            name_to_img[nm] = im

    roi_path = _ensure_roi_file("live", cam_id)
    roi_config = {}
    if roi_path.exists():
        try:
            roi_config = json.loads(roi_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    tiles: List[Optional[np.ndarray]] = [None, None, None, None]

    for i, view_name in enumerate(group_names[:4]):
        bgr = name_to_img.get(view_name)
        if bgr is None:
            continue

        bgr = bgr.copy()

        fisheye_polys = roi_config.get("fisheye_polygons", {})
        view_polys = fisheye_polys.get(view_name, [])
        for poly in view_polys:
            if isinstance(poly, list) and len(poly) >= 3:
                try:
                    pts = np.array([[p["x"], p["y"]] for p in poly], dtype=np.int32)
                    cv2.polylines(bgr, [pts], True, (0, 255, 0), 2)
                except Exception:
                    pass

        bgr = _draw_label_bar(bgr, f"{g} | {view_name.replace('_', ' ').title()}")
        tiles[i] = bgr

    if all(t is None for t in tiles):
        blank = make_2x2_grid(
            [_blank_bgr(), _blank_bgr(), _blank_bgr(), _blank_bgr()],
            480, 640
        )
        return Response(
            content=_jpg_bytes(blank, quality=85),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    valid_tiles = [t for t in tiles if isinstance(t, np.ndarray)]
    if not valid_tiles:
        blank = make_2x2_grid([_blank_bgr(), _blank_bgr(), _blank_bgr(), _blank_bgr()], 640, 480)
        return _jpg_bytes(blank, quality=85)

    cell_h = min(t.shape[0] for t in valid_tiles)
    cell_w = min(t.shape[1] for t in valid_tiles)

    grid = make_2x2_grid(tiles, cell_h, cell_w)
    return _jpg_bytes(grid, quality=85)


@app.get("/api/live/roi/{cam_id}")
def get_live_roi(cam_id: str):
    cam_id_raw = cam_id
    cam_id_used = _live_id(cam_id)
    roi_path = _ensure_roi_file("live", cam_id_used)

    try:
        roi = json.loads(roi_path.read_text(encoding="utf-8"))
    except Exception:
        roi = {"bounding_polygons": [], "fisheye_polygons": {}}

    print(f"[SYSTEM] GET ROI raw cam_id={cam_id_raw}")
    print(f"[SYSTEM] GET ROI used cam_id={cam_id_used}")
    print(f"[SYSTEM] GET ROI path={roi_path}")
    print(f"[SYSTEM] GET ROI json={json.dumps(roi)[:2000]}")

    return {
        **roi,
        "roi_empty": _roi_is_empty(roi),
    }


@app.post("/api/live/roi/{cam_id}")
def save_live_roi(cam_id: str, payload: Dict[str, Any] = Body(...)):
    cam_id_raw = cam_id
    cam_id_used = _live_id(cam_id)
    roi_path = _ensure_roi_file("live", cam_id_used)

    out = _normalize_roi_payload_to_original(cam_id_used, payload, mode="live")

    # ✅ use SAME resolver as static preview
    vtype = "normal"
    src_str = None
    try:
        src_str = resolve_live_source(cam_id_used)
        if src_str:
            vtype = resolve_settings_video_type(cam_id_used, str(src_str))
    except Exception as e:
        print(f"[SYSTEM] save_live_roi unified detect error: {e}")

    if vtype == "fisheye":
        fis = out.get("fisheye_polygons", {}) or {}
        cleaned = {}
        for name in _fisheye_order_names():
            val = fis.get(name, [])
            cleaned[name] = val if isinstance(val, list) else []
        out = {
            "bounding_polygons": [],
            "fisheye_polygons": cleaned,
        }
    else:
        out = {
            "bounding_polygons": out.get("bounding_polygons", []) or [],
            "fisheye_polygons": {},
        }

    print("=" * 80)
    print(f"[SYSTEM] SAVE ROI raw cam_id={cam_id_raw}")
    print(f"[SYSTEM] SAVE ROI used cam_id={cam_id_used}")
    print(f"[SYSTEM] SAVE ROI path={roi_path}")
    print(f"[SYSTEM] SAVE ROI src={src_str}")
    print(f"[SYSTEM] SAVE ROI vtype={vtype}")
    print(f"[SYSTEM] SAVE ROI payload keys={list((payload or {}).keys())}")
    print(
        f"[SYSTEM] SAVE ROI payload fisheye keys={list(((payload or {}).get('fisheye_polygons') or {}).keys()) if isinstance((payload or {}).get('fisheye_polygons'), dict) else 'NOT_DICT'}")
    print(f"[SYSTEM] SAVE ROI out fisheye keys={list((out.get('fisheye_polygons') or {}).keys())}")
    print(f"[SYSTEM] SAVE ROI out json={json.dumps(out)[:2000]}")
    print("=" * 80)

    roi_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    try:
        saved_back = json.loads(roi_path.read_text(encoding="utf-8"))
        print(f"[SYSTEM] SAVE ROI readback path={roi_path}")
        print(f"[SYSTEM] SAVE ROI readback json={json.dumps(saved_back)[:2000]}")
    except Exception as e:
        print(f"[SYSTEM] SAVE ROI readback error={e}")

    p = pipelines_live.get(cam_id_used)
    print(f"[SYSTEM] SAVE ROI pipeline found={p is not None} for cam_id={cam_id_used}")

    if p is not None:
        try:
            setattr(p, "_roi_dirty", True)
        except Exception as e:
            print(f"[SYSTEM] SAVE ROI set dirty error={e}")
        if hasattr(p, "reload_roi"):
            try:
                p.reload_roi()
                print(f"[UI] {cam_id_used} ROI reloaded")
            except Exception as e:
                print(f"[SYSTEM] SAVE ROI reload error={e}")

    return {
        "ok": True,
        "cam_id_raw": cam_id_raw,
        "cam_id_used": cam_id_used,
        "roi_path": str(roi_path),
        "vtype": vtype,
        "roi_empty": _roi_is_empty(out),
    }


@app.get("/api/live/status")
def api_live_status():
    """Get status of all live cameras (which ones have ended)"""
    status = {}

    for cam_id, pipeline in pipelines_live.items():
        cam_status = {
            "cam_id": cam_id,
            "is_running": True,
            "video_ended": False,
            "restart_pending": cam_id in _live_video_restart_pending,
            "last_ended": _live_video_ended.get(cam_id, 0),
            "cooldown_remaining": 0
        }

        # Calculate cooldown remaining if applicable
        if cam_id in _live_video_ended:
            elapsed = time.time() - _live_video_ended[cam_id]
            if elapsed < 5:  # 5 second cooldown
                cam_status["cooldown_remaining"] = 5 - elapsed

        try:
            if hasattr(pipeline, 'eof_reached'):
                cam_status["video_ended"] = pipeline.eof_reached

            if hasattr(pipeline, 'frame_count'):
                cam_status["frame_count"] = pipeline.frame_count

            if hasattr(pipeline, 'total_frames'):
                cam_status["total_frames"] = pipeline.total_frames
                if pipeline.total_frames > 0:
                    cam_status["progress_percent"] = (pipeline.frame_count / pipeline.total_frames) * 100

        except Exception:
            pass

        status[cam_id] = cam_status

    return {"cameras": status, "timestamp": time.time()}


@app.post("/api/live/restart")
def restart_live(target_cam_id: str = None):
    """
    Restart live pipelines.
    - If target_cam_id provided: restart only that specific camera
    - If no target_cam_id: restart ALL cameras (original behavior)
    """
    global pipelines_live, pipelines_settings

    if target_cam_id:
        # Restart single camera
        target_cam_id = _base_id(target_cam_id)
        _system(f"LIVE: Manual restart requested for camera {target_cam_id}")

        # Check if camera exists
        if target_cam_id not in pipelines_live:
            raise HTTPException(status_code=404, detail=f"Camera {target_cam_id} not found")

        # Use your new restart_single_live_camera function
        success = restart_single_live_camera(target_cam_id)

        if success:
            return {
                "ok": True,
                "message": f"Camera {target_cam_id} restarted",
                "restarted_cam": target_cam_id,
                "live_cams": sorted(list(pipelines_live.keys()))
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to restart camera {target_cam_id}")

    else:
        # Original behavior: restart ALL cameras
        _system("LIVE: Manual restart requested for ALL cameras")

        # Stop ALL LIVE pipelines
        for _, p in list(pipelines_live.items()):
            try:
                if hasattr(p, "stop"):
                    p.stop()
            except Exception:
                pass
        pipelines_live = {}

        # Stop SETTINGS pipelines
        for _, p in list(pipelines_settings.items()):
            try:
                if hasattr(p, "stop"):
                    p.stop()
            except Exception:
                pass
        pipelines_settings = {}

        # Start fresh
        start_live_pipelines(limit_normal=8, limit_fisheye=4)

        # Reset restart counters for all cameras
        with _live_video_monitor_lock:
            _live_video_ended.clear()

        with _live_video_restart_lock:
            _live_video_restart_pending.clear()

        _system(f"LIVE: All cameras restarted. Active: {sorted(list(pipelines_live.keys()))}")

        return {
            "ok": True,
            "message": "All cameras restarted",
            "live_cams": sorted(list(pipelines_live.keys()))
        }


# ---------- OFFLINE frame (ROI preview) ----------
@app.get("/api/offline/frame/{stem}/{view_idx}")
def offline_frame(stem: str, view_idx: int):
    """
    ✅ OFFLINE single-view ROI preview
    - Normal video: return ORIGINAL frame size (do NOT resize)
    - Fisheye: return 640x480 dewarped view
    """
    # ✅ only build offline frame when user is on offline page for this stem
    stem = (stem or "").strip()
    if focus_alive() and not focus_is(page="offline", active_id=stem):
        raise HTTPException(status_code=404, detail="not focused (skip offline frame)")
    lock = OFFLINE_LOCKS[stem]
    with lock:
        h264_path = _offline_h264_path(stem)
        if not h264_path.exists() or h264_path.stat().st_size < 256 * 1024:
            raise HTTPException(status_code=404, detail="h264 not ready")

        vtype = detect_video_type_cached(stem, h264_path)

        img = _offline_build_view_image(stem, int(view_idx))
        if img is None or (hasattr(img, "size") and img.size == 0):
            raise HTTPException(status_code=404, detail="no offline view frame")

        # ✅ Only force 640x480 for fisheye
        if vtype == "fisheye":
            img = _force_640x480(img)

        return Response(
            content=_jpg_bytes(img, quality=85),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )


@app.get("/api/offline/group_frame/{stem}/{group}")
def offline_group_frame(stem: str, group: str):
    """
    ✅ OFFLINE fisheye group preview (2x2 grid)
    Uses _offline_build_view_image(stem, view_idx)
    """
    stem = (stem or "").strip()
    if focus_alive() and not focus_is(page="offline", active_id=stem):
        raise HTTPException(status_code=404, detail="not focused (skip offline grid)")
    lock = OFFLINE_LOCKS[stem]
    with lock:
        h264_path = _offline_h264_path(stem)
        vtype = detect_video_type_cached(stem, h264_path)

        # Normal video -> return a single frame
        if vtype != "fisheye":
            img = _offline_build_view_image(stem, 0)
            return Response(
                content=_jpg_bytes(img, quality=85),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
            )

        g = (group or "A").upper()
        if g not in ("A", "B"):
            g = "A"

        idxs = _group_view_indices(g)
        names = _group_names(g)

        tiles: List[Optional[np.ndarray]] = [None, None, None, None]
        for i, vi in enumerate(idxs):
            bgr = _offline_build_view_image(stem, int(vi))
            label = names[i] if i < len(names) else f"view_{vi}"
            bgr = _draw_label_bar(bgr, f"{g} | {label}")
            tiles[i] = bgr

        grid = make_2x2_grid(tiles, 480, 640)

    return Response(
        content=_jpg_bytes(grid, quality=85),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


# ---------- OFFLINE list (from manifest) ----------
@app.get("/api/offline/videos")
def offline_list_videos(request: Request):
    base = str(request.base_url).rstrip("/")

    if focus_alive() and focus_is(page="offline"):
        reconcile_manifest_with_offline_folder()

    items = _manifest_all()

    with _offline_jobs_lock:
        jobs_snapshot = {k: dict(v) for k, v in _offline_jobs.items()}

    for it in items:
        stem = it["id"]
        it["originalUrl"] = f"{base}/offline/{it.get('name')}"

        h264_path = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"
        h264_ready = bool(h264_path.exists())
        it["h264_ready"] = h264_ready

        vtype = "normal"
        views_count = 1

        if h264_ready:
            vtype = detect_video_type_cached(stem, h264_path)
            if vtype == "fisheye":
                views_count = 8
            it["duration"] = it.get("duration") or _video_duration_hms(h264_path)

        it["is_fisheye"] = (vtype == "fisheye")
        it["views_count"] = views_count
        it["h264Url"] = f"{base}/offline/{stem}_h264.mp4" if h264_ready else None

        # ONLY real analysis job status goes here
        job = jobs_snapshot.get(stem) or {}
        job_status = job.get("status")

        if job_status in ("queued", "processing", "done", "failed"):
            it["analysis_status"] = job_status
            it["analysis_error"] = job.get("error")
            it["is_analyzing"] = job_status in ("queued", "processing")
        else:
            it["analysis_status"] = None
            it["analysis_error"] = None
            it["is_analyzing"] = False

    return {"videos": items}

# ---------- OFFLINE upload ----------
@app.post("/api/offline/upload")
async def offline_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(status_code=400, detail="Unsupported format")

    tmp_path = OFFLINE_UPLOAD_DIR / f"__upload_tmp__{int(time.time() * 1000)}{ext}"
    orig_path = OFFLINE_UPLOAD_DIR / Path(file.filename).name
    stem = _safe_stem(file.filename)

    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        if orig_path.exists():
            orig_path.unlink(missing_ok=True)
        tmp_path.rename(orig_path)

        # ✅ ensure manifest matches folder immediately
        reconcile_manifest_with_offline_folder()

        _manifest_upsert(
            stem,
            {
                "name": orig_path.name,
                "size": _human_mb(orig_path.stat().st_size),
                "uploadDate": int(orig_path.stat().st_mtime),
                "status": "processing",
                "h264_ready": False,
                "h264_name": f"{stem}_h264.mp4",
                "roi_ready": False,
                "error": None,
            },
        )

        threading.Thread(
            target=_convert_and_promote_worker,
            args=(stem, orig_path, OFFLINE_UPLOAD_DIR),
            daemon=True,
        ).start()

        return {"ok": True, "id": stem, "status": "processing"}

    except Exception as e:
        _manifest_upsert(stem, {"status": "failed", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------- OFFLINE ROI (ROI first, then analyze) ----------
@app.get("/api/offline/roi/{stem}")
def offline_get_roi(stem: str):
    roi_path = _ensure_roi_file("offline", stem)
    try:
        return json.loads(roi_path.read_text(encoding="utf-8"))
    except Exception:
        return {"bounding_polygons": [], "fisheye_polygons": {}}


@app.post("/api/offline/roi/{stem}")
def offline_save_roi(stem: str, payload: Dict[str, Any] = Body(...)):
    """
    Save offline ROI.
    IMPORTANT:
    - If frontend clears ROI, backend must overwrite roi_config.json with empty ROI.
    - Do NOT merge with existing ROI, otherwise cleared ROI will come back.
    """
    roi_path = _ensure_roi_file("offline", stem)

    # Detect video type
    h264_path = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"
    vtype = "normal"
    if h264_path.exists():
        vtype = detect_video_type_cached(stem, h264_path)

    # Normalize incoming payload
    out = _normalize_roi_payload_to_original(stem, payload, mode="offline")

    # Force clean structure by video type
    if vtype == "fisheye":
        cleaned = {}
        for name in _fisheye_order_names():
            val = (out.get("fisheye_polygons", {}) or {}).get(name, [])
            cleaned[name] = val if isinstance(val, list) else []

        out = {
            "bounding_polygons": [],
            "fisheye_polygons": cleaned,
        }
    else:
        out = {
            "bounding_polygons": out.get("bounding_polygons", []) or [],
            "fisheye_polygons": {},
        }

    # OVERWRITE directly
    roi_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    roi_ready = (not _roi_is_empty(out))
    _manifest_upsert(stem, {"roi_ready": roi_ready})

    return {
        "ok": True,
        "id": stem,
        "roi_ready": roi_ready,
        "vtype": vtype,
    }

# ---------- OFFLINE analyze (requires ROI not empty) ----------
@app.post("/api/offline/analyze")
def offline_analyze(payload: Dict[str, Any] = Body(...)):
    stem = (payload.get("id") or "").strip()
    if not stem:
        raise HTTPException(status_code=400, detail="Missing id")

    roi_path = _ensure_roi_file("offline", stem)
    try:
        roi = json.loads(roi_path.read_text(encoding="utf-8"))
    except Exception:
        roi = {"bounding_polygons": [], "fisheye_polygons": {}}

    if _roi_is_empty(roi):
        raise HTTPException(status_code=400, detail="ROI is empty. Draw ROI first, then Analyze.")

    h264_path = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"

    with _offline_jobs_lock:
        st = (_offline_jobs.get(stem) or {}).get("status")
        if st in ("queued", "processing"):
            return {"ok": True, "started": False, "status": st}

    # ✅ h264 ready -> start now
    if h264_path.exists() and h264_path.stat().st_size > 256 * 1024:
        with _offline_jobs_lock:
            _offline_jobs[stem] = {
                "status": "processing",
                "error": None,
                "started_at": time.time(),
                "ended_at": None,
            }

        th = threading.Thread(target=_run_lostfound_subprocess, args=(stem, h264_path), daemon=True)
        th.start()
        return {"ok": True, "started": True, "status": "processing"}

    # ✅ h264 not ready -> queue
    _manifest_upsert(stem, {"status": "queued", "error": None})
    with _offline_jobs_lock:
        _offline_jobs[stem] = {
            "status": "queued",
            "error": None,
            "started_at": time.time(),
            "ended_at": None,
        }

    queue_analyze_if_needed(stem)
    return {"ok": True, "started": False, "status": "queued", "detail": "h264 not ready; analysis queued"}


def _safe_unlink_with_retry(path: Path, retries: int = 8, delay: float = 0.4) -> None:
    if not path.exists():
        return

    last_err = None
    for _ in range(retries):
        try:
            path.unlink(missing_ok=True)
            if not path.exists():
                return
        except PermissionError as e:
            last_err = e
        except OSError as e:
            last_err = e

        time.sleep(delay)

    if path.exists():
        raise last_err or PermissionError(f"File still in use: {path}")


def _safe_rmtree_with_retry(path: Path, retries: int = 8, delay: float = 0.4) -> None:
    if not path.exists():
        return

    last_err = None
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            if not path.exists():
                return
        except PermissionError as e:
            last_err = e
        except OSError as e:
            last_err = e

        time.sleep(delay)

    if path.exists():
        raise last_err or PermissionError(f"Folder still in use: {path}")


@app.delete("/api/offline/video/{stem}")
def offline_delete_video(stem: str):
    stem = (stem or "").strip()
    if not stem:
        raise HTTPException(status_code=400, detail="Missing stem")

    h264 = OFFLINE_UPLOAD_DIR / f"{stem}_h264.mp4"
    out_dir = _outputs_dir("offline", stem)

    # find original uploaded file (may be mp4/avi/mov/mkv)
    orig_found: Optional[Path] = None
    for f in OFFLINE_UPLOAD_DIR.iterdir():
        if _is_original_upload_file(f) and f.stem == stem:
            orig_found = f
            break

    try:
        # -------------------------------------------------
        # 1) clear queued / running flags
        # -------------------------------------------------
        with _offline_jobs_lock:
            _offline_jobs.pop(stem, None)

        with _analyze_queue_lock:
            _analyze_queued.discard(stem)

        # -------------------------------------------------
        # 2) clear cached frame
        # -------------------------------------------------
        with _offline_frame_cache_lock:
            _offline_frame_cache.pop(stem, None)

        # -------------------------------------------------
        # 3) clear cached fisheye preprocessor
        # -------------------------------------------------
        with _offline_pre_lock:
            rec = _offline_pre_cache.pop(stem, None)
            try:
                pre = rec.get("pre") if isinstance(rec, dict) else None
                if pre is not None and hasattr(pre, "release"):
                    pre.release()
            except Exception:
                pass

        # -------------------------------------------------
        # 4) invalidate video-type cache
        # -------------------------------------------------
        invalidate_video_type_cache(stem)

        # small wait for Windows file handles to release
        time.sleep(0.3)

        # -------------------------------------------------
        # 5) delete original uploaded file
        # -------------------------------------------------
        if orig_found and orig_found.exists():
            _safe_unlink_with_retry(orig_found)

        # -------------------------------------------------
        # 6) delete promoted h264 file
        # -------------------------------------------------
        if h264.exists():
            _safe_unlink_with_retry(h264)

        # -------------------------------------------------
        # 7) delete offline output folder
        # -------------------------------------------------
        if out_dir.exists():
            _safe_rmtree_with_retry(out_dir)

        # -------------------------------------------------
        # 8) remove from manifest
        # -------------------------------------------------
        m = _read_manifest()
        vids = m.get("videos") or {}
        vids.pop(stem, None)
        m["videos"] = vids
        _write_manifest(m)

        return {
            "ok": True,
            "deleted": stem,
            "deleted_original": bool(orig_found),
            "deleted_h264": True,
            "deleted_output_dir": True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    
# ============================================================
# Event Page Processing
# ============================================================
def to_image_url(image_path: str, request_base: str) -> Optional[str]:
    if not image_path:
        return None

    try:
        p = Path(str(image_path))
    except Exception:
        return None

    # If analyzer stored relative paths like:
    # "offline/<stem>/snapshots/xxx.jpg" or "live/<cam>/snapshots/yyy.jpg"
    if not p.is_absolute():
        # Prefer OUTPUTS_LF_DIR because your offline/live snapshots live there
        p = (OUTPUTS_LF_DIR / p).resolve()

    # 1) inside backend outputs -> /lf_outputs/...
    for root in (OUTPUTS_LF_DIR, PROJECT_OUTPUTS_LF_DIR):
        try:
            rel = p.relative_to(root).as_posix()
            return f"{request_base}/lf_outputs/{rel}"
        except Exception:
            pass

    # 2) inside project snapshots -> /snapshots/...
    try:
        # SNAPSHOT_DIR is mounted at /snapshots
        rel2 = p.relative_to(SNAPSHOT_DIR).as_posix()
        return f"{request_base}/snapshots/{rel2}"
    except Exception:
        pass

    return None


def extract_location_from_stem(stem: str) -> str:
    """
    Example stem:
    B100AA_B_Block_B_Block_20251110080959_20251110084059_39454622

    Return:
    B100AA_B_Block_B
    """
    if not stem:
        return "Unknown"

    parts = stem.split("_")

    # We want everything up to the FIRST 'Block_<X>'
    out = []
    for i in range(len(parts) - 1):
        out.append(parts[i])
        if parts[i].lower() == "block":
            # include the block letter
            out.append(parts[i + 1])
            break

    loc = "_".join(out)
    # Safety cleanup
    loc = loc.replace("__", "_").strip("_")

    return loc or "Unknown"


def _ts_from_snapshot_name(path_str: str) -> int:
    # find ..._20260207_170550_...
    m = re.search(r'_(\d{8})_(\d{6})_', path_str)
    if not m:
        return 0
    ymd, hms = m.group(1), m.group(2)
    dt = datetime.datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
    return int(dt.timestamp())


def _pick_label(obj: dict) -> str:
    for k in ("label", "class_name", "name", "cls"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def _pick_status(obj: dict) -> str:
    # normalize to: lost | solved
    s = str(obj.get("status") or obj.get("state") or "").lower().strip()
    if "solve" in s or s == "resolved":
        return "solved"
    return "lost"

def _resolve_item_location(source_id: str) -> str:
    source_id = str(source_id or "").strip()
    if not source_id:
        return "Unknown"

    if source_id.startswith("rtsp_"):
        try:
            rtsp = load_rtsp_sources() or {}
        except Exception:
            rtsp = {}

        rec = rtsp.get(source_id)
        if isinstance(rec, dict):
            name = str(rec.get("name") or "").strip()
            if name:
                return name

        return source_id

    return extract_location_from_stem(source_id)

def _normalize_live_item(cam_id: str, it: dict, request_base: str) -> dict:
    it = it or {}
    raw = it.get("raw") if isinstance(it.get("raw"), dict) else it
    raw = dict(raw or {})  # editable copy

    snap = it.get("snapshot_path") or it.get("snapshot") or it.get("image_path")
    snap_str = str(snap or "").strip()

    resolved_name = _resolve_item_location(cam_id)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(resolved_name or cam_id)).strip("_")

    raw_id = str(
        raw.get("lost_id")
        or it.get("id")
        or it.get("item_id")
        or ""
    ).strip()

    # remove default prefix like B001GB_, keep ROI1_...
    if "_ROI" in raw_id:
        raw_id = "ROI" + raw_id.split("_ROI", 1)[1]

    # write cleaned value back into raw too
    raw["lost_id"] = raw_id

    cam_id_clean = str(cam_id or "").strip()

    if raw_id:
        item_id = f"live-{cam_id_clean}_{safe_name}_{raw_id}"
    elif snap_str:
        snap_name = Path(snap_str).name
        item_id = f"live-{cam_id_clean}_{safe_name}_{snap_name}"
    else:
        item_id = f"live-{cam_id_clean}_{safe_name}_{int(time.time() * 1000)}"

    image_url = to_image_url(snap, request_base)

    try:
        lost_time = float(raw.get("lost_time")) if raw.get("lost_time") is not None else None
    except Exception:
        lost_time = None

    try:
        duration_before_lost = float(raw.get("duration_before_lost") or 0)
    except Exception:
        duration_before_lost = 0.0

    snap_ts = _ts_from_snapshot_name(snap_str) if snap_str else 0
    now_ts = int(time.time())

    # real wall-clock event time
    if snap_ts > 0:
        last_seen_ts = int(snap_ts)
    else:
        last_seen_ts = now_ts

    duration_sec = max(0.0, duration_before_lost)
    first_seen_ts = int(last_seen_ts - duration_sec) if duration_sec > 0 else last_seen_ts

    if first_seen_ts < 0 or first_seen_ts > last_seen_ts:
        first_seen_ts = last_seen_ts

    location = resolved_name

    return {
        "id": item_id,
        "module": "lost_found",
        "source": "live",
        "cameraId": cam_id,
        "location": location,
        "label": _pick_label(it).replace("_", " ").title(),
        "status": _pick_status(it),
        "firstSeenTs": first_seen_ts,
        "lastSeenTs": last_seen_ts,
        "imageUrl": image_url,
        "raw": raw,
    }

def _normalize_offline_item(stem: str, it: dict, request_base: str) -> dict:
    it = it or {}
    raw = it.get("raw") if isinstance(it.get("raw"), dict) else it
    raw = dict(raw or {})  # editable copy

    snap = it.get("snapshot_path") or it.get("snapshot") or it.get("image_path")
    snap_str = str(snap or "").strip()

    resolved_name = _resolve_item_location(stem)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(resolved_name or stem)).strip("_")

    lost_id = raw.get("lost_id") or it.get("lost_id") or it.get("id") or it.get("item_id")
    lost_id = str(lost_id or "").strip()

    # remove default prefix like B001GB_, keep ROI1_...
    if "_ROI" in lost_id:
        lost_id = "ROI" + lost_id.split("_ROI", 1)[1]

    # write cleaned value back into raw too
    raw["lost_id"] = lost_id

    stem_clean = str(stem or "").strip()

    if lost_id:
        item_id = f"offline-{stem_clean}_{safe_name}_{lost_id}"
    elif snap_str:
        item_id = f"offline-{stem_clean}_{safe_name}_{Path(snap_str).name}"
    else:
        item_id = f"offline-{stem_clean}_{safe_name}_{int(time.time() * 1000)}"

    image_url = to_image_url(snap, request_base)

    try:
        lost_time = float(raw.get("lost_time")) if raw.get("lost_time") is not None else None
    except Exception:
        lost_time = None

    try:
        duration_before_lost = float(raw.get("duration_before_lost") or 0)
    except Exception:
        duration_before_lost = 0.0

    snap_ts = _ts_from_snapshot_name(snap_str) if snap_str else 0
    now_ts = int(time.time())

    # real wall-clock event time
    if snap_ts > 0:
        last_seen_ts = int(snap_ts)
    else:
        last_seen_ts = now_ts

    duration_sec = max(0.0, duration_before_lost)
    first_seen_ts = int(last_seen_ts - duration_sec) if duration_sec > 0 else last_seen_ts

    if first_seen_ts < 0 or first_seen_ts > last_seen_ts:
        first_seen_ts = last_seen_ts

    # safety
    if first_seen_ts < 0 or first_seen_ts > last_seen_ts:
        first_seen_ts = last_seen_ts

    return {
        "id": item_id,
        "module": "lost_found",
        "source": "upload",
        "videoId": stem,
        "location": resolved_name,
        "label": _pick_label(it).replace("_", " ").title(),
        "status": _pick_status(it),
        "firstSeenTs": first_seen_ts,
        "lastSeenTs": last_seen_ts,
        "imageUrl": image_url,
        "raw": raw,
    }


def _read_json_file(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"  # same as video_pipeline.py default
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")

# ============================================================
# Lost & Found UI Overrides (status/notes/delete) - persistent
# ============================================================
_overrides_lock = threading.Lock()
_items_store_lock = threading.Lock()

LF_STORE_ITEMS_DIR = OUTPUTS_LF_DIR / "_items_store"
LF_STORE_OVERRIDES_DIR = OUTPUTS_LF_DIR / "_overrides_store"

LF_STORE_ITEMS_DIR.mkdir(parents=True, exist_ok=True)
LF_STORE_OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)

LF_STORE_ITEMS_PREFIX = "items_"
LF_STORE_OVERRIDES_PREFIX = "overrides_"
LF_STORE_SUFFIX = ".json"

LF_ITEMS_PER_FILE = 300
LF_OVERRIDES_PER_FILE = 300

def _lf_shard_path(root: Path, prefix: str, index: int) -> Path:
    return root / f"{prefix}{index:06d}{LF_STORE_SUFFIX}"

def _lf_list_shard_paths(root: Path, prefix: str) -> List[Path]:
    return sorted(root.glob(f"{prefix}*{LF_STORE_SUFFIX}"))

def _lf_load_shard(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _lf_save_shard(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception:
        pass

def _lf_load_all_shards(root: Path, prefix: str) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for p in _lf_list_shard_paths(root, prefix):
        shard = _lf_load_shard(p)
        if isinstance(shard, dict):
            merged.update(shard)
    return merged

def _lf_rewrite_all_shards(
    root: Path,
    prefix: str,
    data: Dict[str, Any],
    per_file: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    for p in _lf_list_shard_paths(root, prefix):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    items = list((data or {}).items())
    if not items:
        _lf_save_shard(_lf_shard_path(root, prefix, 1), {})
        return

    shard_idx = 1
    for i in range(0, len(items), per_file):
        chunk = dict(items[i:i + per_file])
        _lf_save_shard(_lf_shard_path(root, prefix, shard_idx), chunk)
        shard_idx += 1

def _read_overrides() -> Dict[str, Dict[str, Any]]:
    with _overrides_lock:
        data = _lf_load_all_shards(LF_STORE_OVERRIDES_DIR, LF_STORE_OVERRIDES_PREFIX)
        return data if isinstance(data, dict) else {}

def _write_overrides(data: Dict[str, Dict[str, Any]]) -> None:
    with _overrides_lock:
        clean = data if isinstance(data, dict) else {}
        _lf_rewrite_all_shards(
            LF_STORE_OVERRIDES_DIR,
            LF_STORE_OVERRIDES_PREFIX,
            clean,
            LF_OVERRIDES_PER_FILE,
        )

def _read_items_store() -> Dict[str, Dict[str, Any]]:
    with _items_store_lock:
        data = _lf_load_all_shards(LF_STORE_ITEMS_DIR, LF_STORE_ITEMS_PREFIX)
        return data if isinstance(data, dict) else {}

def _write_items_store(data: Dict[str, Dict[str, Any]]) -> None:
    with _items_store_lock:
        clean = data if isinstance(data, dict) else {}
        _lf_rewrite_all_shards(
            LF_STORE_ITEMS_DIR,
            LF_STORE_ITEMS_PREFIX,
            clean,
            LF_ITEMS_PER_FILE,
        )

def _apply_override(item: Dict[str, Any], overrides: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return item

    item_id = str(item.get("id") or "").strip()
    if not item_id:
        return item

    ov = overrides.get(item_id)
    if not isinstance(ov, dict):
        return item

    out = dict(item)

    if isinstance(ov.get("status"), str) and ov["status"] in ("lost", "solved"):
        out["status"] = ov["status"]

    if isinstance(ov.get("notes"), str):
        out["notes"] = ov["notes"]

    if isinstance(ov.get("updated_at"), (int, float)):
        out["updatedAt"] = int(ov["updated_at"])

    if isinstance(ov.get("deleted"), bool):
        out["deleted"] = ov["deleted"]

    if not out.get("snapshot") and isinstance(ov.get("snapshot"), str):
        out["snapshot"] = ov["snapshot"]

    return out

def _should_skip_lf_event_item(item: Dict[str, Any]) -> bool:
    """
    Reject broken / ghost Lost & Found items so they are not stored
    and not shown on the Events page.
    """
    if not isinstance(item, dict):
        return True

    # deleted items should never appear again
    if item.get("deleted") is True:
        return True

    # safe timestamp parsing
    def _to_num(v) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    first_seen = _to_num(item.get("firstSeenTs"))
    last_seen = _to_num(item.get("lastSeenTs"))

    snap = (
        item.get("snapshot")
        or item.get("snapshot_path")
        or item.get("image_path")
        or item.get("imageUrl")
    )

    has_image = bool(str(snap).strip()) if snap is not None else False

    # MAIN RULE YOU ASKED FOR:
    # no valid time + no image => skip completely
    if first_seen <= 0 and last_seen <= 0 and not has_image:
        return True

    return False


def _maybe_emit_lf_notification(item: Dict[str, Any], request_base: str) -> None:
    if not isinstance(item, dict):
        return

    item_id = str(item.get("id") or "").strip()
    if not item_id:
        return

    status = str(item.get("status") or "lost").lower().strip()
    if status != "lost":
        return

    with _lf_notif_lock:
        if item_id in _lf_emitted_ids:
            return
        _lf_emitted_ids.add(item_id)

    ts = item.get("lastSeenTs") or item.get("firstSeenTs") or time.time()
    try:
        ts = float(ts)
    except Exception:
        ts = time.time()

    label = str(item.get("label") or "item")
    location = str(item.get("location") or item.get("cameraId") or item.get("videoId") or "Unknown")
    image_url = item.get("imageUrl")

    severity = "medium"
    if label.lower().replace(" ", "_") in ("wallet", "mobile_phone", "laptop", "tablet"):
        severity = "high"

    payload = {
        "id": item_id,
        "cameraId": str(item.get("cameraId") or item.get("videoId") or ""),
        "cameraName": location,
        "type": "lost-found",
        "timestamp": ts,
        "severity": severity,
        "message": f"Lost item detected: {label} ({location})",
        "imageUrl": image_url,
    }
    _lf_broadcast_notification(payload)


    
@app.get("/api/lostfound/items")
def api_lostfound_items(request: Request):
    base = str(request.base_url).rstrip("/")

    overrides = _read_overrides()

    def _is_deleted_item(item_id: str) -> bool:
        ov = overrides.get(str(item_id or "").strip()) or {}
        return bool(ov.get("deleted") is True)

    def _store_merge(item: Dict[str, Any]) -> None:
        """
        Merge item into persistent store.
        Deleted items are NOT re-merged, but old history already in store is kept.
        """
        if not isinstance(item, dict):
            return

        store = _read_items_store()
        if not isinstance(store, dict):
            store = {}

        iid = str(item.get("id") or "").strip()
        if not iid:
            return

        # do not re-merge deleted items into event history
        if _is_deleted_item(iid):
            return

        prev = store.get(iid) if isinstance(store.get(iid), dict) else {}
        merged = dict(prev)

        # keep earliest firstSeenTs
        prev_first = merged.get("firstSeenTs")
        new_first = item.get("firstSeenTs")
        if prev_first is None and new_first is not None:
            merged["firstSeenTs"] = new_first
        elif prev_first is not None and new_first is not None:
            try:
                merged["firstSeenTs"] = min(float(prev_first), float(new_first))
            except Exception:
                merged["firstSeenTs"] = prev_first

        # keep latest lastSeenTs
        prev_last = merged.get("lastSeenTs")
        new_last = item.get("lastSeenTs")
        if prev_last is None and new_last is not None:
            merged["lastSeenTs"] = new_last
        elif prev_last is not None and new_last is not None:
            try:
                merged["lastSeenTs"] = max(float(prev_last), float(new_last))
            except Exception:
                merged["lastSeenTs"] = new_last

        for k in (
            "module",
            "source",
            "cameraId",
            "videoId",
            "location",
            "label",
            "status",
            "imageUrl",
            "snapshot",
            "snapshot_path",
            "image_path",
            "raw",
            "notes",
            "updatedAt",
        ):
            if k in item and item[k] is not None:
                merged[k] = item[k]

        src_id = str(merged.get("cameraId") or merged.get("videoId") or "").strip()
        source = str(merged.get("source") or "").strip().lower()

        if src_id:
            resolved_name = _resolve_item_location(src_id)
            merged["location"] = resolved_name

            old_id = str(iid).strip()
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(resolved_name or "")).strip("_")

            new_id = old_id

            if safe_name:
                if source == "live":
                    # current normalize format:
                    # live-{cam_id}_{safe_name}_{raw_id}
                    wanted_prefix = f"live-{src_id}_{safe_name}_"

                    if not old_id.startswith(wanted_prefix):
                        if old_id.startswith(f"live-{src_id}_"):
                            rest = old_id[len(f"live-{src_id}_"):]
                            if not rest.startswith(f"{safe_name}_"):
                                new_id = wanted_prefix + rest
                        elif old_id.startswith(f"live-{src_id}-"):
                            rest = old_id[len(f"live-{src_id}-"):]
                            new_id = wanted_prefix + rest

                elif source in ("upload", "offline"):
                    # current normalize format:
                    # offline-{stem}_{safe_name}_{lost_id}
                    wanted_prefix = f"offline-{src_id}_{safe_name}_"

                    if not old_id.startswith(wanted_prefix):
                        if old_id.startswith(f"offline-{src_id}_"):
                            rest = old_id[len(f"offline-{src_id}_"):]
                            if not rest.startswith(f"{safe_name}_"):
                                new_id = wanted_prefix + rest
                        elif old_id.startswith(f"offline-{src_id}-"):
                            rest = old_id[len(f"offline-{src_id}-"):]
                            new_id = wanted_prefix + rest

            if new_id != old_id:
                merged["id"] = new_id
                store.pop(old_id, None)
                iid = new_id
            else:
                merged["id"] = old_id
        else:
            merged["id"] = iid

        store[iid] = merged
        _write_items_store(store)

    # -------------------------
    # 1) LIVE items from running pipelines
    # -------------------------
    try:
        for cam_id, p in list(pipelines_live.items()):
            try:
                items = p.pull_lost_items_for_api() if hasattr(p, "pull_lost_items_for_api") else []
            except Exception:
                items = []

            if not isinstance(items, list):
                continue

            for it in items:
                if not isinstance(it, dict):
                    continue

                norm = _normalize_live_item(cam_id, it, base)
                norm["id"] = str(norm.get("id") or "").strip()
                if not norm["id"]:
                    continue

                snap = it.get("snapshot_path") or it.get("snapshot") or it.get("image_path")
                if snap:
                    snap = str(snap)
                    norm["snapshot"] = snap
                    norm["snapshot_path"] = snap
                    norm["image_path"] = snap
                    try:
                        norm["imageUrl"] = to_image_url(snap, base)
                    except Exception:
                        norm["imageUrl"] = None
                else:
                    norm["imageUrl"] = None

                norm = _apply_override(norm, overrides)

                if _should_skip_lf_event_item(norm):
                    continue

                if norm.get("deleted") is True:
                    continue

                _store_merge(norm)
                _maybe_emit_lf_notification(norm, base)

    except Exception:
        pass

    # -------------------------
    # 1B) LIVE saved items from disk
    # -------------------------
    try:
        if LIVE_ROOT.exists():
            for cam_dir in LIVE_ROOT.iterdir():
                if not cam_dir.is_dir():
                    continue

                cam_id = cam_dir.name
                p_json = cam_dir / "lost_items.json"
                if not p_json.exists():
                    continue

                data = _read_json_file(p_json)

                items = None
                if isinstance(data, dict):
                    items = data.get("items")
                if items is None and isinstance(data, list):
                    items = data
                if not isinstance(items, list):
                    items = []

                for it in items:
                    if not isinstance(it, dict):
                        continue

                    norm = _normalize_live_item(cam_id, it, base)
                    norm["id"] = str(norm.get("id") or "").strip()
                    if not norm["id"]:
                        continue

                    snap = it.get("snapshot_path") or it.get("snapshot") or it.get("image_path")
                    if snap:
                        snap = str(snap)
                        norm["snapshot"] = snap
                        norm["snapshot_path"] = snap
                        norm["image_path"] = snap
                        try:
                            norm["imageUrl"] = to_image_url(snap, base)
                        except Exception:
                            norm["imageUrl"] = None
                    else:
                        norm["imageUrl"] = None

                    norm = _apply_override(norm, overrides)

                    if _should_skip_lf_event_item(norm):
                        continue

                    if norm.get("deleted") is True:
                        continue

                    _store_merge(norm)

    except Exception:
        pass

    # -------------------------
    # 2) OFFLINE analyzed items
    # -------------------------
    try:
        reconcile_manifest_with_offline_folder()
        vids = _manifest_all()

        for v in vids:
            stem = v.get("id")
            if not stem:
                continue

            p_json = _outputs_dir("offline", stem) / "lost_items.json"
            if not p_json.exists():
                continue

            data = _read_json_file(p_json)

            items = None
            if isinstance(data, dict):
                items = data.get("items")
            if items is None and isinstance(data, list):
                items = data
            if not isinstance(items, list):
                items = []

            for it in items:
                if not isinstance(it, dict):
                    continue

                norm = _normalize_offline_item(stem, it, base)
                norm["id"] = str(norm.get("id") or "").strip()
                if not norm["id"]:
                    continue

                snap = it.get("snapshot_path") or it.get("snapshot") or it.get("image_path")
                if snap:
                    snap = str(snap)
                    norm["snapshot"] = snap
                    norm["snapshot_path"] = snap
                    norm["image_path"] = snap
                    try:
                        norm["imageUrl"] = to_image_url(snap, base)
                    except Exception:
                        norm["imageUrl"] = None
                else:
                    norm["imageUrl"] = None

                norm = _apply_override(norm, overrides)

                if _should_skip_lf_event_item(norm):
                    continue

                if norm.get("deleted") is True:
                    continue

                _store_merge(norm)

    except Exception:
        pass

    # IMPORTANT: reload latest store after all merges
    store = _read_items_store()
    if not isinstance(store, dict):
        store = {}

    # -------------------------
    # 3) Build response from STORE (history)
    # -------------------------
    unique: List[dict] = []

    for iid, it in (store or {}).items():
        if not isinstance(it, dict):
            continue

        item = dict(it)
        item["id"] = str(item.get("id") or iid).strip()
        if not item["id"]:
            continue

        item = _apply_override(item, overrides)

        if item.get("deleted") is True:
            continue

        if _should_skip_lf_event_item(item):
            continue

        snap = (
            item.get("snapshot")
            or item.get("snapshot_path")
            or item.get("image_path")
        )

        image_url = None

        if snap:
            try:
                snap_str = str(snap).strip()
                if snap_str:
                    item["snapshot"] = snap_str
                    item["snapshot_path"] = snap_str
                    item["image_path"] = snap_str

                    snap_path = Path(snap_str)
                    if not snap_path.is_absolute():
                        snap_path = (OUTPUTS_LF_DIR / snap_path).resolve()

                    if snap_path.exists():
                        try:
                            image_url = to_image_url(str(snap_path), base)
                        except Exception:
                            image_url = None
            except Exception:
                image_url = None

        item["imageUrl"] = image_url
        unique.append(item)

    # -------------------------
    # 4) Sort newest first
    # -------------------------
    def _sort_key(x: dict) -> int:
        try:
            return int(x.get("lastSeenTs") or x.get("firstSeenTs") or 0)
        except Exception:
            return 0

    unique.sort(key=_sort_key, reverse=True)
    return JSONResponse({"items": unique})

# ============================================================
# Lost & Found Item Actions (Solve / Update Notes / Delete)
# ============================================================
@app.post("/api/lostfound/item/{item_id}/solve")
def api_lf_solve_item(item_id: str):
    item_id = (item_id or "").strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="Missing item_id")

    overrides = _read_overrides()
    rec = overrides.get(item_id) or {}
    rec["status"] = "solved"
    rec["updated_at"] = int(time.time())
    # if previously deleted, keep delete state authoritative
    overrides[item_id] = rec
    _write_overrides(overrides)

    return {"ok": True, "id": item_id, "status": "solved"}


@app.post("/api/lostfound/item/{item_id}/update")
def api_lf_update_item(item_id: str, payload: Dict[str, Any] = Body(default={})):
    item_id = (item_id or "").strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="Missing item_id")

    notes = payload.get("notes", "")
    notes = "" if notes is None else str(notes)

    overrides = _read_overrides()
    rec = overrides.get(item_id) or {}
    rec["notes"] = notes
    rec["updated_at"] = int(time.time())
    overrides[item_id] = rec
    _write_overrides(overrides)

    return {"ok": True, "id": item_id, "notes": notes}

def _mark_item_store_deleted(item_id: str, now_ts: int) -> bool:
    item_id = (item_id or "").strip()
    if not item_id:
        return False

    with _items_store_lock:
        for p in _lf_list_shard_paths(LF_STORE_ITEMS_DIR, LF_STORE_ITEMS_PREFIX):
            shard = _lf_load_shard(p)
            if not isinstance(shard, dict):
                continue

            rec = shard.get(item_id)
            if not isinstance(rec, dict):
                continue

            updated = dict(rec)
            updated["status"] = "deleted"
            updated["deleted"] = True
            updated["updatedAt"] = int(now_ts)
            shard[item_id] = updated

            _lf_save_shard(p, shard)
            return True

    return False

@app.delete("/api/lostfound/item/{item_id}")
def api_lf_delete_item(item_id: str):
    item_id = (item_id or "").strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="Missing item_id")

    now_ts = int(time.time())

    overrides = _read_overrides()
    rec = overrides.get(item_id) or {}
    rec["deleted"] = True
    rec["status"] = "deleted"
    rec["updated_at"] = now_ts
    overrides[item_id] = rec
    _write_overrides(overrides)

    _mark_item_store_deleted(item_id, now_ts)

    return {
        "ok": True,
        "id": item_id,
        "deleted": True,
        "status": "deleted",
    }
    

def _get_lostfound_report_items(request: Request, include_deleted: bool = True):
    base = str(request.base_url).rstrip("/")
    overrides = _read_overrides()
    store = _read_items_store()
    if not isinstance(store, dict):
        store = {}

    out = []
    for iid, it in store.items():
        if not isinstance(it, dict):
            continue

        item = dict(it)
        item["id"] = str(iid).strip()
        if not item["id"]:
            continue

        item = _apply_override(item, overrides)

        if not include_deleted and (
            item.get("deleted") is True or str(item.get("status") or "").lower() == "deleted"
        ):
            continue

        snap = (
            item.get("snapshot")
            or item.get("snapshot_path")
            or item.get("image_path")
        )

        image_url = None

        if snap:
            try:
                snap_str = str(snap).strip()
                if snap_str:
                    item["snapshot"] = snap_str
                    item["snapshot_path"] = snap_str
                    item["image_path"] = snap_str

                    snap_path = Path(snap_str)
                    if not snap_path.is_absolute():
                        snap_path = (OUTPUTS_LF_DIR / snap_path).resolve()

                    if snap_path.exists():
                        try:
                            image_url = to_image_url(str(snap_path), base)
                        except Exception:
                            image_url = None
            except Exception:
                image_url = None

        item["imageUrl"] = image_url
        out.append(item)

    out.sort(
        key=lambda x: int(x.get("lastSeenTs") or x.get("firstSeenTs") or 0),
        reverse=True
    )
    return out

@app.get("/api/lostfound/reports/items")
def api_lostfound_report_items(request: Request):
    items = _get_lostfound_report_items(request, include_deleted=True)
    return JSONResponse({"items": items})


@app.on_event("shutdown")
def persist_live_items_on_shutdown():
    print("Persisting live lost items before shutdown...")

    overrides = _read_overrides()
    store = _read_items_store()
    if not isinstance(store, dict):
        store = {}

    # IMPORTANT:
    # Do NOT use "http://localhost" here because after restart/frontend access
    # the base URL may be different.
    base = ""

    for cam_id, p in list(pipelines.items()):
        try:
            items = p.pull_lost_items_for_api() if hasattr(p, "pull_lost_items_for_api") else []
        except Exception:
            continue

        if not isinstance(items, list):
            continue

        for it in items:
            try:
                norm = _normalize_live_item(cam_id, it, base)
                iid = str(norm.get("id") or "").strip()
                if not iid:
                    continue

                prev = store.get(iid) if isinstance(store.get(iid), dict) else {}
                merged = dict(prev)

                if merged.get("firstSeenTs") is None and norm.get("firstSeenTs") is not None:
                    merged["firstSeenTs"] = norm.get("firstSeenTs")

                # IMPORTANT:
                # Do NOT persist imageUrl because it may become stale after restart.
                for k in (
                    "lastSeenTs",
                    "module",
                    "source",
                    "cameraId",
                    "videoId",
                    "location",
                    "label",
                    "status",
                    "raw",
                    "notes",
                    "updatedAt",
                ):
                    if k in norm and norm[k] is not None:
                        merged[k] = norm[k]

                snap = it.get("snapshot_path") or it.get("snapshot") or it.get("image_path")
                if snap:
                    merged["snapshot"] = str(snap)

                # remove any old stale imageUrl
                merged.pop("imageUrl", None)

                store[iid] = merged

                rec = overrides.get(iid) or {}
                rec["status"] = norm.get("status", "lost")
                rec["notes"] = rec.get("notes", "")
                rec["updated_at"] = int(time.time())
                if snap:
                    rec["snapshot"] = str(snap)
                overrides[iid] = rec

            except Exception:
                continue

    _write_items_store(store)
    _write_overrides(overrides)

    print("Live lost items persisted successfully.")

# ============================================================
# SETTINGS STATIC ROI FRAMES (frozen snapshots)
# ============================================================
_static_lock = threading.Lock()
_static_frame_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}  # (cam_id, view_idx) -> {"jpg": bytes, "ts": float}
_static_group_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (cam_id, group) -> {"jpg": bytes, "ts": float}


def _allowed_live_cam_ids(*, only_enabled: bool = False) -> Set[str]:
    """
    Returns cam_ids used by Live View / Settings page.

    Upload:
    - source of truth = upload folder + _cameras_enabled.json

    RTSP:
    - source of truth = pipelines_live, and if only_enabled=True,
      also respect rtsp_sources.json enabled flag
    """
    allowed: Set[str] = set()

    # -------------------------------------------------
    # 1) Upload cameras from upload folder
    # -------------------------------------------------
    try:
        upload_enabled = load_cameras_enabled() or {}
        files = list_upload_videos_h264_only()

        for f in files:
            cid = _upload_cam_id_from_file(f)
            if not cid:
                continue

            if only_enabled:
                if bool(upload_enabled.get(cid, True)):
                    allowed.add(cid)
            else:
                allowed.add(cid)
    except Exception:
        pass

    # -------------------------------------------------
    # 2) RTSP cameras from running live pipelines
    # -------------------------------------------------
    try:
        rtsp_store = load_rtsp_sources() or {}

        for cid in (pipelines_live or {}).keys():
            cid2 = str(_base_id(cid)).strip()
            if not cid2:
                continue

            if str(cid2).startswith("rtsp_"):
                if only_enabled:
                    rec = rtsp_store.get(cid2) or {}
                    if bool(rec.get("enabled", True)):
                        allowed.add(cid2)
                else:
                    allowed.add(cid2)
            else:
                # upload pipeline ids already handled from upload folder
                if not only_enabled:
                    allowed.add(cid2)
    except Exception:
        pass

    return allowed
@app.get("/api/lostfound/cameras_for_settings")
def get_lostfound_cameras_for_settings(request: Request, start: str = "A"):
    """
    Settings page ONLY:
    return cameras actually used by Live View.

    Important:
    - This endpoint is only for Settings page camera metadata
    - No object detection needed here
    - Must explicitly return fisheye/normal info for frontend
    """
    base = str(request.base_url).rstrip("/")
    cams: List[Dict[str, Any]] = []

    start = (start or "A").upper().strip()
    if start not in ("A", "B"):
        start = "A"

    allowed_live = _allowed_live_cam_ids(only_enabled=True)

    # =========================================================
    # 1) Uploaded videos
    # =========================================================
    files = list_upload_videos_h264_only()

    for file in files:
        stem = file.stem
        cam_id = _cam_id_from_h264_file(file)
        cam_id = str(cam_id).strip()

        if not cam_id:
            cam_id = stem

        # settings page only shows sources that are used in Live View
        if allowed_live and cam_id not in allowed_live:
            continue

        cam_name = cam_name_from_file(file)

        # ✅ use cam_id, not stem
        vtype = resolve_settings_video_type(cam_id, str(file))

        _system(f"SETTINGS CAM UPLOAD: cam_id={cam_id} stem={stem} resolved_vtype={vtype}")

        if vtype == "fisheye":
            outA = grid_path_for_group(stem, "A")
            outB = grid_path_for_group(stem, "B")

            if outA.exists() and outB.exists():
                a_rel = outA.relative_to(GRID_DIR).as_posix()
                b_rel = outB.relative_to(GRID_DIR).as_posix()

                views = [
                    {
                        "id": f"{cam_id}__A",
                        "name": f"{cam_name} (Group A)",
                        "videoUrl": f"{base}/grid/{a_rel}",
                        "filename": a_rel,
                        "order": 0,
                    },
                    {
                        "id": f"{cam_id}__B",
                        "name": f"{cam_name} (Group B)",
                        "videoUrl": f"{base}/grid/{b_rel}",
                        "filename": b_rel,
                        "order": 1,
                    },
                ]

                if start == "B":
                    views = [views[1], views[0]]

                cams.append(
                    {
                        "id": cam_id,  # ✅ use cam_id
                        "name": cam_name,
                        "groupId": cam_id,
                        "order": 0,
                        "classroomId": cam_name,
                        "location": "Block B - Uploaded",
                        "status": "online",
                        "recording": True,
                        "videoUrl": views[0]["videoUrl"],
                        "filename": views[0]["filename"],
                        "video_type": "fisheye",
                        "is_fisheye": True,
                        "views_count": 8,
                        "is_rtsp": False,
                        "views": views,
                    }
                )
            else:
                cams.append(
                    {
                        "id": cam_id,  # ✅ use cam_id
                        "name": cam_name,
                        "groupId": cam_id,
                        "order": 0,
                        "classroomId": cam_name,
                        "location": "Block B - Uploaded",
                        "status": "online",
                        "recording": True,
                        "videoUrl": f"{base}/videos/{file.name}",
                        "filename": file.name,
                        "video_type": "fisheye",
                        "is_fisheye": True,
                        "views_count": 8,
                        "is_rtsp": False,
                        "views": [
                            {"id": f"{cam_id}__A", "name": f"{cam_name} (Group A)", "order": 0},
                            {"id": f"{cam_id}__B", "name": f"{cam_name} (Group B)", "order": 1},
                        ],
                    }
                )

                try:
                    maybe_auto_trigger_grid(stem, file)
                except Exception as e:
                    _system(f"SETTINGS CAM UPLOAD: maybe_auto_trigger_grid failed for {cam_id}: {e}")

            continue

        cams.append(
            {
                "id": cam_id,  # ✅ use cam_id
                "name": cam_name,
                "groupId": cam_id,
                "order": 0,
                "classroomId": cam_name,
                "location": "Block B - Uploaded",
                "status": "online",
                "recording": True,
                "videoUrl": f"{base}/videos/{file.name}",
                "filename": file.name,
                "video_type": "normal",
                "is_fisheye": False,
                "views_count": 1,
                "is_rtsp": False,
            }
        )
    # =========================================================
    # 2) RTSP sources
    # =========================================================
    try:
        rtsp = load_rtsp_sources() or {}

        for rid, rec in rtsp.items():
            # support old plain-string format
            if isinstance(rec, str):
                rec = {"url": rec, "video_type": "auto"}

            if not isinstance(rec, dict):
                continue

            url = (rec.get("url") or "").strip()
            if not url:
                continue

            cam_id = str(rid).strip()
            base_cam_id = _base_id(cam_id)

            # same filtering logic
            if allowed_live and cam_id not in allowed_live and base_cam_id not in allowed_live:
                continue

            cam_name = rec.get("name") or cam_id
            vtype = resolve_settings_video_type(cam_id, url)

            _system(f"SETTINGS CAM RTSP: id={cam_id} resolved_vtype={vtype}")

            if vtype == "fisheye":
                views = [
                    {"id": f"{cam_id}__A", "name": f"{cam_name} (Group A)", "order": 0},
                    {"id": f"{cam_id}__B", "name": f"{cam_name} (Group B)", "order": 1},
                ]

                if start == "B":
                    views = [views[1], views[0]]

                cams.append(
                    {
                        "id": cam_id,
                        "name": cam_name,
                        "groupId": cam_id,
                        "order": 0,
                        "classroomId": cam_name,
                        "location": "RTSP - Live",
                        "status": "online",
                        "recording": bool(rec.get("enabled", True)),
                        "videoUrl": "",
                        "filename": "",
                        "video_type": "fisheye",
                        "is_fisheye": True,
                        "views_count": 8,
                        "is_rtsp": True,
                        "views": views,
                    }
                )
            else:
                cams.append(
                    {
                        "id": cam_id,
                        "name": cam_name,
                        "groupId": cam_id,
                        "order": 0,
                        "classroomId": cam_name,
                        "location": "RTSP - Live",
                        "status": "online",
                        "recording": bool(rec.get("enabled", True)),
                        "videoUrl": "",
                        "filename": "",
                        "video_type": "normal",
                        "is_fisheye": False,
                        "views_count": 1,
                        "is_rtsp": True,
                    }
                )

    except Exception as e:
        _system(f"cameras: load_rtsp_sources error: {e}")

    cams.sort(key=lambda c: (str(c.get("classroomId", "")), str(c.get("groupId", ""))))
    return cams


def _build_static_view_jpg(cam_id: str, view_idx: int) -> bytes:
    cam_id = _live_id(cam_id)

    # ---------------------------------------------------------
    # Resolve live source first
    # ---------------------------------------------------------
    src_str = resolve_live_source(cam_id)
    if not src_str:
        raise HTTPException(status_code=404, detail="live source not found")

    is_rtsp = str(src_str).lower().startswith(("rtsp://", "rtsps://"))

    # ---------------------------------------------------------
    # Use unified settings-page video type resolver
    # ---------------------------------------------------------
    vtype = resolve_settings_video_type(cam_id, str(src_str))

    _system(f"STATIC VIEW: cam_id={cam_id} src={src_str} resolved_vtype={vtype} view_idx={view_idx}")

    # ---------------------------------------------------------
    # NORMAL camera -> just return one raw frame
    # ---------------------------------------------------------
    if vtype != "fisheye":
        cap = cv2.VideoCapture(str(src_str))
        try:
            fr = _read_any_frame(cap)
        finally:
            cap.release()

        if fr is None:
            raise HTTPException(status_code=404, detail="no frame")

        return _jpg_bytes(fr, quality=85)

    # ---------------------------------------------------------
    # FISHEYE camera -> build one dewarped view
    # ---------------------------------------------------------
    try:
        if is_rtsp:
            pre = _get_or_make_live_fisheye_pre_rtsp(cam_id, str(src_str))
        else:
            pre = _get_or_make_live_fisheye_pre(cam_id, Path(str(src_str)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fisheye preprocessor error: {e}")

    cap = cv2.VideoCapture(str(src_str))
    try:
        fr = _read_any_frame(cap)
    finally:
        cap.release()

    if fr is None:
        raise HTTPException(status_code=404, detail="no frame")

    wanted_names = _fisheye_order_names()

    vi = int(view_idx)
    if vi < 0:
        vi = 0
    if vi >= len(wanted_names):
        vi = 0

    name = wanted_names[vi]

    try:
        views = pre.get_views(fr, allowed_names=[name])
    except TypeError:
        views = pre.get_views(fr)

    chosen = None
    for v in (views or []):
        if (v.get("name") or "") == name:
            chosen = v
            break

    if chosen is None or chosen.get("image") is None:
        raise HTTPException(status_code=404, detail=f"fisheye view missing: {name}")

    img = chosen["image"].copy()

    # force same size as live single-view if needed
    h, w = img.shape[:2]
    if h > 0 and w > 0:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    img = _draw_label_bar(img, f"{name.replace('_', ' ').title()}")

    return _jpg_bytes(img, quality=85)

def _build_static_group_jpg(cam_id: str, group: str) -> bytes:
    cam_id = _live_id(cam_id)

    # ---------------------------------------------------------
    # Resolve live source first
    # ---------------------------------------------------------
    src_str = resolve_live_source(cam_id)
    if not src_str:
        raise HTTPException(status_code=404, detail="live source not found")

    is_rtsp = str(src_str).lower().startswith(("rtsp://", "rtsps://"))

    # ---------------------------------------------------------
    # Use unified settings-page video type resolver
    # ---------------------------------------------------------
    vtype = resolve_settings_video_type(cam_id, str(src_str))

    g = (group or "A").upper().strip()
    if g not in ("A", "B"):
        g = "A"

    _system(f"STATIC GROUP: cam_id={cam_id} src={src_str} resolved_vtype={vtype} group={g}")

    # ---------------------------------------------------------
    # NORMAL camera -> just return one raw frame
    # ---------------------------------------------------------
    if vtype != "fisheye":
        cap = cv2.VideoCapture(str(src_str))
        try:
            fr = _read_any_frame(cap)
        finally:
            cap.release()

        if fr is None:
            raise HTTPException(status_code=404, detail="no frame")

        return _jpg_bytes(fr, quality=85)

    # ---------------------------------------------------------
    # FISHEYE camera -> build group A/B 2x2 grid
    # ---------------------------------------------------------
    try:
        if is_rtsp:
            pre = _get_or_make_live_fisheye_pre_rtsp(cam_id, str(src_str))
        else:
            pre = _get_or_make_live_fisheye_pre(cam_id, Path(str(src_str)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fisheye preprocessor error: {e}")

    cap = cv2.VideoCapture(str(src_str))
    try:
        fr = _read_any_frame(cap)
    finally:
        cap.release()

    if fr is None:
        raise HTTPException(status_code=404, detail="no frame")

    group_names = _group_names(g)

    try:
        group_views = pre.get_views(fr, allowed_names=group_names)
    except TypeError:
        group_views = pre.get_views(fr)

    name_to_img: Dict[str, np.ndarray] = {}
    for v in (group_views or []):
        nm = v.get("name", "")
        im = v.get("image")
        if nm and im is not None:
            name_to_img[nm] = im

    tiles: List[Optional[np.ndarray]] = [None, None, None, None]

    for i, view_name in enumerate(group_names[:4]):
        bgr = name_to_img.get(view_name)
        if bgr is None:
            continue

        bgr = bgr.copy()

        # ✅ SETTINGS GROUP PREVIEW SHOULD NOT DRAW SAVED ROI
        bgr = _draw_label_bar(bgr, f"{g} | {view_name.replace('_', ' ').title()}")
        tiles[i] = bgr

        valid_tiles = [t for t in tiles if isinstance(t, np.ndarray)]

    if not valid_tiles:
        blank = make_2x2_grid([_blank_bgr(), _blank_bgr(), _blank_bgr(), _blank_bgr()], 480, 640)
        return _jpg_bytes(blank, quality=85)

    # IMPORTANT:
    # make_2x2_grid(imgs, cell_h, cell_w)
    # so pass HEIGHT first, WIDTH second
    cell_h = min(t.shape[0] for t in valid_tiles)
    cell_w = min(t.shape[1] for t in valid_tiles)

    grid = make_2x2_grid(tiles, cell_h, cell_w)
    return _jpg_bytes(grid, quality=85)

@app.get("/api/settings/static/frame/{cam_id}/{view_idx}")
def settings_static_frame(cam_id: str, view_idx: int, refresh: int = Query(0)):
    """
    ✅ Settings ROI page: frozen single-view image.
    - refresh=0 => reuse same frozen frame
    - refresh=1 => rebuild and replace frozen frame
    """
    cam_id = _live_id(cam_id)
    if focus_alive() and not focus_is(page="settings", active_id=cam_id):
        raise HTTPException(status_code=404, detail="not focused (skip settings static)")
    allowed_live = _allowed_live_cam_ids(only_enabled=False)
    if allowed_live and cam_id not in allowed_live:
        raise HTTPException(status_code=404, detail="camera not in live view")
    key = (cam_id, int(view_idx))

    with _static_lock:
        if not refresh:
            rec = _static_frame_cache.get(key)
            if rec and rec.get("jpg"):
                return Response(rec["jpg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    jpg = _build_static_view_jpg(cam_id, int(view_idx))

    with _static_lock:
        _static_frame_cache[key] = {"ts": time.time(), "jpg": jpg}

    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/settings/static/group_frame/{cam_id}/{group}")
def settings_static_group_frame(cam_id: str, group: str, refresh: int = Query(0)):
    cam_id = _live_id(cam_id)
    if focus_alive() and not focus_is(page="settings", active_id=cam_id):
        raise HTTPException(status_code=404, detail="not focused (skip settings static grid)")
    allowed_live = _allowed_live_cam_ids(only_enabled=False)
    if allowed_live and cam_id not in allowed_live:
        raise HTTPException(status_code=404, detail="camera not in live view")
    g = (group or "A").upper().strip()
    if g not in ("A", "B"):
        g = "A"
    key = (cam_id, g)

    with _static_lock:
        if not refresh:
            rec = _static_group_cache.get(key)
            if rec and rec.get("jpg"):
                return Response(rec["jpg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    jpg = _build_static_group_jpg(cam_id, g)

    with _static_lock:
        _static_group_cache[key] = {"ts": time.time(), "jpg": jpg}

    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ============================================================
# Reports Export (CSV / JSON)
# ============================================================
def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def _matches_filters(it: Dict[str, Any], q: str, status: str, source: str, label: str, location: str) -> bool:
    q = (q or "").strip().lower()
    status = (status or "").strip().lower()
    source = (source or "").strip().lower()
    label = (label or "").strip().lower()
    location = (location or "").strip().lower()

    # status
    it_status = _safe_str(it.get("status", "lost")).lower()
    if status in ("lost", "solved"):
        if status == "lost" and "lost" not in it_status:
            return False
        if status == "solved" and ("solv" not in it_status and it_status != "resolved"):
            return False

    # source
    it_source = _safe_str(it.get("source", "")).lower()
    if source and source != "all":
        if source == "offline":
            looks_offline = bool(it.get("videoId")) and not bool(it.get("cameraId"))
            if not looks_offline and it_source != "offline":
                return False
        else:
            if it_source != source:
                # allow your historical "upload" for offline results
                if not (source == "upload" and it_source == "offline"):
                    return False

    # label/location exact filters
    if label and label != "all":
        if _safe_str(it.get("label", "")).lower() != label:
            return False
    if location and location != "all":
        if _safe_str(it.get("location", "")).lower() != location:
            return False

    # free text search
    if q:
        hay = " ".join(
            [
                _safe_str(it.get("id")),
                _safe_str(it.get("label")),
                _safe_str(it.get("location")),
                _safe_str(it.get("cameraId")),
                _safe_str(it.get("videoId")),
                _safe_str(it.get("source")),
                _safe_str(it.get("status")),
                _safe_str(it.get("notes")),
            ]
        ).lower()
        if q not in hay:
            return False

    return True


@app.get("/api/lostfound/items/export.csv")
def export_lostfound_items_csv(
        request: Request,
        q: str = "",
        status: str = "",
        source: str = "",
        label: str = "",
        location: str = "",
):
    """
    Export the SAME data as /api/lostfound/items, but as CSV.
    Supports filters via querystring:
      ?q=...&status=lost|solved&source=live|upload|offline&label=...&location=...
    """
    # Reuse your existing /api/lostfound/items logic by calling it directly
    # (keeps consistent with overrides + history store + deleted filter)
    data = api_lostfound_items(request)
    # JSONResponse -> .body bytes
    try:
        payload = json.loads(data.body.decode("utf-8"))
    except Exception:
        payload = {"items": []}

    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        items = []

    # apply filters
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if _matches_filters(it, q, status, source, label, location):
            out.append(it)

    # streaming CSV
    def _iter():
        import csv
        import io

        f = io.StringIO()
        w = csv.writer(f)

        w.writerow(
            [
                "id",
                "status",
                "label",
                "location",
                "source",
                "cameraId",
                "videoId",
                "firstSeenTs",
                "lastSeenTs",
                "firstSeen",
                "lastSeen",
                "imageUrl",
                "notes",
            ]
        )
        yield f.getvalue()
        f.seek(0)
        f.truncate(0)

        for it in out:
            first_ts = it.get("firstSeenTs") or 0
            last_ts = it.get("lastSeenTs") or 0

            # keep readable time strings too
            def _fmt(ts):
                try:
                    ts = int(ts or 0)
                    if ts <= 0:
                        return ""
                    dt = datetime.datetime.fromtimestamp(ts)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return ""

            w.writerow(
                [
                    _safe_str(it.get("id")),
                    _safe_str(it.get("status")),
                    _safe_str(it.get("label")),
                    _safe_str(it.get("location")),
                    _safe_str(it.get("source")),
                    _safe_str(it.get("cameraId")),
                    _safe_str(it.get("videoId")),
                    int(first_ts or 0),
                    int(last_ts or 0),
                    _fmt(first_ts),
                    _fmt(last_ts),
                    _safe_str(it.get("imageUrl")),
                    _safe_str(it.get("notes")),
                ]
            )
            yield f.getvalue()
            f.seek(0)
            f.truncate(0)

    headers = {
        "Content-Disposition": "attachment; filename=lost_found_reports.csv",
        "Cache-Control": "no-store",
    }
    return StreamingResponse(_iter(), media_type="text/csv", headers=headers)


@app.get("/api/lostfound/items/export.json")
def export_lostfound_items_json(
        request: Request,
        q: str = "",
        status: str = "",
        source: str = "",
        label: str = "",
        location: str = "",
):
    """
    Optional: export filtered items as JSON.
    """
    data = api_lostfound_items(request)
    try:
        payload = json.loads(data.body.decode("utf-8"))
    except Exception:
        payload = {"items": []}

    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        items = []

    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if _matches_filters(it, q, status, source, label, location):
            out.append(it)

    return JSONResponse({"items": out})


def _dashboard_overlay_flag() -> int:
    """
    Dashboard must always be display-only.
    Return 0 so ROI/overlay is never drawn for dashboard streams.
    """
    return 0


@app.get("/api/live/mjpeg_dashboard/{cam_id}/{group}")
def live_mjpeg_dashboard(cam_id: str, group: str):
    """
    Dashboard-only MJPEG stream.
    Always forces CLEAN dashboard output (no ROI overlay).

    group:
      - "A" / "B" => fisheye group grid
      - "0"       => single view
    """
    cam_id = _live_id(cam_id)

    g = (group or "A").upper().strip()
    if g not in ("A", "B", "0"):
        g = "A"

    overlay = _dashboard_overlay_flag()  # kept for clarity / future use

    FPS = 4.0
    FRAME_DT = 1.0 / FPS
    STALE_OK_SEC = 2.0

    def _detect_live_type() -> str:
        p = pipelines_live.get(cam_id)
        if p is not None:
            try:
                if bool(getattr(p, "is_fisheye", False)):
                    return "fisheye"
                return "normal"
            except Exception:
                pass

        src_str = resolve_live_source(cam_id)
        if not src_str:
            return "normal"

        is_rtsp = str(src_str).lower().startswith(("rtsp://", "rtsps://"))
        if is_rtsp:
            try:
                rec = _rtsp_store_get(load_rtsp_sources() or {}, cam_id)
                vt = (rec.get("video_type") if isinstance(rec, dict) else "auto") or "auto"
                vt = str(vt).strip().lower()
                if vt == "fisheye":
                    return "fisheye"
                if vt == "normal":
                    return "normal"
                return detect_rtsp_video_type(cam_id, str(src_str))
            except Exception:
                return "normal"
        else:
            try:
                return "fisheye" if detect_video_type_cached(cam_id, Path(str(src_str))) == "fisheye" else "normal"
            except Exception:
                return "normal"

    def gen():
        last_ok_ts = 0.0

        while True:
            try:
                p = pipelines_live.get(cam_id)
                live_type = _detect_live_type()
                jpg = None

                if p is not None:
                    # -------------------------------------------------
                    # BEST: use the unified clean dashboard pipeline API
                    # -------------------------------------------------
                    try:
                        jpg = p.pull_dashboard_jpg(g)
                    except Exception:
                        jpg = None

                    # -------------------------------------------------
                    # Fallback: call clean functions directly
                    # -------------------------------------------------
                    if jpg is None:
                        if live_type == "fisheye" and g in ("A", "B"):
                            try:
                                jpg = p.pull_group_grid_jpg_clean(g)
                            except Exception:
                                jpg = None
                        else:
                            try:
                                view_idx = int(g)
                            except Exception:
                                view_idx = 0

                            try:
                                jpg = p.pull_single_view_jpg_clean(view_idx)
                            except Exception:
                                jpg = None

                if jpg:
                    last_ok_ts = time.time()
                    yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n\r\n"
                            + jpg +
                            b"\r\n"
                    )
                else:
                    # no frame available yet; avoid busy loop
                    if (time.time() - last_ok_ts) > STALE_OK_SEC:
                        time.sleep(0.08)
                    else:
                        time.sleep(0.03)

                time.sleep(FRAME_DT)

            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.1)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/live/mjpeg/{cam_id}/{group}")
def live_mjpeg(cam_id: str, group: str, overlay: int = Query(1)):
    """
    MJPEG stream.

    Rules:
    - NORMAL: always stream SINGLE view (ignore group A/B), use overlay=0 to force no ROI.
    - FISHEYE: stream 2x2 grid for group A/B.

    group:
      - "A" / "B" => fisheye grid groups
      - "0"       => force single view
    overlay:
      - 1 => allow overlay (ROI/dets if your rendering adds them)
      - 0 => raw (no ROI) (we force slow-path raw frame for NORMAL)
    """
    
    cam_id = _live_id(cam_id)

    g = (group or "A").upper().strip()
    if g not in ("A", "B", "0"):
        g = "A"

    overlay = 1 if int(overlay) == 1 else 0

    FPS = 4.0
    FRAME_DT = 1.0 / FPS
    STALE_OK_SEC = 2.0

    def _detect_live_type() -> str:
        """
        Prefer pipeline.is_fisheye (fast).
        Fallback: use RTSP store override or probe.
        """
        p = pipelines_live.get(cam_id)
        if p is not None:
            try:
                if bool(getattr(p, "is_fisheye", False)):
                    return "fisheye"
                return "normal"
            except Exception:
                pass

        src_str = resolve_live_source(cam_id)
        if not src_str:
            return "normal"

        is_rtsp = str(src_str).lower().startswith(("rtsp://", "rtsps://"))
        if is_rtsp:
            try:
                rec = _rtsp_store_get(load_rtsp_sources() or {}, cam_id)
                vt = (rec.get("video_type") if isinstance(rec, dict) else "auto") or "auto"
                vt = str(vt).strip().lower()
                if vt == "fisheye":
                    return "fisheye"
                if vt == "normal":
                    return "normal"
                # auto -> probe
                return detect_rtsp_video_type(cam_id, str(src_str))
            except Exception:
                return "normal"
        else:
            try:
                return "fisheye" if detect_video_type_cached(cam_id, Path(str(src_str))) == "fisheye" else "normal"
            except Exception:
                return "normal"

    def _get_jpg_once() -> bytes | None:
        """
        Returns one JPEG frame.
        - NORMAL:
            * overlay=0 -> force RAW (no ROI) by using the same logic as /api/live/group_frame normal path
            * overlay=1 -> allow pipeline cached single if exists, else fallback
        - FISHEYE:
            * use pipeline grid if exists, else fallback group_frame
        """
        vtype = _detect_live_type()

        # -------------------------
        # NORMAL: always single view
        # -------------------------
        if vtype != "fisheye":
            # If user requests raw (overlay=0), DO NOT use pipeline cached images
            # because they might already include ROI overlays.
            if overlay == 0:
                # Use the same slow fallback capture logic, but NO ROI drawing.
                src_str = resolve_live_source(cam_id)
                if not src_str:
                    return None

                cap = cv2.VideoCapture(str(src_str))
                try:
                    fr = _read_any_frame(cap)
                finally:
                    cap.release()

                if fr is None:
                    return None

                return _jpg_bytes(fr, quality=85)

            # overlay==1: try pipeline single view cache first
            p = pipelines_live.get(cam_id)
            if p is not None and hasattr(p, "pull_single_view_jpg"):
                try:
                    det_on = bool(getattr(p, "_detection_enabled", True))
                    jpg = p.pull_single_view_jpg(0, draw_roi_overlay=((overlay == 1) and det_on))       
                    if jpg:
                        return jpg
                except Exception:
                    pass

            # fallback (may draw ROI if your group_frame normal path does)
            # If you already updated /api/live/group_frame to respect overlay, call it indirectly by reusing logic here:
            src_str = resolve_live_source(cam_id)
            if not src_str:
                return None

            roi_path = _ensure_roi_file("live", cam_id)
            roi_config = {}
            if roi_path.exists():
                try:
                    roi_config = json.loads(roi_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            zones = get_zones_for_normal_video(roi_config)

            cap = cv2.VideoCapture(str(src_str))
            try:
                fr = _read_any_frame(cap)
            finally:
                cap.release()

            if fr is None:
                return None

            # overlay==1 => draw zones
            if zones:
                fr = draw_zones_on_view(fr, zones, color=(0, 255, 0), thickness=2, show_id=True)

            return _jpg_bytes(fr, quality=85)

        # -------------------------
        # FISHEYE: grid A/B (or force single if group==0)
        # -------------------------
        p = pipelines_live.get(cam_id)

        # force single fisheye view 0 if requested
        if g == "0":
            if p is not None and hasattr(p, "pull_single_view_jpg"):
                try:
                    jpg = p.pull_single_view_jpg(0)
                    if jpg:
                        return jpg
                except Exception:
                    pass
            # fallback to /api/live/frame logic (optional)
            return None

        # grid A/B
        if p is not None and hasattr(p, "pull_group_grid_jpg"):
            try:
                det_on = bool(getattr(p, "_detection_enabled", True))
                jpg = p.pull_group_grid_jpg(g, draw_roi_overlay=((overlay == 1) and det_on))    
                if jpg:
                    return jpg
            except Exception:
                pass

        # fallback grid_frame (slow) — if you added overlay param to group_frame, you can respect it here too
        # For now: we reuse the same slow logic (your existing live_group_frame already builds 2x2).
        # If you updated live_group_frame signature (overlay param), best is to inline that logic here.
        try:
            # inline minimal: call capture + preprocessor path (same as your live_group_frame fisheye section)
            src_str = resolve_live_source(cam_id)
            if not src_str:
                return None

            is_rtsp = str(src_str).lower().startswith(("rtsp://", "rtsps://"))

            # choose correct preprocessor getter (rtsp vs file)
            if is_rtsp:
                pre = _get_or_make_live_fisheye_pre_rtsp(cam_id, str(src_str))
            else:
                pre = _get_or_make_live_fisheye_pre(cam_id, Path(str(src_str)))

            cap = cv2.VideoCapture(str(src_str))
            try:
                fr = _read_any_frame(cap)
            finally:
                cap.release()

            if fr is None:
                return None

            group_names = _group_names(g)

            try:
                group_views = pre.get_views(fr, allowed_names=group_names)
            except TypeError:
                group_views = pre.get_views(fr)

            name_to_img: Dict[str, np.ndarray] = {}
            for v in (group_views or []):
                nm = v.get("name", "")
                im = v.get("image")
                if nm and im is not None:
                    name_to_img[nm] = im

            roi_path = _ensure_roi_file("live", cam_id)
            roi_config = {}
            if roi_path.exists():
                try:
                    roi_config = json.loads(roi_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            tiles: List[Optional[np.ndarray]] = [None, None, None, None]

            for i, view_name in enumerate(group_names[:4]):
                bgr = name_to_img.get(view_name)
                if bgr is None:
                    continue
                bgr = bgr.copy()

                if overlay == 1:
                    fisheye_polys = roi_config.get("fisheye_polygons", {})
                    view_polys = fisheye_polys.get(view_name, [])
                    for poly in view_polys:
                        if isinstance(poly, list) and len(poly) >= 3:
                            try:
                                pts = np.array([[p["x"], p["y"]] for p in poly], dtype=np.int32)
                                cv2.polylines(bgr, [pts], True, (0, 255, 0), 2)
                            except Exception:
                                pass
                    bgr = _draw_label_bar(bgr, f"{g} | {view_name.replace('_', ' ').title()}")

                tiles[i] = bgr

            valid_tiles = [t for t in tiles if isinstance(t, np.ndarray)]

            if not valid_tiles:
                blank = make_2x2_grid([_blank_bgr(), _blank_bgr(), _blank_bgr(), _blank_bgr()], 640, 480)
                return _jpg_bytes(blank, quality=85)

            cell_h = min(t.shape[0] for t in valid_tiles)
            cell_w = min(t.shape[1] for t in valid_tiles)

            grid = make_2x2_grid(tiles, cell_h, cell_w)
            return _jpg_bytes(grid, quality=85)

        except Exception:
            return None

    def gen():
        boundary = b"--frame\r\n"
        last_jpg = None
        last_jpg_t = 0.0
        next_t = time.perf_counter()

        try:
            while True:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(min(0.01, next_t - now))
                    continue
                next_t += FRAME_DT
                if next_t < now - 0.5:
                    next_t = now + FRAME_DT

                jpg = _get_jpg_once()

                if jpg:
                    last_jpg = jpg
                    last_jpg_t = time.perf_counter()
                else:
                    if last_jpg and (time.perf_counter() - last_jpg_t) <= STALE_OK_SEC:
                        jpg = last_jpg
                    else:
                        time.sleep(0.02)
                        continue

                yield boundary
                yield b"Content-Type: image/jpeg\r\n"
                yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8")
                yield jpg
                yield b"\r\n"

        except GeneratorExit:
            return
        except Exception:
            return

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/api/live/dashboard_frame/{cam_id}/{group}")
def live_dashboard_frame(cam_id: str, group: str):
    cam_id = _live_id(cam_id)

    g = (group or "A").upper().strip()
    if g not in ("A", "B", "0"):
        g = "A"

    no_cache_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }

    p = pipelines_live.get(cam_id)
    if p is None:
        return Response(status_code=204, headers=no_cache_headers)

    try:
        jpg = None

        # Primary source
        if hasattr(p, "pull_dashboard_jpg"):
            jpg = p.pull_dashboard_jpg(g)

        # Fallback 1: fisheye clean grid
        if not jpg:
            if bool(getattr(p, "is_fisheye", False)):
                gg = "B" if g == "B" else "A"
                if hasattr(p, "pull_group_grid_jpg_clean"):
                    jpg = p.pull_group_grid_jpg_clean(gg)

        # Fallback 2: normal single clean frame
        if not jpg:
            if not bool(getattr(p, "is_fisheye", False)):
                if hasattr(p, "pull_single_view_jpg_clean"):
                    jpg = p.pull_single_view_jpg_clean(0)

        # Still no real frame -> tell frontend this frame is unavailable
        if not jpg:
            return Response(status_code=204, headers=no_cache_headers)

        return Response(
            content=jpg,
            media_type="image/jpeg",
            headers=no_cache_headers,
        )

    except Exception as e:
        print(f"[dashboard_frame] cam={cam_id} group={g} failed: {e}")
        return Response(status_code=204, headers=no_cache_headers)
    

@app.post("/api/settings/clear_static_cache")
def clear_static_cache():
    """Clear static frame cache when switching to live view"""
    with _static_lock:
        _static_frame_cache.clear()
        _static_group_cache.clear()
    return {"ok": True}


# Add near the top with other global variables
_detection_toggle_lock = threading.Lock()
_detection_toggle_state: Dict[str, bool] = {}  # cam_id -> detection_enabled
DETECTION_CONFIG_PATH = OUTPUTS_LF_DIR / "_detection_config.json"


def load_detection_config() -> Dict[str, bool]:
    """Load per-camera detection toggle state from disk"""
    with _detection_toggle_lock:
        if DETECTION_CONFIG_PATH.exists():
            try:
                config = json.loads(DETECTION_CONFIG_PATH.read_text(encoding="utf-8"))
                # Ensure we have a dict
                if isinstance(config, dict):
                    return config
            except Exception as e:
                _system(f"Error loading detection config: {e}")
        return {}


def save_detection_config(config: Dict[str, bool]) -> None:
    """Save per-camera detection toggle state to disk"""
    with _detection_toggle_lock:
        try:
            DETECTION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            DETECTION_CONFIG_PATH.write_text(
                json.dumps(config, indent=2, sort_keys=True),
                encoding="utf-8"
            )
            _system(f"Detection config saved: {len(config)} cameras")
        except Exception as e:
            _system(f"Error saving detection config: {e}")


def get_detection_state(cam_id: str = None) -> Union[Dict[str, bool], bool]:
    """
    Get detection state for all cameras or a specific camera
    Returns True by default if no config exists
    """
    config = load_detection_config()
    if cam_id:
        return config.get(cam_id, True)  # Default to True (detection enabled)
    return config


def set_detection_state(cam_id: str, enabled: bool) -> Dict[str, bool]:
    """
    Set detection state for a specific camera
    Returns the updated full config
    """
    cam_id = _live_id(cam_id)
    config = load_detection_config()
    config[cam_id] = enabled
    save_detection_config(config)

    # Update live pipeline if running
    p = pipelines_live.get(cam_id)
    if p and hasattr(p, "set_detection_enabled"):
        try:
            p.set_detection_enabled(enabled)
            _system(f"Detection toggled for {cam_id}: {enabled}")
        except Exception as e:
            _system(f"Failed to toggle detection for {cam_id}: {e}")

    return config


def init_detection_config_from_pipelines() -> None:
    """Initialize detection config from existing pipelines on startup"""
    config = load_detection_config()
    modified = False

    for cam_id in pipelines_live.keys():
        if cam_id not in config:
            config[cam_id] = True  # Default to enabled
            modified = True

    if modified:
        save_detection_config(config)
        _system(f"Initialized detection config for {len(pipelines_live)} cameras")


# Add these API endpoints
@app.get("/api/live/detection/state")
def api_get_detection_state():
    """Get current detection toggle state for all cameras"""
    try:
        config = load_detection_config()
        return JSONResponse(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live/detection/state/{cam_id}")
def api_get_detection_state_cam(cam_id: str):
    """Get detection state for a specific camera"""
    cam_id = _live_id(cam_id)
    try:
        enabled = get_detection_state(cam_id)
        return {"cam_id": cam_id, "detection_enabled": enabled}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/live/detection/toggle/{cam_id}")
def api_toggle_detection(cam_id: str, payload: Dict[str, Any] = Body(...)):
    """Toggle detection on/off for a specific camera"""
    cam_id = _live_id(cam_id)
    enabled = payload.get("enabled", True)

    try:
        config = set_detection_state(cam_id, enabled)
        return {
            "ok": True,
            "cam_id": cam_id,
            "detection_enabled": enabled,
            "all_states": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/live/detection/toggle_all")
def api_toggle_detection_all(payload: Dict[str, Any] = Body(...)):
    """Toggle detection for all cameras at once"""
    enabled = payload.get("enabled", True)

    try:
        config = {}
        for cam_id in pipelines_live.keys():
            config[cam_id] = enabled
            p = pipelines_live.get(cam_id)
            if p and hasattr(p, "set_detection_enabled"):
                try:
                    p.set_detection_enabled(enabled)
                except Exception:
                    pass

        save_detection_config(config)

        return {
            "ok": True,
            "detection_enabled": enabled,
            "cameras_updated": list(pipelines_live.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Update the shutdown handler to save detection config
@app.on_event("shutdown")
def shutdown_handler():
    """Save all states on shutdown"""
    _system("Shutdown: Saving detection config...")

    # Save detection config
    config = load_detection_config()
    save_detection_config(config)

    # Save live items
    persist_live_items_on_shutdown()

    _system("Shutdown complete")


@app.get("/api/debug/videotype/cache")
def debug_video_type_cache():
    """Debug endpoint to see video type cache stats"""
    with _video_type_cache_lock:
        cache_stats = {}
        for stem, data in _video_type_cache.items():
            cache_stats[stem] = {
                "type": data.get("type"),
                "age_seconds": time.time() - data.get("timestamp", 0),
                "hits": data.get("hits", 0)
            }
        return {
            "cache_size": len(_video_type_cache),
            "cache_ttl": VIDEO_TYPE_CACHE_TTL,
            "entries": cache_stats
        }


# ============================================================
# UI Focus (active page / camera)  ✅ add this
# ============================================================
_UI_FOCUS_LOCK = threading.Lock()
_UI_FOCUS_TTL_SEC = 8.0  # if frontend stops sending, focus expires

_UI_FOCUS_STATE: Dict[str, Any] = {
    "page": "none",  # "dashboard"|"live"|"events"|"offline"|"settings"|"none"
    "active_id": None,  # cam_id or stem
    "view": None,  # e.g. "A"/"B"/"0"/"grid"/"view"
    "ts": 0.0,
}


class UIFocusIn(BaseModel):
    page: str = "none"
    active_id: Optional[str] = None
    view: Optional[str] = None


def set_ui_focus(page: str, active_id: Optional[str], view: Optional[str]) -> None:
    now = time.time()
    with _UI_FOCUS_LOCK:
        _UI_FOCUS_STATE["page"] = str(page or "none")
        _UI_FOCUS_STATE["active_id"] = active_id
        _UI_FOCUS_STATE["view"] = view
        _UI_FOCUS_STATE["ts"] = now


def get_ui_focus() -> Dict[str, Any]:
    with _UI_FOCUS_LOCK:
        return dict(_UI_FOCUS_STATE)


def focus_alive() -> bool:
    st = get_ui_focus()
    return (time.time() - float(st.get("ts", 0.0))) <= _UI_FOCUS_TTL_SEC


def focus_is(page: Optional[str] = None,
             active_id: Optional[str] = None,
             view: Optional[str] = None) -> bool:
    """
    True only if focus is fresh and matches provided filters.
    """
    st = get_ui_focus()
    if (time.time() - float(st.get("ts", 0.0))) > _UI_FOCUS_TTL_SEC:
        return False
    if page is not None and st.get("page") != page:
        return False
    if active_id is not None and st.get("active_id") != active_id:
        return False
    if view is not None and st.get("view") != view:
        return False
    return True


# ----------------------------
# Focus endpoints
# ----------------------------
@app.post("/api/ui/focus")
def api_set_ui_focus(payload: UIFocusIn):
    # Normalize ids same as your system
    page = (payload.page or "none").strip().lower()
    active_id = payload.active_id
    if isinstance(active_id, str):
        active_id = active_id.strip()
        # if it's a cam id, normalize
        if page in ("live", "dashboard", "settings"):
            active_id = _base_id(active_id)

    view = payload.view
    if isinstance(view, str):
        view = view.strip().upper()

    set_ui_focus(page, active_id, view)
    return {"ok": True, "focus": get_ui_focus(), "ttl_sec": _UI_FOCUS_TTL_SEC}


@app.get("/api/ui/focus")
def api_get_ui_focus():
    return {"ok": True, "focus": get_ui_focus(), "ttl_sec": _UI_FOCUS_TTL_SEC}


# ==================================
# LIVE SOURCES ENABLE/DISABLE (UPLOAD)
# ==================================
LF_CAMERAS_ENABLED_PATH = OUTPUTS_LF_DIR / "_cameras_enabled.json"
LF_CAMERAS_ENABLED_PATH.parent.mkdir(parents=True, exist_ok=True)


def _upload_cam_id_from_file(f: Path) -> str:
    """
    Canonical upload cam_id used everywhere:
    - stable
    - matches _cameras_enabled.json keys
    """
    return str(_base_id(_cam_id_from_h264_file(f))).strip()


def load_cameras_enabled() -> Dict[str, bool]:
    """
    Load enabled/disabled map for upload cameras.

    ✅ Seeds missing upload videos (default True)
    ✅ Prunes stale keys not in current upload folder
       (so your list matches your REAL upload videos count)
    """
    enabled_map: Dict[str, bool] = {}

    # 1) Load existing file
    try:
        if LF_CAMERAS_ENABLED_PATH.exists():
            raw = json.loads(LF_CAMERAS_ENABLED_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # normalize keys to canonical base id
                enabled_map = {str(_base_id(k)).strip(): bool(v) for k, v in raw.items()}
    except Exception as e:
        _system(f"LIVE: load_cameras_enabled error: {e}")

    # 2) Build current upload cam_id set
    current_ids: Set[str] = set()
    try:
        files = list_upload_videos_h264_only()
        for f in files:
            cid = _upload_cam_id_from_file(f)
            if cid:
                current_ids.add(cid)
    except Exception as e:
        _system(f"LIVE: load_cameras_enabled scan error: {e}")
        # If scan fails, just return what we loaded
        return enabled_map

    # 3) Seed missing ids (default True)
    changed = False
    for cid in current_ids:
        if cid not in enabled_map:
            enabled_map[cid] = True
            changed = True

    # 4) PRUNE keys not in upload folder (so it matches your 6 videos)
    stale = [k for k in enabled_map.keys() if k not in current_ids]
    if stale:
        for k in stale:
            enabled_map.pop(k, None)
        changed = True

    if changed:
        save_cameras_enabled(enabled_map)

    return enabled_map


def save_cameras_enabled(data: Dict[str, bool]) -> Dict[str, bool]:
    """
    Save normalized enabled map (canonical keys only).
    """
    try:
        norm = {str(_base_id(k)).strip(): bool(v) for k, v in (data or {}).items()}
        LF_CAMERAS_ENABLED_PATH.write_text(json.dumps(norm, indent=2), encoding="utf-8")
    except Exception as e:
        _system(f"LIVE: save_cameras_enabled error: {e}")
    return data


@app.get("/api/lostfound/cameras_enabled")
def get_cameras_enabled():
    """
    Frontend helper endpoint
    """
    return {"cameras_enabled": load_cameras_enabled()}


@app.get("/api/lostfound/upload_sources")
def get_upload_sources_for_settings(request: Request):
    """
    ✅ Settings page: list ALL upload videos (exactly what is in upload folder),
    with enabled flag from _cameras_enabled.json.

    This will match your real upload videos count (e.g. 6).
    """
    enabled_map = load_cameras_enabled() or {}
    files = list_upload_videos_h264_only()

    out = []
    for f in files:
        cam_id = _upload_cam_id_from_file(f)
        out.append({
            "id": cam_id,
            "name": cam_name_from_file(f),
            "filename": f.name,
            "enabled": bool(enabled_map.get(cam_id, True)),
        })

    out.sort(key=lambda x: (x["name"], x["id"]))
    return {"sources": out}


@app.post("/api/lostfound/cameras_enabled/toggle/{cam_id}")
def toggle_camera_enabled_api(cam_id: str, payload: Dict[str, Any] = Body(...)):
    cam_id = str(_base_id(cam_id)).strip()

    data = load_cameras_enabled()
    enabled = bool(payload.get("enabled", True))
    data[cam_id] = enabled
    save_cameras_enabled(data)

    if enabled:
        ok = False
        try:
            ok = restart_single_live_camera(cam_id)
        except Exception:
            ok = False

        # force detection re-apply after restart
        try:
            p = pipelines_live.get(cam_id)
            if p and hasattr(p, "set_detection_enabled"):
                det_cfg = load_detection_config()
                p.set_detection_enabled(bool(det_cfg.get(cam_id, True)))
        except Exception:
            pass

        return {"ok": ok, "cam_id": cam_id, "enabled": enabled}
    else:
        p = pipelines_live.get(cam_id)
        if p and hasattr(p, "stop"):
            try:
                p.stop()
            except Exception:
                pass
        pipelines_live.pop(cam_id, None)
        return {"ok": True, "cam_id": cam_id, "enabled": enabled}


# ============================================================
# RTSP Store
# ============================================================
RTSP_STORE_PATH = OUTPUTS_LF_DIR / "rtsp_sources.json"
RTSP_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_rtsp_sources() -> Dict[str, Any]:
    try:
        if RTSP_STORE_PATH.exists():
            return json.loads(RTSP_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_rtsp_sources(data: Dict[str, Any]) -> Dict[str, Any]:
    RTSP_STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# ============================================================
# RTSP APIs
# ============================================================
@app.get("/api/rtsp")
@app.get("/api/lostfound/rtsp_sources")
def get_rtsp_sources():
    return load_rtsp_sources()


@app.post("/api/rtsp")
@app.post("/api/lostfound/rtsp_sources")
def add_or_update_rtsp(payload: Dict[str, Any] = Body(...)):
    data = load_rtsp_sources()

    cam_id = (payload.get("id") or "").strip()
    if not cam_id:
        raise HTTPException(status_code=400, detail="Missing camera id")

    raw_url = (payload.get("url") or "").strip()
    fixed_url = _encode_rtsp_credentials(raw_url)  # ✅ AUTO FIX HERE

    data[cam_id] = {
        "id": cam_id,
        "name": (payload.get("name") or cam_id).strip(),
        "url": fixed_url,
        "enabled": bool(payload.get("enabled", True)),

        # ✅ NEW (optional): "auto" | "normal" | "fisheye"
        "video_type": (payload.get("video_type") or "auto").strip().lower(),
    }

    save_rtsp_sources(data)
    return {"ok": True, "saved_url": fixed_url}


@app.post("/api/rtsp/toggle/{cam_id}")
@app.post("/api/lostfound/rtsp_sources/toggle/{cam_id}")
def toggle_rtsp(cam_id: str, payload: Dict[str, Any] = Body(...)):
    cam_id = (cam_id or "").strip()  # ⚠️ RTSP ids must NOT be base_id
    data = load_rtsp_sources()
    if cam_id not in data:
        raise HTTPException(status_code=404, detail="Camera not found")

    enabled = bool(payload.get("enabled", True))
    data[cam_id]["enabled"] = enabled
    save_rtsp_sources(data)

    # ✅ IMMEDIATE APPLY
    if enabled:
        restart_single_live_camera(cam_id)
    else:
        p = pipelines_live.get(cam_id)
        if p and hasattr(p, "stop"):
            try:
                p.stop()
            except Exception:
                pass
        pipelines_live.pop(cam_id, None)

    return {"ok": True, "cam_id": cam_id, "enabled": enabled}


@app.delete("/api/rtsp/{cam_id}")
@app.delete("/api/lostfound/rtsp_sources/{cam_id}")
def delete_rtsp(cam_id: str):
    cam_id = (cam_id or "").strip()
    data = load_rtsp_sources()
    data.pop(cam_id, None)
    save_rtsp_sources(data)

    # optional: if deleted, also stop live pipeline immediately
    p = pipelines_live.get(cam_id)
    if p and hasattr(p, "stop"):
        try:
            p.stop()
        except Exception:
            pass
    pipelines_live.pop(cam_id, None)

    return {"ok": True}


def _encode_rtsp_credentials(url: str) -> str:
    """
    Ensures username/password are URL-encoded safely (handles @ in password).
    Also prevents double-encoding if user already pasted %40.
    """
    u = (url or "").strip()
    if not u:
        return u
    if not (u.lower().startswith("rtsp://") or u.lower().startswith("rtsps://")):
        return u

    try:
        sp = urlsplit(u)
        netloc = sp.netloc
        if "@" not in netloc:
            return u

        creds, hostpart = netloc.rsplit("@", 1)  # split at LAST @
        if ":" in creds:
            user, pwd = creds.split(":", 1)

            # ✅ decode first to avoid double-encoding
            user_dec = unquote(user)
            pwd_dec = unquote(pwd)

            user_enc = quote(user_dec, safe="")
            pwd_enc = quote(pwd_dec, safe="")
            new_netloc = f"{user_enc}:{pwd_enc}@{hostpart}"
        else:
            user_dec = unquote(creds)
            user_enc = quote(user_dec, safe="")
            new_netloc = f"{user_enc}@{hostpart}"

        return urlunsplit((sp.scheme, new_netloc, sp.path, sp.query, sp.fragment))
    except Exception:
        return u


def _rtsp_store_get(rtsp_store: Any, cam_id: str) -> Optional[Dict[str, Any]]:
    """
    Your load_rtsp_sources() sometimes returns:
      - dict of {id: {...}}
      - or { "sources": [...] }
    This helper supports both.
    """
    if not rtsp_store:
        return None

    # Case 1: dict keyed by id
    if isinstance(rtsp_store, dict):
        # direct hit
        rec = rtsp_store.get(cam_id)
        if isinstance(rec, dict):
            return rec

        # case 2: dict wrapper with "sources": [...]
        srcs = rtsp_store.get("sources")
        if isinstance(srcs, list):
            for s in srcs:
                if isinstance(s, dict) and str(s.get("id")) == str(cam_id):
                    return s
        return None

    # Case 3: list of sources
    if isinstance(rtsp_store, list):
        for s in rtsp_store:
            if isinstance(s, dict) and str(s.get("id")) == str(cam_id):
                return s

    return None


def resolve_live_source(cam_id: str) -> Optional[str]:
    """
    Prefer enabled RTSP source, else fall back to uploaded file.
    Returns a string usable by cv2.VideoCapture / pipeline.
    """
    raw_id = (cam_id or "").strip()
    if not raw_id:
        return None

    is_rtsp_id = raw_id.startswith("rtsp_")

    # 1) RTSP
    try:
        rtsp_store = load_rtsp_sources() or {}
        lookup_id = raw_id if is_rtsp_id else _base_id(raw_id)
        rec = _rtsp_store_get(rtsp_store, lookup_id)

        if isinstance(rec, dict):
            enabled = bool(rec.get("enabled", True))
            url = (rec.get("url") or "").strip()
            if enabled and url:
                return _encode_rtsp_credentials(url)

        elif isinstance(rec, str) and rec.strip():
            return _encode_rtsp_credentials(rec.strip())

    except Exception as e:
        _system(f"LIVE: resolve_live_source RTSP read error for {raw_id}: {e}")

    # 2) Upload fallback
    try:
        stem = _base_id(raw_id)
        src = find_upload_by_stem(stem)
        if src and src.exists():
            return str(src)
    except Exception as e:
        _system(f"LIVE: resolve_live_source upload fallback error for {raw_id}: {e}")

    return None


def resolve_settings_video_type(cam_id: str, src_str: Optional[str] = None) -> str:
    """
    Settings-page-only video type resolver.
    Returns: 'fisheye' or 'normal'
    """
    raw_id = (cam_id or "").strip()
    if not raw_id:
        return "normal"

    base_id = _base_id(raw_id)

    if not src_str:
        src_str = resolve_live_source(raw_id) or ""
    src_str = str(src_str).strip()

    # 1) manual override from settings
    try:
        st = load_lf_settings() or {}
        vt_map = st.get("camera_video_types", {}) or {}

        forced = (
                vt_map.get(raw_id)
                or vt_map.get(base_id)
                or "auto"
        )
        forced = str(forced).strip().lower()

        if forced in ("fisheye", "normal"):
            return forced
    except Exception:
        pass

    # 2) running live pipeline state
    try:
        p = (
                pipelines_live.get(raw_id)
                or pipelines_live.get(base_id)
                or pipelines_live.get(_live_id(raw_id))
        )
        if p is not None:
            return "fisheye" if bool(getattr(p, "is_fisheye", False)) else "normal"
    except Exception:
        pass

    is_rtsp = src_str.lower().startswith(("rtsp://", "rtsps://"))

    # 3) RTSP saved type
    if is_rtsp:
        try:
            rtsp_store = load_rtsp_sources() or {}
            rec = _rtsp_store_get(rtsp_store, raw_id if raw_id.startswith("rtsp_") else base_id)

            if isinstance(rec, dict):
                vt = str(
                    rec.get("video_type")
                    or rec.get("type")
                    or "auto"
                ).strip().lower()

                if vt in ("fisheye", "normal"):
                    return vt
        except Exception:
            pass

    # 4) upload fallback by filename/source
    if not is_rtsp:
        try:
            psrc = Path(src_str)
            stem = psrc.stem if src_str else base_id

            st = load_lf_settings() or {}
            vt_map = st.get("camera_video_types", {}) or {}

            forced2 = (
                    vt_map.get(stem)
                    or vt_map.get(base_id)
                    or vt_map.get(raw_id)
                    or "auto"
            )
            forced2 = str(forced2).strip().lower()

            if forced2 in ("fisheye", "normal"):
                return forced2
        except Exception:
            pass

    # 5) fallback auto detect
    try:
        if is_rtsp:
            return detect_rtsp_video_type(raw_id, src_str)
        else:
            return detect_video_type_cached(base_id, Path(src_str))
    except Exception:
        return "normal"


@app.post("/api/live/rtsp/{cam_id}")
def set_rtsp(cam_id: str, payload: Dict[str, Any] = Body(...)):
    cam_id = _base_id(cam_id)
    url = (payload.get("rtsp") or "").strip()

    # ✅ Frontend passes "type": "fisheye" or "normal".
    # Defaults to "fisheye" because that's what this system is built for.
    vtype = (payload.get("type") or "fisheye").strip().lower()
    if vtype not in ("fisheye", "normal"):
        vtype = "fisheye"

    m = load_rtsp_sources()
    if url:
        # ✅ Store as dict so get_live_source and detect_video_type_cached can both use it
        m[cam_id] = {"url": url, "type": vtype}
    else:
        # empty URL = remove RTSP mapping, fall back to upload file
        if cam_id in m:
            del m[cam_id]

    save_rtsp_sources(m)

    # Invalidate video type cache so next call re-reads from RTSP store, not stale file probe
    invalidate_video_type_cache(cam_id)

    # Restart this camera's pipeline so it immediately connects to the new RTSP source
    restart_single_live_camera(cam_id)
    return {"ok": True, "cam_id": cam_id, "rtsp": m.get(cam_id)}


# ============================================================
# RTSP Video Type Detection (cached)
# ============================================================
_rtsp_type_cache_lock = threading.Lock()
_rtsp_type_cache: Dict[str, Dict[str, Any]] = {}  # cam_id -> {"url":..., "type":..., "ts":...}


def get_live_source(cam_id: str) -> Optional[str]:
    cam_id = _base_id(cam_id)
    m = load_rtsp_sources()
    entry = m.get(cam_id)

    # ✅ Support new dict format {"url": "rtsp://...", "type": "fisheye"}
    if isinstance(entry, dict):
        url = (entry.get("url") or "").strip()
        if url:
            return url
    # ✅ Support old plain-string format (backward compat for existing saved files)
    elif isinstance(entry, str) and entry.strip():
        return entry.strip()

    # Fall back to uploaded file on disk
    src = find_upload_by_stem(cam_id)
    return str(src) if src else None


def detect_rtsp_video_type(cam_id: str, url: str, ttl_sec: float = 300.0) -> str:
    """
    Returns 'normal' or 'fisheye' for RTSP.
    Cached.
    Conservative default = normal.
    """
    cam_id = (cam_id or "").strip()
    url = (url or "").strip()
    if not cam_id or not url:
        return "normal"

    now = time.time()
    with _rtsp_type_cache_lock:
        rec = _rtsp_type_cache.get(cam_id)
        if rec and rec.get("url") == url and (now - float(rec.get("ts", 0))) < ttl_sec:
            return rec.get("type", "normal")

    t = "normal"
    try:
        # ✅ use lostandfound module function
        if hasattr(lf, "detect_video_type"):
            try:
                vv = lf.detect_video_type(url)
            except Exception:
                vv = "normal"
            s = str(vv).lower()
            if "fish" in s:
                t = "fisheye"
    except Exception:
        t = "normal"

    with _rtsp_type_cache_lock:
        _rtsp_type_cache[cam_id] = {"url": url, "type": t, "ts": now}

    return t


_live_rtsp_pre_cache: Dict[str, Any] = {}
_live_rtsp_pre_lock = threading.Lock()


def _get_or_make_live_fisheye_pre_rtsp(cam_id: str, src: Union[str, Path]):
    """
    Build (or return cached) lf.FisheyePreprocessor for this cam.

    ✅ src can be:
      - a file Path  → existing behaviour, use st_mtime as cache key
      - an rtsp:// string → skip stat(), use URL string as cache key

    The lf.FisheyePreprocessor.open(video_path) accepts both because it
    calls cv2.VideoCapture(video_path) internally — same as open_source_capture().
    """
    if not hasattr(lf, "FisheyePreprocessor"):
        raise RuntimeError("lostandfound.py missing FisheyePreprocessor")
    if not FISHEYE_VIEW_CONFIGS:
        raise RuntimeError("lostandfound.py missing FISHEYE_VIEW_CONFIGS")

    src_str = str(src)
    is_rtsp = src_str.startswith("rtsp://") or src_str.startswith("rtsps://")

    # Cache key: for files use path+mtime, for RTSP use URL (never changes mtime)
    cache_key = src_str
    if is_rtsp:
        mtime = 0.0  # RTSP has no mtime
    else:
        try:
            mtime = float(Path(src_str).stat().st_mtime)
        except Exception:
            mtime = 0.0

    with _live_pre_lock:
        rec = _live_pre_cache.get(cam_id)
        if rec and rec.get("path") == cache_key and float(rec.get("mtime") or 0) == mtime:
            return rec["pre"]

        # ✅ Build FisheyePreprocessor the same way lostandfound.create_preprocessor() does
        pre = lf.FisheyePreprocessor(
            view_configs=get_fisheye_view_configs(cam_id),
            output_size=(480, 640),
            input_fov_deg=180.0,
            logger=None,
        )

        pre.open(src_str)

        _live_pre_cache[cam_id] = {"pre": pre, "path": cache_key, "mtime": mtime}
        return pre


def _get_rtsp_entry(cam_id: str) -> Optional[Dict[str, str]]:
    """
    Returns {"url": "rtsp://...", "type": "fisheye"|"normal"} if cam_id has an RTSP mapping.
    Returns None if cam_id uses an upload file (not RTSP).
    Supports both old plain-string and new dict formats in the store.
    """
    cam_id = _base_id(cam_id)
    m = load_rtsp_sources()
    entry = m.get(cam_id)

    if isinstance(entry, dict):
        url = (entry.get("url") or "").strip()
        vtype = (entry.get("type") or "fisheye").strip().lower()
        if url:
            return {"url": url, "type": vtype}

    elif isinstance(entry, str) and entry.strip():
        # Old format had no stored type — default to fisheye
        return {"url": entry.strip(), "type": "fisheye"}

    return None  # not an RTSP cam


# ============================================================
# FISHEYE VIEW CONFIG STORE (yaw/pitch/fov/rotate per camera)
# ============================================================
import threading
import time
import json
import cv2
from typing import Dict, Any
from fastapi import Body, HTTPException, Query
from fastapi.responses import Response

FISHEYE_CFG_PATH = OUTPUTS_LF_DIR / "fisheye_view_configs.json"
FISHEYE_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)

_fisheye_preview_lock_map: Dict[str, threading.Lock] = {}
_fisheye_preview_lock_map_guard = threading.Lock()
_fisheye_preview_last_ts: Dict[str, float] = {}
_fisheye_preview_min_gap_sec = 0.10


def _get_fisheye_preview_lock(cam_id: str) -> threading.Lock:
    cam_id = (cam_id or "").strip()
    with _fisheye_preview_lock_map_guard:
        lk = _fisheye_preview_lock_map.get(cam_id)
        if lk is None:
            lk = threading.Lock()
            _fisheye_preview_lock_map[cam_id] = lk
        return lk


def _safe_float(v, default=0.0):
    try:
        if v is None:
            return float(default)
        if isinstance(v, str) and not v.strip():
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0):
    try:
        if v is None:
            return int(default)
        if isinstance(v, str) and not v.strip():
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _default_fisheye_configs() -> list:
    return json.loads(json.dumps(FISHEYE_VIEW_CONFIGS))


def load_fisheye_view_configs_all() -> Dict[str, Any]:
    try:
        if FISHEYE_CFG_PATH.exists():
            return json.loads(FISHEYE_CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_fisheye_view_configs_all(data: Dict[str, Any]) -> None:
    FISHEYE_CFG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_fisheye_view_configs(cam_id: str) -> list:
    cam_id = (cam_id or "").strip()
    if not cam_id:
        return _default_fisheye_configs()

    allcfg = load_fisheye_view_configs_all()
    cfg = allcfg.get(cam_id)

    if isinstance(cfg, list) and len(cfg) >= 1:
        out = []
        for i, rec in enumerate(cfg):
            if not isinstance(rec, dict):
                continue
            out.append({
                "view_id": _safe_int(rec.get("view_id", i), i),
                "name": str(rec.get("name", f"view_{i}")),
                "yaw": _safe_float(rec.get("yaw", 0.0), 0.0),
                "pitch": _safe_float(rec.get("pitch", 0.0), 0.0),
                "fov": _safe_float(rec.get("fov", 90.0), 90.0),
                "rotate": _safe_int(rec.get("rotate", 0), 0),
            })
        if out:
            return out

    return _default_fisheye_configs()


def set_fisheye_view_configs(cam_id: str, cfg_list: list) -> None:
    cam_id = (cam_id or "").strip()
    if not cam_id:
        raise ValueError("cam_id required")

    if not isinstance(cfg_list, list) or len(cfg_list) == 0:
        raise ValueError("cfg_list must be a list")

    out = []
    for i, rec in enumerate(cfg_list):
        if not isinstance(rec, dict):
            continue

        out.append({
            "view_id": _safe_int(rec.get("view_id", i), i),
            "name": str(rec.get("name", f"view_{i}")),
            "yaw": _safe_float(rec.get("yaw", 0.0), 0.0),
            "pitch": _safe_float(rec.get("pitch", 0.0), 0.0),
            "fov": _safe_float(rec.get("fov", 90.0), 90.0),
            "rotate": _safe_int(rec.get("rotate", 0), 0),
        })

    if not out:
        raise ValueError("no valid config records")

    allcfg = load_fisheye_view_configs_all()
    allcfg[cam_id] = out
    save_fisheye_view_configs_all(allcfg)


def reset_fisheye_view_configs(cam_id: str) -> None:
    cam_id = (cam_id or "").strip()
    if not cam_id:
        raise ValueError("cam_id required")

    allcfg = load_fisheye_view_configs_all()
    if cam_id in allcfg:
        del allcfg[cam_id]
        save_fisheye_view_configs_all(allcfg)


# ============================================================
# FISHEYE CONFIG APIs
# ============================================================
@app.get("/api/lostfound/fisheye_configs/{cam_id}")
def api_get_fisheye_config(cam_id: str):
    cam_id = (cam_id or "").strip()
    return {"cam_id": cam_id, "configs": get_fisheye_view_configs(cam_id)}


@app.post("/api/lostfound/fisheye_configs/{cam_id}")
def api_set_fisheye_config(cam_id: str, payload: Dict[str, Any] = Body(...)):
    cam_id = (cam_id or "").strip()
    cfg = payload.get("configs")

    if not isinstance(cfg, list):
        raise HTTPException(status_code=400, detail="configs must be a list")

    try:
        set_fisheye_view_configs(cam_id, cfg)
    except Exception as e:
        print(f"[ERROR] fisheye config save failed cam={cam_id}: {e}")
        print(f"[ERROR] payload={payload}")
        raise HTTPException(status_code=400, detail=f"invalid fisheye config: {e}")

    # clear fallback/static preprocessor caches
    try:
        with _live_pre_lock:
            _live_pre_cache.pop(cam_id, None)
    except Exception:
        pass

    try:
        with _live_rtsp_pre_lock:
            _live_rtsp_pre_cache.pop(cam_id, None)
    except Exception:
        pass

    # fast apply in running pipeline (do NOT restart while editing)
    try:
        p = pipelines_live.get(cam_id)
        if p is not None:
            setattr(p, "_fisheye_cfg_dirty", True)
            if hasattr(p, "reload_fisheye_config"):
                p.reload_fisheye_config()
    except Exception as e:
        print(f"[WARN] fisheye live apply failed cam={cam_id}: {e}")

    return {"ok": True, "cam_id": cam_id, "applied": True}


@app.post("/api/lostfound/fisheye_configs/{cam_id}/reset")
def api_reset_fisheye_config(cam_id: str):
    cam_id = (cam_id or "").strip()

    try:
        reset_fisheye_view_configs(cam_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        with _live_pre_lock:
            _live_pre_cache.pop(cam_id, None)
    except Exception:
        pass

    try:
        with _live_rtsp_pre_lock:
            _live_rtsp_pre_cache.pop(cam_id, None)
    except Exception:
        pass

    try:
        p = pipelines_live.get(cam_id)
        if p is not None:
            setattr(p, "_fisheye_cfg_dirty", True)
            if hasattr(p, "reload_fisheye_config"):
                p.reload_fisheye_config()
    except Exception as e:
        print(f"[WARN] fisheye reset live apply failed cam={cam_id}: {e}")

    return {"ok": True, "cam_id": cam_id, "reset": True, "applied": True}


@app.get("/api/lostfound/fisheye_preview/{cam_id}/{view_idx}")
def api_fisheye_preview(cam_id: str, view_idx: int, refresh: int = Query(1)):
    cam_id = (cam_id or "").strip()
    if not cam_id:
        raise HTTPException(status_code=400, detail="cam_id required")

    src = get_live_source(cam_id)
    if not src:
        raise HTTPException(status_code=404, detail="live source not found")

    try:
        cfgs = get_fisheye_view_configs(cam_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load fisheye config failed: {e}")

    if not isinstance(cfgs, list) or len(cfgs) == 0:
        raise HTTPException(status_code=500, detail="fisheye config empty")

    cfgs_sorted = sorted(
        [c for c in cfgs if isinstance(c, dict)],
        key=lambda x: int(x.get("view_id", 0))
    )

    wanted_view_id = max(0, min(int(view_idx), 7))

    wanted_cfg = None
    for rec in cfgs_sorted:
        try:
            if int(rec.get("view_id", -1)) == wanted_view_id:
                wanted_cfg = rec
                break
        except Exception:
            pass

    if wanted_cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"view config missing for view_id={wanted_view_id}"
        )

    wanted_name = str(wanted_cfg.get("name") or f"view_{wanted_view_id}")

    lock = _get_fisheye_preview_lock(cam_id)
    acquired = lock.acquire(timeout=1.5)
    if not acquired:
        raise HTTPException(status_code=429, detail="preview busy, try again")

    try:
        # small throttle so rapid typing doesn't hammer preview too hard
        now = time.time()
        last_ts = _fisheye_preview_last_ts.get(cam_id, 0.0)
        dt = now - last_ts
        if dt < _fisheye_preview_min_gap_sec:
            time.sleep(_fisheye_preview_min_gap_sec - dt)
        _fisheye_preview_last_ts[cam_id] = time.time()

        # ==================================================
        # FAST PATH: use running pipeline + cached raw frame
        # ==================================================
        p = pipelines_live.get(cam_id)
        if p is not None and getattr(p, "is_fisheye", False):
            try:
                setattr(p, "_fisheye_cfg_dirty", True)
                if hasattr(p, "reload_fisheye_config"):
                    p.reload_fisheye_config()
            except Exception as e:
                print(f"[WARN] fisheye preview reload failed cam={cam_id}: {e}")

            raw_fr = None
            try:
                if hasattr(p, "_get_latest_raw_frame_copy"):
                    raw_fr = p._get_latest_raw_frame_copy()
            except Exception:
                raw_fr = None

            pre = getattr(p, "preprocessor", None)

            if raw_fr is not None and pre is not None:
                try:
                    try:
                        views = pre.get_views(raw_fr, allowed_names=[wanted_name])
                    except TypeError:
                        views = pre.get_views(raw_fr)

                    chosen = None

                    for v in (views or []):
                        if (v.get("name") or "") == wanted_name:
                            chosen = v
                            break

                    if chosen is None:
                        for v in (views or []):
                            try:
                                if int(v.get("view_id", -1)) == wanted_view_id:
                                    chosen = v
                                    break
                            except Exception:
                                pass

                    if chosen is not None and chosen.get("image") is not None:
                        img = chosen["image"].copy()
                        img = _draw_label_bar(
                            img,
                            f"{wanted_view_id} {wanted_name.replace('_', ' ').title()}"
                        )
                        return Response(
                            content=_jpg_bytes(img, quality=85),
                            media_type="image/jpeg",
                            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
                        )
                except Exception as e:
                    print(f"[WARN] fisheye preview fast-path failed cam={cam_id}: {e}")

        # ==================================================
        # FALLBACK PATH: open source once if cache not ready
        # ==================================================
        pre = None
        cap = None
        try:
            pre = lf.FisheyePreprocessor(
                view_configs=cfgs_sorted,
                config_path=str(_ensure_roi_file("live", cam_id))
            )

            if not pre.open(str(src)):
                raise RuntimeError("cannot open fisheye source")

            cap = cv2.VideoCapture(str(src))
            fr = _read_any_frame(cap)
            if fr is None:
                raise HTTPException(status_code=404, detail="no frame")

            try:
                views = pre.get_views(fr, allowed_names=[wanted_name])
            except TypeError:
                views = pre.get_views(fr)

            chosen = None
            for v in (views or []):
                if (v.get("name") or "") == wanted_name:
                    chosen = v
                    break

            if chosen is None:
                for v in (views or []):
                    try:
                        if int(v.get("view_id", -1)) == wanted_view_id:
                            chosen = v
                            break
                    except Exception:
                        pass

            if chosen is None or chosen.get("image") is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"view missing: id={wanted_view_id}, name={wanted_name}"
                )

            img = chosen["image"].copy()
            img = _draw_label_bar(
                img,
                f"{wanted_view_id} {wanted_name.replace('_', ' ').title()}"
            )

            return Response(
                content=_jpg_bytes(img, quality=85),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
            )

        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                if pre is not None:
                    pre.release()
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fisheye preview failed: {e}")
    finally:
        lock.release()

# =========================================================
# Force-quit shutdown handler
# =========================================================
_FORCE_EXITING = False
_FORCE_EXIT_LOCK = threading.Lock()


def _safe_stop_pipeline_dict(pdict, join_timeout: float = 1.2):
    try:
        if not isinstance(pdict, dict):
            return
        for _, p in list(pdict.items()):
            try:
                if p is not None:
                    p.stop()
            except Exception:
                pass
        for _, p in list(pdict.items()):
            try:
                if p is not None:
                    p.join(timeout=join_timeout)
            except Exception:
                pass
    except Exception:
        pass


def _run_global_backend_cleanup():
    print("[SYSTEM] Shutdown cleanup started")

    try:
        if "pipelines_live" in globals() and isinstance(globals()["pipelines_live"], dict):
            print(f"[SYSTEM] Stopping pipelines_live ({len(globals()['pipelines_live'])})")
            _safe_stop_pipeline_dict(globals()["pipelines_live"])
            globals()["pipelines_live"].clear()
    except Exception as e:
        print(f"[WARN] pipelines_live cleanup failed: {e}")

    try:
        if "pipelines_settings" in globals() and isinstance(globals()["pipelines_settings"], dict):
            print(f"[SYSTEM] Stopping pipelines_settings ({len(globals()['pipelines_settings'])})")
            _safe_stop_pipeline_dict(globals()["pipelines_settings"])
            globals()["pipelines_settings"].clear()
    except Exception as e:
        print(f"[WARN] pipelines_settings cleanup failed: {e}")

    try:
        if "pipelines" in globals() and isinstance(globals()["pipelines"], dict):
            print(f"[SYSTEM] Stopping pipelines ({len(globals()['pipelines'])})")
            _safe_stop_pipeline_dict(globals()["pipelines"])
            globals()["pipelines"].clear()
    except Exception as e:
        print(f"[WARN] pipelines cleanup failed: {e}")

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("[SYSTEM] Shutdown cleanup finished")


@app.on_event("shutdown")
async def on_shutdown():
    print("[SYSTEM] FastAPI shutdown event triggered")
    try:
        _run_global_backend_cleanup()
    except Exception as e:
        print(f"[ERROR] Shutdown cleanup failed: {e}")


def install_force_quit_handler():
    def _handle_exit(sig, frame):
        global _FORCE_EXITING

        with _FORCE_EXIT_LOCK:
            if _FORCE_EXITING:
                print("\n[SYSTEM] Second Ctrl+C -> force kill")
                os._exit(1)
            _FORCE_EXITING = True

        print("\n[SYSTEM] Ctrl+C detected")
        print("[SYSTEM] Cleaning up pipelines... (Press Ctrl+C again to force quit)")

        try:
            _run_global_backend_cleanup()
        except Exception as e:
            print(f"[SYSTEM] Cleanup warning: {e}")

        print("[SYSTEM] Cleanup finished")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)


@app.get("/api/lostfound/upload_sources")
def get_upload_sources():
    enabled_map = load_cameras_enabled() or {}
    files = list_upload_videos_h264_only()

    out = []
    for f in files:
        cam_id = _upload_cam_id_from_file(f)

        try:
            vtype = resolve_settings_video_type(cam_id, str(f))
        except Exception:
            vtype = "normal"

        out.append({
            "id": cam_id,
            "name": cam_name_from_file(f),
            "filename": f.name,
            "enabled": bool(enabled_map.get(cam_id, True)),
            "video_type": vtype,
            "is_fisheye": (vtype == "fisheye"),
            "views_count": 8 if vtype == "fisheye" else 1,
        })

    out.sort(key=lambda x: (x["name"], x["id"]))
    return {"sources": out}

    
VIEW_MODE_OVERRIDE_LOCK = threading.Lock()
VIEW_MODE_OVERRIDE_PATH = OUTPUTS_LF_DIR / "live_view_mode_overrides.json"
VALID_VIEW_MODES = {"auto", "fisheye", "normal"}
LIVE_ACTIVE_SEQUENCE_LOCK = threading.Lock()
LIVE_ACTIVE_SEQUENCE_PATH = OUTPUTS_LF_DIR / "live_active_sequence.json"
VALID_LIVE_MODES = {"lost-found", "attire"}
MAX_ACTIVE_LIVE_STREAMS = 4

def _load_view_mode_overrides():
    try:
        if not VIEW_MODE_OVERRIDE_PATH.exists():
            return {}

        with open(VIEW_MODE_OVERRIDE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            cleaned = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str) and v in VALID_VIEW_MODES:
                    cleaned[k] = v
            return cleaned

        return {}
    except Exception as e:
        print(f"[SYSTEM] Failed to load live view mode overrides: {e}")
        return {}

def _save_view_mode_overrides(data):
    try:
        VIEW_MODE_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = VIEW_MODE_OVERRIDE_PATH.with_suffix(".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        os.replace(tmp_path, VIEW_MODE_OVERRIDE_PATH)
        return True
    except Exception as e:
        print(f"[SYSTEM] Failed to save live view mode overrides: {e}")
        return False
    
@app.get("/api/live/view-mode-overrides")
def get_live_view_mode_overrides():
    with VIEW_MODE_OVERRIDE_LOCK:
        data = _load_view_mode_overrides()
    return data


@app.post("/api/live/view-mode-overrides")
def set_live_view_mode_overrides(payload: dict = Body(...)):
    if not isinstance(payload, dict):
        return {"ok": False, "error": "Payload must be an object"}

    cleaned = {}
    for k, v in payload.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, str):
            continue

        key = k.replace("_h264", "").strip()
        val = v.strip().lower()

        if not key:
            continue
        if val not in VALID_VIEW_MODES:
            continue

        cleaned[key] = val

    with VIEW_MODE_OVERRIDE_LOCK:
        ok = _save_view_mode_overrides(cleaned)

    return {"ok": ok, "count": len(cleaned), "overrides": cleaned}

def _load_live_active_sequence():
    try:
        if not LIVE_ACTIVE_SEQUENCE_PATH.exists():
            return {"lost-found": [], "attire": []}

        with open(LIVE_ACTIVE_SEQUENCE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {"lost-found": [], "attire": []}

        out = {"lost-found": [], "attire": []}
        for mode in ("lost-found", "attire"):
            arr = data.get(mode, [])
            if isinstance(arr, list):
                cleaned = []
                for x in arr:
                    cam_id = str(x or "").replace("_h264", "").strip()
                    if cam_id and cam_id not in cleaned:
                        cleaned.append(cam_id)
                out[mode] = cleaned[:MAX_ACTIVE_LIVE_STREAMS]

        return out
    except Exception as e:
        print(f"[SYSTEM] Failed to load live active sequence: {e}")
        return {"lost-found": [], "attire": []}


def _save_live_active_sequence(data):
    try:
        LIVE_ACTIVE_SEQUENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = LIVE_ACTIVE_SEQUENCE_PATH.with_suffix(".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        os.replace(tmp_path, LIVE_ACTIVE_SEQUENCE_PATH)
        return True
    except Exception as e:
        print(f"[SYSTEM] Failed to save live active sequence: {e}")
        return False
    
@app.get("/api/live/active-sequence")
def get_live_active_sequence(mode: str = "lost-found"):
    mode = str(mode or "lost-found").strip().lower()
    if mode not in VALID_LIVE_MODES:
        raise HTTPException(status_code=400, detail="Invalid mode")

    with LIVE_ACTIVE_SEQUENCE_LOCK:
        data = _load_live_active_sequence()

    return {
        "mode": mode,
        "active_cam_ids": data.get(mode, [])
    }


@app.post("/api/live/active-sequence")
def set_live_active_sequence(payload: dict = Body(...)):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be an object")

    mode = str(payload.get("mode") or "").strip().lower()
    if mode not in VALID_LIVE_MODES:
        raise HTTPException(status_code=400, detail="Invalid mode")

    arr = payload.get("active_cam_ids", [])
    if not isinstance(arr, list):
        raise HTTPException(status_code=400, detail="active_cam_ids must be a list")

    cleaned = []
    for x in arr:
        cam_id = str(x or "").replace("_h264", "").strip()
        if cam_id and cam_id not in cleaned:
            cleaned.append(cam_id)

    cleaned = cleaned[:MAX_ACTIVE_LIVE_STREAMS]

    with LIVE_ACTIVE_SEQUENCE_LOCK:
        data = _load_live_active_sequence()
        data[mode] = cleaned
        ok = _save_live_active_sequence(data)

    return {
        "ok": ok,
        "mode": mode,
        "active_cam_ids": cleaned
    }
