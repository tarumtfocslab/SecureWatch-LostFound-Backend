# lostandfound.py
import json
import gc
import logging
import traceback
import psutil
import torch
import math
import queue
import threading
import os
import cv2
import numpy as np
import re
import ctypes
from collections import defaultdict, deque
from ultralytics import  YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
import subprocess
import time
import csv
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Optional, Dict, Tuple, List
import warnings
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
LOSTFOUND_BACKEND_DIR = CURRENT_FILE.parents[0] 

# ---------------------------------------------------------
#  SET YOUR VIDEO PATH HERE
# ---------------------------------------------------------
# VIDEO_PATH = r"D:\20251110081000-20251110084059\B100D_B_Block_B_Block_20251110081000_20251110084058_39454624.mp4"
VIDEO_PATH = r"D:\DrTew\SecureWatch by QingYing JinXuan\Videos\B100D_B_Block_B_Block_20251110081000_20251110084058_39454624.mp4"
# VIDEO_PATH = r"D:\20251110081000-20251110084059\B001G_B_Block_B_Block_20251110083105_20251110084059_39975898.mp4"
# ---------------------------------------------------------

# ============================================================
# Backend overrides (headless / env-driven)
# ============================================================
LF_HEADLESS = os.getenv("LF_HEADLESS", "0") == "1"

if LF_HEADLESS:
    try:
        cv2.imshow = lambda *a, **k: None
        cv2.waitKey = lambda *a, **k: -1
        cv2.namedWindow = lambda *a, **k: None
        cv2.destroyAllWindows = lambda *a, **k: None
        cv2.setMouseCallback = lambda *a, **k: None

        # ✅ ADD THESE (important on Windows)
        cv2.resizeWindow = lambda *a, **k: None
        cv2.moveWindow = lambda *a, **k: None
        cv2.setWindowProperty = lambda *a, **k: None
        cv2.getWindowProperty = lambda *a, **k: -1
    except Exception:
        pass


# Allow backend to override VIDEO_PATH without editing the file
# (keeps your CLI working too)
try:
    VIDEO_PATH = os.getenv("LF_VIDEO_PATH", VIDEO_PATH)
except Exception:
    pass

# ============================================================
# ROI behavior overrides (backend/automation safe)
# ============================================================
# LF_ROI_MODE:
#   "auto"   -> if ROI exists and not empty: reuse, else error (recommended for backend)
#   "reuse"  -> always reuse (error if missing/empty)
#   "redraw" -> force redraw (CLI only)
LF_ROI_MODE = os.getenv("LF_ROI_MODE", "auto").strip().lower()

# If backend runs headless, default to safe behavior: auto-reuse ROI (no prompt)
if LF_HEADLESS and LF_ROI_MODE not in ("auto", "reuse", "redraw"):
    LF_ROI_MODE = "auto"
# ============================================================
# Fisheye mode (manual / auto_toggle / both)
# ============================================================
LF_FISHEYE_MODE = os.getenv("LF_FISHEYE_MODE", "").strip().lower()

# default behavior:
# - manual when running normally (with UI)
# - both when backend/headless (so it covers A+B => 8 views)
if not LF_FISHEYE_MODE:
    LF_FISHEYE_MODE = "both" if LF_HEADLESS else "manual"

if LF_FISHEYE_MODE not in ("manual", "auto_toggle", "both"):
    LF_FISHEYE_MODE = "both" if LF_HEADLESS else "manual"

# ---------------- FILTER SETTINGS ---------------- #
# ROI-aware size filtering (area in pixels)
MIN_BBOX_AREA = 50            # was 600 (small items become detectable)
ROI_STRICT_MARGIN_PX = 2   # allow bbox slightly outside ROI (more tolerant)
MAX_BBOX_AREA = 250000

# Optional width/height limits (set None to disable)
MIN_BBOX_W, MIN_BBOX_H = None, None
MAX_BBOX_W, MAX_BBOX_H = None, None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your trained items model
WEIGHTS_ITEMS = r"D:\DrTew\SecureWatch by QingYing JinXuan\SecureWatch\lostfound_backend\runs\lostandfound_global_v2\weights\best.pt"
print("ITEM WEIGHTS EXISTS?", os.path.exists(WEIGHTS_ITEMS), WEIGHTS_ITEMS)

# COCO model for person (light + fast)
WEIGHTS_PERSON = "D:\DrTew\SecureWatch by QingYing JinXuan\SecureWatch\lostfound_backend\yolov8x.pt"

COCO_PERSON_CLASS_ID = 0        # in COCO, person == 0
PERSON_NAME = "person"

DEBUG_DETECTIONS = True  # set False later if too noisy
# =========================
# ✅ HELPERS (KEEP)
# =========================
def put_drop_oldest(q: queue.Queue, item):
    """Put item into queue; if full, drop oldest first."""
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            return False

def drain_queue(q: queue.Queue, max_items: int = 999999):
    n = 0
    while n < max_items:
        try:
            q.get_nowait()
            n += 1
        except queue.Empty:
            break

SENTINEL = object()

_progress_lock = threading.Lock()

def write_progress(
    progress_path: str,
    *,
    run_id: str,
    stage: str,                  # "starting" | "running" | "finished" | "error"
    current: int,
    total: int | None,
    message: str,
    fps: float | None = None,
    lost_count: int | None = None,
    group_index: int | None = None,
    views_expected: int | None = None,
):
    with _progress_lock:
        data = {
            "run_id": run_id,
            "stage": stage,
            "current": int(current),
            "total": int(total) if total is not None else None,
            "percentage": (int((current / max(1, total)) * 100) if total else None),
            "fps": float(fps) if fps is not None else None,
            "lost_count": int(lost_count) if lost_count is not None else None,
            "group_index": int(group_index) if group_index is not None else None,
            "views_expected": int(views_expected) if views_expected is not None else None,
            "message": message,
            "updated_at": time.time(),
        }
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

# =========================================================
# OUTPUT CONTROL (clean console)
# =========================================================
DEBUG_RUNTIME = False
DEBUG_LOST_FOUND = False
DEBUG_EVENTS = False
VERBOSE_PREPROCESSOR_SCAN = False  # fisheye/normal scanning prints

def configure_logging(debug: bool = False, mute_info: bool = True):
    """
    debug=False + mute_info=True  -> only show your PHASE prints
    debug=True                   -> allow INFO/DEBUG logs
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    root = logging.getLogger()

    # IMPORTANT: remove existing handlers (your file sets logging elsewhere)
    for h in list(root.handlers):
        root.removeHandler(h)

    level = logging.DEBUG if debug else logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Silence noisy libraries
    for name in [
        "ultralytics",
        "deep_sort_realtime",
        "torch",
        "PIL",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Silence common internal loggers (best-effort)
    for name in [
        "memory_manager",
        "NormalPreprocessor",
        "FisheyePreprocessor",
        "FrameReaderThread",
        "ViewProcessorThread",
        "DetectionWorker",
        "TrackingThread",
        "SupervisorThread",
        "YoloDetector",
        "__main__",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Hard-mute INFO globally
    if mute_info and (not debug):
        logging.disable(logging.INFO)
    else:
        logging.disable(logging.NOTSET)

import builtins

# =========================================================
# STRICT CONSOLE MODE (only allow your PHASE/SYSTEM/UI lines)
# =========================================================
STRICT_CONSOLE = True

_ALLOWED_PREFIXES = (
    "======",        # banners
    "[PHASE",  "[SYSTEM]", "[UI]", "[DONE]", "[WARN]", "[ERROR]", "[SUP]",
)

_original_print = builtins.print

def filtered_print(*args, **kwargs):
    if not STRICT_CONSOLE:
        return _original_print(*args, **kwargs)

    msg = " ".join(str(a) for a in args).strip()
    if not msg:
        return

    # allow only your formatted output
    if msg.startswith(_ALLOWED_PREFIXES):
        return _original_print(*args, **kwargs)

    # block everything else (debug prints, library prints, etc.)
    return

builtins.print = filtered_print


# =========================================================
# CLEAN PRINT HELPERS (your PHASE/PHRASE only)
# =========================================================
def _banner(title: str):
    line = "=" * 70
    print("\n" + line)
    print(title)
    print(line)

def _step(tag: str, msg: str):
    print(f"[{tag}] {msg}")

def _kv(tag: str, **kwargs):
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    print(f"[{tag}] " + " | ".join(parts))

# =========================================================
# VERBOSE PRINT (for preprocessor scan)
# =========================================================
def _vprint(*args, **kwargs):
    if VERBOSE_PREPROCESSOR_SCAN:
        print(*args, **kwargs)

# ---------------------------------------------------------
#  Memory Manager
# ---------------------------------------------------------
class MemoryManager:
    """Comprehensive memory management for CPU, RAM, and GPU"""

    def __init__(self):
        self.cleanup_stats = {
            "cpu_cleanups": 0,
            "ram_cleanups": 0,
            "gpu_cleanups": 0,
            "total_cleanups": 0
        }
        self.memory_thresholds = {
            "ram_threshold": 0.85,  # 85% RAM usage threshold
            "gpu_threshold": 0.80,  # 80% GPU memory threshold
        }

    def get_memory_info(self):
        """Get current memory usage information"""
        memory_info = {}

        # RAM information
        ram = psutil.virtual_memory()
        memory_info['ram'] = {
            'total': ram.total / (1024 ** 3),  # GB
            'available': ram.available / (1024 ** 3),  # GB
            'used': ram.used / (1024 ** 3),  # GB
            'percentage': ram.percent
        }

        # CPU information
        memory_info['cpu'] = {
            'usage_percent': psutil.cpu_percent(interval=0.1),
            'count': psutil.cpu_count(),
            'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }

        # GPU information (if available)
        if torch.cuda.is_available():
            memory_info['gpu'] = {
                'available': True,
                'allocated': torch.cuda.memory_allocated() / (1024 ** 3),  # GB
                'reserved': torch.cuda.memory_reserved() / (1024 ** 3),  # GB
                'max_allocated': torch.cuda.max_memory_allocated() / (1024 ** 3),  # GB
                'percentage': (
                        torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100) if torch.cuda.max_memory_allocated() > 0 else 0
            }
        else:
            memory_info['gpu'] = {'available': False}

        return memory_info

    def cleanup_cpu_memory(self):
        """Clean up CPU-related memory and processes"""
        try:
            # Force garbage collection
            collected = gc.collect()

            # Clear Python's internal caches
            if hasattr(gc, 'set_threshold'):
                # Temporarily lower GC thresholds for more aggressive cleanup
                old_thresholds = gc.get_threshold()
                gc.set_threshold(100, 10, 10)
                gc.collect()
                gc.set_threshold(*old_thresholds)

            # Clear any cached compiled regex patterns
            if hasattr(re, '_cache'):
                re._cache.clear()

            self.cleanup_stats["cpu_cleanups"] += 1
            logger.info(f"CPU memory cleanup performed - collected {collected} objects")
            return collected

        except Exception as e:
            logger.error(f"CPU memory cleanup failed: {e}")
            return 0

    def cleanup_ram_memory(self):
        """Clean up RAM memory"""
        try:
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()

            # Clear NumPy memory pools if available
            try:

                # Note: np.core is deprecated, using gc.collect() for cleanup instead
                pass  # NumPy cleanup is handled by gc.collect()
            except:
                pass

            # Clear OpenCV memory if possible
            try:
                cv2.setUseOptimized(True)  # Ensure optimized operations
            except:
                pass

            # Windows-specific memory cleanup
            if hasattr(ctypes, 'windll'):
                try:
                    # Trim working set (Windows only)
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                except:
                    pass

            self.cleanup_stats["ram_cleanups"] += 1
            logger.info("RAM memory cleanup performed")

        except Exception as e:
            logger.error(f"RAM memory cleanup failed: {e}")

    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        try:
            if torch.cuda.is_available():
                # Clear GPU cache
                torch.cuda.empty_cache()

                # Synchronize GPU operations
                torch.cuda.synchronize()

                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()

                self.cleanup_stats["gpu_cleanups"] += 1
                logger.info("GPU memory cleanup performed")

        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")

    def comprehensive_cleanup(self):
        """Perform comprehensive memory cleanup"""
        logger.info("Starting comprehensive memory cleanup...")

        # Get memory info before cleanup
        before_memory = self.get_memory_info()

        # Perform all cleanups
        self.cleanup_cpu_memory()
        self.cleanup_ram_memory()
        self.cleanup_gpu_memory()

        # Get memory info after cleanup
        after_memory = self.get_memory_info()

        # Update stats
        self.cleanup_stats["total_cleanups"] += 1

        # Log improvement
        ram_before = before_memory['ram']['percentage']
        ram_after = after_memory['ram']['percentage']
        ram_improvement = ram_before - ram_after

        logger.info(
            f"Memory cleanup complete - RAM usage: {ram_before:.1f}% → {ram_after:.1f}% (freed {ram_improvement:.1f}%)")

        return {
            'before': before_memory,
            'after': after_memory,
            'improvement': {
                'ram_freed_percent': ram_improvement,
                'ram_freed_gb': (before_memory['ram']['used'] - after_memory['ram']['used'])
            }
        }

    def should_cleanup_memory(self):
        """Check if memory cleanup is needed"""
        memory_info = self.get_memory_info()

        # Check RAM threshold
        if memory_info['ram']['percentage'] > self.memory_thresholds['ram_threshold'] * 100:
            return True, "RAM usage high"

        # Check GPU threshold
        if torch.cuda.is_available() and memory_info['gpu']['percentage'] > self.memory_thresholds[
            'gpu_threshold'] * 100:
            return True, "GPU memory usage high"

        return False, "Memory usage normal"

    def get_cleanup_stats(self):
        """Get memory cleanup statistics"""
        return self.cleanup_stats.copy()


# Global memory manager instance
memory_manager = MemoryManager()

# ---------------------------------------------------------
#  DEVICE MANAGER (GPU if available, else CPU)
# ---------------------------------------------------------
class DeviceManager:
    """Choose best device: CUDA GPU if available, else CPU."""

    @staticmethod
    def get_device():
        try:
            if torch.cuda.is_available():
                logger.info("✅ CUDA GPU detected – using GPU for models")
                return torch.device("cuda")
            logger.info("⚠️ No GPU available – using CPU")
            return torch.device("cpu")
        except Exception as e:
            logger.error(f"Device detection failed ({e}), falling back to CPU")
            return torch.device("cpu")

# ---------------------------------------------------------
# Detect Video Type
# ---------------------------------------------------------
def is_fisheye_frame_radial(
        frame,
        inner_ratio: float = 0.30,
        corner_crop_ratio: float = 0.20,
        min_center_corner_diff: float = 40.0,
        min_variance: float = 8.0,
        max_corner_brightness: float = 90.0,
) -> bool:
    """
    Conservative fisheye check using:
      - center vs corners brightness
      - downscaled frame for speed
    """
    try:
        if frame is None:
            print("[WARN] is_fisheye_frame_radial(): frame is None")
            return False

        # 🔹 Downscale frame if very big
        h, w = frame.shape[:2]
        target_width = 480  # you can even try 320
        if w > target_width:
            scale = target_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Skip low-variance frames
        var = float(gray.var())
        if var < min_variance:
            # comment this out later to reduce prints
            print(f"[WARN] Frame too uniform (var={var:.2f}), skipping fisheye check.")
            return False

        # Center patch
        ch = int(inner_ratio * h)
        cw = int(inner_ratio * w)
        cy1 = (h - ch) // 2
        cy2 = cy1 + ch
        cx1 = (w - cw) // 2
        cx2 = cx1 + cw
        center_patch = gray[cy1:cy2, cx1:cx2]

        # Corners
        kh = int(corner_crop_ratio * h)
        kw = int(corner_crop_ratio * w)

        tl = gray[0:kh, 0:kw]
        tr = gray[0:kh, w - kw:w]
        bl = gray[h - kh:h, 0:kw]
        br = gray[h - kh:h, w - kw:w]

        corners = np.concatenate([tl.flatten(), tr.flatten(), bl.flatten(), br.flatten()])

        if center_patch.size == 0 or corners.size == 0:
            print("[WARN] Empty center/corner region, skipping frame.")
            return False

        center_mean = float(center_patch.mean())
        corner_mean = float(corners.mean())
        diff = center_mean - corner_mean

        print(
            f"[DEBUG] center_mean={center_mean:.2f}, "
            f"corner_mean={corner_mean:.2f}, diff={diff:.2f}, var={var:.2f}"
        )

        if diff >= min_center_corner_diff and corner_mean <= max_corner_brightness:
            return True
        else:
            return False

    except Exception as e:
        print(f"[ERROR] is_fisheye_frame_radial() failed: {e}")
        traceback.print_exc()
        return False


def detect_video_type(
        video_path,
        max_samples: int = 3,
        max_scan_seconds: float = 3.0,  # only look at first 3 seconds
) -> str:
    """Fast fisheye vs normal detection within the first few seconds."""

    try:
        # Optional: auto-clean if memory is high
        should_clean, reason = memory_manager.should_cleanup_memory()
        if should_clean:
            logger.info(f"Memory high before detect_video_type ({reason}) → running cleanup")
            memory_manager.comprehensive_cleanup()

        print(f"[INFO] Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return "normal"

        # ---- LIMIT TO FIRST FEW SECONDS ----
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1.0:
            fps = 25.0  # fallback

        max_scan_frames = int(fps * max_scan_seconds)
        if max_scan_frames <= 0:
            max_scan_frames = 75  # ~3 seconds at 25 FPS

        print(f"[INFO] FPS ≈ {fps:.2f}, scanning first {max_scan_frames} frames (~{max_scan_seconds}s)")

        # Choose sample indices **within** this small window
        step = max(max_scan_frames // (max_samples + 1), 1)
        sample_indices = {step * (i + 1) for i in range(max_samples)}
        print(f"[INFO] Sampling at frames (within first {max_scan_frames}): {sorted(sample_indices)}")

        fisheye_votes = 0
        normal_votes = 0

        frame_idx = 0
        sampled = 0

        print("[INFO] Start scanning limited frames...")
        while cap.isOpened() and frame_idx <= max_scan_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in sample_indices:
                print(f"\n[INFO] Checking frame {frame_idx}...")
                sampled += 1
                try:
                    if is_fisheye_frame_radial(frame):
                        fisheye_votes += 1
                        print("[INFO] → Frame RESULT: FISHEYE")
                    else:
                        normal_votes += 1
                        print("[INFO] → Frame RESULT: NORMAL")
                except Exception as e:
                    print(f"[ERROR] Frame {frame_idx} detection failed: {e}")
                    traceback.print_exc()

                # 🔹 EARLY EXIT: clearly normal
                if normal_votes >= 2 and normal_votes >= fisheye_votes + 1:
                    print("[INFO] Early stop: clearly NORMAL")
                    break

                # 🔹 EARLY EXIT: clearly fisheye
                if fisheye_votes >= 3:
                    print("[INFO] Early stop: clearly FISHEYE")
                    break

            frame_idx += 1
            if sampled >= max_samples:
                break

        cap.release()

        print(f"\n[INFO] Voting result: fisheye={fisheye_votes}, normal={normal_votes}")

        # Very conservative decision
        if fisheye_votes >= 3 and fisheye_votes >= normal_votes + 2:
            return "fisheye"
        else:
            return "normal"

    except Exception as e:
        print(f"[ERROR] detect_video_type() crashed: {e}")
        traceback.print_exc()
        return "normal"

# ---------------------------------------------------------
# Roi configuration
# ---------------------------------------------------------
def _extract_view_idx(view_name: str) -> int:
    """
    Tries to extract a view index from view_name.
    Examples:
      'view_0' -> 0
      'A0'     -> 0
      'B3'     -> 3
      'cam7'   -> 7
    If cannot parse -> returns -1 (no enforcement).
    """
    if not view_name:
        return -1

    s = str(view_name)

    digits = "".join([c for c in s if c.isdigit()])
    if digits:
        try:
            return int(digits)
        except Exception:
            return -1
    return -1


def filter_detections_by_roi_owner_view(dets: list, zones: list, view_name: str) -> list:
    """
    Each zone can define: zone['owner_view'] = int (0..7 for fisheye)
    If set, only detections from that view are allowed to use this ROI.

    Requirements:
      - det must already have 'roi_id' (your filter_detections_to_zones_by_overlap should assign it)
      - zones contain matching roi_id + owner_view
    """
    view_idx = _extract_view_idx(view_name)
    if view_idx < 0:
        # can't parse -> don't enforce
        return dets

    owner_by_roi = {}
    for z in (zones or []):
        rid = z.get("roi_id", None)
        if rid is None:
            continue
        ov = z.get("owner_view", None)
        if ov is None:
            continue
        try:
            owner_by_roi[str(rid)] = int(ov)
        except Exception:
            continue

    if not owner_by_roi:
        return dets

    out = []
    for d in (dets or []):
        rid = d.get("roi_id", None)
        if rid is None:
            continue

        ov = owner_by_roi.get(str(rid), None)
        # if ROI has owner view defined => enforce
        if ov is not None and int(view_idx) != int(ov):
            continue

        out.append(d)

    return out

def load_config(config_path: str = LOSTFOUND_BACKEND_DIR/"config.json"):
    """Load JSON config with bounding_polygons + fisheye_polygons."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        logger.info(f"[ROI] Loaded config from {config_path}")
        return cfg
    except FileNotFoundError:
        logger.warning(f"[ROI] Config file not found: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"[ROI] Failed to load config: {e}")
        return {}


def normalize_polygon_points(polygon):
    """
    Ensure polygon is a list of (x, y) tuples of floats.

    Accepts:
      - [(x, y), ...]
      - [[x, y], ...]
      - [{'x':.., 'y':..}, ...]
      - values can be int/float/str
    """
    if not polygon:
        return []

    pts = []
    for p in polygon:
        # dict point
        if isinstance(p, dict):
            x = p.get("x", 0)
            y = p.get("y", 0)

        # list/tuple point
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = p[0], p[1]
        else:
            continue

        try:
            x = float(x)
        except (TypeError, ValueError):
            x = 0.0

        try:
            y = float(y)
        except (TypeError, ValueError):
            y = 0.0

        pts.append((x, y))

    return pts


def polygons_to_zones(polygons, start_id=1, id_key="roi_id"):
    """
    polygons:
      [
        [ {x,y}, {x,y}, ... ],   # polygon 1
        [ {x,y}, {x,y}, ... ],   # polygon 2
      ]

    return:
      [{"shape":"polygon","points":[(x,y),...], "<id_key>":id}, ...]
    """
    zones = []
    zid = start_id
    for poly in (polygons or []):
        pts = normalize_polygon_points(poly)
        if len(pts) < 3:
            continue
        zones.append({"shape": "polygon", "points": pts, id_key: zid})
        zid += 1
    return zones


def get_zones_for_normal_video(roi_cfg):
    """Normal video uses roi_cfg['bounding_polygons']."""
    polys = (roi_cfg or {}).get("bounding_polygons", [])
    return polygons_to_zones(polys, start_id=1, id_key="roi_id")


def get_zones_for_fisheye_view(roi_cfg, view_name):
    """Fisheye uses roi_cfg['fisheye_polygons'][view_name] -> list of polygons."""
    fisheye = (roi_cfg or {}).get("fisheye_polygons", {})
    polys = fisheye.get(view_name, [])
    return polygons_to_zones(polys, start_id=1, id_key="roi_id")


def normalize_zones(zones):
    """
    Normalize zones into dict format:
      - dict zone stays dict, but polygon points are normalized to [(float,float),...]
      - raw polygon list becomes {"shape":"polygon","points":[...]}
    """
    out = []
    for z in (zones or []):
        if isinstance(z, dict):
            zz = dict(z)
            if zz.get("shape") == "polygon":
                zz["points"] = normalize_polygon_points(zz.get("points", []))
            out.append(zz)

        elif isinstance(z, (list, tuple)):
            # treat as raw polygon points
            pts = normalize_polygon_points(list(z))
            if len(pts) >= 3:
                out.append({"shape": "polygon", "points": pts, "roi_id": len(out) + 1})
    return out


def _zone_label_anchor(z, W, H):
    """Pick a stable anchor near left-most polygon point (label beside ROI)."""
    shape = z.get("shape", "polygon")

    if shape == "rect":
        x1, y1, x2, y2 = z.get("coords", (0, 0, 0, 0))
        ax, ay = int(x1), int((y1 + y2) / 2)
        return ax, ay

    pts = normalize_polygon_points(z.get("points", []))
    if len(pts) < 3:
        return None

    left_pt = min(pts, key=lambda p: (p[0], p[1]))
    ax, ay = int(left_pt[0]), int(left_pt[1])

    ax = max(0, min(W - 1, ax))
    ay = max(0, min(H - 1, ay))
    return ax, ay


def draw_zones_on_view(img, zones, color=(0, 255, 0), thickness=2, show_id=True):
    if img is None:
        return img

    out = img.copy()
    H, W = out.shape[:2]

    zones = normalize_zones(zones)
    if not zones:
        return out

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FS = 0.7
    FT = 2
    LINE = cv2.LINE_AA

    for z in zones:
        # ✅ use roi_id first (your new format), fallback to zone_id/id
        zid = str(z.get("roi_id") or z.get("zone_id") or z.get("id") or "ROI")
        shape = z.get("shape", "polygon")

        # --- draw shape ---
        if shape == "polygon":
            pts = normalize_polygon_points(z.get("points", []))
            if len(pts) >= 3:
                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts_np], True, color, thickness)

        elif shape == "rect":
            x1, y1, x2, y2 = z.get("coords", (0, 0, 0, 0))
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # --- draw label beside ROI ---
        if show_id:
            anchor = _zone_label_anchor(z, W, H)
            if anchor is None:
                continue
            ax, ay = anchor

            (tw, th), _ = cv2.getTextSize(zid, FONT, FS, FT)

            # place label LEFT of anchor; if out of screen -> RIGHT
            bg_x2 = ax - 8
            bg_x1 = bg_x2 - (tw + 12)
            bg_y1 = ay - (th + 10)
            bg_y2 = ay

            if bg_x1 < 0:
                bg_x1 = ax + 8
                bg_x2 = bg_x1 + (tw + 12)

            if bg_y1 < 0:
                bg_y1 = ay + 8
                bg_y2 = bg_y1 + (th + 10)

            # clamp
            bg_x1 = int(max(0, min(W - 1, bg_x1)))
            bg_x2 = int(max(0, min(W - 1, bg_x2)))
            bg_y1 = int(max(0, min(H - 1, bg_y1)))
            bg_y2 = int(max(0, min(H - 1, bg_y2)))

            cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(out, zid, (bg_x1 + 6, bg_y2 - 6), FONT, FS, (0, 0, 0), FT, LINE)

    return out


def get_largest_roi(zones):
    """Return largest ROI zone by area (polygon/rect)."""
    zones = normalize_zones(zones)
    if not zones:
        return None

    def zone_area(z):
        if z.get("shape") == "rect":
            x1, y1, x2, y2 = z.get("coords", (0, 0, 0, 0))
            return max(0, (x2 - x1)) * max(0, (y2 - y1))

        if z.get("shape") == "polygon":
            pts = normalize_polygon_points(z.get("points", []))  # ✅ important
            if len(pts) < 3:
                return 0.0
            area = 0.0
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                area += x1 * y2 - x2 * y1
            return abs(area) / 2.0

        return 0.0

    return max(zones, key=zone_area)

def _poly_union_bbox_trimmed(zones, trim_ratio=0.10):
    ys, xs = [], []
    for z in zones or []:
        if z.get("shape") != "polygon":
            continue
        for (x, y) in z.get("points", []):
            xs.append(int(x)); ys.append(int(y))
    if not xs:
        return None

    xs.sort()
    ys.sort()
    kx = int(len(xs) * trim_ratio)
    ky = int(len(ys) * trim_ratio)

    xs2 = xs[kx: len(xs)-kx] if len(xs) > 2*kx else xs
    ys2 = ys[ky: len(ys)-ky] if len(ys) > 2*ky else ys

    return (min(xs2), min(ys2), max(xs2), max(ys2))

def _expand_bbox_exclusive(bbox, pad, W, H):
    """
    bbox is (x1,y1,x2,y2) where x2/y2 are inclusive from points.
    Return (x1,y1,x2,y2) where x2/y2 are EXCLUSIVE for slicing.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad + 1)  # +1 to convert inclusive -> exclusive
    y2 = min(H, y2 + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _rect_to_covering_square(bbox, W, H):
    """
    bbox uses EXCLUSIVE (x2,y2). Return EXCLUSIVE square bbox that covers it.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    side = max(w, h)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    sx1 = cx - side // 2
    sy1 = cy - side // 2
    sx2 = sx1 + side
    sy2 = sy1 + side

    # clamp by shifting (keep size)
    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > W:
        shift = sx2 - W
        sx1 -= shift
        sx2 -= shift
    if sy2 > H:
        shift = sy2 - H
        sy1 -= shift
        sy2 -= shift

    sx1 = max(0, sx1); sy1 = max(0, sy1)
    sx2 = min(W, sx2); sy2 = min(H, sy2)

    if sx2 <= sx1 or sy2 <= sy1:
        return None
    return (sx1, sy1, sx2, sy2)

def _shift_zones(zones, dx, dy):
    """
    Shift zones coordinates by (-dx, -dy) when cropping.
    Supports:
      - rect:   zone["coords"] = (x1,y1,x2,y2)
      - polygon: zone["points"] = [(x,y),...]
    """
    out = []
    for z in zones or []:
        zz = dict(z)
        shape = zz.get("shape", "polygon")

        if shape == "rect":
            x1, y1, x2, y2 = zz.get("coords", (0, 0, 0, 0))
            zz["coords"] = (x1 - dx, y1 - dy, x2 - dx, y2 - dy)

        elif shape == "polygon":
            pts = zz.get("points", [])
            zz["points"] = [(float(x) - dx, float(y) - dy) for (x, y) in pts]

        out.append(zz)
    return out


def _shift_detections(detections, dx, dy):
    """Shift detection bbox by (+dx, +dy) to map crop coords back to full frame."""
    for d in detections or []:
        x1, y1, x2, y2 = d["bbox"]
        d["bbox"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
    return detections

# ---------------------------------------------------------
# Detection Calculation (Filtering)
# ---------------------------------------------------------
def filter_detections_by_conf_and_size(detections,
                                      conf_min=None,
                                      min_area=None, max_area=None,
                                      min_w=None, min_h=None,
                                      max_w=None, max_h=None):
    """Filter detections by confidence + bbox size constraints."""
    if not detections:
        return detections

    kept = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        area = w * h
        conf = float(det.get("confidence", 0.0))

        if conf_min is not None and conf < conf_min:
            continue
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        if min_w is not None and w < min_w:
            continue
        if min_h is not None and h < min_h:
            continue
        if max_w is not None and w > max_w:
            continue
        if max_h is not None and h > max_h:
            continue

        kept.append(det)

    return kept

def _point_in_zone(x, y, zone):
    shape = zone.get("shape", "rect")
    if shape == "rect":
        zx1, zy1, zx2, zy2 = zone["coords"]
        return zx1 <= x <= zx2 and zy1 <= y <= zy2
    if shape == "polygon":
        pts = np.array(zone["points"], dtype=np.int32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0
    return False


def _bbox_fully_in_zone(bbox, zone, margin=0):
    """True only if ALL bbox corners are inside the zone."""
    x1, y1, x2, y2 = map(float, bbox)
    x1m, y1m = x1 + margin, y1 + margin
    x2m, y2m = x2 - margin, y2 - margin
    if x2m <= x1m or y2m <= y1m:
        return False
    corners = [(x1m, y1m), (x2m, y1m), (x2m, y2m), (x1m, y2m)]
    return all(_point_in_zone(cx, cy, zone) for (cx, cy) in corners)


def filter_detections_to_zones_strict(detections, zones, margin=0):
    """Keep only detections whose WHOLE bbox is inside at least one ROI zone."""
    if not zones:
        return []
    kept = []
    for det in detections:
        bbox = det["bbox"]
        for z in zones:
            if _bbox_fully_in_zone(bbox, z, margin=margin):
                kept.append(det)
                break
    return kept

def _rect_intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih

def _bbox_area_xyxy(b):
    x1, y1, x2, y2 = map(float, b)
    return max(1.0, (x2 - x1)) * max(1.0, (y2 - y1))

def _bbox_inside_ratio_rect(bbox, rect_zone):
    # rect zone coords assumed (x1,y1,x2,y2)
    x1, y1, x2, y2 = map(float, bbox)
    zx1, zy1, zx2, zy2 = map(float, rect_zone["coords"])
    inter = _rect_intersection_area(x1, y1, x2, y2, zx1, zy1, zx2, zy2)
    return inter / _bbox_area_xyxy(bbox)

def _bbox_inside_ratio_polygon_mask(bbox, poly_zone, img_shape, mask_cache=None):
    """
    Compute (area inside polygon) / (bbox area) using a binary mask.
    Fast enough because we only count pixels inside bbox slice.
    """
    H, W = img_shape[:2]
    x1, y1, x2, y2 = map(int, map(round, bbox))

    # clamp bbox to image
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # cache key based on polygon points + image size
    pts = tuple((int(px), int(py)) for (px, py) in poly_zone.get("points", []))
    key = (H, W, pts)

    if mask_cache is not None and key in mask_cache:
        mask = mask_cache[key]
    else:
        mask = np.zeros((H, W), dtype=np.uint8)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        if mask_cache is not None:
            mask_cache[key] = mask

    inside_pixels = int(cv2.countNonZero(mask[y1:y2, x1:x2]))
    bbox_pixels = max(1, (x2 - x1) * (y2 - y1))
    return inside_pixels / float(bbox_pixels)

def filter_detections_to_zones_by_overlap(detections, zones, img_shape, min_ratio=0.50):
    """
    Keep detections where (area inside ROI / detection box area) >= min_ratio
    Works for rect + polygon zones.
    Also stores:
      - det["roi_inside_ratio"]
      - det["roi_zone_id"]  (the best-matching ROI zone)
    """
    zones = [z for z in (zones or []) if z.get("shape") in ("rect", "polygon")]
    # if not zones:
    #     return []  # no ROI => no detection (strict)

    kept = []
    mask_cache = {}

    for det in detections or []:
        bbox = det.get("bbox", None)
        if not bbox:
            continue

        best_ratio = 0.0
        best_zone_id = None

        for z in zones:
            shape = z.get("shape", "rect")

            if shape == "rect":
                r = _bbox_inside_ratio_rect(bbox, z)
            elif shape == "polygon":
                r = _bbox_inside_ratio_polygon_mask(bbox, z, img_shape, mask_cache=mask_cache)
            else:
                r = 0.0

            if r > best_ratio:
                best_ratio = r
                best_zone_id = z.get("zone_id", None)

        if best_ratio >= float(min_ratio):
            det["roi_inside_ratio"] = float(best_ratio)
            det["roi_zone_id"] = best_zone_id
            kept.append(det)

    return kept

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    if ax2 <= ax1 or ay2 <= ay1:
        return 0.0
    if bx2 <= bx1 or by2 <= by1:
        return 0.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter)

def dedup_by_overlap_ratio(detections, overlap_thr=0.50):
    """
    Deduplicate detections of the SAME class (and same ROI if present)
    using IoU-based overlap.
    Keeps the highest-confidence detection.
    """
    if not detections:
        return []

    # Sort by confidence first (highest wins)
    dets = sorted(
        detections,
        key=lambda d: float(d.get("confidence", 0.0)),
        reverse=True
    )

    kept = []
    for d in dets:
        bbox = d.get("bbox")
        if not bbox:
            continue

        duplicate = False
        for k in kept:
            # class must match
            if k.get("class_name") != d.get("class_name"):
                continue

            # optional: same ROI only
            if k.get("roi_zone_id") != d.get("roi_zone_id"):
                continue

            if _iou_xyxy(bbox, k["bbox"]) >= float(overlap_thr):
                duplicate = True
                break

        if not duplicate:
            kept.append(d)

    return kept
# ---------------------------------------------------------
#  Base Preprocessor
# ---------------------------------------------------------
class BasePreprocessor:
    """Base class for video preprocessing (normal or fisheye)."""
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None

    def open(self) -> bool:
        """Open video capture with error handling."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"[ERROR] Cannot open video in preprocessor: {self.video_path}")
                return False
            return True
        except Exception as e:
            print(f"[ERROR] Failed to open video in preprocessor: {e}")
            traceback.print_exc()
            return False

    def read_frame(self):
        """Read one frame safely."""
        try:
            if self.cap is None:
                return False, None
            ret, frame = self.cap.read()
            return ret, frame
        except Exception as e:
            print(f"[ERROR] Failed to read frame: {e}")
            traceback.print_exc()
            return False, None

    def get_views(self, frame):
        """Return list of view dicts. To be implemented by subclasses."""
        raise NotImplementedError

    def release(self):
        """Release OpenCV capture."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception as e:
            print(f"[WARN] Failed to release capture: {e}")

# ---------------------------------------------------------
#  Normal Preprocessor with ROI
# ---------------------------------------------------------
class NormalPreprocessor(BasePreprocessor):
    """
    Preprocessor for normal CCTV (non-fisheye).
    - Uses config.json["bounding_polygons"] to define ROIs (tables / zones).
    - Each polygon becomes one view + one zone.
    - If no polygons found, falls back to single full-frame view.
    """

    def __init__(
            self,
            video_path: str,
            config_path: str = LOSTFOUND_BACKEND_DIR/"config.json",
            resize_scale: float = 1.0,
    ):
        super().__init__(video_path)
        self.config_path = config_path
        self.resize_scale = resize_scale

        self.config = {}
        self.polygons = []  # list of polygons from config
        self.scale_factor = 1.0  # if you ever need to scale config coords
        self._initialized = False

    def open(self) -> bool:
        """Open video and load ROI config."""
        if not super().open():
            return False

        # optional: test read one frame just to verify stream
        ok, frame = self.read_frame()
        if not ok or frame is None:
            logger.error("[NormalPreprocessor] Failed to read first frame.")
            return False

        # reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # load config
        self.config = load_config(self.config_path)
        self.polygons = self.config.get("bounding_polygons", [])

        if not self.polygons:
            logger.warning(
                "[NormalPreprocessor] No bounding_polygons found in config. "
                "Will use full-frame as single view."
            )
        else:
            logger.info(
                f"[NormalPreprocessor] Loaded {len(self.polygons)} ROI polygons from config."
            )

        self._initialized = True
        return True

    def read_frame(self):
        """Read frame and optionally resize."""
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return ret, frame

        if self.resize_scale != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=self.resize_scale,
                fy=self.resize_scale,
                interpolation=cv2.INTER_AREA,
            )
        return True, frame

    def get_views(self, frame):
        """
        Normal camera = full-frame view with ROI polygons as zones.
        We do NOT crop; we overlay zones later when building the grid.
        """
        views = []
        try:
            if frame is None or not self._initialized:
                return views

            h, w = frame.shape[:2]

            # Build zones from config
            zones = get_zones_for_normal_video(self.config)

            # no ROI => keep empty
            # do NOT convert full frame into a fake ROI
            if not zones:
                zones = []

            logger.info(f"[NormalPreprocessor] full_frame view with {len(zones)} zone(s)")

            views.append({
                "view_id": 0,
                "roi_id": 0,
                "type": "normal_full",
                "name": "full_frame",
                "coords": (0, 0, w, h),
                "zones": zones,
                "image": frame,
            })

            return views

        except Exception as e:
            logger.error(f"[NormalPreprocessor] get_views failed: {e}")
            traceback.print_exc()
            return views
# ---------------------------------------------------------
#  Fisheye Preprocessor
# --------------------------------------------------------
# 8 views around fisheye (every 45 degrees) for max classroom coverage
FISHEYE_VIEW_CONFIGS = [
    # middle
    {"view_id": 0, "name": "middle_row",       "yaw":   100, "pitch": 2, "fov": 120, "rotate": 270},

    # Front-right
    {"view_id": 1, "name": "front_right_row", "yaw":  75, "pitch": 42, "fov": 65, "rotate": 270},

    # Right
    {"view_id": 2, "name": "front_left_row", "yaw": 6.0,  "pitch": 44.0,  "fov": 63.0,"rotate": 270 },

    # Back-right
    {"view_id": 3, "name": "back_right_row",  "yaw": 180, "pitch": 47, "fov": 49, "rotate": 270},

    # Back-left
    {"view_id": 4, "name": "back_left_row",   "yaw": 265, "pitch": 35, "fov": 66, "rotate": 270},

    # Left
    {"view_id": 5, "name": "entrance",        "yaw": 333, "pitch": 140, "fov": 70, "rotate": 270},

    # Front-left
    {"view_id": 6, "name": "back_corridor",  "yaw": 240, "pitch": 35, "fov": 76, "rotate": 270},

    # Corridor (center walkway)
    {"view_id": 7, "name": "front_corridor",        "yaw": 72, "pitch": 17, "fov": 115, "rotate": 270},
]

FISHEYE_GROUPS = [
    # Group A: Front + Right side + Corridor
    [
        "middle_row","front_right_row","front_left_row", "front_corridor",
    ],
    # Group B: Back + Left side
    [
        "back_right_row","back_left_row", "back_corridor", "entrance"
    ],
]
def create_fisheye_remap(
        fisheye_shape,
        output_shape,
        i_fov_deg: float,
        o_fov_deg: float,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
):
    """
    Create map_x, map_y to convert fisheye → perspective view.

    fisheye_shape: (h, w) of fisheye image
    output_shape: (h, w) of output (planar) image
    i_fov_deg: input fisheye FOV (e.g. 180)
    o_fov_deg: output vertical FOV (e.g. 90)
    yaw_deg: rotate around Z (0, 90, 180, 270 for 4 views)
    pitch_deg: tilt up/down
    roll_deg: roll (usually 0)
    """
    i_h, i_w = fisheye_shape
    o_h, o_w = output_shape

    # --- define the 3D projection plane for the output view ---

    # vertical FOV in radians
    v_fov = math.radians(o_fov_deg)
    # top/bottom of plane at z = 1
    y_max = math.tan(v_fov / 2.0)
    # horizontal FOV from aspect ratio
    aspect = o_w / o_h
    x_max = y_max * aspect

    # grid of points on plane (x, y, z=1)
    xs = np.linspace(-x_max, x_max, o_w, dtype=np.float32)
    ys = np.linspace(-y_max, y_max, o_h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)  # shape (o_h, o_w)
    zv = np.ones_like(xv, dtype=np.float32)

    xyz = np.stack([xv, yv, zv], axis=-1)  # (o_h, o_w, 3)
    # normalize to unit length
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz /= np.clip(norm, 1e-8, None)

    # --- rotations: yaw (Z), pitch (X), roll (Y) ---

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # yaw (around Z)
    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    # pitch (around X)
    Rx = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp]
    ], dtype=np.float32)
    # roll (around Y)
    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ], dtype=np.float32)

    R = Rz @ Rx @ Ry  # (3,3)

    # apply rotation to all rays
    rot = xyz @ R.T
    x_r = rot[..., 0]
    y_r = rot[..., 1]
    z_r = rot[..., 2]

    # spherical angles
    theta = np.arctan2(y_r, x_r)  # azimuth around Z
    phi = np.arctan2(np.sqrt(x_r ** 2 + y_r ** 2), z_r)  # angle from forward axis

    # --- project spherical → fisheye image (equidistant model) ---
    i_fov_rad = math.radians(i_fov_deg)
    f = min(i_w, i_h) / i_fov_rad  # focal length scale

    r = f * phi  # radius from center
    cx = i_w / 2.0
    cy0 = i_h / 2.0

    map_x = (cx + r * np.cos(theta)).astype(np.float32)
    map_y = (cy0 + r * np.sin(theta)).astype(np.float32)

    return map_x, map_y

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class FisheyePreprocessor:
    """
    FisheyePreprocessor: build multiple perspective views from one fisheye frame.

    Expected view_config fields per view:
      - view_id (int)
      - name (str)
      - yaw (0..359)
      - pitch (0..359)
      - fov (float/int)

    This class:
      ✅ builds per-view remap maps once in open()
      ✅ get_views() returns list of {"view_id","name","image","zones"}
      ✅ update_view_params(name, yaw/pitch/fov) rebuilds only that view map (real-time)

    NOTE: This version intentionally does NOT apply "rotate".
    """

    def __init__(
        self,
        view_configs,
        config_path=LOSTFOUND_BACKEND_DIR/"config.json",
        output_size=(480, 640),
        input_fov_deg=180.0,
        logger=None,
    ):
        self.logger = logger
        self.view_configs = [dict(v) for v in view_configs]
        self.config_path = config_path
        self.output_size = tuple(output_size)  # (h, w)
        self.input_fov_deg = float(input_fov_deg)
        # video handle
        self.cap = None
        self.video_path = None
        # fisheye input size
        self.in_h = None
        self.in_w = None
        # output size
        self.out_h, self.out_w = self.output_size
        # ROI polygons per view name
        self.fisheye_polys = {}
        # remap maps per view name -> (map_x, map_y)
        self.remap_maps = {}
        # thread lock for live updates
        self._lock = threading.Lock()
        self._initialized = False
        self.roi_cfg = {}

    # ------------------------------
    # Config loading
    # ------------------------------
    def _load_config(self):
        self.fisheye_polys = {}
        self.roi_cfg = {}
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                self.roi_cfg = cfg or {}
                self.fisheye_polys = cfg.get("fisheye_polygons", {}) or {}

                if self.logger:
                    self.logger.info(
                        f"[FisheyePreprocessor] Loaded fisheye_polygons for views: {list(self.fisheye_polys.keys())}"
                    )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[FisheyePreprocessor] Failed to load config: {e}")

    # ------------------------------
    # Video open / close
    # ------------------------------
    def open(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # grab one frame to get fisheye size
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Cannot read first frame from video")

        self.in_h, self.in_w = frame.shape[:2]

        # rewind to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # load config with polygons
        self._load_config()

        # build remap maps for all views
        self._build_all_maps()

        self._initialized = True
        if self.logger:
            self.logger.info(f"[FisheyePreprocessor] initialized with {len(self.view_configs)} views.")
        return True

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self._initialized = False

    # ------------------------------
    # Map building
    # ------------------------------
    def _build_one_map(self, cfg):
        """
        Build one remap map for a single view config.
        """
        map_x, map_y = create_fisheye_remap(
            fisheye_shape=(self.in_h, self.in_w),
            output_shape=(self.out_h, self.out_w),
            i_fov_deg=float(self.input_fov_deg),  # e.g. 180
            o_fov_deg=float(cfg["fov"]),  # output FOV
            yaw_deg=float(cfg["yaw"]),
            pitch_deg=float(cfg["pitch"]),
            roll_deg=0.0,
        )
        return map_x, map_y

    def _build_all_maps(self):
        self.remap_maps = {}
        for cfg in self.view_configs:
            name = cfg["name"]
            mx, my = self._build_one_map(cfg)
            self.remap_maps[name] = (mx, my)

    # ------------------------------
    # Read / produce views
    # ------------------------------
    def read_frame(self):
        if self.cap is None:
            return None, None
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def get_views(self, frame, allowed_names=None):
        """
        frame: BGR fisheye frame
        allowed_names: optional set/list of view names to generate
        Returns list of dict:
          [{"view_id":..,"name":..,"image":..,"zones":[...]}]
        """
        if frame is None:
            return []

        out = []
        for cfg in self.view_configs:
            name = cfg["name"]
            if allowed_names is not None and name not in allowed_names:
                continue

            # get remap maps (may be updated in real-time)
            mx_my = self.remap_maps.get(name, None)
            if mx_my is None:
                # build on demand (shouldn't happen normally)
                mx_my = self._build_one_map(cfg)
                self.remap_maps[name] = mx_my

            mx, my = mx_my

            # remap fisheye -> perspective
            warped = cv2.remap(
                frame, mx, my,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

            # ✅ apply rotate if exists in config (your configs use rotate=270)
            rot = int(cfg.get("rotate", 0) or 0) % 360
            if rot == 90:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                warped = cv2.rotate(warped, cv2.ROTATE_180)
            elif rot == 270:
                warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # ✅ Build zones using unified helper (fisheye per view)
            zones = get_zones_for_fisheye_view(self.roi_cfg, name)

            out.append(
                {
                    "view_id": int(cfg.get("view_id", 0)),
                    "name": name,
                    "image": warped,
                    "zones": zones,
                }
            )

        # stable order by view_id
        out.sort(key=lambda x: x["view_id"])
        return out

    # ------------------------------
    # ✅ Real-time update: yaw/pitch/fov
    # ------------------------------
    def update_view_params(self, view_name, yaw=None, pitch=None, fov=None):
        """
        Real-time update yaw/pitch/fov for ONE view and rebuild only that view's remap map.
        """
        if not self._initialized:
            return False

        cfg = None
        for c in self.view_configs:
            if c.get("name") == view_name:
                cfg = c
                break
        if cfg is None:
            return False

        with self._lock:
            if yaw is not None:
                cfg["yaw"] = float((yaw + 360.0) % 360.0)

            if pitch is not None:
                cfg["pitch"] = float((pitch + 360.0) % 360.0)

            if fov is not None:
                cfg["fov"] = float(_clamp(fov, 60.0, 170.0))

            # rebuild only this view map
            mx, my = self._build_one_map(cfg)
            self.remap_maps[view_name] = (mx, my)

        return True
    
# ---------------------------------------------------------
#  Create Preprocessor
# ---------------------------------------------------------
def create_preprocessor(video_path: str):
    """
    Decide which preprocessor to use based on video type.
    """
    print("\n[PHASE 2] Detecting video type for preprocessor...")
    video_type = detect_video_type(video_path)
    print(f"[PHASE 2] Video type result: {video_type.upper()}")

    try:
        if video_type == "fisheye":
            print("[PHASE 2] Using FisheyePreprocessor (multi-view dewarp).")
            preprocessor = FisheyePreprocessor(
                view_configs=FISHEYE_VIEW_CONFIGS,
                config_path=LOSTFOUND_BACKEND_DIR / "config.json"
            )

            preprocessor.open(video_path)
            return preprocessor


        else:
            print("[PHASE 2] Using NormalPreprocessor (ROI-based).")
            preprocessor = NormalPreprocessor(video_path, config_path=LOSTFOUND_BACKEND_DIR/"config.json")

        if not preprocessor.open():
            print("[ERROR] Failed to open video in preprocessor.")
            return None

        return preprocessor

    except Exception as e:
        print(f"[ERROR] create_preprocessor() failed: {e}")
        traceback.print_exc()
        return None

# ---------------------------------------------------------
#  Build view
# ---------------------------------------------------------
def build_views_grid(views, draw_zones=True):
    """
    Combine multiple view images into a single grid image and draw labels.
    - If 1 view  -> return that image with its name written on it.
    - If 4 views -> arrange as 2x2 grid by view_id, each with its name.
    - draw_zones=False -> do not overlay ROI/zones
    """
    try:
        if not views:
            return None

        def draw_label(img, text):
            if img is None:
                return img
            labeled = img.copy()
            cv2.rectangle(labeled, (5, 5), (250, 35), (0, 0, 0), thickness=-1)
            cv2.putText(
                labeled,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return labeled

        def maybe_draw_zones(img, zones):
            if not draw_zones:
                return img
            return draw_zones_on_view(img, zones or [])

        # ----- case 1: single view -----
        if len(views) == 1:
            v = views[0]
            img = v["image"]
            name = v.get("name", "view_0")

            # draw ROI on original-size image
            img = maybe_draw_zones(img, v.get("zones", []))
            img = draw_label(img, name)
            return img

        # ----- case 2: exactly 4 views -----
        if len(views) == 4:
            views_sorted = sorted(views, key=lambda v: v["view_id"])

            prepared = []
            for v in views_sorted:
                img = v["image"]
                name = v.get("name", f"view_{v.get('view_id', 0)}")

                # IMPORTANT: draw ROI BEFORE resize
                img = maybe_draw_zones(img, v.get("zones", []))
                prepared.append((img, name))

            h = min(img.shape[0] for img, _ in prepared)
            w = min(img.shape[1] for img, _ in prepared)

            def resize_to(img):
                return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

            img0 = draw_label(resize_to(prepared[0][0]), prepared[0][1])
            img1 = draw_label(resize_to(prepared[1][0]), prepared[1][1])
            img2 = draw_label(resize_to(prepared[2][0]), prepared[2][1])
            img3 = draw_label(resize_to(prepared[3][0]), prepared[3][1])

            top_row = np.hstack((img0, img1))
            bottom_row = np.hstack((img2, img3))
            grid = np.vstack((top_row, bottom_row))
            return grid

        # ----- fallback -----
        v = views[0]
        img = v["image"]
        name = v.get("name", "view_0")
        img = maybe_draw_zones(img, v.get("zones", []))
        return draw_label(img, name)

    except Exception as e:
        print(f"[ERROR] build_views_grid() failed: {e}")
        traceback.print_exc()
        return None
    
@dataclass
class SourceConfig:
    source: str | int
    source_type: str            # "file" | "webcam" | "rtsp"
    is_realtime: bool
    buffer_size: int = 50

def detect_source_config(source: str | int) -> SourceConfig:
    if isinstance(source, int):
        return SourceConfig(source=source, source_type="webcam", is_realtime=True, buffer_size=20)

    s = str(source)
    if s.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return SourceConfig(source=s, source_type="rtsp", is_realtime=True, buffer_size=20)

    if os.path.isfile(s):
        return SourceConfig(source=s, source_type="file", is_realtime=False, buffer_size=150)

    # fallback assume file
    return SourceConfig(source=s, source_type="file", is_realtime=False, buffer_size=150)

def open_source_capture(source: str | int):
    """
    Returns cv2.VideoCapture opened from:
      - int -> webcam
      - rtsp/http -> URL
      - file path -> file
    """
    cfg = detect_source_config(source)
    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {cfg.source} ({cfg.source_type})")
    return cap, cfg


# =========================================================
# ✅ THREADING PIPELINES: FrameReader -> ViewProcessor -> DetectionWorker
# =========================================================
class FrameReaderThread(threading.Thread):
    """
    Reads frames from preprocessor and pushes (frame_idx, ts, frame) into frame_queue.

    ts correctness:
      - video file: use CAP_PROP_POS_MSEC (video timeline) fallback to idx/fps
      - realtime: use monotonic wall clock offset
    """
    def __init__(
        self,
        preprocessor,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        *,
        frame_skip: int = 0,
        target_fps: float | None = None,
        is_realtime: bool = False,
    ):
        super().__init__(daemon=True)
        self.preprocessor = preprocessor
        self.frame_queue = frame_queue
        self.stop_event = stop_event

        self.frame_skip = max(0, int(frame_skip))
        self.target_fps = float(target_fps) if target_fps and target_fps > 0 else None
        self.is_realtime = bool(is_realtime)

        # try to read fps from the actual capture (if available)
        cap = getattr(self.preprocessor, "cap", None)
        fps = None
        if cap is not None:
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
            except Exception:
                fps = None
        if not fps or fps <= 1e-6:
            fps = 25.0
        self.fps = float(fps)

        # monotonic base for realtime streams
        self._mono0 = time.monotonic()

    def _ts_for_current_frame(self, frame_idx_fallback: int) -> float:
        """
        Return timestamp in seconds.

        For video files:
          - prefer CAP_PROP_POS_MSEC
          - fallback = CAP_PROP_POS_FRAMES / fps
          - fallback2 = frame_idx_fallback / fps

        For realtime:
          - monotonic seconds since start
        """
        cap = getattr(self.preprocessor, "cap", None)

        if not self.is_realtime and cap is not None:
            # 1) best: video timeline
            try:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if msec and msec > 0:
                    return float(msec) / 1000.0
            except Exception:
                pass

            # 2) fallback: frame position / fps
            try:
                pos_frames = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if pos_frames and pos_frames > 0 and self.fps > 0:
                    # CAP_PROP_POS_FRAMES points to "next frame index", so subtract 1
                    return max(0.0, float(pos_frames - 1.0) / self.fps)
            except Exception:
                pass

            # 3) last fallback: our counter / fps
            return float(frame_idx_fallback) / self.fps

        # realtime: monotonic wall time
        return float(time.monotonic() - self._mono0)

    def run(self):
        logger.info("[FrameReaderThread] Started.")

        dt = (1.0 / self.target_fps) if self.target_fps else 0.0
        next_t = time.monotonic()

        frame_idx = 0
        try:
            while not self.stop_event.is_set():
                ret, frame = self.preprocessor.read_frame()
                if not ret or frame is None:
                    logger.info("[FrameReaderThread] End of stream.")
                    break

                # skip logic
                if self.frame_skip > 0 and (frame_idx % (self.frame_skip + 1) != 0):
                    frame_idx += 1
                    continue

                ts = self._ts_for_current_frame(frame_idx)
                put_drop_oldest(self.frame_queue, (frame_idx, ts, frame))
                frame_idx += 1

                # pacing (optional)
                if dt > 0:
                    next_t += dt
                    sleep_s = next_t - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

        except Exception:
            logger.exception("[FrameReaderThread] crashed")
        finally:
            put_drop_oldest(self.frame_queue, (SENTINEL, None, None))
            logger.info("[FrameReaderThread] Exiting.")

class ViewProcessorThread(threading.Thread):
    """
    Converts frame -> views (fisheye: 4 views; normal: 1 view)
    Packs into ONE bundle job: (frame_idx, gi, views_list)
    """
    def __init__(self, preprocessor, frame_queue: queue.Queue, bundle_job_queue: queue.Queue,
                 stop_event: threading.Event, active_views):
        super().__init__(daemon=True)
        self.preprocessor = preprocessor
        self.frame_queue = frame_queue
        self.bundle_job_queue = bundle_job_queue
        self.stop_event = stop_event
        self.active_views = active_views  # global ActiveViews

    def _normalize_view_dict(self, v: dict) -> dict:
        """Ensure keys exist: image, name, view_id, zones."""
        out = dict(v)
        if "image" not in out and "img" in out:
            out["image"] = out["img"]
        out.setdefault("view_id", 0)
        out.setdefault("name", f"view_{out['view_id']}")
        out.setdefault("zones", [])
        return out

    def run(self):
        logger.info("[ViewProcessorThread] Started.")
        try:
            while not self.stop_event.is_set():
                try:
                    frame_idx, ts, frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if frame_idx is SENTINEL:
                    logger.info("[ViewProcessorThread] Got sentinel, stopping.")
                    break

                try:
                    if isinstance(self.preprocessor, FisheyePreprocessor):
                        allowed, gi = self.active_views.current()
                        views = self.preprocessor.get_views(frame, allowed_names=allowed)
                        views = [self._normalize_view_dict(v) for v in views]
                        expected = len(allowed)  # 4
                    else:
                        gi = 0
                        views = self.preprocessor.get_views(frame)
                        views = [self._normalize_view_dict(v) for v in views]
                        expected = 1

                    if not views:
                        continue

                    if expected == 4 and len(views) != 4:
                        logger.warning(f"[ViewProcessorThread] Frame {frame_idx}: expected 4 views but got {len(views)}")
                        # You can skip instead if you insist on strict 4/4:
                        # continue

                    bundle = (frame_idx, ts, gi, views)
                    put_drop_oldest(self.bundle_job_queue, bundle)

                except Exception:
                    logger.exception(f"[ViewProcessorThread] get_views failed at frame {frame_idx}")
                finally:
                    try:
                        self.frame_queue.task_done()
                    except Exception:
                        pass


        finally:
            # must match (frame_idx, ts, gi, views)
            put_drop_oldest(self.bundle_job_queue, (SENTINEL, None, None, None))
            logger.info("[ViewProcessorThread] Exiting.")

class DetectionWorkerThread(threading.Thread):
    """
    Input bundle:
      (frame_idx, ts, gi, views)

    Output bundle (to det_out_queue):
      (frame_idx, ts, gi, out_views)

    analysis_fps:
      - None / 0 => no throttling
      - >0       => process at this max rate
    """
    def __init__(
        self,
        worker_id: int,
        detector,
        job_queue: queue.Queue,
        out_queue: queue.Queue,
        stop_event: threading.Event,
        batch_size: int = 1,
        analysis_fps: float | None = None,
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.detector = detector
        self.job_queue = job_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.batch_size = max(1, int(batch_size))

        self.analysis_fps = float(analysis_fps) if analysis_fps and analysis_fps > 0 else None
        self._next_analysis_t = 0.0

        self.tracker_mgr = DeepSortTrackerManager(track_items_only=True)
        self._last_frame_idx = {}

    def _should_drop_as_stale(self, view_name: str, frame_idx: int) -> bool:
        last = self._last_frame_idx.get(view_name, -1)
        if frame_idx <= last:
            return True
        self._last_frame_idx[view_name] = frame_idx
        return False

    def _allow_analysis_now(self) -> bool:
        if not self.analysis_fps:
            return True

        now_t = time.monotonic()
        if now_t < self._next_analysis_t:
            return False

        self._next_analysis_t = now_t + (1.0 / float(self.analysis_fps))
        return True

    def run(self):
        logger.info(f"[DetectionWorker-{self.worker_id}] Started.")
        try:
            while not self.stop_event.is_set():
                bundles = []
                try:
                    bundles.append(self.job_queue.get(timeout=0.5))
                except queue.Empty:
                    continue

                for _ in range(self.batch_size - 1):
                    try:
                        bundles.append(self.job_queue.get_nowait())
                    except queue.Empty:
                        break

                for (frame_idx, ts, gi, views) in bundles:
                    try:
                        if frame_idx is SENTINEL:
                            logger.info(f"[DetectionWorker-{self.worker_id}] Got sentinel.")
                            put_drop_oldest(self.out_queue, (SENTINEL, None, None, None))
                            return

                        if not views:
                            continue

                        # throttle HEAVY detection here, not in ViewProcessorThread
                        if not self._allow_analysis_now():
                            continue

                        out_views = []
                        for v in views:
                            img = v.get("image", None)
                            if img is None:
                                continue

                            view_id = v.get("view_id", 0)
                            view_name = v.get("name", f"view_{view_id}")
                            zones = normalize_zones(v.get("zones", []) or [])

                            # largest ROI only
                            largest_roi = get_largest_roi(zones)
                            zones_for_crop = [largest_roi] if largest_roi is not None else []

                            # no ROI => no detection
                            if not zones_for_crop:
                                annotated = draw_zones_on_view(img.copy(), zones_for_crop)
                                out_views.append({
                                    "view_id": view_id,
                                    "name": view_name,
                                    "image": annotated,
                                    "raw_view": img.copy(),
                                    "raw_detections": [],
                                    "zones": zones_for_crop
                                })
                                continue

                            if self._should_drop_as_stale(view_name, frame_idx):
                                continue

                            H, W = img.shape[:2]
                            PAD = 120

                            bbox = _poly_union_bbox_trimmed(zones_for_crop, trim_ratio=0.15)
                            roi_sq = None
                            if bbox is not None:
                                bbox_exp = _expand_bbox_exclusive(bbox, PAD, W, H)
                                if bbox_exp is not None:
                                    roi_sq = _rect_to_covering_square(bbox_exp, W, H)

                            detections = []
                            if roi_sq is not None:
                                rx1, ry1, rx2, ry2 = roi_sq
                                crop = img[ry1:ry2, rx1:rx2].copy()

                                zones_crop = _shift_zones(zones_for_crop, dx=rx1, dy=ry1)
                                det_crop = self.detector.detect_confirmed(crop, zones_crop, view_name=view_name)
                                detections = _shift_detections(det_crop, rx1, ry1)

                            # attach roi_id
                            selected_roi_id = None
                            if isinstance(largest_roi, dict):
                                selected_roi_id = (
                                    largest_roi.get("roi_id")
                                    or largest_roi.get("zone_id")
                                    or largest_roi.get("id")
                                )

                            if selected_roi_id is not None:
                                for d in detections:
                                    d["roi_id"] = str(selected_roi_id)

                            annotated = img.copy()
                            annotated = draw_zones_on_view(annotated, zones_for_crop)
                            annotated = draw_detections_with_id(annotated, detections)

                            out_views.append({
                                "view_id": view_id,
                                "name": view_name,
                                "image": annotated,
                                "raw_view": img.copy(),
                                "raw_detections": list(detections),
                                "zones": zones_for_crop
                            })

                        put_drop_oldest(self.out_queue, (frame_idx, ts, gi, out_views))

                    except Exception:
                        logger.exception(f"[DetectionWorker-{self.worker_id}] failed")
                    finally:
                        try:
                            self.job_queue.task_done()
                        except Exception:
                            pass
        finally:
            logger.info(f"[DetectionWorker-{self.worker_id}] Exiting.")
class SupervisorThread(threading.Thread):
    def __init__(self, reader_thread, bundle_job_queue, stop_event, max_skip=3):
        super().__init__(daemon=True)
        self.reader = reader_thread
        self.q = bundle_job_queue
        self.stop_event = stop_event
        self.max_skip = max_skip
        self.full_count = 0
        self.empty_count = 0

    def run(self):
        logger.info("[SupervisorThread] Started.")
        while not self.stop_event.is_set():
            time.sleep(1.0)
            qsz = self.q.qsize()

            # queue is full
            if qsz >= self.q.maxsize - 1:
                self.full_count += 1
                self.empty_count = 0

                # ✅ only increase skip if full for 3 consecutive seconds
                if self.full_count >= 3:
                    self.reader.frame_skip = min(self.reader.frame_skip + 1, self.max_skip)
                    self.full_count = 0

            # queue is empty
            elif qsz == 0:
                self.empty_count += 1
                self.full_count = 0

                # ✅ only decrease skip if empty for 2 consecutive seconds
                if self.empty_count >= 2:
                    self.reader.frame_skip = max(self.reader.frame_skip - 1, 0)
                    self.empty_count = 0

            else:
                # stable zone
                self.full_count = 0
                self.empty_count = 0

class TrackingThread(threading.Thread):
    def __init__(self, det_out_queue, out_queue, stop_event, detector, confirm_k=3):
        super().__init__(daemon=True)
        self.in_q = det_out_queue
        self.out_q = out_queue
        self.stop_event = stop_event
        self.detector = detector
        self.confirm_k = int(confirm_k)

        self.tracker_mgr = DeepSortTrackerManager(
            track_items_only=False,
            max_age=70,
            n_init=1,
            max_iou_distance=0.75,
            embedder="mobilenet",
            half=True,
            bgr=True,
            min_det_conf=0.10,
            min_box_area=20 * 20,
            max_trackers=16,
        )

        self._seen_count = {}
        self.lost_manager = None

        # optional progress fields (attach from main)
        self.progress_path = None
        self.run_id = "run"
        self.total_frames = None
        self.views_expected = None
        self.fps_hint = None

    def _safe_get_detections(self, v):
        raw = v.get("raw_detections", None)
        if raw is None:
            raw = v.get("detections", None)
        if raw is None:
            raw = v.get("raw", None)
        return raw if raw is not None else []

    def _safe_get_frame(self, v):
        img = v.get("raw_view", None)
        if img is None:
            img = v.get("image", None)
        return img

    def _safe_update_tracker(self, view_name, img, dets_for_track):
        try:
            return self.tracker_mgr.update(view_name, img, dets_for_track)
        except TypeError:
            pass
        try:
            return self.tracker_mgr.update(view_name=view_name, frame_bgr=img, detections=dets_for_track)
        except TypeError:
            pass
        try:
            return self.tracker_mgr.update(img, dets_for_track, view_name=view_name)
        except TypeError:
            pass

        logger.warning("[TrackingThread] tracker_mgr.update signature mismatch -> returning []")
        return []

    def _apply_confirmation(self, track_key, tracks):
        seen = self._seen_count.setdefault(track_key, {})
        current_ids = set()

        for t in tracks:
            tid = int(t.get("track_id", -1))
            if tid < 0:
                continue
            current_ids.add(tid)
            seen[tid] = seen.get(tid, 0) + 1

        for tid in list(seen.keys()):
            if tid not in current_ids:
                seen.pop(tid, None)

        for t in tracks:
            tid = int(t.get("track_id", -1))
            t["stable"] = (tid >= 0 and seen.get(tid, 0) >= self.confirm_k)

        return tracks

    def _filter_for_tracking(self, img, zones, raw):
        persons = [d for d in raw if d.get("class_name") == "person"]
        items   = [d for d in raw if d.get("class_name") != "person"]

        persons = filter_detections_by_conf_and_size(
            persons,
            conf_min=max(0.30, getattr(self.detector, "person_conf", 0.30)),
            min_area=getattr(self.detector, "min_area_person", 40 * 40)
        )
        items = filter_detections_by_conf_and_size(
            items,
            conf_min=max(0.20, getattr(self.detector, "item_capture_conf", 0.25) * 0.8),
            min_area=getattr(self.detector, "min_area_item", 20 * 20)
        )

        # ROI gating for items only
        if zones:
            items = filter_detections_to_zones_strict(items, zones, margin=0)

        return persons + items

    def run(self):
        logger.info("[TrackingThread] Started.")
        try:
            while not self.stop_event.is_set():
                try:
                    frame_idx, ts, gi, det_views = self.in_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if frame_idx is SENTINEL:
                    put_drop_oldest(self.out_q, (SENTINEL, None, None, None))
                    return

                # ✅ timestamp source of truth (video timeline or monotonic realtime)
                now_ts = float(ts) if ts is not None else float(time.monotonic())

                # ✅ progress JSON for UI
                if self.progress_path:
                    lost_count = 0
                    if self.lost_manager is not None:
                        try:
                            lost_count = len(self.lost_manager.lost_items)
                        except Exception:
                            pass

                    write_progress(
                        self.progress_path,
                        run_id=self.run_id,
                        stage="running",
                        current=int(frame_idx),
                        total=self.total_frames,
                        fps=self.fps_hint,
                        lost_count=lost_count,
                        group_index=gi,
                        views_expected=self.views_expected,
                        message=f"Processing frame {frame_idx}",
                    )

                out_views = []

                for v in (det_views or []):
                    img = self._safe_get_frame(v)
                    if img is None:
                        continue

                    zones = v.get("zones", []) or []
                    raw = self._safe_get_detections(v)

                    view_id = v.get("view_id", 0)
                    view_name = v.get("name", "view_0")

                    dets_for_track = self._filter_for_tracking(img, zones, raw)

                    # ---- group detections by roi_id (string)
                    roi_groups = {}
                    for d in dets_for_track:
                        rid = d.get("roi_id")

                        if rid is None and zones and isinstance(zones[0], dict):
                            rid = zones[0].get("zone_id") or zones[0].get("roi_id") or zones[0].get("id")

                        rid = str(rid) if rid is not None else "full_frame"
                        d["roi_id"] = rid
                        roi_groups.setdefault(rid, []).append(d)

                    all_tracks_raw_full = []

                    for rid, dets_roi_full in roi_groups.items():
                        track_key = f"{view_name}__roi{rid}"

                        # optional ROI crop (kept as your logic)
                        roi_img = img
                        roi_offset = (0, 0)

                        roi_poly = None
                        for z in zones:
                            zid = str(z.get("zone_id") or z.get("roi_id") or z.get("id") or "")
                            if zid == str(rid):
                                roi_poly = z.get("points") or z.get("polygon") or z.get("pts")
                                break

                        if roi_poly and len(roi_poly) >= 3:
                            xs = [p[0] for p in roi_poly]
                            ys = [p[1] for p in roi_poly]
                            x1 = max(0, int(min(xs)))
                            y1 = max(0, int(min(ys)))
                            x2 = min(img.shape[1], int(max(xs)))
                            y2 = min(img.shape[0], int(max(ys)))
                            if x2 > x1 and y2 > y1:
                                roi_img = img[y1:y2, x1:x2].copy()
                                roi_offset = (x1, y1)

                        ox, oy = roi_offset

                        # tracker wants ROI-local coords
                        dets_roi_local = []
                        for d in dets_roi_full:
                            bb = d.get("bbox")
                            if bb and len(bb) == 4:
                                x1, y1, x2, y2 = bb
                                d2 = dict(d)
                                d2["bbox"] = [x1 - ox, y1 - oy, x2 - ox, y2 - oy]
                                dets_roi_local.append(d2)
                            else:
                                dets_roi_local.append(dict(d))

                        tracks_local = self._safe_update_tracker(track_key, roi_img, dets_roi_local)
                        tracks_local = self._apply_confirmation(track_key, tracks_local)

                        # convert back to FULL coords
                        tracks_full = []
                        for t in tracks_local:
                            t2 = dict(t)
                            bb = t2.get("bbox")
                            if bb and len(bb) == 4:
                                t2["bbox"] = [bb[0] + ox, bb[1] + oy, bb[2] + ox, bb[3] + oy]
                            db = t2.get("det_bbox")
                            if db and len(db) == 4:
                                t2["det_bbox"] = [db[0] + ox, db[1] + oy, db[2] + ox, db[3] + oy]
                            t2["roi_id"] = str(rid)
                            tracks_full.append(t2)

                        all_tracks_raw_full.extend(tracks_full)

                        # send stable tracks to lost_manager (FULL frame coords!)
                        if self.lost_manager is not None:
                            stable_full = [t for t in tracks_full if t.get("stable")]
                            if stable_full:
                                tracked_objects = []
                                for t in stable_full:
                                    bbox = t.get("det_bbox") or t.get("bbox")
                                    cls = t.get("class_name")
                                    if bbox is None or cls is None:
                                        continue
                                    tracked_objects.append({
                                        "track_id": int(t.get("track_id", -1)),
                                        "class_name": str(cls),
                                        "bbox": list(map(float, bbox)),
                                        "det_bbox": t.get("det_bbox", None),
                                        "confidence": float(t.get("confidence", 1.0)),
                                        "confirmed": True,
                                        "stable": True,
                                        "roi_id": str(rid),
                                    })

                                self.lost_manager.process_tracks(
                                    now_ts,
                                    view_name=view_name,
                                    roi_id=str(rid),
                                    tracked_objects=tracked_objects,
                                    frame_bgr=img,
                                )

                    # attach track IDs back to detections (both FULL coords)
                    dets_for_track = attach_track_ids_to_detections(
                        dets_for_track, all_tracks_raw_full, iou_thr=0.3
                    )

                    annotated = img.copy()
                    annotated = draw_zones_on_view(annotated, zones)
                    annotated = draw_detections_with_id(annotated, dets_for_track)

                    out_views.append({
                        "view_id": view_id,
                        "name": view_name,
                        "image": annotated,
                        "raw_view": img,
                        "raw_detections": raw,
                        "zones": zones
                    })

                # ✅ IMPORTANT: keep ts to UI/output
                put_drop_oldest(self.out_q, (frame_idx, now_ts, gi, out_views))

        except Exception:
            logger.exception("[TrackingThread] crashed")
        finally:
            logger.info("[TrackingThread] Exiting.")


# ---------------------------------------------------------
#  Yolo Detection
# ---------------------------------------------------------
class YoloDetector:
    """
    Dual-model detector (CUSTOM + COCO) with "senior guards":
    ✅ Keep your COCO ID allowlist (person + selected lost items ONLY)
    ✅ Two-stage thresholding (capture low → confirm via temporal voting)
    ✅ Temporal memory + HOLD (blink recovery)
    ✅ ROI filtering (items: center-in-zone; person: no ROI filter)
    ✅ Class-specific area ratio rules (WIDE first; tighten slowly)
    ✅ Prefer custom over COCO on overlap (dedup)
    """
    # -------------------------
    # COCO allowlist (KEEP EXACTLY)
    # -------------------------
    COCO_PERSON_ID = 0

    COCO_LOST_ITEM_IDS = {
        26,  # handbag
        39,  # bottle
        67,  # cell phone
        73,  # book
    }

    # Only rename COCO classes that overlap with custom dataset
    COCO_TO_CUSTOM_NAME = {
        "bottle": "water_bottle",
        "cell phone": "mobile_phone",
    }

    CLASS_COLORS = {
        "backpack": (255, 0, 255),
        "handbag": (255, 128, 128),
        "watch": (0, 255, 0),
        "laptop": (160, 160, 160),  # or hide it
        "tablet": (0, 200, 255),
        "mobile_phone": (0, 255, 255),
        "earphones": (255, 255, 0),
        "powerbank": (255, 180, 0),
        "water_bottle": (255, 200, 0),
        "umbrella": (200, 0, 255),
        "usb_drive": (0, 180, 255),
        "wallet": (255, 0, 0),
        "card": (0, 255, 100),
        "key": (255, 255, 255),
        "headphone": (0, 128, 255),
        "charger_adapter": (128, 255, 0),
        "spectacles": (255, 0, 128),

        # if COCO adds these:
        "book": (120, 255, 120),
    }

    BLOCKED_CLASS_NAMES = {
        "backpack",
        "laptop",
        "handbag",
        "headphone",
        "charger_adapter",
        "spectacles",
        "usb_drive",
    }

    @staticmethod
    def normalize_coco_class_name(name: str) -> str:
        if not name:
            return name
        key = name.strip().lower()
        return YoloDetector.COCO_TO_CUSTOM_NAME.get(key, key)

    def __init__(
        self,
        items_weights: str,
        coco_weights: str,
        device: torch.device,

        # -------------------------
        # Two-stage thresholds
        # -------------------------
        item_capture_conf: float = 0.08,     # 0.05–0.10 keep weak custom detections (high recall)
        coco_capture_conf: float = 0.10,     # 0.10–0.18 keep weak coco detections (high recall)
        person_conf: float = 0.30,           # 0.25–0.35 display persons immediately

        iou: float = 0.50,

        # -------------------------
        # Temporal confirm + hold
        # -------------------------
        track_win: int = 10,     # lookback window
        confirm_k: int = 3,      # must appear K times in window
        hold_frames: int = 15,   # keep visible after last seen
        grid: int = 60,          # spatial bucket size

        # -------------------------
        # Filtering
        # -------------------------
        min_area_item: float = 30.0,     # tiny boxes are usually noise
        min_area_person: float = 1500.0,

        # Class-specific area ratio rules (WIDE first; tighten later)
        class_area_ratio_rules: dict | None = None,

        # Prefer custom over coco if overlap high
        dedup_iou: float = 0.60,

        # Inference image sizes (optional tuning)
        custom_imgsz: int = 640,
        coco_imgsz: int = 832,
    ):
        self.device = device
        self.iou = float(iou)
        self._predict_lock = threading.Lock()

        # stage thresholds
        self.item_capture_conf = float(item_capture_conf)
        self.coco_capture_conf = float(coco_capture_conf)
        self.person_conf = float(person_conf)

        # temporal
        self.track_win = int(track_win)
        self.confirm_k = int(confirm_k)
        self.hold_frames = int(hold_frames)
        self.grid = int(grid)

        # filtering
        self.min_area_item = float(min_area_item)
        self.min_area_person = float(min_area_person)

        self.dedup_iou = float(dedup_iou)
        self.custom_imgsz = int(custom_imgsz)
        self.coco_imgsz = int(coco_imgsz)

        # wide defaults (don’t over-filter)
        self.class_area_ratio_rules = class_area_ratio_rules or {
            # COCO names
            "bottle":     (0.00015, 0.07),
            "cell phone": (0.00008, 0.05),
            "mouse":      (0.00003, 0.03),
            "book":       (0.00020, 0.12),
            "backpack":   (0.00080, 0.35),
            "handbag":    (0.00060, 0.30),


            # Custom naming variants (if your custom model uses these)
            "water_bottle": (0.00015, 0.07),
            "mobile_phone": (0.00008, 0.05),
        }

        # temporal memory store
        self._hist = defaultdict(lambda: deque(maxlen=self.track_win))
        self._last_seen = {}
        self._last_bbox = {}
        self._frame_idx = 0
        self.USE_CUSTOM = True
        self.USE_COCO = True
        self.DEBUG_SRC = True
        self.USE_COCO_PERSON_ONLY = True

        # models
        logger.info(f"[YoloDetector] Loading CUSTOM items model: {items_weights}")
        self.item_model = YOLO(items_weights)

        logger.info(f"[YoloDetector] Loading COCO model: {coco_weights}")
        self.coco_model = YOLO(coco_weights)

        # Force models onto selected device
        try:
            if self.device is not None and getattr(self.device, "type", "") == "cuda":
                self.item_model.to("cuda")
                self.coco_model.to("cuda")
                logger.info("[YoloDetector] Both YOLO models moved to CUDA")
            else:
                self.item_model.to("cpu")
                self.coco_model.to("cpu")
                logger.info("[YoloDetector] Both YOLO models using CPU")
        except Exception as e:
            logger.warning(f"[YoloDetector] Failed to move models to device: {e}")

        self.item_names = self.item_model.names
        logger.info(f"[CUSTOM MODEL CLASSES] {self.item_names}")

        self.coco_names = self.coco_model.names

    # ---------------------------------------------------------
    # Core prediction
    # ---------------------------------------------------------
    def _predict(self, model, img, conf, imgsz):
        dev = 0 if self.device.type == "cuda" else "cpu"
        with self._predict_lock:
            return model.predict(
                source=img,
                conf=float(conf),
                iou=self.iou,
                imgsz=int(imgsz),
                device=dev,
                verbose=False
            )[0]

    def _get_name(self, names, cls_id: int) -> str:
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        return names[cls_id] if cls_id < len(names) else str(cls_id)

    # ---------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------
    @staticmethod
    def _bbox_area(b):
        x1, y1, x2, y2 = b
        return max(1.0, (x2 - x1)) * max(1.0, (y2 - y1))

    def _tkey(self, det, view_name: str):
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (view_name, det["class_name"], int(cx // self.grid), int(cy // self.grid))

    # ---------------------------------------------------------
    # Your original output format + add source tag
    # bbox is stored as list[float] like your code
    # ---------------------------------------------------------
    def _to_det(self, cls_id, cls_name, conf, bbox, source):
        x1, y1, x2, y2 = bbox
        return {
            "class_id": int(cls_id),
            "class_name": str(cls_name),
            "confidence": float(conf),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "source": str(source),  # "custom" | "coco" | "hold"
        }

    # ---------------------------------------------------------
    # Step 1: raw capture (LOW conf) with COCO allowlist enforced
    # ---------------------------------------------------------
    def detect_raw(self, img):
        if img is None:
            return []

        detections = []

        # -----------------------------
        # 1) CUSTOM TRAINED ITEMS
        # -----------------------------
        if getattr(self, "USE_CUSTOM", True):
            r_items = self._predict(self.item_model, img, self.item_capture_conf, self.custom_imgsz)
            if r_items.boxes is not None:
                for b in r_items.boxes:
                    cls_id = int(b.cls.item())
                    cls_name = str(self._get_name(self.item_names, cls_id)).strip().lower()

                    # block unwanted classes
                    if cls_name in self.BLOCKED_CLASS_NAMES:
                        continue

                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    detections.append(
                        self._to_det(
                            cls_id=cls_id,
                            cls_name=cls_name,
                            conf=float(b.conf.item()),
                            bbox=(x1, y1, x2, y2),
                            source="custom",
                        )
                    )

        # -----------------------------
        # 2) COCO MODEL (person + optional selected lost items)
        # -----------------------------
        # You have 2 choices:
        # A) USE_COCO=False => no COCO at all (no person boxes)
        # B) keep person boxes even when USE_COCO=False (recommended):
        #    set self.USE_COCO_PERSON_ONLY = True

        use_coco = getattr(self, "USE_COCO", True)
        use_coco_person_only = getattr(self, "USE_COCO_PERSON_ONLY", False)

        if use_coco or use_coco_person_only:
            r_coco = self._predict(self.coco_model, img, self.coco_capture_conf, self.coco_imgsz)
            if r_coco.boxes is not None:
                for b in r_coco.boxes:
                    coco_id = int(b.cls.item())
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf.item())

                    # person always allowed if coco is used
                    if coco_id == self.COCO_PERSON_ID:
                        detections.append(
                            self._to_det(
                                cls_id=999,
                                cls_name="person",
                                conf=conf,
                                bbox=(x1, y1, x2, y2),
                                source="coco",
                            )
                        )
                        continue

                    # only allow COCO lost items if FULL coco is enabled
                    if use_coco and (coco_id in self.COCO_LOST_ITEM_IDS):
                        raw_name = self._get_name(self.coco_names, coco_id)
                        cls_name = self.normalize_coco_class_name(raw_name)
                        detections.append(
                            self._to_det(
                                cls_id=coco_id,
                                cls_name=cls_name,
                                conf=conf,
                                bbox=(x1, y1, x2, y2),
                                source="coco",
                            )
                        )

        return detections

    # ---------------------------------------------------------
    # Step 2: prefer custom over coco when overlapping
    # ---------------------------------------------------------
    def _dedup_custom_over_coco(self, dets):
        custom = [d for d in dets if d.get("source") == "custom"]
        coco = [d for d in dets if d.get("source") == "coco"]

        keep_coco = []
        for c in coco:
            cb = c["bbox"]
            overlapped = False
            for u in custom:
                if _iou_xyxy(cb, u["bbox"]) >= self.dedup_iou:
                    overlapped = True
                    break
            if not overlapped:
                keep_coco.append(c)

        return custom + keep_coco

    # ---------------------------------------------------------
    # Step 3: wide size filters (avoid monitor->bottle etc.)
    # ---------------------------------------------------------
    def _filter_by_area_ratio(self, dets, img_w, img_h):
        frame_area = float(img_w * img_h)
        out = []
        for d in dets:
            name = d["class_name"]
            if name in self.class_area_ratio_rules:
                mn, mx = self.class_area_ratio_rules[name]
                ratio = self._bbox_area(d["bbox"]) / frame_area
                if ratio < mn or ratio > mx:
                    continue
            out.append(d)
        return out

    # ---------------------------------------------------------
    # Step 4: temporal confirm + hold (blink recovery)
    # ---------------------------------------------------------
    def _temporal_confirm_and_hold(self, items, view_name):
        self._frame_idx += 1

        confirmed = []
        seen_now = set()

        for d in items:
            k = self._tkey(d, view_name)
            seen_now.add(k)
            self._hist[k].append(1)
            self._last_seen[k] = self._frame_idx
            self._last_bbox[k] = d["bbox"]

            if sum(self._hist[k]) >= self.confirm_k:
                confirmed.append(d)

        # HOLD (keep last bbox for a short time even if missing)
        for k, last in list(self._last_seen.items()):
            if (self._frame_idx - last) > self.hold_frames:
                self._last_seen.pop(k, None)
                self._hist.pop(k, None)
                self._last_bbox.pop(k, None)
                continue

            # if not seen this frame but was confirmed recently, keep it visible
            if k not in seen_now and sum(self._hist[k]) >= self.confirm_k:
                vname, cls_name, _, _ = k
                if vname == view_name and k in self._last_bbox:
                    confirmed.append({
                        "class_id": -1,
                        "class_name": cls_name,
                        "confidence": 0.0,
                        "bbox": self._last_bbox[k],
                        "source": "hold",
                    })

        return confirmed

    # ---------------------------------------------------------
    # PUBLIC API: one call per view
    # ---------------------------------------------------------
    def detect_confirmed(self, img, zones=None, view_name="default"):
        """
        Single source of truth for detection logic.
        Enforces: ROI owner view => same item across 8 views only kept from ONE view.
        """
        if img is None:
            return []

        if zones is None:
            zones = []

        # 0) RAW DETECTION (CUSTOM + optional COCO)
        raw = self.detect_raw(img)

        # Prefer custom over coco on overlap
        raw = self._dedup_custom_over_coco(raw)

        persons = [d for d in raw if d["class_name"] == "person"]
        items = [d for d in raw if d["class_name"] != "person"]

        # 1) CONF + SIZE
        # 1) CONF + SIZE
        persons = filter_detections_by_conf_and_size(
            persons,
            conf_min=self.person_conf,
            min_area=self.min_area_person
        )

        # NEW: apply ROI filtering to persons
        persons = filter_detections_to_zones_by_overlap(
            persons, zones, img_shape=img.shape, min_ratio=0.30
        )
        items = filter_detections_by_conf_and_size(
            items,
            conf_min=max(0.01, self.item_capture_conf * 0.8),
            min_area=self.min_area_item
        )

        items = filter_detections_to_zones_by_overlap(
            items, zones, img_shape=img.shape, min_ratio=0.30
        )

        # 2) ROI GATING (items only)
        ROI_MIN_RATIO = 0.40
        items = filter_detections_to_zones_by_overlap(
            items, zones, img_shape=img.shape, min_ratio=ROI_MIN_RATIO
        )

        # ✅ NEW: block detections coming from non-owner views
        items = filter_detections_by_roi_owner_view(items, zones, view_name)

        # ✅ 3) DEDUP overlaps
        persons = dedup_by_overlap_ratio(persons, overlap_thr=0.70)
        items = dedup_by_overlap_ratio(items, overlap_thr=0.45)

        # 4) AREA RATIO FILTER (items only)
        h, w = img.shape[:2]
        items = self._filter_by_area_ratio(items, w, h)

        # 5) TEMPORAL CONFIRM + HOLD
        items = self._temporal_confirm_and_hold(items, view_name)

        if getattr(self, "DEBUG_SRC", False):
            if DEBUG_RUNTIME:
                print(f"[FINAL] view={view_name} items={len(items)} persons={len(persons)}")

        return persons + items

def draw_detections_with_id(img, detections):
    """
    User-friendly drawing:
    - Person boxes: GREEN
    - Item boxes: color by CLASS_COLORS
    - Label background uses same color (brighter)
    - Label shows: ID:<tid> (if any) + class + conf + optional ROI ratio
    """
    out = img.copy()
    H, W = out.shape[:2]

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    LINE_TYPE = cv2.LINE_AA

    base = min(W, H)

    if base <= 540:          # small RTSP frames
        FONT_SCALE = 0.35
        FONT_THICKNESS = 1
        PAD_X = 2
        PAD_Y = 1
        GAP_Y = 1
    elif base <= 720:
        FONT_SCALE = 0.55
        FONT_THICKNESS = 1
        PAD_X = 4
        PAD_Y = 3
        GAP_Y = 4
    else:                    # larger upload/file frames
        FONT_SCALE = 0.60
        FONT_THICKNESS = 1
        PAD_X = 5
        PAD_Y = 4
        GAP_Y = 4

    TEXT_COLOR = (0, 0, 0)

    for d in detections or []:
        # ---- bbox + meta ----
        x1, y1, x2, y2 = map(int, d.get("bbox", [0, 0, 0, 0]))
        cls = str(d.get("class_name", "obj"))
        conf = float(d.get("confidence", 0.0))
        tid = d.get("track_id", None)

        # ---- clamp bbox (safety) ----
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        # ---- choose color ----
        if cls == "person":
            box_color = (0, 255, 0)  # GREEN (BGR)
            thickness = 2
        else:
            box_color = YoloDetector.CLASS_COLORS.get(cls, (180, 180, 180))
            thickness = 2

        # Label background: use same hue but bright enough to see
        BG_COLOR = tuple(max(30, int(c * 0.85)) for c in box_color)

        # ---- draw bbox ----
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)

        # ---- build label lines (clean) ----
        lines = []
        if tid is not None:
            lines.append(f"ID:{tid}")
        lines.append(f"{cls} {conf:.2f}")

        # ✅ ROI ratio line must be appended BEFORE measuring/drawing
        r = d.get("roi_inside_ratio", None)
        if r is not None:
            try:
                lines.append(f"ROI:{float(r):.2f}")
            except Exception:
                pass

        # ---- measure label block ----
        sizes = [cv2.getTextSize(t, FONT, FONT_SCALE, FONT_THICKNESS) for t in lines]
        widths = [s[0][0] for s in sizes]
        heights = [s[0][1] for s in sizes]

        text_w = max(widths) if widths else 0
        text_h = sum(heights) + (len(lines) - 1) * GAP_Y

        bg_w = text_w + PAD_X * 2
        bg_h = text_h + PAD_Y * 2

        # ---- label position (prefer above box) ----
        bg_x1 = max(0, min(x1, W - bg_w))
        bg_y2 = y1
        bg_y1 = bg_y2 - bg_h

        # if above is out -> put below
        if bg_y1 < 0:
            bg_y1 = min(H - bg_h, y1 + 2)
            bg_y2 = bg_y1 + bg_h

        bg_x2 = bg_x1 + bg_w
        bg_y2 = bg_y1 + bg_h

        # ---- draw label background ----
        cv2.rectangle(out, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), BG_COLOR, -1)

        # ---- draw text ----
        y_cursor = int(bg_y1 + PAD_Y)
        for i, t in enumerate(lines):
            th = sizes[i][0][1]  # text height
            ty = y_cursor + th
            cv2.putText(
                out,
                t,
                (int(bg_x1 + PAD_X), int(ty)),
                FONT,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS,
                LINE_TYPE,
            )
            y_cursor += th + GAP_Y

    return out


# =========================================================
# DeepSORT Tracking Layer (NEW CLASS)
# - One tracker per view_name (fisheye multi-view safe)
# - Track items only (recommended for lost-and-found)
# - Returns track objects you can draw/log
# =========================================================
def attach_track_ids_to_detections(detections, tracks, iou_thr=0.3):
    """
    Attach track_id to detections by best IoU match.
    """
    if not detections or not tracks:
        return detections

    for d in detections:
        bbox = d.get("bbox")
        if not bbox:
            continue

        best_iou = 0.0
        best_tid = None

        for t in tracks:
            tb = t.get("bbox")
            if not tb:
                continue

            iou = _iou_xyxy(bbox, tb)
            if iou > best_iou:
                best_iou = iou
                best_tid = t.get("track_id")

        if best_iou >= iou_thr:
            d["track_id"] = best_tid

    return detections

class DeepSortTrackerManager:
    def __init__(
        self,
        track_items_only: bool = False,
        max_age: int = 80,
        n_init: int = 1,
        max_iou_distance: float = 0.8,
        embedder: str = "mobilenet",
        half: bool = True,
        bgr: bool = True,
        # new safety/quality knobs
        min_det_conf: float = 0.10,
        min_box_area: float = 20 * 20,
        max_trackers: int = 16,
    ):
        self.track_items_only = track_items_only
        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.max_iou_distance = float(max_iou_distance)
        self.embedder = embedder
        self.half = bool(half)
        self.bgr = bool(bgr)

        self.min_det_conf = float(min_det_conf)
        self.min_box_area = float(min_box_area)
        self.max_trackers = int(max_trackers)

        self._trackers = {}          # view_name -> DeepSort
        self._tracker_last = {}      # view_name -> last_used_ts
        self._locks = {}             # view_name -> Lock
        self._global_lock = threading.Lock()

    def _ensure_view(self, view_name: str):
        # thread-safe creation + simple LRU cleanup
        with self._global_lock:
            if view_name not in self._locks:
                self._locks[view_name] = threading.Lock()

            if view_name not in self._trackers:
                try:
                    self._trackers[view_name] = DeepSort(
                        max_age=self.max_age,
                        n_init=self.n_init,
                        max_iou_distance=self.max_iou_distance,
                        embedder=self.embedder,
                        half=self.half,
                        bgr=self.bgr,
                        embedder_gpu=torch.cuda.is_available(),
                    )
                except TypeError:
                    self._trackers[view_name] = DeepSort(
                        max_age=self.max_age,
                        n_init=self.n_init,
                        max_iou_distance=self.max_iou_distance,
                        embedder=self.embedder,
                        half=self.half,
                        bgr=self.bgr,
                    )
            self._tracker_last[view_name] = time.time()

            # LRU eviction (optional)
            if len(self._trackers) > self.max_trackers:
                oldest = sorted(self._tracker_last.items(), key=lambda kv: kv[1])[: max(1, len(self._trackers) - self.max_trackers)]
                for k, _ in oldest:
                    self._trackers.pop(k, None)
                    self._tracker_last.pop(k, None)
                    self._locks.pop(k, None)

    @staticmethod
    def _clip_xyxy(b, W, H):
        x1, y1, x2, y2 = b
        x1 = max(0.0, min(float(x1), W - 1.0))
        y1 = max(0.0, min(float(y1), H - 1.0))
        x2 = max(0.0, min(float(x2), W - 1.0))
        y2 = max(0.0, min(float(y2), H - 1.0))
        # enforce proper ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    @staticmethod
    def _to_tlwh(bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def update(self, view_name: str, frame_bgr, detections: list):
        if frame_bgr is None:
            return []

        H, W = frame_bgr.shape[:2]
        self._ensure_view(view_name)

        # Prepare DeepSORT inputs + keep a clean det list for IoU confidence attachment
        clean_dets = []
        ds_inputs = []

        for d in detections or []:
            cls = d.get("class_name", "")
            if self.track_items_only and cls == "person":
                continue

            conf = float(d.get("confidence", 0.0))
            if conf < self.min_det_conf:
                continue

            bbox = d.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue

            bbox = self._clip_xyxy(bbox, W, H)
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue

            tlwh = self._to_tlwh(bbox)
            ds_inputs.append((tlwh, conf, cls))
            clean_dets.append({"bbox": bbox, "confidence": conf, "class_name": cls})

        # Update tracker (per-view lock)
        lock = self._locks.get(view_name)
        if lock is None:
            # fallback
            lock = self._global_lock

        with lock:
            tracker = self._trackers[view_name]
            tracks = tracker.update_tracks(ds_inputs, frame=frame_bgr)

        # Build output tracks and attach confidence via IoU-to-detection matching
        # Build output tracks and attach confidence via IoU-to-detection matching
        out = []
        for t in tracks:
            confirmed = bool(t.is_confirmed())
            x1, y1, x2, y2 = t.to_ltrb()
            tb = [float(x1), float(y1), float(x2), float(y2)]  # tracker box

            # match to best detection by IoU (use detection for class/conf/bbox)
            best_conf = 0.0
            best_iou = 0.0
            best_det = None

            for det in clean_dets:
                iou = _iou_xyxy(tb, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_conf = det["confidence"]
                    best_det = det

            # IMPORTANT:
            # - class_name should come from the matched detection (not t.get_det_class())
            # - det_bbox is the matched detection bbox (best for cropping)
            class_name = best_det["class_name"] if best_det is not None else str(
                getattr(t, "get_det_class", lambda: "unknown")())
            det_bbox = best_det["bbox"] if best_det is not None else None

            out.append({
                "track_id": int(t.track_id),
                "class_name": str(class_name),
                "bbox": tb,  # tracker bbox (for drawing if you want)
                "det_bbox": det_bbox,  # ✅ detection bbox (for cropping!)
                "iou_det": float(best_iou),  # optional debug
                "confirmed": confirmed,
                "confidence": float(best_conf),
            })

        return out


# =========================================================
# PHASE 6 — Lost & Found Manager (Time-based)
# PHASE 7 — Logging & Summary (JSONL + JSON)
# =========================================================
@dataclass
class TrackedObjectState:
    track_id: int
    class_name: str
    view_name: str
    roi_id: str

    first_seen_time: float
    last_seen_time: float
    duration_visible: float = 0.0

    is_marked_lost: bool = False
    lost_id: Optional[str] = None

    # Optional association / attendance logic
    last_attended_time: Optional[float] = None
    last_attended_by_person_id: Optional[int] = None
    last_bbox: Optional[List[float]] = None

    # NEW: ownership lock
    locked_owner_person_id: Optional[int] = None
    owner_candidate_person_id: Optional[int] = None
    owner_candidate_since: Optional[float] = None
    owner_absent_logged: bool = False
    owner_last_seen_time: Optional[float] = None


@dataclass
class LostItem:
    lost_id: str
    track_id: int
    class_name: str
    view_name: str
    roi_id: str

    lost_time: float
    duration_before_lost: float

    status: str = "pending"  # pending | verified | recovered
    image_path: Optional[str] = None

    # for “who likely owned it”
    owner_person_id: Optional[int] = None
    last_attended_time: Optional[float] = None


class ScalableIDGenerator:
    """Generate simplified and meaningful object IDs with format: venue_ROI_CLASSNAME_randomhash"""

    def __init__(self, venue_name: str = None, config_path: str = LOSTFOUND_BACKEND_DIR/"config.json", video_hash: str = None):
        # Load venue name from config.json if not provided
        if venue_name is None:
            try:
                config = load_config(config_path)
                if "venues" in config:
                    # Multi-venue format
                    default_venue = config.get("default_venue", "5GLab")
                    if default_venue in config["venues"]:
                        venue_name = config["venues"][default_venue].get("lab_name", "5GLab")
                    else:
                        venue_name = "5GLab"
                else:
                    # Legacy single-venue format
                    venue_name = config.get("lab_name", "5GLab")
            except Exception as e:
                print(f"Warning: Could not load venue name from config: {e}")
                venue_name = "5GLab"  # Fallback default

        # Simplify venue name for ID generation (max 6 chars, clean format)
        clean_venue = venue_name.replace(" ", "").replace("_", "").upper()
        self.venue_id = clean_venue[:6] if len(clean_venue) > 6 else clean_venue

        self.venue_name = venue_name  # Keep original for reference
        self.object_counts = {}  # Track counts per ROI-class combination
        self.generated_ids = set()  # Track all generated IDs to avoid duplicates

        print(f"ScalableIDGenerator initialized - Venue: {venue_name} -> ID: {self.venue_id}")

    def generate_object_id(self, class_name: str, roi_index: int, timestamp: float = None) -> str:
        """
        Generate simplified object ID: venue_ROI_CLASSNAME_randomhash

        Examples:
        - 5GLAB_ROI0_CELLPH_A1B2
        - 5GLAB_ROI1_BOTTLE_C3D4
        - 5GLAB_ROI2_CELLPH_E5F6
        """
        # Normalize class name
        class_short = class_name.upper().replace(" ", "").replace("_", "")
        if "CELL" in class_short or "PHONE" in class_short:
            class_short = "CELLPH"
        elif "BOTTLE" in class_short:
            class_short = "BOTTLE"
        elif "LAPTOP" in class_short:
            class_short = "LAPTOP"
        elif "BOOK" in class_short:
            class_short = "BOOK"
        else:
            class_short = class_short[:6]  # Limit to 6 chars

        # Generate unique random hash
        import random
        import hashlib

        max_attempts = 100
        for attempt in range(max_attempts):
            # Generate 4-character random hash using timestamp and random data
            timestamp_str = str(timestamp or time.time())
            random_str = f"{random.randint(1000, 9999)}{attempt}"
            hash_input = f"{self.venue_id}_{roi_index}_{class_short}_{timestamp_str}_{random_str}"

            # Create 4-character hash
            hash_obj = hashlib.md5(hash_input.encode())
            random_hash = hash_obj.hexdigest()[:4].upper()

            # Build object ID: venue_ROI_CLASSNAME_randomhash
            object_id = f"{self.venue_id}_ROI{roi_index}_{class_short}_{random_hash}"

            # Check for uniqueness
            if object_id not in self.generated_ids:
                self.generated_ids.add(object_id)

                # Update counts
                roi_class_key = f"{roi_index}_{class_short}"
                self.object_counts[roi_class_key] = self.object_counts.get(roi_class_key, 0) + 1

                print(f"Generated new simplified object ID: {object_id}")
                return object_id

        # Fallback if somehow we can't generate unique ID
        fallback_id = f"{self.venue_id}_ROI{roi_index}_{class_short}_{random.randint(1000, 9999):04X}"
        print(f"Using fallback object ID: {fallback_id}")
        return fallback_id

class JsonlEventLogger:
    """
    Phrase 7: structured event logger to .jsonl
    """
    def __init__(self, out_path: str):
        self.out_path = out_path
        self._lock = Lock()

    def log(self, event_type: str, payload: dict):
        rec = {
            "ts": time.time(),
            "event": event_type,
            "payload": payload,
        }
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        if DEBUG_EVENTS:
            print("[EVENT]", event_type)

def make_run_id(prefix="fisheyeV1_20"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"

def init_run_dirs(run_id: str):
    # Always save inside lostfound_backend/
    base = LOSTFOUND_BACKEND_DIR / "outputs" / "lost_and_found" / run_id
    snap_base = LOSTFOUND_BACKEND_DIR / "snapshots" / run_id

    base.mkdir(parents=True, exist_ok=True)
    snap_base.mkdir(parents=True, exist_ok=True)

    return {
        "run_id": run_id,
        "base": str(base),
        "progress": str(base / "progress.json"),
        "lost_json": str(base / "lost_items.json"),
        "lost_csv": str(base / "lost_items.csv"),
        "event_log": str(base / "event_log.jsonl"),
        "snapshots": str(snap_base),
    }


def setup_run(prefix: str) -> dict:
    forced_base = (os.getenv("LF_OUTPUT_DIR") or "").strip()

    if forced_base:
        base_dir = Path(forced_base)
        if not base_dir.is_absolute():
            base_dir = LOSTFOUND_BACKEND_DIR / forced_base
        base_dir.mkdir(parents=True, exist_ok=True)

        run_id = make_run_id(prefix)

        paths = {
            "run_id": run_id,
            "base": str(base_dir),
            "out_base": str(base_dir),
            "progress": str(base_dir / "progress.json"),
            "event_log": str(base_dir / "event_log.jsonl"),
            "lost_json": str(base_dir / "lost_items.json"),
            "lost_csv": str(base_dir / "lost_items.csv"),
            "snapshots": str(base_dir / "snapshots"),
        }
        Path(paths["snapshots"]).mkdir(parents=True, exist_ok=True)

        try:
            if not os.path.exists(paths["event_log"]):
                open(paths["event_log"], "a", encoding="utf-8").close()
            if not os.path.exists(paths["lost_json"]):
                with open(paths["lost_json"], "w", encoding="utf-8") as f:
                    f.write("[]\n")
            if not os.path.exists(paths["lost_csv"]):
                open(paths["lost_csv"], "a", encoding="utf-8").close()
        except Exception:
            pass

    else:
        run_id = make_run_id(prefix)
        paths = init_run_dirs(run_id)

    event_logger = JsonlEventLogger(paths["event_log"])

    lost_manager = LostAndFoundManager(
        lost_seconds=6.0,
        disappear_seconds=5.0,
        enable_snapshots=True,
        snapshot_dir=paths["snapshots"],
        enable_owner_association=True,

        near_px=140.0,              # tighter than before
        unattended_seconds=5.0,

        owner_lock_px=120.0,        # stricter for owner locking
        owner_lock_seconds=1.0,   

        logger=event_logger,
        autosave_json_path=paths["lost_json"],
        autosave_csv_path=paths["lost_csv"],
        owner_grace_seconds = 4.0,
        item_grace_seconds = 5.0,
    )

    return {
        "paths": paths,
        "event_logger": event_logger,
        "lost_manager": lost_manager,
    }
class LostAndFoundManager:
    """
    Phrase 6 core logic:
      - Maintain per-track lifecycle
      - Decide LOST when:
          ✅ duration_visible >= lost_seconds
          ✅ MUST have an owner (person associated)
          ✅ unattended_seconds >= unattended_seconds
      - Create LostItem records
      - Provide summary save
    """
    def __init__(
        self,
        lost_seconds: float = 20.0,
        disappear_seconds: float = 20.0,
        enable_snapshots: bool = True,
        snapshot_dir: str = None,
        enable_owner_association: bool = True,
        near_px: float = 140.0,
        unattended_seconds: float = 10.0,
        logger: Optional["JsonlEventLogger"] = None,
        autosave_json_path=None,
        autosave_csv_path=None,
        autosave_every=3.0,

        # NEW
        owner_lock_px: float = 90.0,
        owner_lock_seconds: float = 2.0,
        owner_grace_seconds: float = 3.0,
        item_grace_seconds: float = 4.0,
    ):
        self.lost_seconds = float(lost_seconds)
        self.disappear_seconds = float(disappear_seconds)
        self.enable_snapshots = bool(enable_snapshots)
        self.snapshot_dir = snapshot_dir or str(LOSTFOUND_BACKEND_DIR / "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        if autosave_json_path is None or autosave_csv_path is None:
            run_id = time.strftime("run_%Y%m%d_%H%M%S")
            output_base = LOSTFOUND_BACKEND_DIR / "outputs" / "lostandfounds" / run_id
            output_base.mkdir(parents=True, exist_ok=True)

            autosave_json_path = autosave_json_path or str(output_base / "lost_items.json")
            autosave_csv_path = autosave_csv_path or str(output_base / "lost_items.csv")

        self.enable_owner_association = bool(enable_owner_association)
        self.near_px = float(near_px)
        self.unattended_seconds = float(unattended_seconds)

        self.logger = logger

        self.states: Dict[Tuple[str, int], "TrackedObjectState"] = {}
        self.lost_items: Dict[str, "LostItem"] = {}
        self._lock = Lock()

        self.autosave_json_path = autosave_json_path
        self.autosave_csv_path = autosave_csv_path
        self.autosave_every = float(autosave_every)
        self._last_autosave_ts = 0.0

        self.venue_code = "B001G_B_Block_B"
        self.id_gen = ScalableIDGenerator(venue_name=self.venue_code, config_path=LOSTFOUND_BACKEND_DIR/"config.json")
        self.snapshot_dedupe_enabled = True
        self.snapshot_dedupe_every = 60.0
        self.snapshot_similarity_threshold = 0.96
        self.snapshot_time_window_sec = 120.0 
        self._last_snapshot_dedupe_ts = 0.0
        self.owner_lock_px = float(owner_lock_px)
        self.owner_lock_seconds = float(owner_lock_seconds)
        self.owner_grace_seconds = float(owner_grace_seconds)
        self.item_grace_seconds = float(item_grace_seconds)
        
    @staticmethod
    def _center_xy(b):
        x1, y1, x2, y2 = b
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _person_foot_xy(b):
        x1, y1, x2, y2 = b
        return ((x1 + x2) * 0.5, y2)

    @staticmethod
    def _dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    def _class_to_code(self, class_name: str) -> str:
        n = (class_name or "UNK").lower().strip()

        if "cell" in n or "phone" in n:
            return "CELLPH"
        if "bottle" in n:
            return "BOTTLE"
        if "laptop" in n:
            return "LAPTOP"
        if "tablet" in n:
            return "TABLET"
        if "backpack" in n or "bag" in n:
            return "BAG"
        if "handbag" in n:
            return "HANDBG"
        if "wallet" in n:
            return "WALLET"
        if "key" in n:
            return "KEY"
        if "earphone" in n or "headphone" in n:
            return "EARPHN"
        if "usb" in n:
            return "USB"
        if "umbrella" in n:
            return "UMBREL"
        if "book" in n:
            return "BOOK"

        # fallback
        return "".join(ch for ch in n.upper() if ch.isalnum())[:6] or "UNK"

    def _roi_to_index(self, roi_id: str) -> int:
        s = str(roi_id)
        digits = "".join([c for c in s if c.isdigit()])
        return int(digits) if digits else -1

    def _lost_dir(self, roi_id: str) -> str:
        roi_index = self._roi_to_index(roi_id)
        base = os.path.join(self.snapshot_dir, "lost_item")
        folder = os.path.join(base, f"roi_{roi_index if roi_index >= 0 else 'X'}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _next_seq(self, folder: str, prefix: str) -> int:
        try:
            files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".jpg")]
            if not files:
                return 1
            nums = []
            for f in files:
                remain = f[len(prefix):].split("_", 1)[0]
                nums.append(int(remain))
            return max(nums) + 1 if nums else 1
        except Exception:
            return 1

    def _crop_bbox(
        self,
        frame_bgr,
        bbox,
        pad_x=20,
        pad_y=20,
        min_w=120,
        min_h=120,
        square=False,
        max_expand_ratio=2.5,
    ):
        if frame_bgr is None or bbox is None:
            return None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = map(float, bbox)

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # adaptive expansion but prevent over-expanding too much
        expand_x = max(float(pad_x), bw * 0.6)
        expand_y = max(float(pad_y), bh * 0.6)

        expand_x = min(expand_x, bw * float(max_expand_ratio))
        expand_y = min(expand_y, bh * float(max_expand_ratio))

        x1 -= expand_x
        x2 += expand_x
        y1 -= expand_y
        y2 += expand_y

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cw = x2 - x1
        ch = y2 - y1

        if square:
            side = max(cw, ch, float(min_w), float(min_h))
            cw = side
            ch = side
        else:
            cw = max(cw, float(min_w))
            ch = max(ch, float(min_h))

        x1 = int(round(cx - cw / 2.0))
        x2 = int(round(cx + cw / 2.0))
        y1 = int(round(cy - ch / 2.0))
        y2 = int(round(cy + ch / 2.0))

        # clamp and shift back into frame
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            shift = x2 - w
            x1 -= shift
            x2 = w
        if y2 > h:
            shift = y2 - h
            y1 -= shift
            y2 = h

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2].copy()
        return crop if crop.size > 0 else None
    
    def _snapshot_crop_config(self, class_name: str, bbox=None):
        cls = str(class_name or "").strip().lower()

        bw = bh = 0.0
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = map(float, bbox)
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)

        # small flat objects -> keep more context
        if cls in ("mobile_phone", "phone", "cell_phone", "cellphone"):
            return {
                "pad_x": 45,
                "pad_y": 45,
                "min_w": 220,
                "min_h": 220,
                "square": True,
                "max_expand_ratio": 4.0,
            }

        # water bottle -> mimic Version 2 tighter look
        if cls in ("water_bottle", "bottle"):
            return {
                "pad_x": 35,
                "pad_y": 35,
                "min_w": 120,
                "min_h": 160,
                "square": False,
                "max_expand_ratio": 2.5,
            }

        # tablets -> moderate context
        if cls in ("tablet",):
            return {
                "pad_x": 28,
                "pad_y": 28,
                "min_w": 170,
                "min_h": 170,
                "square": False,
                "max_expand_ratio": 2.2,
            }

        # medium items
        if cls in ("backpack", "bag", "handbag", "laptop"):
            return {
                "pad_x": 24,
                "pad_y": 24,
                "min_w": 160,
                "min_h": 140,
                "square": False,
                "max_expand_ratio": 2.2,
            }

        # generic small object fallback
        if bw <= 60 or bh <= 60:
            return {
                "pad_x": 30,
                "pad_y": 30,
                "min_w": 180,
                "min_h": 180,
                "square": True,
                "max_expand_ratio": 3.0,
            }

        # default
        return {
            "pad_x": 20,
            "pad_y": 20,
            "min_w": 130,
            "min_h": 130,
            "square": False,
            "max_expand_ratio": 2.0,
        }

    def _snapshot_signature_group(self, item) -> tuple:
        """
        Group only truly comparable lost items together.

        Compare only within same:
        - class
        - view
        - roi
        - owner
        """
        return (
            str(getattr(item, "class_name", "") or "").strip().lower(),
            str(getattr(item, "view_name", "") or "").strip().lower(),
            str(getattr(item, "roi_id", "") or "").strip().lower(),
            str(getattr(item, "owner_person_id", None) or "no_owner").strip().lower(),
        )

    def _load_gray_for_hash(self, path: str):
        try:
            if not path or not os.path.exists(path):
                return None
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                return None
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            return img
        except Exception:
            return None

    def _ahash_bits(self, gray32):
        if gray32 is None:
            return None
        try:
            small = cv2.resize(gray32, (8, 8), interpolation=cv2.INTER_AREA)
            mean = float(small.mean())
            bits = (small >= mean).astype(np.uint8).flatten()
            return bits
        except Exception:
            return None

    def _hash_similarity(self, path_a: str, path_b: str) -> float:
        ga = self._load_gray_for_hash(path_a)
        gb = self._load_gray_for_hash(path_b)
        if ga is None or gb is None:
            return 0.0

        ba = self._ahash_bits(ga)
        bb = self._ahash_bits(gb)
        if ba is None or bb is None or len(ba) != len(bb):
            return 0.0

        same = float((ba == bb).sum())
        total = float(len(ba))
        return same / total if total > 0 else 0.0

    def _snapshot_quality_score(self, path: str) -> float:
        try:
            if not path or not os.path.exists(path):
                return -1e9

            img = cv2.imread(path)
            if img is None or img.size == 0:
                return -1e9

            h, w = img.shape[:2]
            area_score = float(w * h)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())

            # prefer normal brightness, not too dark / too bright
            brightness_penalty = abs(brightness - 140.0)

            return area_score + (sharpness * 8.0) - (brightness_penalty * 25.0)
        except Exception:
            return -1e9
        
    def _dedupe_snapshots_periodic(self, now_ts: float, force: bool = False):
        """
        Snapshot dedupe:
        - compare only same class/view/roi/owner
        - compare only within time window
        - keep better record + better image
        - delete duplicate physical image files
        - delete duplicate lost_item record from self.lost_items
        """
        if not self.snapshot_dedupe_enabled:
            return False

        if not force and (now_ts - self._last_snapshot_dedupe_ts) < self.snapshot_dedupe_every:
            return False

        changed = False
        self._last_snapshot_dedupe_ts = now_ts

        with self._lock:
            items = list(self.lost_items.values())

            groups = defaultdict(list)
            for it in items:
                if not it:
                    continue
                if not getattr(it, "image_path", None):
                    continue
                if not os.path.exists(it.image_path):
                    continue
                groups[self._snapshot_signature_group(it)].append(it)

            lost_ids_to_delete = set()

            for _, group_items in groups.items():
                if len(group_items) < 2:
                    continue

                group_items.sort(key=lambda x: float(getattr(x, "lost_time", 0.0) or 0.0))
                keepers = []

                for cur in group_items:
                    cur_lost_id = str(getattr(cur, "lost_id", "") or "").strip()
                    cur_path = str(getattr(cur, "image_path", "") or "").strip()

                    if not cur_lost_id or not cur_path:
                        continue
                    if not os.path.exists(cur_path):
                        continue
                    if cur_lost_id in lost_ids_to_delete:
                        continue

                    matched_keeper = None

                    for kept in keepers:
                        kept_lost_id = str(getattr(kept, "lost_id", "") or "").strip()
                        kept_path = str(getattr(kept, "image_path", "") or "").strip()

                        if not kept_lost_id or not kept_path:
                            continue
                        if kept_lost_id in lost_ids_to_delete:
                            continue
                        if not os.path.exists(kept_path):
                            continue

                        # only compare near timestamps
                        if abs(
                            float(getattr(cur, "lost_time", 0.0) or 0.0)
                            - float(getattr(kept, "lost_time", 0.0) or 0.0)
                        ) > self.snapshot_time_window_sec:
                            continue

                        sim = self._hash_similarity(cur_path, kept_path)
                        if sim >= self.snapshot_similarity_threshold:
                            matched_keeper = kept
                            break

                    if matched_keeper is None:
                        keepers.append(cur)
                        continue

                    cur_score = self._snapshot_quality_score(cur_path)

                    keep_lost_id = str(getattr(matched_keeper, "lost_id", "") or "").strip()
                    keep_path = str(getattr(matched_keeper, "image_path", "") or "").strip()
                    keep_score = self._snapshot_quality_score(keep_path)

                    if cur_score > keep_score:
                        # current one is better -> current stays, old keeper removed
                        old_keep_path = keep_path
                        old_keep_id = keep_lost_id

                        # delete old keeper image
                        if old_keep_path and old_keep_path != cur_path and os.path.exists(old_keep_path):
                            try:
                                os.remove(old_keep_path)
                            except Exception:
                                pass

                        # mark old keeper record for deletion
                        if old_keep_id:
                            lost_ids_to_delete.add(old_keep_id)

                        # replace keeper in keepers list with current
                        keepers = [cur if k is matched_keeper else k for k in keepers]

                    else:
                        # old keeper stays -> delete current image + current record
                        if cur_path != keep_path and os.path.exists(cur_path):
                            try:
                                os.remove(cur_path)
                            except Exception:
                                pass

                        if cur_lost_id:
                            lost_ids_to_delete.add(cur_lost_id)

                    changed = True

            # remove duplicate records from lost_items
            for lost_id in lost_ids_to_delete:
                self.lost_items.pop(lost_id, None)

        return changed
    
    def _dedup_lost_items(self, items: list) -> list:
        """
        Deduplicate lost items ONLY by lost_id.

        IMPORTANT:
        - Do NOT dedupe by class_name / roi_id / image_path
        - Different records may legitimately share similar image content
        - We want to preserve event records even if snapshot images are similar
        """
        by_id = {}

        for it in items or []:
            if not isinstance(it, dict):
                continue

            lid = str(it.get("lost_id") or "").strip()
            if not lid:
                continue

            prev = by_id.get(lid)
            if prev is None:
                by_id[lid] = it
                continue

            # keep earlier lost_time if duplicate lost_id appears
            try:
                cur_ts = float(it.get("lost_time", 1e18) or 1e18)
            except Exception:
                cur_ts = 1e18

            try:
                prev_ts = float(prev.get("lost_time", 1e18) or 1e18)
            except Exception:
                prev_ts = 1e18

            if cur_ts < prev_ts:
                by_id[lid] = it

        out = list(by_id.values())
        out.sort(key=lambda x: float(x.get("lost_time", 0.0) or 0.0))
        return out

    def set_lost_item_status(self, lost_id: str, new_status: str):
        with self._lock:
            if lost_id in self.lost_items:
                self.lost_items[lost_id].status = new_status
                if self.logger:
                    self.logger.log("lost_item_status_changed", {
                        "lost_id": lost_id,
                        "new_status": new_status
                    })

    def get_active_lost_items(self) -> List[dict]:
        with self._lock:
            return [asdict(v) for v in self.lost_items.values() if v.status != "recovered"]

    def save_summary(self, out_path: str):
        with self._lock:
            payload = {
                "generated_at": time.time(),
                "lost_items": [asdict(v) for v in self.lost_items.values()],
            }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _autosave_if_needed(self, now_ts: float, force: bool = False):
        if not self.autosave_json_path and not self.autosave_csv_path:
            return

        if (not force) and (now_ts - self._last_autosave_ts < self.autosave_every):
            return

        try:
            if self.autosave_json_path:
                self.export_lost_items_json(self.autosave_json_path)
            if self.autosave_csv_path:
                self.export_lost_items_csv(self.autosave_csv_path)
            self._last_autosave_ts = now_ts
            print(f"[AUTO-SAVE] wrote {self.autosave_json_path} / {self.autosave_csv_path}")
        except Exception as e:
            print("[AUTO-SAVE] failed:", e)

    def process_tracks(
        self,
        now_ts: float,
        view_name: str,
        roi_id: str,
        tracked_objects: List[dict],
        frame_bgr=None,
    ):
        """
        tracked_objects: list of dicts with:
        track_id, class_name, bbox, confirmed, confidence, det_bbox(optional)
        If enable_owner_association=True, you MUST pass BOTH person tracks + item tracks.
        """
        new_lost_created = False

        with self._lock:

            def _as_dict(t):
                if isinstance(t, dict):
                    return t

                d = {}
                if hasattr(t, "track_id"):
                    d["track_id"] = int(t.track_id)

                if hasattr(t, "to_ltrb"):
                    try:
                        d["bbox"] = list(map(float, t.to_ltrb()))
                    except Exception:
                        pass
                elif hasattr(t, "to_tlbr"):
                    try:
                        d["bbox"] = list(map(float, t.to_tlbr()))
                    except Exception:
                        pass

                if hasattr(t, "det_class"):
                    d["class_name"] = t.det_class
                if hasattr(t, "class_name"):
                    d["class_name"] = t.class_name

                if hasattr(t, "det_conf"):
                    d["confidence"] = float(t.det_conf)

                d["confirmed"] = True
                return d

            tracked_objects = [_as_dict(x) for x in (tracked_objects or [])]

            def _cls(x):
                return x.get("class_name") or x.get("class") or x.get("label") or x.get("det_class") or x.get("cls")

            def _bbox(x):
                return x.get("bbox") or x.get("tlbr") or x.get("bbox_xyxy")

            persons = [t for t in tracked_objects if (_cls(t) == "person") and t.get("confirmed", True)]
            items = [t for t in tracked_objects if (_cls(t) not in (None, "person")) and t.get("confirmed", True)]

            seen_keys = set()

            for t in items:
                tid = int(t.get("track_id", -1))
                if tid < 0:
                    continue

                key = (view_name, tid)
                seen_keys.add(key)

                cls = str(_cls(t) or "unknown")
                st = self.states.get(key)
                if st is None:
                    st = TrackedObjectState(
                        track_id=tid,
                        class_name=cls,
                        view_name=view_name,
                        roi_id=str(roi_id),
                        first_seen_time=now_ts,
                        last_seen_time=now_ts,
                    )

                    # IMPORTANT: do not assume attended at start
                    st.last_attended_time = None
                    st.last_attended_by_person_id = None
                    st.owner_absent_logged = False

                    self.states[key] = st

                    if self.logger:
                        self.logger.log("tracking_started", {
                            "view": view_name,
                            "roi_id": roi_id,
                            "track_id": tid,
                            "class_name": cls
                        })

                # update timing
                st.last_seen_time = now_ts
                st.duration_visible = max(0.0, st.last_seen_time - st.first_seen_time)

                # Prefer detection bbox for cropping, fallback to tracker bbox
                bb_det = t.get("det_bbox", None)
                bb_track = _bbox(t)
                bb_use = bb_det if (bb_det is not None and len(bb_det) == 4) else bb_track
                bb_item = bb_use

                if bb_use is not None:
                    st.last_bbox = list(map(float, bb_use))

                # -------------------------
                # Owner association with LOCK
                # -------------------------
                if self.enable_owner_association and bb_item is not None:
                    item_c = self._center_xy(bb_item)

                    best_pid = None
                    best_dist = 1e9

                    for p in persons:
                        pid = int(p.get("track_id", -1))
                        bb_p = _bbox(p)
                        if pid < 0 or bb_p is None:
                            continue

                        pfoot = self._person_foot_xy(bb_p)
                        dpx = self._dist(item_c, pfoot)

                        if dpx < best_dist:
                            best_dist = dpx
                            best_pid = pid

                    # STEP A: no locked owner yet -> try to lock one
                    if st.locked_owner_person_id is None:
                        if best_pid is not None and best_dist <= self.owner_lock_px:
                            if st.owner_candidate_person_id != best_pid:
                                st.owner_candidate_person_id = best_pid
                                st.owner_candidate_since = now_ts
                            else:
                                if (
                                    st.owner_candidate_since is not None
                                    and (now_ts - st.owner_candidate_since) >= self.owner_lock_seconds
                                ):
                                    st.locked_owner_person_id = best_pid
                                    st.last_attended_by_person_id = best_pid
                                    st.last_attended_time = now_ts
                                    st.owner_absent_logged = False
                                    st.owner_last_seen_time = now_ts

                                    if self.logger:
                                        self.logger.log("owner_locked", {
                                            "view": view_name,
                                            "roi_id": roi_id,
                                            "track_id": tid,
                                            "class_name": st.class_name,
                                            "owner_person_id": best_pid,
                                            "lock_distance_px": round(best_dist, 2),
                                        })
                        else:
                            st.owner_candidate_person_id = None
                            st.owner_candidate_since = None

                    # STEP B: already have locked owner -> only locked owner can attend
                    else:
                        locked_pid = st.locked_owner_person_id
                        locked_found = False
                        locked_dist = None

                        for p in persons:
                            pid = int(p.get("track_id", -1))
                            if pid != locked_pid:
                                continue

                            bb_p = _bbox(p)
                            if bb_p is None:
                                continue

                            pfoot = self._person_foot_xy(bb_p)
                            dpx = self._dist(item_c, pfoot)

                            locked_found = True
                            locked_dist = dpx
                            break

                        if locked_found and locked_dist is not None and locked_dist <= self.near_px:
                            st.last_attended_time = now_ts
                            st.last_attended_by_person_id = locked_pid
                            st.owner_last_seen_time = now_ts
                            st.owner_absent_logged = False
                        else:
                            if st.owner_last_seen_time is None and st.last_attended_time is not None:
                                st.owner_last_seen_time = st.last_attended_time
                # -------------------------
                # DEBUG: owner absent tracking (log once only)
                # -------------------------
                if st.locked_owner_person_id is not None and st.last_attended_time is not None:
                    base_attended_time = st.last_attended_time
                    if (
                        st.owner_last_seen_time is not None
                        and (now_ts - st.owner_last_seen_time) <= self.owner_grace_seconds
                    ):
                        base_attended_time = now_ts

                    unattended_for = now_ts - base_attended_time if base_attended_time is not None else 1e9

                    if unattended_for >= self.unattended_seconds and not st.is_marked_lost:
                        if self.logger and not getattr(st, "owner_absent_logged", False):
                            self.logger.log("owner_absent_threshold_reached", {
                                "view": view_name,
                                "roi_id": roi_id,
                                "track_id": tid,
                                "class_name": st.class_name,
                                "owner_person_id": st.locked_owner_person_id,
                                "unattended_for": round(unattended_for, 2),
                            })
                            st.owner_absent_logged = True

                # -------------------------
                # LOST decision (OWNER REQUIRED)
                # -------------------------
                has_owner = (st.locked_owner_person_id is not None)
                unattended_ok = False

                base_attended_time = st.last_attended_time
                if (
                    st.owner_last_seen_time is not None
                    and (now_ts - st.owner_last_seen_time) <= self.owner_grace_seconds
                ):
                    base_attended_time = now_ts

                if has_owner and base_attended_time is not None:
                    unattended_ok = (now_ts - base_attended_time) >= self.unattended_seconds

                if (
                    (not st.is_marked_lost)
                    and has_owner
                    and st.duration_visible >= self.lost_seconds
                    and unattended_ok
                ):
                    roi_index = self._roi_to_index(st.roi_id)
                    lost_id = self.id_gen.generate_object_id(st.class_name, roi_index, timestamp=now_ts)

                    st.is_marked_lost = True
                    st.lost_id = lost_id

                    img_path = None
                    if self.enable_snapshots and frame_bgr is not None:
                        cfg = self._snapshot_crop_config(st.class_name, st.last_bbox)

                        crop = self._crop_bbox(
                            frame_bgr,
                            st.last_bbox,
                            pad_x=cfg["pad_x"],
                            pad_y=cfg["pad_y"],
                            min_w=cfg["min_w"],
                            min_h=cfg["min_h"],
                            square=cfg["square"],
                            max_expand_ratio=cfg["max_expand_ratio"],
                        )

                        # quality guard
                        if crop is not None:
                            h, w = crop.shape[:2]
                            if st.class_name in ("water_bottle", "bottle"):
                               if w < 50 or h < 90:
                                    crop = None
                            elif st.class_name in ("mobile_phone", "phone", "cell_phone", "cellphone"):
                                if w < 120 or h < 120:
                                    crop = None
                            else:
                                if w < 100 or h < 100:
                                    crop = None

                        if crop is not None:
                            roi_index = self._roi_to_index(st.roi_id)
                            class_code = self._class_to_code(st.class_name)
                            roi_code = f"R{roi_index}" if roi_index >= 0 else "RX"

                            folder = self._lost_dir(st.roi_id)
                            prefix = f"{self.venue_code}_{class_code}_{roi_code}_LOST_"
                            seq = self._next_seq(folder, prefix)

                            ts_str = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"{prefix}{seq:03d}_{ts_str}_{lost_id}.jpg"
                            img_path = os.path.join(folder, filename)

                            try:
                                cv2.imwrite(img_path, crop)
                            except Exception:
                                img_path = None

                    lost_item = LostItem(
                        lost_id=lost_id,
                        track_id=tid,
                        class_name=st.class_name,
                        view_name=view_name,
                        roi_id=st.roi_id,
                        lost_time=now_ts,
                        duration_before_lost=st.duration_visible,
                        status="pending",
                        image_path=img_path,
                        owner_person_id=st.locked_owner_person_id,
                        last_attended_time=st.last_attended_time,
                    )

                    if lost_id not in self.lost_items:
                        self.lost_items[lost_id] = lost_item
                        new_lost_created = True
                        if self.logger:
                            self.logger.log("lost_item_detected", asdict(lost_item))

            # Cleanup disappeared
            to_remove = []
            for key, st in list(self.states.items()):
                if key[0] != view_name:
                    continue
                if key in seen_keys:
                    continue
                if (now_ts - st.last_seen_time) >= (self.disappear_seconds + self.item_grace_seconds):
                    to_remove.append(key)

            for key in to_remove:
                st = self.states.pop(key, None)
                if st and self.logger:
                    self.logger.log("tracking_finished", {
                        "view": view_name,
                        "roi_id": roi_id,
                        "track_id": st.track_id,
                        "class_name": st.class_name
                    })

        # autosave outside lock; force only if new lost created
        self._autosave_if_needed(now_ts, force=new_lost_created)

        dedupe_changed = self._dedupe_snapshots_periodic(now_ts, force=new_lost_created)
        if dedupe_changed:
            self._autosave_if_needed(now_ts, force=True)

        if int(now_ts) % 1 == 0:
            if DEBUG_LOST_FOUND:
                print(f"[P6] view={view_name} items_states={len(self.states)} lost_items={len(self.lost_items)}")
    
    def export_lost_items_json(self, out_path: str):
        with self._lock:
            data = [asdict(v) for v in self.lost_items.values()]
        data = self._dedup_lost_items(data)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_lost_items_csv(self, out_path: str):
        with self._lock:
            rows = [asdict(v) for v in self.lost_items.values()]
        rows = self._dedup_lost_items(rows)

        fieldnames = [
            "lost_id", "track_id", "class_name", "view_name", "roi_id",
            "lost_time", "duration_before_lost", "status", "image_path",
            "owner_person_id", "last_attended_time"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------
#  ROI Calling
# ---------------------------------------------------------
def launch_roi_editor(video_path):
    """
    Launch correct ROI editor script using a path relative to this file,
    so it works no matter where Python is started from.
    """
    vtype = detect_video_type(video_path)

    if str(vtype).lower() == "fisheye":
        script_path = LOSTFOUND_BACKEND_DIR / "annotate_fisheye_views.py"
    else:
        script_path = LOSTFOUND_BACKEND_DIR / "annotate_rois.py"

    ret = subprocess.call([sys.executable, str(script_path), str(video_path)])

    if ret != 0:
        raise SystemExit("ROI editor aborted by user")

    return vtype


def roi_config_exists(config_path=LOSTFOUND_BACKEND_DIR/"config.json") -> bool:
    try:
        p = Path(config_path).resolve()
        if not p.exists():
            return False

        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}

        # normal ROI
        has_normal = bool(cfg.get("bounding_polygons"))

        # fisheye ROI
        fisheye_polys = cfg.get("fisheye_polygons", {})
        has_fisheye = isinstance(fisheye_polys, dict) and len(fisheye_polys.keys()) > 0

        return has_normal or has_fisheye

    except Exception:
        return False


def roi_mode_choice(video_path: str, config_path=LOSTFOUND_BACKEND_DIR/"config.json"):
    """
    Returns:
      - "redraw" -> launch ROI editor
      - "reuse"  -> skip ROI editor
    """
    # backend can override config path
    env_config = (os.getenv("LF_ROI_CONFIG") or "").strip()
    if env_config:
        config_path = env_config

    config_path = str(Path(config_path).resolve())

    # ✅ BACKEND / HEADLESS OVERRIDE:
    # If running from backend (or headless), NEVER prompt user.
    # Always reuse if config exists (or create empty and reuse).
    if os.getenv("LF_BACKEND", "0") == "1" or os.getenv("LF_HEADLESS", "0") == "1":
        exists = roi_config_exists(config_path)
        if not exists:
            try:
                p = Path(config_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"bounding_polygons": [], "fisheye_polygons": {}}, f, indent=2)
                print(f"[ROI] Backend: created empty config at {p}")
            except Exception as e:
                print(f"[ROI] Backend: failed to create config at {config_path}: {e}")

        print(f"[ROI] Backend: reuse config -> {config_path}")
        return "reuse"

    # -------------------------
    # original interactive CLI behavior
    # -------------------------
    exists = roi_config_exists(config_path)

    if not exists:
        print("[ROI] No existing ROI config found -> forcing ROI setup.")
        launch_roi_editor(video_path)
        return "redraw"

    print("\n[ROI] ROI config found.")
    print("1 = Redraw ROI now")
    print("2 = Reuse existing ROI and continue")
    choice = input("Choose (1/2): ").strip()

    if choice == "1":
        launch_roi_editor(video_path)
        return "redraw"

    print("[ROI] Reusing existing ROI config.")
    return "reuse"


# ---------------------------------------------------------
#  Active View Controller
# ---------------------------------------------------------
class ActiveViews:
    def __init__(self, groups):
        self.groups = groups
        self.idx = 0
        self.lock = threading.Lock()

    def toggle(self):
        with self.lock:
            self.idx = (self.idx + 1) % len(self.groups)

    def current(self):
        with self.lock:
            return set(self.groups[self.idx]), self.idx


active_views = ActiveViews(FISHEYE_GROUPS)


# ==========================================================
# ✅ REAL-TIME FISHEYE TUNING HELPERS (yaw / pitch / fov only)
# ==========================================================
def handle_realtime_tuning(
    key,
    preprocessor,
    out_views,
    tune_state,
    step_yaw=3.0,
    step_pitch=3.0,
    step_fov=3.0,
):
    """
    Real-time keyboard tuning for fisheye views.
    Keys:
      1-4 : select view in current 2x2 grid
      ← → : yaw -/+
      ↑ ↓ : pitch +/-
      [ ] : fov -/+
      S   : save configs
    """
    if not isinstance(preprocessor, FisheyePreprocessor):
        return tune_state

    if not out_views:
        return tune_state

    shown_views = out_views[:4]
    view_names = [v["name"] for v in shown_views if isinstance(v, dict) and "name" in v]
    if not view_names:
        return tune_state

    tune_state.setdefault("idx", 0)
    tune_state["idx"] = max(0, min(int(tune_state["idx"]), len(view_names) - 1))

    # select tile
    if key in (ord("1"), ord("2"), ord("3"), ord("4")):
        tune_state["idx"] = int(chr(key)) - 1
        tune_state["idx"] = max(0, min(tune_state["idx"], len(view_names) - 1))
        print(f"[TUNE] Selected view: {view_names[tune_state['idx']]}")

    vn = view_names[tune_state["idx"]]

    # find config safely
    cfg = next((c for c in preprocessor.view_configs if c.get("name") == vn), None)
    if not cfg:
        return tune_state

    yaw = float(cfg.get("yaw", 0.0))
    pitch = float(cfg.get("pitch", 0.0))
    fov = float(cfg.get("fov", 90.0))

    changed = False

    # arrow keys (OpenCV keycodes)
    LEFT_KEYS = {81, 2424832}
    UP_KEYS = {82, 2490368}
    RIGHT_KEYS = {83, 2555904}
    DOWN_KEYS = {84, 2621440}

    if key in LEFT_KEYS:
        yaw -= step_yaw
        changed = True
    elif key in RIGHT_KEYS:
        yaw += step_yaw
        changed = True
    elif key in UP_KEYS:
        pitch += step_pitch
        changed = True
    elif key in DOWN_KEYS:
        pitch -= step_pitch
        changed = True
    elif key == ord("["):
        fov -= step_fov
        changed = True
    elif key == ord("]"):
        fov += step_fov
        changed = True

    if changed:
        preprocessor.update_view_params(
            vn,
            yaw=yaw,
            pitch=pitch,
            fov=fov
        )
        print(f"[TUNE] {vn}: yaw={yaw:.1f}, pitch={pitch:.1f}, fov={fov:.1f}")

    # save
    if key in (ord("s"), ord("S")):
        save_fisheye_view_configs(preprocessor)

    return tune_state


def save_fisheye_view_configs(preprocessor, path=None):
    """
    Save fisheye configs inside lostfound_backend by default,
    unless backend overrides the path.
    """
    env_path = (os.getenv("LF_FISHEYE_CONFIG") or "").strip()

    if path is None:
        if env_path:
            path = env_path
        else:
            path = str(LOSTFOUND_BACKEND_DIR / "fisheye_view_configs_live.json")

    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(preprocessor.view_configs, f, indent=2)

    print(f"[TUNE] Saved fisheye view configs -> {p}")


def resolve_roi_config_path() -> Path:
    """
    Resolve ROI config path safely.
    Backend can override with LF_ROI_CONFIG.
    Otherwise default to lostfound_backend/config.json.
    """
    p = (os.getenv("LF_ROI_CONFIG") or "").strip()
    if p:
        return Path(p).resolve()

    return (LOSTFOUND_BACKEND_DIR / "config.json").resolve()


# ---------------------------------------------------------
#  Main Processing
# ---------------------------------------------------------
if __name__ == "__main__":
    # 0) silence spam logs first
    configure_logging(debug=False)

    _banner("LOST & FOUND PIPELINE START")

    # ======================================================
    # PHASE 0 — System + ROI
    # ======================================================
    _step("PHASE 0", "System init + ROI setup")

    _step("SYSTEM", "Memory cleanup...")
    memory_manager.comprehensive_cleanup()

    device = DeviceManager.get_device()
    _kv("SYSTEM", device=device)

    _step("PHASE 0", "ROI setup (redraw/reuse)")

    ROI_CONFIG_PATH = resolve_roi_config_path()

    # show which ROI file is truly used
    _kv("PHASE 0", config=str(ROI_CONFIG_PATH))

    video_type = roi_mode_choice(VIDEO_PATH, config_path=str(ROI_CONFIG_PATH))
    _kv("PHASE 0", roi_mode=str(video_type).upper(), config=str(ROI_CONFIG_PATH))

    # ======================================================
    # PHASE 1 — Run folders + logging + lost manager (Phrase 6/7)
    # ======================================================
    _step("PHASE 1", "Run outputs + Phase 6/7 init")

    run_bundle = setup_run(os.getenv("LF_RUN_PREFIX", "lostandfound"))
    paths = run_bundle["paths"]
    event_logger = run_bundle["event_logger"]
    lost_manager = run_bundle["lost_manager"]

    LOST_JSON_PATH = paths["lost_json"]
    LOST_CSV_PATH = paths["lost_csv"]

    out_dir = paths.get("base", paths.get("out_base", ""))

    _kv("PHASE 6/7", run_id=paths["run_id"], out_dir=out_dir)
    _kv(
        "PHASE 6/7",
        lost_json=os.path.basename(LOST_JSON_PATH),
        lost_csv=os.path.basename(LOST_CSV_PATH),
        event_log=os.path.basename(paths["event_log"])
    )
    _kv("PHASE 6/7", snapshots_dir=paths["snapshots"])

    # ======================================================
    # PHASE 2 — Preprocessor (normal/fisheye) + FPS + source config
    # ======================================================
    _step("PHASE 2", "Preprocessor + Source/FPS policy")

    preprocessor = create_preprocessor(VIDEO_PATH)
    if preprocessor is None:
        _step("ERROR", "Failed to create preprocessor")
        raise SystemExit(1)

    is_fisheye = isinstance(preprocessor, FisheyePreprocessor)
    expected_views = 4 if is_fisheye else 1

    # video fps
    try:
        video_fps = float(preprocessor.cap.get(cv2.CAP_PROP_FPS))
    except Exception:
        video_fps = 0.0
    if not video_fps or video_fps <= 0:
        video_fps = 5.0

    DESIRED_FPS_FISHEYE = 5.0
    DESIRED_FPS_NORMAL = 5.0
    target_fps = min(DESIRED_FPS_FISHEYE, video_fps) if expected_views == 4 else min(DESIRED_FPS_NORMAL, video_fps)

    # source config
    src_cfg = detect_source_config(VIDEO_PATH)

    _kv(
        "PHASE 2",
        video=VIDEO_PATH,
        mode=("FISHEYE" if is_fisheye else "NORMAL"),
        source=("REALTIME" if src_cfg.is_realtime else "FILE"),
        video_fps=round(video_fps, 2),
        target_fps=round(target_fps, 2),
        views_expected=expected_views
    )

    # ======================================================
    # ✅ Backend-controlled auto-toggle for fisheye A/B
    # - LF_FISHEYE_MODE: "A" or "B" (initial group)
    # - LF_AUTO_TOGGLE_SEC: "30" (auto switch interval, 0 disables)
    # ======================================================
    AUTO_TOGGLE_SEC = 0.0
    START_GROUP = os.getenv("LF_FISHEYE_MODE", "").strip().upper()  # "A" or "B"
    try:
        AUTO_TOGGLE_SEC = float(os.getenv("LF_AUTO_TOGGLE_SEC", "0") or 0)
    except Exception:
        AUTO_TOGGLE_SEC = 0.0
    if START_GROUP not in ("A", "B"):
        START_GROUP = "A"

    # ======================================================
    # PHASE 3 — Detector + Threads build
    # ======================================================
    _step("PHASE 3", "Detector + threads setup")

    detector = YoloDetector(
        items_weights=WEIGHTS_ITEMS,
        coco_weights=WEIGHTS_PERSON,
        device=device,
        item_capture_conf=0.08,
        coco_capture_conf=0.1,
        person_conf=0.30,
        track_win=10,
        confirm_k=3,
        hold_frames=15,
    )
    _kv(
        "PHASE 3",
        detector="YOLO",
        item_conf=0.08,
        coco_conf=0.10,
        person_conf=0.30
    )

    stop_event = threading.Event()

    # queues
    frame_queue = queue.Queue(maxsize=60)
    bundle_job_queue = queue.Queue(maxsize=32)
    det_out_queue = queue.Queue(maxsize=4)
    out_queue = queue.Queue(maxsize=2)

    # threads
    reader_thread = FrameReaderThread(
        preprocessor,
        frame_queue,
        stop_event,
        frame_skip=0,
        target_fps=target_fps,
        is_realtime=src_cfg.is_realtime,
    )

    # ======================================================
    # Force initial group BEFORE starting ViewProcessor
    # ======================================================
    if is_fisheye:
        try:
            _, gi_now = active_views.current()
            want_gi = 0 if START_GROUP == "A" else 1
            if gi_now != want_gi:
                active_views.toggle()
                _step("PHASE 2", f"Start group forced -> {START_GROUP}")
        except Exception:
            pass

    processor_thread = ViewProcessorThread(
        preprocessor,
        frame_queue,
        bundle_job_queue,
        stop_event,
        active_views
    )

    num_workers = 1
    workers = [
        DetectionWorkerThread(i, detector, bundle_job_queue, det_out_queue, stop_event)
        for i in range(num_workers)
    ]

    supervisor = SupervisorThread(
        reader_thread=reader_thread,
        bundle_job_queue=bundle_job_queue,
        stop_event=stop_event,
        max_skip=1
    )

    tracking_thread = TrackingThread(det_out_queue, out_queue, stop_event, detector, confirm_k=3)
    tracking_thread.lost_manager = lost_manager

    # progress wiring
    _step("PHASE 3", "Progress.json wiring")
    try:
        tracking_thread.run_id = paths["run_id"]
        tracking_thread.progress_path = paths["progress"]
        tracking_thread.total_frames = int(preprocessor.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (not src_cfg.is_realtime) else None
        tracking_thread.views_expected = expected_views
        tracking_thread.fps_hint = target_fps

        write_progress(
            tracking_thread.progress_path,
            run_id=paths["run_id"],
            stage="starting",
            current=0,
            total=tracking_thread.total_frames,
            fps=tracking_thread.fps_hint,
            lost_count=0,
            group_index=0,
            views_expected=tracking_thread.views_expected,
            message="Starting pipeline...",
        )

        _kv(
            "PHASE 3",
            progress=os.path.basename(paths["progress"]),
            total_frames=tracking_thread.total_frames,
            run_id=paths["run_id"]
        )
    except Exception as e:
        _step("WARN", f"Progress wiring skipped: {e}")

    # ======================================================
    # PHASE 4 — Start threads + UI loop
    # ======================================================
    _step("PHASE 4", "Starting runtime pipeline")

    reader_thread.start()
    processor_thread.start()
    for w in workers:
        w.start()
    tracking_thread.start()
    supervisor.start()

    _kv(
        "PHASE 4",
        reader="ON",
        view_processor="ON",
        workers=num_workers,
        tracking="ON",
        supervisor="ON"
    )

    # UI window
    win_name = "Lost&Found - Multi-View Preview"

    frame_idx, ts, gi = 0, 0.0, 0

    try:
        # screen size
        try:
            user32 = ctypes.windll.user32
            screen_w = int(user32.GetSystemMetrics(0))
            screen_h = int(user32.GetSystemMetrics(1))
        except Exception:
            screen_w, screen_h = 1280, 720

        WINDOW_SCALE = 0.80
        target_w = int(screen_w * WINDOW_SCALE)
        target_h = int(screen_h * WINDOW_SCALE)

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, target_w, target_h)

        display_fps = 25
        display_dt = 1.0 / max(1e-6, display_fps)
        next_show_t = time.time()

        last_out_views = []
        tune_state = {"idx": 0}

        _kv("UI", window=win_name, display_fps=display_fps, scale=WINDOW_SCALE)
        _step("UI", "Controls: Q=Quit | T=Toggle group | 1-4 + arrows + [] + S = Tune/Save")

        def _toggle_views():
            active_views.toggle()
            drain_queue(frame_queue)
            drain_queue(bundle_job_queue)
            drain_queue(det_out_queue)
            drain_queue(out_queue)

            # reset tracker state so IDs don't mix across groups
            try:
                if hasattr(tracking_thread, "reset_state"):
                    tracking_thread.reset_state()
                elif hasattr(tracking_thread, "tracker") and hasattr(tracking_thread.tracker, "reset"):
                    tracking_thread.tracker.reset()
            except Exception:
                pass

            _, new_gi = active_views.current()
            _step("UI", f"Switched view group -> {new_gi}")

        def _handle_key(key, tune_state):
            tune_state = handle_realtime_tuning(key, preprocessor, last_out_views, tune_state)

            if key in (ord('q'), ord('Q')):
                _step("UI", "Quit requested (Q)")
                stop_event.set()
                return False, tune_state

            if key in (ord('t'), ord('T')):
                _toggle_views()

            return True, tune_state

        # auto toggle bookkeeping
        if is_fisheye and AUTO_TOGGLE_SEC > 0:
            tune_state["last_auto_toggle_t"] = time.time()
            _step("UI", f"Auto-toggle enabled every {AUTO_TOGGLE_SEC:.0f}s (start={START_GROUP})")

        # drain grace so tracking/events finish
        drain_start = None

        while not stop_event.is_set():
            latest_bundle = None
            try:
                while True:
                    latest_bundle = out_queue.get_nowait()
            except queue.Empty:
                pass

            key = cv2.waitKeyEx(1)
            ok, tune_state = _handle_key(key, tune_state)
            if not ok:
                break

            # auto toggle every N seconds
            if is_fisheye and AUTO_TOGGLE_SEC > 0:
                last_t = tune_state.get("last_auto_toggle_t", time.time())
                if (time.time() - last_t) >= AUTO_TOGGLE_SEC:
                    tune_state["last_auto_toggle_t"] = time.time()
                    _toggle_views()

            if latest_bundle is None:
                if (not reader_thread.is_alive()) and frame_queue.empty():
                    if bundle_job_queue.empty() and det_out_queue.empty() and out_queue.empty():
                        if drain_start is None:
                            drain_start = time.time()
                        elif time.time() - drain_start > 2.0:
                            _step("UI", "End of stream (pipeline drained)")
                            break
                    else:
                        drain_start = None
                continue

            now = time.time()
            if now < next_show_t:
                continue
            next_show_t = now + display_dt

            frame_idx, ts, gi, out_views = latest_bundle
            last_out_views = out_views or []

            if isinstance(preprocessor, FisheyePreprocessor):
                allowed, _ = active_views.current()
                expected_views_now = len(allowed)
            else:
                expected_views_now = 1

            if out_views and (len(out_views) == expected_views_now):
                out_views = sorted(out_views, key=lambda x: x.get("view_id", 0))
                grid = build_views_grid(out_views)

                if grid is not None:
                    gh, gw = grid.shape[:2]
                    scale = min(target_w / max(1, gw), target_h / max(1, gh))
                    new_w = max(1, int(gw * scale))
                    new_h = max(1, int(gh * scale))
                    grid_resized = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    cv2.imshow(win_name, grid_resized)

        # Done
        _step("DONE", "Writing final progress...")
        try:
            if getattr(tracking_thread, "progress_path", None):
                tot = getattr(tracking_thread, "total_frames", None)
                cur = int(frame_idx) if isinstance(frame_idx, int) else 0
                if isinstance(tot, int) and tot > 0:
                    cur = max(cur, tot)
                write_progress(
                    tracking_thread.progress_path,
                    run_id=getattr(tracking_thread, "run_id", "run"),
                    stage="finished",
                    current=cur,
                    total=tot,
                    fps=getattr(tracking_thread, "fps_hint", None),
                    lost_count=len(lost_manager.lost_items),
                    group_index=int(gi) if isinstance(gi, int) else 0,
                    views_expected=getattr(tracking_thread, "views_expected", None),
                    message="Finished.",
                )
        except Exception:
            pass

        _kv("DONE", lost_count=len(lost_manager.lost_items))

    except KeyboardInterrupt:
        _step("UI", "KeyboardInterrupt -> stopping")
        stop_event.set()

    except Exception as e:
        _step("ERROR", f"Main loop crashed: {e}")
        stop_event.set()

        try:
            # mark progress as error
            if getattr(tracking_thread, "progress_path", None):
                write_progress(
                    tracking_thread.progress_path,
                    run_id=getattr(tracking_thread, "run_id", "run"),
                    stage="error",
                    current=int(frame_idx) if isinstance(frame_idx, int) else 0,
                    total=getattr(tracking_thread, "total_frames", None),
                    fps=getattr(tracking_thread, "fps_hint", None),
                    lost_count=len(lost_manager.lost_items) if lost_manager else 0,
                    group_index=int(gi) if isinstance(gi, int) else 0,
                    views_expected=getattr(tracking_thread, "views_expected", None),
                    message=f"Error: {repr(e)}",
                )
        except Exception:
            pass

    finally:
        # =========================
        # PHASE 5 — Shutdown
        # =========================
        _step("PHASE 5", "Stopping threads + releasing resources")

        stop_event.set()

        # help queues unblock
        try:
            put_drop_oldest(frame_queue, (SENTINEL, None, None))
        except Exception:
            pass

        try:
            put_drop_oldest(bundle_job_queue, (SENTINEL, None, None, None))
        except Exception:
            pass

        try:
            put_drop_oldest(det_out_queue, (SENTINEL, None, None, None))
        except Exception:
            pass

        try:
            put_drop_oldest(out_queue, (SENTINEL, None, None, None))
        except Exception:
            pass

        # join threads
        def _join(t, name, timeout=2.0):
            try:
                if t and t.is_alive():
                    t.join(timeout=timeout)
            except Exception:
                pass

        _join(reader_thread, "reader", 2.0)
        _join(processor_thread, "view_processor", 2.0)

        for w in workers:
            _join(w, "worker", 2.0)

        _join(tracking_thread, "tracking", 2.0)
        _join(supervisor, "supervisor", 2.0)

        # release preprocessor / capture
        try:
            if preprocessor is not None:
                if hasattr(preprocessor, "release"):
                    preprocessor.release()

                cap = getattr(preprocessor, "cap", None)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        except Exception:
            pass

        # close UI windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # final exports
        try:
            if lost_manager is not None:
                lost_manager.export_lost_items_json(LOST_JSON_PATH)
                lost_manager.export_lost_items_csv(LOST_CSV_PATH)
        except Exception:
            pass

        # final progress
        try:
            if getattr(tracking_thread, "progress_path", None):
                tot = getattr(tracking_thread, "total_frames", None)
                cur = int(frame_idx) if isinstance(frame_idx, int) else 0
                if isinstance(tot, int) and tot > 0:
                    cur = max(cur, tot)

                write_progress(
                    tracking_thread.progress_path,
                    run_id=getattr(tracking_thread, "run_id", "run"),
                    stage="finished",
                    current=cur,
                    total=tot,
                    fps=getattr(tracking_thread, "fps_hint", None),
                    lost_count=len(lost_manager.lost_items) if lost_manager else 0,
                    group_index=int(gi) if isinstance(gi, int) else 0,
                    views_expected=getattr(tracking_thread, "views_expected", None),
                    message="Finished.",
                )
        except Exception:
            pass

        # optional final cleanup
        try:
            memory_manager.comprehensive_cleanup()
        except Exception:
            pass

        _kv("DONE", lost_count=(len(lost_manager.lost_items) if lost_manager else 0))
