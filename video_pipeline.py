# video_pipeline.py
# ---------------------------------------------------------
# VideoPipeline that REUSES lostandfound.py threads/managers
# and exposes backend-safe LIVE data:
#   - pull_latest_for_api()  -> views + detections (JSON-safe, no images)
#   - pull_group_grid_jpg()  -> MJPEG-ready 2x2 grid (ROI always, det boxes if enabled)
#   - pull_single_view_jpg() -> MJPEG-ready single view (ROI always, det boxes if enabled)
#
# Conservative version:
# ✅ Mild skip-frame for BOTH RTSP and FILE
# ✅ No aggressive latest-only queue draining
# ✅ No adaptive skip loop
# ✅ Avoids unreasonable “faster than real-world” playback
# ✅ Keeps ROI filtering + fisheye Group A/B preserved
# ---------------------------------------------------------

import os
import time
import json
import queue
import threading
import ctypes
from dataclasses import dataclass, asdict, is_dataclass
from typing import Dict, List, Any, Optional
import logging

import cv2
import numpy as np
import lostandfound as lf

FISHEYE_CFG_STORE = (
    r"D:\DrTew\SecureWatch by QingYing JinXuan\SecureWatch"
    r"\lostfound_backend\backend\outputs\lost_and_found\fisheye_view_configs.json"
)


# ---------------------------------------------------------
# Helpers
# --------------------------------------------------------
def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1280, 720


def _running_under_backend() -> bool:
    return str(os.getenv("LF_BACKEND", "")).strip() == "1"


# ---------------------------------------------------------
# RTSP-safe first-frame loader
# ---------------------------------------------------------
def _rtsp_safe_load_first_frame(video_src: str):
    """
    Load the first readable frame from a video path OR RTSP URL.
    Uses cv2.VideoCapture directly — never pathlib.Path.
    Returns np.ndarray or None.
    """
    try:
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[ERROR] _rtsp_safe_load_first_frame: cannot open: {video_src!r}")
            return None

        is_rtsp = video_src.lower().startswith("rtsp")
        max_attempts = 10 if is_rtsp else 3
        frame = None
        for _ in range(max_attempts):
            ret, frm = cap.read()
            if ret and frm is not None:
                frame = frm
                break
            time.sleep(0.05)

        cap.release()

        if frame is None:
            print(f"[ERROR] _rtsp_safe_load_first_frame: no frame from: {video_src!r}")
        else:
            print(f"[OK] _rtsp_safe_load_first_frame: shape={frame.shape}")
        return frame

    except Exception as e:
        print(f"[ERROR] _rtsp_safe_load_first_frame: {e!r}")
        return None


# ---------------------------------------------------------
# Inline ROI editors
# ---------------------------------------------------------
def _launch_roi_editor_rtsp_safe(video_src: str, config_out_path: str = "config.json"):
    """
    Normal-camera interactive ROI editor. RTSP-safe.
    Writes {"bounding_polygons": [...]} to config_out_path.
    """
    frame = _rtsp_safe_load_first_frame(video_src)
    if frame is None:
        print(f"[WARN] ROI editor: no frame from {video_src!r}. Saving empty ROI.")
        _write_roi_config_safe(config_out_path, [], {})
        return

    KEY_QUIT = ord("q")
    KEY_RESET = ord("r")
    KEY_SAVE_ROI = ord("n")
    WIN = "Left-click=add point | N=save ROI | R=reset | Q=finish & save"

    current_points: List = []
    all_polygons: List = []
    frame_copy = frame.copy()

    def _draw_poly(img, pts, color=(0, 0, 255), close=False):
        if len(pts) < 2:
            return
        for i in range(len(pts)):
            cv2.circle(img, pts[i], 5, color, -1)
            if i + 1 < len(pts):
                cv2.line(img, pts[i], pts[i + 1], color, 2)
        if close and len(pts) >= 3:
            cv2.line(img, pts[-1], pts[0], color, 2)

    def _redraw_saved(base):
        img = base.copy()
        for poly in all_polygons:
            pts = [(int(p["x"]), int(p["y"])) for p in poly]
            _draw_poly(img, pts, color=(0, 255, 0), close=True)
        return img

    def mouse_cb(event, x, y, flags, param):
        nonlocal frame_copy
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
            if len(current_points) > 1:
                cv2.line(frame_copy, current_points[-2], current_points[-1], (0, 0, 255), 2)

    try:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)
        cv2.setMouseCallback(WIN, mouse_cb)
        print("[ROI] L-click=add | N=save ROI | R=reset | Q=finish & save")

        while True:
            cv2.imshow(WIN, frame_copy)
            key = cv2.waitKey(20) & 0xFF

            if key == KEY_QUIT:
                break
            elif key == KEY_RESET:
                current_points.clear()
                frame_copy = _redraw_saved(frame)
            elif key == KEY_SAVE_ROI:
                if len(current_points) < 3:
                    print("[WARN] Need at least 3 points.")
                    continue
                poly = [{"x": float(x), "y": float(y)} for (x, y) in current_points]
                all_polygons.append(poly)
                print(f"[ROI] Saved ROI #{len(all_polygons)} with {len(current_points)} pts")
                current_points.clear()
                frame_copy = _redraw_saved(frame)
    finally:
        try:
            cv2.destroyWindow(WIN)
        except Exception:
            pass

    _write_roi_config_safe(config_out_path, all_polygons, {})
    print(f"[ROI] {len(all_polygons)} polygon(s) saved to {config_out_path!r}")


def _launch_fisheye_roi_editor_rtsp_safe(video_src: str, config_out_path: str = "config.json"):
    """
    Fisheye ROI editor with RTSP warm-up.
    Dewarps views first, lets user annotate each view polygon,
    saves {"fisheye_polygons": {...}} to config_out_path.
    Falls back to plain editor if dewarp fails.
    """
    try:
        view_configs = getattr(lf, "FISHEYE_VIEW_CONFIGS", None) or getattr(lf, "DEFAULT_VIEW_CONFIGS", [])
        prep = lf.FisheyePreprocessor(view_configs=view_configs, config_path=config_out_path)
        opened = prep.open(video_src)
    except Exception as e:
        print(f"[WARN] Fisheye preprocessor failed: {e!r}. Falling back to plain editor.")
        _launch_roi_editor_rtsp_safe(video_src, config_out_path)
        return

    if not opened:
        print("[WARN] Fisheye preprocessor could not open. Falling back to plain editor.")
        _launch_roi_editor_rtsp_safe(video_src, config_out_path)
        try:
            prep.release()
        except Exception:
            pass
        return

    views = []
    try:
        is_rtsp = video_src.lower().startswith("rtsp")
        max_attempts = 15 if is_rtsp else 5
        for attempt in range(max_attempts):
            ok, raw = prep.read_frame()
            if ok and raw is not None:
                views = prep.get_views(raw)
                if views:
                    print(f"[OK] Fisheye views ready after {attempt + 1} frame(s)")
                    break
            time.sleep(0.05)
    except Exception as e:
        print(f"[WARN] Fisheye get_views failed: {e!r}")

    try:
        prep.release()
    except Exception:
        pass

    if not views:
        print("[WARN] No fisheye views. Falling back to plain editor.")
        _launch_roi_editor_rtsp_safe(video_src, config_out_path)
        return

    KEY_NEXT = ord("n")
    KEY_PREV = ord("p")
    KEY_QUIT = ord("q")
    KEY_CLEAR = ord("c")
    KEY_UNDO = ord("u")

    fisheye_polygons: Dict[str, List] = {}
    try:
        if os.path.exists(config_out_path):
            with open(config_out_path, "r", encoding="utf-8") as f:
                fisheye_polygons = (json.load(f) or {}).get("fisheye_polygons", {}) or {}
    except Exception:
        pass

    idx = 0
    views_list = list(views)

    while 0 <= idx < len(views_list):
        v = views_list[idx]
        view_name = str(v.get("name") or f"view_{idx}")
        view_img = v.get("image")

        if not isinstance(view_img, np.ndarray):
            idx += 1
            continue

        print(f"\n=== [{idx + 1}/{len(views_list)}] {view_name} ===")
        print("L-click=add | R-click=close polygon | U=undo | C=clear | N=next | P=prev | Q=quit")

        WIN = view_name
        drawing_poly: List = []
        saved_polys: List = []

        for poly_cfg in fisheye_polygons.get(view_name, []):
            saved_polys.append([(int(p["x"]), int(p["y"])) for p in poly_cfg])

        def mouse_cb_fish(event, x, y, flags, param):
            nonlocal drawing_poly, saved_polys
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing_poly.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(drawing_poly) >= 3:
                    saved_polys.append(drawing_poly.copy())
                    print(f"[CLOSE] {len(drawing_poly)} pts")
                else:
                    print("[WARN] Need >= 3 points")
                drawing_poly.clear()

        try:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, 1280, 720)
            cv2.setMouseCallback(WIN, mouse_cb_fish)
        except Exception:
            idx += 1
            continue

        action = "next"
        while True:
            img = view_img.copy()
            for poly in saved_polys:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            if len(drawing_poly) > 1:
                pts = np.array(drawing_poly, dtype=np.int32)
                cv2.polylines(img, [pts], False, (0, 0, 255), 2)

            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (520, 85), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            cv2.putText(img, "LClick:add  RClick:close  U:undo  C:clear",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "N:next  P:prev  Q:quit",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WIN, img)
            key = cv2.waitKey(20) & 0xFF

            if key == KEY_CLEAR:
                saved_polys.clear()
                drawing_poly.clear()
            elif key == KEY_UNDO:
                if drawing_poly:
                    drawing_poly.pop()
                elif saved_polys:
                    saved_polys.pop()
            elif key == KEY_NEXT:
                action = "next"
                break
            elif key == KEY_PREV:
                action = "prev"
                break
            elif key == KEY_QUIT:
                action = "quit"
                break

        try:
            cv2.destroyWindow(WIN)
        except Exception:
            pass

        fisheye_polygons[view_name] = [
            [{"x": int(x), "y": int(y)} for (x, y) in poly]
            for poly in saved_polys
        ]

        if action == "quit":
            break
        elif action == "next":
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)

    _write_roi_config_safe(config_out_path, [], fisheye_polygons)
    print(f"[ROI] Fisheye polygons saved to {config_out_path!r}")


def _write_roi_config_safe(config_path: str, bounding_polygons: list, fisheye_polygons: dict):
    out = {
        "bounding_polygons": bounding_polygons or [],
        "fisheye_polygons": fisheye_polygons or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)



# ---------------------------------------------------------
# Output roots
# ---------------------------------------------------------
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# IMPORTANT:
# video_pipeline.py is already inside lostfound_backend,
# so use MODULE_DIR directly, not its parent.
OUTPUT_BASE = os.path.join(MODULE_DIR, "outputs")
LAF_BASE = os.path.join(OUTPUT_BASE, "lost_and_found")
SNAPSHOT_BASE = os.path.join(MODULE_DIR, "snapshots")

# ---------------------------------------------------------
# ViewProcessor subclass
# ---------------------------------------------------------
class ViewProcessorThreadCam(lf.ViewProcessorThread):
    def __init__(self, camera_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_id = str(camera_id)

    def _normalize_view_dict(self, v: dict) -> dict:
        out = super()._normalize_view_dict(v)
        out["camera_id"] = self.camera_id
        base_name = out.get("name", f"view_{out.get('view_id', 0)}")
        out["display_name"] = f"{self.camera_id}:{base_name}"
        return out


class TrackingThreadROIMatchLF(lf.TrackingThread):
    """
    Make video_pipeline ROI filtering match lostandfound.py style.
    """

    def _normalize_zone_ids_for_pipeline(self, zones: list) -> list:
        out = []
        for z in lf.normalize_zones(zones or []):
            zz = dict(z)
            rid = zz.get("roi_id", zz.get("zone_id", zz.get("id", None)))
            if rid is not None:
                zz["roi_id"] = rid
                zz["zone_id"] = rid
                zz["id"] = rid
            out.append(zz)
        return out

    def _filter_for_tracking(self, img, zones, raw):
        raw = raw or []
        zones = self._normalize_zone_ids_for_pipeline(zones)

        persons = []
        items = []

        for d in raw:
            if not isinstance(d, dict):
                continue
            cname = str(d.get("class_name", d.get("label", d.get("name", "")))).strip().lower()
            if cname == "person":
                persons.append(d)
            else:
                items.append(d)

        person_conf = max(0.30, float(getattr(self.detector, "person_conf", 0.30)))
        item_conf = max(0.08, float(getattr(self.detector, "item_capture_conf", 0.08)))
        min_area_person = int(getattr(self.detector, "min_area_person", 40 * 40))
        min_area_item = int(getattr(self.detector, "min_area_item", 20 * 20))

        # persons: confidence/size only
        persons = lf.filter_detections_by_conf_and_size(
            persons,
            conf_min=person_conf,
            min_area=min_area_person,
        )

        # items: confidence/size + ROI filter
        items = lf.filter_detections_by_conf_and_size(
            items,
            conf_min=item_conf,
            min_area=min_area_item,
        )

        if zones:
            items = lf.filter_detections_to_zones_by_overlap(
                items,
                zones,
                img_shape=img.shape,
                min_ratio=0.30,
            )
            items = lf.dedup_by_overlap_ratio(items, overlap_thr=0.50)

        return persons + items

    def reload_roi(self):
        try:
            new_cfg = self._read_roi_config() or {}

            with self._roi_lock:
                self._roi_cache = new_cfg
                try:
                    self._roi_mtime = os.path.getmtime(self.cfg.roi_config_path)
                except Exception:
                    self._roi_mtime = time.time()
                self._roi_dirty = False

            self._roi_available_for_detection = self._has_roi_for_detection()

            # -------------------------------------------------
            # IMPORTANT: also push ROI into preprocessor runtime
            # -------------------------------------------------
            try:
                if self.preprocessor is not None:
                    if hasattr(self.preprocessor, "reload_roi_config"):
                        try:
                            self.preprocessor.reload_roi_config(self.cfg.roi_config_path)
                        except TypeError:
                            self.preprocessor.reload_roi_config()
                    else:
                        # fallback: inject latest config directly
                        try:
                            setattr(self.preprocessor, "config_path", self.cfg.roi_config_path)
                        except Exception:
                            pass
                        try:
                            setattr(self.preprocessor, "roi_config_path", self.cfg.roi_config_path)
                        except Exception:
                            pass
                        try:
                            setattr(self.preprocessor, "bounding_polygons", new_cfg.get("bounding_polygons", []) or [])
                        except Exception:
                            pass
                        try:
                            setattr(self.preprocessor, "fisheye_polygons", new_cfg.get("fisheye_polygons", {}) or {})
                        except Exception:
                            pass
            except Exception as e:
                lf._step("WARN", f"{self.cfg.camera_id} preprocessor ROI refresh failed: {e}")

            # -------------------------------------------------
            # IMPORTANT: clear stale queued/cached frames
            # -------------------------------------------------
            try:
                self._drain_q(self.bundle_job_queue, max_items=999999)
                self._drain_q(self.det_out_queue, max_items=999999)
                self._drain_q(self.out_queue, max_items=999999)

                with self._last_bundle_lock:
                    self._last_bundle_by_gi.clear()
                    self._last_meta_by_gi.clear()
            except Exception:
                pass

            lf._step("UI", f"{self.cfg.camera_id} ROI reloaded")
        except Exception as e:
            lf._step("ERROR", f"{self.cfg.camera_id} reload_roi failed: {e}")

    def _refresh_runtime_configs_if_needed(self):
        try:
            self._refresh_roi_cache_if_needed()
        except Exception:
            pass

        try:
            self._refresh_fisheye_config_if_needed()
        except Exception:
            pass

    def _refresh_roi_cache_if_needed(self):
        try:
            need = bool(getattr(self, "_roi_dirty", False))

            try:
                mtime = os.path.getmtime(self.cfg.roi_config_path)
            except Exception:
                mtime = 0.0

            if (not need) and (mtime > float(getattr(self, "_roi_mtime", 0.0))):
                need = True

            if need:
                self.reload_roi()
        except Exception:
            pass

    def _get_roi_cfg_cached(self) -> dict:
        self._refresh_roi_cache_if_needed()
        with self._roi_lock:
            return dict(self._roi_cache or {})
        
    def _largest_poly_only(self, polys):
        try:
            zones = lf.polygons_to_zones(polys or [], start_id=1, id_key="roi_id")
            largest = lf.get_largest_roi(zones)
            if not largest:
                return []
            pts = largest.get("points", []) or []
            return [[{"x": float(x), "y": float(y)} for (x, y) in pts]]
        except Exception:
            return []

    def _draw_polygon_list(self, img, polys, color=(0, 255, 0), thickness=2):
        if img is None:
            return img

        polys = self._largest_poly_only(polys)

        for poly in (polys or []):
            if isinstance(poly, list) and len(poly) >= 3:
                try:
                    pts = np.array(
                        [[int(p["x"]), int(p["y"])] for p in poly if isinstance(p, dict)],
                        dtype=np.int32,
                    )
                    if len(pts) >= 3:
                        cv2.polylines(img, [pts], True, color, thickness)
                except Exception:
                    pass
        return img
        
class RealtimeFreshRTSPReaderThread(threading.Thread):
    """
    RTSP reader with stable 2-3 second delay - NO output until buffer is FULLY loaded
    """

    def __init__(
        self,
        preprocessor,
        out_queue,
        stop_event,
        target_fps: float = 2.5,
        warmup_grabs: float = 2,
        flush_grabs_per_cycle: float = 0,
        reconnect_backoff_sec: float = 0.5,
        max_reconnect_backoff_sec: float = 5.0,
        queue_put_fn=None,
        buffer_delay_sec: float = 2.5,
    ):
        super().__init__(daemon=True)
        self.preprocessor = preprocessor
        self.out_queue = out_queue
        self.stop_event = stop_event

        self.target_fps = max(1.0, float(target_fps or 6.0))
        self.frame_period = 1.0 / self.target_fps

        self.warmup_grabs = max(0, int(warmup_grabs))
        self.flush_grabs_per_cycle = max(0, int(flush_grabs_per_cycle))

        self.reconnect_backoff_sec = float(reconnect_backoff_sec)
        self.max_reconnect_backoff_sec = float(max_reconnect_backoff_sec)

        self.queue_put_fn = queue_put_fn or lf.put_drop_oldest
        self._reconnect_wait = self.reconnect_backoff_sec
        
        # Buffer delay configuration
        self.buffer_delay_sec = float(buffer_delay_sec)
        
        # Required by lostandfound.py supervisor
        self._frame_skip = 0
        
        # Initialize timing attributes
        self.frame_timestamp_count = 0
        self.first_output_time = None
        
        # Frame buffer - will only output when FULLY loaded
        self.frame_buffer = []
        # Need enough frames to cover the full delay
        self.required_buffer_frames = int(self.target_fps * self.buffer_delay_sec) + 5
        self.max_buffer_size = self.required_buffer_frames * 2
        self.buffer_ready = False  # Only becomes True when buffer is FULLY loaded
        
        # Reconnection tracking
        self.reconnecting = False
        
        # Lock for thread-safe buffer access
        self.buffer_lock = threading.Lock()
        
        # Rate limiting
        self.last_output_time = 0
        self.min_output_interval = self.frame_period * 0.98
        
        # Statistics
        self.frames_read = 0
        self.frames_output = 0
        
        print(f"[RTSP] Config: {self.buffer_delay_sec}s delay = {self.required_buffer_frames} frames at {self.target_fps}fps")

    @property
    def frame_skip(self):
        return self._frame_skip

    @frame_skip.setter
    def frame_skip(self, value):
        self._frame_skip = max(0, int(value))

    def _get_cap(self):
        return getattr(self.preprocessor, "cap", None)

    def _apply_low_buffer(self, cap):
        if cap is None:
            return
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _warmup_cap(self, cap):
        if cap is None:
            return
        for _ in range(self.warmup_grabs):
            try:
                if not cap.grab():
                    break
            except Exception:
                break

    def _reset_buffer(self, reason=""):
        """Completely reset buffer - NO output until fully reloaded"""
        with self.buffer_lock:
            self.frame_buffer.clear()
            self.buffer_ready = False
            self.first_output_time = None
            self.frame_timestamp_count = 0
            self.last_output_time = 0
            
            # Clear output queue completely
            try:
                while self.out_queue.qsize() > 0:
                    self.out_queue.get_nowait()
            except Exception:
                pass
            
            print(f"[RTSP] BUFFER RESET for '{reason}' - need {self.required_buffer_frames} frames before output")

    def _safe_reconnect(self) -> bool:
        if self.reconnecting:
            return False
            
        self.reconnecting = True
        print("[RTSP] Attempting reconnection...")
        
        try:
            if hasattr(self.preprocessor, "release"):
                self.preprocessor.release()
        except Exception:
            pass

        time.sleep(min(self._reconnect_wait, self.max_reconnect_backoff_sec))
        self._reconnect_wait = min(self._reconnect_wait * 1.6, self.max_reconnect_backoff_sec)

        ok = False
        try:
            if hasattr(self.preprocessor, "src"):
                ok = bool(self.preprocessor.open(self.preprocessor.src))
            elif hasattr(self.preprocessor, "video_path"):
                ok = bool(self.preprocessor.open(self.preprocessor.video_path))
            else:
                ok = bool(self.preprocessor.open())
        except Exception:
            try:
                ok = bool(self.preprocessor.open())
            except Exception:
                ok = False

        if ok:
            cap = self._get_cap()
            self._apply_low_buffer(cap)
            self._warmup_cap(cap)
            self._reconnect_wait = self.reconnect_backoff_sec
            
            # CRITICAL: Reset buffer on reconnect
            self._reset_buffer("reconnection")
            print("[RTSP] Reconnection successful")
        else:
            print("[RTSP] Reconnection failed")
            
        self.reconnecting = False
        return ok

    def run(self):
        # Initialize with buffer building mode - NO OUTPUT YET
        self._reset_buffer("initial start")
        
        # Apply low-buffer once at start
        try:
            self._apply_low_buffer(self._get_cap())
        except Exception:
            pass

        last_frame_read_time = time.time()
        
        print(f"[RTSP] Reader started - Filling buffer with {self.required_buffer_frames} frames before output...")
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # === READING PHASE - Read frames as fast as reasonable ===
                # Don't read too fast - max 2x target FPS
                read_interval = 1.0 / min(self.target_fps * 2, 30)
                time_since_last_read = current_time - last_frame_read_time
                if time_since_last_read < read_interval:
                    time.sleep(read_interval - time_since_last_read)
                    current_time = time.time()
                
                cap = self._get_cap()
                if cap is None:
                    if not self._safe_reconnect():
                        time.sleep(0.1)
                        continue
                    cap = self._get_cap()
                    if cap is None:
                        time.sleep(0.1)
                        continue

                # Optional flush grabs
                for _ in range(self.flush_grabs_per_cycle):
                    try:
                        if not cap.grab():
                            break
                    except Exception:
                        break

                # Read frame
                ok, frame = cap.read()
                last_frame_read_time = time.time()
                
                if not ok or frame is None:
                    if not self._safe_reconnect():
                        time.sleep(0.1)
                        continue
                    continue

                self.frames_read += 1
                
                # Store frame in buffer with timestamp
                with self.buffer_lock:
                    self.frame_buffer.append((last_frame_read_time, frame.copy()))
                    
                    # Limit buffer size
                    while len(self.frame_buffer) > self.max_buffer_size:
                        self.frame_buffer.pop(0)
                    
                    # CRITICAL: Check if buffer is FULLY loaded (not just partially)
                    if not self.buffer_ready and len(self.frame_buffer) >= self.required_buffer_frames:
                        # Also verify the time difference
                        if len(self.frame_buffer) >= 2:
                            time_diff = self.frame_buffer[-1][0] - self.frame_buffer[0][0]
                            if time_diff >= self.buffer_delay_sec:
                                self.buffer_ready = True
                                self.first_output_time = time.time()
                                print(f"[RTSP] ✓ BUFFER READY! {len(self.frame_buffer)} frames, {time_diff:.2f}s delay")
                                print(f"[RTSP] Starting output at {self.target_fps}fps with {self.buffer_delay_sec}s delay")
                
                # === OUTPUT PHASE - ONLY if buffer is FULLY ready ===
                if self.buffer_ready:
                    # Strict rate limiting - never output faster than target FPS
                    time_since_last_output = time.time() - self.last_output_time
                    if time_since_last_output < self.min_output_interval:
                        # Wait to maintain correct FPS
                        sleep_time = self.min_output_interval - time_since_last_output
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    
                    # Get the next frame to output
                    frame_to_output = None
                    with self.buffer_lock:
                        if self.frame_buffer:
                            # Always output the oldest frame first (FIFO)
                            frame_to_output = self.frame_buffer.pop(0)[1]
                    
                    if frame_to_output is not None:
                        # Clear output queue if it's getting too full
                        try:
                            while self.out_queue.qsize() >= 2:
                                self.out_queue.get_nowait()
                        except Exception:
                            pass
                        
                        # Output the frame
                        output_time = time.time()
                        self.queue_put_fn(self.out_queue, (output_time, None, frame_to_output))
                        
                        self.frames_output += 1
                        self.last_output_time = output_time
                        
                        # Periodic stats
                        if self.frames_output % 30 == 0:
                            with self.buffer_lock:
                                buffer_size = len(self.frame_buffer)
                                print(f"[RTSP] Output: {self.frames_output} frames, buffer: {buffer_size}, target FPS: {self.target_fps}")
                    else:
                        # This shouldn't happen if buffer_ready is True
                        time.sleep(0.001)
                else:
                    # Buffer not ready yet - show progress
                    if self.frames_read % 30 == 0:
                        with self.buffer_lock:
                            buffer_size = len(self.frame_buffer)
                            if buffer_size > 0 and len(self.frame_buffer) >= 2:
                                time_diff = self.frame_buffer[-1][0] - self.frame_buffer[0][0]
                                print(f"[RTSP] Loading buffer: {buffer_size}/{self.required_buffer_frames} frames, {time_diff:.1f}/{self.buffer_delay_sec}s")
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"[ERROR] RTSP reader: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

        # Send sentinel on exit
        try:
            self.queue_put_fn(self.out_queue, (lf.SENTINEL,))
        except Exception:
            pass
        
        print(f"[RTSP] Reader stopped - Read: {self.frames_read}, Output: {self.frames_output}")

    def get_buffer_status(self):
        """Return current buffer status for debugging"""
        with self.buffer_lock:
            return {
                'buffer_ready': self.buffer_ready,
                'buffer_size': len(self.frame_buffer),
                'required_frames': self.required_buffer_frames,
                'target_delay': self.buffer_delay_sec,
                'frames_output': self.frames_output,
                'frames_read': self.frames_read
            }
        
@dataclass
class PipelineConfig:
    camera_id: str
    src: str
    roi_config_path: str
    fisheye_config_path: str = FISHEYE_CFG_STORE
    num_workers: int = 1
    max_skip: float = 2

    desired_fps_fisheye: float = 1.0
    desired_fps_normal: float = 1.0
    display_fps: float = 2.5

    show_ui: bool = True
    enable_detection: bool = True
    force_video_type: Optional[str] = None
    source_kind: str = "FILE"

    detector_group: str = "A"

    base_frame_skip_fisheye_rtsp: int = 0
    base_frame_skip_normal_rtsp: int = 0
    base_frame_skip_fisheye_file: int = 0
    base_frame_skip_normal_file: int = 0

    drop_old_detection_jobs: bool = True
    latest_only_tracking: bool = True

    window_scale: float = 0.80


class VideoPipeline:
    def __init__(self, cfg: PipelineConfig, detector: "lf.YoloDetector"):
        self.cfg = cfg
        if not str(getattr(self.cfg, "fisheye_config_path", "") or "").strip():
            self.cfg.fisheye_config_path = FISHEYE_CFG_STORE
        self.detector = detector
        self.stop_event = threading.Event()
        self.det_stop_event: Optional[threading.Event] = None
        self.preprocessor = None

        self.active_views = lf.ActiveViews(getattr(lf, "FISHEYE_GROUPS", [
            ["middle_row", "front_right_row", "front_left_row", "front_corridor"],
            ["back_right_row", "back_left_row", "entrance", "back_corridor"],
        ]))

        self.frame_queue = queue.Queue(maxsize=4)
        self.raw_frame_queue = queue.Queue(maxsize=4)
        self.frame_queue_a = queue.Queue(maxsize=4)
        self.frame_queue_b = queue.Queue(maxsize=4)
        self.bundle_job_queue = queue.Queue(maxsize=6)
        self.det_out_queue = queue.Queue(maxsize=2)
        self.out_queue = queue.Queue(maxsize=2)

        self.reader_thread = None
        self.processor_thread = None
        self.fanout_thread = None
        self.processor_thread_a = None
        self.processor_thread_b = None
        self.workers: List[threading.Thread] = []
        self.tracking_thread = None
        self.supervisor = None
        self.cache_pump_thread = None
        self.bundle_pump_thread = None

        if _running_under_backend():
            self.cam_out_dir = os.path.dirname(os.path.abspath(self.cfg.roi_config_path))
            os.makedirs(self.cam_out_dir, exist_ok=True)
            self.cam_snap_dir = os.path.join(self.cam_out_dir, "snapshots")
            os.makedirs(self.cam_snap_dir, exist_ok=True)
        else:
            self.cam_out_dir = os.path.join(LAF_BASE, self.cfg.camera_id)
            os.makedirs(self.cam_out_dir, exist_ok=True)
            self.cam_snap_dir = os.path.join(SNAPSHOT_BASE, self.cfg.camera_id)
            os.makedirs(self.cam_snap_dir, exist_ok=True)

        self.event_log_path = os.path.join(self.cam_out_dir, "event_log.jsonl")

        try:
            os.makedirs(os.path.dirname(self.event_log_path), exist_ok=True)

            # touch file so it exists even before first event
            if not os.path.exists(self.event_log_path):
                with open(self.event_log_path, "a", encoding="utf-8"):
                    pass

            self.event_logger = lf.JsonlEventLogger(self.event_log_path)
        except Exception as e:
            print(f"[WARN] event logger init failed for {self.cfg.camera_id}: {e}")
            self.event_logger = None

        try:
            self.lost_manager = lf.LostAndFoundManager(
                lost_seconds=20.0,
                disappear_seconds=20.0,
                enable_snapshots=True,
                snapshot_dir=self.cam_snap_dir,
                enable_owner_association=True,
                near_px=140.0,
                unattended_seconds=10.0,
                logger=self.event_logger,
                autosave_json_path=os.path.join(self.cam_out_dir, "lost_items.json"),
                autosave_csv_path=os.path.join(self.cam_out_dir, "lost_items.csv"),
                autosave_every=3.0,
            )
        except Exception:
            self.lost_manager = None

        self.win_name = f"Lost&Found - Multi-View Preview [{self.cfg.camera_id}]"
        self._tune_state = {"idx": 0}
        self._last_out_views = []

        sw, sh = get_screen_size()
        self.target_w = int(sw * self.cfg.window_scale)
        self.target_h = int(sh * self.cfg.window_scale)

        self.display_dt = 1.0 / max(1e-6, float(self.cfg.display_fps))
        self.next_show_t = time.time()

        self.is_fisheye = False
        self.video_fps = 0.0
        self.target_fps = 0.0
        self.views_expected = 1

        self._detection_enabled = True
        self._roi_available_for_detection = False
        self._last_bundle_by_gi: Dict[int, Any] = {}
        self._last_meta_by_gi: Dict[int, Any] = {}
        self._last_bundle_lock = threading.Lock()

        self.runtime_enabled: bool = True
        self.runtime_enabled_classes: Optional[Dict[str, bool]] = None

        self._roi_cache = None
        self._roi_mtime = 0.0
        self._roi_lock = threading.Lock()
        self._roi_dirty = True

        self._fisheye_cfg_cache = None
        self._fisheye_cfg_mtime = 0.0
        self._fisheye_cfg_lock = threading.Lock()
        self._fisheye_cfg_dirty = True

        self._last_detection_block_reason = None
        self._last_fisheye_cfg_signature = None
        self._last_logged_fisheye_cfg_signature = None
        self._last_fisheye_reload_ts = 0.0

        self._last_raw_frame = None
        self._last_raw_frame_ts = 0.0
        self._last_raw_frame_lock = threading.Lock()

    # ---------------------------------------------------------
    # basic helpers
    # ---------------------------------------------------------
    def _make_fisheye_cfg_signature(self, cfgs: list) -> tuple:
        sig = []
        for x in (cfgs or []):
            if not isinstance(x, dict):
                continue
            sig.append((
                str(x.get("group", "")),
                int(x.get("local_id", 0)),
                int(x.get("view_id", 0)),
                str(x.get("name", "")),
                round(float(x.get("yaw", 0.0)), 4),
                round(float(x.get("pitch", 0.0)), 4),
                round(float(x.get("fov", 0.0)), 4),
                int(x.get("rotate", 0) or 0),
            ))
        return tuple(sig)

    def _is_rtsp_source(self) -> bool:
        try:
            s = str(self.cfg.src or "").strip().lower()
            if s.startswith("rtsp://") or s.startswith("rtsps://"):
                return True
            return str(getattr(self.cfg, "source_kind", "FILE")).upper() == "RTSP"
        except Exception:
            return False

    def _get_base_frame_skip(self) -> int:
        is_rtsp = self._is_rtsp_source()
        if self.is_fisheye:
            return int(
                self.cfg.base_frame_skip_fisheye_rtsp if is_rtsp
                else self.cfg.base_frame_skip_fisheye_file
            )
        return int(
            self.cfg.base_frame_skip_normal_rtsp if is_rtsp
            else self.cfg.base_frame_skip_normal_file
        )

    def _drain_q(self, q: queue.Queue, max_items: int = 5000):
        try:
            n = 0
            while n < max_items:
                q.get_nowait()
                n += 1
        except Exception:
            return

    def _drop_stale_bundle_jobs(self, keep_latest: int = 1):
        try:
            items = []
            while True:
                items.append(self.bundle_job_queue.get_nowait())
        except Exception:
            pass

        if not items:
            return

        sentinels = [x for x in items if isinstance(x, (tuple, list)) and len(x) > 0 and x[0] is lf.SENTINEL]
        normal_items = [x for x in items if not (isinstance(x, (tuple, list)) and len(x) > 0 and x[0] is lf.SENTINEL)]

        if len(normal_items) > keep_latest:
            normal_items = normal_items[-keep_latest:]

        for x in normal_items + sentinels:
            try:
                lf.put_drop_oldest(self.bundle_job_queue, x)
            except Exception:
                pass

    def _drop_stale_tracking_jobs(self, keep_latest: int = 1):
        try:
            items = []
            while True:
                items.append(self.det_out_queue.get_nowait())
        except Exception:
            pass

        if not items:
            return

        sentinels = [x for x in items if isinstance(x, (tuple, list)) and len(x) > 0 and x[0] is lf.SENTINEL]
        normal_items = [x for x in items if not (isinstance(x, (tuple, list)) and len(x) > 0 and x[0] is lf.SENTINEL)]

        if len(normal_items) > keep_latest:
            normal_items = normal_items[-keep_latest:]

        for x in normal_items + sentinels:
            try:
                lf.put_drop_oldest(self.det_out_queue, x)
            except Exception:
                pass

    # ---------------------------------------------------------
    # ROI helpers
    # ---------------------------------------------------------
    def _read_roi_config(self) -> dict:
        try:
            p = self.cfg.roi_config_path
            if p and os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def reload_roi(self):
        try:
            new_cfg = self._read_roi_config() or {}

            with self._roi_lock:
                self._roi_cache = new_cfg
                try:
                    self._roi_mtime = os.path.getmtime(self.cfg.roi_config_path)
                except Exception:
                    self._roi_mtime = time.time()
                self._roi_dirty = False

            self._roi_available_for_detection = self._has_roi_for_detection()

            # IMPORTANT: also push latest ROI into preprocessor runtime
            try:
                if self.preprocessor is not None:
                    if hasattr(self.preprocessor, "reload_roi_config"):
                        try:
                            self.preprocessor.reload_roi_config(self.cfg.roi_config_path)
                        except TypeError:
                            self.preprocessor.reload_roi_config()
                    else:
                        try:
                            setattr(self.preprocessor, "config_path", self.cfg.roi_config_path)
                        except Exception:
                            pass
                        try:
                            setattr(self.preprocessor, "roi_config_path", self.cfg.roi_config_path)
                        except Exception:
                            pass
                        try:
                            setattr(
                                self.preprocessor,
                                "bounding_polygons",
                                new_cfg.get("bounding_polygons", []) or []
                            )
                        except Exception:
                            pass
                        try:
                            setattr(
                                self.preprocessor,
                                "fisheye_polygons",
                                new_cfg.get("fisheye_polygons", {}) or {}
                            )
                        except Exception:
                            pass
            except Exception as e:
                lf._step("WARN", f"{self.cfg.camera_id} preprocessor ROI refresh failed: {e}")

            # IMPORTANT: clear stale cached/queued frames so new ROI appears immediately
            try:
                self._drain_q(self.bundle_job_queue, max_items=999999)
                self._drain_q(self.det_out_queue, max_items=999999)
                self._drain_q(self.out_queue, max_items=999999)

                with self._last_bundle_lock:
                    self._last_bundle_by_gi.clear()
                    self._last_meta_by_gi.clear()
            except Exception:
                pass

            lf._step("UI", f"{self.cfg.camera_id} ROI reloaded")
        except Exception as e:
            lf._step("ERROR", f"{self.cfg.camera_id} reload_roi failed: {e}")

    def _refresh_roi_cache_if_needed(self):
        try:
            need = bool(getattr(self, "_roi_dirty", False))
            try:
                mtime = os.path.getmtime(self.cfg.roi_config_path)
            except Exception:
                mtime = 0.0

            if (not need) and (mtime > float(getattr(self, "_roi_mtime", 0.0))):
                need = True

            if need:
                self.reload_roi()
        except Exception:
            pass

    def _has_normal_roi(self) -> bool:
        try:
            cfg = self._read_roi_config()
            polys = cfg.get("bounding_polygons", []) or []
            return isinstance(polys, list) and len(polys) > 0
        except Exception:
            return False

    def _has_fisheye_roi(self) -> bool:
        try:
            cfg = self._read_roi_config()
            fish = cfg.get("fisheye_polygons", {}) or {}
            if not isinstance(fish, dict):
                return False
            for _, polys in fish.items():
                if isinstance(polys, list) and len(polys) > 0:
                    return True
            return False
        except Exception:
            return False

    def _has_roi_for_detection(self) -> bool:
        return self._has_fisheye_roi() if self.is_fisheye else self._has_normal_roi()

    def _draw_polygon_list(self, img, polys, color=(0, 255, 0), thickness=2):
        if img is None:
            return img
        for poly in (polys or []):
            if isinstance(poly, list) and len(poly) >= 3:
                try:
                    pts = np.array(
                        [[int(p["x"]), int(p["y"])] for p in poly if isinstance(p, dict)],
                        dtype=np.int32,
                    )
                    if len(pts) >= 3:
                        cv2.polylines(img, [pts], True, color, thickness)
                except Exception:
                    pass
        return img

    def _current_normal_polys(self):
        cfg = self._get_roi_cfg_cached()
        return cfg.get("bounding_polygons", []) or []

    def _current_fisheye_polys(self, view_name: str):
        cfg = self._get_roi_cfg_cached()
        fish = cfg.get("fisheye_polygons", {}) or {}
        vals = fish.get(view_name, [])
        return vals if isinstance(vals, list) else []

    # ---------------------------------------------------------
    # fisheye config helpers
    # ---------------------------------------------------------
    def _get_fisheye_cfg_store_path(self) -> str:
        """
        Always prefer the fixed fisheye config store path.
        If cfg provides a custom path, use it only if it exists.
        Otherwise fall back to the fixed path.
        """
        try:
            p = str(getattr(self.cfg, "fisheye_config_path", "") or "").strip()
            if p:
                p = os.path.abspath(p)
                if os.path.exists(p):
                    return p
        except Exception:
            pass

        return os.path.abspath(FISHEYE_CFG_STORE)

    def _default_fisheye_view_configs(self) -> list:
        try:
            cfgs = getattr(lf, "FISHEYE_VIEW_CONFIGS", None)
            if isinstance(cfgs, list) and cfgs:
                return [dict(x) for x in cfgs if isinstance(x, dict)]
        except Exception:
            pass

        try:
            cfgs = getattr(lf, "DEFAULT_VIEW_CONFIGS", None)
            if isinstance(cfgs, list) and cfgs:
                return [dict(x) for x in cfgs if isinstance(x, dict)]
        except Exception:
            pass

        return []

    def _normalize_fisheye_cfg_list(self, cfg_list: list) -> list:
        out = []
        if not isinstance(cfg_list, list):
            return out

        for i, rec in enumerate(cfg_list):
            if not isinstance(rec, dict):
                continue

            try:
                raw_view_id = int(rec.get("view_id", i))
            except Exception:
                raw_view_id = i

            if raw_view_id < 4:
                group = "A"
                local_id = raw_view_id
            else:
                group = "B"
                local_id = raw_view_id - 4

            if rec.get("group") is not None:
                group = str(rec.get("group")).strip().upper()
            if rec.get("local_id") is not None:
                try:
                    local_id = int(rec.get("local_id"))
                except Exception:
                    pass

            out.append({
                "view_id": raw_view_id,
                "group": group,
                "local_id": local_id,
                "name": str(rec.get("name", f"view_{raw_view_id}")),
                "yaw": float(rec.get("yaw", 0.0)),
                "pitch": float(rec.get("pitch", 0.0)),
                "fov": float(rec.get("fov", 90.0)),
                "rotate": int(rec.get("rotate", 0) or 0),
            })

        out.sort(key=lambda x: (
            0 if str(x.get("group", "A")).upper() == "A" else 1,
            int(x.get("local_id", 0)),
        ))
        return out

    def _load_latest_fisheye_configs_for_cam(self) -> list:
        try:
            store_path = self._get_fisheye_cfg_store_path()
            if not store_path or not os.path.exists(store_path):
                print(f"[WARN] [{self.cfg.camera_id}] fisheye cfg file not found: {store_path}")
                return []

            with open(store_path, "r", encoding="utf-8") as f:
                allcfg = json.load(f) or {}

            cam_id = str(self.cfg.camera_id or "").strip()

            candidate_keys = []
            if cam_id:
                candidate_keys.append(cam_id)

                # upload fallback: remove trailing _h264 if any
                if cam_id.endswith("_h264"):
                    candidate_keys.append(cam_id[:-5])

                # basename fallback
                base = os.path.splitext(os.path.basename(str(self.cfg.src or "")))[0]
                if base:
                    candidate_keys.append(base)
                    if base.endswith("_h264"):
                        candidate_keys.append(base[:-5])

            # remove duplicates while keeping order
            seen = set()
            candidate_keys = [k for k in candidate_keys if not (k in seen or seen.add(k))]

            chosen_key = None
            cfg = None
            for k in candidate_keys:
                vv = allcfg.get(k)
                if isinstance(vv, list) and vv:
                    chosen_key = k
                    cfg = vv
                    break

            if not cfg:
                print(f"[WARN] [{self.cfg.camera_id}] no fisheye config found in {store_path}")
                print(f"[DEBUG] candidate_keys={candidate_keys}")
                print(f"[DEBUG] available_keys={list(allcfg.keys())}")
                return []

            out = self._normalize_fisheye_cfg_list(cfg)

            return out

        except Exception as e:
            print(f"[ERROR] [{self.cfg.camera_id}] _load_latest_fisheye_configs_for_cam failed: {e}")
            return []

    def _get_priority_fisheye_configs(self) -> list:
        cfgs = self._load_latest_fisheye_configs_for_cam()
        if cfgs:
            return cfgs

        fallback = self._normalize_fisheye_cfg_list(self._default_fisheye_view_configs())
        print(f"[WARN] [{self.cfg.camera_id}] USING DEFAULT fisheye config")
        print(f"[WARN] [{self.cfg.camera_id}] default views = {[x.get('name') for x in fallback]}")
        return fallback

    def _runtime_fisheye_groups_from_configs(self, cfgs: list | None = None) -> list:
        try:
            cfgs = cfgs or getattr(self.preprocessor, "view_configs", None) or self._get_priority_fisheye_configs()
            cfgs = self._normalize_fisheye_cfg_list(cfgs)

            groups = {
                "A": [None, None, None, None],
                "B": [None, None, None, None],
            }

            for rec in cfgs:
                if not isinstance(rec, dict):
                    continue

                group = str(rec.get("group", "A")).strip().upper()
                if group not in ("A", "B"):
                    group = "A"

                try:
                    local_id = int(rec.get("local_id", 0))
                except Exception:
                    local_id = 0

                if 0 <= local_id <= 3:
                    groups[group][local_id] = str(rec.get("name", f"{group}_view_{local_id}"))

            for g in ("A", "B"):
                for i in range(4):
                    if not groups[g][i]:
                        groups[g][i] = f"{g}_view_{i}"

            return [groups["A"], groups["B"]]
        except Exception:
            return list(getattr(lf, "FISHEYE_GROUPS", [
                ["middle_row", "front_right_row", "front_left_row", "front_corridor"],
                ["back_right_row", "back_left_row", "entrance", "back_corridor"],
            ]))

    def _make_active_views_pair(self, cfgs: list | None = None):
        groups = self._runtime_fisheye_groups_from_configs(cfgs)
        av_a = lf.ActiveViews(groups)
        av_b = lf.ActiveViews(groups)
        try:
            av_b.toggle()
        except Exception:
            pass
        return av_a, av_b

    def _refresh_processor_active_views(self, cfgs: list | None = None):
        if not self.is_fisheye:
            return

        try:
            groups = self._runtime_fisheye_groups_from_configs(cfgs)
            self.active_views = lf.ActiveViews(groups)

            if self.processor_thread_a is not None:
                self.processor_thread_a.active_views = lf.ActiveViews(groups)

            if self.processor_thread_b is not None:
                av_b = lf.ActiveViews(groups)
                try:
                    av_b.toggle()
                except Exception:
                    pass
                self.processor_thread_b.active_views = av_b

        except Exception as e:
            lf._step("WARN", f"{self.cfg.camera_id} _refresh_processor_active_views failed: {e}")

    def _fisheye_expected_names(self, gi: int) -> list:
        try:
            want = "A" if int(gi) == 0 else "B"

            view_configs = getattr(self.preprocessor, "view_configs", None)
            if not view_configs:
                view_configs = self._get_priority_fisheye_configs()

            slots = [None, None, None, None]

            for cfg in (view_configs or []):
                if not isinstance(cfg, dict):
                    continue

                group = str(cfg.get("group", "A")).strip().upper()
                if group != want:
                    continue

                try:
                    li = int(cfg.get("local_id", 0))
                except Exception:
                    continue

                if 0 <= li <= 3:
                    slots[li] = str(cfg.get("name", f"{want}_view_{li}"))

            for i in range(4):
                if not slots[i]:
                    slots[i] = f"{want}_view_{i}"

            return slots
        except Exception:
            return (
                ["A_view_0", "A_view_1", "A_view_2", "A_view_3"]
                if int(gi) == 0 else
                ["B_view_0", "B_view_1", "B_view_2", "B_view_3"]
            )

    def _force_apply_fisheye_configs(self, cfgs: list) -> bool:
        if not self.is_fisheye or self.preprocessor is None:
            return False

        if not isinstance(cfgs, list) or not cfgs:
            return False

        try:
            cfgs = self._normalize_fisheye_cfg_list(cfgs)

            with self._fisheye_cfg_lock:
                self._fisheye_cfg_cache = [dict(x) for x in cfgs]

            setattr(self.preprocessor, "view_configs", [dict(x) for x in cfgs])

            rebuilt = False

            try:
                if hasattr(self.preprocessor, "reload_view_configs"):
                    ok = self.preprocessor.reload_view_configs([dict(x) for x in cfgs])
                    rebuilt = bool(ok)
            except Exception:
                rebuilt = False

            if not rebuilt:
                try:
                    if hasattr(self.preprocessor, "_build_all_maps"):
                        self.preprocessor._build_all_maps()
                        rebuilt = True
                except Exception:
                    rebuilt = False

            if rebuilt:
                try:
                    self._refresh_processor_active_views(cfgs)

                    self._drain_q(self.bundle_job_queue, max_items=999999)
                    self._drain_q(self.out_queue, max_items=999999)
                    self._drain_q(self.det_out_queue, max_items=999999)

                    with self._last_bundle_lock:
                        self._last_bundle_by_gi.clear()
                        self._last_meta_by_gi.clear()
                except Exception:
                    pass

            return rebuilt
        except Exception:
            return False

    def force_fisheye_config_priority_apply(self) -> bool:
        if not self.is_fisheye:
            return False

        try:
            cfgs = self._get_priority_fisheye_configs()
            if not cfgs:
                return False

            ok = self._force_apply_fisheye_configs(cfgs)

            try:
                store_path = self._get_fisheye_cfg_store_path()
                if store_path and os.path.exists(store_path):
                    self._fisheye_cfg_mtime = os.path.getmtime(store_path)
            except Exception:
                pass

            self._fisheye_cfg_dirty = False

            if ok:
                lf._step("UI", f"{self.cfg.camera_id} fisheye config FORCE-applied")
            else:
                lf._step("WARN", f"{self.cfg.camera_id} fisheye config loaded but map rebuild fallback may be partial")

            return ok
        except Exception as e:
            lf._step("ERROR", f"{self.cfg.camera_id} force_fisheye_config_priority_apply failed: {e}")
            return False

    def reload_fisheye_config(self):
        if not self.is_fisheye or self.preprocessor is None:
            return

        try:
            cfgs = self._load_latest_fisheye_configs_for_cam()
            if not cfgs:
                return

            new_sig = self._make_fisheye_cfg_signature(cfgs)
            old_sig = self._last_fisheye_cfg_signature

            # same config -> skip reload completely
            if old_sig is not None and new_sig == old_sig:
                try:
                    store_path = self._get_fisheye_cfg_store_path()
                    if store_path and os.path.exists(store_path):
                        self._fisheye_cfg_mtime = os.path.getmtime(store_path)
                except Exception:
                    pass
                self._fisheye_cfg_dirty = False
                return

            ok = False
            if hasattr(self.preprocessor, "reload_view_configs"):
                ok = bool(self.preprocessor.reload_view_configs(cfgs))

            if not ok:
                try:
                    self.preprocessor.view_configs = [dict(x) for x in cfgs]
                    if hasattr(self.preprocessor, "_build_all_maps"):
                        self.preprocessor._build_all_maps()
                        ok = True
                except Exception:
                    ok = False

            if not ok:
                return

            self._refresh_processor_active_views(cfgs)

            self._drain_q(self.bundle_job_queue, max_items=999999)
            self._drain_q(self.out_queue, max_items=999999)
            self._drain_q(self.det_out_queue, max_items=999999)

            with self._last_bundle_lock:
                self._last_bundle_by_gi.clear()
                self._last_meta_by_gi.clear()

            try:
                store_path = self._get_fisheye_cfg_store_path()
                if store_path and os.path.exists(store_path):
                    self._fisheye_cfg_mtime = os.path.getmtime(store_path)
            except Exception:
                pass

            self._fisheye_cfg_dirty = False
            self._last_fisheye_cfg_signature = new_sig

            # only one concise log per actual config change
            lf._step("UI", f"{self.cfg.camera_id} fisheye config applied")
            lf._step("SYSTEM", f"{self.cfg.camera_id} fisheye views={[x.get('name') for x in cfgs]}")

        except Exception as e:
            lf._step("ERROR", f"{self.cfg.camera_id} reload_fisheye_config failed: {e}")

    def _refresh_fisheye_config_if_needed(self):
        if not self.is_fisheye:
            return

        try:
            store_path = self._get_fisheye_cfg_store_path()
            if not store_path or not os.path.exists(store_path):
                return

            try:
                mtime = os.path.getmtime(store_path)
            except Exception:
                mtime = 0.0

            need = bool(self._fisheye_cfg_dirty)

            if mtime > float(getattr(self, "_fisheye_cfg_mtime", 0.0)):
                need = True

            if not need:
                try:
                    pvc = getattr(self.preprocessor, "view_configs", None)
                    pmaps = getattr(self.preprocessor, "remap_maps", None)
                    if not pvc or not pmaps:
                        need = True
                except Exception:
                    need = True

            if not need:
                return

            cfgs = self._load_latest_fisheye_configs_for_cam()
            if not cfgs:
                return

            new_sig = self._make_fisheye_cfg_signature(cfgs)
            old_sig = self._last_fisheye_cfg_signature

            # file touched but content same -> just update mtime, do not reload/log
            if old_sig is not None and new_sig == old_sig:
                self._fisheye_cfg_mtime = mtime
                self._fisheye_cfg_dirty = False
                return

            self.reload_fisheye_config()

        except Exception as e:
            print(f"[ERROR] [{self.cfg.camera_id}] _refresh_fisheye_config_if_needed failed: {e}")

    def mark_fisheye_config_dirty(self):
        try:
            self._fisheye_cfg_dirty = True
        except Exception:
            pass

    # ---------------------------------------------------------
    # preprocessor open
    # ---------------------------------------------------------
    def _normalized_video_type(self) -> str:
        ft = (self.cfg.force_video_type or "").strip().lower()
        if ft in ("fisheye", "fish"):
            return "FISHEYE"
        if ft in ("normal", "norm"):
            return "NORMAL"

        s = (self.cfg.src or "").strip().lower()
        if s.startswith("rtsp://") or s.startswith("rtsps://"):
            try:
                v = lf.detect_video_type(self.cfg.src, max_samples=3, max_scan_seconds=2.0)
                if "fish" in str(v).lower():
                    return "FISHEYE"
            except Exception:
                pass
            return "NORMAL"

        try:
            v = lf.detect_video_type(self.cfg.src)
            if "fish" in str(v).lower():
                return "FISHEYE"
        except Exception:
            pass
        return "NORMAL"

    def _should_draw_roi_overlay(self) -> bool:
        return bool(self._detection_enabled)

    def _open_preprocessor(self, vtype: str) -> bool:
        if vtype == "FISHEYE":
            try:
                view_configs = self._get_priority_fisheye_configs()
                try:
                    self.preprocessor = lf.FisheyePreprocessor(
                        view_configs=view_configs,
                        config_path=self.cfg.roi_config_path,
                    )
                except TypeError:
                    self.preprocessor = lf.FisheyePreprocessor(
                        view_configs,
                        config_path=self.cfg.roi_config_path,
                    )

                opened = bool(self.preprocessor.open(self.cfg.src))
                if not opened:
                    return False

                self.reload_fisheye_config()

                if self.cfg.src.lower().startswith("rtsp"):
                    self._warm_up_fisheye_cap()

                return True

            except Exception as e:
                print(f"[ERROR] Fisheye preprocessor: {e!r}")
                return False
        try:
            has_normal_roi = self._has_normal_roi()

            logger_obj = getattr(lf, "logger", None)
            old_level = None

            if (not has_normal_roi) and logger_obj is not None:
                try:
                    old_level = logger_obj.level
                    logger_obj.setLevel(logging.ERROR)
                except Exception:
                    old_level = None

            try:
                try:
                    self.preprocessor = lf.NormalPreprocessor(self.cfg.src, config_path=self.cfg.roi_config_path)
                except TypeError:
                    self.preprocessor = lf.NormalPreprocessor(self.cfg.src, self.cfg.roi_config_path)
                return bool(self.preprocessor.open())
            finally:
                if logger_obj is not None and old_level is not None:
                    try:
                        logger_obj.setLevel(old_level)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[ERROR] Normal preprocessor: {e!r}")
            return False

    def _warm_up_fisheye_cap(self):
        try:
            cap = getattr(self.preprocessor, "cap", None)
            if cap is None:
                return
            warmup = 8
            for _ in range(warmup):
                ret, _ = cap.read()
                if not ret:
                    break
            print(f"[OK] Fisheye RTSP warmup: {warmup} frames discarded")
        except Exception as e:
            print(f"[WARN] Fisheye warmup: {e!r}")

    # ---------------------------------------------------------
    # threading / fanout
    # ---------------------------------------------------------
    def _fanout_frames_loop(self):
        while not self.stop_event.is_set():
            try:
                item = self.raw_frame_queue.get(timeout=0.1)
            except Exception:
                continue

            if isinstance(item, (tuple, list)) and len(item) > 0 and item[0] is lf.SENTINEL:
                try:
                    lf.put_drop_oldest(self.frame_queue_a, item)
                except Exception:
                    pass
                try:
                    lf.put_drop_oldest(self.frame_queue_b, item)
                except Exception:
                    pass
                break

            # Cache latest RAW frame before downstream processing
            try:
                if isinstance(item, (tuple, list)) and len(item) >= 3:
                    raw_fr = item[2]
                    if isinstance(raw_fr, np.ndarray):
                        with self._last_raw_frame_lock:
                            self._last_raw_frame = raw_fr.copy()
                            self._last_raw_frame_ts = time.time()
            except Exception:
                pass

            try:
                lf.put_drop_oldest(self.frame_queue_a, item)
            except Exception:
                pass
            try:
                lf.put_drop_oldest(self.frame_queue_b, item)
            except Exception:
                pass

    def _get_latest_raw_frame_copy(self) -> np.ndarray | None:
        try:
            with self._last_raw_frame_lock:
                if isinstance(self._last_raw_frame, np.ndarray):
                    return self._last_raw_frame.copy()
        except Exception:
            pass
        return None
    
    def _encode_jpg(self, img_bgr: np.ndarray | None, quality: int = 70) -> bytes | None:
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            return None
        try:
            ok, buf = cv2.imencode(
                ".jpg",
                img_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            )
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def _get_cached_views_for_group(self, gi: int) -> list:
        try:
            with self._last_bundle_lock:
                ts_views = self._last_bundle_by_gi.get(int(gi))
            if not ts_views:
                return []
            _, views = ts_views
            return views or []
        except Exception:
            return []

    def _start_detection_threads(self):
        self.det_stop_event = threading.Event()
        self._drain_q(self.det_out_queue)
        self._drain_q(self.out_queue)
        self._drain_q(self.bundle_job_queue, max_items=999999)

        analysis_fps = self.target_fps if self._is_rtsp_source() else None

        self.workers = [
            lf.DetectionWorkerThread(
                i,
                self.detector,
                self.bundle_job_queue,
                self.det_out_queue,
                self.det_stop_event,
                batch_size=1,
                analysis_fps=analysis_fps,
            )
            for i in range(int(self.cfg.num_workers))
        ]

        self.tracking_thread = TrackingThreadROIMatchLF(
            self.det_out_queue,
            self.out_queue,
            self.det_stop_event,
            self.detector,
            confirm_k=3,
        )

        if self.lost_manager:
            try:
                self.tracking_thread.lost_manager = self.lost_manager
            except Exception:
                pass

        self.supervisor = lf.SupervisorThread(
            reader_thread=self.reader_thread,
            bundle_job_queue=self.bundle_job_queue,
            stop_event=self.det_stop_event,
            max_skip=int(self.cfg.max_skip),
        )

        for w in self.workers:
            w.start()

        self.tracking_thread.start()
        self.supervisor.start()

        self._stop_display_only_pumps()
        self._start_cache_pump()

    def _stop_detection_threads(self):
        if self.det_stop_event:
            try:
                self.det_stop_event.set()
            except Exception:
                pass

        for w in self.workers:
            try:
                w.join(timeout=0.3)
            except Exception:
                pass

        for t in [self.tracking_thread, self.supervisor]:
            try:
                if t:
                    t.join(timeout=0.3)
            except Exception:
                pass

        self.workers = []
        self.tracking_thread = None
        self.supervisor = None
        self.det_stop_event = None
        self._stop_cache_pump()
        self._drain_q(self.det_out_queue, max_items=999999)
        self._drain_q(self.out_queue, max_items=999999)

    def _start_display_only_pumps(self):
        self._stop_cache_pump()
        if not self.bundle_pump_thread or not self.bundle_pump_thread.is_alive():
            self.bundle_pump_thread = threading.Thread(target=self._bundle_cache_pump, daemon=True)
            self.bundle_pump_thread.start()

    def _stop_display_only_pumps(self):
        pass

    def _start_cache_pump(self):
        if not self.cache_pump_thread or not self.cache_pump_thread.is_alive():
            self.cache_pump_thread = threading.Thread(target=self._cache_pump, daemon=True)
            self.cache_pump_thread.start()

    def _stop_cache_pump(self):
        pass

    # ---------------------------------------------------------
    # public lifecycle
    # ---------------------------------------------------------
    def set_detection_enabled(self, enabled: bool):
        enabled = bool(enabled)
        old_enabled = bool(self._detection_enabled)

        try:
            self._refresh_roi_cache_if_needed()
        except Exception:
            pass

        if enabled and not self._has_roi_for_detection():
            self._roi_available_for_detection = False
            self._detection_enabled = False

            last_reason = getattr(self, "_last_detection_block_reason", None)
            new_reason = "no_roi"
            if old_enabled or last_reason != new_reason:
                self._last_detection_block_reason = new_reason
                lf._step("UI", f"{self.cfg.camera_id} detection remains OFF (no ROI configured)")
            return

        self._last_detection_block_reason = None

        if enabled == self._detection_enabled:
            return

        self._detection_enabled = enabled
        lf._step("UI", f"{self.cfg.camera_id} detection={'ON' if enabled else 'OFF'}")

        try:
            with self._last_bundle_lock:
                self._last_bundle_by_gi.clear()
                self._last_meta_by_gi.clear()
        except Exception:
            pass

        try:
            self._drain_q(self.out_queue, max_items=999999)
            self._drain_q(self.det_out_queue, max_items=999999)
            self._drain_q(self.bundle_job_queue, max_items=999999)
        except Exception:
            pass

        if enabled:
            self._stop_display_only_pumps()
            self._start_detection_threads()
        else:
            self._stop_detection_threads()
            self._start_display_only_pumps()

    def start(self):
        try:
            lf._step("PHASE 2", f"Starting pipeline for {self.cfg.camera_id}")
            try:
                lf._system(
                    f"DETECTOR GROUP MAP cam={self.cfg.camera_id} -> group={self.cfg.detector_group}"
                )
            except Exception:
                print(f"[SYSTEM] DETECTOR GROUP MAP cam={self.cfg.camera_id} -> group={self.cfg.detector_group}")
            # ---------------------------------------------------------
            # reset runtime state
            # ---------------------------------------------------------
            self.stop_event.clear()
            self.det_stop_event = None

            self._detection_enabled = bool(self.cfg.enable_detection)

            try:
                self._roi_dirty = True
                self.reload_roi()
            except Exception:
                self._roi_available_for_detection = self._has_roi_for_detection()

            try:
                self._fisheye_cfg_dirty = True
            except Exception:
                pass

            # ---------------------------------------------------------
            # decide video type
            # ---------------------------------------------------------
            mode = self._normalized_video_type()
            self.is_fisheye = (str(mode).upper() == "FISHEYE")

            # ---------------------------------------------------------
            # open preprocessor
            # ---------------------------------------------------------
            if self.is_fisheye:
                self.preprocessor = lf.FisheyePreprocessor(
                    view_configs=self._load_latest_fisheye_configs_for_cam() or self._default_fisheye_view_configs(),
                    config_path=self.cfg.roi_config_path,
                )
                self.preprocessor.open(self.cfg.src)
                self.views_expected = 8
            else:
                self.preprocessor = lf.NormalPreprocessor(
                    self.cfg.src,
                    config_path=self.cfg.roi_config_path,
                )
                ok = self.preprocessor.open()
                if ok is False:
                    raise RuntimeError(f"Cannot open source: {self.cfg.src}")
                self.views_expected = 1

            # ---------------------------------------------------------
            # read source FPS
            # ---------------------------------------------------------
            cap = getattr(self.preprocessor, "cap", None)
            fps = 0.0
            if cap is not None:
                try:
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                except Exception:
                    fps = 0.0

            if not fps or fps <= 1.0:
                fps = 25.0 if self._is_rtsp_source() else 25.0

            self.video_fps = float(fps)

            desired = self.cfg.desired_fps_fisheye if self.is_fisheye else self.cfg.desired_fps_normal
            desired = float(desired or 1.0)

            if self._is_rtsp_source():
                # RTSP: use requested analysis FPS directly
                self.target_fps = max(0.1, desired)
            else:
                # FILE: do not exceed native FPS
                self.target_fps = min(float(desired), float(self.video_fps))

            # ---------------------------------------------------------
            # refresh ROI availability
            # ---------------------------------------------------------
            self._roi_available_for_detection = self._has_roi_for_detection()

            # ---------------------------------------------------------
            # RTSP or FILE reader choice
            # ---------------------------------------------------------
            is_rtsp = self._is_rtsp_source()

            if self.is_fisheye:
                if is_rtsp:
                    # IMPORTANT:
                    # fisheye RTSP must feed raw_frame_queue because fanout thread reads
                    # raw_frame_queue -> frame_queue_a / frame_queue_b
                    self.reader_thread = RealtimeFreshRTSPReaderThread(
                        self.preprocessor,
                        self.raw_frame_queue,
                        self.stop_event,
                        target_fps=self.cfg.display_fps,
                        warmup_grabs=0,
                        flush_grabs_per_cycle=0,
                        buffer_delay_sec=0.5,  # ADD THIS: 2.5 second delay
                    )
                else:
                    self.reader_thread = lf.FrameReaderThread(
                        self.preprocessor,
                        self.raw_frame_queue,
                        self.stop_event,
                        frame_skip=self._get_base_frame_skip(),
                        target_fps=None,
                        is_realtime=False,
                    )

                self.fanout_thread = threading.Thread(
                    target=self._fanout_frames_loop,
                    daemon=True
                )

                av_a, av_b = self._make_active_views_pair(
                    getattr(self.preprocessor, "view_configs", None)
                )

                self.processor_thread_a = ViewProcessorThreadCam(
                    self.cfg.camera_id,
                    self.preprocessor,
                    self.frame_queue_a,
                    self.bundle_job_queue,
                    self.stop_event,
                    av_a,
                )

                self.processor_thread_b = ViewProcessorThreadCam(
                    self.cfg.camera_id,
                    self.preprocessor,
                    self.frame_queue_b,
                    self.bundle_job_queue,
                    self.stop_event,
                    av_b,
                )

                self.reader_thread.start()
                self.fanout_thread.start()
                self.processor_thread_a.start()
                self.processor_thread_b.start()

            else:
                if is_rtsp:
                    self.reader_thread = RealtimeFreshRTSPReaderThread(
                        self.preprocessor,
                        self.frame_queue,
                        self.stop_event,
                        target_fps=self.cfg.display_fps,
                        warmup_grabs=0,
                        flush_grabs_per_cycle=0,
                        buffer_delay_sec=0.5, 
                    )
                else:
                    self.reader_thread = lf.FrameReaderThread(
                        self.preprocessor,
                        self.frame_queue,
                        self.stop_event,
                        frame_skip=self._get_base_frame_skip(),
                        target_fps=None,
                        is_realtime=False,
                    )

                self.processor_thread = ViewProcessorThreadCam(
                    self.cfg.camera_id,
                    self.preprocessor,
                    self.frame_queue,
                    self.bundle_job_queue,
                    self.stop_event,
                    self.active_views,
                )

                self.reader_thread.start()
                self.processor_thread.start()

            # ---------------------------------------------------------
            # detection or display-only mode
            # ---------------------------------------------------------
            detection_can_run = bool(
                self.cfg.enable_detection
                and self._detection_enabled
                and self._roi_available_for_detection
            )

            if detection_can_run:
                self._start_detection_threads()
            else:
                self._detection_enabled = False
                self._start_display_only_pumps()

                if self.cfg.enable_detection and not self._roi_available_for_detection:
                    lf._step("UI", f"{self.cfg.camera_id} detection skipped (no ROI configured)")

            # ---------------------------------------------------------
            # optional UI window
            # ---------------------------------------------------------
            if self.cfg.show_ui:
                try:
                    cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.win_name, self.target_w, self.target_h)
                except Exception:
                    pass

            # ---------------------------------------------------------
            # concise runtime log
            # ---------------------------------------------------------
            lf._kv(
                "PHASE 2",
                cam=self.cfg.camera_id,
                video=self.cfg.src,
                mode=("FISHEYE" if self.is_fisheye else "NORMAL"),
                source=str(self.cfg.source_kind or "FILE"),
                video_fps=round(self.video_fps, 2),
                target_fps=round(self.target_fps, 2),
                views_expected=self.views_expected,
                roi_ready=self._roi_available_for_detection,
                enable_detection=bool(detection_can_run),
                frame_skip=self._get_base_frame_skip(),
                detector_group=self.cfg.detector_group,
            )

            return True

        except Exception as e:
            lf._step("ERROR", f"{self.cfg.camera_id} start failed: {e}")
            return False

    def stop(self):
        try:
            self.stop_event.set()
        except Exception:
            pass
        try:
            if self.det_stop_event:
                self.det_stop_event.set()
        except Exception:
            pass

        for q_item, sentinel in [
            (self.raw_frame_queue, (lf.SENTINEL, None, None)),
            (self.frame_queue, (lf.SENTINEL, None, None)),
            (self.bundle_job_queue, (lf.SENTINEL, None, None)),
            (self.det_out_queue, (lf.SENTINEL, None, None, None)),
            (self.out_queue, (lf.SENTINEL, None, None, None)),
        ]:
            try:
                lf.put_drop_oldest(q_item, sentinel)
            except Exception:
                pass

    def join(self, timeout=2.0):
        for t in [
            self.reader_thread, self.fanout_thread, self.processor_thread,
            self.processor_thread_a, self.processor_thread_b,
            self.tracking_thread, self.supervisor,
            self.cache_pump_thread, self.bundle_pump_thread,
        ]:
            try:
                if t:
                    t.join(timeout=timeout)
            except Exception:
                pass

        for w in self.workers:
            try:
                w.join(timeout=timeout)
            except Exception:
                pass

        try:
            if self.preprocessor is not None:
                self.preprocessor.release()
        except Exception:
            pass

    def is_finished_local_video(self) -> bool:
        try:
            return (self.reader_thread is not None) and (not self.reader_thread.is_alive()) and self.frame_queue.empty()
        except Exception:
            return False
    def _refresh_runtime_configs_if_needed(self):
        try:
            self._refresh_roi_cache_if_needed()
        except Exception:
            pass

        try:
            self._refresh_fisheye_config_if_needed()
        except Exception:
            pass

    def _get_roi_cfg_cached(self) -> dict:
        self._refresh_roi_cache_if_needed()
        with self._roi_lock:
            return dict(self._roi_cache or {})
    # ---------------------------------------------------------
    # UI preview
    # ---------------------------------------------------------
    def tick_ui(self):
        if not self.cfg.show_ui:
            return

        self._refresh_runtime_configs_if_needed()

        now = time.time()
        if now < self.next_show_t:
            return
        self.next_show_t = now + self.display_dt

        out_views = None

        latest = None
        try:
            while True:
                latest = self.out_queue.get_nowait()
        except Exception:
            pass

        if latest is not None and isinstance(latest, (tuple, list)) and len(latest) >= 4:
            frame_idx, ts, gi, maybe_views = latest[:4]
            if frame_idx is not lf.SENTINEL:
                out_views = maybe_views or []

        if out_views is None:
            try:
                with self._last_bundle_lock:
                    ts_views = self._last_bundle_by_gi.get(0)
                    if ts_views:
                        _, maybe_views = ts_views
                        out_views = maybe_views or []
            except Exception:
                out_views = None

        if not out_views:
            return

        self._last_out_views = out_views or []

        try:
            if (not self.is_fisheye) and (not self._has_normal_roi()):
                v0 = out_views[0] if out_views else None
                img = v0.get("image") if isinstance(v0, dict) else None
                if img is not None:
                    gh, gw = img.shape[:2]
                    scale = min(self.target_w / gw, self.target_h / gh)
                    shown = cv2.resize(
                        img,
                        (max(1, int(gw * scale)), max(1, int(gh * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imshow(self.win_name, shown)
                return
        except Exception:
            pass

        expected = self.views_expected
        try:
            if self.is_fisheye:
                allowed, _ = self.active_views.current()
                expected = len(allowed)
        except Exception:
            pass

        try:
            if out_views and len(out_views) == expected:
                out_views = sorted(out_views, key=lambda x: x.get("view_id", 0))
                grid = lf.build_views_grid(
                    out_views,
                    draw_zones=self._should_draw_roi_overlay()
                )
                if grid is not None:
                    gh, gw = grid.shape[:2]
                    scale = min(self.target_w / gw, self.target_h / gh)
                    grid = cv2.resize(
                        grid,
                        (max(1, int(gw * scale)), max(1, int(gh * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imshow(self.win_name, grid)
        except Exception:
            pass

    # ---------------------------------------------------------
    # cache pumps
    # ---------------------------------------------------------
    def _cache_pump(self):
        while not self.stop_event.is_set():
            if self.is_fisheye:
                self._refresh_fisheye_config_if_needed()

            if not self._detection_enabled:
                time.sleep(0.01)
                continue

            if self._is_rtsp_source() and getattr(self.cfg, "latest_only_tracking", True):
                try:
                    self._drop_stale_tracking_jobs(keep_latest=1)
                except Exception:
                    pass

            latest = None
            try:
                while True:
                    latest = self.out_queue.get_nowait()
            except Exception:
                pass

            if latest is not None:
                try:
                    if isinstance(latest, (tuple, list)) and len(latest) >= 4:
                        frame_idx, ts, gi, out_views = latest[:4]
                        if frame_idx is lf.SENTINEL:
                            continue

                        meta = {}
                        if len(latest) > 4:
                            meta["extras"] = list(latest[4:])
                        meta["raw_latest_type"] = type(latest).__name__
                        meta["raw_latest_len"] = len(latest)

                        with self._last_bundle_lock:
                            self._last_bundle_by_gi[int(gi or 0)] = (
                                float(ts or time.time()),
                                out_views or [],
                            )
                            self._last_meta_by_gi[int(gi or 0)] = meta
                except Exception:
                    pass

            time.sleep(0.005)

    def _bundle_cache_pump(self):
        while not self.stop_event.is_set():
            if self.is_fisheye:
                self._refresh_fisheye_config_if_needed()

            if self._detection_enabled:
                time.sleep(0.01)
                continue

            if self._is_rtsp_source() and getattr(self.cfg, "drop_old_detection_jobs", True):
                try:
                    self._drop_stale_bundle_jobs(keep_latest=1)
                except Exception:
                    pass

            latest = None
            try:
                while True:
                    latest = self.bundle_job_queue.get_nowait()
            except Exception:
                pass

            if latest is not None:
                try:
                    if isinstance(latest, (tuple, list)) and len(latest) >= 3:
                        frame_idx = latest[0]
                        ts = latest[1]
                        gi = 0
                        out_views = None
                        third = latest[2]
                        fourth = latest[3] if len(latest) > 3 else None

                        if isinstance(third, (int, np.integer)) and isinstance(fourth, list):
                            gi = int(third)
                            out_views = fourth
                        elif isinstance(third, list):
                            gi = 0
                            out_views = third

                        if frame_idx is not lf.SENTINEL and out_views is not None:
                            with self._last_bundle_lock:
                                self._last_bundle_by_gi[int(gi)] = (
                                    float(ts or time.time()),
                                    out_views or [],
                                )
                                self._last_meta_by_gi[int(gi)] = {"display_only": True}
                except Exception:
                    pass

            time.sleep(0.003)

    # ---------------------------------------------------------
    # api helpers
    # ---------------------------------------------------------
    def _to_plain(self, x):
        try:
            if x is None:
                return None
            if isinstance(x, (str, int, float, bool)):
                return x
            if isinstance(x, dict):
                return {str(k): self._to_plain(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [self._to_plain(v) for v in x]
            if is_dataclass(x):
                return self._to_plain(asdict(x))
            if hasattr(x, "__dict__"):
                return self._to_plain(dict(x.__dict__))
            return str(x)
        except Exception:
            return str(x)

    def _collect_dets_from_views(self, views):
        dets_out = []
        if not views:
            return dets_out

        candidate_keys = [
            "tracked_detections", "final_detections", "raw_detections",
            "detections", "tracks", "items", "objects",
        ]

        for v in (views or []):
            if not isinstance(v, dict):
                continue
            raw = None
            for k in candidate_keys:
                if k in v and v.get(k) is not None:
                    raw = v.get(k)
                    break
            if raw is None:
                continue
            if not isinstance(raw, list):
                raw = [raw]
            for d in raw:
                dd = self._to_plain(d)
                if isinstance(dd, dict):
                    dets_out.append(dd)
        return dets_out

    def _collect_dets_from_meta(self, meta):
        if not meta or "extras" not in meta:
            return []

        dets_out = []
        for ex in (meta.get("extras") or []):
            ex_plain = self._to_plain(ex)
            if isinstance(ex_plain, list):
                for item in ex_plain:
                    if isinstance(item, dict):
                        dets_out.append(item)
            elif isinstance(ex_plain, dict):
                for k in ("tracks", "detections", "raw_detections", "final_detections"):
                    vv = ex_plain.get(k)
                    if isinstance(vv, list):
                        for item in vv:
                            if isinstance(item, dict):
                                dets_out.append(item)
        return dets_out

    # ---------------------------------------------------------
    # API / MJPEG output
    # ---------------------------------------------------------
    def _draw_label_bar(self, img, text: str):
        if img is None:
            return img
        out = img.copy()
        try:
            cv2.rectangle(out, (5, 5), (260, 35), (0, 0, 0), thickness=-1)
            cv2.putText(
                out,
                str(text),
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass
        return out

    def _fit_with_padding(self, img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Resize while preserving aspect ratio, then center-pad into target cell.
        Never stretch image.
        """
        if img is None:
            return img

        h, w = img.shape[:2]
        if h <= 0 or w <= 0 or target_w <= 0 or target_h <= 0:
            return img

        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _build_fisheye_grid_preserve_aspect(self, prepared_views: list) -> np.ndarray | None:
        try:
            if not prepared_views:
                return None

            views_sorted = sorted(prepared_views, key=lambda x: int(x.get("view_id", 0) or 0))
            if len(views_sorted) < 4:
                return None

            imgs = []
            names = []

            for v in views_sorted[:4]:
                img = v.get("image")
                if not isinstance(img, np.ndarray):
                    return None
                imgs.append(img)
                names.append(str(v.get("name") or f"view_{int(v.get('view_id', 0) or 0)}"))

            # use actual subview size ratio, not hardcoded portrait cells
            cell_h = min(img.shape[0] for img in imgs)
            cell_w = min(img.shape[1] for img in imgs)

            fitted = []
            for img, name in zip(imgs, names):
                x = self._fit_with_padding(img, cell_w, cell_h)
                x = self._draw_label_bar(x, name)
                fitted.append(x)

            top = np.hstack((fitted[0], fitted[1]))
            bottom = np.hstack((fitted[2], fitted[3]))
            grid = np.vstack((top, bottom))
            return grid

        except Exception:
            return None
        
    def _normalize_fisheye_view_640x480(img_bgr):
        if img_bgr is None:
            return None
        try:
            h, w = img_bgr.shape[:2]
            if (w, h) == (640, 480):
                return img_bgr
            return cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_AREA)
        except Exception:
            return img_bgr
        
    def pull_live_tile_jpg(self, group: str = "0") -> bytes | None:
        """
        Live-view tile output.
        - fisheye: A/B -> 2x2 grid with overlay
        - normal: single view with overlay
        """
        try:
            g = (group or "0").upper().strip()

            if self.is_fisheye:
                if g not in ("A", "B"):
                    g = "A"
                return self.pull_group_grid_jpg(g, draw_roi_overlay=True)

            return self.pull_single_view_jpg(0, draw_roi_overlay=True)
        except Exception:
            return None

    def pull_single_view_jpg(self, view_idx_global: int, draw_roi_overlay: bool | None = None) -> bytes | None:
        self._refresh_runtime_configs_if_needed()

        try:
            vi = int(view_idx_global)
        except Exception:
            return None

        gi = 0 if (not self.is_fisheye or vi < 4) else 1

        with self._last_bundle_lock:
            ts_views = self._last_bundle_by_gi.get(int(gi))
            _meta = self._last_meta_by_gi.get(int(gi)) or {}

        if not ts_views:
            return None

        _, views = ts_views
        if not views:
            return None

        wanted_ids = [vi, vi - 4] if (self.is_fisheye and gi == 1) else [vi]

        chosen = None
        for v in views:
            if not isinstance(v, dict):
                continue
            try:
                vid = int(v.get("view_id", -1))
            except Exception:
                continue
            if vid in wanted_ids:
                chosen = v
                break

        if not chosen:
            return None

        img = chosen.get("image")
        if not isinstance(img, np.ndarray):
            return None

        img2 = img.copy()

        if self.is_fisheye:
            view_name = str(chosen.get("name") or "").strip()
            if not view_name:
                names = self._fisheye_expected_names(gi)
                local = (vi - 4) if (gi == 1 and vi >= 4) else vi
                local = max(0, min(3, local))
                view_name = names[local] if local < len(names) else f"view_{local}"
        else:
            # use actual view name if available
            view_name = str(chosen.get("name") or "").strip()
            if not view_name:
                try:
                    vid = int(chosen.get("view_id", 0))
                except Exception:
                    vid = 0
                view_name = f"view_{vid}"
        use_roi = (False if draw_roi_overlay is None else bool(draw_roi_overlay)) and bool(self._detection_enabled)

        if use_roi and (not self.is_fisheye):
            polys = self._current_normal_polys()
            img2 = self._draw_polygon_list(img2, polys, color=(0, 255, 0), thickness=2)

        if not self._detection_enabled and self.is_fisheye:
            try:
                cv2.rectangle(img2, (5, 5), (250, 35), (0, 0, 0), thickness=-1)
                cv2.putText(
                    img2,
                    view_name,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

        try:
            ok, buf = cv2.imencode(".jpg", img2, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def pull_group_grid_jpg(self, group: str, draw_roi_overlay: bool | None = None) -> bytes | None:
        self._refresh_runtime_configs_if_needed()

        group = (group or "A").upper().strip()
        gi = 0 if (not self.is_fisheye or group == "A") else 1

        with self._last_bundle_lock:
            ts_views = self._last_bundle_by_gi.get(int(gi))

        if not ts_views:
            return None

        _, views = ts_views
        if not views or not isinstance(views, list):
            return None

        use_roi = False if draw_roi_overlay is None else bool(draw_roi_overlay)

        prepared = []
        for v in sorted(views, key=lambda x: x.get("view_id", 0)):
            if not isinstance(v, dict):
                continue

            img = v.get("image")
            if not isinstance(img, np.ndarray):
                continue

            img2 = img.copy()
            view_name = str(v.get("name") or "").strip()

            if use_roi and (not self.is_fisheye):
                polys = self._current_normal_polys()
                img2 = self._draw_polygon_list(img2, polys, color=(0, 255, 0), thickness=2)

            prepared.append({
                "view_id": int(v.get("view_id", 0) or 0),
                "name": view_name or f"view_{int(v.get('view_id', 0) or 0)}",
                "image": img2,
            })

        if not prepared:
            return None

        try:
            if self.is_fisheye:
                grid = self._build_fisheye_grid_preserve_aspect(prepared)
            else:
                grid = lf.build_views_grid([
                    {**x, "zones": []} for x in prepared
                ])

            if grid is None:
                return None

            ok, buf = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def pull_single_view_jpg_clean(self, view_idx_global: int) -> bytes | None:
        """
        Dashboard-safe single-view JPG.
        Priority:
          1) fisheye/raw-frame rebuild
          2) cached latest processed view fallback
        Never draws ROI overlay.
        """
        self._refresh_runtime_configs_if_needed()

        try:
            vi = int(view_idx_global)
        except Exception:
            vi = 0

        # -------------------------------------------------
        # FISHEYE CLEAN PATH
        # -------------------------------------------------
        if self.is_fisheye and self.preprocessor is not None:
            gi = 0 if vi < 4 else 1
            local = vi if gi == 0 else (vi - 4)
            local = max(0, min(3, local))

            # 1) Try latest raw frame rebuild first
            raw_fr = self._get_latest_raw_frame_copy()
            if raw_fr is not None:
                try:
                    allowed_names = self._fisheye_expected_names(gi)

                    try:
                        views = self.preprocessor.get_views(raw_fr, allowed_names=allowed_names)
                    except TypeError:
                        views = self.preprocessor.get_views(raw_fr)

                    if views:
                        wanted_name = allowed_names[local] if local < len(allowed_names) else None
                        chosen = None

                        if wanted_name:
                            for v in views:
                                if not isinstance(v, dict):
                                    continue
                                name = str(v.get("name") or "").strip()
                                if name == wanted_name:
                                    chosen = v
                                    break

                        if chosen is None:
                            filtered = []
                            wanted = set(allowed_names)
                            for v in views:
                                if not isinstance(v, dict):
                                    continue
                                name = str(v.get("name") or "").strip()
                                if name and name in wanted:
                                    filtered.append(v)

                            filtered = sorted(filtered, key=lambda x: int(x.get("view_id", 0) or 0))
                            if 0 <= local < len(filtered):
                                chosen = filtered[local]

                        if chosen is not None:
                            img = chosen.get("image")
                            if isinstance(img, np.ndarray):
                                return self._encode_jpg(img.copy(), quality=75)
                except Exception:
                    pass

            # 2) Fallback to cached processed views
            cached_views = self._get_cached_views_for_group(gi)
            if cached_views:
                wanted_ids = [vi, vi - 4] if gi == 1 else [vi]
                for v in cached_views:
                    if not isinstance(v, dict):
                        continue
                    try:
                        vid = int(v.get("view_id", -1))
                    except Exception:
                        continue
                    if vid in wanted_ids:
                        raw = v.get("raw_view")
                        img = raw if isinstance(raw, np.ndarray) else v.get("image")
                        if isinstance(img, np.ndarray):
                            return self._encode_jpg(img.copy(), quality=75)

            return None

        # -------------------------------------------------
        # NORMAL CLEAN PATH
        # -------------------------------------------------
        cached_views = self._get_cached_views_for_group(0)
        if not cached_views:
            return None

        chosen = None
        for v in cached_views:
            if not isinstance(v, dict):
                continue
            try:
                vid = int(v.get("view_id", -1))
            except Exception:
                continue
            if vid == vi:
                chosen = v
                break

        if chosen is None and cached_views:
            chosen = cached_views[0]

        if not chosen:
            return None

        raw = chosen.get("raw_view")
        img = raw if isinstance(raw, np.ndarray) else chosen.get("image")
        if not isinstance(img, np.ndarray):
            return None

        return self._encode_jpg(img.copy(), quality=75)

    def pull_group_grid_jpg_clean(self, group: str) -> bytes | None:
        """
        Dashboard-safe fisheye 2x2 grid JPG.
        Priority:
          1) rebuild from latest raw frame
          2) fallback to cached processed views
        Never draws ROI overlay.
        """
        self._refresh_runtime_configs_if_needed()

        group = (group or "A").upper().strip()
        gi = 0 if group == "A" else 1

        if not self.is_fisheye:
            # normal cameras do not have A/B group grid
            return self.pull_single_view_jpg_clean(0)

        # -------------------------------------------------
        # 1) Try latest raw frame rebuild
        # -------------------------------------------------
        if self.preprocessor is not None:
            raw_fr = self._get_latest_raw_frame_copy()
            if raw_fr is not None:
                try:
                    allowed_names = self._fisheye_expected_names(gi)

                    try:
                        views = self.preprocessor.get_views(raw_fr, allowed_names=allowed_names)
                    except TypeError:
                        views = self.preprocessor.get_views(raw_fr)

                    if views:
                        wanted = set(allowed_names)
                        prepared = []

                        for v in views:
                            if not isinstance(v, dict):
                                continue

                            img = v.get("image")
                            if not isinstance(img, np.ndarray):
                                continue

                            view_name = str(v.get("name") or "").strip()
                            if view_name and view_name not in wanted:
                                continue

                            prepared.append({
                                "view_id": int(v.get("view_id", 0) or 0),
                                "name": view_name or f"view_{int(v.get('view_id', 0) or 0)}",
                                "image": img.copy(),
                            })

                        if prepared:
                            grid = self._build_fisheye_grid_preserve_aspect(prepared)
                            if isinstance(grid, np.ndarray):
                                return self._encode_jpg(grid, quality=70)
                except Exception:
                    pass

        # -------------------------------------------------
        # 2) Fallback to cached processed views
        # -------------------------------------------------
        cached_views = self._get_cached_views_for_group(gi)
        if not cached_views:
            return None

        prepared = []
        for v in sorted(cached_views, key=lambda x: x.get("view_id", 0)):
            if not isinstance(v, dict):
                continue

            raw = v.get("raw_view")
            img = raw if isinstance(raw, np.ndarray) else v.get("image")
            if not isinstance(img, np.ndarray):
                continue

            prepared.append({
                "view_id": int(v.get("view_id", 0) or 0),
                "name": str(v.get("name") or f"view_{int(v.get('view_id', 0) or 0)}"),
                "image": img.copy(),
            })

        if not prepared:
            return None

        grid = self._build_fisheye_grid_preserve_aspect(prepared)
        if not isinstance(grid, np.ndarray):
            return None

        return self._encode_jpg(grid, quality=70)

    def pull_dashboard_jpg(self, target: str) -> bytes | None:
        """
        Unified dashboard-safe output.
        - fisheye:
            A/B -> 2x2 group grid
            0   -> first single view
        - normal:
            always single clean view
        """
        try:
            t = str(target or "A").upper().strip()

            if self.is_fisheye:
                if t in ("A", "B"):
                    return self.pull_group_grid_jpg_clean(t)
                return self.pull_single_view_jpg_clean(0)

            # normal camera
            return self.pull_single_view_jpg_clean(0)

        except Exception:
            return None

    def pull_latest_for_api(self):
        self._refresh_runtime_configs_if_needed()

        with self._last_bundle_lock:
            cache = dict(self._last_bundle_by_gi)
            meta_cache = dict(self._last_meta_by_gi)

        if not cache:
            return []

        payload = []
        for gi, ts_views in cache.items():
            if not ts_views:
                continue

            _, views = ts_views
            if not views:
                continue

            name = "Group A" if gi == 0 else "Group B"
            clean_views = [{k: v[k] for k in v if k != "image"} for v in views if isinstance(v, dict)]

            dets = []
            if self._detection_enabled and self.runtime_enabled:
                dets += self._collect_dets_from_views(views)
                dets += self._collect_dets_from_meta(meta_cache.get(gi))
                enabled = getattr(self, "runtime_enabled_classes", None)
                if isinstance(enabled, dict):
                    dets = [
                        d for d in dets
                        if enabled.get(
                            (d.get("label") or d.get("class_name") or d.get("name") or "").strip().lower(),
                            True,
                        )
                    ]

            payload.append({
                "name": name,
                "views": clean_views,
                "dets": dets,
                "detections": dets,
                "gi": int(gi),
                "detection_enabled": bool(self._detection_enabled),
            })

        return payload

    def pull_lost_items_for_api(self):
        try:
            if self.lost_manager is None:
                return []

            items = self._to_plain(self.lost_manager.get_active_lost_items() or []) or []

            # ✅ filter: only keep items with valid image path
            filtered = []
            for it in items:
                if not isinstance(it, dict):
                    continue

                img = it.get("snapshot_path") or it.get("snapshot") or it.get("image_path")

                if img:   # only keep if exists (not None / not empty)
                    filtered.append(it)

            return filtered

        except Exception:
            return []


# ---------------------------------------------------------
# ONE DB file for all camera ROI configs
# ---------------------------------------------------------
MULTI_ROI_FILE = r"D:\DrTew\SecureWatch by QingYing JinXuan\SecureWatch\lostfound_backend\multivideo_roi_config.json"


def _load_multi_roi_db(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {"cameras": {}}


def _save_multi_roi_db(path: str, db: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def _get_cam_roi(db: dict, cam_id: str) -> dict:
    return (db.get("cameras", {}) or {}).get(cam_id, {}) or {}


def _set_cam_roi(db: dict, cam_id: str, roi_data: dict):
    db.setdefault("cameras", {})
    db["cameras"][cam_id] = roi_data


def _extract_roi_from_config_json(config_path: str) -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
        return {
            "bounding_polygons": cfg.get("bounding_polygons", []) or [],
            "fisheye_polygons": cfg.get("fisheye_polygons", {}) or {},
        }
    except Exception:
        return {"bounding_polygons": [], "fisheye_polygons": {}}


def _read_existing_cam_roi(cam_roi_path: str) -> dict:
    try:
        if os.path.exists(cam_roi_path):
            with open(cam_roi_path, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            return {
                "bounding_polygons": cfg.get("bounding_polygons", []) or [],
                "fisheye_polygons": cfg.get("fisheye_polygons", {}) or {},
            }
    except Exception:
        pass
    return {"bounding_polygons": [], "fisheye_polygons": {}}


def _write_cam_roi_config_file(cam_roi_path: str, roi_data: dict):
    os.makedirs(os.path.dirname(cam_roi_path) or ".", exist_ok=True)
    out = {
        "bounding_polygons": roi_data.get("bounding_polygons", []) or [],
        "fisheye_polygons": roi_data.get("fisheye_polygons", {}) or {},
    }
    with open(cam_roi_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def setup_roi_for_camera(cam_id: str, video_path: str, cam_roi_path: str):
    db = _load_multi_roi_db(MULTI_ROI_FILE)
    existing_db = _get_cam_roi(db, cam_id)
    existing_file = _read_existing_cam_roi(cam_roi_path)

    existing = existing_file
    if not ((existing.get("bounding_polygons", []) or []) or (existing.get("fisheye_polygons", {}) or {})):
        existing = existing_db

    lf._step("PHASE 0", f"{cam_id} ROI setup (draw/reuse/skip)")
    lf._step("PHASE 0", "ROI options: d=draw | Enter=reuse | x=skip")

    try:
        choice = input("").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice == "x":
        roi_data = {"bounding_polygons": [], "fisheye_polygons": {}}
        _set_cam_roi(db, cam_id, roi_data)
        _save_multi_roi_db(MULTI_ROI_FILE, db)
        _write_cam_roi_config_file(cam_roi_path, roi_data)
        lf._kv("PHASE 0", cam=cam_id, roi_mode="SKIP", roi_config=cam_roi_path)
        return

    if choice == "d":
        try:
            cv2.destroyAllWindows()
            time.sleep(0.2)
            lf._step("UI", f"{cam_id} ROI editor open (click window once before keys)")

            tmp_cfg = os.path.join(
                os.path.dirname(os.path.abspath(cam_roi_path)),
                f"roi_editor_tmp_{cam_id}.json"
            )

            try:
                if os.path.exists(tmp_cfg):
                    os.remove(tmp_cfg)
            except Exception:
                pass

            is_fisheye_src = False
            try:
                v = lf.detect_video_type(video_path, max_samples=3, max_scan_seconds=2.0)
                is_fisheye_src = "fish" in str(v).lower()
            except Exception:
                pass

            if is_fisheye_src:
                lf._step("UI", f"{cam_id} → fisheye ROI editor")
                _launch_fisheye_roi_editor_rtsp_safe(video_path, config_out_path=tmp_cfg)
            else:
                lf._step("UI", f"{cam_id} → normal ROI editor")
                _launch_roi_editor_rtsp_safe(video_path, config_out_path=tmp_cfg)

            roi_data = _extract_roi_from_config_json(tmp_cfg)

        except Exception as e:
            print(f"[WARN] ROI editor failed for {cam_id}: {e!r}. Using empty ROI.")
            roi_data = {"bounding_polygons": [], "fisheye_polygons": {}}

        _set_cam_roi(db, cam_id, roi_data)
        _save_multi_roi_db(MULTI_ROI_FILE, db)
        _write_cam_roi_config_file(cam_roi_path, roi_data)
        lf._kv("PHASE 0", cam=cam_id, roi_mode="DRAW", roi_db=MULTI_ROI_FILE, roi_config=cam_roi_path)
        return

    has_existing_roi = bool(
        (existing.get("bounding_polygons", []) or []) or
        (existing.get("fisheye_polygons", {}) or {})
    )

    if not has_existing_roi:
        roi_data = {"bounding_polygons": [], "fisheye_polygons": {}}
        _set_cam_roi(db, cam_id, roi_data)
        _save_multi_roi_db(MULTI_ROI_FILE, db)
        _write_cam_roi_config_file(cam_roi_path, roi_data)
        lf._kv("PHASE 0", cam=cam_id, roi_mode="REUSE_EMPTY", roi_config=cam_roi_path)
        return

    lf._kv(
        "PHASE 0",
        cam=cam_id,
        reuse_normal_roi_count=len(existing.get("bounding_polygons", []) or []),
        reuse_fisheye_roi_views=len(existing.get("fisheye_polygons", {}) or {}),
    )
    _write_cam_roi_config_file(cam_roi_path, existing)
    lf._kv("PHASE 0", cam=cam_id, roi_mode="REUSE", roi_db=MULTI_ROI_FILE, roi_config=cam_roi_path)


def main():
    lf.configure_logging(debug=False, mute_info=True)
    lf._banner("LOST & FOUND MULTI-VIDEO PIPELINE START")
    lf._step("PHASE 0", "System init + ROI setup")

    try:
        lf._step("SYSTEM", "Memory cleanup...")
        lf.memory_manager.comprehensive_cleanup()
    except Exception:
        pass

    try:
        device = lf.DeviceManager.get_device()
    except Exception:
        device = None

    lf._step("PHASE 3", "Detector + threads setup")
    detector = lf.YoloDetector(
        items_weights=lf.WEIGHTS_ITEMS,
        coco_weights=lf.WEIGHTS_PERSON,
        device=device,
        item_capture_conf=0.08,
        coco_capture_conf=0.10,
        person_conf=0.30,
        track_win=10,
        confirm_k=3,
        hold_frames=15,
    )
    lf._kv("PHASE 3", detector="YOLO", item_conf=0.08, coco_conf=0.10, person_conf=0.30)

    sources = {
        "cam1": "rtsp://admin:cctv%402268@10.123.41.192:554/Streaming/Channels/102/",
        "cam2": "rtsp://admin:cctv%402268@10.123.41.193:554/Streaming/Channels/102/",
        "cam3": "rtsp://admin:cctv%402268@10.123.41.194:554/Streaming/Channels/102/",
        "cam4": "rtsp://admin:cctv%402268@10.123.41.195:554/Streaming/Channels/102/",
    }

    cam_roi_paths: Dict[str, str] = {}
    for cam_id in sources.keys():
        cam_out_dir = os.path.join(LAF_BASE, cam_id)
        os.makedirs(cam_out_dir, exist_ok=True)
        cam_roi_path = os.path.join(cam_out_dir, "roi_config.json")
        cam_roi_paths[cam_id] = cam_roi_path
        setup_roi_for_camera(cam_id, sources[cam_id], cam_roi_path)

    lf._step("PHASE 4", "Starting runtime pipeline")
    pipelines: Dict[str, VideoPipeline] = {}
    for cam_id, src in sources.items():
        cfg = PipelineConfig(
            camera_id=cam_id,
            src=src,
            roi_config_path=cam_roi_paths[cam_id],
            fisheye_config_path=r"D:\DrTew\SecureWatch by QingYing JinXuan\SecureWatch\lostfound_backend\backend\outputs\lost_and_found\fisheye_view_configs.json",
            show_ui=True,
            source_kind="RTSP",
            desired_fps_fisheye=4.0,
            desired_fps_normal=4.0,
            base_frame_skip_fisheye_rtsp=0,
            base_frame_skip_normal_rtsp=0,
            base_frame_skip_fisheye_file=0,
            base_frame_skip_normal_file=0,
        )
        p = VideoPipeline(cfg, detector)
        if p.start():
            pipelines[cam_id] = p

    if not pipelines:
        lf._step("ERROR", "No pipelines started.")
        lf._banner("PIPELINE EXIT")
        return

    lf._kv("UI", window="Lost&Found - Multi-View Preview [MULTI]", display_fps=25, scale=0.8)
    lf._step("UI", "Controls: Q=Quit | T=Toggle group | D=Toggle detection | 1-4 + arrows + [] + S = Tune/Save")

    try:
        while True:
            any_alive = False
            for cam_id, p in list(pipelines.items()):
                p.tick_ui()
                if p.is_finished_local_video():
                    p.stop()
                    p.join()
                    try:
                        cv2.destroyWindow(p.win_name)
                    except Exception:
                        pass
                    pipelines.pop(cam_id, None)
                    continue
                any_alive = True
            if not any_alive:
                break

            key = cv2.waitKeyEx(1)
            if key in (ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if key in (ord("t"), ord("T")):
                for cam_id, p in pipelines.items():
                    if p.is_fisheye:
                        p.active_views.toggle()
                        drain_queue(p.frame_queue)
                        drain_queue(p.bundle_job_queue)
                        drain_queue(p.det_out_queue)
                        drain_queue(p.out_queue)
                        lf._step("UI", f"{cam_id} toggled fisheye group")
            if key in (ord("d"), ord("D")):
                for cam_id, p in pipelines.items():
                    if p.cfg.enable_detection:
                        p.set_detection_enabled(not p._detection_enabled)
            if key != -1:
                for cam_id, p in pipelines.items():
                    if p.is_fisheye:
                        try:
                            p._tune_state = lf.handle_realtime_tuning(
                                key, p.preprocessor, p._last_out_views, p._tune_state
                            )
                        except Exception:
                            pass
            time.sleep(0.001)

    except KeyboardInterrupt:
        lf._step("UI", "KeyboardInterrupt -> stopping")
    finally:
        lf._step("SYSTEM", "Stopping threads + cleanup")
        for p in list(pipelines.values()):
            try:
                p.stop()
            except Exception:
                pass
        for p in list(pipelines.values()):
            try:
                p.join()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        lf._banner("PIPELINE EXIT")


if __name__ == "__main__":
    main()
