"""High-performance vision detector and trigger engine for dodge/parry cues."""

from dataclasses import dataclass
from enum import Enum
import queue
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np

from .input_handler import play_audio_cue


class ROIState(Enum):
    IDLE = "IDLE"
    RED_WAITING = "RED_WAITING"
    TRIGGER_GREEN = "TRIGGER_GREEN"


@dataclass
class ROIDetectionResult:
    action_id: str
    key: str
    state: ROIState
    green_count: int
    green_density: float
    red_density: float
    triggered: bool
    bgr_crop: Optional[np.ndarray] = None


class ReactionDetector:
    """Detects neon green action prompts on 5 sub-ROIs with minimal latency."""

    def __init__(
        self,
        config: Dict[str, Any],
        on_trigger_callback: Optional[Callable[[str, str], None]] = None,
        event_queue: Optional[queue.Queue] = None,
    ):
        self.config = config
        self.on_trigger = on_trigger_callback
        self.event_queue = event_queue
        self.armed = False
        self.suppress_detection = False

        # State tracking per action (up, left, down, right, center)
        self.action_states: Dict[str, ROIState] = {}
        self.last_red_time: Dict[str, float] = {}
        self.last_trigger_time: Dict[str, float] = {}

        self.last_results: Dict[str, ROIDetectionResult] = {}
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.frame_count: int = 0
        self.last_fps_calc_time: float = time.perf_counter()

        # Debug visualization buffers
        self.latest_bgr_frame: Optional[np.ndarray] = None
        self.latest_green_mask: Optional[np.ndarray] = None
        self.latest_sub_counts: Dict[str, int] = {
            "up": 0, "left": 0, "down": 0, "right": 0, "center": 0
        }

        self.last_global_trigger_time: float = 0.0
        self.last_reaction_delay_ms: float = 0.0

        self._update_thresholds()
        self._init_states()

    def _update_thresholds(self) -> None:
        th = self.config.get("color_thresholds", {})

        # Calibrated neon green HSV: [35, 70, 120] to [85, 255, 255]
        self.green_lower = np.array(th.get("green_lower", [35, 70, 120]), dtype=np.uint8)
        self.green_upper = np.array(th.get("green_upper", [85, 255, 255]), dtype=np.uint8)

        # Minimum pixel threshold (default: 80 pixels)
        self.min_green_pixels = int(th.get("min_green_pixels", 80))
        self.green_min_density = float(th.get("green_min_density", 0.08))

        # Red baseline (optional)
        self.red_lower_1 = np.array(th.get("red_lower_1", [0, 100, 100]), dtype=np.uint8)
        self.red_upper_1 = np.array(th.get("red_upper_1", [10, 255, 255]), dtype=np.uint8)
        self.red_lower_2 = np.array(th.get("red_lower_2", [170, 100, 100]), dtype=np.uint8)
        self.red_upper_2 = np.array(th.get("red_upper_2", [180, 255, 255]), dtype=np.uint8)
        self.red_min_density = float(th.get("red_min_density", 0.08))

        # Strict transition (require red precursor) vs direct green count trigger
        self.strict_transition = bool(th.get("strict_transition", False))

        timing = self.config.get("timing", {})
        self.cooldown_s = float(timing.get("cooldown_ms", 350)) / 1000.0

        # Pre-press reaction delay & humanizer jitter
        self.pre_press_delay_ms = float(
            self.config.get("pre_press_delay_ms", timing.get("pre_press_delay_ms", 45))
        )
        self.delay_jitter_ms = float(
            self.config.get("delay_jitter_ms", timing.get("delay_jitter_ms", 10))
        )
        self.key_press_duration_ms = float(
            self.config.get("key_press_duration_ms", timing.get("key_press_duration_ms", 40))
        )
        self.audio_cues_enabled = bool(self.config.get("audio_cues_enabled", True))
        self.pre_press_delay_s = self.pre_press_delay_ms / 1000.0

        # Priority resolution order (highest to lowest)
        self.priority_order = [
            str(k).upper() for k in self.config.get("priority_order", ["A", "F", "D", "S", "W"])
        ]

    def _init_states(self) -> None:
        self.last_global_trigger_time = 0.0
        actions = ["up", "left", "down", "right", "center"]
        for act in actions:
            self.action_states[act] = ROIState.IDLE
            self.last_red_time[act] = 0.0
            self.last_trigger_time[act] = 0.0

    def get_sniper_box(self) -> Dict[str, int]:
        sbox = self.config.get("sniper_box", {"x": 860, "y": 440, "width": 200, "height": 200})
        return {
            "left": int(sbox.get("x", 860)),
            "top": int(sbox.get("y", 440)),
            "width": int(sbox.get("width", 200)),
            "height": int(sbox.get("height", 200)),
        }

    def get_relative_sub_rois(self, box_w: int, box_h: int) -> Dict[str, Tuple[int, int, int, int]]:
        """Return 5 relative sub-zones, using calibrated config if available."""
        sub_cfg = self.config.get("sub_rois")
        if sub_cfg and isinstance(sub_cfg, dict):
            res = {}
            for act in ["up", "down", "left", "right", "center"]:
                if act in sub_cfg:
                    c = sub_cfg[act]
                    rx = max(0, min(box_w, int(c.get("rx", 0))))
                    ry = max(0, min(box_h, int(c.get("ry", 0))))
                    rw = max(10, min(box_w - rx, int(c.get("rw", 40))))
                    rh = max(10, min(box_h - ry, int(c.get("rh", 40))))
                    res[act] = (rx, ry, rx + rw, ry + rh)
            if len(res) == 5:
                return res

        # Default geometric fallback
        x_left_end = int(box_w * 0.30)
        x_right_start = int(box_w * 0.70)
        y_top_end = int(box_h * 0.30)
        y_bottom_start = int(box_h * 0.70)

        return {
            "up": (x_left_end, 0, x_right_start, y_top_end),
            "down": (x_left_end, y_bottom_start, x_right_start, box_h),
            "left": (0, y_top_end, x_left_end, y_bottom_start),
            "right": (x_right_start, y_top_end, box_w, y_bottom_start),
            "center": (x_left_end, y_top_end, x_right_start, y_bottom_start),
        }

    def _resolve_key(self, action_id: str) -> str:
        keybinds = self.config.get("keybinds", {}).get("actions", {})
        if action_id in keybinds:
            return keybinds[action_id]
        if "rois" in self.config and action_id in self.config["rois"]:
            return self.config["rois"][action_id].get("key", "w" if action_id == "up" else "f")
        default_keys = {"up": "w", "left": "a", "down": "s", "right": "d", "center": "f"}
        return default_keys.get(action_id, "f")

    def get_priority_rank(self, action_id: str, key: str) -> int:
        """Return priority index (0 = highest priority) based on key or action."""
        k = key.upper().strip()
        if k in self.priority_order:
            return self.priority_order.index(k)
        action_default_map = {
            "left": "A",
            "center": "F",
            "right": "D",
            "down": "S",
            "up": "W",
        }
        def_k = action_default_map.get(action_id, "").upper()
        if def_k in self.priority_order:
            return self.priority_order.index(def_k)
        return 999

    def analyze_crop(
        self,
        bgr_crop: np.ndarray,
        action_id: str,
        now: Optional[float] = None,
        can_trigger: bool = True,
        candidate_keys: Optional[List[str]] = None,
    ) -> ROIDetectionResult:
        """Analyze single sub-ROI crop for green flash."""
        if now is None:
            now = time.perf_counter()

        key = self._resolve_key(action_id)
        total_pixels = bgr_crop.shape[0] * bgr_crop.shape[1]
        if total_pixels == 0:
            return ROIDetectionResult(action_id, key, ROIState.IDLE, 0, 0.0, 0.0, False)

        # Convert to HSV
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

        # Neon green mask
        mask_green = cv2.inRange(hsv, self.green_lower, self.green_upper)
        green_count = int(cv2.countNonZero(mask_green))
        green_density = green_count / total_pixels
        self.latest_sub_counts[action_id] = green_count

        # Red mask
        mask_red_1 = cv2.inRange(hsv, self.red_lower_1, self.red_upper_1)
        mask_red_2 = cv2.inRange(hsv, self.red_lower_2, self.red_upper_2)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
        red_count = int(cv2.countNonZero(mask_red))
        red_density = red_count / total_pixels

        current_state = self.action_states.get(action_id, ROIState.IDLE)
        last_red = self.last_red_time.get(action_id, 0.0)
        last_trigger = self.last_trigger_time.get(action_id, 0.0)
        triggered = False

        # Diagnostic log whenever green exceeds 10% of threshold
        diag_min = max(5, int(self.min_green_pixels * 0.10))
        if green_count >= diag_min:
            print(f"[Detector Diagnostic] {action_id.upper()}: {green_count} green px ({green_density*100:.1f}%) | Target: >={self.min_green_pixels} px")

        # Check Red Baseline
        if red_density >= self.red_min_density:
            current_state = ROIState.RED_WAITING
            self.last_red_time[action_id] = now

        # Check Green Flash Trigger
        is_green_trigger = (green_count >= self.min_green_pixels) and (green_density >= self.green_min_density)
        has_red_precursor = (now - last_red) <= 1.0

        if is_green_trigger and can_trigger:
            valid_transition = (not self.strict_transition) or (
                current_state == ROIState.RED_WAITING or has_red_precursor
            )
            not_on_cooldown = (
                ((now - self.last_global_trigger_time) >= self.cooldown_s)
                and ((now - last_trigger) >= self.cooldown_s)
            )

            if valid_transition and not_on_cooldown:
                triggered = True
                current_state = ROIState.TRIGGER_GREEN
                self.last_trigger_time[action_id] = now
                self.last_global_trigger_time = now

                k_upper = key.upper()
                cands = candidate_keys or [k_upper]

                # Compute humanized reaction delay with random jitter
                jitter = random.uniform(-self.delay_jitter_ms, self.delay_jitter_ms) if self.delay_jitter_ms > 0 else 0.0
                actual_delay_ms = max(0.0, self.pre_press_delay_ms + jitter)
                self.last_reaction_delay_ms = actual_delay_ms
                actual_delay_s = actual_delay_ms / 1000.0
                hold_duration_ms = max(10.0, self.key_press_duration_ms)
                hold_duration_s = hold_duration_ms / 1000.0

                log_msg = f"[Triggered] '{k_upper}' | Candidates: {cands} | Delay: {actual_delay_ms:.1f}ms | Hold: {int(hold_duration_ms)}ms"
                print(f"{log_msg} | Green px: {green_count} ({green_density*100:.1f}%)")

                if self.armed and self.on_trigger:
                    play_audio_cue("trigger", enabled=self.audio_cues_enabled)
                    try:
                        self.on_trigger(action_id, key, actual_delay_s, hold_duration_s)
                    except TypeError:
                        self.on_trigger(action_id, key)

                if self.event_queue:
                    action_name = self.config.get("action_names", {}).get(action_id, action_id)
                    self.event_queue.put({
                        "time": time.strftime("%H:%M:%S") + f".{int((now % 1) * 1000):03d}",
                        "action": action_id,
                        "key": k_upper,
                        "name": action_name,
                        "green_px": green_count,
                        "green_pct": green_density * 100.0,
                        "delay_ms": round(actual_delay_ms, 1),
                        "hold_ms": int(hold_duration_ms),
                        "candidates": cands,
                        "log_msg": log_msg,
                    })
            elif not valid_transition:
                current_state = ROIState.IDLE
        elif red_density < self.red_min_density:
            if current_state == ROIState.TRIGGER_GREEN:
                current_state = ROIState.IDLE
            elif (now - last_red) > 1.0:
                current_state = ROIState.IDLE

        self.action_states[action_id] = current_state

        res = ROIDetectionResult(
            action_id=action_id,
            key=key,
            state=current_state,
            green_count=green_count,
            green_density=green_density,
            red_density=red_density,
            triggered=triggered,
            bgr_crop=bgr_crop,
        )
        self.last_results[action_id] = res
        return res

    # Backwards compatibility alias
    analyze_roi_buffer = analyze_crop

    def process_sniper_buffer(
        self,
        bgr_sniper_img: np.ndarray,
        now: Optional[float] = None,
    ) -> List[ROIDetectionResult]:
        """Process full sniper image, evaluate multi-target green cues with priority, and fire trigger."""
        if self.suppress_detection:
            return []

        t_start = time.perf_counter()
        if now is None:
            now = t_start

        self.latest_bgr_frame = bgr_sniper_img

        # Compute full green mask for debug view
        hsv_full = cv2.cvtColor(bgr_sniper_img, cv2.COLOR_BGR2HSV)
        self.latest_green_mask = cv2.inRange(hsv_full, self.green_lower, self.green_upper)

        # Full red mask for red baseline detection
        mask_red_1 = cv2.inRange(hsv_full, self.red_lower_1, self.red_upper_1)
        mask_red_2 = cv2.inRange(hsv_full, self.red_lower_2, self.red_upper_2)
        mask_red_full = cv2.bitwise_or(mask_red_1, mask_red_2)

        h, w = bgr_sniper_img.shape[:2]
        sub_rois = self.get_relative_sub_rois(w, h)

        # Step 1: Pre-calculate counts, densities, and crops for all 5 sub-zones
        zone_data = {}
        for action_id, (x1, y1, x2, y2) in sub_rois.items():
            cx1 = max(0, min(w, x1))
            cy1 = max(0, min(h, y1))
            cx2 = max(0, min(w, x2))
            cy2 = max(0, min(h, y2))

            if cx2 > cx1 and cy2 > cy1:
                crop = bgr_sniper_img[cy1:cy2, cx1:cx2]
                sub_mask = self.latest_green_mask[cy1:cy2, cx1:cx2]
                cnt = int(cv2.countNonZero(sub_mask))
                tot_px = crop.shape[0] * crop.shape[1]
                green_density = cnt / tot_px if tot_px > 0 else 0.0

                sub_red_mask = mask_red_full[cy1:cy2, cx1:cx2]
                red_cnt = int(cv2.countNonZero(sub_red_mask))
                red_density = red_cnt / tot_px if tot_px > 0 else 0.0
            else:
                crop = np.zeros((10, 10, 3), dtype=np.uint8)
                cnt = 0
                green_density = 0.0
                red_density = 0.0

            zone_data[action_id] = {
                "crop": crop,
                "green_cnt": cnt,
                "green_density": green_density,
                "red_density": red_density,
            }

        # Step 2: Identify all zones exceeding the green pixel threshold
        not_on_global_cooldown = (now - self.last_global_trigger_time) >= self.cooldown_s
        qualifying_actions = []
        if not_on_global_cooldown:
            for action_id, z in zone_data.items():
                is_green = (z["green_cnt"] >= self.min_green_pixels) and (z["green_density"] >= self.green_min_density)
                last_red = self.last_red_time.get(action_id, 0.0)
                curr_st = self.action_states.get(action_id, ROIState.IDLE)
                has_red = (now - last_red) <= 1.0
                valid_transition = (not self.strict_transition) or (curr_st == ROIState.RED_WAITING or has_red)
                not_on_act_cooldown = (now - self.last_trigger_time.get(action_id, 0.0)) >= self.cooldown_s

                if is_green and valid_transition and not_on_act_cooldown:
                    qualifying_actions.append(action_id)

        # Step 3: Prioritized Multi-Detection Resolution
        # Sort qualifying candidates strictly by priority index (e.g. A > F > D > S > W)
        winner_action = None
        candidate_keys = []
        if qualifying_actions:
            qualifying_actions.sort(
                key=lambda act: (self.get_priority_rank(act, self._resolve_key(act)), -zone_data[act]["green_cnt"])
            )
            winner_action = qualifying_actions[0]
            candidate_keys = [self._resolve_key(act).upper() for act in qualifying_actions]

        # Step 4: Analyze each sub-ROI, allowing only the winning candidate to trigger
        results = []
        for action_id, z in zone_data.items():
            can_trig = (action_id == winner_action)
            res = self.analyze_crop(
                z["crop"],
                action_id,
                now=now,
                can_trigger=can_trig,
                candidate_keys=candidate_keys if can_trig else None,
            )
            results.append(res)

        self._update_fps_and_latency(time.perf_counter() - t_start, now)
        return results

    def _update_fps_and_latency(self, frame_latency: float, now: float) -> None:
        self.frame_count += 1
        self.latency_ms = frame_latency * 1000.0
        delta = now - self.last_fps_calc_time
        if delta >= 0.5:
            self.fps = self.frame_count / delta
            self.frame_count = 0
            self.last_fps_calc_time = now

    def capture_and_process_once(self, sct: mss.base.MSSBase) -> List[ROIDetectionResult]:
        """Capture sniper box from screen and process all 5 zones."""
        sbox = self.get_sniper_box()
        raw = sct.grab(sbox)
        frame_bgr = np.ascontiguousarray(np.array(raw)[:, :, :3])
        return self.process_sniper_buffer(frame_bgr, now=time.perf_counter())
