"""Unit tests for deterministic priority resolution (A > F > D > S > W) and pre-press delay."""

import copy
import queue
import time
import unittest
import numpy as np

from dodge_parry_macro.detector import ReactionDetector, ROIState
from dodge_parry_macro.input_handler import DirectInputHandler


TEST_CONFIG = {
    "sniper_box": {
        "x": 860,
        "y": 440,
        "width": 200,
        "height": 200,
    },
    "menu_always_on_top": True,
    "always_on_top": True,
    "pre_press_delay_ms": 60,
    "delay_jitter_ms": 10,
    "audio_cues_enabled": False,
    "roi_locked": False,
    "priority_order": ["A", "F", "D", "S", "W"],
    "sub_rois": {
        "up": {"rx": 60, "ry": 0, "rw": 80, "rh": 60},
        "left": {"rx": 0, "ry": 60, "rw": 60, "rh": 80},
        "center": {"rx": 60, "ry": 60, "rw": 80, "rh": 80},
        "right": {"rx": 140, "ry": 60, "rw": 60, "rh": 80},
        "down": {"rx": 60, "ry": 140, "rw": 80, "rh": 60},
    },
    "keybinds": {
        "toggle_arm": "f6",
        "toggle_overlay": "f7",
        "quit": "f8",
        "toggle_lock": "f9",
        "panic": "f12",
        "actions": {
            "up": "w",
            "left": "a",
            "down": "s",
            "right": "d",
            "center": "f",
        },
    },
    "action_names": {
        "up": "Up (W)",
        "left": "Left (A)",
        "down": "Down (S)",
        "right": "Right (D)",
        "center": "Center Parry (F)",
    },
    "color_thresholds": {
        "green_lower": [35, 70, 120],
        "green_upper": [85, 255, 255],
        "min_green_pixels": 80,
        "green_min_density": 0.08,
        "red_lower_1": [0, 100, 100],
        "red_upper_1": [10, 255, 255],
        "red_lower_2": [170, 100, 100],
        "red_upper_2": [180, 255, 255],
        "red_min_density": 0.08,
        "strict_transition": False,
    },
    "timing": {
        "cooldown_ms": 300,
        "pre_press_delay_ms": 60,
        "delay_jitter_ms": 10,
        "key_press_duration_ms": 40,
        "scan_interval_ms": 1,
    },
}


def make_blank_frame(w: int = 200, h: int = 200) -> np.ndarray:
    """Create background grass frame (no green flash)."""
    np.random.seed(42)
    b = np.random.randint(15, 30, (h, w), dtype=np.uint8)
    g = np.random.randint(50, 80, (h, w), dtype=np.uint8)
    r = np.random.randint(15, 30, (h, w), dtype=np.uint8)
    return np.dstack([b, g, r])


def create_synthetic_frame(
    w: int = 200,
    h: int = 200,
    green_zones: list[str] | None = None,
    pixel_counts: dict[str, int] | None = None,
) -> np.ndarray:
    """Generate a 200x200 synthetic frame with neon green in specified sub-zones."""
    frame = make_blank_frame(w, h)
    neon_green = [20, 255, 57]  # BGR

    coords = {
        "up": (60, 0, 140, 60),
        "down": (60, 140, 140, 200),
        "left": (0, 60, 60, 140),
        "right": (140, 60, 200, 140),
        "center": (60, 60, 140, 140),
    }

    green_zones = green_zones or []
    pixel_counts = pixel_counts or {}

    for zone in green_zones:
        if zone in coords:
            x1, y1, x2, y2 = coords[zone]
            desired_px = pixel_counts.get(zone)
            if desired_px is not None:
                total_zone_area = (x2 - x1) * (y2 - y1)
                count = min(desired_px, total_zone_area)
                idx = 0
                for yy in range(y1, y2):
                    for xx in range(x1, x2):
                        if idx < count:
                            frame[yy, xx] = neon_green
                            idx += 1
            else:
                cx1 = x1 + int((x2 - x1) * 0.15)
                cx2 = x2 - int((x2 - x1) * 0.15)
                cy1 = y1 + int((y2 - y1) * 0.15)
                cy2 = y2 - int((y2 - y1) * 0.15)
                frame[cy1:cy2, cx1:cx2] = neon_green

    return frame


class TestPriorityResolution(unittest.TestCase):
    def setUp(self):
        self.triggered_events = []
        self.event_queue = queue.Queue()
        self.detector = ReactionDetector(
            config=copy.deepcopy(TEST_CONFIG),
            on_trigger_callback=lambda act, k, d, h: self.triggered_events.append((act, k, d, h)),
            event_queue=self.event_queue,
        )
        self.detector.armed = True

    def test_simultaneous_a_f_d_triggers_strictly_a(self):
        """Passing synthetic frame where A, F, and D are all green results strictly in triggering A."""
        frame = create_synthetic_frame(green_zones=["left", "center", "right"])

        results = self.detector.process_sniper_buffer(frame, now=10.0)

        # Check triggered actions
        triggered_results = [r for r in results if r.triggered]
        self.assertEqual(len(triggered_results), 1, "Expected exactly one triggered candidate!")
        winner = triggered_results[0]

        # Verify A (left) was chosen
        self.assertEqual(winner.action_id, "left")
        self.assertEqual(winner.key, "a")
        self.assertEqual(len(self.triggered_events), 1)
        self.assertEqual(self.triggered_events[0][0], "left")
        self.assertEqual(self.triggered_events[0][1], "a")

        # Verify F and D were ignored (not triggered)
        center_res = next(r for r in results if r.action_id == "center")
        right_res = next(r for r in results if r.action_id == "right")
        self.assertFalse(center_res.triggered, "F (center) should NOT have triggered!")
        self.assertFalse(right_res.triggered, "D (right) should NOT have triggered!")
        self.assertGreaterEqual(center_res.green_count, 80)
        self.assertGreaterEqual(right_res.green_count, 80)

        # Check debug event queue log message
        self.assertFalse(self.event_queue.empty())
        evt = self.event_queue.get_nowait()
        self.assertEqual(evt["action"], "left")
        self.assertEqual(evt["key"], "A")
        self.assertIn("Candidates:", evt["log_msg"])
        self.assertIn("'A'", evt["log_msg"])
        self.assertIn("'F'", evt["log_msg"])
        self.assertIn("'D'", evt["log_msg"])
        self.assertGreaterEqual(evt["delay_ms"], 50.0)
        self.assertLessEqual(evt["delay_ms"], 70.0)

    def test_priority_overrides_raw_pixel_count(self):
        """Even if D has higher pixel count than A, A must win because priority (A > D) takes precedence."""
        frame = create_synthetic_frame(
            green_zones=["left", "center", "right"],
            pixel_counts={"right": 1500, "center": 1000, "left": 500},
        )

        results = self.detector.process_sniper_buffer(frame, now=20.0)
        self.assertEqual(len(self.triggered_events), 1)
        self.assertEqual(self.triggered_events[0][0], "left")
        self.assertEqual(self.triggered_events[0][1], "a")

        # Verify pixel counts
        left_res = next(r for r in results if r.action_id == "left")
        right_res = next(r for r in results if r.action_id == "right")
        self.assertGreater(right_res.green_count, left_res.green_count)
        self.assertTrue(left_res.triggered)
        self.assertFalse(right_res.triggered)

    def test_fallback_priority_f_over_d_when_a_absent(self):
        """When A is not green, but F, D, and S are green, F must be chosen (F > D > S)."""
        frame = create_synthetic_frame(green_zones=["center", "right", "down"])

        results = self.detector.process_sniper_buffer(frame, now=30.0)
        self.assertEqual(len(self.triggered_events), 1)
        self.assertEqual(self.triggered_events[0][0], "center")
        self.assertEqual(self.triggered_events[0][1], "f")

        center_res = next(r for r in results if r.action_id == "center")
        self.assertTrue(center_res.triggered)

    def test_fallback_priority_d_over_s_and_w(self):
        """When A and F are absent, D wins over S and W (D > S > W)."""
        frame = create_synthetic_frame(green_zones=["right", "down", "up"])

        results = self.detector.process_sniper_buffer(frame, now=40.0)
        self.assertEqual(len(self.triggered_events), 1)
        self.assertEqual(self.triggered_events[0][0], "right")
        self.assertEqual(self.triggered_events[0][1], "d")

    def test_global_lockout_cooldown_suppresses_trailing_triggers(self):
        """After triggering A, trailing frames within 300ms cooldown must not fire any key."""
        frame1 = create_synthetic_frame(green_zones=["left", "center"])
        self.detector.process_sniper_buffer(frame1, now=50.0)
        self.assertEqual(len(self.triggered_events), 1)
        self.assertEqual(self.triggered_events[0][0], "left")

        # Frame 100ms later with D now turning green (inside 300ms cooldown)
        frame2 = create_synthetic_frame(green_zones=["right"])
        results2 = self.detector.process_sniper_buffer(frame2, now=50.1)
        self.assertEqual(len(self.triggered_events), 1, "Trailing trigger fired during global cooldown!")
        right_res2 = next(r for r in results2 if r.action_id == "right")
        self.assertFalse(right_res2.triggered)

        # Frame 350ms later (cooldown expired) fires normally
        frame3 = create_synthetic_frame(green_zones=["right"])
        results3 = self.detector.process_sniper_buffer(frame3, now=50.35)
        self.assertEqual(len(self.triggered_events), 2)
        self.assertEqual(self.triggered_events[1][0], "right")

    def test_calibrated_sub_rois_loading(self):
        """Verify custom sub_rois from config are loaded accurately into detector."""
        custom_cfg = copy.deepcopy(TEST_CONFIG)
        custom_cfg["sub_rois"] = {
            "up": {"rx": 50, "ry": 5, "rw": 90, "rh": 45},
            "left": {"rx": 5, "ry": 55, "rw": 45, "rh": 90},
            "center": {"rx": 65, "ry": 65, "rw": 70, "rh": 70},
            "right": {"rx": 145, "ry": 55, "rw": 45, "rh": 90},
            "down": {"rx": 50, "ry": 145, "rw": 90, "rh": 45},
        }
        det = ReactionDetector(custom_cfg)
        rois = det.get_relative_sub_rois(200, 200)
        self.assertEqual(rois["up"], (50, 5, 140, 50))
        self.assertEqual(rois["left"], (5, 55, 50, 145))
        self.assertEqual(rois["center"], (65, 65, 135, 135))
        self.assertEqual(rois["right"], (145, 55, 190, 145))
        self.assertEqual(rois["down"], (50, 145, 140, 190))

    def test_reaction_jitter_distribution(self):
        """Verify reaction delay with jitter 60ms ± 10ms stays strictly within bounds."""
        frame = create_synthetic_frame(green_zones=["left"])
        for i in range(10):
            det = ReactionDetector(
                config=copy.deepcopy(TEST_CONFIG),
                on_trigger_callback=lambda act, k, d, h: None,
            )
            det.armed = True
            det.process_sniper_buffer(frame, now=100.0 + i * 2.0)
            delay_ms = det.last_reaction_delay_ms
            # 60ms ± 10ms -> 50.0ms to 70.0ms
            self.assertGreaterEqual(delay_ms, 50.0)
            self.assertLessEqual(delay_ms, 70.0)

    def test_pre_press_delay_in_input_handler(self):
        """DirectInputHandler honors pre_press_delay_ms before triggering key_down."""
        handler = DirectInputHandler(default_press_duration_ms=20, pre_press_delay_ms=50, mock=True)

        t0 = time.perf_counter()
        handler.press_key_sync("a")
        elapsed = time.perf_counter() - t0

        # Should sleep at least 50ms pre-press delay + 20ms key press
        self.assertGreaterEqual(elapsed, 0.065)
        self.assertEqual(handler.last_pressed_key, "a")
        self.assertEqual(handler.total_presses, 1)

    def test_panic_killswitch_releases_all_keys(self):
        """Emergency failsafe release_all_keys releases active keys."""
        handler = DirectInputHandler(mock=True)
        handler.key_down("w")
        handler.key_down("a")
        self.assertIn("w", handler.active_keys)
        self.assertIn("a", handler.active_keys)

        handler.release_all_keys()
        self.assertEqual(len(handler.active_keys), 0)

    def test_audio_cue_playback_safe(self):
        """Verify play_audio_cue can be called safely without exceptions."""
        from dodge_parry_macro.input_handler import play_audio_cue
        # Should not raise exception
        play_audio_cue("trigger", enabled=True)
        play_audio_cue("arm", enabled=False)
        play_audio_cue("panic", enabled=True)

    def test_start_stop_toggle_no_thread_leak(self):
        """Rapidly toggle start/stop 20+ times without creating or accumulating threads."""
        import threading
        from dodge_parry_macro.main import MacroOrchestrator

        orch = MacroOrchestrator(mock_input=True)
        orch.start_capture_thread()
        initial_thread = orch.capture_thread
        self.assertIsNotNone(initial_thread)
        self.assertTrue(initial_thread.is_alive())

        # Toggle once to initialize any lazy persistent daemon worker (audio queue)
        orch.toggle_arm()
        thread_count_after_warmup = threading.active_count()

        # Rapidly toggle 25 times
        for _ in range(25):
            orch.toggle_arm()

        # Capture thread must be the exact same persistent thread
        self.assertIs(orch.capture_thread, initial_thread)
        self.assertTrue(orch.capture_thread.is_alive())
        # Active thread count must not have grown
        self.assertEqual(threading.active_count(), thread_count_after_warmup)

        orch.stop()

    def test_dynamic_hotkey_rebinding(self):
        """Verify hotkeys can be rebound dynamically without restarting."""
        from dodge_parry_macro.main import HotkeyManager

        triggered = []
        mapping = {"toggle_arm": "f6", "panic": "f12"}
        callbacks = {
            "toggle_arm": lambda: triggered.append("arm"),
            "panic": lambda: triggered.append("panic"),
        }

        mgr = HotkeyManager(mapping, callbacks)
        self.assertEqual(mgr.key_mapping["toggle_arm"], "f6")

        # Dynamically rebind toggle_arm to f10
        mgr.rebind("toggle_arm", "f10")
        self.assertEqual(mgr.key_mapping["toggle_arm"], "f10")

        mgr.stop()

    def test_overlay_colors_have_no_green_overlap(self):
        """Verify that all sub-box colors have zero pixels overlapping with neon green HSV thresholds."""
        import cv2
        from PyQt6 import QtGui

        # Sub-box colors from sniper_overlay.py
        colors = {
            "up": QtGui.QColor(239, 68, 68),      # Red
            "left": QtGui.QColor(59, 130, 246),   # Blue
            "down": QtGui.QColor(249, 115, 22),   # Orange
            "right": QtGui.QColor(168, 85, 247),  # Purple
            "center": QtGui.QColor(6, 182, 212),  # Cyan
        }

        green_lower = np.array([35, 70, 120], dtype=np.uint8)
        green_upper = np.array([85, 255, 255], dtype=np.uint8)

        for name, col in colors.items():
            bgr_pixel = np.array([[[col.blue(), col.green(), col.red()]]], dtype=np.uint8)
            hsv = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, green_lower, green_upper)
            self.assertEqual(
                int(cv2.countNonZero(mask)),
                0,
                f"Sub-box '{name}' color {col.name()} overlaps with green trigger thresholds!",
            )

    def test_calibration_suppress_detection(self):
        """Verify setting suppress_detection=True prevents any trigger even on pure neon green."""
        frame = create_synthetic_frame(green_zones=["left", "center", "right", "up", "down"])
        self.detector.suppress_detection = True

        results = self.detector.process_sniper_buffer(frame, now=999.0)
        self.assertEqual(len(results), 0)
        self.assertEqual(len(self.triggered_events), 0)


if __name__ == "__main__":
    unittest.main()


