"""Unit tests for ReactionDetector using synthetic buffers and image_d4f267.png."""

import copy
import os
import queue
import unittest
import cv2
import numpy as np

from dodge_parry_macro.detector import ReactionDetector, ROIState


IMAGE_D4F267_PATH = os.path.join(os.path.dirname(__file__), "image_d4f267.png")

TEST_CONFIG = {
    "sniper_box": {
        "x": 860,
        "y": 440,
        "width": 200,
        "height": 200,
    },
    "keybinds": {
        "toggle_arm": "f6",
        "toggle_overlay": "f7",
        "quit": "f8",
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
        "green_lower": [35, 120, 120],
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
        "cooldown_ms": 350,
        "key_press_duration_ms": 40,
        "scan_interval_ms": 1,
    },
}


def make_grass_patch(h: int = 50, w: int = 50) -> np.ndarray:
    np.random.seed(123)
    b = np.random.randint(15, 30, (h, w), dtype=np.uint8)
    g = np.random.randint(60, 95, (h, w), dtype=np.uint8)
    r = np.random.randint(15, 30, (h, w), dtype=np.uint8)
    return np.dstack([b, g, r])


def make_red_cue(h: int = 50, w: int = 50) -> np.ndarray:
    img = make_grass_patch(h, w)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = [20, 20, 225]
    return img


def make_neon_green_flash(h: int = 50, w: int = 50) -> np.ndarray:
    img = make_grass_patch(h, w)
    # Neon green #39FF14: BGR [20, 255, 57]
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = [20, 255, 57]
    return img


class TestMultiWindowDetector(unittest.TestCase):
    def setUp(self):
        self.triggered_events = []
        self.event_queue = queue.Queue()
        self.detector = ReactionDetector(
            config=copy.deepcopy(TEST_CONFIG),
            on_trigger_callback=lambda act, k: self.triggered_events.append((act, k)),
            event_queue=self.event_queue,
        )
        self.detector.armed = True

    def test_mock_image_d4f267_triggers_parry_f(self):
        """Test with provided image_d4f267.png to prove detector counts green pixels and fires F."""
        self.assertTrue(os.path.exists(IMAGE_D4F267_PATH), "image_d4f267.png not found!")
        frame = cv2.imread(IMAGE_D4F267_PATH)
        self.assertIsNotNone(frame)

        results = self.detector.process_sniper_buffer(frame, now=1.0)
        center_res = next((r for r in results if r.action_id == "center"), None)

        self.assertIsNotNone(center_res)
        self.assertGreaterEqual(center_res.green_count, 80, f"Expected >= 80 green pixels, got {center_res.green_count}")
        self.assertTrue(center_res.triggered, "Center parry did not trigger!")
        self.assertEqual(center_res.key, "f")
        self.assertIn(("center", "f"), self.triggered_events)

        # Check debug event emitted to queue
        self.assertFalse(self.event_queue.empty())
        evt = self.event_queue.get_nowait()
        self.assertEqual(evt["action"], "center")
        self.assertEqual(evt["key"], "F")
        self.assertGreaterEqual(evt["green_px"], 80)

    def test_pure_red_does_not_trigger_keypress(self):
        """Frames with pure red cue must enter RED_WAITING and not fire any key."""
        red_frame = make_red_cue()
        res = self.detector.analyze_crop(red_frame, "center", now=1.0)

        self.assertEqual(res.state, ROIState.RED_WAITING)
        self.assertFalse(res.triggered)
        self.assertEqual(len(self.triggered_events), 0)
        self.assertTrue(self.event_queue.empty())

    def test_green_transition_triggers_key_and_debug_event(self):
        """Transition to neon green must trigger keypress and emit debug event."""
        green_frame = make_neon_green_flash()

        res = self.detector.analyze_crop(green_frame, "center", now=1.12)
        self.assertTrue(res.triggered)
        self.assertEqual(res.key, "f")
        self.assertEqual(self.triggered_events, [("center", "f")])

        self.assertFalse(self.event_queue.empty())
        evt = self.event_queue.get_nowait()
        self.assertEqual(evt["action"], "center")
        self.assertEqual(evt["key"], "F")
        self.assertGreater(evt["green_px"], 50)

    def test_cooldown_prevents_duplicate_firing(self):
        """Subsequent green frames during cooldown (350ms) are suppressed."""
        green_frame = make_neon_green_flash()

        res1 = self.detector.analyze_crop(green_frame, "up", now=10.0)
        self.assertTrue(res1.triggered)
        self.assertEqual(len(self.triggered_events), 1)

        # Frame at t=10.15 (150ms later, inside 350ms cooldown)
        res2 = self.detector.analyze_crop(green_frame, "up", now=10.15)
        self.assertFalse(res2.triggered, "Duplicate trigger fired inside cooldown!")
        self.assertEqual(len(self.triggered_events), 1)

    def test_custom_keybind_remapping(self):
        """Detector respects remapped keybinds from config."""
        self.detector.config["keybinds"]["actions"]["center"] = "space"
        green_frame = make_neon_green_flash()

        res = self.detector.analyze_crop(green_frame, "center", now=20.1)
        self.assertTrue(res.triggered)
        self.assertEqual(res.key, "space")
        self.assertIn(("center", "space"), self.triggered_events)

    def test_sub_roi_relative_mapping(self):
        """Verify 5 sub-zones mapping: W top 30%, S bottom 30%, A left 30%, D right 30%, F center 40%."""
        w, h = 200, 200
        sub_rois = self.detector.get_relative_sub_rois(w, h)

        self.assertEqual(sub_rois["up"], (60, 0, 140, 60))
        self.assertEqual(sub_rois["down"], (60, 140, 140, 200))
        self.assertEqual(sub_rois["left"], (0, 60, 60, 140))
        self.assertEqual(sub_rois["right"], (140, 60, 200, 140))
        self.assertEqual(sub_rois["center"], (60, 60, 140, 140))


if __name__ == "__main__":
    unittest.main()
