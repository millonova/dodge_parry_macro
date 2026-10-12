"""Synthetic unit tests for ReactionDetector and color transition state machine."""

import copy
import unittest
import numpy as np

from dodge_parry_macro.detector import ReactionDetector, ROIState
from dodge_parry_macro.input_handler import DirectInputHandler, SCANCODES


TEST_CONFIG = {
    "screen_resolution": [1920, 1080],
    "rois": {
        "up": {"key": "w", "name": "Up (W)", "x": 100, "y": 100, "w": 40, "h": 40},
        "left": {"key": "a", "name": "Left (A)", "x": 50, "y": 150, "w": 40, "h": 40},
        "down": {"key": "s", "name": "Down (S)", "x": 100, "y": 200, "w": 40, "h": 40},
        "right": {"key": "d", "name": "Right (D)", "x": 150, "y": 150, "w": 40, "h": 40},
        "center": {"key": "f", "name": "Parry (F)", "x": 100, "y": 150, "w": 40, "h": 40},
    },
    "color_thresholds": {
        "red_lower_1": [0, 100, 100],
        "red_upper_1": [10, 255, 255],
        "red_lower_2": [170, 100, 100],
        "red_upper_2": [180, 255, 255],
        "green_lower": [35, 100, 120],
        "green_upper": [85, 255, 255],
        "red_min_density": 0.08,
        "green_min_density": 0.15,
        "strict_transition": True,
    },
    "timing": {
        "cooldown_ms": 350,
        "key_press_duration_ms": 35,
        "scan_interval_ms": 1,
    },
    "hotkeys": {
        "toggle_arm": "f6",
        "recalibrate": "f7",
        "quit": "f8",
    },
}


def make_grass_patch(height: int = 40, width: int = 40) -> np.ndarray:
    """Generate synthetic grass background (muted natural green)."""
    # Muted dark green BGR: B=20-30, G=70-90, R=20-35
    np.random.seed(42)
    b = np.random.randint(15, 35, (height, width), dtype=np.uint8)
    g = np.random.randint(60, 95, (height, width), dtype=np.uint8)
    r = np.random.randint(15, 35, (height, width), dtype=np.uint8)
    return np.dstack([b, g, r])


def make_red_indicator(height: int = 40, width: int = 40) -> np.ndarray:
    """Generate synthetic red indicator arrow/crest with white letter."""
    img = make_grass_patch(height, width)
    # Bright red triangle/arrow shape in center: BGR [20, 20, 220]
    y1, y2 = height // 4, 3 * height // 4
    x1, x2 = width // 4, 3 * width // 4
    img[y1:y2, x1:x2] = [20, 20, 225]
    # White letter in center: BGR [255, 255, 255]
    cy, cx = height // 2, width // 2
    img[cy - 2 : cy + 2, cx - 2 : cx + 2] = [255, 255, 255]
    return img


def make_green_flash_indicator(height: int = 40, width: int = 40) -> np.ndarray:
    """Generate synthetic green reaction flash."""
    img = make_grass_patch(height, width)
    # Bright neon green flash: BGR [20, 245, 30] (HSV H~60, S~235, V~245)
    y1, y2 = height // 4, 3 * height // 4
    x1, x2 = width // 4, 3 * width // 4
    img[y1:y2, x1:x2] = [20, 245, 30]
    # White letter in center
    cy, cx = height // 2, width // 2
    img[cy - 2 : cy + 2, cx - 2 : cx + 2] = [255, 255, 255]
    return img


class TestReactionDetector(unittest.TestCase):
    def setUp(self):
        self.triggered_keys = []
        self.detector = ReactionDetector(
            config=copy.deepcopy(TEST_CONFIG),
            on_trigger_callback=lambda roi_id, key: self.triggered_keys.append((roi_id, key)),
        )
        self.detector.armed = True

    def test_grass_background_suppression(self):
        """Natural grass background must not trigger any keypress."""
        grass = make_grass_patch(40, 40)
        res = self.detector.analyze_roi_buffer(grass, "up", now=10.0)

        self.assertFalse(res.triggered, "Grass background caused false trigger!")
        self.assertNotEqual(res.state, ROIState.TRIGGER_GREEN)
        self.assertEqual(len(self.triggered_keys), 0)

    def test_red_baseline_detection(self):
        """Red indicator sets state to RED_WAITING, does not trigger keypress."""
        red_crop = make_red_indicator(40, 40)
        res = self.detector.analyze_roi_buffer(red_crop, "center", now=10.0)

        self.assertEqual(res.state, ROIState.RED_WAITING)
        self.assertGreater(res.red_density, 0.08)
        self.assertFalse(res.triggered, "Red indicator prematurely triggered keypress!")
        self.assertEqual(len(self.triggered_keys), 0)

    def test_red_to_green_transition_triggers_correct_key(self):
        """Transitioning from Red to Green flash must trigger target key."""
        red_crop = make_red_indicator(40, 40)
        green_crop = make_green_flash_indicator(40, 40)

        # Step 1: Red frame at t=10.0
        res1 = self.detector.analyze_roi_buffer(red_crop, "up", now=10.0)
        self.assertEqual(res1.state, ROIState.RED_WAITING)
        self.assertFalse(res1.triggered)

        # Step 2: Green flash at t=10.15 (150ms later)
        res2 = self.detector.analyze_roi_buffer(green_crop, "up", now=10.15)
        self.assertEqual(res2.state, ROIState.TRIGGER_GREEN)
        self.assertTrue(res2.triggered, "Green flash did not trigger action!")
        self.assertIn(("up", "w"), self.triggered_keys)

    def test_strict_mode_prevents_unheralded_green(self):
        """If green flash appears without a prior red indicator, strict mode rejects it."""
        green_crop = make_green_flash_indicator(40, 40)

        # No red frame beforehand
        res = self.detector.analyze_roi_buffer(green_crop, "left", now=50.0)
        self.assertFalse(res.triggered, "Strict mode allowed trigger without red precursor!")
        self.assertNotEqual(res.state, ROIState.TRIGGER_GREEN)
        self.assertEqual(len(self.triggered_keys), 0)

    def test_cooldown_retrigger_prevention(self):
        """Successive green frames within cooldown period must not re-trigger."""
        red_crop = make_red_indicator(40, 40)
        green_crop = make_green_flash_indicator(40, 40)

        # Trigger sequence
        self.detector.analyze_roi_buffer(red_crop, "center", now=100.0)
        res_first = self.detector.analyze_roi_buffer(green_crop, "center", now=100.1)
        self.assertTrue(res_first.triggered)
        self.assertEqual(len(self.triggered_keys), 1)

        # Frame immediately following (40ms later, cooldown is 350ms)
        res_second = self.detector.analyze_roi_buffer(green_crop, "center", now=100.14)
        self.assertFalse(res_second.triggered, "Cooldown failed to prevent duplicate trigger!")
        self.assertEqual(len(self.triggered_keys), 1)

    def test_directional_key_mapping(self):
        """All 5 directions map to W, A, S, D, F correctly."""
        mappings = [
            ("up", "w"),
            ("left", "a"),
            ("down", "s"),
            ("right", "d"),
            ("center", "f"),
        ]

        t = 200.0
        for roi_id, expected_key in mappings:
            red_crop = make_red_indicator(40, 40)
            green_crop = make_green_flash_indicator(40, 40)

            self.detector.analyze_roi_buffer(red_crop, roi_id, now=t)
            res = self.detector.analyze_roi_buffer(green_crop, roi_id, now=t + 0.1)
            self.assertTrue(res.triggered, f"Failed to trigger {roi_id}")
            self.assertEqual(res.key, expected_key)
            self.assertIn((roi_id, expected_key), self.triggered_keys)
            t += 1.0  # advance past cooldown

    def test_direct_input_handler(self):
        """Verify DirectInputHandler mock mode tracks scancodes."""
        handler = DirectInputHandler(default_press_duration_ms=10, mock=True)
        handler.press_key_sync("w")
        self.assertEqual(handler.last_pressed_key, "w")
        self.assertEqual(handler.total_presses, 1)

        for key in ["a", "s", "d", "f"]:
            self.assertIn(key, SCANCODES)


if __name__ == "__main__":
    unittest.main()
