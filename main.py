"""Main entry point and orchestrator for multi-window dodge/parry reaction macro."""

# Enforce Per-Monitor DPI Awareness before any GUI / display calls
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import argparse
import json
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

import mss
from PyQt6 import QtWidgets

if __package__ is None or __package__ == "":
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dodge_parry_macro.detector import ReactionDetector
    from dodge_parry_macro.input_handler import DirectInputHandler, play_audio_cue
    from dodge_parry_macro.settings_gui import SettingsDashboardWindow
    from dodge_parry_macro.sniper_overlay import SniperOverlayWindow
else:
    from .detector import ReactionDetector
    from .input_handler import DirectInputHandler, play_audio_cue
    from .settings_gui import SettingsDashboardWindow
    from .sniper_overlay import SniperOverlayWindow


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


class HotkeyManager:
    """Manages global hotkeys with dynamic re-binding"""

    def __init__(self, key_mapping: Dict[str, str], callbacks: Dict[str, Callable[[], None]]):
        self.key_mapping = {k: v.lower().strip() for k, v in key_mapping.items()}
        self.callbacks = callbacks
        self._lock = threading.Lock()
        self.backend = None
        self._pynput_listener = None
        self._init_listener()

    def _init_listener(self) -> None:
        try:
            from pynput import keyboard as pk

            def _on_press(key):
                key_str = None
                if hasattr(key, "name") and key.name:
                    key_str = key.name.lower()
                elif hasattr(key, "char") and key.char:
                    key_str = key.char.lower()
                else:
                    key_str = str(key).lower().replace("key.", "")

                if not key_str:
                    return

                with self._lock:
                    for action, bound_key in self.key_mapping.items():
                        if key_str == bound_key:
                            cb = self.callbacks.get(action)
                            if cb:
                                try:
                                    cb()
                                except Exception as e:
                                    print(f"[HotkeyManager] Error executing {action}: {e}")

            self._pynput_listener = pk.Listener(on_press=_on_press)
            self._pynput_listener.daemon = True
            self._pynput_listener.start()
            self.backend = "pynput"
            print(f"[HotkeyManager] Dynamic pynput global listener active.")
        except Exception as pynput_err:
            try:
                import keyboard
                self.backend = "keyboard"
                self._rebind_keyboard()
                print(f"[HotkeyManager] Dynamic keyboard module hotkeys active.")
            except Exception as kb_err:
                print(f"[HotkeyManager] Global hotkeys unavailable ({pynput_err} / {kb_err}).")

    def _rebind_keyboard(self) -> None:
        import keyboard
        try:
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        with self._lock:
            for action, key_name in self.key_mapping.items():
                cb = self.callbacks.get(action)
                if cb and key_name:
                    try:
                        keyboard.add_hotkey(key_name, cb)
                    except Exception as e:
                        print(f"[HotkeyManager] Could not bind {key_name} to {action}: {e}")

    def rebind(self, action: str, new_key: str) -> None:
        k = new_key.lower().strip()
        with self._lock:
            self.key_mapping[action] = k
        if self.backend == "keyboard":
            self._rebind_keyboard()
        print(f"[HotkeyManager] Action '{action}' dynamically rebound to '{k.upper()}'. Instant effect.")

    def stop(self) -> None:
        if self._pynput_listener:
            try:
                self._pynput_listener.stop()
            except Exception:
                pass
        if self.backend == "keyboard":
            try:
                import keyboard
                keyboard.unhook_all_hotkeys()
            except Exception:
                pass


class MacroOrchestrator:
    """Manages multi-window application, persistent background worker thread, and dynamic hotkeys."""

    def __init__(self, config_path: str = CONFIG_PATH, mock_input: bool = False):
        self.config_path = config_path
        self.config = self.load_config()
        self.mock_input = mock_input

        timing = self.config.get("timing", {})
        pre_press_delay_ms = float(
            self.config.get("pre_press_delay_ms", timing.get("pre_press_delay_ms", 60))
        )
        self.input_handler = DirectInputHandler(
            default_press_duration_ms=timing.get("key_press_duration_ms", 40),
            pre_press_delay_ms=pre_press_delay_ms,
            mock=mock_input,
        )

        self.event_queue: queue.Queue = queue.Queue()
        self.detector = ReactionDetector(
            config=self.config,
            on_trigger_callback=self.handle_trigger,
            event_queue=self.event_queue,
        )

        # Thread lifecycle management: single persistent daemon thread
        self.running_event = threading.Event()
        self.armed_event = threading.Event()
        self.suppress_detection = False
        self.capture_thread: Optional[threading.Thread] = None

        # Dynamic Hotkey manager
        self.hotkey_manager: Optional[HotkeyManager] = None

        # GUI windows
        self.overlay_window: Optional[SniperOverlayWindow] = None
        self.settings_window: Optional[SettingsDashboardWindow] = None

    def load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def reload_config(self) -> None:
        self.config = self.load_config()
        self.detector.config = self.config
        self.detector._update_thresholds()
        self.detector._init_states()
        self.input_handler.pre_press_delay_s = getattr(self.detector, "pre_press_delay_s", 0.0)

    def handle_trigger(
        self,
        action_id: str,
        key: str,
        delay_s: Optional[float] = None,
        hold_s: Optional[float] = None,
    ) -> None:
        if delay_s is None:
            delay_s = getattr(self.detector, "pre_press_delay_s", 0.0)
        if hold_s is None:
            hold_s = self.input_handler.default_press_duration
        self.input_handler.press_key_async(key, duration_s=hold_s, delay_s=delay_s)

    def set_always_on_top(self, enabled: bool) -> None:
        self.config["menu_always_on_top"] = enabled

    def toggle_arm(self) -> None:
        """Toggles bot state cleanly via threading.Event without creating or killing worker threads."""
        if self.armed_event.is_set():
            self.armed_event.clear()
            self.detector.armed = False
        else:
            self.armed_event.set()
            self.detector.armed = True

        status_text = "ACTIVE" if self.detector.armed else "PAUSED"
        status = "ACTIVE ⚡" if self.detector.armed else "PAUSED 🛑"
        play_audio_cue("arm" if self.detector.armed else "disarm", enabled=self.detector.audio_cues_enabled)
        print(f"[Macro] Status: {status_text}")
        if self.settings_window:
            self.settings_window.update_arm_state()
            self.settings_window.add_log(f"Bot state: {status}")

    def toggle_roi_lock(self) -> None:
        is_locked = not bool(self.config.get("roi_locked", False))
        self.config["roi_locked"] = is_locked
        play_audio_cue("lock" if is_locked else "unlock", enabled=self.detector.audio_cues_enabled)
        if self.overlay_window:
            self.overlay_window.set_locked(is_locked)
        if self.settings_window:
            self.settings_window.set_roi_locked(is_locked)

    def panic(self) -> None:
        """Hardware killswitch: instantly disarm, release all keys, clear queues."""
        self.armed_event.clear()
        self.detector.armed = False
        self.input_handler.release_all_keys()
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
        play_audio_cue("panic", enabled=self.detector.audio_cues_enabled)
        print("[PANIC KILLSWITCH] Disarmed and released all held keys!")
        if self.settings_window:
            self.settings_window.update_arm_state()
            self.settings_window.add_log("🚨 PANIC KILLSWITCH TRIGGERED! Bot disarmed & all keys released.")

    def toggle_overlay(self) -> None:
        if self.overlay_window:
            if self.overlay_window.isVisible():
                self.overlay_window.hide()
                if self.settings_window:
                    self.settings_window.add_log("Sniper Overlay: Hidden")
            else:
                self.overlay_window.show()
                if self.settings_window:
                    self.settings_window.add_log("Sniper Overlay: Shown")

    def on_overlay_geometry_changed(self, x: int, y: int, w: int, h: int) -> None:
        self.config["sniper_box"] = {"x": x, "y": y, "width": w, "height": h}
        if self.overlay_window:
            self.config["sub_rois"] = self.overlay_window.get_relative_sub_rois_dict()
            self.config["rois"] = self.overlay_window.get_absolute_rois()
        self.detector.config = self.config

    def on_calibration_active(self, active: bool) -> None:
        """Suppresses detection while user is actively dragging/resizing ROIs on screen."""
        self.suppress_detection = active
        self.detector.suppress_detection = active

    def on_keybind_changed(self, action_id: str, new_key: str) -> None:
        """Dynamically rebinds hotkeys and updates overlay labels in real-time."""
        self.reload_config()
        if action_id in ["toggle_arm", "toggle_overlay", "panic", "toggle_lock", "quit"]:
            if self.hotkey_manager:
                self.hotkey_manager.rebind(action_id, new_key)
        else:
            if self.overlay_window:
                self.overlay_window.update_action_label(action_id, new_key.upper())

    def capture_loop(self) -> None:
        """Persistent worker loop. Runs continuously for process lifetime, sleeping when disarmed."""
        with mss.MSS() as sct:
            while self.running_event.is_set():
                if not self.armed_event.is_set() or self.suppress_detection:
                    time.sleep(0.05)
                    continue
                try:
                    self.detector.capture_and_process_once(sct)
                    time.sleep(0.001)
                except Exception:
                    time.sleep(0.05)

    def start_capture_thread(self) -> None:
        """Spawns the single persistent worker daemon thread once."""
        if self.capture_thread and self.capture_thread.is_alive():
            return
        self.running_event.set()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self) -> None:
        """Gracefully halts the persistent worker loop and hotkey listener."""
        self.running_event.clear()
        self.armed_event.clear()
        if self.hotkey_manager:
            self.hotkey_manager.stop()

    def setup_hotkeys(self) -> None:
        keybinds = self.config.get("keybinds", {})
        key_mapping = {
            "toggle_arm": keybinds.get("toggle_arm", "f6"),
            "toggle_overlay": keybinds.get("toggle_overlay", "f7"),
            "toggle_lock": keybinds.get("toggle_lock", "f9"),
            "panic": keybinds.get("panic", "f12"),
            "quit": keybinds.get("quit", "f8"),
        }
        callbacks = {
            "toggle_arm": self.toggle_arm,
            "toggle_overlay": self.toggle_overlay,
            "toggle_lock": self.toggle_roi_lock,
            "panic": self.panic,
            "quit": lambda: os._exit(0),
        }
        self.hotkey_manager = HotkeyManager(key_mapping, callbacks)


def main():
    parser = argparse.ArgumentParser(description="Multi-Window Dodge/Parry Reaction Macro")
    parser.add_argument("--test-mode", action="store_true", help="Run with mock hardware input (safe test)")
    parser.add_argument("--headless", action="store_true", help="Run without Qt GUI")
    args = parser.parse_args()

    orchestrator = MacroOrchestrator(mock_input=args.test_mode)
    orchestrator.setup_hotkeys()
    orchestrator.start_capture_thread()

    if args.headless:
        print("[Macro] Headless mode active. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            orchestrator.stop()
    else:
        app = QtWidgets.QApplication(sys.argv)

        # 1. Window 1: Sniper Overlay (Permanently Topmost)
        overlay = SniperOverlayWindow(config_path=orchestrator.config_path)
        overlay.geometry_changed.connect(orchestrator.on_overlay_geometry_changed)
        overlay.calibration_active.connect(orchestrator.on_calibration_active)
        orchestrator.overlay_window = overlay
        overlay.show()

        # 2. Window 2: Settings & Debug Dashboard (Independent Pinning)
        settings = SettingsDashboardWindow(
            detector=orchestrator.detector,
            on_toggle_arm=orchestrator.toggle_arm,
            on_toggle_overlay=orchestrator.toggle_overlay,
            on_toggle_always_on_top=orchestrator.set_always_on_top,
            on_toggle_roi_lock=orchestrator.toggle_roi_lock,
            on_keybind_changed=orchestrator.on_keybind_changed,
            config_path=orchestrator.config_path,
        )
        orchestrator.settings_window = settings
        settings.show()

        ret = app.exec()
        orchestrator.stop()
        sys.exit(ret)


if __name__ == "__main__":
    main()
