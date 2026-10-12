"""Window 2: Settings, Keybind Manager, Live Debug View, and Real-Time Green Mask Diagnostics."""

import json
import os
import queue
import sys
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .detector import ReactionDetector


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

DASHBOARD_STYLE = """
QWidget {
    background-color: #0b0f19;
    color: #f1f5f9;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 8px;
    margin-top: 18px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    color: #94a3b8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #38bdf8;
}
QPushButton {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    color: #e2e8f0;
}
QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}
QPushButton:pressed {
    background-color: #0f172a;
}
QPushButton#btnArm {
    background-color: #16a34a;
    border: 1px solid #22c55e;
    color: #ffffff;
    font-size: 15px;
    font-weight: bold;
    padding: 12px 20px;
    border-radius: 8px;
}
QPushButton#btnArm:hover {
    background-color: #22c55e;
}
QPushButton#btnArm[armed="true"] {
    background-color: #dc2626;
    border: 1px solid #ef4444;
}
QPushButton#btnArm[armed="true"]:hover {
    background-color: #ef4444;
}
QPushButton.keybindBtn {
    background-color: #0f172a;
    border: 1.5px solid #38bdf8;
    color: #38bdf8;
    font-family: 'Consolas', monospace;
    font-size: 13px;
    font-weight: bold;
    min-width: 65px;
    padding: 5px 10px;
    border-radius: 6px;
}
QPushButton.keybindBtn:hover {
    background-color: #1e293b;
}
QPushButton.keybindBtn[listening="true"] {
    background-color: #854d0e;
    border: 1.5px solid #facc15;
    color: #fef08a;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #38bdf8;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f8fafc;
    border: 1px solid #64748b;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QListWidget {
    background-color: #030712;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 6px;
    color: #94a3b8;
    font-family: 'Consolas', monospace;
    font-size: 12px;
}
QListWidget::item {
    padding: 4px 6px;
    border-bottom: 1px solid #0f172a;
}
"""


class SettingsDashboardWindow(QtWidgets.QMainWindow):
    """Main Settings, Keybind Manager, and Live Diagnostic Green Mask View."""

    def __init__(
        self,
        detector: ReactionDetector,
        on_toggle_arm: Optional[Callable[[], None]] = None,
        on_toggle_overlay: Optional[Callable[[], None]] = None,
        on_toggle_always_on_top: Optional[Callable[[bool], None]] = None,
        on_toggle_roi_lock: Optional[Callable[[], None]] = None,
        on_keybind_changed: Optional[Callable[[str, str], None]] = None,
        config_path: str = CONFIG_PATH,
    ):
        super().__init__()
        self.detector = detector
        self.on_toggle_arm = on_toggle_arm
        self.on_toggle_overlay = on_toggle_overlay
        self.on_toggle_always_on_top = on_toggle_always_on_top
        self.on_toggle_roi_lock = on_toggle_roi_lock
        self.on_keybind_changed = on_keybind_changed
        self.config_path = config_path
        self.config: Dict[str, Any] = self.load_config()

        # Key rebinding state
        self.listening_action: Optional[str] = None
        self.keybind_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self.px_count_labels: Dict[str, QtWidgets.QLabel] = {}

        self.init_ui()

        # Timer to read from detector event queue & refresh live mask (30 FPS)
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.poll_updates)
        self.poll_timer.start(33)

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_config(self) -> None:
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def init_ui(self) -> None:
        self.setWindowTitle("⚡ Dodge & Parry Reaction Macro - Control Dashboard")
        self.resize(920, 800)
        self.setStyleSheet(DASHBOARD_STYLE)

        if self.config.get("menu_always_on_top", True):
            self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        # --- Top Header & Master Controls ---
        top_bar = QtWidgets.QHBoxLayout()

        toggle_key = self.config.get("keybinds", {}).get("toggle_arm", "f6").upper()
        self.btn_arm = QtWidgets.QPushButton(f"⚡ ARM BOT ({toggle_key})")
        self.btn_arm.setObjectName("btnArm")
        self.btn_arm.setProperty("armed", "false")
        self.btn_arm.clicked.connect(self.handle_arm_clicked)
        top_bar.addWidget(self.btn_arm)

        self.lbl_status_pill = QtWidgets.QLabel("PAUSED 🛑")
        self.lbl_status_pill.setStyleSheet("""
            background-color: #334155;
            color: #94a3b8;
            font-weight: bold;
            padding: 8px 14px;
            border-radius: 6px;
        """)
        top_bar.addWidget(self.lbl_status_pill)

        self.lbl_fps = QtWidgets.QLabel("FPS: -- | Latency: -- ms")
        self.lbl_fps.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 14px; padding: 0 10px;")
        top_bar.addWidget(self.lbl_fps)

        top_bar.addStretch()

        # ROI Lock Button (Game mode click-through)
        is_locked = bool(self.config.get("roi_locked", False))
        self.btn_lock = QtWidgets.QPushButton("🔒 Lock ROIs (Game Mode)" if is_locked else "🔓 Unlock ROIs (Edit Mode)")
        self.btn_lock.setStyleSheet("background-color: #0f766e; color: #ffffff;" if is_locked else "")
        self.btn_lock.clicked.connect(self.handle_lock_clicked)
        top_bar.addWidget(self.btn_lock)

        self.btn_overlay = QtWidgets.QPushButton("🎯 Show/Hide Sniper Overlay (F7)")
        self.btn_overlay.clicked.connect(self.handle_overlay_clicked)
        top_bar.addWidget(self.btn_overlay)

        main_layout.addLayout(top_bar)

        # --- Middle: Live ROI & Neon Green Mask View ---
        grp_mask = QtWidgets.QGroupBox("LIVE ROI & NEON GREEN MASK DIAGNOSTICS")
        mask_layout = QtWidgets.QHBoxLayout(grp_mask)
        mask_layout.setSpacing(16)

        # Left: Screen Crop Preview
        v_crop = QtWidgets.QVBoxLayout()
        lbl_c_title = QtWidgets.QLabel("Captured ROI View")
        lbl_c_title.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.lbl_preview_crop = QtWidgets.QLabel()
        self.lbl_preview_crop.setFixedSize(160, 140)
        self.lbl_preview_crop.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_crop.setStyleSheet("background-color: #030712; border: 1px solid #1e293b; border-radius: 6px;")
        v_crop.addWidget(lbl_c_title)
        v_crop.addWidget(self.lbl_preview_crop)
        mask_layout.addLayout(v_crop)

        # Center: Green Mask Preview
        v_mask = QtWidgets.QVBoxLayout()
        lbl_m_title = QtWidgets.QLabel("Binary Neon Green Mask [35-85 HSV]")
        lbl_m_title.setStyleSheet("font-weight: bold; color: #22c55e;")
        self.lbl_preview_mask = QtWidgets.QLabel()
        self.lbl_preview_mask.setFixedSize(160, 140)
        self.lbl_preview_mask.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_mask.setStyleSheet("background-color: #030712; border: 1px solid #1e293b; border-radius: 6px;")
        v_mask.addWidget(lbl_m_title)
        v_mask.addWidget(self.lbl_preview_mask)
        mask_layout.addLayout(v_mask)

        # Right: Real-Time Sub-Zone Pixel Counts
        v_counts = QtWidgets.QVBoxLayout()
        lbl_cnt_title = QtWidgets.QLabel("Calculated Green Pixels per Sub-ROI")
        lbl_cnt_title.setStyleSheet("font-weight: bold; color: #e2e8f0;")
        v_counts.addWidget(lbl_cnt_title)

        grid_cnt = QtWidgets.QGridLayout()
        grid_cnt.setSpacing(6)
        sub_names = [
            ("up", "Top [W]:"),
            ("left", "Left [A]:"),
            ("center", "Center [F] Parry:"),
            ("right", "Right [D]:"),
            ("down", "Bottom [S]:"),
        ]
        for idx, (act, name) in enumerate(sub_names):
            lbl_n = QtWidgets.QLabel(name)
            lbl_n.setStyleSheet("color: #94a3b8; font-weight: 600;")
            lbl_v = QtWidgets.QLabel("0 px")
            lbl_v.setStyleSheet("font-family: monospace; font-weight: bold; color: #38bdf8;")
            self.px_count_labels[act] = lbl_v
            grid_cnt.addWidget(lbl_n, idx, 0)
            grid_cnt.addWidget(lbl_v, idx, 1)

        v_counts.addLayout(grid_cnt)
        v_counts.addStretch()
        mask_layout.addLayout(v_counts, 1)

        main_layout.addWidget(grp_mask)

        # --- Lower Split: Key Mapping & Sensitivity Sliders ---
        mid_layout = QtWidgets.QHBoxLayout()

        # Group 1: Interactive Key Mapping
        grp_keys = QtWidgets.QGroupBox("ACTION KEYBINDS (CLICK TO REBIND)")
        key_grid = QtWidgets.QGridLayout(grp_keys)
        key_grid.setVerticalSpacing(8)
        key_grid.setHorizontalSpacing(14)

        actions_map = self.config.get("keybinds", {}).get("actions", {
            "up": "w", "left": "a", "down": "s", "right": "d", "center": "f"
        })
        action_labels = [
            ("up", "Up (Dodge Up):", actions_map.get("up", "w")),
            ("left", "Left (Dodge Left):", actions_map.get("left", "a")),
            ("down", "Down (Dodge Down):", actions_map.get("down", "s")),
            ("right", "Right (Dodge Right):", actions_map.get("right", "d")),
            ("center", "Center (Parry):", actions_map.get("center", "f")),
            ("toggle_arm", "Master Arm Toggle:", self.config.get("keybinds", {}).get("toggle_arm", "f6")),
            ("panic", "Emergency Panic Key:", self.config.get("keybinds", {}).get("panic", "f12")),
            ("toggle_lock", "Lock/Unlock ROIs:", self.config.get("keybinds", {}).get("toggle_lock", "f9")),
        ]

        row = 0
        for action_id, title, current_key in action_labels:
            lbl = QtWidgets.QLabel(title)
            lbl.setStyleSheet("font-weight: 600; color: #cbd5e1;")
            btn = QtWidgets.QPushButton(current_key.upper())
            btn.setProperty("class", "keybindBtn")
            btn.setProperty("listening", "false")
            btn.clicked.connect(lambda checked, a=action_id: self.start_key_rebinding(a))

            self.keybind_buttons[action_id] = btn
            key_grid.addWidget(lbl, row, 0)
            key_grid.addWidget(btn, row, 1)
            row += 1

        mid_layout.addWidget(grp_keys, 1)

        # Group 2: Sensitivity & Precision Tuning
        grp_tuning = QtWidgets.QGroupBox("DETECTION SENSITIVITY & TIMING")
        tune_layout = QtWidgets.QFormLayout(grp_tuning)
        tune_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # Min Green Pixel Count Slider
        self.slider_min_px = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_min_px.setRange(20, 400)
        self.slider_min_px.setValue(self.detector.min_green_pixels)
        self.lbl_min_px_val = QtWidgets.QLabel(f"{self.slider_min_px.value()} px")
        self.slider_min_px.valueChanged.connect(self.on_min_px_changed)
        tune_layout.addRow("Min Green Trigger Pixels:", self._wrap_slider(self.slider_min_px, self.lbl_min_px_val))

        # Action Cooldown Slider
        self.slider_cd = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_cd.setRange(100, 1000)
        self.slider_cd.setValue(int(self.detector.cooldown_s * 1000))
        self.lbl_cd_val = QtWidgets.QLabel(f"{self.slider_cd.value()} ms")
        self.slider_cd.valueChanged.connect(self.on_cd_changed)
        tune_layout.addRow("Action Cooldown:", self._wrap_slider(self.slider_cd, self.lbl_cd_val))

        # Pre-Press Delay Base Slider (0 to 500 ms)
        self.slider_delay = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_delay.setRange(0, 500)
        curr_delay = int(self.config.get("pre_press_delay_ms", getattr(self.detector, "pre_press_delay_ms", 45)))
        self.slider_delay.setValue(curr_delay)
        self.lbl_delay_val = QtWidgets.QLabel(f"{self.slider_delay.value()} ms")
        self.slider_delay.valueChanged.connect(self.on_delay_changed)
        tune_layout.addRow("Pre-Press Delay Base:", self._wrap_slider(self.slider_delay, self.lbl_delay_val))

        # Delay Random Jitter Slider (0 to 100 ms)
        self.slider_jitter = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_jitter.setRange(0, 100)
        curr_jitter = int(self.config.get("delay_jitter_ms", getattr(self.detector, "delay_jitter_ms", 10)))
        self.slider_jitter.setValue(curr_jitter)
        self.lbl_jitter_val = QtWidgets.QLabel(f"± {self.slider_jitter.value()} ms")
        self.slider_jitter.valueChanged.connect(self.on_jitter_changed)
        tune_layout.addRow("Reaction Jitter (± ms):", self._wrap_slider(self.slider_jitter, self.lbl_jitter_val))

        # Key Hold Duration Slider (10 to 150 ms)
        self.slider_hold = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_hold.setRange(10, 150)
        curr_hold = int(self.config.get("key_press_duration_ms", getattr(self.detector, "key_press_duration_ms", 40)))
        self.slider_hold.setValue(curr_hold)
        self.lbl_hold_val = QtWidgets.QLabel(f"{self.slider_hold.value()} ms")
        self.slider_hold.valueChanged.connect(self.on_hold_changed)
        tune_layout.addRow("Key Hold Duration:", self._wrap_slider(self.slider_hold, self.lbl_hold_val))

        # Strict transition checkbox
        self.chk_strict = QtWidgets.QCheckBox("Strict Transition (Require Red Outline before Green)")
        self.chk_strict.setChecked(self.detector.strict_transition)
        self.chk_strict.stateChanged.connect(self.on_strict_changed)
        tune_layout.addRow("", self.chk_strict)

        # Audio Cues checkbox
        self.chk_audio = QtWidgets.QCheckBox("Audio Feedback / Sound Cues (Beep on Trigger/Arm/Lock)")
        self.chk_audio.setChecked(self.config.get("audio_cues_enabled", True))
        self.chk_audio.stateChanged.connect(self.on_audio_changed)
        tune_layout.addRow("", self.chk_audio)

        # Menu Always on top checkbox (only controls settings menu layer)
        self.chk_menu_always_on_top = QtWidgets.QCheckBox("Menu Always on Top (Pin Settings Window)")
        self.chk_menu_always_on_top.setChecked(self.config.get("menu_always_on_top", True))
        self.chk_menu_always_on_top.stateChanged.connect(self.on_menu_always_on_top_changed)
        tune_layout.addRow("", self.chk_menu_always_on_top)

        mid_layout.addWidget(grp_tuning, 1)
        main_layout.addLayout(mid_layout)

        # --- Bottom: Live Event Log ---
        grp_log = QtWidgets.QGroupBox("LIVE DEBUG & TRIGGER EVENT LOG")
        log_layout = QtWidgets.QVBoxLayout(grp_log)
        self.list_logs = QtWidgets.QListWidget()
        log_layout.addWidget(self.list_logs)

        btn_clear = QtWidgets.QPushButton("Clear Log")
        btn_clear.setFixedWidth(100)
        btn_clear.clicked.connect(self.list_logs.clear)
        log_layout.addWidget(btn_clear, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        main_layout.addWidget(grp_log, 1)

    def _wrap_slider(self, slider: QtWidgets.QSlider, label: QtWidgets.QLabel) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(slider, 1)
        label.setFixedWidth(55)
        label.setStyleSheet("font-weight: bold; color: #38bdf8;")
        l.addWidget(label)
        return w

    def start_key_rebinding(self, action_id: str) -> None:
        if self.listening_action and self.listening_action in self.keybind_buttons:
            old_btn = self.keybind_buttons[self.listening_action]
            old_btn.setProperty("listening", "false")
            old_btn.style().unpolish(old_btn)
            old_btn.style().polish(old_btn)

        self.listening_action = action_id
        btn = self.keybind_buttons[action_id]
        btn.setProperty("listening", "true")
        btn.setText("Press key...")
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        self.setFocus()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.listening_action:
            key = event.key()
            if key == QtCore.Qt.Key.Key_Escape:
                self.restore_keybind_buttons()
                self.listening_action = None
                return

            key_name = QtGui.QKeySequence(key).toString().lower()
            if not key_name:
                key_name = event.text().lower()

            if key_name:
                self.apply_rebound_key(self.listening_action, key_name)
                self.listening_action = None
                return

        super().keyPressEvent(event)

    def apply_rebound_key(self, action_id: str, key_name: str) -> None:
        if "keybinds" not in self.config:
            self.config["keybinds"] = {}
        if "actions" not in self.config["keybinds"]:
            self.config["keybinds"]["actions"] = {}

        if action_id in ["toggle_arm", "toggle_overlay", "panic", "toggle_lock", "quit"]:
            self.config["keybinds"][action_id] = key_name
            if action_id == "toggle_arm":
                self.btn_arm.setText(f"⚡ ARM BOT ({key_name.upper()})")
        else:
            self.config["keybinds"]["actions"][action_id] = key_name

        self.save_config()
        self.restore_keybind_buttons()
        self.add_log(f"Key rebound: {action_id} -> [{key_name.upper()}]")

        if self.on_keybind_changed:
            self.on_keybind_changed(action_id, key_name)

    def restore_keybind_buttons(self) -> None:
        actions_map = self.config.get("keybinds", {}).get("actions", {})
        for act, btn in self.keybind_buttons.items():
            btn.setProperty("listening", "false")
            if act in ["toggle_arm", "toggle_overlay", "panic", "toggle_lock", "quit"]:
                btn.setText(self.config.get("keybinds", {}).get(act, "").upper())
            else:
                btn.setText(actions_map.get(act, "").upper())
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def handle_lock_clicked(self) -> None:
        if self.on_toggle_roi_lock:
            self.on_toggle_roi_lock()
        else:
            curr = bool(self.config.get("roi_locked", False))
            self.set_roi_locked(not curr)

    def set_roi_locked(self, locked: bool) -> None:
        self.config["roi_locked"] = locked
        self.save_config()
        if hasattr(self, "btn_lock"):
            self.btn_lock.setText("🔒 Lock ROIs (Game Mode)" if locked else "🔓 Unlock ROIs (Edit Mode)")
            self.btn_lock.setStyleSheet("background-color: #0f766e; color: #ffffff;" if locked else "")
        self.add_log(f"ROI Mode: {'LOCKED (Game Click-Through)' if locked else 'UNLOCKED (Calibration Mode)'}")

    def handle_arm_clicked(self) -> None:
        if self.on_toggle_arm:
            self.on_toggle_arm()
        else:
            self.detector.armed = not self.detector.armed
            self.update_arm_state()

    def handle_overlay_clicked(self) -> None:
        if self.on_toggle_overlay:
            self.on_toggle_overlay()

    def update_arm_state(self) -> None:
        armed = self.detector.armed
        toggle_key = self.config.get("keybinds", {}).get("toggle_arm", "f6").upper()
        self.btn_arm.setProperty("armed", "true" if armed else "false")
        self.btn_arm.setText(f"🛑 STOP BOT ({toggle_key})" if armed else f"⚡ ARM BOT ({toggle_key})")
        self.btn_arm.style().unpolish(self.btn_arm)
        self.btn_arm.style().polish(self.btn_arm)

        if armed:
            self.lbl_status_pill.setText("ACTIVE ⚡")
            self.lbl_status_pill.setStyleSheet("background-color: #16a34a; color: #ffffff; font-weight: bold; padding: 8px 14px; border-radius: 6px;")
        else:
            self.lbl_status_pill.setText("PAUSED 🛑")
            self.lbl_status_pill.setStyleSheet("background-color: #334155; color: #94a3b8; font-weight: bold; padding: 8px 14px; border-radius: 6px;")

    def on_min_px_changed(self, val: int) -> None:
        self.lbl_min_px_val.setText(f"{val} px")
        self.detector.min_green_pixels = val
        self.config["color_thresholds"]["min_green_pixels"] = val
        self.save_config()

    def on_cd_changed(self, val: int) -> None:
        self.lbl_cd_val.setText(f"{val} ms")
        self.detector.cooldown_s = val / 1000.0
        self.config["timing"]["cooldown_ms"] = val
        self.save_config()

    def on_strict_changed(self, state: int) -> None:
        self.detector.strict_transition = (state == QtCore.Qt.CheckState.Checked.value)
        self.config["color_thresholds"]["strict_transition"] = self.detector.strict_transition
        self.save_config()

    def on_delay_changed(self, val: int) -> None:
        self.lbl_delay_val.setText(f"{val} ms")
        self.detector.pre_press_delay_ms = float(val)
        self.detector.pre_press_delay_s = float(val) / 1000.0
        self.config["pre_press_delay_ms"] = val
        if "timing" in self.config:
            self.config["timing"]["pre_press_delay_ms"] = val
        self.save_config()

    def on_jitter_changed(self, val: int) -> None:
        self.lbl_jitter_val.setText(f"± {val} ms")
        self.detector.delay_jitter_ms = float(val)
        self.config["delay_jitter_ms"] = val
        if "timing" in self.config:
            self.config["timing"]["delay_jitter_ms"] = val
        self.save_config()

    def on_hold_changed(self, val: int) -> None:
        self.lbl_hold_val.setText(f"{val} ms")
        self.detector.key_press_duration_ms = float(val)
        self.config["key_press_duration_ms"] = val
        if "timing" in self.config:
            self.config["timing"]["key_press_duration_ms"] = val
        self.save_config()

    def on_audio_changed(self, state: int) -> None:
        is_checked = (state == QtCore.Qt.CheckState.Checked.value or state == 2)
        self.config["audio_cues_enabled"] = is_checked
        self.detector.audio_cues_enabled = is_checked
        self.save_config()
        self.add_log(f"Audio Feedback: {'Enabled' if is_checked else 'Disabled'}")

    def on_menu_always_on_top_changed(self, state: int) -> None:
        is_checked = (state == QtCore.Qt.CheckState.Checked.value or state == 2)
        self.config["menu_always_on_top"] = is_checked
        self.save_config()
        self.set_menu_always_on_top(is_checked)
        self.add_log(f"Menu Always on Top: {'Enabled' if is_checked else 'Disabled'}")

    def set_menu_always_on_top(self, enabled: bool) -> None:
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.show()

    # Backwards compatibility alias
    on_always_on_top_changed = on_menu_always_on_top_changed
    set_always_on_top = set_menu_always_on_top

    def add_log(self, text: str) -> None:
        item = QtWidgets.QListWidgetItem(text)
        self.list_logs.addItem(item)
        if self.list_logs.count() > 100:
            self.list_logs.takeItem(0)
        self.list_logs.scrollToBottom()

    def poll_updates(self) -> None:
        self.update_arm_state()

        # Update FPS and latency
        fps = self.detector.fps
        lat = self.detector.latency_ms
        self.lbl_fps.setText(f"FPS: {fps:5.1f} | Latency: {lat:4.1f} ms")

        # Update Live ROI Previews (Screen crop & Green mask)
        frame_bgr = self.detector.latest_bgr_frame
        if frame_bgr is not None and frame_bgr.size > 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg_crop = QtGui.QImage(rgb.data, w, h, w * ch, QtGui.QImage.Format.Format_RGB888)
            pix_crop = QtGui.QPixmap.fromImage(qimg_crop).scaled(
                160, 140, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.FastTransformation
            )
            self.lbl_preview_crop.setPixmap(pix_crop)

        mask_green = self.detector.latest_green_mask
        if mask_green is not None and mask_green.size > 0:
            h, w = mask_green.shape
            qimg_mask = QtGui.QImage(mask_green.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
            pix_mask = QtGui.QPixmap.fromImage(qimg_mask).scaled(
                160, 140, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.FastTransformation
            )
            self.lbl_preview_mask.setPixmap(pix_mask)

        # Update Sub-ROI Green Pixel Counts
        for act, lbl in self.px_count_labels.items():
            cnt = self.detector.latest_sub_counts.get(act, 0)
            lbl.setText(f"{cnt} px")
            if cnt >= self.detector.min_green_pixels:
                lbl.setStyleSheet("font-family: monospace; font-weight: bold; color: #22c55e;")
            elif cnt > 0:
                lbl.setStyleSheet("font-family: monospace; font-weight: bold; color: #facc15;")
            else:
                lbl.setStyleSheet("font-family: monospace; font-weight: bold; color: #64748b;")

        # Drain event queue from detector
        if self.detector.event_queue:
            while not self.detector.event_queue.empty():
                try:
                    evt = self.detector.event_queue.get_nowait()
                    if "log_msg" in evt:
                        line = f"[{evt['time']}] ⚡ {evt['log_msg']} - {evt['green_px']} px ({evt['green_pct']:.1f}%)"
                    else:
                        line = f"[{evt['time']}] ⚡ TRIGGER '{evt['key']}' ({evt['name']}) - {evt['green_px']} px ({evt['green_pct']:.1f}%)"
                    self.add_log(line)
                except queue.Empty:
                    break
