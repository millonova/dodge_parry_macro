"""Real-time PyQt6 Debug HUD & Control Window."""

import datetime
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dodge_parry_macro.detector import ReactionDetector, ROIDetectionResult, ROIState
    from dodge_parry_macro.roi_manager import open_roi_calibration
else:
    from .detector import ReactionDetector, ROIDetectionResult, ROIState
    from .roi_manager import open_roi_calibration


STYLE_SHEET = """
QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 18px;
    padding-top: 12px;
    font-weight: bold;
    color: #8b949e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #21262d;
    border: 1px solid #363b42;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    color: #c9d1d9;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
}
QPushButton:pressed {
    background-color: #161b22;
}
QPushButton#btnArm {
    background-color: #238636;
    border: 1px solid #2ea043;
    color: #ffffff;
    font-size: 15px;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 8px;
}
QPushButton#btnArm:hover {
    background-color: #2ea043;
}
QPushButton#btnArm[armed="true"] {
    background-color: #da3633;
    border: 1px solid #f85149;
}
QPushButton#btnArm[armed="true"]:hover {
    background-color: #f85149;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #21262d;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #38bdf8;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f0f6fc;
    border: 1px solid #8b949e;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QListWidget {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
    color: #8b949e;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
QListWidget::item {
    padding: 4px;
    border-bottom: 1px solid #21262d;
}
"""


class ROICardWidget(QtWidgets.QFrame):
    """Card displaying live video crop, HSV mask, and status for one ROI."""

    def __init__(self, roi_id: str, title: str, key: str, parent=None):
        super().__init__(parent)
        self.roi_id = roi_id
        self.title = title
        self.key = key

        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            ROICardWidget {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header: Key badge & Title
        hdr = QtWidgets.QHBoxLayout()
        self.lbl_key = QtWidgets.QLabel(f"[{key.upper()}]")
        self.lbl_key.setStyleSheet("""
            background-color: #38bdf8;
            color: #0b1329;
            font-weight: bold;
            font-size: 12px;
            padding: 2px 6px;
            border-radius: 4px;
        """)
        self.lbl_title = QtWidgets.QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: bold; color: #e6edf3;")
        hdr.addWidget(self.lbl_key)
        hdr.addWidget(self.lbl_title)
        hdr.addStretch()

        self.lbl_status = QtWidgets.QLabel("IDLE")
        self.lbl_status.setStyleSheet("""
            background-color: #21262d;
            color: #8b949e;
            font-weight: bold;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
        """)
        hdr.addWidget(self.lbl_status)
        layout.addLayout(hdr)

        # Live Image Preview
        self.lbl_preview = QtWidgets.QLabel()
        self.lbl_preview.setFixedSize(110, 80)
        self.lbl_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("background-color: #090d13; border-radius: 4px;")
        layout.addWidget(self.lbl_preview, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Metrics text
        self.lbl_metrics = QtWidgets.QLabel("R: 0.0% | G: 0.0%")
        self.lbl_metrics.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_metrics.setStyleSheet("color: #8b949e; font-size: 11px; font-family: monospace;")
        layout.addWidget(self.lbl_metrics)

    def update_data(self, res: ROIDetectionResult) -> None:
        r_pct = res.red_density * 100.0
        g_pct = res.green_density * 100.0
        self.lbl_metrics.setText(f"R:{r_pct:4.1f}% | G:{g_pct:4.1f}%")

        # Update status pill
        if res.state == ROIState.TRIGGER_GREEN or res.triggered:
            self.lbl_status.setText("GREEN FLASH!")
            self.lbl_status.setStyleSheet("background-color: #238636; color: #ffffff; font-weight: bold; padding: 2px 6px; border-radius: 4px;")
            self.setStyleSheet("ROICardWidget { background-color: #1a2e1d; border: 1.5px solid #2ea043; border-radius: 8px; }")
        elif res.state == ROIState.RED_WAITING:
            self.lbl_status.setText("RED READY")
            self.lbl_status.setStyleSheet("background-color: #da3633; color: #ffffff; font-weight: bold; padding: 2px 6px; border-radius: 4px;")
            self.setStyleSheet("ROICardWidget { background-color: #2d181a; border: 1.5px solid #f85149; border-radius: 8px; }")
        else:
            self.lbl_status.setText("IDLE")
            self.lbl_status.setStyleSheet("background-color: #21262d; color: #8b949e; font-weight: bold; padding: 2px 6px; border-radius: 4px;")
            self.setStyleSheet("ROICardWidget { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }")

        # Update preview frame
        if res.bgr_crop is not None and res.bgr_crop.size > 0:
            crop_rgb = cv2.cvtColor(res.bgr_crop, cv2.COLOR_BGR2RGB)
            # Upscale for visibility
            h, w, ch = crop_rgb.shape
            scale_w, scale_h = 100, 70
            resized = cv2.resize(crop_rgb, (scale_w, scale_h), interpolation=cv2.INTER_NEAREST)
            qimg = QtGui.QImage(resized.data, scale_w, scale_h, scale_w * 3, QtGui.QImage.Format.Format_RGB888)
            self.lbl_preview.setPixmap(QtGui.QPixmap.fromImage(qimg))


class MacroMainWindow(QtWidgets.QMainWindow):
    """Main window with real-time HUD, tuning sliders, and logs."""

    def __init__(
        self,
        detector: ReactionDetector,
        on_toggle_arm: Optional[Callable[[], None]] = None,
        config_path: str = "dodge_parry_macro/config.json",
    ):
        super().__init__()
        self.detector = detector
        self.on_toggle_arm = on_toggle_arm
        self.config_path = config_path

        self.cards: Dict[str, ROICardWidget] = {}
        self.init_ui()

        # UI refresh timer (30 FPS)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(33)

    def init_ui(self) -> None:
        self.setWindowTitle("⚡ Dodge & Parry Reaction Macro - DirectInput HUD")
        self.resize(880, 680)
        self.setStyleSheet(STYLE_SHEET)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # --- Top Header Bar ---
        top_bar = QtWidgets.QHBoxLayout()

        # Arm / Disarm Button
        self.btn_arm = QtWidgets.QPushButton("⚡ ARM MACRO (F6)")
        self.btn_arm.setObjectName("btnArm")
        self.btn_arm.setProperty("armed", "false")
        self.btn_arm.clicked.connect(self.handle_arm_clicked)
        top_bar.addWidget(self.btn_arm)

        # Metrics Label
        self.lbl_fps = QtWidgets.QLabel("FPS: -- | Latency: -- ms")
        self.lbl_fps.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8; padding: 0 10px;")
        top_bar.addWidget(self.lbl_fps)

        top_bar.addStretch()

        # Calibration Button
        self.btn_calibrate = QtWidgets.QPushButton("🎯 Calibrate ROIs (F7)")
        self.btn_calibrate.clicked.connect(self.handle_calibrate_clicked)
        top_bar.addWidget(self.btn_calibrate)

        main_layout.addLayout(top_bar)

        # --- Middle: ROI Grid ---
        grp_rois = QtWidgets.QGroupBox("TARGET REGIONS OF INTEREST (5 ROIS)")
        grid_layout = QtWidgets.QGridLayout(grp_rois)
        grid_layout.setSpacing(12)

        # Cross pattern arrangement:
        #          [Up]
        #   [Left] [Center] [Right]
        #         [Down]
        rois = self.detector.config.get("rois", {})
        roi_defs = [
            ("up", 0, 1, "Up (W)", "w"),
            ("left", 1, 0, "Left (A)", "a"),
            ("center", 1, 1, "Center (F)", "f"),
            ("right", 1, 2, "Right (D)", "d"),
            ("down", 2, 1, "Down (S)", "s"),
        ]

        for roi_id, r, c, name, default_key in roi_defs:
            key = rois.get(roi_id, {}).get("key", default_key)
            card = ROICardWidget(roi_id, name, key)
            self.cards[roi_id] = card
            grid_layout.addWidget(card, r, c)

        main_layout.addWidget(grp_rois)

        # --- Bottom Split: Tuning & Logs ---
        btm_layout = QtWidgets.QHBoxLayout()

        # Tuning Controls
        grp_tuning = QtWidgets.QGroupBox("SENSITIVITY & TIMING TUNING")
        tune_layout = QtWidgets.QFormLayout(grp_tuning)
        tune_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # Red Min Density Slider
        self.slider_red = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_red.setRange(1, 40)
        self.slider_red.setValue(int(self.detector.red_min_density * 100))
        self.lbl_red_val = QtWidgets.QLabel(f"{self.slider_red.value()}%")
        self.slider_red.valueChanged.connect(self.on_red_slider_changed)
        tune_layout.addRow("Red Min Density:", self._wrap_slider(self.slider_red, self.lbl_red_val))

        # Green Min Density Slider
        self.slider_green = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_green.setRange(5, 60)
        self.slider_green.setValue(int(self.detector.green_min_density * 100))
        self.lbl_green_val = QtWidgets.QLabel(f"{self.slider_green.value()}%")
        self.slider_green.valueChanged.connect(self.on_green_slider_changed)
        tune_layout.addRow("Green Min Density:", self._wrap_slider(self.slider_green, self.lbl_green_val))

        # Cooldown Slider
        self.slider_cd = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_cd.setRange(100, 1000)
        self.slider_cd.setValue(int(self.detector.cooldown_s * 1000))
        self.lbl_cd_val = QtWidgets.QLabel(f"{self.slider_cd.value()} ms")
        self.slider_cd.valueChanged.connect(self.on_cd_slider_changed)
        tune_layout.addRow("Key Cooldown:", self._wrap_slider(self.slider_cd, self.lbl_cd_val))

        # Strict transition checkbox
        self.chk_strict = QtWidgets.QCheckBox("Strict Transition (Require Red before Green)")
        self.chk_strict.setChecked(self.detector.strict_transition)
        self.chk_strict.stateChanged.connect(self.on_strict_toggled)
        tune_layout.addRow("", self.chk_strict)

        btm_layout.addWidget(grp_tuning, 1)

        # Trigger Event Log
        grp_logs = QtWidgets.QGroupBox("ACTION TRIGGER LOG")
        log_layout = QtWidgets.QVBoxLayout(grp_logs)
        self.list_logs = QtWidgets.QListWidget()
        log_layout.addWidget(self.list_logs)
        btm_layout.addWidget(grp_logs, 1)

        main_layout.addLayout(btm_layout)

    def _wrap_slider(self, slider: QtWidgets.QSlider, label: QtWidgets.QLabel) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(slider, 1)
        label.setFixedWidth(50)
        l.addWidget(label)
        return w

    def handle_arm_clicked(self) -> None:
        if self.on_toggle_arm:
            self.on_toggle_arm()
        else:
            self.detector.armed = not self.detector.armed
            self.update_arm_button_state()

    def update_arm_button_state(self) -> None:
        armed = self.detector.armed
        self.btn_arm.setProperty("armed", "true" if armed else "false")
        self.btn_arm.setText("🛑 DISARM MACRO (F6)" if armed else "⚡ ARM MACRO (F6)")
        self.btn_arm.style().unpolish(self.btn_arm)
        self.btn_arm.style().polish(self.btn_arm)

    def handle_calibrate_clicked(self) -> None:
        open_roi_calibration(self.config_path)
        # Reload config in detector
        self.detector._update_thresholds()
        self.detector._init_states()

    def on_red_slider_changed(self, val: int) -> None:
        self.lbl_red_val.setText(f"{val}%")
        self.detector.red_min_density = val / 100.0

    def on_green_slider_changed(self, val: int) -> None:
        self.lbl_green_val.setText(f"{val}%")
        self.detector.green_min_density = val / 100.0

    def on_cd_slider_changed(self, val: int) -> None:
        self.lbl_cd_val.setText(f"{val} ms")
        self.detector.cooldown_s = val / 1000.0

    def on_strict_toggled(self, state: int) -> None:
        self.detector.strict_transition = (state == QtCore.Qt.CheckState.Checked.value)

    def add_log_event(self, text: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        item = QtWidgets.QListWidgetItem(f"[{ts}] {text}")
        self.list_logs.addItem(item)
        if self.list_logs.count() > 80:
            self.list_logs.takeItem(0)
        self.list_logs.scrollToBottom()

    def refresh_ui(self) -> None:
        self.update_arm_button_state()
        fps = self.detector.fps
        ms = (1000.0 / fps) if fps > 0 else 0.0
        self.lbl_fps.setText(f"FPS: {fps:5.1f} | Latency: {ms:4.1f} ms")

        # Update ROI Cards
        for roi_id, card in self.cards.items():
            if roi_id in self.detector.last_results:
                card.update_data(self.detector.last_results[roi_id])
