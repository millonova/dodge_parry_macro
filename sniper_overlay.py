"""Window 1: Floating multi-box ROI sniper overlay with independent sub-boxes and lock mode."""

import ctypes
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


class SubROIWidget(QtWidgets.QWidget):
    """Independently draggable and resizable child ROI sub-box."""

    geometry_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        action_id: str,
        label: str,
        color: QtGui.QColor,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.action_id = action_id
        self.label = label
        self.color = color
        self.locked = False

        self._is_dragging = False
        self._is_resizing = False
        self._drag_start_pos = QtCore.QPoint()
        self._drag_start_geo = QtCore.QRect()

        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def set_locked(self, locked: bool) -> None:
        self.locked = locked
        if locked:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.update()

    def set_label(self, label: str) -> None:
        self.label = label
        self.update()

    def _in_resize_grip(self, pos: QtCore.QPoint) -> bool:
        return (self.width() - pos.x() <= 14) and (self.height() - pos.y() <= 14)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.locked:
            event.ignore()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.globalPosition().toPoint()
            self._drag_start_geo = self.geometry()
            if self._in_resize_grip(event.pos()):
                self._is_resizing = True
            else:
                self._is_dragging = True
            parent = self.parentWidget()
            if parent and hasattr(parent, "set_calibration_active"):
                parent.set_calibration_active(True)
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.locked:
            event.ignore()
            return

        parent = self.parentWidget()
        pw = parent.width() if parent else 9999
        ph = parent.height() if parent else 9999

        if self._is_resizing and event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_start_pos
            new_w = max(24, min(pw - self.x(), self._drag_start_geo.width() + delta.x()))
            new_h = max(24, min(ph - self.y(), self._drag_start_geo.height() + delta.y()))
            self.resize(new_w, new_h)
            self.geometry_changed.emit()
            event.accept()
        elif self._is_dragging and event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_start_pos
            new_x = max(0, min(pw - self.width(), self._drag_start_geo.x() + delta.x()))
            new_y = max(18, min(ph - self.height(), self._drag_start_geo.y() + delta.y()))
            self.move(new_x, new_y)
            self.geometry_changed.emit()
            event.accept()
        else:
            if self._in_resize_grip(event.pos()):
                self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
            else:
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.locked:
            event.ignore()
            return
        self._is_dragging = False
        self._is_resizing = False
        parent = self.parentWidget()
        if parent and hasattr(parent, "set_calibration_active"):
            parent.set_calibration_active(False)
        event.accept()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        r, g, b = self.color.red(), self.color.green(), self.color.blue()

        # Fill & border
        fill_alpha = 35 if self.locked else 55
        border_alpha = 200 if self.locked else 240
        painter.setBrush(QtGui.QColor(r, g, b, fill_alpha))
        painter.setPen(QtGui.QPen(QtGui.QColor(r, g, b, border_alpha), 1.6))
        painter.drawRoundedRect(rect, 4, 4)

        # Action Label
        painter.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.Bold))
        painter.setPen(QtGui.QColor(255, 255, 255, 240))
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, self.label)

        # Interactive resize corner indicator when unlocked
        if not self.locked:
            grip_size = 8
            w, h = self.width(), self.height()
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(255, 255, 255, 180))
            path = QtGui.QPainterPath()
            path.moveTo(w - 2, h - 2)
            path.lineTo(w - 2 - grip_size, h - 2)
            path.lineTo(w - 2, h - 2 - grip_size)
            path.closeSubpath()
            painter.drawPath(path)


class SniperOverlayWindow(QtWidgets.QWidget):
    """Semi-transparent floating sniper window covering 5 calibrated action cues."""

    geometry_changed = QtCore.pyqtSignal(int, int, int, int)
    lock_state_changed = QtCore.pyqtSignal(bool)
    calibration_active = QtCore.pyqtSignal(bool)

    def __init__(self, config_path: str = CONFIG_PATH, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.config = self.load_config()

        # Lock mode state (controls mouse pass-through)
        self.locked = bool(self.config.get("roi_locked", False))

        # Drag state for parent window
        self._is_dragging = False
        self._drag_position = QtCore.QPoint()

        # Sub-ROI widgets
        self.sub_widgets: Dict[str, SubROIWidget] = {}

        # Debounce timer for saving geometry
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(200)
        self._save_timer.timeout.connect(self.save_geometry_to_config)

        self.init_ui()
        self.set_locked(self.locked)

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _apply_capture_exclusion(self) -> None:
        """Excludes overlay window from desktop screen capture (WDA_EXCLUDEFROMCAPTURE = 0x00000011)."""
        if sys.platform == "win32":
            try:
                hwnd = int(self.winId())
                WDA_EXCLUDEFROMCAPTURE = 0x00000011
                res = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
                if res:
                    print(f"[SniperOverlay] WDA_EXCLUDEFROMCAPTURE applied to HWND {hwnd}")
            except Exception as e:
                print(f"[SniperOverlay] Warning: SetWindowDisplayAffinity failed: {e}")

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._apply_capture_exclusion()

    def set_calibration_active(self, active: bool) -> None:
        self.calibration_active.emit(active)

    def update_action_label(self, action_id: str, label: str) -> None:
        if action_id in self.sub_widgets:
            self.sub_widgets[action_id].set_label(label)

    def init_ui(self) -> None:
        self.setWindowTitle("🎯 Dodge/Parry Sniper Overlay")
        # Sniper overlay is PERMANENTLY topmost
        flags = (
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.SubWindow
        )
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(120, 120)

        # Apply capture exclusion immediately upon initialization
        self._apply_capture_exclusion()

        # Load saved geometry from config
        sbox = self.config.get("sniper_box", {"x": 860, "y": 440, "width": 215, "height": 189})
        bw = int(sbox.get("width", 215))
        bh = int(sbox.get("height", 189))
        self.setGeometry(int(sbox.get("x", 860)), int(sbox.get("y", 440)), bw, bh)

        # Create 5 calibrated child sub-boxes
        actions_map = self.config.get("keybinds", {}).get("actions", {
            "up": "w", "left": "a", "down": "s", "right": "d", "center": "f"
        })

        sub_configs = self.config.get("sub_rois", {})
        default_sub_rois = self._compute_default_sub_rois(bw, bh)

        # Defensive palette: pure neon green completely stripped out to prevent self-detection loops
        colors = {
            "up": QtGui.QColor(239, 68, 68),      # Crimson Red
            "left": QtGui.QColor(59, 130, 246),   # Azure Blue
            "down": QtGui.QColor(249, 115, 22),   # Vivid Orange (Replaces greenish yellow)
            "right": QtGui.QColor(168, 85, 247),  # Electric Purple
            "center": QtGui.QColor(6, 182, 212),  # Electric Cyan (Replaces Emerald Green)
        }

        for action_id in ["up", "left", "center", "right", "down"]:
            key_label = actions_map.get(action_id, "").upper()
            w = SubROIWidget(
                action_id=action_id,
                label=key_label,
                color=colors.get(action_id, QtGui.QColor(255, 255, 255)),
                parent=self,
            )
            w.geometry_changed.connect(self.trigger_save)
            self.sub_widgets[action_id] = w

            # Determine initial rect from config or default
            if action_id in sub_configs:
                sc = sub_configs[action_id]
                rx, ry = int(sc.get("rx", 0)), int(sc.get("ry", 0))
                rw, rh = int(sc.get("rw", 50)), int(sc.get("rh", 50))
            else:
                rx, ry, rw, rh = default_sub_rois[action_id]

            w.setGeometry(rx, ry, rw, rh)

        # Bottom-right parent resize grip
        self.size_grip = QtWidgets.QSizeGrip(self)
        self.size_grip.setStyleSheet("""
            QSizeGrip {
                width: 16px;
                height: 16px;
                background-color: #38bdf8;
                border-radius: 3px;
                margin: 2px;
            }
        """)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Position size grip in bottom right
        if hasattr(self, "size_grip"):
            self.size_grip.move(self.width() - 18, self.height() - 18)
        self.trigger_save()

    def set_always_on_top(self, enabled: bool) -> None:
        """Enforces permanent topmost status on the Sniper Overlay."""
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        if self.isVisible():
            self.show()

    def set_locked(self, locked: bool) -> None:
        """Lock sub-boxes and activate mouse click-through for gameplay."""
        self.locked = locked
        for w in self.sub_widgets.values():
            w.set_locked(locked)

        self._update_click_through()
        self.update()
        self.lock_state_changed.emit(locked)

    def _update_click_through(self) -> None:
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, self.locked)
        if sys.platform == "win32":
            try:
                hwnd = int(self.winId())
                GWL_EXSTYLE = -20
                WS_EX_TRANSPARENT = 0x00000020
                WS_EX_LAYERED = 0x00080000
                style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                if self.locked:
                    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED)
                else:
                    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, (style | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT)
            except Exception:
                pass

    def _compute_default_sub_rois(self, box_w: int, box_h: int) -> Dict[str, Tuple[int, int, int, int]]:
        x_left_end = int(box_w * 0.30)
        x_right_start = int(box_w * 0.70)
        y_top_end = int(box_h * 0.30)
        y_bottom_start = int(box_h * 0.70)
        cw = max(24, x_right_start - x_left_end)
        ch = max(24, y_bottom_start - y_top_end)

        return {
            "center": (x_left_end, y_top_end, cw, ch),
            "up": (x_left_end, 18, cw, max(24, y_top_end - 18)),
            "down": (x_left_end, y_bottom_start, cw, max(24, box_h - y_bottom_start)),
            "left": (0, y_top_end, max(24, x_left_end), ch),
            "right": (x_right_start, y_top_end, max(24, box_w - x_right_start), ch),
        }

    def calculate_sub_rois(self) -> Dict[str, QtCore.QRect]:
        return {action: w.geometry() for action, w in self.sub_widgets.items()}

    def get_relative_sub_rois_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            action: {
                "rx": w.x(),
                "ry": w.y(),
                "rw": w.width(),
                "rh": w.height(),
            }
            for action, w in self.sub_widgets.items()
        }

    def get_absolute_rois(self) -> Dict[str, Dict[str, int]]:
        wx = self.x()
        wy = self.y()
        return {
            action: {
                "x": wx + w.x(),
                "y": wy + w.y(),
                "w": w.width(),
                "h": w.height(),
            }
            for action, w in self.sub_widgets.items()
        }

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Outer semi-transparent container
        bg_color = QtGui.QColor(15, 23, 42, 60 if self.locked else 100)
        border_color = QtGui.QColor(56, 189, 248, 120 if self.locked else 220)
        painter.setBrush(bg_color)
        painter.setPen(QtGui.QPen(border_color, 1.8))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

        # Header banner instruction
        status_text = "🔒 ROI LOCKED (CLICK-THROUGH)" if self.locked else "🎯 SNIPER OVERLAY (CALIBRATION MODE)"
        painter.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Weight.Bold))
        painter.setPen(QtGui.QColor(148, 163, 184, 180 if self.locked else 230))
        painter.drawText(
            QtCore.QRect(0, 2, self.width(), 16),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            status_text,
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.locked:
            event.ignore()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.set_calibration_active(True)
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.locked:
            event.ignore()
            return
        if self._is_dragging and event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.locked:
            event.ignore()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._is_dragging = False
            self.set_calibration_active(False)
            self.trigger_save()
            event.accept()

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:
        super().moveEvent(event)
        self.trigger_save()

    def trigger_save(self) -> None:
        self.geometry_changed.emit(self.x(), self.y(), self.width(), self.height())
        self._save_timer.start()

    def save_geometry_to_config(self) -> None:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        else:
            cfg = {}

        cfg["sniper_box"] = {
            "x": self.x(),
            "y": self.y(),
            "width": self.width(),
            "height": self.height(),
        }

        cfg["sub_rois"] = self.get_relative_sub_rois_dict()
        cfg["rois"] = self.get_absolute_rois()
        cfg["roi_locked"] = self.locked

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SniperOverlayWindow()
    window.show()
    sys.exit(app.exec())
