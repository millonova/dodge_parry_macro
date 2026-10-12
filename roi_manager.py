"""Interactive ROI sniper/calibration tool using PyQt6 overlay."""

import json
import os
import sys
from typing import Any, Dict, Optional

from PyQt6 import QtCore, QtGui, QtWidgets


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

ROI_COLORS = {
    "up": QtGui.QColor(239, 68, 68, 200),      # Red / Up (W)
    "left": QtGui.QColor(59, 130, 246, 200),   # Blue / Left (A)
    "down": QtGui.QColor(234, 179, 8, 200),    # Yellow / Down (S)
    "right": QtGui.QColor(168, 85, 247, 200),  # Purple / Right (D)
    "center": QtGui.QColor(16, 185, 129, 200), # Emerald / Center (F)
}


class ROICalibrationOverlay(QtWidgets.QWidget):
    """Fullscreen semi-transparent overlay to reposition and resize 5 ROIs."""

    def __init__(self, config_path: str = CONFIG_PATH, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.config: Dict[str, Any] = self.load_config()
        self.rois: Dict[str, Dict[str, Any]] = self.config.get("rois", {})

        self.selected_roi: Optional[str] = "center"
        self.dragging: bool = False
        self.resizing: bool = False
        self.drag_start_pos = QtCore.QPoint()
        self.drag_start_roi_rect = QtCore.QRect()

        self.init_ui()

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"rois": {}}

    def save_config(self) -> None:
        self.config["rois"] = self.rois
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def init_ui(self) -> None:
        self.setWindowTitle("ROI Sniper Overlay - Calibrate W, A, S, D, F")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        screen = QtGui.QGuiApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())

        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def get_roi_rect(self, roi_id: str) -> QtCore.QRect:
        r = self.rois[roi_id]
        return QtCore.QRect(r["x"], r["y"], r["w"], r["h"])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Dimmed backdrop
        painter.fillRect(self.rect(), QtGui.QColor(10, 15, 25, 140))

        # Instructions banner at top center
        banner_rect = QtCore.QRect(self.width() // 2 - 320, 24, 640, 70)
        painter.setBrush(QtGui.QColor(18, 24, 38, 230))
        painter.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248), 1.5))
        painter.drawRoundedRect(banner_rect, 10, 10)

        painter.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.Bold))
        painter.setPen(QtGui.QColor(241, 245, 249))
        painter.drawText(
            banner_rect.adjusted(0, 8, 0, -32),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "🎯 ROI SNIPER CALIBRATION - W, A, S, D, F",
        )

        painter.setFont(QtGui.QFont("Segoe UI", 9))
        painter.setPen(QtGui.QColor(148, 163, 184))
        painter.drawText(
            banner_rect.adjusted(0, 34, 0, -6),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "Drag boxes | Drag corner to resize | Arrow keys to nudge | Enter to Save | Esc to Cancel",
        )

        # Draw each ROI box
        for roi_id, roi in self.rois.items():
            rect = self.get_roi_rect(roi_id)
            color = ROI_COLORS.get(roi_id, QtGui.QColor(255, 255, 255, 200))
            is_selected = (roi_id == self.selected_roi)

            # Box fill & outline
            fill_color = QtGui.QColor(color.red(), color.green(), color.blue(), 50 if not is_selected else 90)
            pen_width = 3.0 if is_selected else 1.5
            pen_color = QtGui.QColor(color.red(), color.green(), color.blue(), 255 if is_selected else 180)

            painter.setBrush(fill_color)
            painter.setPen(QtGui.QPen(pen_color, pen_width))
            painter.drawRect(rect)

            # Corner crosshairs
            ch = 6
            painter.drawLine(rect.left(), rect.top() + rect.height() // 2, rect.left() + ch, rect.top() + rect.height() // 2)
            painter.drawLine(rect.right(), rect.top() + rect.height() // 2, rect.right() - ch, rect.top() + rect.height() // 2)
            painter.drawLine(rect.left() + rect.width() // 2, rect.top(), rect.left() + rect.width() // 2, rect.top() + ch)
            painter.drawLine(rect.left() + rect.width() // 2, rect.bottom(), rect.left() + rect.width() // 2, rect.bottom() - ch)

            # Resize handle on bottom-right
            if is_selected:
                handle_size = 8
                handle_rect = QtCore.QRect(
                    rect.right() - handle_size,
                    rect.bottom() - handle_size,
                    handle_size,
                    handle_size,
                )
                painter.fillRect(handle_rect, QtGui.QColor(255, 255, 255, 240))

            # Label badge
            label = f"{roi.get('key', '').upper()} ({roi.get('name', roi_id)})"
            painter.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Weight.Bold))
            fm = painter.fontMetrics()
            badge_w = fm.horizontalAdvance(label) + 10
            badge_h = 18
            badge_rect = QtCore.QRect(rect.left(), rect.top() - badge_h - 2, badge_w, badge_h)

            painter.fillRect(badge_rect, pen_color)
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(badge_rect, QtCore.Qt.AlignmentFlag.AlignCenter, label)

            # Dimension coordinates below
            dim_str = f"{rect.width()}x{rect.height()} ({rect.x()},{rect.y()})"
            painter.setFont(QtGui.QFont("Segoe UI", 7))
            painter.setPen(QtGui.QColor(203, 213, 225))
            painter.drawText(
                QtCore.QRect(rect.left(), rect.bottom() + 3, 120, 16),
                QtCore.Qt.AlignmentFlag.AlignLeft,
                dim_str,
            )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = event.pos()

            # First check if clicking on bottom-right resize handle of selected ROI
            if self.selected_roi:
                sel_rect = self.get_roi_rect(self.selected_roi)
                handle_rect = QtCore.QRect(sel_rect.right() - 10, sel_rect.bottom() - 10, 16, 16)
                if handle_rect.contains(pos):
                    self.resizing = True
                    self.drag_start_pos = pos
                    self.drag_start_roi_rect = sel_rect
                    return

            # Check click on any ROI
            clicked_roi = None
            for roi_id in reversed(list(self.rois.keys())):
                rect = self.get_roi_rect(roi_id)
                if rect.contains(pos):
                    clicked_roi = roi_id
                    break

            if clicked_roi:
                self.selected_roi = clicked_roi
                self.dragging = True
                self.drag_start_pos = pos
                self.drag_start_roi_rect = self.get_roi_rect(clicked_roi)
                self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = event.pos()
        delta = pos - self.drag_start_pos

        if self.dragging and self.selected_roi:
            r = self.rois[self.selected_roi]
            r["x"] = max(0, self.drag_start_roi_rect.x() + delta.x())
            r["y"] = max(0, self.drag_start_roi_rect.y() + delta.y())
            self.update()
        elif self.resizing and self.selected_roi:
            r = self.rois[self.selected_roi]
            r["w"] = max(10, self.drag_start_roi_rect.width() + delta.x())
            r["h"] = max(10, self.drag_start_roi_rect.height() + delta.y())
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.dragging = False
            self.resizing = False

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()

        # Save and apply
        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.save_config()
            self.close()
            return

        # Cancel
        if key == QtCore.Qt.Key.Key_Escape:
            self.close()
            return

        # Cycle selection with Tab or 1..5
        if key == QtCore.Qt.Key.Key_Tab:
            roi_keys = list(self.rois.keys())
            if self.selected_roi in roi_keys:
                idx = (roi_keys.index(self.selected_roi) + 1) % len(roi_keys)
                self.selected_roi = roi_keys[idx]
            else:
                self.selected_roi = roi_keys[0] if roi_keys else None
            self.update()
            return

        # Direct selection by key
        key_map = {
            QtCore.Qt.Key.Key_W: "up",
            QtCore.Qt.Key.Key_A: "left",
            QtCore.Qt.Key.Key_S: "down",
            QtCore.Qt.Key.Key_D: "right",
            QtCore.Qt.Key.Key_F: "center",
        }
        if key in key_map and key_map[key] in self.rois:
            self.selected_roi = key_map[key]
            self.update()
            return

        # Nudge position
        step = 5 if (event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier) else 1
        if self.selected_roi:
            r = self.rois[self.selected_roi]
            if key == QtCore.Qt.Key.Key_Left:
                r["x"] = max(0, r["x"] - step)
            elif key == QtCore.Qt.Key.Key_Right:
                r["x"] += step
            elif key == QtCore.Qt.Key.Key_Up:
                r["y"] = max(0, r["y"] - step)
            elif key == QtCore.Qt.Key.Key_Down:
                r["y"] += step
            elif key in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
                r["w"] += step
                r["h"] += step
            elif key in (QtCore.Qt.Key.Key_Minus, QtCore.Qt.Key.Key_Underscore):
                r["w"] = max(10, r["w"] - step)
                r["h"] = max(10, r["h"] - step)
            self.update()


def open_roi_calibration(config_path: str = CONFIG_PATH) -> None:
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True

    overlay = ROICalibrationOverlay(config_path=config_path)
    overlay.show()

    if owns_app:
        sys.exit(app.exec())


if __name__ == "__main__":
    open_roi_calibration()
