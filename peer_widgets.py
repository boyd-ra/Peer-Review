from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class LineSwatchWidget(QtWidgets.QWidget):
    def __init__(self, color_rgb: Tuple[int, int, int], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.color_rgb = color_rgb
        self.setFixedSize(28, 12)

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor(*self.color_rgb))
        pen.setWidth(2)
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        y = self.height() / 2.0
        painter.drawLine(1, int(round(y)), self.width() - 1, int(round(y)))
        painter.end()


class RangeSlider(QtWidgets.QWidget):
    valuesChanged = QtCore.Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 1000
        self._lower = 0
        self._upper = 1000
        self._active_handle: Optional[str] = None
        self.setMinimumHeight(28)
        self.setMouseTracking(True)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(220, 28)

    def setRange(self, minimum: int, maximum: int):
        self._minimum = int(minimum)
        self._maximum = max(int(maximum), self._minimum)
        self.setValues(self._lower, self._upper)

    def values(self) -> Tuple[int, int]:
        return self._lower, self._upper

    def setValues(self, lower: int, upper: int):
        lower = int(np.clip(lower, self._minimum, self._maximum))
        upper = int(np.clip(upper, lower, self._maximum))
        changed = lower != self._lower or upper != self._upper
        self._lower = lower
        self._upper = upper
        self.update()
        if changed:
            self.valuesChanged.emit(self._lower, self._upper)

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        margin = 10.0
        groove_height = 6.0
        handle_radius = 7.0
        groove_y = self.height() / 2.0
        groove_left = margin
        groove_right = max(margin, self.width() - margin)

        groove_rect = QtCore.QRectF(
            groove_left,
            groove_y - groove_height / 2.0,
            max(groove_right - groove_left, 1.0),
            groove_height,
        )
        lower_pos = self._value_to_pos(self._lower)
        upper_pos = self._value_to_pos(self._upper)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#d0d4dc"))
        painter.drawRoundedRect(groove_rect, groove_height / 2.0, groove_height / 2.0)

        selected_rect = QtCore.QRectF(
            lower_pos,
            groove_y - groove_height / 2.0,
            max(upper_pos - lower_pos, 1.0),
            groove_height,
        )
        painter.setBrush(QtGui.QColor("#2979ff"))
        painter.drawRoundedRect(selected_rect, groove_height / 2.0, groove_height / 2.0)

        for handle_pos in (lower_pos, upper_pos):
            painter.setBrush(QtGui.QColor("#ffffff"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#2979ff"), 1.5))
            painter.drawEllipse(QtCore.QPointF(handle_pos, groove_y), handle_radius, handle_radius)

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return

        pos_x = float(event.position().x())
        lower_dist = abs(pos_x - self._value_to_pos(self._lower))
        upper_dist = abs(pos_x - self._value_to_pos(self._upper))
        self._active_handle = "lower" if lower_dist <= upper_dist else "upper"
        self._move_active_handle(pos_x)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._active_handle is None:
            event.ignore()
            return
        self._move_active_handle(float(event.position().x()))
        event.accept()

    def mouseReleaseEvent(self, event):
        self._active_handle = None
        event.accept()

    def _move_active_handle(self, pos_x: float):
        value = self._pos_to_value(pos_x)
        if self._active_handle == "lower":
            self.setValues(value, self._upper)
        elif self._active_handle == "upper":
            self.setValues(self._lower, value)

    def _value_to_pos(self, value: int) -> float:
        span = max(self._maximum - self._minimum, 1)
        usable_width = max(self.width() - 20.0, 1.0)
        return 10.0 + usable_width * ((value - self._minimum) / span)

    def _pos_to_value(self, pos_x: float) -> int:
        usable_width = max(self.width() - 20.0, 1.0)
        ratio = (pos_x - 10.0) / usable_width
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))


class WindowLevelSlider(QtWidgets.QWidget):
    valuesChanged = QtCore.Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = -1200
        self._maximum = 2000
        self._lower = -160
        self._center = 40
        self._upper = 240
        self._active_handle: Optional[str] = None
        self.setMinimumHeight(28)
        self.setMouseTracking(True)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(220, 28)

    def setRange(self, minimum: int, maximum: int):
        self._minimum = int(minimum)
        self._maximum = max(int(maximum), self._minimum)
        self.setValues(self._lower, self._center, self._upper)

    def values(self) -> Tuple[int, int, int]:
        return self._lower, self._center, self._upper

    def setValues(self, lower: int, center: int, upper: int):
        lower = int(np.clip(lower, self._minimum, self._maximum))
        center = int(np.clip(center, lower, self._maximum))
        upper = int(np.clip(upper, center, self._maximum))
        changed = (lower, center, upper) != (self._lower, self._center, self._upper)
        self._lower, self._center, self._upper = lower, center, upper
        self.update()
        if changed:
            self.valuesChanged.emit(self._lower, self._center, self._upper)

    def setWindowLevel(self, width: int, level: int):
        half_width = max(int(round(width / 2.0)), 1)
        center = int(np.clip(level, self._minimum, self._maximum))
        max_half_width = min(center - self._minimum, self._maximum - center)
        half_width = max(1, min(half_width, max_half_width))
        self.setValues(center - half_width, center, center + half_width)

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        groove_y = self.height() / 2.0
        groove_height = 6.0
        handle_radius = 7.0
        groove_rect = QtCore.QRectF(10.0, groove_y - groove_height / 2.0, max(self.width() - 20.0, 1.0), groove_height)

        lower_pos = self._value_to_pos(self._lower)
        center_pos = self._value_to_pos(self._center)
        upper_pos = self._value_to_pos(self._upper)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#d0d4dc"))
        painter.drawRoundedRect(groove_rect, groove_height / 2.0, groove_height / 2.0)

        selected_rect = QtCore.QRectF(
            lower_pos,
            groove_y - groove_height / 2.0,
            max(upper_pos - lower_pos, 1.0),
            groove_height,
        )
        painter.setBrush(QtGui.QColor("#2979ff"))
        painter.drawRoundedRect(selected_rect, groove_height / 2.0, groove_height / 2.0)

        for handle_pos, fill in ((lower_pos, "#ffffff"), (center_pos, "#2979ff"), (upper_pos, "#ffffff")):
            painter.setBrush(QtGui.QColor(fill))
            painter.setPen(QtGui.QPen(QtGui.QColor("#2979ff"), 1.5))
            painter.drawEllipse(QtCore.QPointF(handle_pos, groove_y), handle_radius, handle_radius)

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        pos_x = float(event.position().x())
        positions = {
            "lower": self._value_to_pos(self._lower),
            "center": self._value_to_pos(self._center),
            "upper": self._value_to_pos(self._upper),
        }
        self._active_handle = min(positions, key=lambda key: abs(pos_x - positions[key]))
        self._move_active_handle(pos_x)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._active_handle is None:
            event.ignore()
            return
        self._move_active_handle(float(event.position().x()))
        event.accept()

    def mouseReleaseEvent(self, event):
        self._active_handle = None
        event.accept()

    def _move_active_handle(self, pos_x: float):
        value = self._pos_to_value(pos_x)
        if self._active_handle == "center":
            half_width = self._upper - self._center
            center = int(np.clip(value, self._minimum + half_width, self._maximum - half_width))
            self.setValues(center - half_width, center, center + half_width)
            return

        half_width = abs(value - self._center)
        max_half_width = min(self._center - self._minimum, self._maximum - self._center)
        half_width = int(np.clip(half_width, 1, max_half_width))
        self.setValues(self._center - half_width, self._center, self._center + half_width)

    def _value_to_pos(self, value: int) -> float:
        span = max(self._maximum - self._minimum, 1)
        usable_width = max(self.width() - 20.0, 1.0)
        return 10.0 + usable_width * ((value - self._minimum) / span)

    def _pos_to_value(self, pos_x: float) -> int:
        usable_width = max(self.width() - 20.0, 1.0)
        ratio = float(np.clip((pos_x - 10.0) / usable_width, 0.0, 1.0))
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))
